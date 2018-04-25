import csv
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import argparse
import pickle
import numpy as np
import html
import pandas as pd

from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from torch.autograd import Variable
from torch import optim

import pdb

use_cuda = torch.cuda.is_available()
#use_cuda = False # use CPU

HIDDEN_DIM = 64 

#labels_to_id = {'none': 0, 'racism': 1, 'sexism': 1}
labels_to_id = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

# Class for converting from words to ids and vice-versa
class Vocab:
    def __init__(self):
        self.word_to_id = defaultdict(int)
        self.id_to_word = []
        self.word_to_id['<UNK>'] = 0
        self.id_to_word.append('<UNK>')
        self.word_to_id['<SENT>'] = 1
        self.id_to_word.append('<SENT>')
        self.word_to_id['</SENT>'] = 2
        self.id_to_word.append('</SENT>')

    # Given a corpus, build token to id and id to token dictionaries for words
    # that occur more frequently than unk_threshold
    def build_vocab(self, corpus, unk_threshold=1):
        word_count = defaultdict(int)
        for text in corpus:
            for word in text:
                word_count[word] += 1

        for word, count in word_count.items():
            if count > unk_threshold and not word in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word.append(word)

    def get_word_id(self, word):
        return self.word_to_id[word]

    def get_word_from_id(self, word_id):
        return self.id_to_word[word_id]

    def size(self):
        return len(self.id_to_word)

class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BidirectionalEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.bi_lstm = nn.LSTM(self.hidden_size, self.hidden_size,
            bidirectional=True)

    def forward(self, instance_input, hidden):
        embedded = self.embedding(instance_input).view(1, 1, -1)
        output = embedded
        output, hidden = self.bi_lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2, 1, self.hidden_size)).cuda(), 
                Variable(torch.zeros(2, 1, self.hidden_size)).cuda())
        else:
            result = (Variable(torch.zeros(2, 1, self.hidden_size)),
                Variable(torch.zeros(2, 1, self.hidden_size)))

        return result

class AttentionClassifier(nn.Module):
    def __init__(self, num_labels_to_id, hidden_size):
        super(AttentionClassifier, self).__init__()
        self.num_labels_to_id = num_labels_to_id
        self.hidden_size = hidden_size

        self.layer1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_labels_to_id)

    def forward(self, encoder_outputs, encoder_hidden):
        interaction = torch.bmm(encoder_outputs.unsqueeze(0),
            encoder_hidden[0].view(self.hidden_size * 2, -1).unsqueeze(0))
        attn_weights = F.softmax(interaction, dim=1).view(1, 1, -1)
        attn_applied = torch.bmm(attn_weights,
            encoder_outputs.unsqueeze(0))
        output = F.relu(self.layer1(attn_applied))
        output = F.relu(self.layer2(output))
        output = self.out(output[0])

        return output, attn_weights

# Preprocesses text for model input
def process_raw_text(raw_text):
    tokenizer = TweetTokenizer()
    text = tokenizer.tokenize(raw_text)
    text = ['<SENT>'] + text
    text.append('</SENT>')

    return text

# Read in file containing text for building the training vocabulary
def read_corpus_file(corpus_filename, text_colname):
    corpus = []
    tokenizer = TweetTokenizer()
    reader = csv.DictReader(open(corpus_filename, 'r'))
    for row in reader:
        if row[text_colname]:
            text = process_raw_text(row[text_colname])
            corpus.append(text)

    return corpus

# Read in file containing list of slurs to ignore
def read_slur_file(slur_filename, vocab):
    slur_set = set() 
    for line in open(slur_filename, 'r'):
        slur = line.strip().lower()
        slur_id = vocab.get_word_id(slur)
        if not slur_id == vocab.get_word_id('<UNK>'):
            slur_set.add(slur_id)

    return slur_set

# Read file containing text and label pairs and converts them into Variables
def process_instances(instances_filename, vocab, labels_to_id, text_colname):
    instances = []
    reader = csv.DictReader(open(instances_filename, 'r'))
    for row in reader:
        if not text_colname in row:
            raise ValueError("No text column found")
        text = process_raw_text(row[text_colname])
        text_ids = [vocab.get_word_id(word) for word in text]
        text_variable = Variable(torch.LongTensor(text_ids).view(-1, 1))
        label = Variable(torch.LongTensor([labels_to_id[row['label']]]))
        if use_cuda:
            text_variable = text_variable.cuda()
            label = label.cuda()

        instances.append((text_variable, label))

    return instances

# Update the model for the given instance
def update_model(instance, encoder, encoder_optimizer, classifier, classifier_optimizer,
    criterion, slur_set, reverse_gradient, grad_lambda_val=.1):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    instance_input, instance_label = instance
    input_length = instance_input.size()[0]

    instance_ids = instance_input.view(-1).data.cpu().numpy()
    instance_slurs = [1 if x in slur_set else 0 for x in instance_ids]
    slur_mask = Variable(torch.FloatTensor(instance_slurs).view(1, 1, -1))
    zero_attention = Variable(torch.zeros(1, 1, input_length))
    grad_lambda = Variable(torch.FloatTensor([grad_lambda_val]))
    if use_cuda:
        slur_mask = slur_mask.cuda()
        zero_attention = zero_attention.cuda()
        grad_lambda = grad_lambda.cuda()

    encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size*2))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            instance_input[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    classifier_output, classifier_attention = classifier(encoder_outputs, encoder_hidden)
    if reverse_gradient:
        slur_attention = classifier_attention * slur_mask
        slur_attention_criterion = torch.nn.MSELoss()
        loss = criterion(classifier_output, instance_label) + grad_lambda * slur_attention_criterion(slur_attention,
            zero_attention)
    else:
        loss = criterion(classifier_output, instance_label)

    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()

    return loss

# Trains the model over training_instances for a given number of epochs
def train_epochs(training_instances, dev_instances, encoder, classifier, vocab, labels_to_id, out_dirpath, weight_filepath, preds_filepath, slur_set, criterion=torch.nn.CrossEntropyLoss(), reverse_gradient=False, n_epochs=30, print_every=500, learning_rate=.1):
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

    dev_inputs = [x[0] for x in dev_instances]
    dev_labels = [x[1] for x in dev_instances]

    print_loss_total = 0

    best_dev_score = -np.inf
    best_dev_results = ''

    best_encoder = None
    best_classifier = None

    starting_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M") # starting timestamp
    dirpath = out_dirpath + '{}'.format(starting_ts)

    for i in range(n_epochs):
        print_epoch_loss_total = 0
        print 'Training epoch ' + str(i)
        random.shuffle(training_instances)
        for j, instance in enumerate(training_instances):
            # Check if instance nonempty
            if len(instance[0].size()) > 0:
                loss = update_model(instance, encoder, encoder_optimizer, classifier, classifier_optimizer,
                    criterion, slur_set, reverse_gradient)
                print_loss_total += loss
                print_epoch_loss_total += loss

            if j !=0 and j % print_every == 0:
                print_loss_avg = print_loss_total/print_every
                print_loss_total = 0
                print('Epoch %d iteration %d loss: %.4f' % (i, j, print_loss_avg))

        print_epoch_loss_avg = print_epoch_loss_total/len(training_instances)
        print('Epoch %d avg loss: %.4f' % (i, print_epoch_loss_avg))
        predicted_dev_labels, attention_weights = classify(dev_inputs, encoder, classifier, vocab, labels_to_id)
        acc = evaluate_accuracy(dev_labels, predicted_dev_labels)
        prec, rec, f1 = evaluate_f1(dev_labels, predicted_dev_labels)
        
        print('Epoch %d dev accuracy: %.4f' % (i, acc))
        print('Epoch %d dev f1: %.4f' % (i, f1))
        print('Epoch %d dev precision: %.4f' % (i, prec))
        print('Epoch %d dev recall: %.4f' % (i, rec))
        print('----------------------------------')

        score = f1

        if score > best_dev_score:
            best_dev_score = score
            best_dev_results = 'Best so far f1 %.4f, precision %.4f, recall %.4f' % (f1, prec, rec)

            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

            # Save encoder
            torch.save(encoder, os.path.join(dirpath, 'encoder.model'))
            best_encoder = encoder

            # Save classifier
            torch.save(classifier, os.path.join(dirpath, 'classifier.model'))
            best_classifier = classifier

            # Save vocab
            with open(os.path.join(dirpath, 'vocab.pkl'), 'wb') as f:
                pickle.dump(vocab, f)

            # Save attn weights
            with open(weight_filepath, 'wb') as f:
                pickle.dump(attention_weights, f)

            # Save predictions
            preds = [p.data.cpu().tolist()[0] for p in predicted_dev_labels]
            with open(preds_filepath, 'wb') as f:
                pickle.dump(preds, f)

        print(best_dev_results)

    return best_encoder, best_classifier, starting_ts


# Runs the model as a classifier on the given instance_inputs
def classify(instance_inputs, encoder, classifier, vocab, labels_to_id):

    predicted_labels = []
    attention_weights = []

    for instance_input in instance_inputs:
        encoder_hidden = encoder.init_hidden()
        if len(instance_input.size()) == 0:
            continue

        input_length = instance_input.size()[0]

        encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size*2))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                instance_input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        classifier_output, classifier_attention = classifier(encoder_outputs, encoder_hidden)
        top_value, top_label = classifier_output.data.topk(1)
        predicted_labels.append(Variable(top_label.squeeze(0)))
        attention_weights.append(classifier_attention.data.cpu().tolist())
            
    return predicted_labels, attention_weights

def evaluate_accuracy(true_labels, predicted_labels):
    correct = 0
    total = 0
    for tl, pl in zip(true_labels, predicted_labels):
        total += 1
        if tl.data[0] == pl.data[0]:
            correct += 1

    return float(correct)/total * 100

def evaluate_f1(true_labels, predicted_labels):
    """ Evaluate f1 on  predicting hate speech """

    total = 0
    pred_hs = 0
    actual_hs = 0
    correct_hs = 0

    for tl, pl in zip(true_labels, predicted_labels):
        total += 1
        if tl.data[0] == 1: # is actually hate speech
            actual_hs += 1
        if pl.data[0] == 1: # is predicted hate speech
            pred_hs += 1
        if tl.data[0] == pl.data[0] == 1:
            correct_hs += 1

    if pred_hs == 0:
        prec = 0.0
    else:
        prec = float(correct_hs)/pred_hs

    rec = float(correct_hs)/actual_hs

    if prec == rec == 0:
       f1 = 0.0 
    else:
       f1 = 2 * prec * rec / (prec + rec) * 100  

    return prec*100, rec*100, f1

def load_model(model_path, model_ts):
    dirpath = os.path.join(model_path + model_ts)
    clf_path = os.path.join(dirpath, "classifier.model")
    encoder_path = os.path.join(dirpath, "encoder.model")
    vocab_path = os.path.join(dirpath, "vocab.pkl")

    assert os.path.exists(clf_path) and os.path.exists(encoder_path) and os.path.exists(vocab_path)

    encoder = torch.load(encoder_path)
    classifier = torch.load(clf_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    return encoder, classifier, vocab

def color_attn(val, total_max, total_min):
    """ Returns 0-1 for highlighting """
    
    scale = 1/total_max
    val = (val-total_min) * scale
    return val

def attention_visualization(weight_filepath, preds_filepath, viz_filepath, dev_filename, dev_labels, text_colname):

    # Load weights
    with open(weight_filepath, 'rb') as f:
        wts = pickle.load(f)
        wts = [w[0][0] for w in wts]

    # Load predictions
    with open(preds_filepath, 'rb') as f:
        preds = pickle.load(f)

    # Load, tokenize text
    text = read_corpus_file(dev_filename, text_colname)

    # Check lengths
    for i, (w,t) in enumerate(zip(wts, text)):
        if len(w) != len(t):
            print('{}: {} - {}'.format(i, len(w), len(t)))

    # Make visualization string
    total_max = max(d for wt in wts for d in wt)
    total_min = min(d for wt in wts for d in wt)

    wts_viz = []
    for i, (wt, sent) in enumerate(zip(wts, text)):

        vals = [color_attn(d, total_max, total_min) for d in wt]
        try:
            wts_viz.append(''.join(["<span style='background-color: rgba(255,0,0,{})'>{}</span>&nbsp".format(val, html.escape(w)) for val,w in zip(vals, sent)]))
        except UnicodeEncodeError:
            continue

    # Match attention weights with predictions
    out = pd.DataFrame(list(zip(wts_viz, preds, dev_labels)), columns=['attention_weights', 'predicted_label', 'actual_label'])
    out['attention_weights'].map(lambda x: x.encode('utf8'))

    pd.set_option('display.max_colwidth', -1)

    out.to_html(viz_filepath, escape=False)

def main():

    # Argparse
    parser = argparse.ArgumentParser(description='Train model to identify hate speech.')
    parser.add_argument('--load-model', nargs='?', dest='load', help='timestamp of model to load in format YYYY-MM-DDTHH-MM-SS', default='')
    parser.add_argument('--dataset', nargs='?', dest='dataset_name', help='timestamp of model to load in format YYYY-MM-DDTHH-MM-SS', default='')
    parser.add_argument('--text-colname', nargs='?', dest='text_colname', help='name of column with input tweet text', default='')
    parser.add_argument('--epochs', nargs='?', dest='n_epochs', help='Number of epochs', type=int, default=30)
    parser.add_argument('--reverse-gradient', action='store_true', dest='grad', help='run the gradient reversal version of the model')
    #parser.add_argument('--just-eval', dest='just_eval', action='store_true')
    #parser.set_defaults(just_eval=False)
    parser.set_defaults(grad=False)
    args = parser.parse_args()

    if args.dataset_name == 'davidson':
        #training_filename = 'data/davidson/debug.csv'
        #dev_filename = 'data/davidson/debug.csv'
        #test_filename = 'data/davidson/debug.csv'

        training_filename = 'data/davidson/train.csv'
        dev_filename = 'data/davidson/dev.csv'
        test_filename = 'data/davidson/test.csv' 

    elif args.dataset_name == 'zeerak_naacl':
        training_filename = 'data/zeerak_naacl/train.csv'
        dev_filename = 'data/zeerak_naacl/dev.csv'
        test_filename = 'data/zeerak_naacl/test.csv' 

    else:
        raise ValueError("No dataset name given")

    #text_colname = 'tweet'
    #text_colname = 'text'
    #text_colname = 'tweet_unk_slur'
    #text_colname = 'tweet_no_slur'

    fold_name = os.path.splitext(os.path.basename(dev_filename))[0] # to examine predictions
    out_dirpath =  'models/{}_'.format(args.dataset_name) # path to save the model files to
    weight_filepath = 'output/{}_{}_{}_attn.pkl'.format(args.dataset_name, args.text_colname, fold_name) # filepath for attention weights
    preds_filepath = 'output/{}_{}_{}_preds.pkl'.format(args.dataset_name, args.text_colname, fold_name) # filepath for predictions

    # If loading an existing model
    if args.load:
        print("Loading model...")
        ts = args.load
        encoder, classifier, vocab = load_model(out_dirpath, ts)

    else:
        training_corpus = read_corpus_file(training_filename, args.text_colname)
        vocab = Vocab()
        vocab.build_vocab(training_corpus)

        encoder = BidirectionalEncoder(vocab.size(), HIDDEN_DIM)
        classifier = AttentionClassifier(len(labels_to_id), HIDDEN_DIM)

    if use_cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    training_instances = process_instances(training_filename, vocab, labels_to_id, args.text_colname)
    dev_instances = process_instances(dev_filename, vocab, labels_to_id, args.text_colname)
    test_instances = process_instances(test_filename, vocab, labels_to_id, args.text_colname)

    slur_filename = 'data/hatebase_slurs.txt'
    slur_set = read_slur_file(slur_filename, vocab)

    if not args.load:
        encoder, classifier, ts = train_epochs(training_instances, dev_instances, encoder, classifier, vocab, labels_to_id, out_dirpath, weight_filepath, preds_filepath, slur_set, 
            print_every=500, reverse_gradient=args.grad, n_epochs=args.n_epochs)

    viz_filepath = 'output/{}_{}_attn_viz.html'.format(args.dataset_name, ts) # filepath for predictions

    # Evaluate on test
    print("Evaluating on test...")
    test_inputs = [x[0] for x in test_instances]
    test_labels = [x[1] for x in test_instances]
    # Check for maximum index
    #max_ind = max([max(el.data.cpu().numpy().flatten()) for el in test_inputs])
    preds, attn_weights = classify(test_inputs, encoder, classifier, vocab, labels_to_id)
    prec, rec, f1 = evaluate_f1(test_labels, preds)
    print('test f1: %.4f' % f1)
    print('test precision: %.4f' % prec)
    print('test recall: %.4f' % rec)

    # Make attention weight visualization
    dev_labels = [x[1].data[0] for x in dev_instances]
    attention_visualization(weight_filepath, preds_filepath, viz_filepath, dev_filename, dev_labels, args.text_colname)

if __name__ == '__main__': main()
