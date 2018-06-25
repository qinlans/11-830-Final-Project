import csv
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import argparse
import pickle
import numpy as np
import html
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from torch.autograd import Variable
from torch import optim

import pdb

use_cuda = torch.cuda.is_available()
#use_cuda = False # use CPU

HIDDEN_DIM = 64 

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
        if row['label'] == '':
            pdb.set_trace()
        label = Variable(torch.LongTensor([labels_to_id[row['label']]]))
        if use_cuda:
            text_variable = text_variable.cuda()
            label = label.cuda()

        instances.append((text_variable, label))

    return instances

# Update the model for the given instance
def update_model(instance, encoder, encoder_optimizer, classifier, classifier_optimizer,
    criterion, slur_set, reverse_gradient, grad_lambda_val=.5):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    instance_input, instance_label = instance
    input_length = instance_input.size()[0]

    encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size*2))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            instance_input[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    classifier_output, classifier_attention = classifier(encoder_outputs, encoder_hidden)
    if reverse_gradient:
        instance_ids = instance_input.view(-1).data.cpu().numpy()
        instance_slurs = [1 if x in slur_set else 0 for x in instance_ids]
        slur_mask = Variable(torch.FloatTensor(instance_slurs).view(1, 1, -1))
        zero_attention = Variable(torch.zeros(1, 1, input_length))
        grad_lambda = Variable(torch.FloatTensor([grad_lambda_val]))
        if use_cuda:
            slur_mask = slur_mask.cuda()
            zero_attention = zero_attention.cuda()
            grad_lambda = grad_lambda.cuda()

        slur_attention = classifier_attention * slur_mask
        slur_attention_criterion = torch.nn.MSELoss()
        loss = criterion(classifier_output, instance_label) + grad_lambda * slur_attention_criterion(slur_attention,
            zero_attention)
    else:
        loss = criterion(classifier_output, instance_label)

    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()

    del classifier
    del encoder
    del encoder_outputs
    del classifier_output
    del encoder_optimizer
    del classifier_optimizer

    return loss.data.cpu()[0]

def train_epoch(i, training_instances, encoder, encoder_optimizer, classifier, classifier_optimizer, criterion, slur_set, reverse_gradient, print_every, output_dirpath):
    print_epoch_loss_total = 0
    print_loss_total = 0
    for j, instance in enumerate(tqdm(training_instances, ncols=50)):
        # Check if instance nonempty
        if len(instance[0].size()) > 0:
            loss = update_model(instance, encoder, encoder_optimizer, classifier, classifier_optimizer,
                criterion, slur_set, reverse_gradient, grad_lambda_val=0.5)
            print_loss_total += loss
            print_epoch_loss_total += loss

        if j !=0 and j % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            tqdm.write('Epoch %d iteration %d loss: %.4f' % (i, j, print_loss_avg))

    print_epoch_loss_avg = print_epoch_loss_total/len(training_instances)
    print('Epoch %d avg loss: %.4f' % (i, print_epoch_loss_avg))
    # Save avg losses to a file
    loss_fpath = os.path.join(output_dirpath, 'loss.txt')
    with open(loss_fpath, 'a') as f:
        f.write('{}\t{:.4f}\n'.format(i, print_epoch_loss_avg))

    return loss

# Trains the model over training_instances for a given number of epochs
def train_epochs(training_instances, dev_instances, encoder, classifier, vocab,
    labels_to_id, model_dirpath, output_dirpath, slur_set,
    criterion=torch.nn.CrossEntropyLoss(), reverse_gradient=False, n_epochs=30,
    print_every=500, learning_rate=.1, categorical=False):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    
    id_to_labels = {v: k for k, v in labels_to_id.items()}

    dev_inputs = [x[0] for x in dev_instances]
    dev_labels = [x[1] for x in dev_instances]

    print_loss_total = 0

    best_dev_results = None
    best_dev_score = -np.inf

    if print_every == -1: # Don't print log info
        print_every = sys.maxsize

    for i in range(n_epochs):
        print_epoch_loss_total = 0
        print('Training epoch ' + str(i))
        random.shuffle(training_instances)

        if not os.path.exists(model_dirpath):
            os.mkdir(model_dirpath)
        if not os.path.exists(output_dirpath):
            os.mkdir(output_dirpath)

        loss = train_epoch(i, training_instances, encoder, encoder_optimizer, classifier, classifier_optimizer, criterion, slur_set, reverse_gradient, print_every, output_dirpath)
        print_loss_total += loss

        predicted_dev_labels, attention_weights = classify(dev_inputs, encoder, classifier, labels_to_id)
        results, prec, rec, f1, acc = evaluate(dev_labels, predicted_dev_labels, labels_to_id, epoch_num=i, categorical=categorical)
        
        print('Epoch %d dev accuracy: %.4f' % (i, acc))
        print('Epoch %d dev f1: %.4f' % (i, f1))
        print('Epoch %d dev precision: %.4f' % (i, prec))
        print('Epoch %d dev recall: %.4f' % (i, rec))
        print('----------------------------------')

        if f1 > best_dev_score or best_dev_score is None:
            # Update all our best results
            best_dev_results = results
            prec, rec, f1, _, _, = tuple(best_dev_results.loc[id_to_labels[1]])
            best_epoch = best_dev_results.index.name
            best_dev_score = f1

            # Save encoder
            torch.save(encoder, os.path.join(model_dirpath, 'encoder.model'))

            # Save classifier
            torch.save(classifier, os.path.join(model_dirpath, 'classifier.model'))

            # Save vocab
            with open(os.path.join(model_dirpath, 'vocab.pkl'), 'wb') as f:
                pickle.dump(vocab, f)

            # Save attn weights
            with open(os.path.join(output_dirpath, 'dev_attn.pkl'), 'wb') as f:
                pickle.dump(attention_weights, f)

            # Save predictions
            preds = [p.data.cpu().tolist()[0] for p in predicted_dev_labels]
            with open(os.path.join(output_dirpath, 'dev_preds.pkl'), 'wb') as f:
                pickle.dump(preds, f)

            # Save the best scores
            best_dev_results.to_csv(os.path.join(output_dirpath, 'dev_scores.csv'))

        print()
        print("Best dev results so far:")
        print(best_dev_results)
        print()

def evaluate(y, y_pred, labels_to_id, epoch_num='', return_all=True, categorical=False):
    """Compute the performance on the data."""

    id_to_labels = {v: k for k, v in labels_to_id.items()}
    y = [*map(lambda v: v.data.cpu().numpy()[0], y)]
    y_pred = [*map(lambda v: v.data.cpu().numpy()[0], y_pred)]

    # Set up the output DataFrame
    index = [id_to_labels[v] for v in list(set(y))] + ['weighted_average']
    columns = ['precision', 'recall', 'f1_score', 'accuracy', 'support']
    results = pd.DataFrame(index=index, columns=columns)
    results.index.name = 'Epoch {}'.format(epoch_num)
    
    # Compute everything
    acc = accuracy_score(y, y_pred)
    res = precision_recall_fscore_support(y, y_pred)
    #pos_labels = list(range(1,max(set(y))))
    #res_weight = precision_recall_fscore_support(y, y_pred, labels=pos_labels, average='weighted')
    res_weight = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    # Compile result numbers
    prec = np.concatenate([res[0], [res_weight[0]]])
    rec = np.concatenate([res[1], [res_weight[1]]])
    f1 =  np.concatenate([res[2], [res_weight[2]]])
    sup = np.concatenate([res[3], [sum(res[3])]])
    
    # Put into results and return
    results['accuracy']['weighted_average'] = acc
    results['precision'] = prec
    results['recall'] = rec
    results['f1_score'] = f1
    results['support'] = sup
    
    if return_all:
        if categorical:
            prec, rec, f1, _, _, = tuple(results.loc['weighted_average'])
        else:
            prec, rec, f1, _, _, = tuple(results.loc[id_to_labels[1]])
        return results, prec, rec, f1, acc
    else:
        return results

# Runs the model as a classifier on the given instance_inputs
def classify(instance_inputs, encoder, classifier, labels_to_id):

    predicted_labels = []
    attention_weights = []

    for instance_input in tqdm(instance_inputs, ncols=50):
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

def load_model(model_path):
    clf_path = os.path.join(model_path, "classifier.model")
    encoder_path = os.path.join(model_path, "encoder.model")
    vocab_path = os.path.join(model_path, "vocab.pkl")

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

def zero_division(n, d):
    return n/d if d else 0

def attention_visualization(output_dirpath, fold, filename, labels, text_colname):
    viz_filepath = os.path.join(output_dirpath, '{}_attn_viz.html'.format(fold))
    weight_filepath = os.path.join(output_dirpath, '{}_attn.pkl'.format(fold))
    word_weight_filepath= os.path.join(output_dirpath, '{}_attn_top_weights.csv'.format(fold))
    preds_filepath = os.path.join(output_dirpath, '{}_preds.pkl'.format(fold))
    types = ['total', 'hate', 'none']
    word_weights = {t: defaultdict(float) for t in types}
    word_ctr = {t: defaultdict(int) for t in types}

    # Load weights
    with open(weight_filepath, 'rb') as f:
        wts = pickle.load(f)
        wts = [w[0][0] for w in wts]

    # Load predictions
    with open(preds_filepath, 'rb') as f:
        preds = pickle.load(f)

    # Load, tokenize text
    text = read_corpus_file(filename, text_colname)

    # Check lengths
    for i, (w,t) in enumerate(zip(wts, text)):
        if len(w) != len(t):
            print('Length mismatch for attention visualization instance {}: {} - {}'.format(i, len(w), len(t)))

    # Make visualization string
    total_max = max(d for wt in wts for d in wt)
    total_min = min(d for wt in wts for d in wt)

    wts_viz = []
    for i, (wt, sent, pred) in enumerate(zip(wts, text, preds)):

        # Accumulate top word weights
        for (wd, d) in zip(sent, wt):
            word_weights['total'][wd] += d
            word_ctr['total'][wd] += 1 
            if pred == 1:
                word_weights['hate'][wd] += d
                word_ctr['hate'][wd] += 1 
            else:
                word_weights['none'][wd] += d
                word_ctr['none'][wd] += 1 
        
        vals = [color_attn(d, total_max, total_min) for d in wt]
        wts_viz.append(''.join(["<span style='background-color: rgba(255,0,0,{})'>{}</span>&nbsp".format(val, html.escape(w)) for val,w in zip(vals, sent)]))

    # Match attention weights with predictions
    out = pd.DataFrame(list(zip(wts_viz, preds, labels)), columns=['attention_weights', 'predicted_label', 'actual_label'])
    out['attention_weights'].map(lambda x: x.encode('utf8'))

    pd.set_option('display.max_colwidth', -1)

    out.to_html(viz_filepath, escape=False)

    # Save top word weights
    outlines = [[wd, 
                zero_division(word_weights['total'][wd], word_ctr['total'][wd]), 
                word_ctr['total'][wd], 
                word_weights['total'][wd],
                zero_division(word_weights['hate'][wd], word_ctr['hate'][wd]), 
                word_ctr['hate'][wd], 
                word_weights['hate'][wd],
                zero_division(word_weights['none'][wd], word_ctr['none'][wd]), 
                word_ctr['none'][wd], 
                word_weights['none'][wd],
                ] for wd in word_weights['total']]
    top_wd_wts = pd.DataFrame(outlines, 
                        columns=['word', 'average_weight', 'count', 'total_weight', 'average_hate_weight', 'hate_count', 'hate_weight', 'average_none_weight', 'none_count', 'average_none_weight'])
    top_wd_wts.sort_values('average_hate_weight', inplace=True, ascending=False)
    top_wd_wts.to_csv(word_weight_filepath, index=False)

def main():

    # Argparse
    parser = argparse.ArgumentParser(description='Train model to identify hate speech.')
    parser.add_argument('--load-model', nargs='?', dest='load', help='Name of model to load (e.g. dataset_YYYY-MM-DDTHH-MM-SS if no model name provided)', default='')
    parser.add_argument('--dataset', nargs='?', dest='dataset_name', help='Name of dataset to load', default='')
    parser.add_argument('--text-colname', nargs='?', dest='text_colname', help='name of column with input tweet text', default='')
    parser.add_argument('--lambda', nargs='?', dest='grad_lambda', help='gradient reversal lambda', default=0.1)
    parser.add_argument('--model-name', nargs='?', dest='model_name', help='name of model to save to', default=None)
    parser.add_argument('--epochs', nargs='?', dest='n_epochs', help='Number of epochs', type=int, default=30)
    parser.add_argument('--reverse-gradient', action='store_true', dest='grad', help='run the gradient reversal version of the model')
    parser.add_argument('--categorical', action='store_true', dest='categorical', help='do three-way classification')
    parser.add_argument('--debug', action='store_true', dest='debug', help='load small dataset for debugging', default=False)
    parser.add_argument('--cross-domain', action='store_true', dest='cross', help='cross domain running', default=False)
    parser.set_defaults(grad=False)
    parser.set_defaults(categorical=False)
    args = parser.parse_args()

    if args.dataset_name == 'davidson':
        if args.debug:
            training_filename = 'data/davidson/debug.csv'
            dev_filename = 'data/davidson/debug.csv'
            test_filename = 'data/davidson/debug.csv' 
        else:
            training_filename = 'data/davidson/train.csv'
            dev_filename = 'data/davidson/dev.csv'
            test_filename = 'data/davidson/test.csv' 
        
        if args.categorical:
            labels_to_id = {'neither': 0, 'offensive_language': 1, 'hate_speech': 2}
        else:
            labels_to_id = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

    elif args.dataset_name == 'zeerak_naacl':
        training_filename = 'data/zeerak_naacl/train.csv'
        dev_filename = 'data/zeerak_naacl/dev.csv'
        test_filename = 'data/zeerak_naacl/test.csv' 
        if args.categorical:
            labels_to_id = {'none': 0, 'racism': 1, 'sexism': 2}
        else:
            labels_to_id = {'none': 0, 'racism': 1, 'sexism': 1}

    elif args.dataset_name == 'davidson-zeerak':
        training_filename = 'data/davidson/train.csv'
        dev_filename = 'data/davidson/dev.csv'
        test_filename = 'data/zeerak_naacl/test.csv' 
        labels_to_id = {'none': 0, 'racism': 1, 'sexism': 1}

    elif args.dataset_name == 'zeerak-davidson':
        training_filename = 'data/zeerak_naacl/train.csv'
        dev_filename = 'data/zeerak_naacl/dev.csv'
        test_filename = 'data/davidson/test.csv' 
        labels_to_id = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

    elif args.dataset_name == 'sexism-racism_zeerak_naacl':
        training_filename = 'data/zeerak_naacl/sexism_train.csv'
        dev_filename = 'data/zeerak_naacl/sexism_dev.csv'
        test_filename = 'data/zeerak_naacl/racism.csv' 
        labels_to_id = {'none': 0, 'racism': 1, 'sexism': 1}

    elif args.dataset_name == 'racism-sexism_zeerak_naacl':
        training_filename = 'data/zeerak_naacl/racism_train.csv'
        dev_filename = 'data/zeerak_naacl/racism_dev.csv'
        test_filename = 'data/zeerak_naacl/sexism.csv' 
        labels_to_id = {'none': 0, 'racism': 1, 'sexism': 1}

    elif args.dataset_name == 'sexism_zeerak_naacl':
        training_filename = 'data/zeerak_naacl/sexism_train.csv'
        dev_filename = 'data/zeerak_naacl/sexism_dev.csv'
        test_filename = 'data/zeerak_naacl/sexism_test.csv' 
        labels_to_id = {'none': 0, 'racism': 0, 'sexism': 1}

    elif args.dataset_name == 'racism_zeerak_naacl':
        training_filename = 'data/zeerak_naacl/racism_train.csv'
        dev_filename = 'data/zeerak_naacl/racism_dev.csv'
        test_filename = 'data/zeerak_naacl/racism_test.csv' 
        labels_to_id = {'none': 0, 'racism': 1, 'sexism': 0}

    else:
        raise ValueError("No dataset name given")

    fold_name = os.path.splitext(os.path.basename(dev_filename))[0] # to examine predictions

    if args.model_name:
        model_dirpath =  'models/{}'.format(args.model_name) # path to save the model files to
        if args.cross:
            output_dirpath =  'output/{}_{}'.format(args.dataset_name, args.model_name) # path to save the output files to
            if not os.path.exists(output_dirpath):
                os.mkdir(output_dirpath)
        else:
            output_dirpath =  'output/{}'.format(args.model_name) # path to save the output files to
    elif args.load:
        model_dirpath =  'models/{}'.format(args.load) # path to save the model files to
        if args.cross:
            output_dirpath =  'output/{}_{}'.format(args.dataset_name, args.load) # path to save the output files to
            if not os.path.exists(output_dirpath):
                os.mkdir(output_dirpath)
        else:
            output_dirpath =  'output/{}'.format(args.load) # path to save the output files to

    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M") # starting timestamp
        model_dirpath =  'models/{}_{}'.format(args.dataset_name, ts) # path to save the model files to
        output_dirpath =  'output/{}_{}'.format(args.dataset_name, ts) # path to save the output files to

    # If loading an existing model
    if args.load:
        print("Loading model...")
        encoder, classifier, vocab = load_model(model_dirpath)

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

    if args.dataset_name.endswith('davidson'):
        slur_filename = 'data/hatebase_slurs.txt'
    elif args.dataset_name.endswith('zeerak_naacl') or args.dataset_name.endswith('zeerak'):
        slur_filename = 'data/hatebase+zeerak_exclude_slurs.txt'
    else:
        slur_filename = ''
    slur_set = read_slur_file(slur_filename, vocab)

    if not args.load:
        train_epochs(training_instances, dev_instances, encoder,
            classifier, vocab, labels_to_id, model_dirpath, output_dirpath, slur_set, 
            print_every=-1, reverse_gradient=args.grad, n_epochs=args.n_epochs)
        encoder, classifier, vocab = load_model(model_dirpath)

    # Evaluate on test
    print("Evaluating on test...")
    test_inputs = [x[0] for x in test_instances]
    test_labels = [x[1] for x in test_instances]
    # Check for maximum index
    #max_ind = max([max(el.data.cpu().numpy().flatten()) for el in test_inputs])
    preds, attn_weights = classify(test_inputs, encoder, classifier, labels_to_id)

    # Save attn weights
    with open(os.path.join(output_dirpath, 'test_attn.pkl'), 'wb') as f:
        pickle.dump(attn_weights, f)

    # Save predictions
    pred_vals = [p.data.cpu().tolist()[0] for p in preds]
    with open(os.path.join(output_dirpath, 'test_preds.pkl'), 'wb') as f:
        pickle.dump(pred_vals, f)

    results, prec, rec, f1, _ = evaluate(test_labels, preds, labels_to_id, categorical=args.categorical)
    print('test f1: %.4f' % f1)
    print('test precision: %.4f' % prec)
    print('test recall: %.4f' % rec)
    results.to_csv(os.path.join(output_dirpath, 'test_scores.csv'))

    # Make attention weight visualizations from dev and test weights
    #dev_labels = [x[1].data[0] for x in dev_instances]
    test_labels = [x[1].data[0] for x in test_instances]
    #attention_visualization(output_dirpath, 'dev', dev_filename, dev_labels, args.text_colname)
    attention_visualization(output_dirpath, 'test', test_filename, test_labels, args.text_colname)

if __name__ == '__main__': main()
