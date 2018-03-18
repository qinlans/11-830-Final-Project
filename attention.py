import csv
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import argparse
import pickle

from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from torch.autograd import Variable
from torch import optim

import pdb

use_cuda = torch.cuda.is_available()
HIDDEN_DIM = 64 
#labels_to_id = {'none': 0, 'racism': 1, 'sexism': 2}
labels_to_id = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

# Class for converting from words to ids and vice-versa
class Vocab:
    def __init__(self):
        self.word_to_id = defaultdict(lambda: 0)
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
        word_count = defaultdict(lambda: 0)
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
    text = tokenizer.tokenize(raw_text.lower())
    text = ['<SENT>'] + text
    text.append('</SENT>')
    return text

# Read in file containing text for building the training vocabulary
def read_corpus_file(corpus_filename):
    corpus = []
    tokenizer = TweetTokenizer()
    reader = csv.DictReader(open(corpus_filename, 'r'))
    for row in reader:
        if row['text']:
            text = process_raw_text(row['text'])
            corpus.append(text)

    return corpus

# Read file containing text and label pairs and converts them into Variables
def process_instances(instances_filename, vocab, labels_to_id):
    instances = []
    reader = csv.DictReader(open(instances_filename, 'r'))
    for row in reader:
        text = process_raw_text(row['text'])
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
    criterion):
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
    loss = criterion(classifier_output, instance_label)
    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()

    return loss

# Trains the model over training_instances for a given number of epochs
def train_epochs(training_instances, dev_instances, encoder, classifier, vocab, labels_to_id, out_filepath, weight_filepath, preds_filepath,
    n_epochs=30, print_every=500, learning_rate=.1):
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    dev_inputs = [x[0] for x in dev_instances]
    dev_labels = [x[1] for x in dev_instances]

    print_loss_total = 0

    best_dev_score = 0

    starting_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") # starting timestamp

    for i in range(n_epochs):
        print_epoch_loss_total = 0
        print 'Training epoch ' + str(i)
        random.shuffle(training_instances)
        for j, instance in enumerate(training_instances):
            loss = update_model(instance, encoder, encoder_optimizer, classifier, classifier_optimizer,
                criterion)
            print_loss_total += loss
            print_epoch_loss_total += loss

            if j !=0 and j % print_every == 0:
                print_loss_avg = print_loss_total/print_every
                print_loss_total = 0
                print('Epoch %d iteration %d loss: %.4f' % (i, j, print_loss_avg))

        print_epoch_loss_avg = print_epoch_loss_total/len(training_instances)
        print('Epoch %d avg loss: %.4f' % (i, print_epoch_loss_avg))
        predicted_dev_labels = classify(dev_inputs, encoder, classifier, vocab, labels_to_id, weight_filepath, preds_filepath)
        #acc = evaluate_accuracy(dev_labels, predicted_dev_labels)
        prec, rec, f1 = evaluate_f1(dev_labels, predicted_dev_labels)
        
        #print('Epoch %d dev accuracy: %.4f' % (i, score))
        print('Epoch %d dev f1: %.4f' % (i, f1))
        print('Epoch %d dev precision: %.4f' % (i, prec))
        print('Epoch %d dev recall: %.4f' % (i, rec))
        print('----------------------------------')

        score = f1

        if score > best_dev_score:
            best_dev_score = score

            # Save encoder
            torch.save(encoder, out_filepath + 'encoder_{}.model'.format(starting_ts))

            # Save classifier
            torch.save(classifier, out_filepath + 'classifier_{}.model'.format(starting_ts))

# Runs the model as a classifier on the given instance_inputs
def classify(instance_inputs, encoder, classifier, vocab, labels_to_id, weight_filepath, preds_filepath):

    predicted_labels = []
    attention_weights = []

    for instance_input in instance_inputs:
        encoder_hidden = encoder.init_hidden()
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

    # Save attn weights
    with open(weight_filepath, 'wb') as f:
        pickle.dump(attention_weights, f)

    # Save predictions
    preds = [p.data.cpu().tolist()[0] for p in predicted_labels]
    with open(preds_filepath, 'wb') as f:
        pickle.dump(preds, f)
            
    return predicted_labels

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

    return prec, rec, f1

def load_model(model_path, model_ts):
    clf_path = "{}classifier_{}.model".format(model_path, model_ts)
    encoder_path = "{}encoder_{}.model".format(model_path, model_ts)
    if os.path.exists(clf_path) and os.path.exists(encoder_path):
        encoder = torch.load(encoder_path)
        classifier = torch.load(clf_path)

    return (encoder, classifier)

def main():
#    training_filename = 'data/davidson/debug.csv'
#    dev_filename = 'data/davidson/debug.csv'

    training_filename = 'data/davidson/train.csv'
    dev_filename = 'data/davidson/dev.csv'

    test_filename = 'data/davidson/test.csv' 

    dataset_name = os.path.split(os.path.dirname(training_filename))[1] # parent dir of training filename
    fold_name = os.path.splitext(os.path.basename(dev_filename))[0] # to examine predictions
    out_filepath =  'models/{}_'.format(dataset_name) # path to save the model to
    weight_filepath = 'output/{}_{}_attn.pkl'.format(dataset_name, fold_name) # filepath for attention weights
    preds_filepath = 'output/{}_{}_preds.pkl'.format(dataset_name, fold_name) # filepath for predictions

    # Argparse
    parser = argparse.ArgumentParser(description='Train model to identify hate speech.')
    parser.add_argument('--load-model', nargs='?', dest='load', help='timestamp of model to load in format YYYY-MM-DDTHH-MM-SS', default='')
    args = parser.parse_args()

    training_corpus = read_corpus_file(training_filename)
    vocab = Vocab()
    vocab.build_vocab(training_corpus)

    training_instances = process_instances(training_filename, vocab, labels_to_id)
    dev_instances = process_instances(dev_filename, vocab, labels_to_id)
    """
    test_instances = process_instances(test_filename, vocab, labels_to_id)
    """

    # If loading an existing model
    if args.load:
        encoder, classifier = load_model(out_filepath, args.load)

    else:
        encoder = BidirectionalEncoder(vocab.size(), HIDDEN_DIM)
        classifier = AttentionClassifier(len(labels_to_id), HIDDEN_DIM)

    if use_cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    train_epochs(training_instances, dev_instances, encoder, classifier, vocab, labels_to_id, out_filepath, weight_filepath, preds_filepath, print_every=200)

if __name__ == '__main__': main()
