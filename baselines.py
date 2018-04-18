import os
import sys
import numpy as np
import pandas as pd
import html
import pdb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from nltk.tokenize import TweetTokenizer


class DataHandler():

    def __init__(self):
        self.folds = ['train', 'dev', 'test']
        self.data_dirpath = None
        self.data = None
        self.outcome_colname = None


    def load_data(self, data_dirpath):
        self.data_dirpath = data_dirpath
        self.data = {}
        for f in self.folds:
            self.data[f] = pd.read_csv(os.path.join(self.data_dirpath, f'{f}.csv'))

    
    def process_data(self, text_colname, outcome_colname,
                    multiclass_transform=None, feats='unigrams'):
        """ Preprocesses, vectorizes data """
    
        self.outcome_colname = outcome_colname
        self.text_colname = text_colname

        # Use tweet tokenizer
        tokenizer = TweetTokenizer()
        text_data = {}
        for f in self.folds:
            text_data[f] = self.data[f][self.text_colname].map(lambda x: ' '.join(tokenizer.tokenize(x.lower()))).tolist()

        if feats=='unigrams':
            vec = CountVectorizer(ngram_range=(1,1))
        elif feats=='bigrams':
            vec = CountVectorizer(ngram_range=(1,2))
        else:
            raise ValueError("No feature types specified")
        vec.fit(text_data['train'])

        bow = {}
        bow['train'] = vec.transform(text_data['train'])
        bow['dev'] = vec.transform(text_data['dev'])
        bow['test'] = vec.transform(text_data['test'])

        labels = {}
        for f in self.folds:
            if multiclass_transform is not None:
                labels[f] = [multiclass_transform[val] for val in self.data[f][self.outcome_colname].tolist()]
            else:
                labels[f] = self.data[f][self.outcome_colname].tolist()

        return bow['train'], labels['train'], bow['dev'], labels['dev'], bow['test'], labels['test']


def evaluate(preds, y, multiclass=False, labels=None):

    if multiclass:
        prec = precision_score(preds, y, average='micro', labels=labels)
        rec = recall_score(preds, y, average='micro', labels=labels)
        f1 = f1_score(preds, y, average='micro', labels=labels)

    else:
        prec = precision_score(preds, y)
        rec = recall_score(preds, y)
        f1 = f1_score(preds, y)

    acc = accuracy_score(preds, y)

    return prec, rec, f1, acc


def main():

    # Settings
    #dataset = 'davidson'
    dataset = 'zeerak_naacl'

    #multiclass = False
    multiclass = True
    labels = [1,2] # for multiclass evaluation

    #feats = 'unigrams'
    feats = 'bigrams'

    base_dirpath = '/usr0/home/mamille2/11-830-Final-Project/data/' # for misty
    #base_dirpath = '/usr2/mamille2/11-830-Final-Project/data/' # for erebor
    data_dirpath = os.path.join(base_dirpath, dataset)

    dh = DataHandler()

    print("Loading data...", end=" ")
    sys.stdout.flush()
    dh.load_data(data_dirpath)
    print("done.")
    sys.stdout.flush()

    print("Processing data...", end=" ")
    sys.stdout.flush()
    
    if multiclass:
        if dataset == 'davidson':
            X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'label', feats=feats, multiclass_transform = {'neither': 0, 'offensive_language': 1, 'hate_speech': 2})

        elif dataset == 'zeerak_naacl':
            X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'label', feats=feats, multiclass_transform = {'none': 0, 'racism': 1, 'sexism': 2}) 

    else:
        if dataset=='davidson':
            X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'hate_speech', feats=feats) 

        elif dataset == 'zeerak_naacl':
            X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'label', feats=feats, multiclass_transform = {'none': 0, 'racism': 1, 'sexism': 1}) 

    print("done.")
    sys.stdout.flush()

    print("Training classifier...", end=" ")
    sys.stdout.flush()
    if multiclass:
        clf = OneVsRestClassifier(LogisticRegression())
    else:
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("done.")
    sys.stdout.flush()

    print("Evaluating classifier...", end=" ")
    sys.stdout.flush()
    preds = clf.predict(X_test)

    if multiclass:
        prec, rec, f1, acc = evaluate(preds, y_test, multiclass=multiclass, labels=labels)

    else:
        prec, rec, f1, acc = evaluate(preds, y_test, multiclass=multiclass)

    print("done.")
    sys.stdout.flush()

    print()
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1: {f1}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    main()
