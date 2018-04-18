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


def evaluate(preds, y):

    prec = precision_score(preds, y, average='micro')
    rec = recall_score(preds, y, average='micro')
    f1 = f1_score(preds, y, average='micro')
    acc = accuracy_score(preds, y)

    return prec, rec, f1, acc


def main():

    base_dirpath = '/usr0/home/mamille2/11-830-Final-Project/data/' # for misty
    #base_dirpath = '/usr2/mamille2/11-830-Final-Project/data/' # for erebor
    #data_dirpath = os.path.join(base_dirpath, 'zeerak_naacl')
    data_dirpath = os.path.join(base_dirpath, 'davidson')

    dh = DataHandler()

    clf_type = 'multiclass'
    #clf_type = 'binary'

    print("Loading data...", end=" ")
    sys.stdout.flush()
    dh.load_data(data_dirpath)
    print("done.")
    sys.stdout.flush()

    print("Processing data...", end=" ")
    sys.stdout.flush()
    #X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'hate')
    #X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('text', 'hate_speech') # davidson, binary classification

    # davidson, multiclass
    X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('text', 'label', feats='bigrams', multiclass_transform = {'neither': 0, 'offensive_language': 1, 'hate_speech': 2})
    print("done.")
    sys.stdout.flush()

    print("Training classifier...", end=" ")
    sys.stdout.flush()
    if clf_type == 'multiclass':
        clf = OneVsRestClassifier(LogisticRegression())
    else:
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("done.")
    sys.stdout.flush()

    print("Evaluating classifier...", end=" ")
    sys.stdout.flush()
    #preds = clf.predict(X_dev)
    #prec, rec, f1, acc = evaluate(preds, y_dev)
    preds = clf.predict(X_test)
    prec, rec, f1, acc = evaluate(preds, y_test)
    print("done.")
    sys.stdout.flush()

    print()
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1: {f1}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    main()
