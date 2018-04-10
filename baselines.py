import os
import numpy as np
import pandas as pd
import html
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
            self.data[f] = pd.read_csv(os.path.join(self.data_dirpath, f'{f}_utf8.csv'))

    
    def process_data(self, text_colname, outcome_colname):
        """ Preprocesses, vectorizes data """
    
        self.outcome_colname = outcome_colname
        self.text_colname = text_colname

        # Use tweet tokenizer
        tokenizer = TweetTokenizer()
        text_data = {}
        for f in self.folds:
            text_data[f] = self.data[f][self.text_colname].map(lambda x: ' '.join(tokenizer.tokenize(x.lower()))).tolist()

        vec = CountVectorizer(ngram_range=(1,2))
        vec.fit(text_data['train'])

        bow = {}
        bow['train'] = vec.transform(text_data['train'])
        print(bow['train'].shape)
        bow['dev'] = vec.transform(text_data['dev'])
        print(bow['dev'].shape)

        labels = {}
        for f in self.folds:
            labels[f] = self.data[f][self.text_colname].tolist()

        return bow['train'], labels['train'], bow['dev'], labels['dev']


def evaluate(preds, y):

    #correct_hs = 0
    #correct_athg = 0
    #pred_hs = sum(preds)
    #actual_hs = sum(gold)
    #total = len(preds)

    #for tl, pl in zip(preds, gold):
    #    if tl == pl == 1:
    #        correct_hs += 1
    #        correct_athg += 1
    #    elif tl == pl == 0:
    #        correct_athg += 1
    #        
    #prec = correct_hs/pred_hs
    #rec = correct_hs/actual_hs
    #f1 = 2 * prec * rec / (prec + rec)
    #acc = correct_athg/total

    prec = precision_score(preds, y)
    rec = recall_score(preds, y)
    f1 = f1_score(preds, y)
    acc = accuracy_score(preds, y)

    return prec, rec, f1, acc


def main():

    dh = DataHandler()

    print("Loading data...", end=" ")
    sys.stdout.flush()
    dh.load_data('/usr0/home/mamille2/11-830-Final-Project/data/zeerak_naacl/')
    print("done.")
    sys.stdout.flush()

    print("Processing data...", end=" ")
    sys.stdout.flush()
    X_train, y_train, X_dev, y_dev = dh.process_data('tweet', 'hate')
    print("done.")
    sys.stdout.flush()

    print("Training, evaluating classifiers...", end=" ")
    sys.stdout.flush()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_dev)
    print("done.")
    sys.stdout.flush()

    prec, rec, f1, acc = evaluate(preds, y_dev)

    print()
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1: {f1}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    main()
