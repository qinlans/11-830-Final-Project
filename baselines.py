import os
import sys
import numpy as np
import pandas as pd
import html
import pdb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support

from nltk.tokenize import TweetTokenizer


class DataHandler():

    def __init__(self):
        self.folds = ['train', 'dev', 'test']
        self.data_dirpath = None
        self.data = None
        self.outcome_colname = None


    def load_data(self, train_data_dirpath, test_data_dirpath, train_prefix=None, test_prefix=None):
        self.data_dirpath = {'train': train_data_dirpath,
                            'dev': train_data_dirpath,
                            'test': test_data_dirpath}
        self.data = {}
        if train_prefix and test_prefix:
            self.data['train'] = pd.read_csv(os.path.join(self.data_dirpath['train'], f'{train_prefix}_train.csv'))
            self.data['dev'] = pd.read_csv(os.path.join(self.data_dirpath['dev'], f'{train_prefix}_dev.csv'))
            self.data['test'] = pd.read_csv(os.path.join(self.data_dirpath['test'], f'{test_prefix}.csv'))
        else:
            for f in self.folds:
                self.data[f] = pd.read_csv(os.path.join(self.data_dirpath[f], f'{f}.csv'))

    
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
                labels[f] = [multiclass_transform[f][val] for val in self.data[f][self.outcome_colname].tolist()]
            else:
                labels[f] = self.data[f][self.outcome_colname].tolist()

        return bow['train'], labels['train'], bow['dev'], labels['dev'], bow['test'], labels['test']

# From attention.py
def evaluate(y, y_pred, labels_to_id, return_all=True, categorical=False):
    """Compute the performance on the data."""

    id_to_labels = {v: k for k, v in labels_to_id.items()}

    # Set up the output DataFrame
    index = [id_to_labels[v] for v in list(set(y))] + ['weighted_average']
    columns = ['precision', 'recall', 'f1_score', 'accuracy', 'support']
    results = pd.DataFrame(index=index, columns=columns)
    
    # Compute everything
    acc = accuracy_score(y, y_pred)
    res = precision_recall_fscore_support(y, y_pred)
    #res_weight = precision_recall_fscore_support(y, y_pred, average='weighted')
    res_weight = precision_recall_fscore_support(y, y_pred, average='macro')
    
    # Compile result numbers
    prec = np.concatenate([res[0], [res_weight[0]]])
    rec = np.concatenate([res[1], [res_weight[1]]])
    f1 =  np.concatenate([res[2], [res_weight[2]]])
    sup = np.concatenate([res[3], [sum(res[3])]])
    
    # Put into results and return
    results['accuracy']['average'] = acc
    results['precision'] = prec
    results['recall'] = rec
    results['f1_score'] = f1
    results['support'] = sup

    return results
    

#def evaluate(preds, y, multiclass=False, labels=None):
#
#    if multiclass:
#        #prec = precision_score(preds, y, average='micro', labels=labels)
#        #rec = recall_score(preds, y, average='micro', labels=labels)
#        #f1 = f1_score(preds, y, average='micro', labels=labels)
#        prec = precision_score(preds, y, average='weighted')
#        rec = recall_score(preds, y, average='weighted')
#        f1 = f1_score(preds, y, average='weighted')
#
#    else:
#        prec = precision_score(preds, y)
#        rec = recall_score(preds, y)
#        f1 = f1_score(preds, y)
#
#    acc = accuracy_score(preds, y)
#
#    return prec, rec, f1, acc


def main():

    # Settings
    #train_dataset = 'davidson'
    train_dataset = 'zeerak_naacl'
    #test_dataset = 'davidson'
    test_dataset = 'zeerak_naacl'

    #train_prefix = 'racism'
    train_prefix = 'sexism'
    #train_prefix = None

    #test_prefix = 'sexism'
    test_prefix = 'racism'
    #test_prefix = None

    multiclass = False
    #multiclass = True
    #labels = [1,2] # for multiclass evaluation

    feats = 'unigrams'
    #feats = 'bigrams'

    base_dirpath = '/usr0/home/mamille2/11-830-Final-Project/data/' # for misty
    #base_dirpath = '/usr2/mamille2/11-830-Final-Project/data/' # for erebor
    train_data_dirpath = os.path.join(base_dirpath, train_dataset)
    test_data_dirpath = os.path.join(base_dirpath, test_dataset)

    dh = DataHandler()

    print("Loading data...", end=" ")
    sys.stdout.flush()
    dh.load_data(train_data_dirpath, test_data_dirpath, train_prefix=train_prefix, test_prefix=test_prefix)
    print("done.")
    sys.stdout.flush()

    print("Processing data...", end=" ")
    sys.stdout.flush()
    
    if multiclass:
        if train_dataset == 'davidson':
            multiclass_transform = {'train': {'neither': 0, 'offensive_language': 1, 'hate_speech': 2}, 
                                    'dev': {'neither': 0, 'offensive_language': 1, 'hate_speech': 2},
                                    'test': {'neither': 0, 'offensive_language': 1, 'hate_speech': 2}}
        elif train_dataset == 'zeerak_naacl':
            multiclass_transform = {'train': {'none': 0, 'racism': 1, 'sexism': 2},
                                    'dev': {'none': 0, 'racism': 1, 'sexism': 2},
                                    'test': {'none': 0, 'racism': 1, 'sexism': 2}}

    else:
        multiclass_transform = {}
        if train_dataset=='davidson':
            multiclass_transform['train'] = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}
            multiclass_transform['dev'] = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

        elif train_dataset == 'zeerak_naacl':
            multiclass_transform['train'] = {'none': 0, 'racism': 1, 'sexism': 1}
            multiclass_transform['dev'] = {'none': 0, 'racism': 1, 'sexism': 1}

        if test_dataset=='davidson':
            multiclass_transform['test'] = {'neither': 0, 'offensive_language': 0, 'hate_speech': 1}

        elif test_dataset=='zeerak_naacl':
            multiclass_transform['test'] = {'none': 0, 'racism': 1, 'sexism': 1}

    X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'label', 
                    feats=feats, multiclass_transform=multiclass_transform) 

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

    print("Evaluating classifier...")
    sys.stdout.flush()
    preds = clf.predict(X_test)

    #prec, rec, f1, acc = evaluate(preds, y_test , categorical=multiclass)
    results = evaluate(preds, y_test, labels_to_id=multiclass_transform['test'], categorical=multiclass)
    print(results)

    print("done.")
    sys.stdout.flush()

    print()
    #print(f'Precision: {prec}')
    #print(f'Recall: {rec}')
    #print(f'F1: {f1}')
    #print(f'Accuracy: {acc}')


if __name__ == '__main__':
    main()
