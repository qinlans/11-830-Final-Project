import os
import pandas as pd
import re
import pdb


# Settings
base_dirpath = '/usr0/home/mamille2'
data_dirpath = os.path.join(base_dirpath, '11-830-Final-Project/data')
dataset = 'zeerak_naacl'
slurpath = os.path.join(data_dirpath, 'hatebase_slurs.txt')


# Load data
folds = ['train', 'dev', 'test']
data = {}
for f in folds:
    data[f] = pd.read_csv(os.path.join(data_dirpath, dataset, f'{f}.csv'))
    

# Load slurs
with open(slurpath) as f:
    slurs = f.read().splitlines()


# # Unk/remove slurs from datasets

# UNK
print("Unking slurs...")
slurs_p = re.compile(r'|'.join(slurs))

for f in folds:
    data[f]['tweet_unk_slur'] = data[f]['tweet'].map(lambda x: re.sub(slurs_p, '<UNK>', x))


# Remove
print("Removing slurs...")
for f in folds:
    data[f]['tweet_no_slur'] = data[f]['tweet'].map(lambda x: re.sub(r'\s+', ' ', re.sub(slurs_p, '', x)))


# Save out
for f in folds:
    data[f].to_csv(os.path.join(data_dirpath, dataset, f'{f}.csv'), index=False)
print("Saved files.")
