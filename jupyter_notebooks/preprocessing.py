from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pickle
import html
import re

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('im')
clean_pattern = re.compile(
    "([^\s\w@_<>#"  # alphanumeric, '_', '@', '#', and '<', '>'
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)])+')
    "])+")
url_pattern = re.compile('http\S+')
MENTION = '<MENTION>'
HASHTAG = '<HASHTAG>'
URL = '<URL>'


def clean_tweets(df):
    """Clean the tweets in a dataframe."""
    # First we convert everything from html encoding to unicode
    print('html removal')
    df['tweet'] = [*map(html.unescape, df['tweet'])]
    # Then we save a version of the original tweet before processing
    print('url removal')
    df['original_tweet'] = df['tweet']
    df['tweet'] = [*map(replace_url, df['tweet'])]
    # Remvoe certain characters (as per )
    print('cleaning and extracting mentions/hashtags')
    df['tweet'] = [*map(strip_text, df['tweet'])]
    df['tweet'], df['mentions'], df['hashtags'] = \
        replace_all_mentions(df['tweet'])
    return df

def strip_text(text):
    """Strip text.

    Keeps: alphanumeric, [@, _, <, >, #], and emojis
    Removes: other special characters and the phrase 'RT'
    """
    text = clean_pattern.sub(' ', text)
    text = text.replace('RT', ' ')
    return ' '.join(text.split())


def replace_url(text, replace_with=URL):
    """Replace all URL's with a given string."""
    return url_pattern.sub(replace_with, text)


def replace_mentions(tweet, remove_stopwords=False):
    """Replace all mentions in a tweet with <MENTION>.
    
    Returns: the new tweet, and a list of mentions.
    """
    words = tweet.split(' ')
    mentions = []
    hashtags = []
    cleaned_tweet = []
    
    i = 0
    while(i < len(words)):
        w = words[i]
        if w[0] == '@':
            if len(w) == 1:
                try:
                    mentions.append('@{}'.format(words[i+1]))
                    i += 1
                except IndexError:
                    pass  # The '@' was at the end of the tweet
            else:
                mentions.append(w)
            cleaned_tweet.append(MENTION)
        elif w[0] == '#':
            if len(w) == 1:
                try:
                    hashtags.append('#{}'.format(words[i+1]))
                    i += 1
                except IndexError:
                    pass  # The '#' was at the end of the tweet
            else:
                hashtags.append(w)
            # cleaned_tweet.append(HASHTAG)
            cleaned_tweet.append('#')
            cleaned_tweet.append(w[1:].lower())
        else:
            if w[0] != '<':
                w = w.lower()
            if remove_stopwords and w not in STOPWORDS:
                cleaned_tweet.append(w)
            else:
                cleaned_tweet.append(w)
        i += 1

    return ' '.join(cleaned_tweet), mentions, hashtags



def replace_all_mentions(tweets):
    """Replace mentions for all tweets"""
    cleaned_tweets = []
    mentions_list = []
    hashtags_list = []

    for tweet in tweets:
        cleaned_tweet, mentions, hashtags = \
            replace_mentions(tweet, remove_stopwords=False)
        cleaned_tweets.append(cleaned_tweet)
        mentions_list.append(mentions)
        hashtags_list.append(hashtags)

    return cleaned_tweets, mentions_list, hashtags_list


def one_hot_encode(classes):
    """Returns a one-hot encoded matrix for the classes."""
    return np.array([*map(get_one_hot_vector, classes)], dtype=int)


def get_one_hot_vector(i, size=3):
    """Makes a onehot vector of a given size, with only index i as one."""
    vec = np.zeros(size)
    vec[i] = 1
    return vec


def make_debug_df(df, size=30, cols=['hate_speech', 'offensive_language', 'neither']):
    """Make a small, balanced DataFrame for debugging."""
    sample_size = int(size / 3)
    debug = None
    for col in cols:
        msk = df[col] == 1
        if debug is None:
            debug = df[msk].sample(sample_size)
        else:
            debug = debug.append(df[msk].sample(sample_size))
    return debug.sample(frac=1)