#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
# Download required NLTK data
nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import re, string

# Initialize lemmatizer and stop words set
document_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def process_tokens(tweet_tokens):
    """
    Cleans and lemmatizes tweet tokens:
      - lowercases
      - removes URLs, mentions, hashtags
      - filters out stop words & pure punctuation
      - lemmatizes based on POS tags
    """
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        tok = token.lower()

        # drop URLs, mentions, hashtags
        if re.match(r'https?://\S+', tok) or tok.startswith('@') or tok.startswith('#'):
            continue
        # drop stop words
        if tok in stop_words:
            continue
        # drop pure punctuation
        if all(ch in string.punctuation for ch in tok):
            continue

        # map POS tag to WordNet tag
        if tag.startswith('N'):
            wn_tag = 'n'
        elif tag.startswith('V'):
            wn_tag = 'v'
        elif tag.startswith('J'):
            wn_tag = 'a'
        else:
            wn_tag = 'n'

        # lemmatize and collect
        lemma = document_lemmatizer.lemmatize(tok, wn_tag)
        cleaned_tokens.append(lemma)

    return cleaned_tokens


# Prepare cleaned token lists for tweets
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = [process_tokens(tokens) for tokens in positive_tweet_tokens]
negative_cleaned_tokens_list = [process_tokens(tokens) for tokens in negative_tweet_tokens]

