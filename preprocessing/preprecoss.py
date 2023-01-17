import re

import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

def clean_text(raw):
    # Removes special symbols and just keep
    # words in lower or upper form
    raw = [x.lower() for x in raw]
    raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in raw]
    return raw

def text_tokenization(raw):
    # Tokenizes each sentence by implementing the nltk tool
    raw = [word_tokenize(x) for x in raw]
    return raw

def build_vocabulary(raw):
    # Builds the vocabulary and keeps the "x" most frequent word
    num_words = 0
    vocabulary = dict()
    fdist = nltk.FreqDist()
    
    for sentence in raw:
      for word in sentence:
        fdist[word] += 1
        
    common_words = fdist.most_common(num_words)
    
    for idx, word in enumerate(common_words):
        vocabulary[word[0]] = (idx+1)

    return vocabulary

def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def count_unique_words(text):
    return len(set(text.split()))

def count_sent(text):
    return len(nltk.sent_tokenize(text))

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

def stopwords_vs_count_words(df):
    df['stopwords_vs_words'] = df['stopword_count']/df['word_count']
    return df

def main():
    train = pd.read_csv("../data/raw/train.csv")
    train['Character Count Title'] = train['Title'].apply(count_chars)
    train['Character Count Description'] = train['Description'].apply(count_chars)
    train['Word Count Title'] = train['Title'].apply(count_words)
    train['Word Count Description'] = train['Description'].apply(count_words)
    train['Sentence Count Description'] = train['Description'].apply(count_sent)
    train['Unique Word Count Title'] = train['Title'].apply(count_unique_words)
    train['Unique Count Description'] = train['Description'].apply(count_unique_words)
    train['Avg Word Length Title'] = train['Character Count Title'] / train['Word Count Title']
    train['Avg Word Length Description'] = train['Character Count Description'] / train['Word Count Description']
    train['Avg Sentence Length Description'] = train['Word Count Description'] / train['Sentence Count Description']
    train['Unique Words vs Word Count Title'] = train['Unique Word Count Title'] / train['Word Count Title']
    train['Unique Words vs Word Count Description'] = train['Unique Count Description'] / train['Word Count Description']
    
    test = pd.read_csv("../data/raw/test.csv")
    test['Character Count Title'] = test['Title'].apply(count_chars)
    test['Character Count Description'] = test['Description'].apply(count_chars)
    test['Word Count Title'] = test['Title'].apply(count_words)
    test['Word Count Description'] = test['Description'].apply(count_words)
    test['Sentence Count Description'] = test['Description'].apply(count_sent)
    test['Unique Word Count Title'] = test['Title'].apply(count_unique_words)
    test['Unique Count Description'] = test['Description'].apply(count_unique_words)
    test['Avg Word Length Title'] = test['Character Count Title'] / test['Word Count Title']
    test['Avg Word Length Description'] = test['Character Count Description'] / test['Word Count Description']
    test['Avg Sentence Length Description'] = test['Word Count Description'] / test['Sentence Count Description']
    test['Unique Words vs Word Count Title'] = test['Unique Word Count Title'] / test['Word Count Title']
    test['Unique Words vs Word Count Description'] = test['Unique Count Description'] / test['Word Count Description']

    train.to_csv('../data/processed/train.csv', index=False)
    test.to_csv('../data/processed/test.csv', index=False)

if __name__ == "__main__":
    main()