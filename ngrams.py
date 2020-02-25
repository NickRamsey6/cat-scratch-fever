import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob


df = pd.read_csv('shcComs.csv')

def senti(text):
    return TextBlob(text).sentiment
df['CommentString'] = df['Comment'].astype(str)
df['senti_score'] = df['CommentString'].apply(senti)
df['Polarity'] = df['senti_score'].apply(lambda x: x[0])
df['Subjectivity'] = df['senti_score'].apply(lambda x: x[1])


def remove_punctuation(text):
	no_punct = "".join([c for c in text if c not in string.punctuation])
	return no_punct

df['Comment'] = df['Comment'].apply(lambda x: remove_punctuation(x))

# def remove_stopwords(text):
# 	words = [w for w in text if w not in stopwords.words('english')]
# 	return words
#
# df['Comment'] = df['Comment'].apply(lambda x : remove_stopwords(x))

def ngram_getter(text):
    return TextBlob(text).ngrams(n=2)
df['CommentString'] = df['Comment'].astype(str)
df['ngrams'] = df['CommentString'].apply(ngram_getter)

# ngramFreq = collection.Counter(df['ngrams'])
# print(ngramFreq.most_common(10))

# print(df['ngrams'])

s = df.explode('ngrams')
s.to_excel("explodedBiGrams.xlsx", index=False)
