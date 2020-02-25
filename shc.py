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

def senti(x):
    return TextBlob(x).sentiment
df['CommentString'] = df['Comment'].astype(str)
df['senti_score'] = df['CommentString'].apply(senti)
df['Polarity'] = df['senti_score'].apply(lambda x: x[0])
df['Subjectivity'] = df['senti_score'].apply(lambda x: x[1])



def remove_punctuation(text):
	no_punct = "".join([c for c in text if c not in string.punctuation])
	return no_punct

df['Comment'] = df['Comment'].apply(lambda x: remove_punctuation(x))

#I nstantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

df['Comment'] = df['Comment'].apply(lambda x: tokenizer.tokenize(x.lower()))

def remove_stopwords(text):
	words = [w for w in text if w not in stopwords.words('english')]
	return words

df['Comment'] = df['Comment'].apply(lambda x : remove_stopwords(x))



df.drop(['CommentString', 'senti_score'], axis=1, inplace=True)

# print(df['senti_score'][3][0])
s = df.explode('Comment')
def posTag(x):
	return TextBlob(x).tags
s['CommentString'] = s['Comment'].astype(str)
s['tags'] = s['CommentString'].apply(posTag)
s.to_excel("explodedComs3.xlsx", index=False)

#1st_col = 'commentz'

#r = pd.DataFrame({
#		col:np.repeat(df[col1].values, df[1st_col].str.len())
#		for col in df.columns.drop(1st_col)}
#	).assign(**{1st_col:np.concatenate(df[1st_col].values)})[df.columns]

#print(r.head(20))
# Instantiate lemmatizer
#lemmatizer = WordNetLemmatizer()

#def word_lemmatizer(text):
#	lem_text = [lemmatizer.lemmatize(i) for i in text]
#	return lem_text

#df['Comment'].apply(lambda x: word_lemmatizer(x))

# Instatiate Stemmer
#stemmer = PorterStemmer()

#def word_stemmer(text):
#	stem_text = " ".join([stemmer.stem(i) for i in text])
#	return stem_text

#df['Comment'] = df['Comment'].apply(lambda x: word_stemmer(x))
