import numpy as np
import pandas as pd
import re

from nltk.stem.porter import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from nltk.corpus import stopwords

def data_clean(dt):
    
    dt          = dt.fillna('')
    dt["total"] = dt['title'] + " " + dt["author"]  
    return dt

def text_cleaning(df) :
    
    ps = PorterStemmer()
    corpus = []
    for i in range(len(df)):
        m = re.sub("[^a-zA-Z]", " ", df["total"][i])
        m = m.lower()
        m = m.split()
        m = [ps.stem(word) for word in m if not word in stopwords.words('english')]
        clean_text = " ".join(m)
        corpus.append(clean_text)
        
    return corpus

def onehot(corpus, VOCAB_SIZE = 5000):
    return [one_hot(words, VOCAB_SIZE) for words in corpus]

def padding(onehot_text) :
    return np.array(pad_sequences(onehot_text, padding = "pre", maxlen = 25))

def get_label(df):
    return np.array(df["label"])

def make_clean(df) :
        
    corpus = data_clean(df)
    corpus = text_cleaning(corpus)
    corpus = onehot(corpus)
    corpus = padding(corpus)
    return   corpus

def pp(text) :
    
    ps = PorterStemmer()
    m = re.sub("[^a-zA-Z]", " ", text)
    m = m.lower()
    m = m.split()
    m = [ps.stem(word) for word in m if not word in stopwords.words('english')]
    clean_text = " ".join(m)
    corpus = onehot(clean_text)
    temp = []
    for element in corpus :
        if len(element) == 0 :
            temp.append(0)
        else : 
            temp.append(element[0])
    corpus = padding([temp])
    return   corpus


def pp_ml(text) :
    
    ps = PorterStemmer()
    m = re.sub("[^a-zA-Z]", " ", text)
    m = m.lower()
    m = m.split()
    m = [ps.stem(word) for word in m if not word in stopwords.words('english')]
    clean_text = " ".join(m)
    corpus = onehot(clean_text)
    temp = []
    for element in corpus :
        if len(element) == 0 :
            temp.append(0)
        else : 
            temp.append(element[0])
            
    return np.array(pad_sequences([temp], padding = "pre", maxlen = 30))
    