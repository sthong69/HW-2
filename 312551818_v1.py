# Deep Neural Network
# Long Short-Term Memory
# Bidirectional Encoder Representations from Transformers

import json

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import string as strg
from sklearn.model_selection import train_test_split

import xml
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

train_set = "train.json"
test_set = "train.json"

with open(train_set, "r") as json_file:
    data_test = json.load(json_file)

def nltk_pipeline(string):
    # Tokenize
    res = nltk.word_tokenize(string)

    # Remove punctuation
    res = [_ for _ in res if _ not in strg.punctuation]

    # Lowercase
    res = [_.lower() for _ in res]

    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    res = [lemmatizer.lemmatize(_) for _ in res]

    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    #res =  [_ for _ in res if _ not in stopwords]

    # Remove HTML tags

    # Untokenize
    alt_res = ""
    for token in res:
        alt_res += token
    res = alt_res

    return res

def extract_features(data, list):
    res = []
    for i in range(len(list[0])):
        feature = []
        for j in range(len(data)):
            feature.append(data[j][list[0][i]])
        res.append(feature)
    return res        

def clean_features(list, index_to_clean):
    res = []
    for i in range(len(list)):
        if i in index_to_clean:
            feature = []
            for j in range(len(list[i])):
                feature.append(nltk_pipeline(list[i][j]))
        else:
            feature = list[i].copy()
        res.append(feature)
    return res



features_list = ['text', 'rating'], 

X = clean_features(extract_features(data_test,features_list),[0])

X = pd.DataFrame(np.transpose(X), columns=features_list)

print(X)