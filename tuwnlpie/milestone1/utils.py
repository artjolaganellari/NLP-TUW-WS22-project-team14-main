import csv
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk.metrics import precision,recall,ConfusionMatrix
import collections
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import seaborn as sns

import pandas as pd


def read_docs_from_csv(filename):
    docs = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for text, label in tqdm(reader):
            words = nltk.word_tokenize(text)
            docs.append((words, label))

    return docs


def split_train_dev_test(docs, train_ratio=0.8, dev_ratio=0.1):
    np.random.seed(2022)
    np.random.shuffle(docs)
    train_size = int(len(docs) * train_ratio)
    dev_size = int(len(docs) * dev_ratio)
    return (
        docs[:train_size],
        docs[train_size : train_size + dev_size],
        docs[train_size + dev_size :],
    )

def split_train_test_sklearn(df):
    return train_test_split(df['sentence'],df['target'],test_size=0.2,random_state=1024)


def calculate_tp_fp_fn(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp += 1
        else:
            if true == "positive":
                fn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, fscore


def lemmatize_dataset(df):
    stop_words = list(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([word for word in word_tokenize(x, language='english') if re.match('\w', word)]))
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))
        
    return df

def prepare_dataset(in_path):
    df = pd.read_csv(in_path, delimiter=',')
    #drop columns we do not need and
    #Transform boolean is_cause and is_treat to label variable with 3 labels ("cause","treat" and "neutral")
    df['target']= 'neutral'
    df.loc[df[df.is_cause == 1.0].index.values, 'target'] = 'cause'
    df.loc[df[df.is_treat == 1.0].index.values, 'target'] = 'treat'
    
    df = df.loc[:, ['sentence', 'target']]

    return df


def transform_to_dictionary(df):
    docs = []
    for index,row in df.iterrows():
        words = row['sentence'] #tokenizing is done in the next step
        docs.append((words, row['target']))

    all_words = set(word.lower() for sentence in docs for word in word_tokenize(sentence[0]))
    t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in docs]
    return t


def train_NaiveBayes(df):
    return NaiveBayesClassifier.train(df)

def show_features(model):
    return model.show_most_informative_features(30)

def evaluate_nltk_nb(model,testdata):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    labels = []
    tests = []
    for i, (text, label) in enumerate(testdata):
        refsets[label].add(i)
        observed = model.classify(text)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)

    print('Overall Accuracy:', nltk.classify.accuracy(model, testdata))
    print('Cause precision:', nltk.precision(refsets['cause'], testsets['cause']))
    print('Cause recall:', nltk.recall(refsets['cause'], testsets['cause']))
    print('Cause F-measure:', nltk.f_measure(refsets['cause'], testsets['cause']))
    print('Treat precision:', nltk.precision(refsets['treat'], testsets['treat']))
    print('Treat recall:', nltk.recall(refsets['treat'], testsets['treat']))
    print('Treat F-measure:', nltk.f_measure(refsets['treat'], testsets['treat']))
    print('Neutral precision:', nltk.precision(refsets['neutral'], testsets['neutral']))
    print('Neutral recall:', nltk.recall(refsets['neutral'], testsets['neutral']))
    print('Neutral F-measure:', nltk.f_measure(refsets['neutral'], testsets['neutral']))
    print(nltk.ConfusionMatrix(labels,tests))

def test_prediction_for_new_sentence(df,model,text):
    docs = []
    for index,row in df.iterrows():
        words = row['sentence'] #tokenizing is done in the next step
        docs.append((words, row['target']))

    all_words = set(word.lower() for sentence in docs for word in word_tokenize(sentence[0]))
    doc1={word: (word in nltk.word_tokenize(text.lower())) for word in all_words}
    return model.classify(doc1)


def train_multi_nb_classifier(X_train,y_train):
    multi_nb_classifier = Pipeline([('v', CountVectorizer()), ('tfidf', TfidfTransformer()), ('mnb', MultinomialNB())])
    return(multi_nb_classifier.fit(X_train, y_train))

def train_svm_classifier(X_train,y_train):
    svm_classifier = Pipeline([('v', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SGDClassifier(random_state=1024))])
    return(svm_classifier.fit(X_train, y_train))

def undersampling_dataset(df):
    n = 150
    msk = df.groupby('target')['target'].transform('size') >= n
    df = pd.concat((df[msk].groupby('target').sample(n=n), df[~msk]), ignore_index=True)
    return df