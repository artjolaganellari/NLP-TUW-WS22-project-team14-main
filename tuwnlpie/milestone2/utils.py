import csv
import time

import nltk
import pandas as pd
import torch
import re

from tuwnlpie import logger

# Set the optimizer and the loss function!
# https://pytorch.org/docs/stable/optim.fun
import torch.optim as optim
from torch.optim import Adam
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gc
from sklearn.model_selection import train_test_split as split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchtext
import numpy as np
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab as _Vocab
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
#from torchtext.data.functional import to_map_style_dataset
from tuwnlpie.milestone2.model import RNNClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy


def to_map_style_dataset(iter_data): #copied from torchtext version 13 because we use older version
    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):
        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)

# This is just for measuring training time!
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

 #function for transform booleans into target labels
def label_treats_causes(row):  
    if row['is_cause'] < 1 and row['is_treat'] > 0:
        return 'treat'
    elif row['is_cause'] > 0 and row['is_treat'] < 1:
        return 'cause'
    return 'neutral'
    
def build_vocabulary(datasets): #helper function for generating iterator for bulding vocabulary for train and test datasets
    tokenizer = get_tokenizer("basic_english")
    for dataset in datasets:
        for text, _ in dataset:
            yield tokenizer(text)



class Vocab(_Vocab):
    def lookup_indices(self, tokens):
        return [vocab[token] for token in tokens]

    def __call__(self, tokens):
        return self.lookup_indices(tokens)
        
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


class TreatDiseaseDataset:
    def __init__(self, data_path, BATCH_SIZE=64,MAX_WORDS=20):
        self.data_path = data_path
        # Initialize the correct device
        # It is important that every array should be on the same device or the training won't work
        # A device could be either the cpu or the gpu if it is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_WORDS = MAX_WORDS
        self.df = self.read_df_from_csv(self.data_path)
        self.df=self.transform(self.df)
        self.tr_data, self.te_data = self.split_data(self.df)

        self.tr_tuples,self.te_tuples=self.create_tuples(self.tr_data,self.te_data)
        #print(self.tr_tuples[5:10]) #uncomment to look at data structure
        self.vocab=self.build_vocab(self.tr_tuples,self.te_tuples)
        self.train_dataset,self.test_dataset = to_map_style_dataset(self.tr_tuples), to_map_style_dataset(self.te_tuples)
        self.target_classes = ["cause", "treat", "neutral"]
        
        self.VOCAB_SIZE = len(self.vocab)
        self.OUT_DIM = 3

        

        (
            self.train_iterator,
            self.test_iterator,
        ) = self.create_dataloader_iterators(
            self.train_dataset,
            self.test_dataset,
            self.BATCH_SIZE
        )

    def read_df_from_csv(self, path):
        df = pd.read_csv(path)
        
        return df
    
    
    def transform(self, df):
        
        df['label'] = df.apply(lambda x: label_treats_causes(x),axis=1)
        #drop columns we do not need
        df=df.drop(columns=['Unnamed: 0','disease_doid','food_entity','disease_entity','is_cause','is_treat'])
        labels = {"cause": 0, "treat": 1, "neutral":2}
        df["label"] = [labels[item] for item in df.label]
        stop_words = list(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    
        df['sentence'] = df['sentence'].apply(lambda x: ' '.join([word for word in nltk.word_tokenize(x, language='english') if re.match('\w', word)]))
        df['sentence'] = df['sentence'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))
    

        return df

    def split_data(self, train_data, random_seed=2022):
        tr_data, te_data = split(train_data, test_size=0.2, random_state=random_seed)

        return tr_data, te_data
        
    def create_tuples(self, train_data, test_data):
        # Create a list of tuples for Dataframe rows using list comprehension
        train_tuples = [tuple(row) for row in train_data.values]
        test_tuples = [tuple(row) for row in test_data.values]
        return train_tuples, test_tuples
        
    def build_vocab(self, train_data,test_data):
        vocab = build_vocab_from_iterator(build_vocabulary([train_data, test_data]))
        return(vocab)

    # def prepare_vectorizer(self, tr_data):
        # vectorizer = CountVectorizer(
            # max_features=3000, tokenizer=LemmaTokenizer(), stop_words="english"
        # )

        # word_to_ix = vectorizer.fit(tr_data.text)

        # return word_to_ix

    # # Preparing the data loaders for the training and the validation sets
    # # PyTorch operates on it's own datatype which is very similar to numpy's arrays
    # # They are called Torch Tensors: https://pytorch.org/docs/stable/tensors.html
    # # They are optimized for training neural networks
    # def prepare_dataloader(self, tr_data, val_data, te_data, word_to_ix, device):
        # # First we transform the text into one-hot encoded vectors
        # # Then we create Torch Tensors from the list of the vectors
        # # It is also inportant to send the Tensors to the correct device
        # # All of the tensors should be on the same device when training
        # tr_data_vecs = torch.FloatTensor(
            # word_to_ix.transform(tr_data.text).toarray()
        # ).to(device)
        # tr_labels = torch.LongTensor(tr_data.label.tolist()).to(device)

        # val_data_vecs = torch.FloatTensor(
            # word_to_ix.transform(val_data.text).toarray()
        # ).to(device)
        # val_labels = torch.LongTensor(val_data.label.tolist()).to(device)

        # te_data_vecs = torch.FloatTensor(
            # word_to_ix.transform(te_data.text).toarray()
        # ).to(device=device)
        # te_labels = torch.LongTensor(te_data.label.tolist()).to(device=device)

        # tr_data_loader = [
            # (sample, label) for sample, label in zip(tr_data_vecs, tr_labels)
        # ]
        # val_data_loader = [
            # (sample, label) for sample, label in zip(val_data_vecs, val_labels)
        # ]

        # te_data_loader = [
            # (sample, label) for sample, label in zip(te_data_vecs, te_labels)
        # ]

        # return tr_data_loader, val_data_loader, te_data_loader

    def vectorize_batch(self,batch):
        tokenizer = get_tokenizer("basic_english")
        max_words=20 #tested with different max_words sizes
        X, Y = list(zip(*batch))
        X = [self.vocab(tokenizer(text)) for text in X]
        X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] ## Bringing all samples to max_words length.

        return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)    
    # The DataLoader(https://pytorch.org/docs/stable/data.html) class helps us to prepare the training batches
    # It has a lot of useful parameters, one of it is _shuffle_ which will randomize the training dataset in each epoch
    # This can also improve the performance of our model
    def create_dataloader_iterators(
        self, tr_data, te_data, BATCH_SIZE
    ):
        train_iterator = DataLoader(
            tr_data,
            batch_size=BATCH_SIZE,
            collate_fn=self.vectorize_batch,
            shuffle=True,
        )

        test_iterator = DataLoader(
            te_data,
            batch_size=BATCH_SIZE,
            collate_fn=self.vectorize_batch,
            shuffle=False,
        )

        return train_iterator, test_iterator


def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for (sentence, sentence_len), labels in val_loader:
            preds = model(sentence)
            loss = loss_fn(preds, labels)
            losses.append(loss.item())

            Y_shuffled.append(labels)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


def TrainModel(model, optimizer, train_loader, val_loader, epochs=10):
    loss_fn = nn.CrossEntropyLoss()
    for i in range(1, epochs+1):
        losses = []
        for (sentence, sentence_len), labels in train_loader:
            Y_preds = model(sentence)

            loss = loss_fn(Y_preds, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        CalcValLossAndAccuracy(model, loss_fn, val_loader)

def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for (sentence, sentence_len), labels in loader:
        preds = model(sentence)
        Y_preds.append(preds)
        Y_shuffled.append(labels)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()
    
def EvaluatePerformance(model,loader,target_classes):
    Y_actual, Y_preds = MakePredictions(model, loader)
    print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
    print("\nClassification Report : ")
    print(classification_report(Y_actual, Y_preds, target_names=target_classes))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_actual, Y_preds))


def save_metrics(save_path, train_loss, valid_loss, i):
    if not save_path:
        raise ValueError('Provided save path is none, metrics can\'t be saved!')
    
    state_dict = {'train_loss': train_loss,
                  'valid_loss': valid_loss,
                  'i': i}
    
    torch.save(state_dict, save_path)
    logger.info(f'Metrics saved to {save_path}')


def load_metrics(load_path):
    if not load_path:
        raise ValueError('Provided load path is none, metrics can\'t be loaded!')
    
    state_dict = torch.load(load_path)
    logger.info(f'Metrics loaded from: {load_path}')
    
    return state_dict['train_loss'], state_dict['valid_loss'], state_dict['i']


def create_loss_plot(train_loss, valid_loss, steps):
    _, ax = plt.subplots(figsize=(10, 8))
    ax.set(xlabel='steps', ylabel='loss', title='Training loss')
    plt.plot(steps, train_loss, label='Train')
    plt.plot(steps, valid_loss, label='Valid')
    plt.legend()
    plt.show()


def get_iterator(dataset, batch_size, device='cpu'):
        return BucketIterator(dataset, batch_size=batch_size, sort_key=lambda x: len(x.sentence),
                                device=device, sort=True, sort_within_batch=True)


def get_loaders_lstm(data_path='./data/', 
                     train_name='train_lstm.csv',
                     valid_name='valid_lstm.csv',
                     test_name='test_lstm.csv',
                     vocab_min_freq=3, 
                     batch_size=32, 
                     device='cpu'):
    spacy.load('en_core_web_sm')

    target_field = LabelField(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
    text_field = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, include_lengths=True, batch_first=True)
    fields = [('sentence', text_field), ('label', target_field)]

    train, valid, test = TabularDataset.splits(path=data_path, train=train_name, validation=valid_name, test=test_name,
                                            format='CSV', fields=fields, skip_header=True)

    text_field.build_vocab(train, min_freq=vocab_min_freq)

    train_loader = get_iterator(dataset=train, batch_size=batch_size, device=device)
    valid_loader = get_iterator(dataset=valid, batch_size=batch_size, device=device)
    test_loader = get_iterator(dataset=test, batch_size=batch_size, device=device)

    return train_loader, valid_loader, test_loader, len(text_field.vocab)


def prepare_dataset_lstm(path='./data'):
    #df = pd.read_csv(f'{path}/food_disease_dataset_processed.csv')
    #df = df.replace({'cause': 0, 'treat': 1, 'neutral': 2})
    df = pd.read_csv(f'{path}/food_disease_dataset.csv')
    df['label'] = df.apply(lambda x: label_treats_causes(x),axis=1)
    #drop columns we do not need
    df=df.drop(columns=['Unnamed: 0','disease_doid','food_entity','disease_entity','is_cause','is_treat'])
    labels = {"cause": 0, "treat": 1, "neutral":2}
    df["label"] = [labels[item] for item in df.label]
    stop_words = list(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([word for word in nltk.word_tokenize(x, language='english') if re.match('\w', word)]))
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))

    X_train, X_test = split(df, test_size=0.2, random_state=42, stratify=df['label'])
    X_train, X_valid = split(X_train, test_size=0.2, random_state=42, stratify=X_train['label'])

    X_train.to_csv(f'{path}/train_lstm.csv', index=False)
    X_valid.to_csv(f'{path}/valid_lstm.csv', index=False)
    X_test.to_csv(f'{path}/test_lstm.csv', index=False)