import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np
import pandas as pd
import os
import nltk 
from nltk.tokenize import word_tokenize
import collections
from collections import Counter
# from sklearn.preprocessing import Scaler 
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_scor e,f1_score,confusion_matrix 


# Resolve path to hamlet.txt relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
hamlet_path = os.path.join(script_dir, '..', 'LSTM RNN', 'hamlet.txt')

with open(hamlet_path, 'r', encoding='utf-8') as file:
    text=file.read().lower()

text=text.replace(","," ").replace("."," ").replace(":"," ").replace(";"," ").replace("?"," ")
token = word_tokenize(text)
 
vocab = {'unk':0}

for tok in Counter(token).keys() :
    if tok not in vocab :
        vocab[tok]=len(vocab)
print(len(vocab))

#sentence extracting 

sentences = [line.strip() for line in text.split("\n") if line.strip()]
print(sentences)


def text_indices (sentences,vocab):
    index_sent= []
    for sentence in sentences :
        token_sent=word_tokenize(sentence)
        

print(token_sent)



# X = []
# y = []

# for i in range(0, len(encoded_text) - seq_length):
#     seq = encoded_text[i:i + seq_length]
#     target = encoded_text[i + seq_length]
#     X.append(seq)
#     y.append(target)

# # Convert to PyTorch tensors
# X = torch.tensor(X, dtype=torch.long)
# y = torch.tensor(y, dtype=torch.long)

# print(f"Total words: {len(words)}")
# print(f"Unique words (Vocab size): {len(word_to_int)}")
# print(f"Number of sequences: {len(X)}")
# print(f"Sample X (shape {X.shape}):\n", X[:2])
# print(f"Sample y (shape {y.shape}):\n", y[:2])

