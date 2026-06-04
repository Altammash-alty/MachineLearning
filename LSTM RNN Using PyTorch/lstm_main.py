import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np
import pandas as pd
import nltk 
from nltk import tokenize 
# from sklearn.preprocessing import Scaler 
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_score,f1_score,confusion_matrix 


with open('LSTM RNN\hamlet.txt','r') as file:
    text=file.read().lower()

import string
from collections import Counter

# 1. Clean the text properly (strings are immutable, so we must reassign)
for p in string.punctuation:
    text = text.replace(p, " ")

# 2. Tokenize into words using NLTK
words = tokenize.word_tokenize(text)

# 3. Build a vocabulary
word_counts = Counter(words)
# Sort words by frequency (optional, but good practice)
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

# Create dictionaries to map words to integers and vice versa
word_to_int = {word: i for i, word in enumerate(sorted_words)}
int_to_word = {i: word for i, word in enumerate(sorted_words)}

# 4. Convert the entire text into integers
encoded_text = [word_to_int[word] for word in words]

# 5. Create sequences (X) and targets (y)
seq_length = 20 # Number of previous words to look at
X = []
y = []

for i in range(0, len(encoded_text) - seq_length):
    seq = encoded_text[i:i + seq_length]
    target = encoded_text[i + seq_length]
    X.append(seq)
    y.append(target)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

print(f"Total words: {len(words)}")
print(f"Unique words (Vocab size): {len(word_to_int)}")
print(f"Number of sequences: {len(X)}")
print(f"Sample X (shape {X.shape}):\n", X[:2])
print(f"Sample y (shape {y.shape}):\n", y[:2])

