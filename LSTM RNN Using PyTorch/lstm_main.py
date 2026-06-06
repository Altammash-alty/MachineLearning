import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import time


# Resolve path to hamlet.txt relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
hamlet_path = os.path.join(script_dir, '..', 'LSTM RNN', 'hamlet.txt')

with open(hamlet_path, 'r', encoding='utf-8') as file:
    document = file.read().lower()

tokens = word_tokenize(document)
vocab = {'<unk>': 0}

for token in Counter(tokens).keys():
    if token not in vocab:
        vocab[token] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")


input_sentences = document.split('\n')


def text_to_indices(sentence, vocab):
    numerical_sentence = []
    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<unk>'])
    return numerical_sentence


input_numerical_sentences = []

for sentence in input_sentences:
    tokenized = word_tokenize(sentence.lower())
    if tokenized:  # skip empty lines
        input_numerical_sentences.append(text_to_indices(tokenized, vocab))


training_sequence = []
for sentence in input_numerical_sentences:
    for i in range(1, len(sentence)):
        training_sequence.append(sentence[:i + 1])

print(f"Number of training sequences: {len(training_sequence)}")

len_list = [len(seq) for seq in training_sequence]
max_len = max(len_list)
print(f"Max sequence length: {max_len}")


MAX_SEQ_LEN = 100
if max_len > MAX_SEQ_LEN:
    training_sequence = [seq for seq in training_sequence if len(seq) <= MAX_SEQ_LEN]
    max_len = MAX_SEQ_LEN

padded_training_sequence = []
for sequence in training_sequence:
    padded_training_sequence.append([0] * (max_len - len(sequence)) + sequence)

padded_training_sequence = torch.tensor(padded_training_sequence, dtype=torch.long)

X = padded_training_sequence[:, :-1]
y = padded_training_sequence[:, -1]



class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Dataset size: {len(dataset)}")


class LSTMModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output


# -------------------------------------------------------------------
# 10. Initialize model, loss, optimizer
# -------------------------------------------------------------------

model = LSTMModel(len(vocab))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------------------------------------------
# 11. Training loop
# -------------------------------------------------------------------

for epoch in range(epochs):
    total_loss = 0

    for batch_x, batch_y in dataloader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        total_loss = total_loss + loss.item()

    print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")


# -------------------------------------------------------------------
# 12. Prediction function
# -------------------------------------------------------------------

def prediction(model, vocab, text, seq_len):
    """Predict the next word given input text."""
    # tokenize
    tokenized_text = word_tokenize(text.lower())

    # text -> numerical indices
    numerical_text = text_to_indices(tokenized_text, vocab)

    # truncate if too long
    if len(numerical_text) > seq_len:
        numerical_text = numerical_text[-seq_len:]

    # padding
    padded_text = torch.tensor(
        [0] * (seq_len - len(numerical_text)) + numerical_text,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    # send to model
    model.eval()
    with torch.no_grad():
        output = model(padded_text)

    # predicted index
    value, index = torch.max(output, dim=1)

    # merge with text
    return text + " " + list(vocab.keys())[index.item()]



input_seq_len = max_len - 1


num_tokens = 10
input_text = "to be or not"

for i in range(num_tokens):
    output_text = prediction(model, vocab, input_text, input_seq_len)
    print(output_text)
    input_text = output_text
    time.sleep(0.3)


dataloader_eval = DataLoader(dataset, batch_size=64, shuffle=False)


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)

            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total * 100
    return accuracy


accuracy = calculate_accuracy(model, dataloader_eval, device)
print(f"\nModel Accuracy: {accuracy:.2f}%")