import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# get read document.txt
with open('data/document.txt','r',encoding='utf-8') as f:
    document = f.read()

#tokenize
tokens = word_tokenize(document.lower())

#build vocab
vocab = {'<UNK>':0}
for token in Counter(tokens).keys():
    if token not in vocab:
        vocab[token] = len(vocab)

# print(f'Vocab size: {len(vocab)}')

# list sentences
input_sentences = document.split('\n')

# convert text to indices
def text_to_indices(sentence,vocab):
    numerical_sentence = []

    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<UNK>'])
    
    return numerical_sentence


# create list of indices questions
input_numerical_sentences = []

for sentence in input_sentences:
    input_numerical_sentences.append(text_to_indices(word_tokenize(sentence.lower()),vocab))

# print(input_numerical_sentences[:5])


# making training sequence
training_sequences = []
for sentence in input_numerical_sentences:
    for i in range(1,len(sentence)):
        training_sequences.append((sentence[:i+1]))

# print(f'Total training sequences: {len(training_sequences)}')

#padding sequences to make it uniform length
# --- find max length of sequence ---
len_list = []
for seq in training_sequences:
    len_list.append(len(seq))

max_len = max(len_list)

# --- pad sequences ---
padded_training_sequences = []

for sequence in training_sequences:
    padded_training_sequences.append([0]*(max_len - len(sequence)) + sequence)

# print(padded_training_sequences[:5])

# convert to tensor
padded_training_sequences = torch.tensor(padded_training_sequences, dtype=torch.long)
print(padded_training_sequences.shape)


#split input and output from each sequence
X = padded_training_sequences[:,:-1]
y = padded_training_sequences[:,-1]

# print(X.shape)
# print(y.shape)


# datasets and dataloaders
class NextWordDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X[0])
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = NextWordDataset(X,y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# test dataloader
# for inputs, targets in dataloader:
#     print(inputs.shape)
#     print(targets.shape)
#     break

