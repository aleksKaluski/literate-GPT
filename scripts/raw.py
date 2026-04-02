"""
Tokenize, encode and decode the training text.
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

# set current dir as .\literate-GPT
current_dir = os.getcwd()
if current_dir[-4:-1] != 'data':
    os.chdir(os.path.dirname(current_dir))
print(f"\nCurrent working directory: {os.getcwd()}")


with open("data/lecture_on_ethics.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all characters
chars = sorted(list(set(text)))

# encode the characters
stoi = {ch:i for i, ch in enumerate(chars)} # set of char-number pairs
itos = {i:ch for i, ch in enumerate(chars)}

# encoder
encode = lambda s: [stoi[c] for c in s]

# decoder
decode = lambda l: ''.join([itos[i] for i in l])

# encode the text and wrap it into data tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])

# train-test split
n = int(0.9 * len(data)) # 90%
train_data = data[:n]
test_data = data[n:]

# make the training as batch

batch_size = 4
block_size = 8

def get_batch(split):
    """
    Generate a batch of data of inputs (x) and targets (y).
    :param split: decide weather you use train or test data; split must be in ['train', 'test']
    :return: batch of inputs (x), targets (y)
    """
    data = train_data if split == 'train' else test_data

    # randomly grab a chunk
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # block size characters
    x = torch.stack([data[i:i+block_size] for i in ix])

    # offset of bock chars by 1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    """
    Define a bigram language model, which a simple NN. Essentially it is a massive scoreboard where
    every word is a word and every column is it's possible successor.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits







