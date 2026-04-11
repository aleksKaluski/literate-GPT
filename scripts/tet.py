"""
Tokenize, encode and decode the training text.
"""
import os
import string
import torch
import torch.nn as nn
from torch.nn import functional as F

# set current dir as .\literate-GPT
current_dir = os.getcwd()
if current_dir[-4:-1] != 'data':
    os.chdir(os.path.dirname(current_dir))
print(f"\nCurrent working directory: {os.getcwd()}")

# load the data
with open("data/lecture_on_ethics.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # text = "".join(lines[100:-200])
    text = "".join(lines)

text = text.replace("\n", " ")
allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
delete_dict = {ord(c): None for c in text if c not in allowed_chars}
text = text.translate(delete_dict)
# print(text[:100])

# define hyperparams
######################################
torch.manual_seed(42)
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 64
n_head = 6
n_layer = 6

dropout = 0.2

max_iters = 5000
eval_interval = 300
learning_rate = 2e-3
eval_iters = 200

# all characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode the characters
stoi = {ch:i for i, ch in enumerate(chars)} # set of char-number pairs
itos = {i:ch for i, ch in enumerate(chars)}

# encoder
encode = lambda s: [stoi[c] for c in s]

# decoder
decode = lambda l: ''.join([itos[i] for i in l])

# encode the text and wrap it into data tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train-test split
n = int(0.9 * len(data)) # 90%
train_data = data[:n]
test_data = data[n:]

print(test_data)