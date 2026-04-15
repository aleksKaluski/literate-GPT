from src import model as md
import os
import string
import torch


# set current dir as .\literate-GPT
# current_dir = os.getcwd()
# if current_dir[-4:-1] != 'data':
#     os.chdir(os.path.dirname(current_dir))
print(f"\nCurrent working directory: {os.getcwd()}")

#########################################################################################
# load the data
with open("data/lecture_on_ethics.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # text = "".join(lines[100:-200])
    text = "".join(lines)

# delete non-ACII characters
text = text.replace("\n", " ")
allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
delete_dict = {ord(c): None for c in text if c not in allowed_chars}
text = text.translate(delete_dict)

#########################################################################################
# encode and decode chars

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

assert len(train_data) != 0
assert len(test_data) != 0

#########################################################################################
# define key params
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
block_size = 32
n_embed = 64
n_head = 8
n_layer = 6
dropout = 0.2
max_iters = 5000
eval_interval = 500
learning_rate = 2e-3
eval_iters = 200
head_size = n_embed // n_head


print(f"{'Dataset Metrics':-^30}")
print(f"Train Length: {len(train_data)}")
print(f"Val Length:   {len(test_data)}")
print(f"Block Size:   {block_size}")

print(f"\n{'Model Hyperparameters':-^30}")
print(f"Vocab Size:   {vocab_size}")
print(f"Embed Dim:    {n_embed}")
print(f"Heads:        {n_head}")
print(f"Head Size:    {n_embed // n_head}")
print("-" * 30)
#########################################################################################
