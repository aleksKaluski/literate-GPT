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

print("train len:", len(train_data), "val len:", len(test_data), "block_size:", block_size)
print("vocab_size:", vocab_size)
print("n_embd:", n_embed, "n_head:", n_head, "head_size:", n_embed // n_head)
######################################


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


@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # initialize keys, queries and values for attention
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # tril is a buffer (not a parameter of the model)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # make sure that the feature does not communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

        # projection for residual connections
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # projection is a linear transformation of the outcome of
        # this layer
        out = torch.cat([h(x) for h in self.n_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        # a very simple layer: linear MLP followed by a ReLU activation
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed),
                                 nn.ReLU(),
                                 nn.Linear(n_embed, n_embed), # projection layer going back to the
                                 # residual pathway
                                 nn.Dropout(dropout)
                                 )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd ,n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)

        # layer normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # by adding x we introduce the residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    Define a bigram language model, which a simple NN. Essentially it is a massive scoreboard where
    every word is a word and every column is it's possible successor.
    """
    def __init__(self):
        super().__init__()

        # token identity encoding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # token position encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # initialize block
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_head) for _ in range(n_layer)])

        # self-attention head (hanged with blocks)
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) #we have 4 communication channels
        # self.ffwd = FeedForward(n_embed)

        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm

        # self-attention head
        self.lm_head = nn.Linear(n_embed, vocab_size) # language modeling head


        # token position encoding
        # self.position_embedding_table = nn.Embedding(vocab_size, n_embed)


    def forward(self, idx, targets=None):

        B, T = idx.shape
        # position embdeding
        pos_emd = self.position_embedding_table(torch.arange(T).to(device=device)) # -> (T, C)
        tok_emb = self.token_embedding_table(idx) # B, T, C (embed C)

        # we add token embeddings and position emeddings
        x = tok_emb + pos_emd
        x = self.blocks(x) # feed the into to self-attention head
        # x = self.ffwd(x) # feedforward after self-attention
        logits = self.lm_head(x) # B, T, C (vocac size C)

        if targets is None:
            loss = None
        else:
            # reshape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Negative Log Likelihood
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx = (B, T)
        for _ in range(max_new_tokens):

            # crop the size to block size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # predictions
            logits = logits[:, -1, :]  # -> (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

vocab_size = len(chars)
model = GPTLanguageModel()
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
