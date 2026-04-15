"""
Here we keep the core architecture of the GPT model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_batch(split: str,
              block_size: int,
              batch_size: int,
              train_data: torch.long,
              test_data: torch.long) -> torch.utils.data.DataLoader:
    """
    Generate a batch of data of inputs (x) and targets (y) for training.
    :param split: decide weather you use train or test data; split must be in ['train', 'test']
    :param batch_size: decides how many rows are in x and y
    :param block_size: decides how many tokens are in each row
    :param train_data: torch.long containing training data
    :param test_data: torch.long containing test data
    :return: batch of inputs (x), targets (y)
    """

    assert split in ['train', 'test'], "Split must be in ['train', 'test']!"

    data = train_data if split == 'train' else test_data

    # randomly grab a chunk
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # block size characters
    x = torch.stack([data[i:i+block_size] for i in ix])

    # offset of bock chars by 1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int = 200):
    """
    Estimate the loss of the GPT model while training.
    :param model: PyTorch model for which to estimate loss.
    :param eval_iters: how many iterations to evaluate the model
    :return: evaluation loss
    """
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
    """
    A class that implements a head of attention mechanism in the GPT model.
    """
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float):
        """
        Initialize the head of attention mechanism.
        :param head_size: size of attention heads
        :param n_embed: number of embedding dimensions on which attention mechanism is applied
        :param block_size: the maximum context length (the number of tokens which we use to train the model)
        :param dropout: how much dropout we want to use (meaning: how many activations we want to randomly zero)
        """
        super().__init__()

        # initialize keys, queries and values for attention
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # tril is a buffer (not a parameter of the model)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.long) ->torch.Tensor:
        """
        The way in which the attention mechanism is applied (defines the computational graph).
        :param x: the input sequence of token embeddings for the attention head
        :return: logits (s a context-aware representation of the input tokens)
        """
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
        return out # (Batch, Time, Head Size)


class MultiHeadAttention(nn.Module):
    """
    Initialize the multi-head attention mechanism.
    """
    def __init__(self, head_size: int, n_embed: int, n_heads: int , dropout: float):
        """
        Initialize the multi-head attention mechanism.
        :param head_size: size of attention heads
        :param n_embed: number of embedding dimensions on which attention mechanism is applied
        :param n_heads: number of attention heads to initialize
        :param dropout: how much dropout we want to use (meaning: how many activations we want to randomly zero)
        """
        super().__init__()
        self.n_heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

        # projection for residual connections
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x:  torch.long) -> torch.Tensor:
        """
        Use projection layer to mix the information from all individual heads.
        :param x:
        :return:
        """
        # projection is a linear transformation of the outcome of
        # this layer
        out = torch.cat([h(x) for h in self.n_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out # (B, T, C)


class FeedForward(nn.Module):
    """
    A very simple linear MLP with ReLu activation function.
    """
    def __init__(self, n_embed: int, dropout: float):
        """
        Initialize a simple feed-forward layer.
        :param n_embed: number of embedding dimensions on which feed-forward layer is applied
        :param dropout: how much dropout we want to use
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed),
                                 nn.ReLU(),
                                 # projection layer going back to the
                                 # residual pathway
                                 nn.Linear(n_embed, n_embed),
                                 nn.Dropout(dropout)
                                 )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Initialize the block of attention mechanism with normalization layers.
    """
    def __init__(self, n_embed: int, n_heads: int):
        """
        A single block of attention.
        :param n_embed: number of embedding dimensions on which attention mechanism is applied
        :param n_heads: number of attention heads to initialize
        """
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)

        # normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # by adding x we introduce the residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x