from src import model as md
from src import preprocessing as dp
import os
import torch
import tiktoken


os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"\nCurrent working directory: {os.getcwd()}")

#########################################################################################
# load the data
text = dp.clean_text(path_to_file=r'data/mock_data.txt')

# encode and decode chars
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

#########################################################################################
# encode and decode chars

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode the text and wrap it into data tensor
data = torch.tensor(token_ids, dtype=torch.long)

# train-test split
n = int(0.9 * len(data)) # 70%
train_data = data[:n]
test_data = data[n:]

assert len(train_data) != 0
assert len(test_data) != 0

#########################################################################################
# define key params
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 8
block_size = 16
n_embed = 16
n_head = 4
n_layer = 4
dropout = 0.2
max_iters = 1000
eval_interval = 200
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
print(f"Head Size:    {head_size}")
print("-" * 30)

# pass arguments as kwargs
params = {'vocab_size': len(chars),
          'block_size': block_size,
          'batch_size': batch_size,
          'n_heads': n_head,
          'n_layer': n_layer,
          'n_embed': n_embed,
          'dropout': dropout,
          'head_size': head_size,
          'device': device
          }

#########################################################################################
# initialize model
model = md.GPTLanguageModel(**params)
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = md.estimate_loss(model=model,
                                  train_data=train_data,
                                  test_data=test_data,
                                  **params)

        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = md.get_batch(split='train',
                          train_data=train_data,
                          test_data=test_data,
                          **params)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context,
                        max_new_tokens=500,
                        **params)[0].tolist()))
