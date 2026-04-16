from src import model as md
import os
import string
import torch


os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"\nCurrent working directory: {os.getcwd()}")

#########################################################################################
# load the data
with open(r"data/mock_data.txt", "r", encoding="utf-8") as f:
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
print(decode(m.generate(context,
                        max_new_tokens=500,
                        **params)[0].tolist()))
