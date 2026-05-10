"""
This script trains the model and make an output with timestamps.
"""

from src import model as md
from src import preprocessing as pp
import os
import torch
import tiktoken
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from src.conversation import ConversationHistory
from src.preprocessing import clean, get_chat_tokenizer

parent_dir = Path(__file__).resolve().parent.parent
os.chdir(parent_dir)

print(f"Current working directory: {os.getcwd()}")
#########################################################################################
# load the data
# text = dp.clean_text(path_to_file=r'data/mock_data.txt')
df_full = pd.read_parquet("hf://datasets/ruggsea/stanford-encyclopedia-of-philosophy_chat_multi_turn/data/train-00000-of-00001.parquet")
df_full = df_full.head(300)


text = pp.process_conversational_dataset(df_full, column="conversation")
tokenizer = get_chat_tokenizer()

# add conversational tokens
token_ids = tokenizer.encode(text, allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"})

#########################################################################################
vocab_size = tokenizer.n_vocab + 2

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

# load the params
with open("data/cfg.json", "r", encoding="utf-8") as f:
    params = json.load(f)


params['vocab_size'] = tokenizer.n_vocab + 2
params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
params['head_size'] = int(params['n_embed']) // int(params['n_heads'])


print(f"{'Dataset Metrics':-^30}")
print(f"Train Length: {len(train_data)}")
print(f"Val Length:   {len(test_data)}")
print(f"Block Size:   {params['batch_size']}")

print(f"\n{'Model Hyperparameters':-^30}")
print(f"Vocab Size:\t{vocab_size}")
print(f"Embed Dim:\t{params['n_embed']}")
print(f"Heads:\t{params['n_heads']}")
print(f"Head Size:\t{params['head_size'] }")
print(f"Device:\t{params['device']}")
print(f"checkpoint_interval:\t{params['checkpoint_interval']}")

# add README file to get to now the state of the model
metadata = open("data/metadata.txt", 'w',  encoding='utf-8')
metadata.write(f"{'Dataset Metrics':-^30}\n")
metadata.write(f"Train Length: {len(train_data)}\n")
metadata.write(f"Val Length:   {len(test_data)}\n")
metadata.write(f"Block Size:   {params['batch_size']}\n")
metadata.write(f"\n{'Model Hyperparameters':-^30}\n")
metadata.write(f"Vocab Size:\t{params['n_embed']}\n")
metadata.write(f"Embed Dim:\t{params['n_embed']}\n")
metadata.write(f"Head Size:\t{params['n_heads']}\n")
metadata.write(f"Head Size:\t{params['n_heads']}\n")
metadata.write(f"Device:\t{params['device']}\n")
metadata.write(f"checkpoint_interval:\t{params['checkpoint_interval']}\n\n")

#########################################################################################
# initialize model
model = md.GPTLanguageModel(**params)
m = model.to(device)

print(f"\n{'Model':-^30}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print("Training:")

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])

# introduce Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['max_iters'])

last_train_loss = 0
last_val_loss = 0
for iter in range(params['max_iters']):

    # every once in a while evaluate the loss on train and val sets
    if iter % params['eval_interval'] == 0:
        losses = md.estimate_loss(model=model,
                                  train_data=train_data,
                                  test_data=test_data,
                                  **params)

        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        last_train_loss = losses['train']
        last_val_loss = losses['val']

    if iter != 0 and iter % params['checkpoint_interval'] == 0:
        timestamp = datetime.now().strftime('%H-%M-%S_%d_%m_%Y')
        filename = f"models/william_james_{timestamp}.pt"
        os.makedirs("models", exist_ok=True)

        torch.save(model.state_dict(), filename)
        print("-"*30)
        print(f"Model saved to {filename}")
        print("-" * 30)

        metadata.write(f"Model: william_james_{timestamp}\n")
        metadata.write(f"Number of steps:{iter}\n")
        metadata.write(f"Last train loss {last_train_loss:.4f}\n")
        metadata.write(f"Last val loss {last_val_loss:.4f}\n")
        metadata.write("-" * 30 + "\n")

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
    scheduler.step()

#########################################################################################
# save the model
timestamp = datetime.now().strftime('%H-%M-%S_%d_%m_%Y')
filename = f"models/william_james_{timestamp}.pt"
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), filename)
print(f"\nFinal model saved to {filename}")

metadata.write(f"Model: william_james_{timestamp}\n")
metadata.write(f"Number of steps:{params['max_iters']}\n")
metadata.write(f"Last train loss {last_train_loss:.4f}\n")
metadata.write(f"Last val loss {last_val_loss:.4f}\n")
metadata.write("-" * 30 + "\n")
metadata.close()

# sve the params to JSON
with open("data/cfg.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)
