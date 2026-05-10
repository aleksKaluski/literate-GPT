from src import model as md
from src import preprocessing as pp
import os
import torch
import tiktoken
import pandas as pd
from src.conversation import ConversationHistory
from src.preprocessing import clean

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"\nCurrent working directory: {os.getcwd()}")

#########################################################################################
# load the data
# text = dp.clean_text(path_to_file=r'data/mock_data.txt')
df_full = pd.read_parquet("hf://datasets/ruggsea/stanford-encyclopedia-of-philosophy_chat_multi_turn/data/train-00000-of-00001.parquet")
df_full = df_full.head(100)


text = pp.process_conversational_dataset(df_full, column="conversation")


# add special tokens
special_tokens = {
    "<|user|>": 50257,
    "<|assistant|>": 50258
}

# encode and decode chars
basic_tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = tiktoken.Encoding(
    name="chat_gpt2",
    pat_str=basic_tokenizer._pat_str,
    mergeable_ranks=basic_tokenizer._mergeable_ranks,
    special_tokens={**basic_tokenizer._special_tokens, **special_tokens}
)

# add conversational tokens
token_ids = tokenizer.encode(text, allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"})

#########################################################################################
vocab_size = tokenizer.n_vocab + len(special_tokens)

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
print(f"Device: {device}")

batch_size = 16
block_size = 32
n_embed = 32
n_head = 8
n_layer = 8
dropout = 0.2
max_iters = 1000
eval_interval = 200
learning_rate = 2e-3
eval_iters = 400
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
params = {'vocab_size': vocab_size,
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

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# introduce Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

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
    scheduler.step()

#########################################################################################
# generate from the model
# initialize history
history = ConversationHistory(lm_memory_size=block_size, tokenizer=tokenizer ,device=device)

print(f"Welcome to the API with GPT-transformer!")
print(f"To end the conversation, type 'exit'.")
context = history.return_context()

# say 'hi' to the user
output_tensor = m.generate(context, max_new_tokens=50, **params)
response = tokenizer.decode(output_tensor[0, context.shape[1]:].tolist())
print(f"[james]: {response}")
history.append(text=response, role="assistant")

while True:

    user_input = input("[user]: ")
    user_input_cleaned = clean(user_input)
    if user_input == "exit":
        break
    history.append(text=user_input_cleaned, role="user")
    context = history.return_context()

    # normal response returns the entire context
    output_tensor = m.generate(context, max_new_tokens=50, **params)

    # take just newly generated tokens
    response = tokenizer.decode(output_tensor[0, context.shape[1]:].tolist())

    print(f"James: {response}")
    history.append(text=response, role="assistant")


