"""
A script that loads the model and then enables us to talk with it.
"""

from src import model as md
import os
import torch
from src.conversation import ConversationHistory
from src.preprocessing import clean, get_chat_tokenizer
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"\nCurrent working directory: {os.getcwd()}")

#########################################################################################
# Load the params
with open("data/cfg.json", "r", encoding="utf-8") as f:
    params = json.load(f)

torch.manual_seed(42)
device = params['device']

#########################################################################################
# load tokenizer
tokenizer = get_chat_tokenizer()

# initialize model
model = md.GPTLanguageModel(**params)
load_path = "models/william_james_16-15-30_11_05_2026.pt"

# load and inject the weights into the model
if os.path.exists(load_path):
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Successfully restored the persona from {load_path}")
else:
    print("No saved weights found. Initiating a blank slate.")


model.to(device)
model.eval()  # set to evaluation mode for conversation

#########################################################################################
# generate from the model
# initialize history
history = ConversationHistory(lm_memory_size=params['block_size'], tokenizer=tokenizer ,device=device)

print(f"Welcome to the API with GPT-transformer!")
print(f"To end the conversation, type 'exit'.")
context = history.return_context()

# say 'hi' to the user
output_tensor = model.generate(context, max_new_tokens=50, **params)
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
    output_tensor = model.generate(context, max_new_tokens=50, **params)

    # take just newly generated tokens
    response = tokenizer.decode(output_tensor[0, context.shape[1]:].tolist())

    print(f"[james]: {response}")
    history.append(text=response, role="assistant")


