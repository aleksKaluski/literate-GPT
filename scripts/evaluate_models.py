"""
A script to evaluate a bunch of trained models. It loads each of them, then it asks some questions regarding the
basic cognitive functions. The results are saved to "logs" directory.
"""
from pathlib import Path
import os
import json
import torch

from src.preprocessing import get_chat_tokenizer, clean
from src.model import GPTLanguageModel
from src.conversation import ConversationHistory

parent_dir = Path(__file__).resolve().parent.parent
os.chdir(parent_dir)

print(f"Current working directory: {os.getcwd()}")
models_dir = "models"

for model_name in os.listdir(models_dir):
    # skip non-wav files
    if not model_name.endswith('.pt'): continue

    # save model's path
    model_path = os.path.join(models_dir, model_name)

    # Load the params
    with open("data/cfg.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    torch.manual_seed(42)
    device = params['device']

    # load tokenizer
    tokenizer = get_chat_tokenizer()

    # initialize model
    model = GPTLanguageModel(**params)

    # load and inject the weights into the model
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully restored the persona from {model_path}")
    else:
        print("No saved weights found. Initiating a blank slate.")

    model.to(device)
    model.eval()

    # make a test conversation with the model
    history = ConversationHistory(lm_memory_size=params['block_size'], tokenizer=tokenizer, device=device)
    context = history.return_context()

    with open('data/test_file.txt', 'r', encoding='utf-8') as f1, \
            open(f'data/logs/{model_name}.txt', 'w', encoding='utf-8') as f2:
        lines_list = f1.readlines()

        f2.write(f"TESTS\n")
        for line in lines_list:
            user_input = clean(line)
            history.append(text=user_input, role="user")
            context = history.return_context()

            # normal response returns the entire context
            output_tensor = model.generate(context, max_new_tokens=50, **params)

            # take just newly generated tokens
            response = tokenizer.decode(output_tensor[0, context.shape[1]:].tolist())
            history.append(text=response, role="assistant")
            f2.write(f"User: {user_input}\n")
            f2.write(f"James: {response}\n\n")




