import json
import torch
import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
os.chdir(parent_dir)

print(f"Current working directory: {os.getcwd()}")


with open("data/cfg.json", "r", encoding="utf-8") as f:
    params = json.load(f)


params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
print(params)