"""
Pretrains model on txt files.
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