"""
A set of functions designed for quick date encoding and preprocessing.
"""
import os
import re
import tiktoken

def preprocess_data(path_to_file: str, start_line: int =1, end_line: int =1) -> str:

    assert os.path.exists(path_to_file), "Path does not exist!"

    with open(path_to_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        text = "".join(lines[start_line:-end_line])
        assert len(text) != 0, "No data to train and evaluate! Your file probably has only one line."

        # clean text
        cleaned = re.sub(r'[^a-zA-Z .,:()]', '', text)
        return cleaned


# tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like coffee?"
# token_ids = tokenizer.encode(text)
# print(token_ids)