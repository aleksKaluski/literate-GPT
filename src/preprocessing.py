"""
A set of functions designed for quick date encoding and preprocessing.
"""
import os
import re
import pandas as pd


def process_classical_txt(path_to_file: str, start_line: int =1, end_line: int =1) -> str:
    """
    Extract the data from TXT files.
    :param path_to_file: path to file
    :param start_line: line to start at
    :param end_line: line to end at
    :return: cleaned text (list of strings)
    """
    assert os.path.exists(path_to_file), "Path does not exist!"

    with open(path_to_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        text = "".join(lines[start_line:-end_line])
        assert len(text) != 0, "No data to train and evaluate! Your file probably has only one line."

        # clean text
        cleaned = clean(text)
        return cleaned


def clean(text: str) -> str:
    """
    Clean text and return a single string.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9 .,:()<>|\n]', '', text)
    return "".join(cleaned)


def process_conversational_dataset(df: pd.DataFrame, column: str) -> list[str]:
    """
    Process HF dataset to create conversational dataset.
    """
    text = ""
    for row in df[column]:
        conversation = ""
        for turn in row:
            if turn['role'] == 'user':
                content = clean(turn['content'])
                conversation += "<|user|>\n"
                conversation += content + "\n"
            elif turn['role'] == 'assistant':
                content = clean(turn['content'])
                conversation += "<|assistant|>\n"
                conversation += content + "\n"
        # last token
        conversation += "<|endoftext|>" + '\n'
        text += conversation
    return text


