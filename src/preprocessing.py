"""
A set of functions designed for quick date encoding and preprocessing.
"""
import os
import re
import pandas as pd


def clean_text(path_to_file: str, start_line: int =1, end_line: int =1) -> str:
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
        cleaned = re.sub(r'[^a-zA-Z .,:()]', '', text)
        return cleaned


def process_row(row: dict[str, str]) -> str:
    """
    Process a single row of HF dataset in order to create
    conversational dataset.
    """
    result = ""
    for turn in row:
        if turn['role'] == 'user':
            result += "<|user|>\n"
            result += turn['content'] + "\n"
        elif turn['role'] == 'assistant':
            result += "<|assistant|>\n"
            result += turn['content'] + "\n"

    # last token
    result += "<|endoftext|>"
    return result


def process_dataset(df: pd.DataFrame, column: str) -> list[str]:
    """
    Process HF dataset to create conversational dataset.
    """
    text = ""
    for row in df[column]:
        text += process_row(row)
    return text

