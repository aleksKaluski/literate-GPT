from pathlib import Path
import os
from src.preprocessing import clean


parent_dir = Path(__file__).resolve().parent.parent
os.chdir(parent_dir)

print(f"Current working directory: {os.getcwd()}")



with open("data/test_file.txt", "r", encoding="utf-8") as f:
    lines_list = f.readlines()

lines_list = [clean(line) for line in lines_list]
print(lines_list)
for line in lines_list:
    print(clean(line))
