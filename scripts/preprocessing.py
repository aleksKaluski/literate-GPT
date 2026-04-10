import re
import os
import string

# set current dir as .\literate-GPT
current_dir = os.getcwd()
if current_dir[-4:-1] != 'data':
    os.chdir(os.path.dirname(current_dir))
print(f"\nCurrent working directory: {os.getcwd()}")

# clean the text
with open("data/swans_way_complete.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    text = "".join(lines[100:-200])

text = text.replace("\n", " ")
allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
delete_dict = {ord(c): None for c in text if c not in allowed_chars}
text = text.translate(delete_dict)


chars = sorted(list(set(text)))
print(chars)
print(text[:100])
