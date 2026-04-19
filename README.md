# Literate GPT
This repository contains GPT code implemented by Andrej Karpathy in his famous YT tutorial. As for now, it is a simple
model for text generation. However, we aim to introduce certain improvements and make it conversational. 

## Project Structure 
* `data` - keeps mock example for training
* `src`
  * `model.py` - core GPT architecture
* `main.py` - script for initializing the model and training

## Quick Start
1. Install requirements by using pip: `pip install requirements.txt`
2. Make set your parameters in `main.py`
3. Train the model on a mock example by using `python main.py`

## Interesting improvements 
* We introduce `tiktoken` tokenization, that was orginally used in GPT-2 model. 
* 

## Sources
* https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6033s
