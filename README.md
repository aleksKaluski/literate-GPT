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
* We use **flash attention** mechanism. Normal attention has complexity if $O(n**2)$, which is dramatically high. 
Flash attention reduces the complexity by using ultra-fast SRAM GPU memory.
* We emply **weight tieing**. By setting the same weights of `lm_head` and `token_embedding_table` in GPT model.
Thanks to that we reduce the paramters of the model and save VRAM memory.

## Sources
* https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6033s
* https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
* https://towardsdatascience.com/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton-5609f0b143ea/
