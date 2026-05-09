"""
This file keeps all the important conversational tools that we need to have.
"""
import torch

class ConversationHistory():
    """
    Keeps track of conversation history, updates it and return the amount of
    information that keeps track of the context.
    """
    def __init__(self, lm_memory_size, tokenizer, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.llm_memory_size = lm_memory_size
        self.conversation_history = tokenizer.encode("Your name is William James. Say hello and introduce yourself.")


    def append(self, text: str, role: str):
        """
        Updates the conversation history.
        """
        if role == "user":
            # wrap text into tags
            wrapped_text = "\n<|user|>\n" + text + "\n<|assistant|>\n"
            self.conversation_history += self.tokenizer.encode(wrapped_text, allowed_special= {'<|assistant|>', '<|user|>', '<|endoftext|>'}) # encode
        if role == "assistant":
            text = text + "<|endoftext|>"
            self.conversation_history += self.tokenizer.encode(text, allowed_special= {'<|assistant|>', '<|user|>', '<|endoftext|>'})


    def return_context(self):
        """
        Returns the right amount of context for LLMs encoded by the tokenizer.
        """
        context = self.conversation_history[-self.llm_memory_size:]

        # GPT requires tensor (time, batch), so unsqueeze it!
        return torch.as_tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)


    def to_string(self):
        """
        Prints the conversation history for a given memory size (mostly for testing).
        """
        print(f"Conversation History:")
        print(self.conversation_history[-self.llm_memory_size:])
