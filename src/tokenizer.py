import re
import torch
import tiktoken

class Vocabulary:
    def __init__(self, file_path: str):
        # Open the file and read the text
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Preprocess the text: split by whitespace and punctuation
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        # Remove empty strings and strip whitespace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        # Create a set of unique tokens
        self.data = {token:integer for integer,token in enumerate(all_tokens)}

class Tokenizer:
    def __init__(self, vocab: Vocabulary):
        # Store the vocabulary
        self.str_to_int = vocab.data
        # Create a reverse mapping from integer to string
        self.int_to_str = {i : s for s, i in vocab.data.items()}

    def encode(self, text: str):
        # Preprocess the text: split by whitespace and punctuation
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove empty strings and strip whitespace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown tokens with "<unk>"
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        # Convert tokens to their corresponding integer IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

    def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())