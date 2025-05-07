import os
import urllib.request
import torch
from model import Model
from tokenizer import Tokenizer
import tiktoken

file_name = "gpt2-small-124M.pth"

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True,       # Query-key-value bias
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_file = {
    "gpt2-small (124M)": "gpt2-small-124M.pth",
    "gpt2-medium (355M)": "gpt2-medium-355M.pth",
    "gpt2-large (774M)": "gpt2-large-774M.pth",
    "gpt2-xl (1558M)": "gpt2-xl-1558M.pth",
}

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{model_file[CHOOSE_MODEL]}"

if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)
    print(f"Downloaded to {file_name}")

tokenizer = tiktoken.get_encoding("gpt2")

gpt = Model(BASE_CONFIG, file_name)
gpt.eval()

token_ids = gpt.generate(
    idx=Tokenizer.text_to_token_ids("Every effort moves you", tokenizer).to(BASE_CONFIG["device"]),
    max_new_tokens=100,
    context_size=BASE_CONFIG["context_length"],
    top_k=15,
    temperature=1.0
)

print("Output text:\n", Tokenizer.token_ids_to_text(token_ids, tokenizer))