import tiktoken
import torch
from model import Model

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # Query-key-value bias
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

model = Model(GPT_CONFIG_124M)
model.eval() # disable dropout

tokenizer = tiktoken.get_encoding("gpt2")

def test_model():
    expected_token_size = 4
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    assert len(encoded) == expected_token_size, f"Expected length 4, got {len(encoded)}"

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    assert encoded_tensor.shape == (1, expected_token_size), f"Expected shape (1, 4), got {encoded_tensor.shape}"

    max_new_tokens = 100

    out = model.generate(
        idx=encoded_tensor, 
        max_new_tokens=max_new_tokens, 
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    assert out.shape == (1, max_new_tokens+expected_token_size), f"Expected shape (1, {max_new_tokens+expected_token_size}), got {out.shape}"

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    assert decoded_text.startswith(f"{start_context}"), f"Expected text to start with '{start_context}', got '{decoded_text}'"
