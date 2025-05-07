import tiktoken
import torch
from model_trainer import ModelTrainer
from model import Model
from dataset import GptDataset

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

model = Model(GPT_CONFIG_124M, "gpt_model.pth")
tokenizer = tiktoken.get_encoding("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
train_ratio = 0.90
split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]

train_loader = GptDataset.create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = GptDataset.create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

modelTrainer = ModelTrainer(model, optimizer, tokenizer)
train_losses, val_losses, tokens_seen = modelTrainer.train(
    train_loader,
    val_loader,
    num_epochs=10,
    eval_freq=5, 
    eval_iter=5,
    start_context="Every effort moves you"
)

model.save("gpt_model.pth")

model.eval() # disable dropout

while True:
    prompt = input("Please enter your prompt: ")
    max_new_tokens = int(input("Please enter the expected token size: "))

    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = model.generate(
        idx=encoded_tensor, 
        max_new_tokens=max_new_tokens, 
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print(tokenizer.decode(out.squeeze(0).tolist()))