import torch
import tiktoken
import time
from tokenizer import Tokenizer
from model import Model
from model_trainer import ModelTrainer
from dataset import GptDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False,      # Query-key-value bias
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

tokenizer = tiktoken.get_encoding("gpt2")

# Open the file and read the text
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

def test_model_trainer():
    torch.manual_seed(123)

    start_time = time.time()

    model = Model(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    modelTrainer = ModelTrainer(model, optimizer, tokenizer)
    train_losses, val_losses, tokens_seen = modelTrainer.train(
        train_loader,
        val_loader,
        num_epochs=10,
        eval_freq=5, 
        eval_iter=5,
        start_context="Every effort moves you"
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, 10, len(train_losses))
    modelTrainer.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    model.to("cpu")
    model.eval()

    token_ids = model.generate(
        idx=Tokenizer.text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", Tokenizer.token_ids_to_text(token_ids, tokenizer))