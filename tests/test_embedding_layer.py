import torch
from dataset import GptDataset
from embedding_layer import EmbeddingLayer

# Token embeddings
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

batch_size = 8
max_length = 4
dataloader = GptDataset.create_dataloader(
    raw_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

vocab_size = 50257  # GPT-2 vocab size
output_dim = 256  # Embedding dimension

def test_embed_tokens():
    # [8, 4, 256]
    # batch_size is 8 so it has 8 rows
    # max_length is 4 so it has 4 tokens in a row
    # output_dim is 256 so each token is embedded into a 256-dimensional vector
    token_embeddings = EmbeddingLayer.embed_tokens(vocab_size=vocab_size, input_ids=inputs, output_dim=output_dim)
    assert token_embeddings.shape == (batch_size, max_length, output_dim), "Token embeddings shape mismatch"

def test_embed_positions():
    # [4, 256]
    # max_length is 4 so it has 4 tokens in a row
    # output_dim is 256 so each token is embedded into a 256-dimensional vector
    pos_embeddings = EmbeddingLayer.embed_positions(pos_size=max_length, output_dim=output_dim)
    assert pos_embeddings.shape == (max_length, output_dim), "Position embeddings shape mismatch"

def test_embed_input():
    token_embeddings = EmbeddingLayer.embed_tokens(vocab_size=vocab_size, input_ids=inputs, output_dim=output_dim)
    pos_embeddings = EmbeddingLayer.embed_positions(pos_size=max_length, output_dim=output_dim)
    input_embeddings = token_embeddings + pos_embeddings
    assert input_embeddings.shape == (batch_size, max_length, output_dim), "Input embeddings shape mismatch"