import torch

class EmbeddingLayer:
    def embed_tokens(vocab_size: int, input_ids: torch.Tensor, output_dim: int) -> torch.Tensor:
        embedding = torch.nn.Embedding(vocab_size, output_dim)
        return embedding(input_ids)

    def embed_positions(pos_size: int, output_dim: int) -> torch.Tensor:
        embedding = torch.nn.Embedding(pos_size, output_dim)
        return embedding(torch.arange(pos_size))