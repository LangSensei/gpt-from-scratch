import torch
from attention import SelfAttention, SelfAttentionV2, CausalAttention, MultiHeadAttentionWrapper, MultiHeadAttention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
batch_size = batch.shape[0]
d_out = 2

def test_self_attention():
    sa = SelfAttention(inputs.shape[-1], d_out)
    output = sa(inputs)
    assert output.shape == (inputs.shape[0], d_out), f"Expected shape ({inputs.shape[0]}, {d_out}), got {output.shape}"

def test_self_attention_v2():
    sa_v2 = SelfAttentionV2(inputs.shape[-1], d_out, inputs.shape[0])
    output = sa_v2(inputs)
    assert output.shape == (inputs.shape[0], d_out), f"Expected shape ({inputs.shape[0]}, {d_out}), got {output.shape}"

def test_causal_attention():
    ca = CausalAttention(batch.shape[-1], d_out, batch.shape[1], 0.0)
    output = ca(batch)
    assert output.shape == (batch.shape[0], batch.shape[1], d_out), f"Expected shape ({batch.shape[0]}, {batch.shape[1]}, {d_out}), got {output.shape}"

def test_multi_head_attention_wrapper():
    num_heads = 2
    mha = MultiHeadAttentionWrapper(batch.shape[-1], d_out, batch.shape[1], 0.0, num_heads=num_heads)
    output = mha(batch)
    assert output.shape == (batch.shape[0], batch.shape[1], d_out* num_heads), f"Expected shape ({batch.shape[0]}, {batch.shape[1]}, {d_out* num_heads}), got {output.shape}"

def test_multi_head_attention():
    num_heads = 2
    mha = MultiHeadAttention(batch.shape[-1], d_out * num_heads, batch.shape[1], 0.0, num_heads=num_heads)
    output = mha(batch)
    assert output.shape == (batch.shape[0], batch.shape[1], d_out* num_heads), f"Expected shape ({batch.shape[0]}, {batch.shape[1]}, {d_out* num_heads}), got {output.shape}"