import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.w_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.w_query
        keys = x @ self.w_key
        values = x @ self.w_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)

        context_vecs = attn_weights @ values
        return context_vecs

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        attn_scores = queries @ keys.T
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        context_vecs = attn_weights @ values

        return context_vecs

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: int, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = attn_weights @ values

        return context_vecs

class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: int, num_heads: int, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: int, num_heads: int, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec