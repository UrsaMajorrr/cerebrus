import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        torch.manual_seed(123)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
        print(attn_weights)
        context_vector = attn_weights @ values
        return context_vector

class SelfAttentionStable(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
        print(attn_weights)
        context_vector = attn_weights @ values
        return context_vector
        

inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.22, 0.88, 0.33],
    [0.57, 0.12, 0.99],
    [0.71, 0.23, 0.44],
    [0.11, 0.54, 0.77],
    [0.98, 0.36, 0.22]
])

self_attention = SelfAttention(d_in=3, d_out=2)
print(self_attention(inputs))
self_attention_stable = SelfAttentionStable(d_in=3, d_out=2)
print(self_attention_stable(inputs))

queries = self_attention_stable.W_query(inputs)
keys = self_attention_stable.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

context_legnth = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_legnth, context_legnth))
masked_simple = attn_weights*mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_legnth, context_legnth), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

torch.manual_seed(123)
dropout = nn.Dropout(0.5)
print(dropout(attn_weights))

batch = torch.stack([inputs, inputs], dim=0)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ values
        return context_vector

context_length = batch.shape[1]
ca = CausalSelfAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
context_vector = ca(batch)
print(context_vector)

class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

multi_head_attention = MultiHeadAttentionSimple(d_in=3, d_out=1, context_length=context_length, n_heads=2, dropout=0.0)
context_vector = multi_head_attention(batch)
print(context_vector)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).to(keys.device).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector