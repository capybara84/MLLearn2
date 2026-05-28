import math

import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = weights @ v
    return out, weights


torch.manual_seed(0)

batch_size = 2
seq_len = 4
d_k = 8

q = torch.randn(batch_size, seq_len, d_k)
k = torch.randn(batch_size, seq_len, d_k)
v = torch.randn(batch_size, seq_len, d_k)

out, weights = scaled_dot_product_attention(q, k, v)

print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)
print("weights:", weights.shape)
print("out:", out.shape)
print("weight row sums:")
print(weights.sum(dim=-1))

mask = torch.tril(torch.ones(seq_len, seq_len))
masked_out, masked_weights = scaled_dot_product_attention(q, k, v, mask)

print()
print("causal mask:")
print(mask)
print("masked weights for first batch:")
print(masked_weights[0])
print("masked out:", masked_out.shape)
