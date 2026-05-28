import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return weights @ v


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


torch.manual_seed(0)

batch_size = 2
seq_len = 4
d_model = 8
d_ff = 32

x = torch.randn(batch_size, seq_len, d_model)
mask = torch.tril(torch.ones(seq_len, seq_len))

block = TransformerBlock(d_model, d_ff)
y = block(x, mask)

print("x:", x.shape)
print("mask:", mask.shape)
print("y:", y.shape)
print("shape is preserved:", x.shape == y.shape)
