import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, d_model, attn_dim):
        super().__init__()
        self.d_model = d_model
        self.attn_dim = attn_dim
        self.scaling_factor = attn_dim ** 0.5
        self.W_q = nn.Linear(d_model, attn_dim, bias=False)
        self.W_k = nn.Linear(d_model, attn_dim, bias=False)
        self.W_v = nn.Linear(d_model, attn_dim, bias=False)

    def forward(self, x, mask=None):
        queries = self.W_q(x)
        keys = self.W_k(x)
        scores = queries @ keys.transpose(-2, -1) / self.scaling_factor

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        values = self.W_v(x)
        return attn_weights @ values

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, attn_dim, num_heads):
        super().__init__()

        self.d_model = d_model
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        self.heads = nn.ModuleList([Attention(d_model, attn_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(attn_dim * num_heads, d_model)

    def forward(self, x, mask=None):
        # print("Input shape:", x.shape)  # Debug print
        head_outputs = [head(x, mask) for head in self.heads]
        # print("Head outputs shapes:", [h.shape for h in head_outputs])  # Debug print
        concatenated = torch.cat(head_outputs, dim=-1)
        # print("Concatenated shape:", concatenated.shape)  # Debug print
        output = self.output_projection(concatenated)
        # print("Output shape:", output.shape)  # Debug print
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, attn_dim, num_heads, ff_dim=1024, dropout=0.2):
        super().__init__()
        self.attn = MultiheadAttention(d_model, attn_dim, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, causal_mask=None):
        x = self.norm1(x + self.drop1(self.attn(x, mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CausalLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len=256, d_model=256,
                 num_layers=3, dropout=0.2):
        super().__init__()
        pad_id = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, 32, 8, ff_dim=512, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)  # (1, T, T)

        if mask is not None:
            # mask is (B, T), expand to (B, 1, T) then broadcast with (1, T, T)
            combined = causal * mask.unsqueeze(1)  # (B, T, T)
        else:
            combined = causal

        embd = self.embedding(x)
        embd = self.pos_encoding(embd)
        out = self.embed_drop(embd)

        for layer in self.layers:
            out = layer(out, combined)

        return self.lm_head(out)