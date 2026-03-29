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

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / self.scaling_factor

        if mask is not None:
            keys_mask = mask.unsqueeze(1)  # (B, 1, T)
            scores = scores.masked_fill(keys_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        values = self.W_v(x)
        attn_output = attn_weights @ values
        return attn_output

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
    def __init__(self, d_model, attn_dim, num_heads, ff_dim=1024):
        super().__init__()
        self.attn = MultiheadAttention(d_model, attn_dim, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
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

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_seq_len=256, d_model=512):
        super().__init__()
        pad_id = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        self.layer_1 = TransformerBlock(d_model, 64, 8)
        self.layer_2 = TransformerBlock(d_model, 64, 8)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq)
        batch_size, seq_len = x.shape
        
        embd = self.embedding(x)
        # print("Embedding shape:", embd.shape)  # Debug print
        embd = self.pos_encoding(embd)


        out = self.layer_1(embd, mask)

        out = self.layer_2(out, mask)
        # print("After layer 2 shape:", out.shape)  # Debug print

        if mask is not None:
            mask = mask.unsqueeze(-1).float()      # (B, T, 1)
            out = out * mask
            pooled = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)
        # print("Pooled shape:", pooled.shape)  # Debug print
        return self.cls_head(pooled)