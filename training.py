# %%
import numpy as np
import torch
from torch import nn

# %%
class Attention(nn.Module):
    def __init__(self, d_model, attn_dim):
        super().__init__()

        self.d_model = d_model
        self.attn_dim = attn_dim 

        self.scaling_factor = attn_dim ** 0.5

        self.W_q = nn.Linear(d_model, attn_dim, bias=False)
        self.W_k = nn.Linear(d_model, attn_dim, bias=False)
        self.W_v = nn.Linear(d_model, attn_dim, bias=False)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / self.scaling_factor
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

    def forward(self, x):
        print("Input shape:", x.shape)  # Debug print
        head_outputs = [head(x) for head in self.heads]
        print("Head outputs shapes:", [h.shape for h in head_outputs])  # Debug print
        concatenated = torch.cat(head_outputs, dim=-1)
        print("Concatenated shape:", concatenated.shape)  # Debug print
        output = self.output_projection(concatenated)
        print("Output shape:", output.shape)  # Debug print
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

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
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
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        self.layer_1 = TransformerBlock(d_model, 64, 8)
        self.layer_2 = TransformerBlock(d_model, 64, 8)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        # x: (batch, seq)
        batch_size, seq_len = x.shape
        
        embd = self.token_embedding(x)
        print("Embedding shape:", embd.shape)  # Debug print
        embd = self.pos_encoding(embd)


        out = self.layer_1(embd)
        print("After layer 1 shape:", out.shape)  # Debug print
        out = self.layer_2(out)
        print("After layer 2 shape:", out.shape)  # Debug print

        pooled = out.mean(dim=1)   # sequence classification
        print("Pooled shape:", pooled.shape)  # Debug print
        return self.cls_head(pooled)

# %% [markdown]
# ## Breaking down 

# %%
true_strings: list[str] = []
corrupted_strings: list[str] = []
with open("challenge-data/train.txt", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", maxsplit=1)
        if len(parts) < 2:
            continue
        true_strings.append(parts[0])
        corrupted_strings.append(parts[1])

# %%
from bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tt = tokenizer.load("bpe_tokenizer.json")

# %%
tt.encode(true_strings[0])

# %%
model = TransformerClassifier(vocab_size=tt.vocab.keys().__len__())
model(torch.tensor([tt.encode(true_strings[0])]))

# %% [markdown]
# f

# %%
max_len = float("-inf")
min_len = float("inf")
tokenized = []
for s in true_strings:
    enc = tt.encode(s)
    tokenized.append(enc)
    if len(enc) > max_len:
        max_len = len(enc)

    if len(enc) < min_len:
        min_len = len(enc)
print("Max sequence length:", max_len)
print("Min sequence length:", min_len)

# %%
buckets_bounds = np.arange(0, max_len + 1, 50)
buckets_bounds

# %%
buckets = {b: [] for b in buckets_bounds}
for t in tokenized:
    l = len(t)
    for b in buckets_bounds:
        if l <= b:
            buckets[b].append(t)
            break

for k, v in buckets.items():
    print(f"Bucket {k}: {len(v)} sequences")

# %%



