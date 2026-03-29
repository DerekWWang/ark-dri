import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import tqdm

from bpe_tokenizer import BPETokenizer
from part1.dataloader import ClassificationDataset, BucketBatchSampler, collate_fn
from part1.model import TransformerClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-4
GRAD_CLIP    = 1.0
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
USE_AMP = DEVICE.type == "cuda"   # AMP only supported on CUDA
print(f"Using device: {DEVICE}  |  AMP: {USE_AMP}")

# ── Data ───────────────────────────────────────────────────────────────────────
true_strings: list[str] = []
corrupted_strings: list[str] = []
with open("challenge-data/train.txt", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", maxsplit=1)
        if len(parts) < 2:
            continue
        true_strings.append(parts[0])
        corrupted_strings.append(parts[1])

tt = BPETokenizer.load("bpe_tokenizer.json")
PAD_ID = len(tt.vocab)

print("Tokenizing true strings...")
true_tokenized = [tt.encode(s) for s in true_strings]
print("Tokenizing corrupted strings...")
corrupted_tokenized = [tt.encode(s) for s in corrupted_strings]

dataset = ClassificationDataset(true_tokenized, corrupted_tokenized, bucket_size=50)
print(f"Dataset size: {len(dataset):,} samples")

# Use max sequence length across the whole dataset so positional encoding
# is always large enough, regardless of which bucket a batch comes from.
MAX_SEQ_LEN = max(len(ids) for ids in true_tokenized + corrupted_tokenized)
print(f"Max sequence length: {MAX_SEQ_LEN}")

sampler = BucketBatchSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=lambda b: collate_fn(b, pad_id=PAD_ID),
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)
print(f"Batches per epoch: {len(sampler):,}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = TransformerClassifier(
    vocab_size=len(tt.vocab),
    max_seq_len=MAX_SEQ_LEN,
    d_model=512,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(sampler),
    epochs=EPOCHS,
)
loss_fn = nn.BCEWithLogitsLoss()
scaler  = GradScaler(device=DEVICE.type, enabled=USE_AMP)

# ── Stats tracking ─────────────────────────────────────────────────────────────
history: list[dict] = []

def save_checkpoint(epoch: int, is_best: bool):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
    }
    path = CHECKPOINT_DIR / f"ckpt_epoch{epoch:03d}.pt"
    torch.save(state, path)
    if is_best:
        torch.save(state, CHECKPOINT_DIR / "best.pt")
    print(f"  Saved checkpoint: {path}")

def load_checkpoint(path: str):
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    history.extend(state.get("history", []))
    return state["epoch"]

# Resume from latest checkpoint if one exists
start_epoch = 0
ckpts = sorted(CHECKPOINT_DIR.glob("ckpt_epoch*.pt"))
if ckpts:
    start_epoch = load_checkpoint(str(ckpts[-1]))
    print(f"Resumed from epoch {start_epoch}")

# ── Train / eval loops ─────────────────────────────────────────────────────────
def run_epoch(dataloader, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for tokens, mask, labels in tqdm.tqdm(
            dataloader, desc="Train" if train else "Eval", leave=False
        ):
            tokens = tokens.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(tokens, mask).squeeze(-1)   # (B,) — fix for squeeze bug
                loss   = loss_fn(logits.float(), labels.float())

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * tokens.size(0)
            preds = (logits.sigmoid() > 0.5).long()
            correct += (preds == labels).sum().item()
            total   += tokens.size(0)

    return total_loss / total, correct / total


best_loss = float("inf")
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(loader, train=True)
    elapsed = time.time() - t0

    is_best = train_loss < best_loss
    if is_best:
        best_loss = train_loss

    stats = {
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 6),
        "train_acc":  round(train_acc,  6),
        "lr":         scheduler.get_last_lr()[0],
        "elapsed_s":  round(elapsed, 1),
    }
    history.append(stats)

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"loss {train_loss:.4f} | acc {train_acc:.4f} | "
        f"lr {stats['lr']:.2e} | {elapsed:.1f}s"
        + (" ★ best" if is_best else "")
    )

    save_checkpoint(epoch + 1, is_best)

    # Persist full history to JSON after every epoch
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
