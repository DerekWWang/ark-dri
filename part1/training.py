import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
import tqdm

from bpe_tokenizer import BPETokenizer
from dataloader import CausalLMDataset, BucketBatchSampler, collate_fn
from model import CausalLM

# Configs
BATCH_SIZE   = 256
TRAIN_STEPS  = 25000
LR           = 3e-4
GRAD_CLIP    = 1.0
SAVE_EVERY_STEPS = 10000
LOG_EVERY_STEPS = 500
MAX_SEQ_LEN = 256
VAL_RATIO = 0.15
VAL_EVERY_STEPS = 5000
EVAL_BATCH_PAIRS = 128
MIN_LR_RATIO = 1e-3
TARGET_WARMUP_STEPS = 2000
CHECKPOINT_DIR = Path("checkpoints-large")
CHECKPOINT_DIR.mkdir(exist_ok=True)
TOKENIZED_CACHE_PATH = CHECKPOINT_DIR / "tokenized_cache.pt"
TRAIN_DATA_PATH = Path("/workspace/ark-dri/part1/train.txt")
TOKENIZER_PATH = Path("/workspace/ark-dri/bpe_tokenizer.json")

DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
USE_AMP = DEVICE.type == "cuda"   # AMP only supported on CUDA
print(f"Using device: {DEVICE}  |  AMP: {USE_AMP}")

sentence_pairs: list[tuple[str, str]] = []
with open(TRAIN_DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", maxsplit=1)
        if len(parts) < 2:
            continue
        sentence_pairs.append((parts[0], parts[1]))

tt = BPETokenizer.load(str(TOKENIZER_PATH))
PAD_ID = len(tt.vocab)

pair_tokenized: list[tuple[list[int], list[int]]]
cache_loaded = False

if TOKENIZED_CACHE_PATH.exists():
    try:
        cache = torch.load(TOKENIZED_CACHE_PATH, map_location="cpu")
        train_stat = TRAIN_DATA_PATH.stat()
        tok_stat = TOKENIZER_PATH.stat()
        cache_meta = cache.get("meta", {})
        if (
            cache_meta.get("train_mtime_ns") == train_stat.st_mtime_ns
            and cache_meta.get("train_size") == train_stat.st_size
            and cache_meta.get("tokenizer_mtime_ns") == tok_stat.st_mtime_ns
            and cache_meta.get("tokenizer_size") == tok_stat.st_size
        ):
            if "pair_tokenized" in cache:
                pair_tokenized = cache["pair_tokenized"]
            else:
                clean_list = cache.get("clean_tokenized", cache.get("true_tokenized"))
                corrupt_list = cache.get("corrupted_tokenized")
                if clean_list is None or corrupt_list is None:
                    raise KeyError("Cache missing pair_tokenized or clean/corrupt tokenized lists")
                if len(clean_list) != len(corrupt_list):
                    keep_n = min(len(clean_list), len(corrupt_list))
                    print(
                        "Warning: cache clean/corrupt lengths differ "
                        f"({len(clean_list)} vs {len(corrupt_list)}). Using first {keep_n}."
                    )
                    clean_list = clean_list[:keep_n]
                    corrupt_list = corrupt_list[:keep_n]
                pair_tokenized = list(zip(clean_list, corrupt_list))

            cache_loaded = True
            print(f"Loaded tokenized cache: {TOKENIZED_CACHE_PATH}")
        else:
            print("Tokenized cache is stale. Recomputing tokenization...")
    except Exception as e:
        print(f"Failed to load tokenized cache ({e}). Recomputing tokenization...")

if not cache_loaded:
    print("Tokenizing clean/corrupted sentence pairs...")
    pair_tokenized = [(tt.encode(clean), tt.encode(corrupt)) for clean, corrupt in sentence_pairs]

    train_stat = TRAIN_DATA_PATH.stat()
    tok_stat = TOKENIZER_PATH.stat()
    torch.save(
        {
            "pair_tokenized": pair_tokenized,
            "meta": {
                "train_mtime_ns": train_stat.st_mtime_ns,
                "train_size": train_stat.st_size,
                "tokenizer_mtime_ns": tok_stat.st_mtime_ns,
                "tokenizer_size": tok_stat.st_size,
            },
        },
        TOKENIZED_CACHE_PATH,
    )
    print(f"Saved tokenized cache: {TOKENIZED_CACHE_PATH}")

def split_holdout(items, val_ratio: float, seed: int = 1337):
    n = len(items)
    if n < 2:
        return items[:], []

    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    val_idx = set(idxs[:n_val])

    train_split = [items[i] for i in range(n) if i not in val_idx]
    val_split = [items[i] for i in range(n) if i in val_idx]
    return train_split, val_split


filtered_pairs = [
    (clean_ids, corrupt_ids)
    for clean_ids, corrupt_ids in pair_tokenized
    if 2 <= len(clean_ids) <= MAX_SEQ_LEN and 2 <= len(corrupt_ids) <= MAX_SEQ_LEN
]
train_pairs, val_pairs = split_holdout(filtered_pairs, VAL_RATIO, seed=1337)
train_clean = [clean_ids for clean_ids, _ in train_pairs]

train_dataset = CausalLMDataset(
    train_clean,
    bucket_size=50,
    max_seq_len=MAX_SEQ_LEN,
)
use_validation = len(val_pairs) > 0

print(f"Train dataset size: {len(train_dataset):,} samples")
if use_validation:
    print(f"Validation pair count: {len(val_pairs):,}")
else:
    print("Validation dataset is empty after filtering; validation is disabled")

# usse max sequence length across the whole dataset so positional encoding
# is always large enough, regardless of which bucket a batch comes from.
print(f"Max sequence length cap: {MAX_SEQ_LEN}")

train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=lambda b: collate_fn(b, pad_id=PAD_ID),
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)
print(f"Train batches per epoch: {len(train_sampler):,}")

# model
model = CausalLM(
    vocab_size=len(tt.vocab),
    max_seq_len=MAX_SEQ_LEN,
    d_model=512,
    num_layers=4,
    dropout=0.35
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
total_train_steps = max(1, TRAIN_STEPS)
if total_train_steps <= 1000:
    warmup_steps = max(1, total_train_steps // 2)
else:
    warmup_steps = min(5000, max(1000, int(0.1 * total_train_steps)))
warmup_steps = min(warmup_steps, total_train_steps)
decay_steps = max(1, total_train_steps - warmup_steps)

def lr_lambda(current_step: int) -> float:
    step = current_step + 1
    if step <= warmup_steps:
        return max(MIN_LR_RATIO, step / max(1, warmup_steps))

    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
print(
    f"Scheduler: total_steps={total_train_steps:,}, warmup_steps={warmup_steps:,}, "
    f"decay_steps={decay_steps:,}"
)
scaler  = GradScaler(device=DEVICE.type, enabled=USE_AMP)


"""
log checkpoining and validaiton for display later
"""
history: list[dict] = []

def save_checkpoint(global_step: int, best_val_loss: float, is_best: bool, save_periodic: bool):
    state = {
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
    }
    if save_periodic:
        path = CHECKPOINT_DIR / f"ckpt_step{global_step:09d}.pt"
        torch.save(state, path)
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"weights_step{global_step:09d}.pt")
        print(f"  Saved checkpoint: {path}")
    if is_best:
        torch.save(state, CHECKPOINT_DIR / "best.pt")
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best_weights.pt")
        print("  Saved best checkpoint and weights")

def load_checkpoint(path: str):
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    history.extend(state.get("history", []))
    global_step = state.get("global_step", 0)
    best_val_loss = state.get("best_val_loss", float("inf"))
    return global_step, best_val_loss

# TODO: Resume from latest checkpoint if one exists (need to handle loading optimizer/scheduler state later)
# global_step = 0
# best_val_loss = float("inf")
# ckpts = sorted(CHECKPOINT_DIR.glob("ckpt_step*.pt"))
# if ckpts:
#     global_step, best_val_loss = load_checkpoint(str(ckpts[-1]))
#     print(f"Resumed from global step {global_step}")

# scoring loop
def score_sentence(model, token_ids, device):
    """Return average negative log-likelihood (perplexity proxy)."""
    tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
    mask = (tokens != PAD_ID).long()
    with torch.no_grad():
        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            lm_logits = model(tokens, mask)
        log_probs = F.log_softmax(lm_logits[:, :-1], dim=-1)
        targets = tokens[:, 1:]
        target_mask = (targets != PAD_ID)
        safe_targets = targets.clamp(min=0, max=log_probs.size(-1) - 1)
        nll = -log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
        nll = nll * target_mask
        denom = target_mask.sum().clamp(min=1)
        return (nll.sum() / denom).item()


def score_sentences_batch(model, token_batch: list[list[int]], device) -> list[float]:
    """Return mean NLL per sentence for a batch of tokenized sentences."""
    tokens, mask = collate_fn([torch.tensor(ids, dtype=torch.long) for ids in token_batch], pad_id=PAD_ID)
    tokens = tokens.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    with torch.no_grad():
        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            lm_logits = model(tokens, mask)
            log_probs = F.log_softmax(lm_logits[:, :-1], dim=-1)

    targets = tokens[:, 1:]
    target_mask = (targets != PAD_ID)
    safe_targets = targets.clamp(min=0, max=log_probs.size(-1) - 1)
    token_nll = -log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)

    nll_sum = (token_nll * target_mask).sum(dim=1)
    nll_count = target_mask.sum(dim=1).clamp(min=1)
    return (nll_sum / nll_count).detach().cpu().tolist()


def evaluate(val_pairs) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    total_true_nll, total = 0.0, 0
    correct_pairs = 0

    for i in tqdm.tqdm(range(0, len(val_pairs), EVAL_BATCH_PAIRS), desc="ValPairs", leave=False):
        chunk = val_pairs[i : i + EVAL_BATCH_PAIRS]
        clean_chunk = [clean for clean, _ in chunk]
        corrupt_chunk = [corrupt for _, corrupt in chunk]
        scores = score_sentences_batch(model, clean_chunk + corrupt_chunk, DEVICE)
        n = len(clean_chunk)
        clean_scores = scores[:n]
        corrupt_scores = scores[n:]

        for score_a, score_b in zip(clean_scores, corrupt_scores):
            label = "A" if score_a < score_b else "B"
            total_true_nll += score_a
            correct_pairs += int(label == "A")
            total += 1

    if total == 0:
        if was_training:
            model.train()
        return float("inf"), float("inf"), 0.0

    avg_true_nll = total_true_nll / total
    ppl = float(math.exp(min(20.0, avg_true_nll)))
    pair_acc = correct_pairs / total
    if was_training:
        model.train()
    return avg_true_nll, ppl, pair_acc


model.train()
running_loss, running_tokens = 0.0, 0
window_start_time = time.time()
train_iter = iter(train_loader)
prog = tqdm.tqdm(total=TRAIN_STEPS, initial=global_step, desc="Train", leave=False)

while global_step < TRAIN_STEPS:
    try:
        tokens, mask = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        tokens, mask = next(train_iter)

    tokens = tokens.to(DEVICE, non_blocking=True)
    mask   = mask.to(DEVICE, non_blocking=True)

    with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            lm_logits = model(tokens, mask)
            shift_logits = lm_logits[:, :-1]
            shift_targets = tokens[:, 1:]

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=PAD_ID
            )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    valid_tokens = (shift_targets != PAD_ID).sum().item()
    running_loss += loss.item() * valid_tokens
    running_tokens += valid_tokens

    global_step += 1
    prog.update(1)
    prog.set_postfix(loss=f"{(running_loss / max(1, running_tokens)):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    if global_step % LOG_EVERY_STEPS == 0:
        elapsed = time.time() - window_start_time
        train_loss = running_loss / max(1, running_tokens)
        train_ppl = float(math.exp(min(20.0, train_loss)))
        stats = {
            "type": "step_train",
            "step": global_step,
            "train_loss": round(train_loss, 6),
            "train_ppl": round(train_ppl, 6),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(stats)
        print(
            f"Step {global_step:9d}/{TRAIN_STEPS} | "
            f"train_loss {train_loss:.4f} | train_ppl {train_ppl:.2f} | "
            f"lr {stats['lr']:.2e} | {elapsed:.1f}s"
        )
        running_loss, running_tokens = 0.0, 0
        window_start_time = time.time()

        with open(CHECKPOINT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if use_validation and global_step % VAL_EVERY_STEPS == 0:
        val_loss, val_ppl, val_pair_acc = evaluate(val_pairs)
        val_stats = {
            "type": "step_val",
            "step": global_step,
            "val_loss": round(val_loss, 6),
            "val_ppl": round(val_ppl, 6),
            "val_pair_acc": round(val_pair_acc, 6),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(val_stats)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        print(
            f"Step {global_step:9d} | val_loss {val_loss:.4f} | "
            f"val_ppl {val_ppl:.2f} | pair_acc {val_pair_acc:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
            + (" ★ best" if is_best else "")
        )
        if is_best:
            save_checkpoint(global_step, best_val_loss, is_best=True, save_periodic=False)

        with open(CHECKPOINT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if global_step % SAVE_EVERY_STEPS == 0:
        save_checkpoint(global_step, best_val_loss, is_best=False, save_periodic=True)

save_checkpoint(global_step, best_val_loss, is_best=False, save_periodic=True)
if use_validation and global_step % VAL_EVERY_STEPS != 0:
    val_loss, val_ppl, val_pair_acc = evaluate(val_pairs)
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        save_checkpoint(global_step, best_val_loss, is_best=True, save_periodic=False)
    print(
        f"Final val | loss {val_loss:.4f} | ppl {val_ppl:.2f} | pair_acc {val_pair_acc:.4f} | "
        f"lr {scheduler.get_last_lr()[0]:.2e}"
        + (" ★ best" if is_best else "")
    )

with open(CHECKPOINT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)
