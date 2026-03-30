from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.amp import autocast
import tqdm

from part1.bpe_tokenizer import BPETokenizer
from part1.model import CausalLM


# Config: edit these values directly.
TEST_PATH = Path("test.rand.txt")
OUT_PATH = Path("part1.txt")
CHECKPOINT_PATH = Path("checkpoints/best_weights.pt")
TOKENIZER_PATH = Path("bpe_tokenizer.json")
MAX_SEQ_LEN = 256
D_MODEL: int | None = None
NUM_LAYERS: int | None = None
DROPOUT = 0.35
LIMIT: int | None = None


def load_model(
    checkpoint_path: Path,
    max_seq_len: int,
    d_model: int | None,
    num_layers: int | None,
    dropout: float,
    device: torch.device,
):
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    # Infer architecture directly from checkpoint to prevent config mismatches.
    inferred_vocab_size = state_dict["lm_head.weight"].shape[0]
    inferred_d_model = state_dict["embedding.weight"].shape[1]
    layer_ids = {
        int(m.group(1))
        for key in state_dict.keys()
        if (m := re.match(r"layers\.(\d+)\.", key))
    }
    inferred_num_layers = (max(layer_ids) + 1) if layer_ids else 0

    final_d_model = inferred_d_model if d_model is None else d_model
    final_num_layers = inferred_num_layers if num_layers is None else num_layers

    model = CausalLM(
        vocab_size=inferred_vocab_size,
        max_seq_len=max_seq_len,
        d_model=final_d_model,
        num_layers=final_num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def score_sentence(model, token_ids: list[int], pad_id: int, device: torch.device, use_amp: bool) -> float:
    """Return average negative log-likelihood (perplexity proxy)."""
    if len(token_ids) < 2:
        return float("inf")

    tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
    mask = (tokens != pad_id).long()

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            lm_logits = model(tokens, mask)  # (1, T, vocab)

        # shift: predict token t+1 from position t
        log_probs = F.log_softmax(lm_logits[:, :-1], dim=-1)
        targets = tokens[:, 1:]
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return nll.mean().item()


def main() -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp}")

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
    pad_id = len(tokenizer.vocab)
    model = load_model(
        CHECKPOINT_PATH,
        MAX_SEQ_LEN,
        D_MODEL,
        NUM_LAYERS,
        DROPOUT,
        device,
    )

    lines = TEST_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    if LIMIT is not None:
        lines = lines[: LIMIT]

    predictions: list[str] = []
    for line in tqdm.tqdm(lines, desc="Scoring pairs"):
        parts = line.split("\t", maxsplit=1)
        if len(parts) < 2:
            predictions.append("A")
            continue

        sent_a, sent_b = parts[0], parts[1]
        ids_a = tokenizer.encode(sent_a)[:MAX_SEQ_LEN]
        ids_b = tokenizer.encode(sent_b)[:MAX_SEQ_LEN]

        score_a = score_sentence(model, ids_a, pad_id, device, use_amp)
        score_b = score_sentence(model, ids_b, pad_id, device, use_amp)
        label = "A" if score_a < score_b else "B"  # lower NLL = more natural
        predictions.append(label)

    OUT_PATH.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions):,} labels to {OUT_PATH}")


if __name__ == "__main__":
    main()
