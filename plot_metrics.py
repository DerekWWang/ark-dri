from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(history_path: Path):
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    train_steps = []
    train_loss = []
    val_steps = []
    val_acc = []

    for row in history:
        row_type = row.get("type")
        step = row.get("step")
        if step is None:
            continue

        if row_type == "step_train" and "train_loss" in row:
            train_steps.append(step)
            train_loss.append(row["train_loss"])
        elif row_type == "step_val" and "val_acc" in row:
            val_steps.append(step)
            val_acc.append(row["val_acc"])

    return train_steps, train_loss, val_steps, val_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train loss and validation accuracy across training steps."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("checkpoints/history.json"),
        help="Path to history JSON file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path (e.g. plots/metrics.png).",
    )
    args = parser.parse_args()

    train_steps, train_loss, val_steps, val_acc = load_metrics(args.history)

    if not train_steps and not val_steps:
        raise SystemExit(f"No metrics found in {args.history}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if train_steps:
        ax1.plot(train_steps, train_loss, color="tab:blue", label="Train Loss", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    if val_steps:
        ax2.plot(val_steps, val_acc, color="tab:orange", label="Val Accuracy", linewidth=2)
    ax2.set_ylabel("Val Accuracy", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    if lines:
        ax1.legend(lines, labels, loc="best")

    plt.title("Train Loss and Validation Accuracy vs Step")
    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
