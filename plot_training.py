"""
Training Curve Plotter
======================
Reads history.pickle files produced by mobilenet_card_classifier.py
and generates publication-quality training/validation curves.

Single experiment:
    python plot_training.py

Compare multiple experiments (clean vs CRF variants):
    python plot_training.py --compare

Custom paths:
    python plot_training.py --histories working/history.pickle \\
                                         working/crf28/history.pickle \\
                                         working/crf38/history.pickle \\
                            --labels "Clean" "CRF-28" "CRF-38" \\
                            --output working/training_curves.png
"""

import os
import pickle
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# -----------------------------------------------------------------------
# Colours / style  (colour-blind safe palette)
# -----------------------------------------------------------------------
PALETTE = [
    "#2171b5",   # blue   — clean
    "#cb181d",   # red    — crf28
    "#238b45",   # green  — crf38
    "#6a51a3",   # purple
    "#d94801",   # orange
]


def load_history(path: str):
    """Load a history.pickle and return lists of train_loss, val_loss, val_acc."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    train_loss = [r["train_loss"] for r in records]
    val_loss   = [r["val_loss"]   for r in records]
    val_acc    = [r["val_acc"]    for r in records]
    return train_loss, val_loss, val_acc


def plot_single(history_path: str, output_path: str):
    """Three-panel figure: train loss | val loss | val accuracy."""
    train_loss, val_loss, val_acc = load_history(history_path)
    epochs = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("MobileNet Card Classifier — Training Curves", fontsize=14, fontweight="bold")

    color_train = PALETTE[0]
    color_val   = PALETTE[1]

    # ── Train loss ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color=color_train, linewidth=2, label="Train loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Val loss ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, val_loss, color=color_val, linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy Loss", fontsize=11)
    ax.set_title("Validation Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Val accuracy ─────────────────────────────────────────────────────
    ax = axes[2]
    acc_pct = [v * 100 for v in val_acc]
    ax.plot(epochs, acc_pct, color=color_val, linewidth=2, label="Val accuracy")
    best_acc = max(acc_pct)
    best_ep  = acc_pct.index(best_acc) + 1
    ax.axhline(best_acc, color="gray", linestyle="--", linewidth=1,
               label=f"Best {best_acc:.2f}% (ep {best_ep})")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Validation Accuracy", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _add_epoch_count(axes, len(epochs))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_combined(history_paths, labels, output_path: str):
    """
    Two-panel figure for a research paper:
        Left  : train loss for all experiments
        Right : val accuracy for all experiments
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "MobileNet Card Classifier — Training vs. H.264 CRF Compression",
        fontsize=13, fontweight="bold",
    )

    max_epochs = 0

    for i, (path, label) in enumerate(zip(history_paths, labels)):
        if not os.path.exists(path):
            print(f"  SKIP {path}  (not found)")
            continue

        train_loss, val_loss, val_acc = load_history(path)
        epochs  = list(range(1, len(train_loss) + 1))
        max_epochs = max(max_epochs, len(epochs))
        acc_pct = [v * 100 for v in val_acc]
        color   = PALETTE[i % len(PALETTE)]

        ax_loss.plot(epochs, train_loss, color=color, linewidth=2,
                     label=label)
        ax_acc.plot(epochs, acc_pct, color=color, linewidth=2,
                    label=f"{label}  (best {max(acc_pct):.2f}%)")

    # ── Loss panel ───────────────────────────────────────────────────────
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Training Cross-entropy Loss", fontsize=12)
    ax_loss.set_title("Training Loss", fontsize=12)
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # ── Accuracy panel ───────────────────────────────────────────────────
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Validation Top-1 Accuracy (%)", fontsize=12)
    ax_acc.set_title("Validation Accuracy", fontsize=12)
    ax_acc.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_acc.legend(fontsize=10, loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_all_metrics(history_paths, labels, output_dir: str):
    """
    Four-panel figure (2×2) — best for a research paper appendix:
        Train loss  |  Val loss
        Val acc     |  Train-val acc gap (overfitting indicator)
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MobileNet Card Classifier — Full Training Diagnostics",
        fontsize=14, fontweight="bold",
    )

    ax_tl, ax_vl = axes[0]
    ax_va, ax_gap = axes[1]

    for i, (path, label) in enumerate(zip(history_paths, labels)):
        if not os.path.exists(path):
            print(f"  SKIP {path}  (not found)")
            continue

        train_loss, val_loss, val_acc = load_history(path)
        epochs  = list(range(1, len(train_loss) + 1))
        acc_pct = [v * 100 for v in val_acc]
        # proxy train accuracy from loss (lower loss ≈ higher acc on train set)
        # We don't record train acc separately, so use val_acc - gap as proxy
        # Just show loss curves and val accuracy + overfitting gap via losses
        color = PALETTE[i % len(PALETTE)]
        ls    = ["-", "--", ":", "-."][i % 4]

        ax_tl.plot(epochs, train_loss, color=color, linewidth=2,
                   linestyle=ls, label=label)
        ax_vl.plot(epochs, val_loss,   color=color, linewidth=2,
                   linestyle=ls, label=label)
        ax_va.plot(epochs, acc_pct,    color=color, linewidth=2,
                   linestyle=ls, label=f"{label}  ({max(acc_pct):.2f}%)")

        # Generalisation gap: val_loss - train_loss  (positive = overfitting)
        gap = [vl - tl for tl, vl in zip(train_loss, val_loss)]
        ax_gap.plot(epochs, gap, color=color, linewidth=2,
                    linestyle=ls, label=label)

    for ax, title, ylabel in [
        (ax_tl,  "Training Loss",        "Cross-entropy Loss"),
        (ax_vl,  "Validation Loss",      "Cross-entropy Loss"),
        (ax_va,  "Validation Accuracy",  "Top-1 Accuracy (%)"),
        (ax_gap, "Generalisation Gap\n(Val Loss − Train Loss)",
                                         "Loss Gap  (↑ = overfitting)"),
    ]:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_va.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_gap.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    out = os.path.join(output_dir, "full_diagnostics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _add_epoch_count(axes, n_epochs):
    for ax in axes:
        ax.set_xlim(1, n_epochs)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

DEFAULT_HISTORIES = [
    "working/history.pickle",
    "working/crf28/history.pickle",
    "working/crf38/history.pickle",
]
DEFAULT_LABELS = ["Clean (original)", "CRF-28 train", "CRF-38 train"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves from history.pickle files.")
    parser.add_argument("--histories", nargs="+",
                        default=DEFAULT_HISTORIES,
                        help="One or more history.pickle paths")
    parser.add_argument("--labels",    nargs="+",
                        default=DEFAULT_LABELS,
                        help="Legend label for each history file")
    parser.add_argument("--output",    default="working/training_curves.png",
                        help="Output path for the combined plot")
    parser.add_argument("--compare",   action="store_true",
                        help="Also generate the 4-panel diagnostics figure")
    parser.add_argument("--single",    default=None,
                        help="Plot single-experiment 3-panel figure for this pickle")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.single:
        # Single experiment — 3-panel figure
        out = args.single.replace(".pickle", "_curves.png")
        plot_single(args.single, out)

    else:
        # Filter to existing files only
        existing = [(h, l) for h, l in zip(args.histories, args.labels)
                    if os.path.exists(h)]

        if not existing:
            print("No history files found. Run training first.")
        else:
            paths, labels = zip(*existing)

            if len(paths) == 1:
                # Only one experiment — use the 3-panel single view
                plot_single(paths[0], args.output)
            else:
                # Multiple experiments — combined comparison
                plot_combined(list(paths), list(labels), args.output)

            if args.compare or len(paths) > 1:
                plot_all_metrics(list(paths), list(labels),
                                 os.path.dirname(args.output) or "working")
