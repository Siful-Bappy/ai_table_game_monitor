"""
Compression Degradation Evaluation
====================================
Evaluates MobileNet card classification accuracy across H.264 CRF levels.

Experiment design
-----------------
    Train  : clean augmented images         (working/best.pt)
    Test   : CRF-degraded val images        (dataset_gkl_cards/compressed_val/)
    Metric : Top-1 accuracy, cross-entropy loss at each CRF level

Expected output
---------------
    working/compression_eval/
        results.json            raw numbers
        results.csv             for spreadsheet / paper table
        accuracy_vs_crf.png     drop curve plot

Usage
-----
    # Step 1 — generate degraded val sets (run once)
    python compress_val_dataset.py

    # Step 2 — evaluate and plot
    python compression_eval.py

    # Custom paths
    python compression_eval.py --checkpoint   working/best.pt \\
                                --clean-val    dataset_gkl_cards/augmented/val \\
                                --compressed   dataset_gkl_cards/compressed_val \\
                                --output       working/compression_eval
"""

import os
import json
import csv
from glob import glob
from torch.utils.data import DataLoader
from torchvision import transforms

import mobilenet_card_classifier as mobilenet_card
from mobilenet_card_classifier import MyDataset, DeviceDL, TrainHelper


# DEFAULT_CRF_LEVELS = [ 42, 47, 51]
DEFAULT_CRF_LEVELS = [0, 18, 23, 28, 35, 42, 47, 51]


# -----------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------

def _make_val_loader(val_dir: str, card_label_list: list,
                     mean: list, std: list, batch_size: int):
    """Build a DataLoader for one val directory."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img_paths = glob(os.path.join(val_dir, "*/*.jpg"))
    if not img_paths:
        return None
    return DeviceDL(DataLoader(
        MyDataset(img_paths, card_label_list, tf=tf),
        batch_size=batch_size, shuffle=False, num_workers=4,
    ))


def evaluate_split(model, val_dir: str, card_label_list: list,
                   mean: list, std: list, batch_size: int = 64):
    """
    Evaluate the model on one val directory.

    Returns
    -------
    dict with keys: val_acc, val_loss, n_images  — or None if dir is empty
    """
    loader = _make_val_loader(val_dir, card_label_list, mean, std, batch_size)
    if loader is None:
        return None
    result = TrainHelper().evaluation(model, loader)
    result["n_images"] = len(glob(os.path.join(val_dir, "*/*.jpg")))
    return result


def run_compression_eval(
    checkpoint_path: str,
    clean_val_dir: str,
    compressed_dir: str,
    card_labels_path: str = "card_labels.json",
    crf_levels: list = None,
    output_dir: str = "working/compression_eval",
    batch_size: int = 64,
):
    """
    Full compression degradation evaluation pipeline.

    Parameters
    ----------
    checkpoint_path  : path to trained model weights (.pt)
    clean_val_dir    : path to clean val images
    compressed_dir   : root folder containing crf_18/, crf_23/, ... sub-folders
    card_labels_path : path to card_labels.json
    crf_levels       : list of CRF values to test
    output_dir       : where to save results and plot
    batch_size       : DataLoader batch size
    """
    crf_levels = crf_levels or DEFAULT_CRF_LEVELS
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    with open(card_labels_path, encoding="utf-8") as f:
        card_label_list = json.load(f)
    print(f"Classes  : {len(card_label_list)}")

    # Load dataset stats (mean/std from training)
    mean, std = mobilenet_card.load_dataset_stats()

    # Load trained model
    model = mobilenet_card.load_model(checkpoint_path)
    if model is None:
        raise RuntimeError(f"Could not load model from '{checkpoint_path}'")

    results = []

    # ── Baseline: clean val ─────────────────────────────────────────────
    print("\nEvaluating clean val ...")
    r = evaluate_split(model, clean_val_dir, card_label_list, mean, std, batch_size)
    if r:
        entry = {"crf": "clean", **r}
        results.append(entry)
        print(f"  clean     →  acc={r['val_acc']:.4f}  loss={r['val_loss']:.5f}"
              f"  ({r['n_images']} images)")

    # ── Each CRF level ──────────────────────────────────────────────────
    for crf in crf_levels:
        val_dir = os.path.join(compressed_dir, f"crf_{crf}")
        if not os.path.isdir(val_dir):
            print(f"  SKIP crf_{crf}  — folder not found: {val_dir}")
            print(f"       Run  python compress_val_dataset.py  first.")
            continue
        print(f"Evaluating CRF {crf:>2} ...")
        r = evaluate_split(model, val_dir, card_label_list, mean, std, batch_size)
        if r:
            entry = {"crf": crf, **r}
            results.append(entry)
            print(f"  CRF {crf:>2}    →  acc={r['val_acc']:.4f}  loss={r['val_loss']:.5f}"
                  f"  ({r['n_images']} images)")

    if not results:
        print("No results collected — nothing to save.")
        return []

    # ── Save results ────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "results.json")
    csv_path  = os.path.join(output_dir, "results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crf", "val_acc", "val_loss", "n_images"])
        writer.writeheader()
        writer.writerows(results)

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'CRF':<10} {'Top-1 Acc (%)':>15} {'Loss':>12} {'Images':>10}")
    print("-" * 55)
    clean_acc = None
    for r in results:
        acc_pct = r["val_acc"] * 100
        drop    = f"  (−{clean_acc - acc_pct:.2f}%)" if clean_acc and r["crf"] != "clean" else ""
        print(f"{str(r['crf']):<10} {acc_pct:>14.2f}% {r['val_loss']:>12.5f}"
              f" {r['n_images']:>10}{drop}")
        if r["crf"] == "clean":
            clean_acc = acc_pct
    print("=" * 55)
    print(f"\nSaved → {json_path}")
    print(f"Saved → {csv_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    _plot_accuracy_curve(results, output_dir)

    return results


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def _plot_accuracy_curve(results: list, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    clean  = next((r for r in results if r["crf"] == "clean"), None)
    crf_rs = [r for r in results if r["crf"] != "clean"]
    if not crf_rs:
        return

    crfs = [r["crf"] for r in crf_rs]
    accs = [r["val_acc"] * 100 for r in crf_rs]

    _, ax1 = plt.subplots(figsize=(9, 5))

    # Accuracy line
    color_acc = "steelblue"
    ax1.plot(crfs, accs, marker='o', linewidth=2.5, color=color_acc,
             label="Top-1 Accuracy")
    ax1.set_xlabel("H.264 CRF  (←  high quality  |  low quality  →)", fontsize=12)
    ax1.set_ylabel("Top-1 Accuracy (%)", color=color_acc, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_xticks(crfs)
    ax1.set_ylim(max(0, min(accs) - 5), 101)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Loss on secondary axis
    losses  = [r["val_loss"] for r in crf_rs]
    ax2 = ax1.twinx()
    color_loss = "tomato"
    ax2.plot(crfs, losses, marker='s', linewidth=2, linestyle='--',
             color=color_loss, label="Cross-entropy Loss")
    ax2.set_ylabel("Cross-entropy Loss", color=color_loss, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_loss)

    # Clean baseline
    if clean:
        ax1.axhline(clean["val_acc"] * 100, color='green', linestyle=':',
                    linewidth=1.5, label=f"Clean baseline  ({clean['val_acc']*100:.2f}%)")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    ax1.set_title("MobileNet Card Classification\n"
                  "Top-1 Accuracy & Loss vs. H.264 Compression (CRF)",
                  fontsize=13)
    ax1.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "accuracy_vs_crf.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved → {plot_path}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate MobileNet accuracy vs H.264 CRF compression level.")
    parser.add_argument("--checkpoint",   default="working/best.pt",
                        help="Trained model weights (.pt)")
    parser.add_argument("--clean-val",    default="dataset_gkl_cards/augmented/val",
                        help="Clean validation folder")
    parser.add_argument("--compressed",   default="dataset_gkl_cards/compressed_val",
                        help="Root folder with crf_XX/ sub-folders")
    parser.add_argument("--labels",       default="card_labels.json")
    parser.add_argument("--crf", type=int, nargs="+", default=DEFAULT_CRF_LEVELS,
                        help="CRF levels to evaluate  (default: 18 23 28 35 42 51)")
    parser.add_argument("--output",       default="working/compression_eval",
                        help="Output folder for results and plot")
    parser.add_argument("--batch-size",   type=int, default=64)
    args = parser.parse_args()

    run_compression_eval(
        checkpoint_path  = args.checkpoint,
        clean_val_dir    = args.clean_val,
        compressed_dir   = args.compressed,
        card_labels_path = args.labels,
        crf_levels       = args.crf,
        output_dir       = args.output,
        batch_size       = args.batch_size,
    )
