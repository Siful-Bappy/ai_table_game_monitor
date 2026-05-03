"""
Generates Figure 1 — System Architecture block diagram for the paper.
Output: additionals/fig1_system_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")

# ── colour palette ────────────────────────────────────────────────────────
C_INPUT  = "#dce8f5"   # light blue  — input
C_PROC   = "#d4edda"   # light green — processing
C_MODEL  = "#fff3cd"   # light amber — DL model
C_OUTPUT = "#f8d7da"   # light red   — output
EDGE     = "#333333"
TXT      = "#111111"

# ── layout constants ─────────────────────────────────────────────────────
BOX_H   = 1.1
BOX_W   = 1.9
Y_TOP   = 3.7          # top row  (main pipeline)
Y_BOT   = 1.2          # bottom row (sub-detail)
ARROW_Y = Y_TOP + BOX_H / 2

def box(ax, x, y, w, h, label, sublabel="", color=C_PROC, fontsize=10):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=TXT, zorder=4)
        ax.text(x + w/2, y + h*0.28, sublabel,
                ha="center", va="center", fontsize=8,
                color="#444444", style="italic", zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=TXT, zorder=4)

def arrow(ax, x_start, x_end, y, color="#555555"):
    ax.annotate(
        "", xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.6, mutation_scale=16),
        zorder=2,
    )

def down_arrow(ax, x, y_start, y_end, color="#888888"):
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.2, mutation_scale=12),
        zorder=2,
    )

# ── x positions for the 6 main boxes ─────────────────────────────────────
xs = [0.2, 2.3, 4.4, 6.5, 8.6, 10.7]
labels = [
    ("Video\nInput",       "",                   C_INPUT),
    ("Table\nAlignment",   "Homography",          C_PROC),
    ("YOLO\nDetection",    "YOLOv11-nano",        C_MODEL),
    ("SORT\nTracking",     "Multi-object",        C_PROC),
    ("Card\nClassifier",   "MobileNetV1-1.25×",   C_MODEL),
    ("Baccarat\nLogic",    "Score + Overlay",     C_OUTPUT),
]

for i, (lbl, sub, col) in enumerate(labels):
    box(ax, xs[i], Y_TOP, BOX_W, BOX_H, lbl, sub, color=col)

# arrows between main boxes
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + BOX_W, xs[i+1], ARROW_Y)

# ── bottom row: detail boxes under YOLO and Classifier ───────────────────
# Under YOLO (xs[2])
box(ax, xs[2], Y_BOT, BOX_W, 0.9,
    "Hand regions",  "Player / Banker", C_PROC, fontsize=9)
down_arrow(ax, xs[2] + BOX_W/2, Y_TOP, Y_BOT + 0.9)

# Under Classifier (xs[4])
box(ax, xs[4], Y_BOT, BOX_W, 0.9,
    "53 classes",  "52 cards + back", C_MODEL, fontsize=9)
down_arrow(ax, xs[4] + BOX_W/2, Y_TOP, Y_BOT + 0.9)

# ── Video source label ────────────────────────────────────────────────────
ax.text(xs[0] + BOX_W/2, Y_TOP + BOX_H + 0.22,
        "IP Camera\n(H.264)", ha="center", fontsize=8,
        color="#555555", style="italic")

# ── Output label ─────────────────────────────────────────────────────────
ax.text(xs[5] + BOX_W/2, Y_TOP + BOX_H + 0.22,
        "Annotated\nStream / MP4", ha="center", fontsize=8,
        color="#555555", style="italic")

# ── Title ─────────────────────────────────────────────────────────────────
ax.text(7.1, 4.85,
        "Fig. 1.  AI Baccarat Table Monitoring — System Pipeline",
        ha="center", va="center", fontsize=11, fontweight="bold", color=TXT)

# ── Legend ────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=EDGE, label="Input / Output"),
    mpatches.Patch(facecolor=C_PROC,   edgecolor=EDGE, label="Classical processing"),
    mpatches.Patch(facecolor=C_MODEL,  edgecolor=EDGE, label="Deep learning model"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=EDGE, label="Game logic"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=8, framealpha=0.85, edgecolor="#aaaaaa",
          bbox_to_anchor=(1.0, 0.01))

plt.tight_layout(pad=0.3)
out = "fig1_system_architecture.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
