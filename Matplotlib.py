import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Load CSV logs
# =========================

base = pd.read_csv("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/base/mlp_learning_curves.csv")
cnn  = pd.read_csv("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/Target/cnn_learning_curves.csv")

# =========================
# Global matplotlib style
# =========================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# =========================
# Create figure (2x2)
# =========================

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), sharex='col')

# -------------------------
# (a) Base Accuracy
# -------------------------
axes[0, 0].plot(base["epoch"], base["accuracy"],
                linestyle='-', linewidth=1.8, label="Train")
axes[0, 0].plot(base["epoch"], base["val_accuracy"],
                linestyle='--', linewidth=1.8, label="Validation")
axes[0, 0].set_title("(a) Base Model – Accuracy")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend(frameon=False)

# -------------------------
# (b) Base Loss
# -------------------------
axes[0, 1].plot(base["epoch"], base["loss"],
                linestyle='-', linewidth=1.8)
axes[0, 1].plot(base["epoch"], base["val_loss"],
                linestyle='--', linewidth=1.8)
axes[0, 1].set_title("(b) Base Model – Loss")
axes[0, 1].set_ylabel("Loss")

# -------------------------
# (c) CNN Accuracy
# -------------------------
axes[1, 0].plot(cnn["epoch"], cnn["accuracy"],
                linestyle='-', linewidth=1.8)
axes[1, 0].plot(cnn["epoch"], cnn["val_accuracy"],
                linestyle='--', linewidth=1.8)
axes[1, 0].set_title("(c) CNN – Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")

# -------------------------
# (d) CNN Loss
# -------------------------
axes[1, 1].plot(cnn["epoch"], cnn["loss"],
                linestyle='-', linewidth=1.8)
axes[1, 1].plot(cnn["epoch"], cnn["val_loss"],
                linestyle='--', linewidth=1.8)
axes[1, 1].set_title("(d) CNN – Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")

# =========================
# Layout & export
# =========================

for ax in axes.flat:
    ax.grid(False)
    ax.set_xlim(left=0)

plt.tight_layout()

plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/learning_curves_neurips_ieee.png", bbox_inches="tight")
plt.show()
