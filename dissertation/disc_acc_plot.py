import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def smooth(y, weight=0.85):
    last = y[0]
    result = []
    for v in y:
        last = last * weight + (1 - weight) * v
        result.append(last)
    return np.array(result)

df = pd.read_csv("disc_acc.csv")
x_col = "train/frame"

runs = {
    "Walker": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedwalker_pixels_cnn_1 - train/diayn_acc",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedwalker_pixels_clip_1 - train/diayn_acc",
    },
    "Quadruped": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedquadruped_pixels_cnn_1 - train/diayn_acc",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedquadruped_pixels_clip_1 - train/diayn_acc",
    },
    "Cheetah": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedcheetah_pixels_cnn_1 - train/diayn_acc",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedcheetah_pixels_clip_1 - train/diayn_acc",
    },
}

colors = {
    "DIAYN-CLIP": "#EE6677",
    "DIAYN-CNN":  "#4477AA",
}

fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=False)

for ax, (domain, methods) in zip(axes, runs.items()):
    for method, mean_col in methods.items():
        sub = df[[x_col, mean_col]].dropna()
        x    = sub[x_col].values
        mean = sub[mean_col].values

        ax.plot(x, smooth(mean), label=method,
                color=colors[method], linewidth=1.2)

    ax.axhline(1/16, linestyle="--", color="grey",
               linewidth=0.8, label="Chance (6.25%)")
    ax.set_title(f"Textured{domain}")
    ax.set_xlabel("Pretraining frames")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Discriminator accuracy")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.12))

plt.tight_layout()
plt.savefig("fig_disc_acc.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_disc_acc.pdf")