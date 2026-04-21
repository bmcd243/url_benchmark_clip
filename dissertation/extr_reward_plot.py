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

df = pd.read_csv("extr_reward.csv")
x_col = "train/frame"

runs = {
    "Walker": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedwalker_pixels_cnn_1 - train/extr_reward",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedwalker_pixels_clip_1 - train/extr_reward",
        "APS-CNN":    "pretrain_aps_cnn_texturedwalker_pixels_cnn_1 - train/extr_reward",
        "APS-CLIP":   "pretrain_aps_clip_texturedwalker_pixels_clip_1 - train/extr_reward",
    },
    "Quadruped": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedquadruped_pixels_cnn_1 - train/extr_reward",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedquadruped_pixels_clip_1 - train/extr_reward",
        "APS-CNN":    "pretrain_aps_cnn_texturedquadruped_pixels_cnn_1 - train/extr_reward",
        "APS-CLIP":   "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/extr_reward",
    },
    "Cheetah": {
        "DIAYN-CNN":  "pretrain_diayn_cnn_texturedcheetah_pixels_cnn_1 - train/extr_reward",
        "DIAYN-CLIP": "pretrain_diayn_clip_texturedcheetah_pixels_clip_1 - train/extr_reward",
        "APS-CNN":    "pretrain_aps_cnn_texturedcheetah_pixels_cnn_1 - train/extr_reward",
        "APS-CLIP":   "pretrain_aps_clip_texturedcheetah_pixels_clip_1 - train/extr_reward",
    },
}

colors = {
    "DIAYN-CLIP": "#EE6677",
    "DIAYN-CNN":  "#4477AA",
    "APS-CLIP":   "#228833",
    "APS-CNN":    "#CCBB44",
}

linestyles = {
    "DIAYN-CLIP": "-",
    "DIAYN-CNN":  "-",
    "APS-CLIP":   "--",
    "APS-CNN":    "--",
}

fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=False)

for ax, (domain, methods) in zip(axes, runs.items()):
    for method, mean_col in methods.items():
        sub = df[[x_col, mean_col]].dropna()
        x    = sub[x_col].values
        mean = sub[mean_col].values

        ax.plot(x, smooth(mean), label=method,
                color=colors[method],
                linestyle=linestyles[method],
                linewidth=1.2)

    ax.set_title(f"Textured{domain}")
    ax.set_xlabel("Pretraining frames")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(0, None)

axes[0].set_ylabel("Extrinsic reward")

all_handles, all_labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in all_labels:
            all_handles.append(handle)
            all_labels.append(label)

fig.legend(all_handles, all_labels, loc="lower center",
           ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
plt.savefig("fig_extr_reward.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_extr_reward.pdf")