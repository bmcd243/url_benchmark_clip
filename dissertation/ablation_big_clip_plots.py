import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
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

x_col = "train/frame"

runs = {
    "APS-CLIP (ViT-B/32)": "pretrain_aps_clip_texturedquadruped_pixels_clip_1",
    "APS-CLIP (ViT-g/14)": "big_clip_pretrain_aps_clip_texturedquadruped_pixels_clip_1",
}

colors = {
    "APS-CLIP (ViT-B/32)": "#228833",
    "APS-CLIP (ViT-g/14)": "#AA3377",
}

plots = [
    ("ablation_big_clip_extr.csv",             "train/extr_reward",      "Extrinsic reward",          (0, 0)),
    ("ablation_big_clip_intr.csv",             "train/intr_reward",      "Intrinsic reward",          (0, 1)),
    ("ablation_big_clip_intr_ent_reward.csv",  "train/intr_ent_reward",  "$r^{\\mathrm{ent}}$",       (1, 0)),
    ("ablation_big_clip_intr_sf_reward.csv",   "train/intr_sf_reward",   "$r^{\\mathrm{sf}}$",        (1, 1)),
]

fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))

for csv_file, metric, ylabel, (row, col) in plots:
    ax = axes[row][col]
    df = pd.read_csv(csv_file)
    for label, prefix in runs.items():
        col_name = f"{prefix} - {metric}"
        if col_name not in df.columns:
            print(f"Missing: {col_name}")
            continue
        sub = df[[x_col, col_name]].dropna()
        xs = sub[x_col].values
        ys = smooth(sub[col_name].values)
        ax.plot(xs, ys, label=label, color=colors[label], linewidth=1.2)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.set_ylim(0, None)
    if row == 1:
        ax.set_xlabel("Pretraining frames")

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig("fig_big_clip_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_big_clip_ablation.pdf")