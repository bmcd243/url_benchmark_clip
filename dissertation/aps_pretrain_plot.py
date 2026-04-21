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

df_ent = pd.read_csv("intr_ent_reward.csv")
df_sf  = pd.read_csv("intr_sf_reward.csv")
x_col  = "train/frame"

domains = ["Walker", "Quadruped", "Cheetah"]

ent_runs = {
    "Walker": {
        "APS-CNN":  "pretrain_aps_cnn_texturedwalker_pixels_cnn_1 - train/intr_ent_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedwalker_pixels_clip_1 - train/intr_ent_reward",
    },
    "Quadruped": {
        "APS-CNN":  "pretrain_aps_cnn_texturedquadruped_pixels_cnn_1 - train/intr_ent_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/intr_ent_reward",
    },
    "Cheetah": {
        "APS-CNN":  "pretrain_aps_cnn_texturedcheetah_pixels_cnn_1 - train/intr_ent_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedcheetah_pixels_clip_1 - train/intr_ent_reward",
    },
}

sf_runs = {
    "Walker": {
        "APS-CNN":  "pretrain_aps_cnn_texturedwalker_pixels_cnn_1 - train/intr_sf_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedwalker_pixels_clip_1 - train/intr_sf_reward",
    },
    "Quadruped": {
        "APS-CNN":  "pretrain_aps_cnn_texturedquadruped_pixels_cnn_1 - train/intr_sf_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/intr_sf_reward",
    },
    "Cheetah": {
        "APS-CNN":  "pretrain_aps_cnn_texturedcheetah_pixels_cnn_1 - train/intr_sf_reward",
        "APS-CLIP": "pretrain_aps_clip_texturedcheetah_pixels_clip_1 - train/intr_sf_reward",
    },
}

colors = {
    "APS-CLIP": "#228833",
    "APS-CNN":  "#CCBB44",
}

fig, axes = plt.subplots(2, 3, figsize=(7, 4.2), sharey=False)

for col, domain in enumerate(domains):
    # Row 0: entropy reward
    ax = axes[0][col]
    for method, col_name in ent_runs[domain].items():
        if col_name not in df_ent.columns:
            print(f"Missing: {col_name}")
            continue
        sub = df_ent[[x_col, col_name]].dropna()
        x = sub[x_col].values
        y = sub[col_name].values
        ax.plot(x, smooth(y), label=method,
                color=colors[method], linewidth=1.2)
    ax.set_title(f"Textured{domain}")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    if col == 0:
        ax.set_ylabel("$r^{\\mathrm{ent}}$")
    ax.set_ylim(0, None)

    # Row 1: SF reward
    ax = axes[1][col]
    for method, col_name in sf_runs[domain].items():
        if col_name not in df_sf.columns:
            print(f"Missing: {col_name}")
            continue
        sub = df_sf[[x_col, col_name]].dropna()
        x = sub[x_col].values
        y = sub[col_name].values
        ax.plot(x, smooth(y), label=method,
                color=colors[method], linewidth=1.2)
    ax.set_xlabel("Pretraining frames")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    if col == 0:
        ax.set_ylabel("$r^{\\mathrm{sf}}$")
    ax.set_ylim(0, None)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig("fig_aps_intr_rewards.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_aps_intr_rewards.pdf")