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
    "8 skills":  "ablation_skill_8_pretrain_diayn_clip_texturedquadruped_pixels_clip_1",
    "16 skills": "pretrain_diayn_clip_texturedquadruped_pixels_clip_1",
    "32 skills": "ablation_skill_32_pretrain_diayn_clip_texturedquadruped_pixels_clip_1",
}

colors = {
    "8 skills":  "#4477AA",
    "16 skills": "#EE6677",
    "32 skills": "#228833",
}

chance = {
    "8 skills":  1/8,
    "16 skills": 1/16,
    "32 skills": 1/32,
}

for metric, csv_file, ylabel, outfile in [
    ("train/diayn_acc",   "ablation_num_skills_disc_acc.csv",  "Discriminator accuracy", "fig_skill_acc.pdf"),
    ("train/extr_reward", "ablation_num_skills_extr_reward.csv", "Extrinsic reward",       "fig_skill_extr.pdf"),
]:
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(5, 3))

    for label, prefix in runs.items():
        col = f"{prefix} - {metric}"
        if col not in df.columns:
            print(f"Missing: {col}")
            continue
        sub = df[[x_col, col]].dropna()
        xs = sub[x_col].values
        ys = smooth(sub[col].values)
        ax.plot(xs, ys, label=label, color=colors[label], linewidth=1.2)

    if metric == "train/diayn_acc":
        for label, c in chance.items():
            ax.axhline(c, color=colors[label], linewidth=0.7,
                       linestyle="--", alpha=0.5)

    ax.set_xlabel("Pretraining frames")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")