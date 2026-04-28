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
    "Frame stack 1": "ablation_frame_stack_pretrain_aps_clip_texturedquadruped_pixels_clip_1",
    "Frame stack 3": "pretrain_aps_clip_texturedquadruped_pixels_clip_1",
}

colors = {
    "Frame stack 1": "#4477AA",
    "Frame stack 3": "#228833",
}

plots = [
    ("frame_stack_extr_reward.csv",      "train/extr_reward",     "Extrinsic reward"),
    ("frame_stack_intr_ent_reward.csv",  "train/intr_ent_reward", "$r^{\\mathrm{ent}}$"),
    ("frame_stack_intr_sf_reward.csv",   "train/intr_sf_reward",  "$r^{\\mathrm{sf}}$"),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2),
                         gridspec_kw={"wspace": 0.35})

for ax, (csv_file, metric, ylabel) in zip(axes, plots):
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
    ax.set_xlabel("Pretraining frames")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.set_ylim(0, None)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, -0.12))

plt.savefig("fig_frame_stack_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_frame_stack_ablation.pdf")