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

df = pd.read_csv("extr_reward_texture_ablation.csv")
x = "train/frame"

plain    = "ablation_background_pretrain_aps_clip_quadruped_pixels_clip_1 - train/extr_reward"
textured = "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/extr_reward"

plain_min    = "ablation_background_pretrain_aps_clip_quadruped_pixels_clip_1 - train/extr_reward__MIN"
plain_max    = "ablation_background_pretrain_aps_clip_quadruped_pixels_clip_1 - train/extr_reward__MAX"
textured_min = "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/extr_reward__MIN"
textured_max = "pretrain_aps_clip_texturedquadruped_pixels_clip_1 - train/extr_reward__MAX"

fig, ax = plt.subplots(figsize=(5, 3))

for col, col_min, col_max, label, color in [
    (plain,    plain_min,    plain_max,    "APS-CLIP (Plain)",    "#4477AA"),
    (textured, textured_min, textured_max, "APS-CLIP (Textured)", "#228833"),
]:
    sub = df[[x, col, col_min, col_max]].dropna()
    xs  = sub[x].values
    ys  = smooth(sub[col].values)
    lo  = smooth(sub[col_min].values)
    hi  = smooth(sub[col_max].values)
    ax.plot(xs, ys, label=label, color=color, linewidth=1.2)
    ax.fill_between(xs, lo, hi, alpha=0.15, color=color)

ax.set_xlabel("Pretraining frames")
ax.set_ylabel("Extrinsic reward")
ax.set_title("Textured Quadruped vs Plain Quadruped (APS-CLIP)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("fig_texture_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_texture_ablation.pdf")