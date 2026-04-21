import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

steps = [100_000, 500_000, 1_000_000, 2_000_000]
labels = ["100k", "500k", "1M", "2M"]

cnn_mean  = [265, 281, 275, 331]
cnn_std   = [48,  51,  48,  43]
clip_mean = [304, 476, 381, 538]
clip_std  = [130, 31,  238, 140]

colors = {
    "APS-CNN":  "#CCBB44",
    "APS-CLIP": "#228833",
}

fig, ax = plt.subplots(figsize=(5, 3.5))

for label, means, stds in [
    ("APS-CNN",  cnn_mean,  cnn_std),
    ("APS-CLIP", clip_mean, clip_std),
]:
    means = np.array(means)
    stds  = np.array(stds)
    ax.plot(steps, means, marker="o", markersize=4,
            label=label, color=colors[label], linewidth=1.2)
    ax.fill_between(steps,
                    means - stds,
                    means + stds,
                    alpha=0.15, color=colors[label])

ax.set_xticks(steps)
ax.set_xticklabels(labels)
ax.set_xlabel("Pretraining frames")
ax.set_ylabel("Episode reward (Stand)")
ax.set_ylim(0, None)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("fig_scalability_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_scalability_ablation.pdf")