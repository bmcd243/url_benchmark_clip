import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

api = wandb.Api()

def get_run_history(run_id, keys, entity="latent-skills", project="urlb"):
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.scan_history(keys=keys)
    df = pd.DataFrame(history)
    return df

def smooth(values, weight=0.85):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)

# --- Configure your run IDs here ---
runs = {
    "DIAYN-CNN":  ["run_id_seed1", "run_id_seed2", "run_id_seed3"],
    "DIAYN-CLIP": ["run_id_seed1", "run_id_seed2", "run_id_seed3"],
}

metric = "train/diayn_acc"

# Academic style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})

fig, ax = plt.subplots(figsize=(4.5, 3))

colors = {"DIAYN-CNN": "#4477AA", "DIAYN-CLIP": "#EE6677"}

for label, run_ids in runs.items():
    all_values = []
    frames = None
    for run_id in run_ids:
        df = get_run_history(run_id, keys=["train/frame", metric])
        df = df.dropna().sort_values("train/frame")
        if frames is None:
            frames = df["train/frame"].values
        # interpolate to common frame axis
        interp = np.interp(frames, df["train/frame"].values,
                           df[metric].values)
        all_values.append(interp)

    mean = np.mean(all_values, axis=0)
    std  = np.std(all_values, axis=0)
    mean_s = smooth(mean)

    ax.plot(frames, mean_s, label=label,
            color=colors[label], linewidth=1.5)
    ax.fill_between(frames,
                    smooth(mean - std),
                    smooth(mean + std),
                    alpha=0.15, color=colors[label])

# Chance level for DIAYN
ax.axhline(1/16, linestyle="--", color="grey",
           linewidth=0.8, label="Chance (6.25%)")

ax.set_xlabel("Pretraining frames")
ax.set_ylabel("Discriminator accuracy")
ax.set_title("TexturedCheetah — DIAYN pretraining")
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("pretrain_cheetah_diayn.pdf", bbox_inches="tight")
plt.close()