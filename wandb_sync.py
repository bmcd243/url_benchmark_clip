import subprocess
import wandb
import os
from pathlib import Path

# Map run IDs to your desired names
run_names = {
    "9dhx58tx": "CLIP APS WALKER",
    "2qqdzakl": "CLIP DIAYN WALKER",
    "0si6foph": "CLIP APS QUADRUPED",
    "7xulrn65": "CLIP DIAYN QUADRUPED",
    "0i7pirwd": "ablation frame stack 1",
    "pxkgfzz3": "ablation feature dim 50",
    "3g4xvup6": "ablation 8 skills",
    "d39a9phl": "ablation 32 skills",
    "0ko53k7q": "ablation large clip",
    "s1xzog1x": "CNN DIAYN CHEETAH",
    "02jyzdrv": "CNN APS CHEETAH",
    "g4jpms0d": "CLIP DIAYN CHEETAH",
    "qun682ve": "CLIP APS CHEETAH"
}

WANDB_PROJECT = "urlb"
WANDB_ENTITY = "bm844"  # your wandb username

# Find all wandb run directories
base_dirs = [
    "/mnt/faster0/bm844/exp_local/pretrain",
    "/mnt/faster0/bm844/exp_local/finetune",
]

run_dirs = []
for base in base_dirs:
    run_dirs.extend(Path(base).rglob("run-*"))

print(f"Found {len(run_dirs)} wandb run directories")

# Sync each run
for run_dir in run_dirs:
    print(f"Syncing {run_dir}...")
    result = subprocess.run(
        ["wandb", "sync", "--no-mark-synced", str(run_dir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: {result.stderr.strip()}")
    else:
        print(f"  OK")

# Rename runs on wandb
api = wandb.Api()
for run_id, name in run_names.items():
    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        run.name = name
        run.save()
        print(f"Renamed {run_id} to {name}")
    except Exception as e:
        print(f"Failed to rename {run_id}: {e}")

print("Done")