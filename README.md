# CLIP-Enhanced Unsupervised Reinforcement Learning Benchmark (URLB)

This codebase extends [URLB](https://github.com/rll-research/url_benchmark) with frozen OpenCLIP visual encoders as a drop-in replacement for the default CNN encoder, investigating whether rich pretrained visual representations improve unsupervised skill discovery under pixel observations.

The base codebase was adapted from [DrQv2](https://github.com/facebookresearch/drqv2). The DDPG agent and training scripts were developed by Denis Yarats. All authors contributed to developing individual baselines for URLB.

## Key Contributions

- **CLIP encoder integration**: a frozen OpenCLIP ViT-B/32 encoder replaces the trainable CNN for pixel observations, with stacked-frame concatenation and pre-cached embeddings for training efficiency
- **Textured environments**: custom MuJoCo domains (`texturedwalker`, `texturedquadruped`, `texturedcheetah`) with realistic sky, floor, and robot textures designed to better utilise CLIP's visual priors
- **CLIP variants of skill-based agents**: `diayn_clip` and `aps_clip` replace the CNN encoder with CLIP while keeping the skill discovery objective unchanged

## Requirements

A CUDA-capable GPU is required. Install dependencies via:
```sh
pip install -r requirements.txt
```

The key additional dependencies over the original URLB are `open_clip_torch` and `clip`.

## Implemented Agents

| Agent | CNN command | CLIP command | Paper |
|---|---|---|---|
| DIAYN | `agent=diayn_cnn` | `agent=diayn_clip` | [paper](https://arxiv.org/abs/1802.06070) |
| APS | `agent=aps_cnn` | `agent=aps_clip` | [paper](http://proceedings.mlr.press/v139/liu21b.html) |
| DDPG | `agent=ddpg` | — | [paper](https://arxiv.org/abs/1509.02971) |

## Available Domains

| Domain | Tasks | Notes |
|---|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` | Original URLB |
| `quadruped` | `walk`, `run`, `stand`, `jump` | Original URLB |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` | Original URLB |
| `texturedwalker` | `stand`, `walk`, `run`, `flip` | Textured variant |
| `texturedquadruped` | `stand`, `walk`, `run`, `jump` | Textured variant |
| `texturedcheetah` | `run`, `flip` | Textured variant |

## Observation Modes

| Mode | Command |
|---|---|
| States | `obs_type=states` |
| Pixels (CNN encoder) | `obs_type=pixels encoder_type=cnn` |
| Pixels (CLIP encoder) | `obs_type=pixels encoder_type=clip` |

## Instructions

### Pre-training

To pre-train DIAYN with a CLIP encoder on the textured walker:
```sh
python pretrain.py agent=diayn_clip domain=texturedwalker obs_type=pixels
```

To pre-train APS with a CLIP encoder on the textured quadruped:
```sh
python pretrain.py agent=aps_clip domain=texturedquadruped obs_type=pixels
```

Snapshots are saved at `100k`, `500k`, `1M`, and `2M` frames to the directory specified by `snapshot_dir` in `pretrain.yaml`. A final snapshot is also saved unconditionally when training completes.

To run with Weights and Biases logging:
```sh
python pretrain.py agent=diayn_clip domain=texturedwalker obs_type=pixels use_wandb=true
```

### Fine-tuning

Fine-tuning loads a pre-trained snapshot and trains on a downstream task using extrinsic reward. Pass the snapshot path directly:
```sh
python finetune.py \
  agent=diayn_clip \
  task=texturedwalker_run \
  obs_type=pixels \
  snapshot_path=/path/to/snapshot_2000000.pt \
  reward_free=false \
  seed=1
```

To run three seeds in parallel using Hydra multirun:
```sh
python finetune.py -m \
  agent=diayn_clip \
  task=texturedwalker_run \
  obs_type=pixels \
  snapshot_path=/path/to/snapshot_2000000.pt \
  reward_free=false \
  seed=1,2,3 \
  use_wandb=true
```

### Monitoring

Logs are stored under the directory specified by `hydra.run.dir` in `pretrain.yaml` / `finetune.yaml`. To launch tensorboard:
```sh
tensorboard --logdir /path/to/exp_local
```

Console output format:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```

| Key | Meaning |
|---|---|
| F | Total environment frames |
| S | Total agent steps |
| E | Total episodes |
| L | Episode length |
| R | Episode return |
| FPS | Training throughput |
| T | Total training time |