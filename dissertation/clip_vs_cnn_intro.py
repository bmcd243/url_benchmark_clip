import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dmc

def get_clip_distance(img_a, img_b, model, preprocess, device):
    """Cosine distance in CLIP space (0=identical, 2=maximally dissimilar)."""
    with torch.no_grad():
        a = preprocess(Image.fromarray(img_a)).unsqueeze(0).to(device)
        b = preprocess(Image.fromarray(img_b)).unsqueeze(0).to(device)
        ea = model.encode_image(a)
        eb = model.encode_image(b)
        ea = ea / ea.norm(dim=-1, keepdim=True)
        eb = eb / eb.norm(dim=-1, keepdim=True)
        cosine_sim = (ea * eb).sum().item()
    return round(1 - cosine_sim, 4)

def get_pixel_distance(img_a, img_b):
    """L2 distance in pixel space."""
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    return round(np.linalg.norm(a - b), 1)

def set_standing_pose(physics, seed=0):
    rng = np.random.RandomState(seed)
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.0
        physics.named.data.qpos['rooty'] = rng.uniform(-0.05, 0.05)
        physics.named.data.qpos['right_hip'] = rng.uniform(-0.1, 0.1)
        physics.named.data.qpos['left_hip'] = rng.uniform(-0.1, 0.1)
        physics.named.data.qpos['right_knee'] = rng.uniform(-0.1, 0.0)
        physics.named.data.qpos['left_knee'] = rng.uniform(-0.1, 0.0)
        physics.named.data.qpos['rootx'] = rng.uniform(-0.5, 0.5)
        # Randomise lighting for background variation
        physics.model.light_pos[0] = [
            rng.uniform(-4, 4),
            rng.uniform(-4, -1),
            rng.uniform(0.5, 6),
        ]
        intensity = rng.uniform(0.2, 1.0)
        physics.model.light_ambient[0] = [intensity * 0.4] * 3
        physics.model.light_diffuse[0] = [intensity] * 3

def set_standing_pose_night(physics, seed=0):
    rng = np.random.RandomState(seed)
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.0
        physics.named.data.qpos['rooty'] = rng.uniform(-0.05, 0.05)
        physics.named.data.qpos['right_hip'] = rng.uniform(-0.1, 0.1)
        physics.named.data.qpos['left_hip'] = rng.uniform(-0.1, 0.1)
        physics.named.data.qpos['right_knee'] = rng.uniform(-0.1, 0.0)
        physics.named.data.qpos['left_knee'] = rng.uniform(-0.1, 0.0)
        physics.named.data.qpos['rootx'] = rng.uniform(-0.5, 0.5)
        # Nighttime: very dark ambient, minimal diffuse, blue-tinted
        physics.model.light_pos[0] = [0.0, -2.0, 8.0]
        physics.model.light_ambient[0] = [0.1, 0.1, 0.05]  # near black, slight blue
        physics.model.light_diffuse[0] = [0.1, 0.1, 0.12]  # dim blue moonlight
        physics.model.light_specular[0] = [0.0, 0.0, 0.0]

def set_splits_pose(physics):
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.3    # raised to clear floor
        physics.named.data.qpos['rooty'] = 0.0
        physics.named.data.qpos['right_hip'] = 1.3
        physics.named.data.qpos['left_hip'] = -1.3
        physics.named.data.qpos['right_knee'] = 0.0
        physics.named.data.qpos['left_knee'] = 0.0
        # Ankles bent outward for visual effect
        physics.named.data.qpos['right_ankle'] = 0.4
        physics.named.data.qpos['left_ankle'] = -0.4
        physics.named.data.qpos['rootx'] = 0.0
        physics.model.light_pos[0] = [0.0, -2.0, 3.0]
        physics.model.light_ambient[0] = [0.3] * 3
        physics.model.light_diffuse[0] = [0.8] * 3

def set_standing_left(physics, seed=0):
    rng = np.random.RandomState(seed)
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.0
        physics.named.data.qpos['rooty'] = 0.0
        physics.named.data.qpos['right_hip'] = 0.0
        physics.named.data.qpos['left_hip'] = 0.0
        physics.named.data.qpos['right_knee'] = -0.05
        physics.named.data.qpos['left_knee'] = -0.05
        physics.named.data.qpos['rootx'] = -100.8   # far left
        physics.model.light_pos[0] = [0.0, -2.0, 3.0]
        physics.model.light_ambient[0] = [0.3] * 3
        physics.model.light_diffuse[0] = [0.8] * 3

def set_standing_right(physics, seed=0):
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.0
        physics.named.data.qpos['rooty'] = 0.0
        physics.named.data.qpos['right_hip'] = 0.0
        physics.named.data.qpos['left_hip'] = 0.0
        physics.named.data.qpos['right_knee'] = -0.05
        physics.named.data.qpos['left_knee'] = -0.05
        physics.named.data.qpos['rootx'] = 100.8    # far right
        physics.model.light_pos[0] = [0.0, -2.0, 3.0]
        physics.model.light_ambient[0] = [0.3] * 3
        physics.model.light_diffuse[0] = [0.8] * 3

def set_minor_postural_change(physics, seed=0):
    rng = np.random.RandomState(seed)
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = 0.0
        physics.named.data.qpos['rooty'] = rng.uniform(-0.05, 0.05)
        # Slightly different arm/leg position
        physics.named.data.qpos['right_hip'] = 0.05
        physics.named.data.qpos['left_hip'] = -0.35
        physics.named.data.qpos['right_knee'] = -0.2
        physics.named.data.qpos['left_knee'] = -0.2
        physics.named.data.qpos['rootx'] = rng.uniform(-0.5, 0.5)
        # Same lighting as comparison A
        physics.model.light_pos[0] = [0.0, -2.0, 3.0]
        physics.model.light_ambient[0] = [0.3] * 3
        physics.model.light_diffuse[0] = [0.8] * 3

def set_fallen_pose(physics, seed=0):
    rng = np.random.RandomState(seed)
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)
        physics.named.data.qpos['rootz'] = -1.23
        physics.named.data.qpos['rooty'] = rng.uniform(1.5, 1.7)
        physics.named.data.qpos['right_hip'] = rng.uniform(-0.5, 0.5)
        physics.named.data.qpos['left_hip'] = rng.uniform(-0.5, 0.5)
        physics.named.data.qpos['right_knee'] = rng.uniform(-1.0, 0.0)
        physics.named.data.qpos['left_knee'] = rng.uniform(-1.0, 0.0)
        physics.named.data.qpos['rootx'] = rng.uniform(-0.5, 0.5)
        physics.model.light_pos[0] = [0.0, -2.0, 3.0]
        physics.model.light_ambient[0] = [0.3] * 3
        physics.model.light_diffuse[0] = [0.8] * 3


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    env = dmc.make("texturedwalker_flip", obs_type="pixels",
                   frame_stack=1, action_repeat=1, seed=42)
    env.reset()
    physics = env.physics

    # ------------------------------------------------------------------ #
    # Render all frames
    # ------------------------------------------------------------------ #
    H, W = 224, 224

    # Comparison 1: Same pose, different background (lighting)
    set_standing_pose(physics, seed=0)
    physics.model.light_pos[0] = [0.0, -2.0, 3.0]
    physics.model.light_ambient[0] = [0.3] * 3
    physics.model.light_diffuse[0] = [0.8] * 3
    img_1a = physics.render(height=H, width=W, camera_id=0)

    set_standing_pose_night(physics, seed=0)
    img_1b = physics.render(height=H, width=W, camera_id=0)

    # Comparison 2: Same pose, minor postural change
    set_standing_left(physics, seed=0)
    physics.model.light_pos[0] = [0.0, -2.0, 3.0]
    physics.model.light_ambient[0] = [0.3] * 3
    physics.model.light_diffuse[0] = [0.8] * 3
    img_2a = physics.render(height=H, width=W, camera_id=0)

    set_standing_right(physics, seed=0)
    img_2b = physics.render(height=H, width=W, camera_id=0)

    # Comparison 3: Standing vs splits
    set_standing_pose(physics, seed=0)
    physics.model.light_pos[0] = [0.0, -2.0, 3.0]
    physics.model.light_ambient[0] = [0.3] * 3
    physics.model.light_diffuse[0] = [0.8] * 3
    img_3a = physics.render(height=H, width=W, camera_id=0)

    set_splits_pose(physics)
    img_3b = physics.render(height=H, width=W, camera_id=0)

    # ------------------------------------------------------------------ #
    # Compute distances
    # ------------------------------------------------------------------ #
    comparisons = [
        (img_1a, img_1b, "Same pose,\nday vs night"),
        (img_2a, img_2b, "Same pose,\nminor postural change"),
        (img_3a, img_3b, "Standing vs\nsplits"),
    ]

    pixel_dists = [get_pixel_distance(a, b) for a, b, _ in comparisons]
    clip_dists  = [get_clip_distance(a, b, model, preprocess, device)
                   for a, b, _ in comparisons]

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.5, wspace=0.15)

    for row, (img_a, img_b, label) in enumerate(comparisons):
        ax_a = fig.add_subplot(gs[row, 0])
        ax_b = fig.add_subplot(gs[row, 1])
        ax_t = fig.add_subplot(gs[row, 2])

        ax_a.imshow(img_a)
        ax_a.axis('off')
        if row == 0:
            ax_a.set_title("Image A", fontsize=9, fontweight='bold')

        ax_b.imshow(img_b)
        ax_b.axis('off')
        if row == 0:
            ax_b.set_title("Image B", fontsize=9, fontweight='bold')

        ax_t.axis('off')
        if row == 0:
            ax_t.set_title("Distances", fontsize=9, fontweight='bold')

        text = (
            f"{label}\n\n"
            f"Pixel L2:  {pixel_dists[row]:,.0f}\n"
            f"CLIP cosine:  {clip_dists[row]:.4f}"
        )
        ax_t.text(0.05, 0.5, text,
                  transform=ax_t.transAxes,
                  fontsize=8.5,
                  verticalalignment='center',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='#f5f5f5',
                            edgecolor='#cccccc'))

    fig.suptitle(
        "Pixel distance vs CLIP cosine distance as semantic similarity measures",
        fontsize=10, fontweight='bold', y=0.98
    )

    plt.savefig("clip_vs_pixel_three_comparisons.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved clip_vs_pixel_three_comparisons.png")

    print("\n=== Distance Summary ===")
    labels = ["Same pose, different lighting",
              "Same pose, minor postural change",
              "Semantically distinct poses"]
    for i, lbl in enumerate(labels):
        print(f"{lbl}:")
        print(f"  Pixel L2    : {pixel_dists[i]:,.0f}")
        print(f"  CLIP cosine : {clip_dists[i]:.4f}")