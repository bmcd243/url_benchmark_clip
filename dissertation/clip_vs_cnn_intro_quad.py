import os
import sys
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dmc

def get_clip_distance(img_a, img_b, model, preprocess, device):
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
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    return round(np.linalg.norm(a - b), 1)

def set_neutral_lighting(physics):
    physics.model.light_pos[0] = [0.0, -2.0, 3.0]
    physics.model.light_ambient[0] = [0.3] * 3
    physics.model.light_diffuse[0] = [0.8] * 3

def set_standing(physics, x=0.0, y=0.0):
    with physics.reset_context():
        physics.data.qpos[:] = 0.0
        # root position and upright quaternion
        physics.named.data.qpos['root'][:3] = [x, y, 0.57]
        physics.named.data.qpos['root'][3:] = [1, 0, 0, 0]
        # all leg joints at 0 = natural standing
        set_neutral_lighting(physics)

def set_standing_night(physics, x=0.0, y=0.0):
    with physics.reset_context():
        physics.data.qpos[:] = 0.0
        physics.named.data.qpos['root'][:3] = [x, y, 0.57]
        physics.named.data.qpos['root'][3:] = [1, 0, 0, 0]
        physics.model.light_pos[0] = [0.0, -2.0, 8.0]
        physics.model.light_ambient[0] = [0.02, 0.02, 0.05]
        physics.model.light_diffuse[0] = [0.05, 0.05, 0.12]
        physics.model.light_specular[0] = [0.0, 0.0, 0.0]

def fix_camera(physics, cam_id=0, pos=[0.0, -4.0, 1.5], quat=[0.8, 0.6, 0.0, 0.0]):
    """Set camera to fixed mode at a given position and orientation."""
    physics.model.cam_mode[cam_id] = 0  # fixed
    physics.model.cam_pos[cam_id] = pos
    physics.model.cam_quat[cam_id] = quat

def set_agent_position(physics, x=0.0):
    with physics.reset_context():
        physics.data.qpos[:] = 0.0
        physics.named.data.qpos['root'][:3] = [x, 0.0, 0.57]
        physics.named.data.qpos['root'][3:] = [1, 0, 0, 0]
        set_neutral_lighting(physics)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    env = dmc.make("texturedquadruped_stand", obs_type="pixels",
                   frame_stack=1, action_repeat=1, seed=42)
    env.reset()
    physics = env.physics

    H, W = 224, 224
    CAM = 0

    # Comparison 1: day vs night
    set_standing(physics, x=0.0)
    img_1a = physics.render(height=H, width=W, camera_id=2)

    set_standing_night(physics, x=0.0)
    img_1b = physics.render(height=H, width=W, camera_id=2)

    # Comparison 2: same pose, different position in frame
    # 1. Agent in center, move camera RIGHT (+1.0) -> Agent appears on the LEFT
    set_agent_position(physics, x=0.0)
    fix_camera(physics, cam_id=CAM, pos=[1.0, -4.0, 1.5], quat=[0.8, 0.6, 0.0, 0.0])
    img_2a = physics.render(height=H, width=W, camera_id=CAM)

    # 2. Agent in center, move camera LEFT (-1.0) -> Agent appears on the RIGHT
    set_agent_position(physics, x=0.0)
    fix_camera(physics, cam_id=CAM, pos=[-1.0, -4.0, 1.5], quat=[0.8, 0.6, 0.0, 0.0])
    img_2b = physics.render(height=H, width=W, camera_id=CAM)

    # Comparison 3: same pose, close vs far
    set_agent_position(physics, x=0.0)
    fix_camera(physics, cam_id=CAM, pos=[0.0, -2.5, 1.2], quat=[0.8, 0.6, 0.0, 0.0])
    img_3a = physics.render(height=H, width=W, camera_id=CAM)

    set_agent_position(physics, x=0.0)
    fix_camera(physics, cam_id=CAM, pos=[0.0, -8.0, 2.8], quat=[0.8, 0.6, 0.0, 0.0])
    img_3b = physics.render(height=H, width=W, camera_id=CAM)

    comparisons = [
        (img_1a, img_1b, "Same pose, day vs night"),
        (img_2b, img_3a, "Same pose, different position"),
        # (img_3a, img_3b, "Same pose,\nclose vs far"),
    ]
    pixel_dists = [get_pixel_distance(a, b) for a, b, _ in comparisons]
    clip_dists  = [get_clip_distance(a, b, model, preprocess, device)
                   for a, b, _ in comparisons]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

    n = len(comparisons)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.8 * n),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.45,
                                          "width_ratios": [1, 1, 0.75]})

    # Column titles on row 0 — all three use set_title so they align naturally
    col_titles = ["Image A", "Image B", "Distances"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row, (img_a, img_b, label) in enumerate(comparisons):
        for col, img in enumerate([img_a, img_b]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Row label centred below both images as a single horizontal line
        ax_left  = axes[row][0]
        ax_right = axes[row][1]
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        x_mid = (pos_l.x0 + pos_r.x1) / 2
        y_bot = pos_l.y0 - 0.02
        fig.text(x_mid, y_bot, label.replace("\n", ", "),
                 fontsize=10, ha="center", va="top",
                 style="italic", color="#444444")

        ax_t = axes[row][2]
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for spine in ax_t.spines.values():
            spine.set_visible(False)

        for y_pos, metric, value in [
            (0.72, "Pixel L2", f"{pixel_dists[row]:,.0f}"),
            (0.28, "CLIP cosine", f"{clip_dists[row]:.4f}"),
        ]:
            ax_t.text(0.5, y_pos + 0.09, metric,
                      transform=ax_t.transAxes,
                      fontsize=10, fontweight="bold",
                      ha="center", va="center", color="#333333")
            ax_t.text(0.5, y_pos - 0.07, value,
                      transform=ax_t.transAxes,
                      fontsize=13,
                      ha="center", va="center", color="#111111",
                      bbox=dict(boxstyle="round,pad=0.5",
                                facecolor="#f0f0f0",
                                edgecolor="#bbbbbb",
                                linewidth=0.8))

    fig.suptitle(
        "Pixel L2 distance vs CLIP cosine distance as semantic similarity measures",
        fontsize=11, fontweight="bold", y=1.01
    )

    plt.savefig("clip_vs_pixel_three_comparisons.png",
                dpi=300, bbox_inches="tight")
    plt.close()