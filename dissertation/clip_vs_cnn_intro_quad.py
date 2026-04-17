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
        (img_1a, img_1b, "Same pose,\nday vs night"),
        (img_2b, img_3a, "Same pose,\ndifferent position"),
        (img_3a, img_3b, "Same pose,\nclose vs far"), # Fixed img_2a to img_3a here!
    ]
    pixel_dists = [get_pixel_distance(a, b) for a, b, _ in comparisons]
    clip_dists  = [get_clip_distance(a, b, model, preprocess, device)
                   for a, b, _ in comparisons]

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.15)

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
    labels = ["Same pose, day vs night",
              "Same pose, different position",
              "Same pose, close vs far"]
    for i, lbl in enumerate(labels):
        print(f"{lbl}:")
        print(f"  Pixel L2    : {pixel_dists[i]:,.0f}")
        print(f"  CLIP cosine : {clip_dists[i]:.4f}")