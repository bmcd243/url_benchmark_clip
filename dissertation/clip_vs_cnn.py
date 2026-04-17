import os

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from PIL import Image
import torch
import clip
import open_clip
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import dmc # Your URLB environment loader
from agent.ddpg import Encoder
from custom_dmc_tasks import texturedwalker

from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def extract_embeddings(frames, device):
    print("Extracting embeddings...")

    # ViT-B/32 via openai clip
    clip_model_b32, clip_preprocess_b32 = clip.load("ViT-B/32", device=device)
    clip_model_b32.eval()

    # ViT-g-14 via open_clip
    clip_model_g14, _, clip_preprocess_g14 = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k", device=device
    )
    clip_model_g14.eval()

    # Randomly initialised URLB CNN
    urlb_cnn = Encoder(obs_shape=(3, 84, 84)).to(device)
    urlb_cnn.eval()

    clip_b32_embeddings = []
    clip_g14_embeddings = []
    cnn_embeddings = []

    with torch.no_grad():
        for frame in frames:
            pil_img = Image.fromarray(frame)

            # ViT-B/32
            clip_b32_input = clip_preprocess_b32(pil_img).unsqueeze(0).to(device)
            clip_b32_emb = clip_model_b32.encode_image(clip_b32_input)
            clip_b32_embeddings.append(clip_b32_emb.cpu().numpy().flatten())

            # ViT-g-14
            clip_g14_input = clip_preprocess_g14(pil_img).unsqueeze(0).to(device)
            clip_g14_emb = clip_model_g14.encode_image(clip_g14_input)
            clip_g14_embeddings.append(clip_g14_emb.cpu().numpy().flatten())

            # CNN
            cnn_input = torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
            cnn_emb = urlb_cnn(cnn_input)
            cnn_embeddings.append(cnn_emb.cpu().numpy().flatten())

    return (
        np.array(clip_b32_embeddings),
        np.array(clip_g14_embeddings),
        np.array(cnn_embeddings),
    )

def plot_tsne(embeddings, labels, title, filename):
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    reduced_embs = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))

    unique_labels = sorted(set(labels))
    # Colourblind-friendly palette
    palette = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
               '#66CCEE', '#AA3377', '#BBBBBB']
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        ax.scatter(reduced_embs[indices, 0], reduced_embs[indices, 1],
                   c=palette[i], label=label, alpha=0.8, s=15,
                   edgecolors='none')

    ax.set_title('CLIP ViT-B/32 — coloured by pose', fontsize=12)
    ax.legend(fontsize=9, markerscale=2, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def force_pose(physics, pose_name):
    with physics.reset_context():
        physics.data.qpos[:] = np.zeros(physics.data.qpos.shape)

        # Continuous random lighting
        physics.model.light_pos[0] = [
            np.random.uniform(-4, 4),
            np.random.uniform(-4, -1),
            np.random.uniform(0.5, 6),
        ]
        intensity = np.random.uniform(0.2, 1.0)
        physics.model.light_ambient[0] = [intensity * 0.4] * 3
        physics.model.light_diffuse[0] = [intensity] * 3

        # Much smaller x offset — keep agent in frame
        # Camera is at x=0, agent at x=0 is centred
        
        rootx = np.random.uniform(-1.0, 1.0)
        physics.named.data.qpos['rootx'] = rootx
        # physics.model.cam_pos[0][0] = rootx  # camera tracks agent x

        if pose_name == "Standing":
            physics.named.data.qpos['rootz'] = 0.0
            physics.named.data.qpos['rooty'] = np.random.uniform(-0.15, 0.15)
            physics.named.data.qpos['right_hip'] = np.random.uniform(-0.2, 0.2)
            physics.named.data.qpos['left_hip'] = np.random.uniform(-0.2, 0.2)
            physics.named.data.qpos['right_knee'] = np.random.uniform(-0.2, 0.0)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-0.2, 0.0)

        elif pose_name == "Mid-Stride":
            physics.named.data.qpos['rootz'] = -0.1
            physics.named.data.qpos['rooty'] = np.random.uniform(0.1, 0.3)
            physics.named.data.qpos['right_hip'] = np.random.uniform(0.5, 0.9)
            physics.named.data.qpos['right_knee'] = np.random.uniform(-0.6, -0.3)
            physics.named.data.qpos['left_hip'] = np.random.uniform(-0.9, -0.5)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-0.8, -0.4)

        elif pose_name == "Fallen":
            physics.named.data.qpos['rootz'] = -1.23
            physics.named.data.qpos['rooty'] = np.random.uniform(1.5, 1.9)  # was 1.2–1.9
            physics.named.data.qpos['right_hip'] = np.random.uniform(-1.0, 1.0)
            physics.named.data.qpos['left_hip'] = np.random.uniform(-1.0, 1.0)
            physics.named.data.qpos['right_knee'] = np.random.uniform(-1.5, 0)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-1.5, 0)

        elif pose_name == "Flipping":
            # rootz is an offset from 1.3m. A small positive offset simulates a jump.
            physics.named.data.qpos['rootz'] = np.random.uniform(0.2, 0.5)
            physics.named.data.qpos['rooty'] = np.random.uniform(3.0, 3.8) # Upside down
            physics.named.data.qpos['right_hip'] = np.random.uniform(-0.5, 0.5)
            physics.named.data.qpos['left_hip'] = np.random.uniform(-0.5, 0.5)
            physics.named.data.qpos['right_knee'] = np.random.uniform(-1.0, 0)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-1.0, 0)

        elif pose_name == "Leaning-Fwd":
            physics.named.data.qpos['rootz'] = -0.05
            physics.named.data.qpos['rootz'] = -0.05
            physics.named.data.qpos['rooty'] = np.random.uniform(0.3, 0.5)  # was fixed 0.5
            physics.named.data.qpos['right_hip'] = np.random.uniform(0.0, 0.2)
            physics.named.data.qpos['left_hip'] = np.random.uniform(0.0, 0.2)
            physics.named.data.qpos['right_knee'] = np.random.uniform(-0.15, 0.0)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-0.15, 0.0)

        elif pose_name == "One-Leg":
            physics.named.data.qpos['rootz'] = -0.05
            physics.named.data.qpos['rooty'] = np.random.uniform(-0.05, 0.05)  # was -0.1–0.1
            physics.named.data.qpos['right_hip'] = np.random.uniform(1.1, 1.3)  # was fixed 1.2
            physics.named.data.qpos['right_knee'] = np.random.uniform(-1.1, -0.9)  # was fixed -1.0
            physics.named.data.qpos['left_hip'] = np.random.uniform(-0.05, 0.05)  # was fixed 0.0
            physics.named.data.qpos['left_knee'] = np.random.uniform(-0.08, -0.02)  # was fixed -0.05

        elif pose_name == "Landing":
            physics.named.data.qpos['rootz'] = np.random.uniform(-0.35, -0.25)
            physics.named.data.qpos['rooty'] = np.random.uniform(-0.05, 0.05)  # was -0.1–0.1, tighter
            physics.named.data.qpos['right_hip'] = np.random.uniform(0.9, 1.1)   # was 0.8, more bent
            physics.named.data.qpos['right_knee'] = np.random.uniform(-1.5, -1.2) # was -1.3
            physics.named.data.qpos['left_hip'] = np.random.uniform(0.9, 1.1)
            physics.named.data.qpos['left_knee'] = np.random.uniform(-1.5, -1.2)


def collect_data_balanced(images_per_class=200, save_dir="sanity_check_images"):
    print(f"Generating balanced dataset in '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)

    env = dmc.make("texturedwalker_flip", obs_type="pixels", frame_stack=1, action_repeat=1, seed=42)
    env.reset()

    poses = [
    "Standing",       # upright, arms down
    "Mid-Stride",     # one leg forward
    "Fallen",         # on ground
    "Flipping",       # upside down
    "Leaning-Fwd",    # torso tilted forward ~30deg, still standing
    "One-Leg",        # standing on one leg, other raised
    "Landing",        # just landed, knees bent deep, torso upright
]


    frames = []
    labels = []

    for pose in poses:
        print(f"  Generating {images_per_class} images for pose: {pose}")
        for i in range(images_per_class):
            force_pose(env.physics, pose)
            frame_hwc = env.physics.render(height=84, width=84, camera_id=0)

            filename = f"{pose}_{i:03d}.png"
            Image.fromarray(frame_hwc).save(os.path.join(save_dir, filename))

            frames.append(frame_hwc)
            labels.append(pose)

    print("Done!")
    return frames, labels


def evaluate_clustering(embeddings, labels, name):
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    # Silhouette score — how well separated the clusters are
    # Range [-1, 1] — higher is better
    sil = silhouette_score(embeddings, label_ids, metric='cosine')

    # 5-fold cross-validated kNN accuracy — can a classifier separate poses?
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_acc = cross_val_score(knn, embeddings, label_ids, cv=5).mean()

    print(f"\n{name}:")
    print(f"  Silhouette score : {sil:.4f}  (higher = better clusters)")
    print(f"  kNN accuracy     : {knn_acc:.4f}  (higher = more separable)")
    return sil, knn_acc


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames, labels = collect_data_balanced(images_per_class=200)

    clip_b32_embs, cnn_embs = extract_embeddings(frames, device)

    print("\n=== Quantitative Evaluation ===")
    evaluate_clustering(clip_b32_embs, labels, "CLIP ViT-B/32")
    evaluate_clustering(clip_g14_embs, labels, "CLIP ViT-g-14")
    evaluate_clustering(cnn_embs,      labels, "CNN (random URLB)")
    print("================================\n")

#     # Single plot set — coloured by pose only
#     # CNN should fail because lighting + position varies continuously
#     # so CNN can't use pixel location or brightness as a shortcut
    plot_tsne(clip_b32_embs, labels,
              "CLIP ViT-B/32 — coloured by pose",
              "tsne_clip_b32_by_pose.png")
    plot_tsne(clip_g14_embs, labels,
              "CLIP ViT-g-14 — coloured by pose",
              "tsne_clip_g14_by_pose.png")
    plot_tsne(cnn_embs, labels,
              "CNN (random) — coloured by pose",
              "tsne_cnn_by_pose.png")