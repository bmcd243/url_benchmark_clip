import torch
import torchvision.transforms as T
import open_clip

import utils
from agent.ddpg import DDPGAgent

class SemanticAPTAgent(DDPGAgent):
    def __init__(self, clip_model_name, clip_pretrained, 
                 knn_rms, knn_k, knn_avg, knn_clip, update_encoder, **kwargs):
        super().__init__(**kwargs)
        
        self.update_encoder = update_encoder
        
        # 1. Load the frozen CLIP model
        print(f"Loading OpenCLIP model: {clip_model_name}...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=self.device
        )
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 2. Setup the Particle-Based Entropy (PBE) estimator
        # For ViT-B-32, the embedding dimension is 512.
        self.clip_embed_dim = 512 
        
        # URLB requires an RMS object to track and normalize the KNN distances
        rms = utils.RMS(self.device)
        
        # Pass the arguments in the exact order required by your utils.py
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        
        # 3. Create the image transformation pipeline for CLIP
        self.clip_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def compute_intr_reward(self, next_obs, step):
        # Extract the most recent frame (last 3 channels from the stack)
        if next_obs.shape[1] > 3:
            recent_frame = next_obs[:, -3:, :, :]
        else:
            recent_frame = next_obs
            
        # Normalize to [0, 1] for CLIP transforms
        recent_frame = recent_frame.float() / 255.0
        clip_input = self.clip_transform(recent_frame)
        
        with torch.no_grad():
            # Get the 512-dim semantic representation from CLIP
            rep = self.clip_model.encode_image(clip_input)
            rep /= rep.norm(dim=-1, keepdim=True)
            
        # Calculate the K-Nearest Neighbor entropy reward in CLIP space!
        reward = self.pbe(rep)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # 1. Compute Intrinsic Reward (Semantic Entropy)
        if self.reward_free:
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(next_obs, step)
            reward = intr_reward
            
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # 2. Update DDPG Actor and Critic
        obs_enc = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs_enc = self.aug_and_encode(next_obs)

        if not self.update_encoder:
            obs_enc = obs_enc.detach()
            next_obs_enc = next_obs_enc.detach()

        metrics.update(
            self.update_critic(obs_enc.detach(), action, reward, discount,
                               next_obs_enc.detach(), step))
        metrics.update(self.update_actor(obs_enc.detach(), step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics