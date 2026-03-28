# agent/lsd.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent

from torch.nn.utils import spectral_norm

class TrajEncoder(nn.Module):
    """
    Representation function phi mapping observations (or embeddings) to a
    latent skill space.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, skill_dim: int):
        super().__init__()
        
        # Build the standard unconstrained network first
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),
        )
        
        #   Apply standard URLB weight initialization
        self.apply(utils.weight_init)

        # Apply the custom LSD spectral norm to the linear layers
        # The linear layers are at indices 0, 2, and 4 in the Sequential block
        spectral_norm(self.net[0])
        spectral_norm(self.net[2])
        spectral_norm(self.net[4])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LSDAgent(DDPGAgent):

    def __init__(
        self,
        skill_dim: int,
        update_encoder: bool,
        encoder_type: str = 'cnn',
        **kwargs,
    ):
        self.skill_dim = skill_dim
        self.update_encoder = update_encoder

        # Inject encoder type and skill meta-dim before building base agent
        kwargs['encoder_type'] = encoder_type
        kwargs['meta_dim'] = self.skill_dim

        super().__init__(**kwargs)

        # phi input dim = encoder output dim (obs_dim already includes meta_dim)
        enc_dim = self.obs_dim - self.skill_dim
        self.traj_encoder = TrajEncoder(
            enc_dim, kwargs['hidden_dim'], self.skill_dim
        ).to(self.device)

        self.traj_encoder_opt = torch.optim.Adam(
            self.traj_encoder.parameters(), lr=self.lr
        )
        self.traj_encoder.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        """Sample skill z ~ N(0, I); return solved_meta after finetuning."""
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.random.randn(self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        """Resample skill at episode boundaries during pre-training."""
        if not finetune and time_step.last():
            return self.init_meta()
        return meta


    def compute_intr_reward(
        self,
        skill: torch.Tensor,
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        r = (phi(e') - phi(e))^T z      shape: (B, 1)
        """
        with torch.no_grad():
            delta_phi = self.traj_encoder(next_obs_enc) - self.traj_encoder(obs_enc)
        reward = (delta_phi * skill).sum(dim=1, keepdim=True)
        return reward

    # ------------------------------------------------------------------
    # Trajectory encoder update
    # ------------------------------------------------------------------

    def update_traj_encoder(
        self,
        skill: torch.Tensor,
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        step: int,
    ) -> dict:
        metrics = {}

        # 1. STRICT DETACH: This absolutely prevents the OOM by cutting off the CNN/CLIP graph
        obs_enc = obs_enc.detach()
        next_obs_enc = next_obs_enc.detach()

        # 2. Forward pass through the spectral_norm MLP
        phi_s      = self.traj_encoder(obs_enc)
        phi_s_next = self.traj_encoder(next_obs_enc)
        
        # 3. Mathematically safe single-pass loss (No spectral_norm collapse)
        delta_phi  = phi_s_next - phi_s                          
        loss = -(delta_phi * skill).sum(dim=1).mean()

        # 4. Standard Optimization
        self.traj_encoder_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        self.traj_encoder_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['lsd_loss']             = loss.item()
            metrics['lsd_delta_phi_norm']   = delta_phi.norm(dim=1).mean().item()
            metrics['lsd_phi_s_norm']       = phi_s.norm(dim=1).mean().item()

        return metrics

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def update(self, replay_iter, step: int) -> dict:
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        # ---- encode ----
        # For CLIP:  aug_and_encode returns the pre-cached embedding directly.
        # For CNN:   aug_and_encode applies augmentation + CNN encoder.
        obs_enc      = self.aug_and_encode(obs)
        next_obs_enc = self.aug_and_encode(next_obs)

        # L2-normalise CLIP embeddings (stored normalised but kept safe here)
        if self.obs_precached:
            obs_enc      = F.normalize(obs_enc.float(),      dim=-1)
            next_obs_enc = F.normalize(next_obs_enc.float(), dim=-1)

        # Detach encoder output when we do NOT want to update encoder weights
        # (CLIP always; CNN when update_encoder=False)
        if not self.update_encoder:
            obs_enc      = obs_enc.detach()
            next_obs_enc = next_obs_enc.detach()

        # ---- trajectory encoder + intrinsic reward ----
        if self.reward_free:
            metrics.update(
                self.update_traj_encoder(skill, obs_enc, next_obs_enc, step)
            )
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, obs_enc, next_obs_enc)
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # Always detach before feeding actor / critic
        obs_enc      = obs_enc.detach()
        next_obs_enc = next_obs_enc.detach()

        # Concatenate skill for actor / critic (same layout as DIAYN / LGSD)
        obs_with_skill      = torch.cat([obs_enc,      skill], dim=1)
        next_obs_with_skill = torch.cat([next_obs_enc, skill], dim=1)

        metrics.update(
            self.update_critic(
                obs_with_skill, action, reward,
                discount, next_obs_with_skill, step,
            )
        )
        metrics.update(self.update_actor(obs_with_skill, step))
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    # ------------------------------------------------------------------
    # Finetuning — select best skill from extrinsic reward signal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def regress_meta(self, replay_iter, step: int) -> OrderedDict:
        """
        Identify the skill direction most correlated with extrinsic reward.

        weighted_delta = E[ r_ext * (phi(e') - phi(e)) ]   shape: (skill_dim,)

        The skill is then normalised to unit norm for consistency with
        Gaussian sampling scale.
        """
        obs, _, extr_reward, _, next_obs, _ = utils.to_torch(
            next(replay_iter), self.device
        )

        obs_enc      = self.aug_and_encode(obs)
        next_obs_enc = self.aug_and_encode(next_obs)

        if self.obs_precached:
            obs_enc      = F.normalize(obs_enc.float(),      dim=-1)
            next_obs_enc = F.normalize(next_obs_enc.float(), dim=-1)

        delta_phi = (
            self.traj_encoder(next_obs_enc) - self.traj_encoder(obs_enc)
        )  # (B, skill_dim)

        # Reward-weighted average displacement
        weighted = (extr_reward * delta_phi).mean(dim=0)   # (skill_dim,)
        skill    = weighted / (weighted.norm() + 1e-12)
        skill    = skill.cpu().numpy().astype(np.float32)

        meta = OrderedDict()
        meta['skill'] = skill
        self.solved_meta = meta

        print(
            f'[LSD] regress_meta: skill norm={np.linalg.norm(skill):.4f}, '
            f'skill={np.round(skill, 3)}'
        )
        return meta