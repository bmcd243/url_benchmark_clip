# agent/lsd.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent
from agent.spectral_norm import spectral_norm


class TrajEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, skill_dim: int):
        super().__init__()
        te_hidden_dim = skill_dim * 16

        self.net = nn.Sequential(
            nn.Linear(obs_dim,       te_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(te_hidden_dim, te_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(te_hidden_dim, skill_dim),
        )
        self.apply(utils.weight_init)

        spectral_norm(self.net[0])
        spectral_norm(self.net[2])
        spectral_norm(self.net[4])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class LSDAgent(DDPGAgent):

    def __init__(self, skill_dim, update_encoder, encoder_type='cnn', **kwargs):
        self.skill_dim = skill_dim
        self.update_encoder = update_encoder
        # Remove this line:
        # self.te_spectral_coef = float(kwargs.pop('te_spectral_coef', 1.0))
        self.te_lr = float(kwargs.pop('traj_encoder_lr', kwargs.get('lr', 1e-4)))

        kwargs['encoder_type'] = encoder_type
        kwargs['meta_dim'] = self.skill_dim
        super().__init__(**kwargs)

        enc_dim = self.obs_dim - self.skill_dim
        self.traj_encoder = TrajEncoder(
            enc_dim,
            kwargs['hidden_dim'],
            self.skill_dim,
            # No spectral_coef argument
        ).to(self.device)

        self.traj_encoder_opt = torch.optim.Adam(
            self.traj_encoder.parameters(), lr=self.te_lr
        )
        self.traj_encoder.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        """Sample skill z ~ N(0, I); return solved_meta after finetuning."""
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.random.randn(self.skill_dim).astype(np.float32)
        skill = skill / (np.linalg.norm(skill) + 1e-12)
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
        skill = F.normalize(skill.float(), dim=1, eps=1e-12)
        obs_enc = obs_enc.float()
        next_obs_enc = next_obs_enc.float()
        with torch.no_grad():
            phi_next = self.traj_encoder(next_obs_enc)
            phi_cur = self.traj_encoder(obs_enc)
            if not (torch.isfinite(phi_next).all() and torch.isfinite(phi_cur).all()):
                return torch.zeros(
                    (obs_enc.size(0), 1),
                    device=obs_enc.device,
                    dtype=obs_enc.dtype,
                )
            delta_phi = phi_next - phi_cur
        reward = (delta_phi * skill).sum(dim=1, keepdim=True)
        if not torch.isfinite(reward).all():
            reward = torch.zeros_like(reward)
        return reward

    # ------------------------------------------------------------------
    # Trajectory encoder update
    # ------------------------------------------------------------------

    def update_traj_encoder(self, skill, obs_enc, next_obs_enc, step):
        metrics = {}

        obs_enc      = obs_enc.detach().float()
        next_obs_enc = next_obs_enc.detach().float()
        skill        = F.normalize(skill.detach().float(), dim=1, eps=1e-12)

        phi_s      = self.traj_encoder(obs_enc)
        phi_s_next = self.traj_encoder(next_obs_enc)
        delta_phi  = phi_s_next - phi_s
        loss       = -(delta_phi * skill).sum(dim=1).mean()

        self.traj_encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.traj_encoder.parameters(), max_norm=1.0)
        self.traj_encoder_opt.step()
        # No project_spectral() needed — hook fires on next forward pass

        if self.use_tb or self.use_wandb:
            metrics['lsd_loss']           = loss.item()
            metrics['lsd_delta_phi_norm'] = delta_phi.norm(dim=1).mean().item()
            metrics['lsd_phi_var']        = phi_s.var(dim=0).mean().item()

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
        skill = skill.float()
        skill_norm = F.normalize(skill, dim=1, eps=1e-12)

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
                self.update_traj_encoder(skill_norm, obs_enc, next_obs_enc, step)
            )
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(
                    skill_norm, obs_enc, next_obs_enc
                )
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
        obs_with_skill = torch.cat([obs_enc, skill_norm], dim=1)
        next_obs_with_skill = torch.cat([next_obs_enc, skill_norm], dim=1)

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