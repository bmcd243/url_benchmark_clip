# agent/lgsd.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dm_env import specs


import utils
from agent.ddpg import DDPGAgent


class TrajEncoder(nn.Module):
    """
    Representation function phi: E -> Z mapping CLIP embeddings to
    latent space. Lipschitz constraint enforced via Lagrange multiplier
    rather than spectral norm, following LGSD (Rho et al., 2025).
    Spectral norm enforces Lipschitz w.r.t. input L2 norm only;
    Lagrange multiplier directly penalises violations of
    ||phi(e') - phi(e)|| <= d_CLIP(o, o').
    """
    def __init__(self, clip_dim, hidden_dim, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, e):
        return self.net(e)


class LGSDAgent(DDPGAgent):
    """
    LGSD-inspired skill discovery using CLIP embeddings as the semantic
    distance metric. Replaces LGSD's LLM+sentence-transformer pipeline
    with frozen CLIP, enabling annotation-free pixel-based skill discovery.

    Key differences from original LSD (Park et al., 2022):
    - Lipschitz constraint w.r.t. CLIP cosine distance, not Euclidean
      state distance
    - Constraint enforced via Lagrange multiplier (dual gradient descent),
      not spectral normalisation
    - phi takes CLIP embeddings as input, not raw proprioceptive states

    Key differences from LGSD (Rho et al., 2025):
    - CLIP replaces LLM captioner + sentence transformer
    - Pixel-based observations, no symbolic state access required
    - URLB/DDPG backbone rather than PPO/Isaac Gym
    """
    def __init__(
        self,
        skill_dim,
        lgsd_scale,
        update_encoder,
        lagrange_init,
        lagrange_lr,
        lipschitz_epsilon,
        encoder_type='clip',
        **kwargs,
    ):
        self.skill_dim = skill_dim
        self.lgsd_scale = lgsd_scale
        self.update_encoder = update_encoder
        kwargs["encoder_type"] = encoder_type

        # Lagrange multiplier for Lipschitz constraint (LGSD Section 4.1)
        # lambda >= 0 enforced throughout
        self.lagrange      = lagrange_init
        self.lagrange_lr   = lagrange_lr
        self.lipschitz_eps = lipschitz_epsilon

        # Inject skill dim so actor/critic see [enc(obs) || skill]
        kwargs["meta_dim"] = self.skill_dim
        super().__init__(**kwargs)

        # phi takes CLIP embedding (512-dim) as input, not raw obs
        clip_dim = self.obs_dim - self.skill_dim
        self.traj_encoder = TrajEncoder(
            clip_dim, kwargs['hidden_dim'], self.skill_dim
        ).to(self.device)

        self.traj_encoder_opt = torch.optim.Adam(
            self.traj_encoder.parameters(), lr=self.lr
        )
        self.traj_encoder.train()

    # ------------------------------------------------------------------ #
    # Skill interface
    # ------------------------------------------------------------------ #

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        """
        Sample skill from isotropic Gaussian following LGSD.
        Note: LGSD uses z ~ N(0,I), not unit sphere.
        The isotropic sampling fosters diverse directions naturally.
        """
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.random.randn(self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if not finetune and time_step.last():
            return self.init_meta()
        return meta

    # ------------------------------------------------------------------ #
    # CLIP cosine distance — the semantic metric d_CLIP
    # Replaces LGSD's d_lang (Equation 2 in paper)
    # ------------------------------------------------------------------ #

    def _clip_cosine_distance(self, e, e_next):
        """
        d_CLIP(o, o') = 1 - cosine_similarity(e, e')
        where e = f_CLIP(o) / ||f_CLIP(o)||

        Since embeddings are already L2-normalised when cached,
        this reduces to: d = 1 - (e * e').sum(dim=1)

        Returns shape (B,), range [0, 2].
        Satisfies conditions (i) d(x,x)=0 and (ii) d(x,y)=d(y,x).
        Triangle inequality holds via path-length argument
        (LGSD Claim 1 / Appendix C).
        """
        # Clamp for numerical stability before acos if needed
        cos_sim = (e * e_next).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        return 1.0 - cos_sim  # (B,)

    # ------------------------------------------------------------------ #
    # Intrinsic reward
    # ------------------------------------------------------------------ #

    def compute_intr_reward(self, skill, e, e_next, step):
        """
        r(s, z, s') = (phi(e') - phi(e))^T z
        Same as LSD but phi operates in CLIP semantic space.
        """
        with torch.no_grad():
            phi_s      = self.traj_encoder(e)
            phi_s_next = self.traj_encoder(e_next)
        delta_phi = phi_s_next - phi_s
        reward = (delta_phi * skill).sum(dim=1, keepdim=True)
        return reward * self.lgsd_scale

    # ------------------------------------------------------------------ #
    # Trajectory encoder update with Lagrange Lipschitz constraint
    # LGSD Algorithm 1, lines 16-17
    # ------------------------------------------------------------------ #

    def update_traj_encoder(self, skill, e, e_next, step):
        metrics = dict()

        phi_s      = self.traj_encoder(e)
        phi_s_next = self.traj_encoder(e_next)
        delta_phi  = phi_s_next - phi_s

        # LSD/LGSD objective: minimise negative alignment 
        lsd_loss = -(delta_phi * skill).sum(dim=1).mean()

        # Lipschitz constraint violation (Using Squared L2 to match LGSD Appendix E)
        # phi_dist_sq = delta_phi.pow(2).sum(dim=1)           # (B,)
        phi_dist  = delta_phi.norm(dim=1)
        d_clip      = self._clip_cosine_distance(e, e_next) # (B,)
        violation   = phi_dist - d_clip                  # (B,)

        # Lagrange penalty (Minimisation framework)
        # Heavy penalty for violation > 0. Capped bonus for violation < 0.
        clamped_violation = torch.min(
            torch.full_like(violation, -self.lipschitz_eps),
            violation
        )
        lipschitz_loss = self.lagrange * clamped_violation.mean()

        loss = lsd_loss + lipschitz_loss

        self.traj_encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.traj_encoder_opt.step()

        # Dual gradient descent: update lambda
        with torch.no_grad():
            self.lagrange += self.lagrange_lr * violation.mean().item()
            self.lagrange  = max(0.0, self.lagrange)  # lambda >= 0

        if self.use_tb or self.use_wandb:
            metrics['lgsd_lsd_loss']         = lsd_loss.item()
            metrics['lgsd_lipschitz_loss']   = lipschitz_loss.item()
            metrics['lgsd_violation_mean']   = violation.mean().item()
            metrics['lgsd_violation_frac']   = (violation > 0).float().mean().item()
            metrics['lgsd_phi_dist']      = phi_dist.mean().item()
            metrics['lgsd_d_clip']           = d_clip.mean().item()
            metrics['lgsd_lagrange']         = self.lagrange

        return metrics

    # ------------------------------------------------------------------ #
    # Main update loop
    # ------------------------------------------------------------------ #

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        # obs is already a CLIP embedding (512-dim) if caching is active
        # Normalise here in case embeddings weren't normalised at cache time
        e      = F.normalize(obs.float(),      dim=-1)
        e_next = F.normalize(next_obs.float(), dim=-1)

        if self.reward_free:
            # 1. Update trajectory encoder with Lipschitz constraint
            metrics.update(
                self.update_traj_encoder(skill, e, e_next, step)
            )

            # 2. Compute intrinsic reward
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(
                    skill, e, e_next, step
                )

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # Detach — CLIP encoder is frozen, nothing to propagate
        e      = e.detach()
        e_next = e_next.detach()

        # Concatenate skill for actor/critic
        obs_with_skill      = torch.cat([e,      skill], dim=1)
        next_obs_with_skill = torch.cat([e_next, skill], dim=1)

        # Standard DDPG updates
        metrics.update(
            self.update_critic(
                obs_with_skill.detach(), action, reward,
                discount, next_obs_with_skill.detach(), step
            )
        )
        metrics.update(self.update_actor(obs_with_skill.detach(), step))
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    # ------------------------------------------------------------------ #
    # Finetuning skill selection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        """
        Select best skill using reward-weighted displacement.
        Same principle as LSD variant but now displacement is in
        CLIP-constrained latent space.
        """
        obs, _, extr_reward, _, next_obs, _ = utils.to_torch(
            next(replay_iter), self.device
        )

        obs_enc      = self.aug_and_encode(obs)
        next_obs_enc = self.aug_and_encode(next_obs)
        e      = F.normalize(obs_enc.float(),      dim=-1)
        e_next = F.normalize(next_obs_enc.float(), dim=-1)

        phi_s      = self.traj_encoder(e)
        phi_s_next = self.traj_encoder(e_next)
        delta_phi  = phi_s_next - phi_s   # (B, skill_dim)

        # Find skill direction most correlated with extrinsic reward
        weighted = (extr_reward * delta_phi).mean(dim=0)  # (skill_dim,)
        # LGSD uses Gaussian skills, so no unit-norm constraint needed
        # but normalise for consistency with skill sampling scale
        skill = weighted / (weighted.norm() + 1e-12)
        skill = skill.cpu().numpy().astype(np.float32)

        meta = OrderedDict()
        meta['skill'] = skill
        self.solved_meta = meta
        return meta