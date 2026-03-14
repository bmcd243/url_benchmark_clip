# agent/lsd.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class SpectralNormLinear(nn.Module):
    """Linear layer with manual spectral normalization. Avoids PyTorch's
    hook/parametrization machinery which has device-placement bugs in 2.x."""
    def __init__(self, in_features, out_features, n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.n_power_iterations = n_power_iterations
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('u', F.normalize(torch.randn(out_features), dim=0, eps=eps))
        self.register_buffer('v', F.normalize(torch.randn(in_features), dim=0, eps=eps))

    def forward(self, x):
        weight = self.weight
        if self.training:
            with torch.no_grad():
                u = self.u.clone()
                v = self.v.clone()
                for _ in range(self.n_power_iterations):
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps)
                    u = F.normalize(torch.mv(weight, v), dim=0, eps=self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        else:
            u = self.u
            v = self.v
        sigma = torch.dot(u, torch.mv(weight, v)).clamp(min=self.eps)
        return F.linear(x, weight / sigma, self.bias)


class TrajEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, skill_dim):
        super().__init__()
        self.fc1 = SpectralNormLinear(obs_dim, hidden_dim)
        self.fc2 = SpectralNormLinear(hidden_dim, hidden_dim)
        self.fc3 = SpectralNormLinear(hidden_dim, skill_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LSDAgent(DDPGAgent):
    def __init__(
        self,
        skill_dim,
        lsd_scale,
        update_encoder,
        encoder_type='clip',
        **kwargs,
    ):
        self.skill_dim = skill_dim
        self.lsd_scale = lsd_scale
        self.update_encoder = update_encoder

        # Inject skill dim into obs so actor/critic see [enc(obs) || skill]
        kwargs["meta_dim"] = self.skill_dim
        kwargs["encoder_type"] = encoder_type

        super().__init__(**kwargs)

        # Trajectory encoder: obs_dim here is the *encoder output* dim,
        # before the skill is concatenated. So we subtract skill_dim back out.
        traj_encoder_input_dim = self.obs_dim - self.skill_dim

        self.traj_encoder = TrajEncoder(
            traj_encoder_input_dim,
            kwargs['hidden_dim'],
            self.skill_dim,
        ).to(self.device)

        self.traj_encoder_opt = torch.optim.Adam(
            self.traj_encoder.parameters(), lr=self.lr
        )

        self.traj_encoder.train()

    # ------------------------------------------------------------------ #
    # Skill (meta) interface - same pattern as DIAYN/APS
    # ------------------------------------------------------------------ #

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.random.randn(self.skill_dim).astype(np.float32)
        # Original LSD samples raw Gaussian skills, no normalization
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        # During pretraining, resample skill every episode (at episode end)
        if not finetune and time_step.last():
            return self.init_meta()
        return meta

    # ------------------------------------------------------------------ #
    # Core LSD reward
    # ------------------------------------------------------------------ #

    def compute_intr_reward(self, skill, obs, next_obs, step):
        """
        r_t = dot(w, phi(s_{t+1}) - phi(s_t))

        This is the inner product of the skill direction with the
        displacement in embedding space. Maximising this reward encourages
        the agent to move in the direction specified by the skill vector.
        """
        with torch.no_grad():
            phi_s      = self.traj_encoder(obs)
            phi_s_next = self.traj_encoder(next_obs)

        # Displacement in embedding space
        delta_phi = phi_s_next - phi_s  # (B, skill_dim)

        # Project displacement onto skill direction
        # skill: (B, skill_dim), already on unit sphere
        reward = (delta_phi * skill).sum(dim=1, keepdim=True)  # (B, 1)

        return reward * self.lsd_scale

    # ------------------------------------------------------------------ #
    # Trajectory encoder update
    # ------------------------------------------------------------------ #

    def update_traj_encoder(self, skill, obs, next_obs, step):
        """
        Minimise -E[dot(w, phi(s') - phi(s))]
        i.e. maximise the inner product (same objective as the reward,
        but now with gradients flowing through traj_encoder).
        """
        metrics = dict()

        phi_s      = self.traj_encoder(obs)
        phi_s_next = self.traj_encoder(next_obs)
        delta_phi  = phi_s_next - phi_s

        # Negative because we want to maximise
        loss = -(delta_phi * skill).sum(dim=1).mean()

        self.traj_encoder_opt.zero_grad(set_to_none=True)
        # Only update CNN encoder if it's trainable (not CLIP)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # need to add this manual clipping as using ddpg as opposed to sac which automatically bounds the reward via entropy regularisation
        torch.nn.utils.clip_grad_norm_(self.traj_encoder.parameters(), max_norm=1.0)
        

        self.traj_encoder_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['lsd_te_loss'] = loss.item()
            metrics['lsd_delta_phi_norm'] = delta_phi.norm(dim=1).mean().item()

        return metrics

    # ------------------------------------------------------------------ #
    # Main update loop - mirrors diayn.py structure exactly
    # ------------------------------------------------------------------ #

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        # Encode pixels -> repr (or identity for states)
        obs_enc      = self.aug_and_encode(obs)
        next_obs_enc = self.aug_and_encode(next_obs)

        if self.reward_free:
            # 1. Update trajectory encoder
            metrics.update(
                self.update_traj_encoder(skill, obs_enc.detach(), next_obs_enc.detach(), step)
            )

            # 2. Compute intrinsic reward (no grad needed here)
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(
                    skill, obs_enc.detach(), next_obs_enc.detach(), step
                )

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # Detach encoder output if not updating it
        if not self.update_encoder:
            obs_enc      = obs_enc.detach()
            next_obs_enc = next_obs_enc.detach()

        # Concatenate skill to obs for actor/critic (same as DIAYN)
        obs_with_skill      = torch.cat([obs_enc,      skill], dim=1)
        next_obs_with_skill = torch.cat([next_obs_enc, skill], dim=1)

        # 3. Standard DDPG actor/critic updates
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
    # Finetuning skill selection (URLB standard: 4k probe steps)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        """
        Select best skill using displacement-based scoring over a batch
        of replay transitions collected with extrinsic reward.
        This mirrors APSAgent.regress_meta() in structure.
        """
        obs, _, extr_reward, _, next_obs, _ = utils.to_torch(
            next(replay_iter), self.device
        )

        obs_enc      = self.aug_and_encode(obs)
        next_obs_enc = self.aug_and_encode(next_obs)

        phi_s      = self.traj_encoder(obs_enc)
        phi_s_next = self.traj_encoder(next_obs_enc)
        delta_phi  = (phi_s_next - phi_s)  # (B, skill_dim)

        # Find the skill direction that best explains the reward signal:
        # solve w* = argmax_w  E[r_ext * dot(w, delta_phi)]
        # Closed form: w* = mean(r_ext * delta_phi), then normalise
        weighted = (extr_reward * delta_phi).mean(dim=0)  # (skill_dim,)
        skill = weighted / (weighted.norm() + 1e-12)
        skill = skill.cpu().numpy().astype(np.float32)

        meta = OrderedDict()
        meta['skill'] = skill
        self.solved_meta = meta
        return meta