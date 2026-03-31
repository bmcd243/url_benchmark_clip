import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import NamedTuple

class LGSDRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    skills: torch.Tensor
    next_observations: torch.Tensor
    d_lang: torch.Tensor

class LGSDRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, skill_dim=16, lang_embed_dim=512, **kwargs):
        self.skill_dim = skill_dim
        self.lang_embed_dim = lang_embed_dim
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.skills = np.zeros((self.buffer_size, self.n_envs, self.skill_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, self.lang_embed_dim), dtype=np.float32)
        self.d_lang = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, skill=None, next_obs=None, d_lang=None, **kwargs) -> None:
        self.skills[self.pos] = np.array(skill).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.d_lang[self.pos] = np.array(d_lang).copy()
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> LGSDRolloutBufferSamples:
        samples = super()._get_samples(batch_inds, env)
        return LGSDRolloutBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            old_values=samples.old_values,
            old_log_prob=samples.old_log_prob,
            advantages=samples.advantages,
            returns=samples.returns,
            skills=self.to_torch(self.skills.reshape(-1, self.skill_dim)[batch_inds]),
            next_observations=self.to_torch(self.next_observations.reshape(-1, self.lang_embed_dim)[batch_inds]),
            d_lang=self.to_torch(self.d_lang.reshape(-1)[batch_inds])
        )

class LGSDNetworks(nn.Module):
    def __init__(self, obs_dim, skill_dim, lang_embed_dim, init_lambda=300.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, skill_dim)
        )
        self.psi = nn.Sequential(
            nn.Linear(lang_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, skill_dim)
        )
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda)))

    @property
    def lambda_val(self):
        return torch.exp(self.log_lambda)

# --- Minimal Mock Environment to satisfy SB3 Initialization ---
class MockEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

class LGSD_PPO_Agent:
    def __init__(self, name, obs_type, obs_shape, action_shape, num_expl_steps,
                 encoder_type, lang_embed_dim, skill_dim, init_lambda, epsilon,
                 lgsd_lr, learning_rate, batch_size, n_steps, clip_range,
                 n_epochs, gae_lambda, gamma, ent_coef, device='cuda', **kwargs):
        
        self.device = torch.device(device)
        self.skill_dim = skill_dim
        self.lang_embed_dim = lang_embed_dim
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.ppo_obs_dim = obs_shape[0] + skill_dim
        self.action_dim = action_shape[0]
        
        if encoder_type == 'clip':
            import clip 
            self.encoder, _ = clip.load("ViT-B/32", device=self.device)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.lgsd_nets = LGSDNetworks(obs_shape[0], skill_dim, lang_embed_dim, init_lambda).to(self.device)
        self.lgsd_optimizer = torch.optim.Adam(self.lgsd_nets.parameters(), lr=lgsd_lr)
        
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ppo_obs_dim,))
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,))
        
        # Wrap the MockEnv so PPO initializes perfectly
        dummy_env = DummyVecEnv([lambda: MockEnv(obs_space, act_space)])
        
        self.ppo = PPO("MlpPolicy", env=dummy_env, learning_rate=learning_rate, n_steps=n_steps,
                       batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                       gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, device=self.device)

        sb3_logger = configure(None, format_strings=[])
        self.ppo.set_logger(sb3_logger)
        
        self.rollout_buffer = LGSDRolloutBuffer(
            self.n_steps, obs_space, act_space, device=self.device, gamma=gamma,
            gae_lambda=gae_lambda, n_envs=1, skill_dim=skill_dim, lang_embed_dim=lang_embed_dim
        )

    def get_action_and_value(self, obs, z):
        policy_input_np = np.concatenate([obs, z], axis=-1)
        policy_input_ts = torch.as_tensor(policy_input_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.ppo.policy.get_distribution(policy_input_ts)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            value = self.ppo.policy.predict_values(policy_input_ts)
        return action.cpu().numpy()[0], value, log_prob, policy_input_np

    def compute_intrinsic_reward(self, obs, next_obs, z):
        with torch.no_grad():
            phi_s = self.lgsd_nets.phi(torch.as_tensor(obs).float().to(self.device))
            phi_s_next = self.lgsd_nets.phi(torch.as_tensor(next_obs).float().to(self.device))
            z_ts = torch.as_tensor(z).float().to(self.device)
            r_int = torch.sum((phi_s_next - phi_s) * z_ts).cpu().item()
        return r_int

    def update(self, last_obs, last_z, last_done):
        with torch.no_grad():
            last_input = np.concatenate([last_obs, last_z], axis=-1)
            last_input_ts = torch.as_tensor(last_input).float().to(self.device).unsqueeze(0)
            last_value = self.ppo.policy.predict_values(last_input_ts)
        
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([last_done]))

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            # rollout_data.observations contains [state, skill]. 
            # We slice off the skill dimensions to get the pure state (1536-d).
            s = rollout_data.observations[:, :-self.skill_dim]
            s_next = rollout_data.next_observations
            z = rollout_data.skills
            d_lang = rollout_data.d_lang

            phi_s = self.lgsd_nets.phi(s)
            phi_s_next = self.lgsd_nets.phi(s_next)
            
            latent_dist_sq = torch.sum((phi_s - phi_s_next) ** 2, dim=1)
            constraint = torch.clamp(d_lang - latent_dist_sq, max=self.epsilon)
            
            r_int_batch = torch.sum((phi_s_next - phi_s) * z, dim=1)
            phi_loss = -torch.mean(r_int_batch + self.lgsd_nets.lambda_val.detach() * constraint)
            lambda_loss = torch.mean(self.lgsd_nets.lambda_val * constraint.detach())
            
            psi_e_s = self.lgsd_nets.psi(s)
            psi_loss = F.mse_loss(psi_e_s, z)
            
            total_lgsd_loss = phi_loss + lambda_loss + psi_loss
            self.lgsd_optimizer.zero_grad()
            total_lgsd_loss.backward()
            self.lgsd_optimizer.step()

        self.ppo.rollout_buffer = self.rollout_buffer
        self.ppo.train()
        
        metrics = {'train/lgsd_lambda': self.lgsd_nets.lambda_val.item()}
        if self.ppo.logger is not None:
            for key, val in self.ppo.logger.name_to_value.items():
                # Format the SB3 keys (e.g. 'train/loss') to avoid URLB WandB conflicts
                formatted_key = f"train/ppo_{key.replace('/', '_')}"
                metrics[formatted_key] = val
        
        self.rollout_buffer.reset()
        return metrics