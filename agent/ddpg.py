from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import torchvision.transforms as T

import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class CLIPEncoder(nn.Module):
    """Frozen OpenCLIP encoder with optional stacked-frame concatenation."""
    def __init__(self,
                 obs_shape,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 clip_model_name="ViT-g-14",
                 clip_pretrained="laion2b_s34b_b88k",
                 concat_stacked_frames=True,
                 normalize_embeddings=True,
                 freeze=True):
        super().__init__()
        self.device = device
        self.concat_stacked_frames = concat_stacked_frames

        assert len(obs_shape) == 3, "Expected obs_shape=(C,H,W)"
        in_channels = obs_shape[0]
        assert in_channels % 3 == 0, f"Pixel channels must be divisible by 3, got {in_channels}"
        self.num_frames = in_channels // 3

        # Load OpenCLIP once
        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            device=self.device)
        self.model.eval()
        
        # Freeze all CLIP parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.base_dim = int(self.model.visual.output_dim)
        self.repr_dim = (self.base_dim * self.num_frames
                 if self.concat_stacked_frames else self.base_dim)

        # Transform to resize URLB's 84x84 images up to CLIP's expected 224x224
        self.resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)

        mean = getattr(self.model.visual, 'image_mean',
                       (0.48145466, 0.4578275, 0.40821073))
        std = getattr(self.model.visual, 'image_std',
                      (0.26862954, 0.26130258, 0.27577711))
        self.normalize = T.Normalize(mean=mean, std=std)
        self.use_amp = str(self.device).startswith('cuda')
        self.normalize_embeddings = normalize_embeddings

    def forward(self, obs):
        # obs is stacked RGB: (B, 3*T, H, W)
        B, C, H, W = obs.shape
        assert C % 3 == 0, f"Expected channels divisible by 3, got {C}"
        num_frames = C // 3

        # URLB stores pixel observations in [0,255] uint8.
        obs = obs / 255.0

        # (B, T, 3, H, W) -> (B*T, 3, H, W)
        frames = obs.reshape(B, num_frames, 3, H, W).reshape(B * num_frames,
                                                              3, H, W)
        frames = self.resize(frames)
        frames = self.normalize(frames)

        with torch.no_grad():
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb = self.model.encode_image(frames)
            else:
                emb = self.model.encode_image(frames)

        emb = emb.float()
        if self.normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        emb = emb.reshape(B, num_frames, self.base_dim)
        if self.concat_stacked_frames:
            emb = emb.reshape(B, num_frames * self.base_dim)
        else:
            emb = emb.mean(dim=1)
        return emb


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 meta_dim=0,
                 encoder_type='cnn',
                 clip_model_name='ViT-B-32',
                 clip_pretrained='openai',
                 clip_concat_stacked_frames=True,
                 clip_normalize=True):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            if encoder_type == 'clip':
                self.encoder = CLIPEncoder(
                    obs_shape=obs_shape,
                    device=device,
                    clip_model_name=clip_model_name,
                    clip_pretrained=clip_pretrained,
                    concat_stacked_frames=clip_concat_stacked_frames,
                    normalize_embeddings=clip_normalize,
                    freeze=not update_encoder
                    ).to(device)
                self.update_encoder = update_encoder  # CLIP weights are frozen — no optimizer needed
                self.obs_precached = True
                print(
                    f"[DDPGAgent] OpenCLIP encoder loaded ({clip_model_name}, {clip_pretrained}), "
                    f"frames={self.encoder.num_frames}, repr_dim={self.encoder.repr_dim}")
            elif encoder_type == 'cnn':
                self.encoder = Encoder(obs_shape).to(device)
                self.update_encoder = True
                self.obs_precached = False
                print(f"[DDPGAgent] Using CNN Encoder, repr_dim={self.encoder.repr_dim}")
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
            self.update_encoder = False
            self.obs_precached = False

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if obs_type == 'pixels' and self.update_encoder:
            # Only create encoder optimizer for trainable encoders (CNN)
            # CLIP encoder has frozen weights so no optimizer needed
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # Only copy encoder if it was trained (not frozen CLIP)
        if self.update_encoder:
            utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode, obs_already_encoded=False):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = obs if obs_already_encoded else self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        # actor_loss = -Q.mean()
        actor_loss = -Q.mean() - 0.01 * dist.entropy().sum(dim=-1).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    # supports with and without precached embeddings
    def aug_and_encode(self, obs):
        if self.obs_precached:
            return obs  # already precached embedding, skip encoder entirely
        if self.update_encoder:
            obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
