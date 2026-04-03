# python finetune_lgsd.py agent=lgsd_clip task=texturedcheetah_run obs_type=pixels snapshot_base_dir=/path/to/your/saved/pretrain_snapshot

import os
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
import wandb

import dmc
import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

class FinetuneLGSDWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Logging setup
        if cfg.use_wandb:
            exp_name = '_'.join(['finetune', cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, str(cfg.seed)])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # Env setup (Finetuning uses cfg.task, e.g., 'cheetah_run')
        pixel_size = 84
        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, pixel_size=pixel_size)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, pixel_size=pixel_size)

        # Agent setup
        cfg.agent.obs_type = cfg.obs_type
        obs_shape = (512 * cfg.frame_stack,) if cfg.obs_type == 'pixels' else self.train_env.observation_spec()['observations'].shape
        cfg.agent.obs_shape = obs_shape
        cfg.agent.action_shape = self.train_env.action_spec().shape
        cfg.agent.num_expl_steps = 0 # Not used for finetuning

        self.agent = hydra.utils.instantiate(cfg.agent)

        # Load Pretrained Snapshot
        self._load_snapshot(cfg)

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in cfg.task else 2,
            use_wandb=self.cfg.use_wandb
        )
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in cfg.task else 2,
            use_wandb=self.cfg.use_wandb
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def _load_snapshot(self, cfg):
        """Finds and loads the PPO policy weights from the pretraining phase."""
        try:
            snapshot_dir = Path(cfg.snapshot_base_dir)
            snapshot_path = None
            
            # Recursively find the snapshot.pt file
            for path in snapshot_dir.rglob('*.pt'):
                snapshot_path = path
                break
                
            if snapshot_path and snapshot_path.exists():
                print(f"Loading pretrained agent from: {snapshot_path}")
                payload = torch.load(snapshot_path, map_location=self.device)
                
                # Extract and load the SB3 PPO state dict
                if 'agent' in payload:
                    try:
                        self.agent.ppo.policy.load_state_dict(payload['agent'].ppo.policy.state_dict())
                        print("✅ Pretrained weights loaded successfully.")
                    except Exception as e:
                        print(f"⚠️ Could not load exact state dict: {e}")
            else:
                print("⚠️ WARNING: No pretrained snapshot found. Training from scratch.")
        except Exception as e:
            print(f"⚠️ Snapshot loading bypassed: {e}")

    def _encode_obs(self, obs):
        if self.cfg.obs_type != 'pixels':
            return obs
        with torch.no_grad():
            t = torch.as_tensor(obs, device=self.device).float()
            t = t.view(-1, 3, t.shape[1], t.shape[2])
            t = torch.nn.functional.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
            emb = self.agent.encoder.encode_image(t) 
            emb = emb.view(-1).cpu().numpy()
        return emb

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
        while eval_until_episode(episode):
            # Use a zeroed skill vector during finetuning to provide a consistent baseline
            current_z = np.zeros(self.agent.skill_dim, dtype=np.float32) 
            time_step = self.eval_env.reset()
            current_emb = self._encode_obs(time_step.observation)
            self.video_recorder.init(self.eval_env, enabled=True)
            
            while not time_step.last():
                with torch.no_grad():
                    policy_input = self._get_policy_input(current_emb, current_z)
                    action = self.agent.ppo.policy.predict(policy_input, deterministic=True)[0]
                
                time_step = self.eval_env.step(action)
                current_emb = self._encode_obs(time_step.observation)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                
            episode += 1
            self.video_recorder.save(f'eval_{self._global_step * self.cfg.action_repeat}_{episode}.mp4', 
                                     wandb_key=f'eval/video_{episode}')
            
        with self.logger.log_and_dump_ctx(self._global_step * self.cfg.action_repeat, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self._global_episode)
            log('step', self._global_step)

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        
        time_step = self.train_env.reset()
        current_emb = self._encode_obs(time_step.observation)
        
        # Zero out the skill vector so the network learns to rely purely on the visual state 
        # to maximize the downstream task reward.
        current_z = np.zeros(self.agent.skill_dim, dtype=np.float32) 
        
        episode_step, episode_reward = 0, 0
        self.train_video_recorder.init(self.train_env)

        while train_until_step(self._global_step):
            if eval_every_step(self._global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self._global_step * self.cfg.action_repeat)
                self.eval()

            # 1. Collect rollout using EXTRINSIC Rewards
            for step in range(self.agent.n_steps):
                action, value, log_prob, ppo_input = self.agent.get_action_and_value(current_emb, current_z)
                
                time_step = self.train_env.step(action)
                next_emb = self._encode_obs(time_step.observation)
                
                # --- KEY DIFFERENCE: Use the task reward instead of the LGSD intrinsic reward ---
                extrinsic_reward = time_step.reward if time_step.reward is not None else 0.0
                
                self.agent.rollout_buffer.add(
                    obs=ppo_input,
                    action=action,
                    reward=np.array([extrinsic_reward], dtype=np.float32),
                    episode_start=np.array([episode_step == 0], dtype=bool),
                    value=value,
                    log_prob=log_prob,
                    skill=np.expand_dims(current_z, 0),
                    next_obs=np.expand_dims(next_emb, 0),
                    d_lang=np.array([0.0], dtype=np.float32) # Dummy value, unused in finetuning
                )

                episode_reward += extrinsic_reward
                episode_step += 1
                self._global_step += 1
                self.train_video_recorder.record(self.train_env)

                if time_step.last():
                    self._global_episode += 1
                    self.train_video_recorder.save(f'train_{self._global_step * self.cfg.action_repeat}.mp4')
                    
                    with self.logger.log_and_dump_ctx(self._global_step * self.cfg.action_repeat, ty='train') as log:
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_step * self.cfg.action_repeat)
                        log('episode', self._global_episode)
                        log('step', self._global_step)
                    
                    time_step = self.train_env.reset()
                    next_emb = self._encode_obs(time_step.observation)
                    episode_step, episode_reward = 0, 0
                    self.train_video_recorder.init(self.train_env)

                current_emb = next_emb

            # 2. Phase B Only: Standard SB3 PPO Updates (Skip LGSD representation updates)
            with torch.no_grad():
                last_input = self._get_policy_input(current_emb, current_z)
                last_input_ts = torch.as_tensor(last_input).float().to(self.device).unsqueeze(0)
                last_value = self.agent.ppo.policy.predict_values(last_input_ts)
            
            self.agent.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([time_step.last()]))

            # We bypass agent.update() completely and just run PPO.train()
            self.agent.ppo.rollout_buffer = self.agent.rollout_buffer
            self.agent.ppo.train()
            
            # Extract SB3 Metrics
            metrics = {}
            if self.agent.ppo.logger is not None:
                for key, val in self.agent.ppo.logger.name_to_value.items():
                    formatted_key = f"train/ppo_{key.replace('/', '_')}"
                    metrics[formatted_key] = val
            
            self.agent.rollout_buffer.reset()
            self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')
    
    def _get_policy_input(self, emb, z):
        emb_ts = torch.as_tensor(emb).float().to(self.device)
        with torch.no_grad():
            projected = self.agent.lgsd_nets.clip_projector(emb_ts)
        return np.concatenate([projected.cpu().numpy(), z], axis=-1)

@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    workspace = FinetuneLGSDWorkspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()