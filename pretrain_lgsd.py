import os
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
import wandb
import omegaconf

import dmc
import utils
from logger import Logger
from video import TrainVideoRecorder

torch.backends.cudnn.benchmark = True
from dmc_benchmark import PRIMAL_TASKS

class LGSDWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Logging setup
        if cfg.use_wandb:
            exp_name = '_'.join([cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type, str(cfg.seed)])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # Env setup
        task = PRIMAL_TASKS[self.cfg.domain]
        pixel_size = 84
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, pixel_size=pixel_size)

        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, pixel_size=pixel_size)
        from video import VideoRecorder # Make sure this is imported at the top!
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in cfg.domain else 2,
            use_wandb=self.cfg.use_wandb
        )

        # Agent setup
        cfg.agent.obs_type = cfg.obs_type
        if cfg.obs_type == 'pixels':
            # 512 dim for each frame in the stack
            obs_shape = (512 * cfg.frame_stack,)
        else:
            obs_shape = self.train_env.observation_spec()['observations'].shape
            
        cfg.agent.obs_shape = obs_shape
        cfg.agent.action_shape = self.train_env.action_spec().shape
        cfg.agent.num_expl_steps = cfg.num_seed_frames // cfg.action_repeat

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in cfg.domain else 2,
            use_wandb=self.cfg.use_wandb
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def _encode_obs(self, obs):
        if self.cfg.obs_type != 'pixels':
            return obs
        with torch.no_grad():
            # obs shape is (9, 84, 84)
            t = torch.as_tensor(obs, device=self.device).float()
            
            # Split the 9 channels back into (3 frames, 3 channels, 84, 84)
            t = t.view(-1, 3, t.shape[1], t.shape[2])
            
            # Resize to CLIP's expected 224x224 resolution
            t = torch.nn.functional.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Encode all frames simultaneously: output is (3, 512)
            emb = self.agent.encoder.encode_image(t) 
            
            # Flatten to a single 1D array of shape (1536,)
            emb = emb.view(-1).cpu().numpy()
            
        return emb

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        
        time_step = self.train_env.reset()
        current_emb = self._encode_obs(time_step.observation)
        current_z = np.random.randn(self.agent.skill_dim).astype(np.float32)
        
        episode_step, episode_reward = 0, 0
        self.train_video_recorder.init(self.train_env)

        # PPO On-Policy Loop
        while train_until_step(self._global_step):
            # 1. Collect a rollout of length n_steps
            for step in range(self.agent.n_steps):
                action, value, log_prob, ppo_input = self.agent.get_action_and_value(current_emb, current_z)
                
                time_step = self.train_env.step(action)
                next_emb = self._encode_obs(time_step.observation)
                
                # LGSD specific calculations
                r_int = self.agent.compute_intrinsic_reward(current_emb, next_emb, current_z)
                d_lang_val = np.linalg.norm(current_emb - next_emb) # Proxy for semantic distance
                
                # Add to custom buffer
                self.agent.rollout_buffer.add(
                    obs=ppo_input,
                    action=action,
                    reward=np.array([r_int], dtype=np.float32),
                    episode_start=np.array([episode_step == 0], dtype=bool),
                    value=value,
                    log_prob=log_prob,
                    skill=np.expand_dims(current_z, 0),
                    next_obs=np.expand_dims(next_emb, 0),
                    d_lang=np.array([d_lang_val], dtype=np.float32)
                )

                episode_reward += time_step.reward
                episode_step += 1
                self._global_step += 1
                self.train_video_recorder.record(self.train_env)

                if time_step.last():
                    self._global_episode += 1
                    self.train_video_recorder.save(f'{self._global_step * self.cfg.action_repeat}.mp4')
                    
                    with self.logger.log_and_dump_ctx(self._global_step * self.cfg.action_repeat, ty='train') as log:
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_step * self.cfg.action_repeat)
                        log('episode', self._global_episode)
                        log('step', self._global_step)
                    
                    time_step = self.train_env.reset()
                    next_emb = self._encode_obs(time_step.observation)
                    current_z = np.random.randn(self.agent.skill_dim).astype(np.float32)
                    episode_step, episode_reward = 0, 0
                    self.train_video_recorder.init(self.train_env)

                current_emb = next_emb

            # 2. Update PPO and LGSD Networks (Phase A & B)
            metrics = self.agent.update(last_obs=current_emb, last_z=current_z, last_done=time_step.last())
            self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')

@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    workspace = LGSDWorkspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()