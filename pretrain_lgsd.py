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
from video import VideoRecorder

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

        if cfg.use_wandb:
            try:
                full_cfg = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True)
                wandb.config.update(full_cfg, allow_val_change=True)
            except omegaconf.errors.MissingMandatoryValue:
                # Ignore unresolved Hydra placeholders and still log available keys.
                partial_cfg = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=False)
                wandb.config.update(partial_cfg, allow_val_change=True)

        # self.train_video_recorder = TrainVideoRecorder(
        #     self.work_dir if cfg.save_train_video else None,
        #     camera_id=0 if 'quadruped' not in cfg.domain else 2,
        #     use_wandb=self.cfg.use_wandb
        # )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._saved_snapshots = set()

    
    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
        while eval_until_episode(episode):
            # Sample a fixed skill for the duration of the eval episode
            current_z = np.random.randn(self.agent.skill_dim).astype(np.float32)
            time_step = self.eval_env.reset()
            current_emb = self._encode_obs(time_step.observation)
            
            self.video_recorder.init(self.eval_env, enabled=True)
            
            while not time_step.last():
                with torch.no_grad():
                    # Deterministic action prediction from the SB3 PPO Policy
                    policy_input = np.concatenate([current_emb, current_z], axis=-1)
                    policy_input_ts = torch.as_tensor(policy_input).float().to(self.device).unsqueeze(0)
                    action = self.agent.ppo.policy.predict(policy_input_ts, deterministic=True)[0][0]
                
                time_step = self.eval_env.step(action)
                current_emb = self._encode_obs(time_step.observation)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                
            episode += 1
            # Save the video to wandb under a unique skill identifier
            self.video_recorder.save(f'{self._global_step * self.cfg.action_repeat}_skill{episode}.mp4', 
                                     wandb_key=f'eval/video_skill{episode}')
            
        with self.logger.log_and_dump_ctx(self._global_step * self.cfg.action_repeat, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self._global_episode)
            log('step', self._global_step)
    
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

    def save_snapshot(self, frame_tag=None):
        frame = self._global_step * self.cfg.action_repeat
        tag = frame if frame_tag is None else frame_tag
        
        # Mirror pretrain.py snapshot_dir logic
        snapshot_dir = Path(self.cfg.snapshot_dir.replace(
            '${obs_type}', self.cfg.obs_type
        ).replace(
            '${domain}', self.cfg.domain
        ).replace(
            '${agent.name}', self.cfg.agent.name
        ).replace(
            '${seed}', str(self.cfg.seed)
        ))
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        run_id = f"{self.work_dir.parent.name}_{self.work_dir.name}"
        snapshot = snapshot_dir / f'snapshot_{run_id}_{tag}.pt'
        
        payload = {
            'agent': self.agent,
            '_global_step': self._global_step,
            '_global_episode': self._global_episode
        }
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        if self.cfg.use_wandb:
            wandb.save(str(snapshot), base_path=str(snapshot_dir), policy='now')
        print(f"[pretrain_lgsd] Snapshot saved: {snapshot}")

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        # eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        
        time_step = self.train_env.reset()
        current_emb = self._encode_obs(time_step.observation)
        current_z = np.random.randn(self.agent.skill_dim).astype(np.float32)
        
        episode_step, episode_reward = 0, 0
        # self.train_video_recorder.init(self.train_env)

        # PPO On-Policy Loop
        while train_until_step(self._global_step):
            # if eval_every_step(self._global_step):
            #     self.logger.log('eval_total_time', self.timer.total_time(), self._global_step * self.cfg.action_repeat)
            #     self.eval()
            # 1. Collect a rollout of length n_steps
            for step in range(self.agent.n_steps):
                action, value, log_prob, ppo_input = self.agent.get_action_and_value(current_emb, current_z)
                
                time_step = self.train_env.step(action)
                next_emb = self._encode_obs(time_step.observation)
                
                # LGSD specific calculations
                r_int = self.agent.compute_intrinsic_reward(current_emb, next_emb, current_z)
                # L2 Normalize the embeddings
                curr_norm = current_emb / (np.linalg.norm(current_emb) + 1e-8)
                next_norm = next_emb / (np.linalg.norm(next_emb) + 1e-8)
                # Compute Cosine Distance: 1 - Cosine Similarity
                d_lang_val = 1.0 - np.dot(curr_norm, next_norm)
                
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
                    d_lang=np.array([d_lang_val], dtype=np.float32),
                    raw_obs=np.expand_dims(current_emb, 0)
                )

                episode_reward += time_step.reward
                episode_step += 1
                self._global_step += 1
                # self.train_video_recorder.record(self.train_env)

                if time_step.last():
                    self._global_episode += 1
                    # self.train_video_recorder.save(f'{self._global_step * self.cfg.action_repeat}.mp4')
                    
                    with self.logger.log_and_dump_ctx(self._global_step * self.cfg.action_repeat, ty='train') as log:
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_step * self.cfg.action_repeat)
                        log('episode', self._global_episode)
                        log('step', self._global_step)
                    
                    time_step = self.train_env.reset()
                    next_emb = self._encode_obs(time_step.observation)
                    current_z = np.random.randn(self.agent.skill_dim).astype(np.float32)
                    episode_step, episode_reward = 0, 0
                    # self.train_video_recorder.init(self.train_env)

                current_emb = next_emb

            # 2. Update PPO and LGSD Networks (Phase A & B)
            metrics = self.agent.update(last_obs=current_emb, last_z=current_z, last_done=time_step.last())
            prev_frame = (self._global_step - self.agent.n_steps) * self.cfg.action_repeat
            curr_frame = self._global_step * self.cfg.action_repeat
            for snapshot_frame in self.cfg.snapshots:
                if prev_frame < snapshot_frame <= curr_frame and snapshot_frame not in self._saved_snapshots:
                    self.save_snapshot(frame_tag=snapshot_frame)
                    self._saved_snapshots.add(snapshot_frame)
                
            self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')

        final_frame = self._global_step * self.cfg.action_repeat
        if final_frame not in self._saved_snapshots:
            self.save_snapshot(frame_tag=final_frame)

@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    workspace = LGSDWorkspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()