import faulthandler
faulthandler.enable()

import warnings
import copy

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
import omegaconf
from dm_env import specs


import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            encoder_type = getattr(cfg.agent, 'encoder_type', 'cnn')
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                encoder_type, str(cfg.seed)
            ])
            tags = [cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type, encoder_type]
            # Do NOT pass config= here: agent.obs_type / obs_shape / action_shape
            # are still '???' at this point and will cause OmegaConf to raise
            # MissingMandatoryValue. The important fields are logged below via
            # wandb.config.update after make_agent() fills them in.
            wandb.init(
                project="urlb",
                group=cfg.agent.name,
                name=exp_name,
                tags=tags,
            )

        snapshot_dir = Path(self.cfg.snapshot_dir.replace(
            '${obs_type}', cfg.obs_type
        ).replace(
            '${domain}', cfg.domain
        ).replace(
            '${agent.name}', cfg.agent.name
        ).replace(
            '${seed}', str(cfg.seed)
        ))

        if cfg.use_wandb:
            encoder_type = getattr(cfg.agent, 'encoder_type', 'cnn')
            wandb.config.update({
                'snapshot_dir':    str(snapshot_dir.resolve()),
                'encoder_type':    encoder_type,
                'agent_name':      cfg.agent.name,
                'domain':          cfg.domain,
                'obs_type':        cfg.obs_type,
                'seed':            cfg.seed,
                'run_id':          wandb.run.id
            })
        
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        encoder_type = getattr(cfg.agent, 'encoder_type', 'cnn')
        # pixel_size = 224 if (cfg.obs_type == 'pixels' and encoder_type == 'clip') else 84
        pixel_size = 84
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                cfg.action_repeat, cfg.seed, pixel_size=pixel_size)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                cfg.action_repeat, cfg.seed, pixel_size=pixel_size)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # Now that make_agent() has filled in obs_type/obs_shape/action_shape,
        # log the full resolved config to wandb.
        if cfg.use_wandb:
            try:
                full_cfg = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True)
                wandb.config.update(full_cfg, allow_val_change=True)
            except omegaconf.errors.MissingMandatoryValue:
                pass  # any remaining ??? fields are intentionally unset

        encoder_type = getattr(cfg.agent, 'encoder_type', 'cnn')
        self._clip_precache = (cfg.obs_type == 'pixels' and encoder_type == 'clip')

        if self._clip_precache:
            # Override the observation spec to store 512-d embeddings
            obs_spec = specs.Array((self.agent.encoder.repr_dim,), np.float32, 'observation')
        else:
            obs_spec = self.train_env.observation_spec()

        # create replay buffer  (replace the existing data_specs block)
        data_specs = (obs_spec,
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
        self.work_dir if cfg.save_video else None,
        camera_id=0 if 'quadruped' not in cfg.domain else 2,
        use_wandb=self.cfg.use_wandb)

        self.train_video_recorder = TrainVideoRecorder(
        self.work_dir if cfg.save_train_video else None,
        camera_id=0 if 'quadruped' not in cfg.domain else 2,
        use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _encode_obs(self, obs):
        """Encode a single obs with CLIP if precaching, otherwise return as-is."""
        if not self._clip_precache:
            return obs
        with torch.no_grad():
            t = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0).float()
            emb = self.agent.encoder(t)
        return emb.squeeze(0).cpu().numpy()

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            meta = self.agent.init_meta()
            skill_idx = np.argmax(meta['skill']) if 'skill' in meta else episode
            time_step = self.eval_env.reset()
            current_emb = self._encode_obs(time_step.observation)  # encode once
            self.video_recorder.init(self.eval_env, enabled=True)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(current_emb, meta, self.global_step,
                                            eval_mode=True,
                                            obs_already_encoded=self._clip_precache)
                time_step = self.eval_env.step(action)
                current_emb = self._encode_obs(time_step.observation)  # encode next, reuse next iter
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            episode += 1
            self.video_recorder.save(
                f'{self.global_frame}_skill{skill_idx}.mp4',
                wandb_key=f'eval/video_skill{skill_idx}'
            )
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        current_emb = self._encode_obs(time_step.observation)  # encode once at reset
        self.replay_storage.add(time_step._replace(observation=current_emb), meta)
        self.train_video_recorder.init(self.train_env)
        metrics = None

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                current_emb = self._encode_obs(time_step.observation)  # encode once at reset
                self.replay_storage.add(time_step._replace(observation=current_emb), meta)
                self.train_video_recorder.init(self.train_env)
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            # Act using already-encoded obs — no CLIP call here
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(current_emb, meta, self.global_step,
                                        eval_mode=False,
                                        obs_already_encoded=self._clip_precache)

            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            time_step = self.train_env.step(action)
            episode_reward += time_step.reward

            # Encode next obs exactly once, then carry forward as current for next step
            next_emb = self._encode_obs(time_step.observation)
            self.replay_storage.add(time_step._replace(observation=next_emb), meta)
            self.train_video_recorder.record(self.train_env)
            current_emb = next_emb

            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        # Build a separate dict for saving without mutating the live agent
        if hasattr(self.agent, 'encoder') and hasattr(self.agent.encoder, 'model'):
            save_agent = copy.copy(self.agent)  # shallow copy
            del save_agent.encoder
            payload['agent'] = save_agent
            payload['encoder_type'] = 'clip'

        with snapshot.open('wb') as f:
            torch.save(payload, f)

        if self.cfg.use_wandb:
            wandb.log({
                'snapshot/frame':    self.global_frame,
                'snapshot/path':     str(snapshot.resolve()),
            }, step=self.global_frame)
        
        print(f"[pretrain] Snapshot saved: {snapshot}")

    def _maybe_encode_obs(self, obs):
        """Encode obs with CLIP before storing in replay buffer."""
        if not self._clip_precache:
            return obs
        with torch.no_grad():
            t = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0).float()
            emb = self.agent.encoder(t)
        return emb.squeeze(0).cpu().numpy()


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()