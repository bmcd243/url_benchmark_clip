import warnings
import wandb

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import omegaconf
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


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

        encoder_type = getattr(cfg.agent, 'encoder_type', 'cnn')
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment,
                cfg.agent.name,
                cfg.task,
                cfg.obs_type,
                encoder_type,
                f'seed{cfg.seed}'
            ])
            run_config = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=False
            )
            wandb.init(
                project="urlb",
                group=f"{cfg.agent.name}_{cfg.task}",
                name=exp_name,
                tags=[cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, encoder_type],
                config=run_config
            )

        # create logger
        self.logger = Logger(self.work_dir,
                                use_tb=cfg.use_tb,
                                use_wandb=cfg.use_wandb)
        # create envs

        pixel_size = 224 if (cfg.obs_type == 'pixels' and encoder_type == 'clip') else 84
        
        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                          cfg.action_repeat, cfg.seed, pixel_size=pixel_size)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                cfg.action_repeat, cfg.seed, pixel_size=pixel_size)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

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

        # initialize from pretrained
        # if cfg.snapshot_ts > 0:
        #     pretrained_agent = self.load_snapshot()['agent']
        #     self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        # Decide whether replay stores raw pixels or precomputed CLIP embeddings
        self._clip_precache = (cfg.obs_type == 'pixels' and encoder_type == 'clip')

        if self._clip_precache:
            obs_spec = specs.Array(
                (self.agent.encoder.repr_dim,), np.float32, 'observation')
        else:
            obs_spec = self.train_env.observation_spec()

        # create replay buffer specs
        data_specs = (
            obs_spec,
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
        )

        # create data storage
        self.replay_storage = ReplayBufferStorage(
            data_specs, meta_specs, self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.replay_buffer_size,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
            False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in cfg.task else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in cfg.task else 2,
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
        first_episode = True
        while eval_until_episode(episode):
            meta = self.agent.init_meta()
            time_step = self.eval_env.reset()
            current_emb = self._encode_obs(time_step.observation)
            self.video_recorder.init(self.eval_env, enabled=first_episode)  # only record first
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
                f'{self.global_frame}.mp4',
                wandb_key=f'eval/video_skill'
            )
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        eval_every_step  = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        # ------------------------------------------------------------------ #
        # PROBE PHASE (4096 steps)
        # Collect transitions with extrinsic reward using the pretrained policy,
        # then call regress_meta to select the best skill/task direction.
        # This applies to DIAYN, APS, and LGSD equally.
        # ------------------------------------------------------------------ #
        if hasattr(self.agent, 'regress_meta'):
            print(f"[finetune] Starting probe phase ({self.cfg.num_init_steps} steps)...")
            probe_until = utils.Until(self.cfg.num_init_steps, self.cfg.action_repeat)
            probe_step  = 0

            time_step = self.train_env.reset()
            meta      = self.agent.init_meta()
            current_emb = self._encode_obs(time_step.observation)

            # Store probe transitions — extrinsic reward is available here
            # (reward_free=False during finetuning, so extr_reward is real)
            if self._clip_precache:
                self.replay_storage.add(
                    time_step._replace(observation=current_emb), meta)
            else:
                self.replay_storage.add(time_step, meta)

            while probe_until(probe_step):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        current_emb, meta, probe_step,
                        eval_mode=False,
                        obs_already_encoded=self._clip_precache)

                time_step = self.train_env.step(action)
                next_emb  = self._encode_obs(time_step.observation)
                meta      = self.agent.update_meta(meta, probe_step, time_step)

                if self._clip_precache:
                    self.replay_storage.add(
                        time_step._replace(observation=next_emb), meta)
                else:
                    self.replay_storage.add(time_step, meta)

                current_emb = next_emb
                probe_step += 1

                if time_step.last():
                    time_step   = self.train_env.reset()
                    meta        = self.agent.init_meta()
                    current_emb = self._encode_obs(time_step.observation)

            # Select best skill from probe transitions
            # This sets agent.solved_meta, which init_meta() returns from now on
            meta = self.agent.regress_meta(self.replay_iter, 0)
            print(f"[finetune] Probe complete. Skill selected: {meta}")

        # ------------------------------------------------------------------ #
        # FINETUNE PHASE (96k steps)
        # Fixed skill from regress_meta — no further skill updates
        # ------------------------------------------------------------------ #
        episode_step, episode_reward = 0, 0
        time_step   = self.train_env.reset()
        meta        = self.agent.init_meta()   # returns solved_meta if set
        current_emb = self._encode_obs(time_step.observation)

        if self._clip_precache:
            self.replay_storage.add(
                time_step._replace(observation=current_emb), meta)
        else:
            self.replay_storage.add(time_step, meta)

        self.train_video_recorder.init(self.train_env)
        metrics = None

        while train_until_step(self._global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps',            episode_frame / elapsed_time)
                        log('total_time',     total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode',        self.global_episode)
                        log('buffer_size',    len(self.replay_storage))
                        log('step',           self.global_step)

                time_step   = self.train_env.reset()
                meta        = self.agent.init_meta()   # still returns solved_meta
                current_emb = self._encode_obs(time_step.observation)

                if self._clip_precache:
                    self.replay_storage.add(
                        time_step._replace(observation=current_emb), meta)
                else:
                    self.replay_storage.add(time_step, meta)

                self.train_video_recorder.init(self.train_env)
                episode_step  = 0
                episode_reward = 0

            if eval_every_step(self._global_step):
                self.logger.log('eval_total_time',
                                self.timer.total_time(), self.global_frame)
                self.eval()

            # Skill is fixed — no update_meta during finetune phase
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    current_emb, meta, self._global_step,
                    eval_mode=False,
                    obs_already_encoded=self._clip_precache)

            metrics = self.agent.update(self.replay_iter, self._global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty='train')

            time_step      = self.train_env.step(action)
            episode_reward += time_step.reward
            next_emb       = self._encode_obs(time_step.observation)

            if self._clip_precache:
                self.replay_storage.add(
                    time_step._replace(observation=next_emb), meta)
            else:
                self.replay_storage.add(time_step, meta)

            self.train_video_recorder.record(self.train_env)
            current_emb    = next_emb
            episode_step   += 1
            self._global_step += 1

    def load_snapshot(self):
        if self.cfg.get('snapshot_path', None) is not None:
            snapshot = Path(self.cfg.snapshot_path)
            if not snapshot.exists():
                raise FileNotFoundError(f"snapshot_path not found: {snapshot}")
            with snapshot.open('rb') as f:
                return torch.load(f, weights_only=False, map_location=self.device)

        # fallback: directory-based lookup using snapshot_ts
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = (snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name)

        def try_load(seed):
            snapshot = snapshot_dir / str(seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                return torch.load(f, weights_only=False, map_location=self.device)

        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload

    def save_snapshot(self):
        snapshot_dir = self.work_dir / 'snapshots'
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
