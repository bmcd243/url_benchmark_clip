
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import hydra
import torch
import numpy as np
import clip
from PIL import Image
from pathlib import Path
import warnings
import wandb

import dmc
import utils
from video import VideoRecorder

warnings.filterwarnings('ignore')

class ZeroShotEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.work_dir = Path.cwd()
        
        # --- 1. W&B Integration ---
        if getattr(cfg, 'use_wandb', False):
            exp_name = '_'.join([
                getattr(cfg, 'experiment', 'zero_shot'), 
                cfg.agent.name, 
                cfg.task, 
                cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)
        
        # --- 2. Setup Environment ---
        print(f"Loading Environment: {cfg.task}...")
        self.env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed)
        
        # --- 3. Setup Agent ---
        print(f"Loading Agent: {cfg.agent.name}...")
        self.agent = hydra.utils.instantiate(
            cfg.agent, 
            obs_type=cfg.obs_type,
            obs_shape=self.env.observation_spec().shape,
            action_shape=self.env.action_spec().shape,
            num_expl_steps=0
        )
            
        payload = self._load_snapshot()
        self.agent.init_from(payload['agent'])
        self.agent.train(False)
        
        # --- 5. Load CLIP & Video Recorder ---
        print("Loading CLIP (ViT-B/32)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        cam_id = 2 if 'quadruped' in cfg.task else 0
        # We set use_wandb=False here to prevent the default 'eval/video' logging, 
        # allowing us to do custom prompt-based logging later.
        self.video_recorder = VideoRecorder(self.work_dir, camera_id=cam_id, use_wandb=False)

    def _load_snapshot(self):
        if self.cfg.get('snapshot_path', None) is not None:
            snapshot = Path(self.cfg.snapshot_path)
            if not snapshot.exists():
                raise FileNotFoundError(f"snapshot_path not found: {snapshot}")
            
            try:
                frame_count = int(snapshot.stem.split('_')[-1])
                print(f"[zero_shot] Loading snapshot:")
                print(f"  path  : {snapshot}")
                print(f"  frames: {frame_count:,}")
                print(f"  agent : {self.cfg.agent.name}")
                print(f"  task  : {self.cfg.task}")
                print(f"  seed  : {self.cfg.seed}")
            except (ValueError, IndexError):
                print(f"[zero_shot] Loading snapshot: {snapshot}")

            with snapshot.open('rb') as f:
                return torch.load(f, weights_only=False, map_location=self.device)

        # fallback: directory-based lookup using snapshot_ts
        domain = self.cfg.task.split('_')[0]
        snapshot_dir = (Path(self.cfg.snapshot_base_dir) / self.cfg.obs_type / 
                        domain / self.cfg.agent.name / str(self.cfg.seed))
        snapshot_path = snapshot_dir / f'snapshot_{self.cfg.snapshot_ts}.pt'
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found at {snapshot_path}")
        
        print(f"[zero_shot] Loading snapshot:")
        print(f"  path  : {snapshot_path}")
        print(f"  frames: {self.cfg.snapshot_ts:,}")
        print(f"  agent : {self.cfg.agent.name}")
        print(f"  task  : {self.cfg.task}")
        print(f"  seed  : {self.cfg.seed}")

        with snapshot_path.open('rb') as f:
            return torch.load(f, weights_only=False, map_location=self.device) 
    
    
    def generate_candidate_skills(self):
        candidates = []
        if 'diayn' in self.cfg.agent.name:
            for i in range(self.agent.skill_dim):
                skill = np.zeros(self.agent.skill_dim, dtype=np.float32)
                skill[i] = 1.0
                candidates.append(skill)
        elif 'aps' in self.cfg.agent.name:
            for _ in range(self.cfg.num_candidates):
                task = np.random.randn(self.agent.sf_dim).astype(np.float32)
                task = task / np.linalg.norm(task)
                candidates.append(task)
        else:
            raise NotImplementedError("Zero shot only implemented for DIAYN and APS")
        return candidates

    @torch.no_grad()
    def evaluate_skill(self, skill, text_features):
        meta_key = 'skill' if 'diayn' in self.cfg.agent.name else 'task'
        meta = {meta_key: skill}
        
        time_step = self.env.reset()
        total_similarity = 0.0
        steps_taken = 0
        
        cam_id = 2 if 'quadruped' in self.cfg.task else 0
        
        for _ in range(self.cfg.eval_steps):
            if time_step.last():
                break
                
            obs = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0).float()
            with torch.no_grad():
                emb = self.agent.encoder(obs).squeeze(0).cpu().numpy()
            action = self.agent.act(emb, meta, 0, eval_mode=True, obs_already_encoded=True)
            time_step = self.env.step(action)
            
            img_array = self.env.physics.render(height=224, width=224, camera_id=cam_id)
            pil_image = Image.fromarray(img_array)
            img_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            img_features = self.clip_model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            similarity = (img_features * text_features).sum(dim=-1).item()
            total_similarity += similarity
            steps_taken += 1
            
        return total_similarity / max(1, steps_taken)

    @torch.no_grad()
    def interact(self):
        print("\n" + "="*50)
        print("Ready! Enter a prompt to find a matching skill.")
        print("Type 'quit' to exit.")
        print("="*50)
        
        step_idx = 0
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            candidates = self.generate_candidate_skills()
            best_skill = None
            best_score = -float('inf')
            
            print(f"Testing {len(candidates)} candidate skills in the background...")
            
            for i, skill in enumerate(candidates):
                score = self.evaluate_skill(skill, text_features)
                if score > best_score:
                    best_score = score
                    best_skill = skill
            
            print(f"--> Found best skill with alignment score: {best_score:.4f}")
            
            print(f"Recording video of the best skill...")
            meta_key = 'skill' if 'diayn' in self.cfg.agent.name else 'task'
            meta = {meta_key: best_skill}
            
            time_step = self.env.reset()
            self.video_recorder.init(self.env, enabled=True)
            
            steps = 0
            while not time_step.last() and steps < 250:
                obs = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0).float()
                with torch.no_grad():
                    emb = self.agent.encoder(obs).squeeze(0).cpu().numpy()
                action = self.agent.act(emb, meta, 0, eval_mode=True, obs_already_encoded=True)
                time_step = self.env.step(action)
                self.video_recorder.record(self.env)
                steps += 1
                
            if getattr(self.cfg, 'use_wandb', False):
                print(f"Uploading video to W&B...")
                # Extract frames directly from memory
                frames = np.transpose(np.array(self.video_recorder.frames), (0, 3, 1, 2))
                safe_prompt_name = prompt.replace(" ", "_")
                
                wandb.log({
                    f"zero_shot_videos/{safe_prompt_name}": wandb.Video(frames[::8, :, ::2, ::2], fps=6, format="gif"),
                    "zero_shot/alignment_score": best_score,
                    "zero_shot/prompt": prompt
                }, step=step_idx)
            else:
                print("W&B is disabled. Video was not saved locally or remotely.")
                
            # Clear the frames from memory to prevent RAM leaks
            self.video_recorder.frames = []
            
            step_idx += 1

@hydra.main(config_path='.', config_name='zero_shot')
def main(cfg):
    evaluator = ZeroShotEvaluator(cfg)
    evaluator.interact()

if __name__ == '__main__':
    main()