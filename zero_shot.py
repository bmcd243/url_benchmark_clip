import os
import hydra
import torch
import numpy as np
import clip
from PIL import Image
from pathlib import Path
import warnings

import dmc
import utils
from video import VideoRecorder

warnings.filterwarnings('ignore')

class ZeroShotEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.work_dir = Path.cwd()
        
        # 1. Setup Environment
        print(f"Loading Environment: {cfg.task}...")
        self.env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed)
        
        # 2. Setup Agent
        print(f"Loading Agent: {cfg.agent.name}...")
        self.agent = hydra.utils.instantiate(
            cfg.agent, 
            obs_type=cfg.obs_type,
            obs_shape=self.env.observation_spec().shape,
            action_shape=self.env.action_spec().shape,
            num_expl_steps=0
        )
        
        # 3. Load Snapshot
        snapshot_dir = Path(cfg.snapshot_base_dir) / cfg.obs_type / cfg.task.split('_')[0] / cfg.agent.name / str(cfg.seed)
        snapshot_path = snapshot_dir / f'snapshot_{cfg.snapshot_ts}.pt'
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found at {snapshot_path}")
            
        with snapshot_path.open('rb') as f:
            payload = torch.load(f)
        self.agent.init_from(payload['agent'])
        self.agent.train(False) # Set to eval mode
        
        # 4. Load CLIP
        print("Loading CLIP (ViT-B/32)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.video_recorder = VideoRecorder(self.work_dir)

    def generate_candidate_skills(self):
        """Generates candidate skills depending on if the agent is discrete (DIAYN) or continuous (APS)"""
        candidates = []
        if 'diayn' in self.cfg.agent.name:
            # DIAYN: Test all discrete skills (one-hot vectors)
            for i in range(self.agent.skill_dim):
                skill = np.zeros(self.agent.skill_dim, dtype=np.float32)
                skill[i] = 1.0
                candidates.append(skill)
        elif 'aps' in self.cfg.agent.name:
            # APS: Sample random continuous vectors on the unit sphere
            for _ in range(self.cfg.num_candidates):
                task = np.random.randn(self.agent.sf_dim).astype(np.float32)
                task = task / np.linalg.norm(task)
                candidates.append(task)
        else:
            raise NotImplementedError("Zero shot only implemented for DIAYN and APS")
        return candidates

    @torch.no_grad()
    def evaluate_skill(self, skill, text_features):
        """Rolls out a skill for N steps and scores the rendered images against the text prompt"""
        # Format meta dict based on agent type
        meta_key = 'skill' if 'diayn' in self.cfg.agent.name else 'task'
        meta = {meta_key: skill}
        
        time_step = self.env.reset()
        total_similarity = 0.0
        steps_taken = 0
        
        for _ in range(self.cfg.eval_steps):
            if time_step.last():
                break
                
            # Get action
            action = self.agent.act(time_step.observation, meta, 0, eval_mode=True)
            time_step = self.env.step(action)
            
            # Render a clean image directly from the physics engine for CLIP
            # Camera 0 is side view, Camera 2 is specific to quadruped
            cam_id = 2 if 'quadruped' in self.cfg.task else 0
            img_array = self.env.physics.render(height=224, width=224, camera_id=cam_id)
            
            # Preprocess for CLIP
            pil_image = Image.fromarray(img_array)
            img_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get CLIP image features and score
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
        
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            # 1. Encode Text
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 2. Get Candidates
            candidates = self.generate_candidate_skills()
            best_skill = None
            best_score = -float('inf')
            
            print(f"Testing {len(candidates)} candidate skills in the background...")
            
            # 3. Test Candidates
            for i, skill in enumerate(candidates):
                score = self.evaluate_skill(skill, text_features)
                if score > best_score:
                    best_score = score
                    best_skill = skill
            
            print(f"--> Found best skill with alignment score: {best_score:.4f}")
            
            # 4. Record a full video of the winning skill
            print(f"Recording video of the best skill...")
            meta_key = 'skill' if 'diayn' in self.cfg.agent.name else 'task'
            meta = {meta_key: best_skill}
            
            time_step = self.env.reset()
            self.video_recorder.init(self.env, enabled=True)
            
            steps = 0
            while not time_step.last() and steps < 250: # Max 250 frames for the video
                action = self.agent.act(time_step.observation, meta, 0, eval_mode=True)
                time_step = self.env.step(action)
                self.video_recorder.record(self.env)
                steps += 1
                
            vid_name = f"{prompt.replace(' ', '_')}.mp4"
            self.video_recorder.save(vid_name)
            print(f"Saved to: {self.work_dir}/eval_video/{vid_name}")


@hydra.main(config_path='.', config_name='zero_shot')
def main(cfg):
    evaluator = ZeroShotEvaluator(cfg)
    evaluator.interact()

if __name__ == '__main__':
    main()