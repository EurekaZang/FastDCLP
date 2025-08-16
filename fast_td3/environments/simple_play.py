#!/usr/bin/env python3
"""
Simple script to quickly test a trained FastTD3 model in IsaacLab with GUI.

Usage:
    python simple_play.py path/to/model.pt Isaac-Velocity-Flat-H1-v0
"""

import sys
import torch
import os

# Add FastTD3 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_td3.fast_td3 import Actor
from fast_td3.fast_td3_utils import EmpiricalNormalization
from fast_td3.environments.isaaclab_env import IsaacLabEnv


def simple_play(model_path: str, task_name: str, num_episodes: int = 3):
    """Simple function to play trained model in IsaacLab."""
    
    print(f"Loading model: {model_path}")
    print(f"Task: {task_name}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create actor (adjust parameters as needed)
    actor = Actor(
        n_obs=checkpoint.get('args', {}).get('n_obs', 1090),
        n_act=checkpoint.get('args', {}).get('n_act', 2),
        num_envs=1,
        init_scale=0.1,
        hidden_dim=checkpoint.get('args', {}).get('actor_hidden_dim', 512),
        device=device
    )
    
    # Load actor weights
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # Load normalizer if available
    obs_normalizer = None
    if 'obs_normalizer_state' in checkpoint:
        obs_normalizer = EmpiricalNormalization(
            shape=checkpoint.get('args', {}).get('n_obs', 1090), 
            device=device
        )
        obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
        obs_normalizer.eval()
        print("Loaded observation normalizer")
    
    # Create environment with GUI
    print("Creating IsaacLab environment...")
    env = IsaacLabEnv(
        task_name=task_name,
        device=device,
        num_envs=1,
        seed=42,
        action_bounds=1.0,
        render_mode="human",  # Enable GUI
        headless=False        # Show GUI
    )
    
    print(f"Environment created: obs={env.num_obs}, act={env.num_actions}")
    print("Starting episodes...")
    
    # Play episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        obs = env.reset(random_start_init=False)
        episode_reward = 0
        step = 0
        
        with torch.no_grad():
            for step in range(env.max_episode_steps):
                # Normalize observation
                if obs_normalizer is not None:
                    norm_obs = obs_normalizer(obs, update=False)
                else:
                    norm_obs = obs
                
                # Get action
                action = actor.explore(norm_obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward.sum().item()
                
                # Render
                try:
                    env.render()
                except:
                    pass  # Ignore render errors
                
                if step % 50 == 0:
                    print(f"Step {step}: reward={episode_reward:.2f}")
                
                if done.any():
                    break
        
        print(f"Episode {episode + 1} finished: {step} steps, reward={episode_reward:.2f}")
    
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_play.py <model_path> <task_name> [num_episodes]")
        print("Example: python simple_play.py models/model.pt Isaac-Velocity-Flat-H1-v0 3")
        sys.exit(1)
    
    model_path = sys.argv[1]
    task_name = sys.argv[2]
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    simple_play(model_path, task_name, num_episodes)
