#!/usr/bin/env python3
"""
Simple script to quickly test a trained DCLP model in IsaacLab with GUI.

Usage:
    python simple_play_dclp.py path/to/model.pt Isaac-Navigation-Flat-Turtlebot2-v0
"""

import sys
import torch
import os

# Add FastTD3 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_td3.fast_td3_utils import EmpiricalNormalization
from fast_td3.environments.isaaclab_env import IsaacLabEnv
from fast_td3.dclp import DCLP, MLPActorCritic


def simple_play_dclp(model_path: str, task_name: str, num_episodes: int = 3):
    """Simple function to play trained DCLP model in IsaacLab."""
    
    print(f"Loading DCLP model: {model_path}")
    print(f"Task: {task_name}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    args_dict = checkpoint.get('args', {})
    state_dim = args_dict.get('n_obs', 280)  # Default for DCLP LiDAR setup
    action_dim = args_dict.get('n_act', 2)   # Default for navigation
    
    print(f"Model dimensions - State: {state_dim}, Action: {action_dim}")
    
    # Create DCLP model
    dclp = DCLP(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args_dict.get('actor_learning_rate', 3e-4),
        gamma=args_dict.get('gamma', 0.99),
        tau=args_dict.get('tau', 0.005),
        alpha=args_dict.get('alpha', 0.2),
        hidden_sizes=tuple(args_dict.get('actor_hidden_dim', [128, 128, 128, 128])),
        device=device
    )
    
    # Load model weights
    try:
        # Check if this is a DCLP checkpoint with the expected format
        if 'actor_critic' in checkpoint:
            # Use DCLP's load method
            dclp.load(model_path)
            print("Loaded DCLP model using native load method")
        else:
            # Try to find actor_critic state dict in training checkpoint
            model_state_dict = checkpoint.get('model_state_dict', 
                                            checkpoint.get('dclp_state_dict', 
                                                         checkpoint.get('actor_critic_state_dict')))
            if model_state_dict:
                dclp.actor_critic.load_state_dict(model_state_dict)
                print("Loaded DCLP model from training checkpoint")
            else:
                print("Warning: Could not find model state dict, using initialized weights")
    except Exception as e:
        print(f"Warning: Error loading model weights: {e}")
    
    dclp.actor_critic.eval()
    
    # Load normalizer if available
    obs_normalizer = None
    if 'obs_normalizer_state' in checkpoint:
        obs_normalizer = EmpiricalNormalization(
            shape=state_dim, 
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
                
                # Get action from DCLP
                action = dclp.get_action(norm_obs, deterministic=True)
                
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
        print("Usage: python simple_play_dclp.py <model_path> <task_name> [num_episodes]")
        print("Example: python simple_play_dclp.py models/model.pt Isaac-Navigation-Flat-Turtlebot2-v0 3")
        sys.exit(1)
    
    model_path = sys.argv[1]
    task_name = sys.argv[2]
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    simple_play_dclp(model_path, task_name, num_episodes)
