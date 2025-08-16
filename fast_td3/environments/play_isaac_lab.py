#!/usr/bin/env python3
"""
play_isaac_lab.py - Play and visualize trained FastTD3 models in IsaacLab environments

This script loads a trained FastTD3 model and runs it in an IsaacLab environment with rendering enabled.
The IsaacSim GUI will open to show the robot following the trained policy.

Usage:
    python play_isaac_lab.py --model_path models/your_model.pt --task_name Isaac-Velocity-Flat-H1-v0
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from typing import Optional, Dict, Any

# Add the parent directory to sys.path to import fast_td3 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_td3 import Actor, MultiTaskActor, Critic
from fast_td3_simbav2 import Actor as SimbaV2Actor, MultiTaskActor as SimbaV2MultiTaskActor, Critic as SimbaV2Critic
from fast_td3_utils import EmpiricalNormalization
from environments.isaaclab_env import IsaacLabEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play trained FastTD3 model in IsaacLab")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True,
        help="IsaacLab task name (e.g., Isaac-Velocity-Flat-H1-v0)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=5,
        help="Number of episodes to play"
    )
    parser.add_argument(
        "--max_episode_steps", 
        type=int, 
        default=None,
        help="Maximum steps per episode (default: use environment default)"
    )
    parser.add_argument(
        "--deterministic", 
        action="store_true",
        help="Use deterministic policy (no exploration noise)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--action_bounds", 
        type=float, 
        default=1.0,
        help="Action bounds for the environment"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=60,
        help="Target FPS for rendering (affects sleep time between steps)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    
    return parser.parse_args()


def load_trained_model(model_path: str, device: str) -> tuple[Actor, Optional[EmpiricalNormalization], Dict[str, Any]]:
    """
    Load trained FastTD3 model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (actor_model, obs_normalizer, checkpoint_info)
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print("Available keys in checkpoint:", list(checkpoint.keys()))
    
    # Extract training arguments
    args_dict = checkpoint.get('args', {})
    
    # Get actor state dict
    actor_state_dict = None
    for key in ['actor_state_dict', 'actor', 'policy']:
        if key in checkpoint:
            actor_state_dict = checkpoint[key]
            break
    
    if actor_state_dict is None:
        raise ValueError("Could not find actor state dict in checkpoint")
    
    # Detect architecture type based on state dict keys
    is_simbav2 = any('embedder' in key or 'encoder' in key or 'predictor' in key 
                     for key in actor_state_dict.keys())
    
    print(f"Detected architecture: {'SimbaV2' if is_simbav2 else 'Standard FastTD3'}")
    
    # Get the correct number of environments from the checkpoint
    noise_scales_shape = actor_state_dict.get('noise_scales', torch.tensor([[1.0]])).shape
    original_num_envs = noise_scales_shape[0] if len(noise_scales_shape) > 1 else 1
    
    # Infer dimensions from model architecture if not in args
    n_obs = args_dict.get('n_obs')
    n_act = args_dict.get('n_act')
    
    # If dimensions not in args, try to infer from state dict
    if n_obs is None or n_act is None:
        if 'embedder.w.w.weight' in actor_state_dict:
            # For SimbaV2, embedder adds +1 to input, so we need to subtract 1
            embedder_input_dim = actor_state_dict['embedder.w.w.weight'].shape[1]
            inferred_n_obs = embedder_input_dim - 1  # Subtract the +1 that embedder adds
            if n_obs is None:
                n_obs = inferred_n_obs
                print(f"Inferred observation dimension: {n_obs} (embedder input: {embedder_input_dim})")
        
        if 'predictor.mean_bias' in actor_state_dict:
            inferred_n_act = actor_state_dict['predictor.mean_bias'].shape[0]
            if n_act is None:
                n_act = inferred_n_act
                print(f"Inferred action dimension: {n_act}")
    
    # Use defaults if still not found
    if n_obs is None:
        n_obs = 1090
        print(f"Using default observation dimension: {n_obs}")
    if n_act is None:
        n_act = 2
        print(f"Using default action dimension: {n_act}")
    
    # Determine actor parameters from args or use defaults
    base_actor_kwargs = {
        'n_obs': n_obs,
        'n_act': n_act,
        'num_envs': 1,  # Single environment for play
        'hidden_dim': args_dict.get('actor_hidden_dim', 512),
        'device': device,
        'std_min': args_dict.get('std_min', 0.001),
        'std_max': args_dict.get('std_max', 0.4)
    }
    
    if is_simbav2:
        # SimbaV2 specific parameters - calculate defaults like in train.py
        import math
        actor_hidden_dim = base_actor_kwargs['hidden_dim']
        actor_num_blocks = args_dict.get('actor_num_blocks', 1)
        
        actor_kwargs = {
            **base_actor_kwargs,
            'scaler_init': args_dict.get('scaler_init', math.sqrt(2.0 / actor_hidden_dim)),
            'scaler_scale': args_dict.get('scaler_scale', math.sqrt(2.0 / actor_hidden_dim)),
            'alpha_init': args_dict.get('alpha_init', 1.0 / (actor_num_blocks + 1)),
            'alpha_scale': args_dict.get('alpha_scale', 1.0 / math.sqrt(actor_hidden_dim)),
            'expansion': args_dict.get('expansion', 4),
            'c_shift': args_dict.get('c_shift', 3.0),
            'num_blocks': actor_num_blocks
        }
        
        # Check if this is a multi-task actor
        if 'num_tasks' in args_dict:
            actor_kwargs.update({
                'num_tasks': args_dict['num_tasks'],
                'task_embedding_dim': args_dict.get('task_embedding_dim', 32)
            })
            actor = SimbaV2MultiTaskActor(**actor_kwargs)
            print("Loaded SimbaV2 MultiTaskActor")
        else:
            actor = SimbaV2Actor(**actor_kwargs)
            print("Loaded SimbaV2 Actor")
    else:
        # Standard FastTD3 actor
        actor_kwargs = {
            **base_actor_kwargs,
            'init_scale': args_dict.get('init_scale', 0.1)
        }
        
        # Check if this is a multi-task actor
        if 'num_tasks' in args_dict:
            actor_kwargs.update({
                'num_tasks': args_dict['num_tasks'],
                'task_embedding_dim': args_dict.get('task_embedding_dim', 32)
            })
            actor = MultiTaskActor(**actor_kwargs)
            print("Loaded standard MultiTaskActor")
        else:
            actor = Actor(**actor_kwargs)
            print("Loaded standard Actor")
    
    # Fix noise_scales dimension if needed
    if original_num_envs != 1:
        # Create new noise_scales with correct shape for single environment
        new_noise_scales = actor_state_dict['noise_scales'][:1].clone()
        actor_state_dict = actor_state_dict.copy()
        actor_state_dict['noise_scales'] = new_noise_scales
        print(f"Adjusted noise_scales from {original_num_envs} envs to 1 env")
    
    # Load actor state dict
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    
    # Load observation normalizer if available
    obs_normalizer = None
    obs_normalizer_state = checkpoint.get('obs_normalizer_state')
    if obs_normalizer_state is not None:
        obs_normalizer = EmpiricalNormalization(
            shape=base_actor_kwargs['n_obs'], 
            device=device
        )
        obs_normalizer.load_state_dict(obs_normalizer_state)
        obs_normalizer.eval()
        print("Loaded observation normalizer")
    else:
        print("No observation normalizer found in checkpoint")
    
    print(f"✅ Model loaded successfully!")
    print(f"Actor architecture: {actor}")
    
    return actor, obs_normalizer, args_dict


def create_isaaclab_env(args, render_mode: str = "rgb_array") -> IsaacLabEnv:
    """
    Create IsaacLab environment with rendering enabled.
    
    Args:
        args: Command line arguments
        render_mode: Rendering mode ("rgb_array" or "human")
        
    Returns:
        IsaacLab environment instance
    """
    print(f"Creating IsaacLab environment: {args.task_name}")
    print(f"Device: {args.device}")
    print(f"Render mode: {render_mode}")
    print(f"Headless: {args.headless}")
    
    env = IsaacLabEnv(
        task_name=args.task_name,
        device=args.device,
        num_envs=1,  # Single environment for visualization
        seed=args.seed,
        action_bounds=args.action_bounds,
        render_mode=render_mode,
        headless=args.headless
    )
    
    print(f"✅ Environment created successfully!")
    print(f"Observation space: {env.num_obs}")
    print(f"Action space: {env.num_actions}")
    print(f"Max episode steps: {env.max_episode_steps}")
    
    return env


def play_episode(
    env: IsaacLabEnv,
    actor: Actor,
    obs_normalizer: Optional[EmpiricalNormalization],
    episode_num: int,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    target_fps: int = 60,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Play a single episode with the trained model.
    
    Args:
        env: IsaacLab environment
        actor: Trained actor model
        obs_normalizer: Observation normalizer (if used during training)
        episode_num: Episode number for logging
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        target_fps: Target FPS for rendering
        verbose: Whether to print step information
        
    Returns:
        Episode statistics dictionary
    """
    print(f"\n🎮 Playing Episode {episode_num + 1}")
    print("=" * 50)
    
    # Reset environment
    obs = env.reset(random_start_init=False)  # Consistent starting position for visualization
    
    episode_reward = 0.0
    episode_length = 0
    done = False
    
    if max_steps is None:
        max_steps = env.max_episode_steps
    
    # Calculate sleep time for target FPS
    sleep_time = 1.0 / target_fps if target_fps > 0 else 0
    
    with torch.no_grad():
        while not done and episode_length < max_steps:
            step_start_time = time.time()
            
            # Normalize observation if normalizer is available
            if obs_normalizer is not None:
                normalized_obs = obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            
            # Get action from actor
            action = actor.explore(normalized_obs, deterministic=deterministic)
            
            # Step environment
            next_obs, reward, dones, info = env.step(action)
            
            # Update episode statistics
            episode_reward += reward.sum().item()
            episode_length += 1
            done = dones.any().item()
            
            # Render if not headless
            if not hasattr(env, 'headless') or not env.headless:
                try:
                    env.render()
                except Exception as e:
                    if verbose:
                        print(f"Render warning: {e}")
            
            # Update observation for next step
            obs = next_obs
            
            # Progress logging
            if verbose and episode_length % 100 == 0:
                print(f"Step {episode_length:4d} | Reward: {episode_reward:8.2f} | Action: {action[0].cpu().numpy()}")
            
            # Control FPS
            step_time = time.time() - step_start_time
            if sleep_time > step_time:
                time.sleep(sleep_time - step_time)
    
    # Episode summary
    episode_stats = {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'average_reward_per_step': episode_reward / max(episode_length, 1),
        'success': episode_reward > 0,  # Adjust success criteria as needed
    }
    
    print(f"📊 Episode {episode_num + 1} Results:")
    print(f"   Total Reward: {episode_reward:.2f}")
    print(f"   Episode Length: {episode_length}")
    print(f"   Avg Reward/Step: {episode_stats['average_reward_per_step']:.4f}")
    print(f"   Success: {episode_stats['success']}")
    
    return episode_stats


def main():
    """Main execution function."""
    args = parse_args()
    
    print("🚀 FastTD3 IsaacLab Player")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task_name}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Target FPS: {args.fps}")
    print(f"Headless: {args.headless}")
    print("=" * 60)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    
    try:
        # Load trained model
        actor, obs_normalizer, model_info = load_trained_model(args.model_path, device)
        
        # Create environment
        render_mode = "human" if not args.headless else "rgb_array"
        env = create_isaaclab_env(args, render_mode=render_mode)
        
        # Validate model and environment compatibility
        expected_obs_dim = env.num_obs
        expected_action_dim = env.num_actions
        
        print(f"\n🔍 Compatibility Check:")
        print(f"Environment - Obs: {expected_obs_dim}, Actions: {expected_action_dim}")
        # Note: We can't easily check actor dimensions without forward pass
        print(f"Model loaded successfully - compatibility assumed")
        
        if not args.headless:
            print(f"\n🎬 Starting visualization...")
            print(f"IsaacSim GUI should open shortly...")
            print(f"Press Ctrl+C to stop early")
        
        # Play episodes
        all_episode_stats = []
        
        for episode in range(args.num_episodes):
            try:
                episode_stats = play_episode(
                    env=env,
                    actor=actor,
                    obs_normalizer=obs_normalizer,
                    episode_num=episode,
                    max_steps=args.max_episode_steps,
                    deterministic=args.deterministic,
                    target_fps=args.fps,
                    verbose=True
                )
                all_episode_stats.append(episode_stats)
                
                # Short pause between episodes
                if episode < args.num_episodes - 1:
                    print(f"\n⏸️  Pausing 2 seconds before next episode...")
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopped by user after {episode + 1} episodes")
                break
            except Exception as e:
                print(f"\n❌ Error in episode {episode + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final summary
        if all_episode_stats:
            print(f"\n📈 Final Summary ({len(all_episode_stats)} episodes):")
            print("=" * 50)
            
            rewards = [stats['episode_reward'] for stats in all_episode_stats]
            lengths = [stats['episode_length'] for stats in all_episode_stats]
            successes = [stats['success'] for stats in all_episode_stats]
            
            print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"Success Rate: {np.mean(successes):.2%}")
            print(f"Best Reward: {np.max(rewards):.2f}")
            print(f"Worst Reward: {np.min(rewards):.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n✅ Playback completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
