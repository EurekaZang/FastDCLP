#!/usr/bin/env python3
"""
play_dclp_isaac_lab.py - Play and visualize trained DCLP models in IsaacLab environments

This script loads a trained DCLP model and runs it in an IsaacLab environment with rendering enabled.
The IsaacSim GUI will open to show the robot following the trained policy.

Usage:
    python play_dclp_isaac_lab.py --model_path models/your_model.pt --task_name Isaac-Navigation-Flat-Turtlebot2-v0
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

from fast_td3_utils import EmpiricalNormalization
from environments.isaaclab_env import IsaacLabEnv
from dclp import FastDCLP


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play trained DCLP model in IsaacLab")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained DCLP model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="IsaacLab task name (e.g., Isaac-Navigation-Flat-Turtlebot2-v0)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of environments to run in parallel"
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
        default=1,
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


def load_trained_dclp_model(model_path: str, device: str) -> tuple[FastDCLP, Optional[EmpiricalNormalization], Dict[str, Any]]:
    """
    Load trained DCLP model from checkpoint.
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    Returns:
        Tuple of (dclp_model, obs_normalizer, checkpoint_info)
    """
    print(f"Loading DCLP model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print("Available keys in checkpoint:", list(checkpoint.keys()))

    # Extract training arguments
    args_dict = checkpoint.get('args', {})
    
    # Get dimensions from args or use defaults
    state_dim = args_dict.get('n_obs', 548)  # DCLP typically uses 280 for LiDAR + robot state
    action_dim = args_dict.get('n_act', 2)   # Typically 2 for navigation (linear, angular velocity)

    print(f"Model dimensions - State: {state_dim}, Action: {action_dim}")

    # Extract FastDCLP specific parameters
    actor_hidden_dim = args_dict.get('actor_hidden_dim', 256)
    critic_hidden_dim = args_dict.get('critic_hidden_dim', 256)
    num_hidden_layers = args_dict.get('num_hidden_layers', 4)
    
    actor_hidden_sizes = [actor_hidden_dim for _ in range(num_hidden_layers)]
    critic_hidden_sizes = [critic_hidden_dim for _ in range(num_hidden_layers)]

    # Create FastDCLP instance with same parameters as training
    dclp = FastDCLP(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=args_dict.get('actor_learning_rate', 3e-4),
        critic_lr=args_dict.get('critic_learning_rate', 3e-4),
        gamma=args_dict.get('gamma', 0.99),
        tau=args_dict.get('tau', 0.005),
        alpha=args_dict.get('alpha', 0.2),
        actor_hidden_sizes=actor_hidden_sizes,
        critic_hidden_sizes=critic_hidden_sizes,
        num_atoms=args_dict.get('num_atoms', 251),
        v_min=args_dict.get('v_min', -100.0),
        v_max=args_dict.get('v_max', 100.0),
        use_grad_norm_clipping=args_dict.get('use_grad_norm_clipping', True),
        max_grad_norm=args_dict.get('max_grad_norm', 1.0),
        device=device,
        # Default to no AMP/compile for inference/play
        amp_enabled=False,
        compile_mode="default"
    )

    # Load model state dict
    try:
        # Check if this is a DCLP checkpoint with the expected format
        if 'actor_critic' in checkpoint:
            # Use DCLP's built-in load method
            dclp.load(model_path)
            print("✅ Loaded DCLP model using native load method")
        else:
             print("⚠️  Warning: Could not find 'actor_critic' in checkpoint.")
    except Exception as e:
        print(f"⚠️  Warning: Error loading state dict: {e}")
        import traceback
        traceback.print_exc()

    # Set to evaluation mode
    dclp.actor_critic.eval()

    # Load observation normalizer if available
    obs_normalizer = None
    obs_normalizer_state = checkpoint.get('obs_normalizer_state')
    if obs_normalizer_state is not None:
        obs_normalizer = EmpiricalNormalization(
            shape=state_dim, 
            device=device
        )
        obs_normalizer.load_state_dict(obs_normalizer_state)
        obs_normalizer.eval()
        print("✅ Loaded observation normalizer")
    else:
        print("ℹ️  No observation normalizer found in checkpoint")

    print(f"✅ DCLP model loaded successfully!")

    return dclp, obs_normalizer, args_dict


def create_isaaclab_env(args, render_mode: str = "human") -> IsaacLabEnv:
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
        num_envs=args.num_envs,
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
    dclp: FastDCLP,
    obs_normalizer: Optional[EmpiricalNormalization],
    episode_num: int,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    target_fps: int = 60,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Play a single episode with the trained DCLP model.
    
    Args:
        env: IsaacLab environment
        dclp: Trained DCLP model
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
            
            # Get action from DCLP model
            action = dclp.get_action(normalized_obs, deterministic=deterministic)
            
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
            if verbose and episode_length % 50 == 0:
                print(f"Step {episode_length:4d} | Reward: {episode_reward:8.2f} | Action: {action[0].cpu().numpy() if torch.is_tensor(action) else action}")
            
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
    
    print("🚀 DCLP IsaacLab Player")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task_name}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Target FPS: {args.fps}")
    print("=" * 60)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load trained DCLP model
        print("\n📥 Loading DCLP model...")
        dclp, obs_normalizer, model_info = load_trained_dclp_model(args.model_path, device)
        
        # Create environment
        print("\n🌍 Creating environment...")
        render_mode = "human" if not args.headless else "rgb_array"
        env = create_isaaclab_env(args, render_mode)
        
        # Validate dimensions
        if env.num_obs != dclp.actor_critic.state_dimension:
            print(f"⚠️  Warning: Environment obs dim ({env.num_obs}) != model state dim ({dclp.actor_critic.state_dimension})")
            print("This might cause issues. Consider retraining or using a compatible environment.")
        
        if env.num_actions != dclp.actor_critic.action_dimension:
            print(f"⚠️  Warning: Environment action dim ({env.num_actions}) != model action dim ({dclp.actor_critic.action_dimension})")
        
        # Play episodes
        print(f"\n🎮 Playing {args.num_episodes} episodes...")
        all_episode_stats = []
        
        for episode in range(args.num_episodes):
            episode_stats = play_episode(
                env=env,
                dclp=dclp,
                obs_normalizer=obs_normalizer,
                episode_num=episode,
                max_steps=args.max_episode_steps,
                deterministic=args.deterministic,
                target_fps=args.fps,
                verbose=True
            )
            all_episode_stats.append(episode_stats)
            
            # Sleep between episodes
            if episode < args.num_episodes - 1:
                print(f"\n⏸️  Waiting 3 seconds before next episode...")
                time.sleep(3.0)
        
        # Final statistics
        print("\n📈 Final Statistics:")
        print("=" * 60)
        
        avg_reward = np.mean([stats['episode_reward'] for stats in all_episode_stats])
        avg_length = np.mean([stats['episode_length'] for stats in all_episode_stats])
        success_rate = np.mean([stats['success'] for stats in all_episode_stats])
        
        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Episodes: {len(all_episode_stats)}")
        
        print("\n✅ All episodes completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🏁 Shutting down...")


if __name__ == "__main__":
    main()
