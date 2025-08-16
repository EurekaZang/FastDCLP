#!/usr/bin/env python3
"""
Example script for evaluating and visualizing trained FastTD3 models using PlayIsaacLabEnv.

Usage:
    python evaluate_model.py --model_path models/your_model.pt --task_name Isaac-Cartpole-v0
"""

import argparse
import torch
import os
import sys

# Add the parent directory to sys.path to import fast_td3 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_td3.fast_td3 import Actor, MultiTaskActor
from fast_td3.environments.play_isaaclab_env import PlayIsaacLabEnv, load_trained_model
from fast_td3.fast_td3_utils import EmpiricalNormalization


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained FastTD3 model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True,
        help="IsaacLab task name (e.g., Isaac-Cartpole-v0)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run evaluation on (cuda/cpu)"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--deterministic", 
        action="store_true",
        help="Use deterministic policy (no exploration noise)"
    )
    parser.add_argument(
        "--record_video", 
        action="store_true",
        help="Record videos of the episodes"
    )
    parser.add_argument(
        "--video_path", 
        type=str, 
        default="./evaluation_videos/",
        help="Path to save recorded videos"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run without visual rendering"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for evaluation"
    )
    
    return parser.parse_args()


def load_model_and_normalizers(model_path: str, device: str):
    """
    Load the trained model and normalizers from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (actor, obs_normalizer, critic_obs_normalizer)
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Print available keys in checkpoint
    print("Available keys in checkpoint:", list(checkpoint.keys()))
    
    # Load actor
    actor_state_dict = None
    actor_kwargs = None
    
    # Try different possible key names for actor
    for key in ['actor_state_dict', 'actor', 'policy']:
        if key in checkpoint:
            actor_state_dict = checkpoint[key]
            break
    
    # Try to get actor kwargs
    for key in ['actor_kwargs', 'model_config', 'args']:
        if key in checkpoint:
            actor_kwargs = checkpoint[key]
            break
    
    if actor_state_dict is None:
        raise ValueError("Could not find actor state dict in checkpoint")
    
    # Create actor instance
    # You may need to adjust these parameters based on your specific model
    if actor_kwargs and hasattr(actor_kwargs, 'num_tasks'):
        # Multi-task actor
        actor = MultiTaskActor(**actor_kwargs)
    else:
        # Regular actor - you may need to manually specify parameters
        # These should match the parameters used during training
        default_actor_kwargs = {
            'n_obs': 1090,  # Adjust based on your observation space
            'n_act': 2,     # Adjust based on your action space
            'num_envs': 1,  # For evaluation
            'init_scale': 0.1,
            'hidden_dim': 256,
            'device': device,
        }
        
        if actor_kwargs:
            default_actor_kwargs.update(actor_kwargs)
            
        actor = Actor(**default_actor_kwargs)
    
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    
    # Load observation normalizers if available
    obs_normalizer = None
    critic_obs_normalizer = None
    
    if 'obs_normalizer' in checkpoint:
        obs_normalizer = EmpiricalNormalization(shape=actor.net[0].in_features, device=device)
        obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
        obs_normalizer.eval()
    
    if 'critic_obs_normalizer' in checkpoint:
        # Determine critic observation size
        critic_obs_size = checkpoint.get('critic_obs_size', actor.net[0].in_features)
        critic_obs_normalizer = EmpiricalNormalization(shape=critic_obs_size, device=device)
        critic_obs_normalizer.load_state_dict(checkpoint['critic_obs_normalizer'])
        critic_obs_normalizer.eval()
    
    print(f"Model loaded successfully. Actor type: {type(actor).__name__}")
    
    return actor, obs_normalizer, critic_obs_normalizer


def main():
    args = parse_args()
    
    print("="*60)
    print("FastTD3 Model Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task_name}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("="*60)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    
    try:
        # Load model and normalizers
        actor, obs_normalizer, critic_obs_normalizer = load_model_and_normalizers(
            args.model_path, device
        )
        
        # Create play environment
        play_env = PlayIsaacLabEnv(
            task_name=args.task_name,
            device=device,
            num_envs=1,
            seed=args.seed,
            enable_viewport=not args.headless,
            record_video=args.record_video,
            video_path=args.video_path,
        )
        
        print(f"Environment created successfully")
        print(f"Observation space: {play_env.num_obs}")
        print(f"Action space: {play_env.num_actions}")
        print(f"Max episode steps: {play_env.max_episode_steps}")
        
        # Evaluate model
        print("\nStarting evaluation...")
        results = play_env.evaluate_model(
            actor_model=actor,
            num_episodes=args.num_episodes,
            obs_normalizer=obs_normalizer,
            deterministic=args.deterministic,
            verbose=True
        )
        
        # Save evaluation results
        results_file = f"evaluation_results_{args.task_name.replace('-', '_')}.txt"
        with open(results_file, 'w') as f:
            f.write("FastTD3 Model Evaluation Results\n")
            f.write("="*40 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Task: {args.task_name}\n")
            f.write(f"Episodes: {args.num_episodes}\n")
            f.write(f"Deterministic: {args.deterministic}\n")
            f.write("\nResults:\n")
            for key, value in results.items():
                if key not in ['episode_rewards', 'episode_lengths']:
                    f.write(f"{key}: {value}\n")
            f.write("\nEpisode Rewards:\n")
            f.write(str(results['episode_rewards']))
            f.write("\nEpisode Lengths:\n")
            f.write(str(results['episode_lengths']))
        
        print(f"\nResults saved to: {results_file}")
        
        if args.record_video:
            print(f"Videos saved to: {args.video_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
