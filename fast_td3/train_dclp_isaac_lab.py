#!/usr/bin/env python3
"""
train_dclp_isaac_lab.py - Training script for DCLP model with IsaacLab environments

This script trains the DCLP (Deep Contextual Lidar Processing) model using FastTD3 algorithm
with IsaacLab navigation environments. It's optimized for efficient training with LiDAR data.

Usage:
    python train_dclp_isaac_lab.py --env_name Isaac-Navigation-Flat-Turtlebot2-v0 --total_timesteps 1000000
"""

import os
import sys
import random
import time
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import wandb
import tyro

# Environment setup
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.amp import autocast, GradScaler
from tensordict import TensorDict

from fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from environments.isaaclab_env import IsaacLabEnv
from dclp import DCLP, MLPActorCritic

torch.set_float32_matmul_precision("high")


@dataclass
class DCLPArgs:
    """DCLP training arguments"""
    # Environment
    env_name: str = "Isaac-Navigation-Flat-Turtlebot2-v0"
    """IsaacLab environment name"""
    
    # Training
    seed: int = 1
    """Random seed"""
    total_timesteps: int = 1000000
    """Total training timesteps"""
    learning_starts: int = 25000
    """Steps before learning starts"""
    num_envs: int = 1024
    """Number of parallel environments"""
    
    # Algorithm
    agent: str = "dclp_td3"
    """Agent type"""
    gamma: float = 0.99
    """Discount factor"""
    tau: float = 0.005
    """Target network soft update rate"""
    batch_size: int = 4096
    """Training batch size"""
    buffer_size: int = 1024 * 5
    """Replay buffer size"""
    
    # Learning rates
    actor_learning_rate: float = 3e-4
    """Actor learning rate"""
    critic_learning_rate: float = 3e-4
    """Critic learning rate"""
    actor_learning_rate_end: float = 3e-5
    """Actor final learning rate"""
    critic_learning_rate_end: float = 3e-5
    """Critic final learning rate"""
    
    # Network architecture
    actor_hidden_dim: int = 512
    """Actor hidden dimension"""
    critic_hidden_dim: int = 1024
    """Critic hidden dimension"""
    init_scale: float = 0.01
    """Actor initialization scale"""
    
    # TD3 specific
    policy_noise: float = 0.1
    """Policy noise for target smoothing"""
    noise_clip: float = 0.5
    """Noise clipping for target smoothing"""
    policy_frequency: int = 2
    """Policy update frequency"""
    num_updates: int = 1
    """Number of updates per step"""
    
    # Normalization
    obs_normalization: bool = True
    """Use observation normalization"""
    reward_normalization: bool = False
    """Use reward normalization"""
    
    # Hardware
    cuda: bool = True
    """Use CUDA if available"""
    device_rank: int = 0
    """Device rank"""
    torch_deterministic: bool = True
    """PyTorch deterministic mode"""
    
    # Optimization
    amp: bool = True
    """Use automatic mixed precision"""
    amp_dtype: str = "fp16"
    """AMP dtype (bf16 or fp16)"""
    compile: bool = True
    """Compile model with torch.compile"""
    compile_mode: str = "reduce-overhead"
    """Compile mode"""
    weight_decay: float = 0.0
    """Weight decay"""
    use_grad_norm_clipping: bool = False
    """Use gradient norm clipping"""
    max_grad_norm: float = 0.5
    """Maximum gradient norm"""
    
    # Logging
    use_wandb: bool = True
    """Use Weights & Biases logging"""
    project: str = "DCLP-IsaacLab"
    """W&B project name"""
    exp_name: str = "dclp_training"
    """Experiment name"""
    eval_interval: int = 10000
    """Evaluation interval"""
    save_interval: int = 50000
    """Model save interval"""
    
    # Environment specific
    action_bounds: float = 1.0
    """Action bounds scaling"""
    max_episode_steps: Optional[int] = None
    """Maximum episode steps override"""
    
    # DCLP specific
    lidar_points: int = 90
    """Number of LiDAR points (270/3)"""
    lidar_features: int = 3
    """LiDAR features per point (sin, cos, distance)"""
    use_cnn_features: bool = True
    """Use CNN for LiDAR feature extraction"""


def main():
    args = tyro.cli(DCLPArgs)
    print(f"Training DCLP with args: {args}")
    
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    
    # Setup mixed precision
    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda" if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() 
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Setup device
    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create IsaacLab environment
    print(f"Creating IsaacLab environment: {args.env_name}")
    envs = IsaacLabEnv(
        task_name=args.env_name,
        device=device.type,
        num_envs=args.num_envs,
        seed=args.seed,
        action_bounds=args.action_bounds,
        headless=True,  # Always headless for training
    )
    
    # Get environment dimensions
    n_obs = envs.num_obs
    n_act = envs.num_actions
    n_critic_obs = n_obs  # DCLP uses same obs for critic
    
    print(f"Environment specs - Obs: {n_obs}, Actions: {n_act}, Envs: {args.num_envs}")
    
    # Setup normalization
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()
    
    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(gamma=args.gamma, device=device)
    else:
        reward_normalizer = nn.Identity()
    
    # Create DCLP networks
    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
    }
    
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": 101,  # Fixed for distributional critic
        "v_min": -10.0,
        "v_max": 10.0,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }
    
    print("Creating DCLP network...")
    # Create single DCLP instance that contains both actor and critic
    dclp = DCLP(
        state_dim=n_obs,
        action_dim=n_act,
        lr=args.actor_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=0.2,  # Entropy coefficient
        hidden_sizes=(args.actor_hidden_dim, args.actor_hidden_dim, args.actor_hidden_dim, args.actor_hidden_dim),
        device=device
    )
    
    print("DCLP network initialized successfully")
    
    # Optimizers are already initialized within the DCLP class
    
    print("Optimizers and schedulers initialized successfully")
    
    # Replay buffer
    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=False,  # DCLP uses symmetric observations
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
    )
    
    print("Replay buffer initialized successfully")
    
    # Note: Model compilation is handled internally by DCLP if needed
    
    def evaluate():
        """Evaluation function"""
        print("Running evaluation...")
        
        eval_returns = []
        eval_lengths = []
        
        for eval_ep in range(min(10, args.num_envs)):
            obs = envs.reset(random_start_init=False)
            episode_return = 0.0
            episode_length = 0
            
            with torch.no_grad():
                for step in range(envs.max_episode_steps):
                    # Apply normalization if needed
                    if args.obs_normalization:
                        norm_obs = obs_normalizer(obs, update=False)
                    else:
                        norm_obs = obs
                        
                    # Get action from DCLP
                    action_np = dclp.get_action(norm_obs[0].cpu().numpy(), deterministic=True)
                    action = torch.from_numpy(action_np).unsqueeze(0).to(device)
                    
                    obs, reward, done, info = envs.step(action)
                    episode_return += reward[0].item()
                    episode_length += 1
                    
                    if done[0]:
                        break
            
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
        
        mean_return = np.mean(eval_returns)
        mean_length = np.mean(eval_lengths)
        
        print(f"Eval: Mean Return = {mean_return:.2f}, Mean Length = {mean_length:.1f}")
        
        if args.use_wandb:
            wandb.log({
                "eval/mean_return": mean_return,
                "eval/mean_length": mean_length,
            })
        
        return mean_return
    
    def update_dclp(data, logs_dict):
        """Update DCLP network (both actor and critic)"""
        
        # Extract data from TensorDict format
        observations = data["observations"].float()  # Convert from float16 to float32
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        next_observations = data["next"]["observations"].float()  # Convert from float16 to float32
        dones = data["next"]["dones"].bool()
        
        # Apply normalization if needed
        if args.obs_normalization:
            normalized_obs = obs_normalizer(observations, update=True)
            normalized_next_obs = obs_normalizer(next_observations, update=False)
        else:
            normalized_obs = observations
            normalized_next_obs = next_observations
        
        # Prepare batch data for DCLP
        batch = {
            'state': normalized_obs.cpu().numpy(),
            'action': actions.cpu().numpy(),
            'reward': rewards.cpu().numpy(),
            'next_state': normalized_next_obs.cpu().numpy(),
            'done': dones.cpu().numpy()
        }
        
        # Train DCLP
        train_metrics = dclp.train_step(batch)
        
        # Log metrics
        logs_dict.update(train_metrics)
    
    # Training loop
    global_step = 0
    obs = envs.reset(random_start_init=False)
    dones = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = time.time()
    
    print("🚀 Starting DCLP training...")
    
    while global_step < args.total_timesteps:
        # Collect experience
        with torch.no_grad():
            if global_step < args.learning_starts:
                # Random actions during warmup
                actions = torch.rand(args.num_envs, n_act, device=device) * 2 - 1
            else:
                # Policy actions from DCLP
                if args.obs_normalization:
                    norm_obs = obs_normalizer(obs, update=False)
                else:
                    norm_obs = obs
                
                # Get actions from all environments
                action_list = []
                for i in range(args.num_envs):
                    obs_np = norm_obs[i].cpu().numpy()
                    action_np = dclp.get_action(obs_np, deterministic=False)
                    action_list.append(action_np)
                
                actions = torch.from_numpy(np.array(action_list)).to(device)
        
        next_obs, rewards, next_dones, infos = envs.step(actions)
        
        # Normalize rewards
        if hasattr(reward_normalizer, 'update_stats'):
            reward_normalizer.update_stats(rewards, next_dones)
        normalized_rewards = reward_normalizer(rewards)
        
        # Store transitions using TensorDict format
        tensor_dict = TensorDict({
            "observations": obs,
            "actions": actions,
            "critic_observations": obs if not envs.asymmetric_obs else obs,  # DCLP uses same obs
            "next": TensorDict({
                "observations": next_obs,
                "critic_observations": next_obs if not envs.asymmetric_obs else next_obs,
                "rewards": normalized_rewards,
                "dones": dones,
                "truncations": next_dones,
            })
        })
        
        rb.extend(tensor_dict)
        
        # Update normalizers
        if args.obs_normalization:
            obs_normalizer(obs)
            critic_obs_normalizer(obs)
        
        obs = next_obs
        dones = next_dones
        global_step += args.num_envs
        
        # Training updates
        if global_step >= args.learning_starts:
            for _ in range(args.num_updates):
                data = rb.sample(args.batch_size)
                logs_dict = {}
                
                # Update DCLP (both actor and critic)
                update_dclp(data, logs_dict)
                
                # Log training metrics
                if args.use_wandb and len(logs_dict) > 0:
                    logs_dict["train/global_step"] = global_step
                    wandb.log(logs_dict)
        
        # Periodic evaluation
        if global_step % args.eval_interval == 0 and global_step > 0:
            evaluate()
        
        # Periodic model saving
        if global_step % args.save_interval == 0 and global_step > 0:
            os.makedirs("models", exist_ok=True)
            save_path = f"models/{run_name}_{global_step}.pt"
            dclp.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Update progress bar
        if global_step % 1000 == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
            pbar.set_description(
                f"Step: {global_step}, FPS: {steps_per_sec:.0f}, "
                f"Buffer: {rb.ptr}/{rb.size}"
            )
        pbar.update(args.num_envs)
    
    # Final evaluation and save
    print("🏁 Training completed!")
    final_return = evaluate()
    
    os.makedirs("models", exist_ok=True)
    final_save_path = f"models/{run_name}_final.pt"
    dclp.save(final_save_path)
    
    print(f"✅ Final evaluation return: {final_return:.2f}")
    print(f"💾 Model saved to: {final_save_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
    main()
