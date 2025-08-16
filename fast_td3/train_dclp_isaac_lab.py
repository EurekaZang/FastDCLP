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
    buffer_size: int = 1000000
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
    amp_dtype: str = "bf16"
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
    
    print("Creating DCLP Actor and Critic networks...")
    actor = Actor(**actor_kwargs)
    critic = Critic(**critic_kwargs)
    critic_target = Critic(**critic_kwargs)
    
    # Initialize target network
    critic_target.load_state_dict(critic.state_dict())
    
    print("Actor initialized successfully")
    print("Critic and target critic initialized successfully")
    
    # Setup optimizers
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    critic_optimizer = optim.AdamW(
        list(critic.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    
    # Learning rate schedulers
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    
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
    
    # Compile models if requested
    if args.compile:
        print("Compiling models...")
        actor = torch.compile(actor, mode=args.compile_mode)
        critic = torch.compile(critic, mode=args.compile_mode)
        critic_target = torch.compile(critic_target, mode=args.compile_mode)
    
    def evaluate():
        """Evaluation function"""
        print("Running evaluation...")
        actor.eval()
        
        eval_returns = []
        eval_lengths = []
        
        for eval_ep in range(min(10, args.num_envs)):
            obs = envs.reset(random_start_init=False)
            episode_return = 0.0
            episode_length = 0
            
            with torch.no_grad():
                for step in range(envs.max_episode_steps):
                    norm_obs = obs_normalizer(obs, update=False)
                    action = actor.explore(norm_obs, deterministic=True)
                    
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
        
        actor.train()
        return mean_return
    
    def update_critic(data, logs_dict):
        """Update critic network"""
        critic_optimizer.zero_grad()
        
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            # Get current Q-values
            q1_curr, q2_curr = critic(data["obs"], data["act"])
            
            # Compute target Q-values
            with torch.no_grad():
                target_noise = torch.randn_like(data["act"]) * args.policy_noise
                target_noise = torch.clamp(target_noise, -args.noise_clip, args.noise_clip)
                
                next_action = actor.explore(
                    obs_normalizer(data["obs2"], update=False), 
                    deterministic=True
                ) + target_noise
                next_action = torch.clamp(next_action, -1.0, 1.0)
                
                target_q1, target_q2 = critic_target(data["obs2"], next_action)
                target_q = torch.min(target_q1, target_q2)
                
                y = data["rew"] + args.gamma * (1 - data["done"]) * target_q
            
            # Critic loss
            critic_loss = F.mse_loss(q1_curr, y) + F.mse_loss(q2_curr, y)
        
        # Backward pass
        scaler.scale(critic_loss).backward()
        
        if args.use_grad_norm_clipping:
            scaler.unscale_(critic_optimizer)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        
        scaler.step(critic_optimizer)
        scaler.update()
        
        logs_dict["train/critic_loss"] = critic_loss.item()
    
    def update_actor(data, logs_dict):
        """Update actor network"""
        actor_optimizer.zero_grad()
        
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            actions = actor.explore(
                obs_normalizer(data["obs"], update=False),
                deterministic=True
            )
            q1, q2 = critic(data["obs"], actions)
            actor_loss = -torch.min(q1, q2).mean()
        
        # Backward pass
        scaler.scale(actor_loss).backward()
        
        if args.use_grad_norm_clipping:
            scaler.unscale_(actor_optimizer)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        
        scaler.step(actor_optimizer)
        scaler.update()
        
        logs_dict["train/actor_loss"] = actor_loss.item()
    
    @torch.no_grad()
    def soft_update_target():
        """Soft update target network"""
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(
                args.tau * param.data + (1 - args.tau) * target_param.data
            )
    
    # Training loop
    global_step = 0
    obs = envs.reset(random_start_init=False)
    dones = torch.zeros(args.num_envs, device=device)
    
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
                # Policy actions
                norm_obs = obs_normalizer(obs, update=False)
                actions = actor.explore(norm_obs, dones=dones)
        
        next_obs, rewards, next_dones, infos = envs.step(actions)
        
        # Normalize rewards
        if hasattr(reward_normalizer, 'update_stats'):
            reward_normalizer.update_stats(rewards, next_dones)
        normalized_rewards = reward_normalizer(rewards)
        
        # Store transitions
        rb.add(obs, actions, normalized_rewards, next_obs, dones, next_dones)
        
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
                
                # Update critic
                update_critic(data, logs_dict)
                
                # Update actor (delayed)
                if global_step % args.policy_frequency == 0:
                    update_actor(data, logs_dict)
                    soft_update_target()
                
                # Update learning rates
                actor_scheduler.step()
                critic_scheduler.step()
                
                # Log training metrics
                if args.use_wandb and len(logs_dict) > 0:
                    logs_dict["train/global_step"] = global_step
                    logs_dict["train/actor_lr"] = actor_optimizer.param_groups[0]['lr']
                    logs_dict["train/critic_lr"] = critic_optimizer.param_groups[0]['lr']
                    wandb.log(logs_dict)
        
        # Periodic evaluation
        if global_step % args.eval_interval == 0 and global_step > 0:
            evaluate()
        
        # Periodic model saving
        if global_step % args.save_interval == 0 and global_step > 0:
            save_params(
                global_step,
                actor,
                critic,
                critic_target,
                obs_normalizer,
                critic_obs_normalizer,
                args,
                f"models/{run_name}_{global_step}.pt",
            )
        
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
    
    save_params(
        global_step,
        actor,
        critic,
        critic_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )
    
    print(f"✅ Final evaluation return: {final_return:.2f}")
    print(f"💾 Model saved to: models/{run_name}_final.pt")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
