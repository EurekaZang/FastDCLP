#!/usr/bin/env python3
"""
Fastdclp with DSAC

Usage:
    python train_dclp_isaac_lab.py --env_name Isaac-Navigation-Flat-Jackal-v0     --num_envs 4096 --buffer_size 10240 --batch_size 32768    --actor_hidden_dim 256 --critic_hidden_dim 256     --learning_starts 10000 --total_timesteps 1000000   --exp_name "try_training_real_fastdclp_with_entropy" --no-headless
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
import gc
import psutil

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
from dclp_utils import DCLPArgs, calculate_turtlebot_velocities

torch.set_float32_matmul_precision("high")


def main():
    args = tyro.cli(DCLPArgs)
    print(f"Training DCLP with args: {args}")

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    # Setup mixed precision
    amp_enabled = False
    amp_device_type = "cpu"
    amp_dtype = torch.float32

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
        device=str(device),
        num_envs=args.num_envs,
        seed=args.seed,
        action_bounds=args.action_bounds,
        render_mode=args.render_mode,
        headless=args.headless,
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

    print("Creating DCLP network...")
    # Create single DCLP instance that contains both actor and critic
    dclp = DCLP(
        state_dim=n_obs,
        action_dim=n_act,
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        hidden_sizes=(args.actor_hidden_dim, args.actor_hidden_dim, args.actor_hidden_dim, args.actor_hidden_dim),
        use_grad_norm_clipping=args.use_grad_norm_clipping,
        max_grad_norm=args.max_grad_norm,
        device=device
    )

    print("DCLP network initialized successfully")
    print("Optimizers and schedulers initialized successfully")

    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dclp.critic_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    print("Q-scheduler initialized successfully")
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dclp.actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )
    print("Actor-scheduler initialized successfully")

    # Replay buffer
    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
    )
    print("Replay buffer initialized successfully")
    #FIXME: Any problem?
    def evaluate():
        """Evaluation function"""
        print("Running evaluation...")
        print(f"Max episode steps: {envs.max_episode_steps}")
        print(f"Number of environments: {envs.num_envs}")
        eval_returns = []
        eval_lengths = []
        num_eval_episodes = min(3, args.num_envs)
        eval_pbar = tqdm.tqdm(range(num_eval_episodes), desc="🔍 Evaluation Episodes", leave=False)
        for evaluate_episode in eval_pbar:
            obs = envs.reset(random_start_init=False)
            episode_return = 0.0
            episode_length = 0
            print(f"\nStarting evaluation episode {evaluate_episode+1}")
            max_eval_steps = min(1000, envs.max_episode_steps)
            step_pbar = tqdm.tqdm(range(max_eval_steps),
                                desc=f"Episode {evaluate_episode+1}/{num_eval_episodes}",
                                leave=False)
            with torch.no_grad():
                for step in step_pbar:
                    if args.obs_normalization:
                        norm_obs = obs_normalizer(obs, update=False)
                    else:
                        norm_obs = obs
                    actions = dclp.get_action(norm_obs)
                    obs, reward, done, info = envs.step(actions)
                    episode_return += reward.mean().item()
                    episode_length += 1
                    # Update step progress bar with current reward
                    step_pbar.set_postfix({
                        'Return': f'{episode_return:.2f}',
                        'Reward': f'{reward.mean().item():.3f}',
                        'Length': episode_length,
                        'Done': done.sum().item()
                    })
                    if done[0]:
                        print(f"Episode {evaluate_episode+1} finished at step {episode_length} (done=True)")
                        break
                    if episode_length >= max_eval_steps:
                        print(f"Episode {evaluate_episode+1} truncated at {max_eval_steps} steps for faster evaluation")
                        break
                    if episode_length % 1000 == 0:
                        print(f"Episode {evaluate_episode+1}: Step {episode_length}, Return: {episode_return:.2f}, Last Reward: {reward.mean().item():.3f}")
            step_pbar.close()
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
            eval_pbar.set_postfix({
                'Avg Return': f'{np.mean(eval_returns):.2f}',
                'Avg Length': f'{np.mean(eval_lengths):.1f}'
            })
        eval_pbar.close()
        mean_return = np.mean(eval_returns)
        mean_length = np.mean(eval_lengths)
        print(f"Eval: Mean Return = {mean_return:.2f}, Mean Length = {mean_length:.1f}")
        if args.use_wandb:
            wandb.log({
                "eval/mean_return": mean_return,
                "eval/mean_length": mean_length,
            })
        
        return mean_return
    
    

    # def update_dclp(data, logs_dict):
    #     """Update DCLP network (both actor and critic)"""
    #     #TODO: Add autocast? Be same to FastTD3
    #     # Extract data from TensorDict format
    #     observations = data["observations"].float()  # Convert from float16 to float32
    #     actions = data["actions"]
    #     rewards = data["next"]["rewards"]
    #     next_observations = data["next"]["observations"].float()  # Convert from float16 to float32
    #     dones = data["next"]["dones"].bool()

    #     if args.obs_normalization:
    #         normalized_obs = obs_normalizer(observations,  update=True)
    #         normalized_next_obs = obs_normalizer(next_observations, update=False)
    #     else:
    #         normalized_obs = observations
    #         normalized_next_obs = next_observations

    #     # Prepare batch data for DCLP
    #     batch = {
    #         'state': normalized_obs.cpu().numpy(),
    #         'action': actions.cpu().numpy(),
    #         'reward': rewards.cpu().numpy(),
    #         'next_state': normalized_next_obs.cpu().numpy(),
    #         'done': dones.cpu().numpy()
    #     }

    #     # Train DCLP
    #     train_metrics = dclp.train_step(batch)

    #     # Calculate total episode reward
    #     total_episode_reward = rewards.sum().item()

    #     # Log total episode reward to W&B
    #     if args.use_wandb:
    #         logs_dict["total_episode_reward"] = total_episode_reward

    #     # Log metrics
    #     logs_dict.update(train_metrics)

    # Log hyperparameters to W&B
    if args.use_wandb:
        wandb.config.update({
            "actor_learning_rate": args.actor_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "gamma": args.gamma,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "num_envs": args.num_envs,
            "total_timesteps": args.total_timesteps,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
        })

    # Log system metrics to W&B
    if args.use_wandb:
        wandb.log({
            "system/gpu_memory_allocated": torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0,
            "system/gpu_memory_reserved": torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0,
            "system/cpu_memory_usage": psutil.virtual_memory().percent,
        })



    # Training loop
    global_step = 0
    obs = envs.reset(random_start_init=False)
    dones = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = time.time()
    print("🚀 Starting DCLP training...")
    #TODO: Check the logic
    while global_step < args.total_timesteps:
        with torch.no_grad():
            if global_step < args.learning_starts:
                actions = torch.rand(args.num_envs, n_act, device=device) * 2 - 1
            else:
                if args.obs_normalization:
                    norm_obs = obs_normalizer(obs, update=False)
                else:
                    norm_obs = obs
                actions = dclp.get_action(norm_obs)
                # print("debug actions",actions)
                wandb.log({
                    "env0_linear_action": actions[0][0] * 10,
                    "env0_angular_action": actions[0][1] * 10,
                })
        next_obs, rewards, next_dones, infos = envs.step(actions)
        if hasattr(reward_normalizer, 'update_stats'):
            reward_normalizer.update_stats(rewards, next_dones)
        normalized_rewards = reward_normalizer(rewards) # If --reward-normalization = False, this is identity
        
        # if envs.asymmetric_obs:
        #     next_critic_obs = infos["observations"]["critic"]
        # # Compute 'true' next_obs and next_critic_obs for saving
        # true_next_obs = torch.where(
        #     dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        # )
        # if envs.asymmetric_obs:
        #     true_next_critic_obs = torch.where(
        #         dones[:, None] > 0,
        #         infos["observations"]["raw"]["critic_obs"],
        #         next_critic_obs,
        #     )

        tensor_dict = TensorDict({
            "observations": obs,
            "actions": actions,
            "critic_observations": obs if not envs.asymmetric_obs else obs,
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
        logs_dict = {} # 用于存储 *最后一次* 更新的指标
        if global_step >= args.learning_starts:
            for _ in range(args.num_updates):
                data = rb.sample(args.batch_size)
                
                # --- 数据准备 (从 update_dclp 移入) ---
                observations = data["observations"].float()
                actions_batch = data["actions"]
                rewards_batch = data["next"]["rewards"]
                next_observations = data["next"]["observations"].float()
                dones_batch = data["next"]["dones"].bool()

                if args.obs_normalization:
                    normalized_obs = obs_normalizer(observations,  update=True)
                    normalized_next_obs = obs_normalizer(next_observations, update=False)
                else:
                    normalized_obs = observations
                    normalized_next_obs = next_observations

                batch = {
                    'state': normalized_obs.cpu().numpy(),
                    'action': actions_batch.cpu().numpy(),
                    'reward': rewards_batch.cpu().numpy(),
                    'next_state': normalized_next_obs.cpu().numpy(),
                    'done': dones_batch.cpu().numpy()
                }

                # Train DCLP 并获取指标
                logs_dict = dclp.train_step(batch)
                # 在优化器 step 完成后更新学习率调度器
                q_scheduler.step()
                actor_scheduler.step()
                # 存储用于日志的 buffer 奖励
                logs_dict["buffer_rewards"] = rewards_batch.mean().item()
            
            # --- 新增: 每 100 步记录一次日志 ---
            if global_step % 100 == 0 and args.use_wandb:
                elapsed_time = time.time() - start_time
                speed = global_step / elapsed_time if elapsed_time > 0 else 0

                wandb_logs = {
                    "speed": speed,
                    "frame": global_step * args.num_envs,
                    "critic_lr": q_scheduler.get_last_lr()[0],
                    "actor_lr": actor_scheduler.get_last_lr()[0],
                    "env_rewards": rewards.mean().item(), # 'rewards' 是来自 env.step 的最新奖励
                    
                    # --- 添加来自 DCLP 的指标 ---
                    # (使用 .get() 避免在 logs_dict 为空时出错)
                    "actor_loss": logs_dict.get('actor_loss'),
                    "qf_loss": logs_dict.get('qf_loss'),
                    "actor_grad_norm": logs_dict.get('actor_grad_norm'),
                    "critic_grad_norm": logs_dict.get('critic_grad_norm'),
                    "buffer_rewards": logs_dict.get('buffer_rewards'),
                    "qf_max": logs_dict.get('q1_mean'), # 使用 q1_mean 作为 qf_max 的代理
                    "qf_min": logs_dict.get('q2_mean'), # 使用 q2_mean 作为 qf_min 的代理
                    "policy_q_mean": logs_dict.get('policy_q_mean'),
                    "target_q_mean": logs_dict.get('target_q_mean'),
                    "log_probs_mean": logs_dict.get('log_probs_mean'),
                    
                    # --- 动作日志 ---
                    "env0_linear_action": actions[0][0] * 10,
                    "env0_angular_action": actions[0][1] * 10,
                }
                
                # 过滤掉 None 值
                wandb_logs = {k: v for k, v in wandb_logs.items() if v is not None}
                wandb.log(wandb_logs, step=global_step)
        # Periodic evaluation
        if global_step % (args.eval_interval * args.num_envs) == 0 and global_step > 0:
            evaluate()
        # Periodic model saving
        if global_step % args.save_interval == 0 and global_step > 0:
            os.makedirs("models", exist_ok=True)
            save_path = f"./models/{run_name}_{global_step}.pt"
            dclp.save(save_path)
            print(f"Model saved to {save_path}")
            # Log model checkpoint to W&B
            if args.use_wandb:
                artifact = wandb.Artifact(run_name, type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
        # Update progress bar
        rewards = rewards.mean(dim=-1).item()
        if global_step % (100 * args.num_envs) == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
            pbar.set_description(
                f"step: {global_step},"
                f"buffer: {rb.ptr}/{rb.size},"
                f"reward: {rewards:.4f},"
                f"SPS: {steps_per_sec:.2f}" # --- 添加 SPS 到 pbar ---
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