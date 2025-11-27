#!/usr/bin/env python3
"""
train_fastdclp_isaac_lab.py - Training script for DCLP model with IsaacLab environments

This script trains the DCLP model using FastTD3 algorithm
with IsaacLab navigation environments. It's optimized for efficient training with LiDAR data.

Usage:
python ./fast_td3/train_fastdclp_isaac_lab.py --env_name Isaac-Navigation-Flat-Jackal-v0 --num_envs 1024 --buffer_size 5120 --batch_size 32768 --actor_hidden_dim 256 --critic_hidden_dim 256 --learning_starts 10 --actor_learning_rate 0.0003 --critic_learning_rate 0.0003 --actor_learning_rate_end 0.0003 --critic_learning_rate_end 0.0003 --total_timesteps 100000 --exp_name fastdclp_with_entropy_optimedcudagraph_w_step_punish_changeparams_refined_vminmax --num_steps 8 --num_updates 1 --v_min -250 --v_max 250 --compile_mode default --alpha 0.1

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
from dclp import FastDCLP
from dclp_utils import DCLPArgs, calculate_turtlebot_velocities

torch.set_float32_matmul_precision("high")


def main():
    args = tyro.cli(DCLPArgs)
    print(f"Training DCLP with args: {args}")

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
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
    dclp = FastDCLP(
        state_dim=n_obs,
        action_dim=n_act,
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        actor_hidden_sizes=[args.actor_hidden_dim for _ in range(args.num_hidden_layers)],
        critic_hidden_sizes=[args.critic_hidden_dim for _ in range(args.num_hidden_layers)],
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        std_min=args.std_min,
        std_max=args.std_max,
        use_grad_norm_clipping=args.use_grad_norm_clipping,
        max_grad_norm=args.max_grad_norm,
        device=device,
        scalar = scaler,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        compile_mode=args.compile_mode,
        policy_frequency=args.policy_frequency,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        num_envs=args.num_envs,
    )

    if args.compile:
        dclp.enable_compile()

    print("DCLP network initialized successfully")
    print("Optimizers and schedulers initialized successfully")

    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dclp.critic_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.critic_learning_rate_end,
    )
    print("Q-scheduler initialized successfully")
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dclp.actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.actor_learning_rate_end,
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
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,     #torch.device("cpu"),
    )
    print("Replay buffer initialized successfully")
    def evaluate():
        """
        Evaluation function for IsaacLab environments.
        
        Note: IsaacLab automatically resets sub-environments when they terminate.
        We track only the FIRST episode completion for each environment.
        """
        num_eval_envs = envs.num_envs
        
        # Track cumulative stats for current episode
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
        episode_successes = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs = envs.reset(random_start_init=False)

        max_eval_steps = envs.max_episode_steps
        for step in range(max_eval_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):              
                if args.obs_normalization:
                    norm_obs = obs_normalizer(obs, update=False)
                else:
                    norm_obs = obs
                actions = dclp.get_action(norm_obs, deterministic=True)
            
            next_obs, rewards, dones, infos = envs.step(actions.float())
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            episode_successes = torch.where(
                ~done_masks, episode_successes | infos["successes"], episode_successes
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs
        episode_successes_rate = episode_successes.float().mean().item()
        return episode_returns.mean().item(), episode_lengths.mean().item(), episode_successes_rate

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
    if args.compile:
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    global_step = 0
    total_eps = 0
    total_success = 0
    window_eps_log = args.log_interval
    window_eps_acc = 0
    window_succ_acc = 0
    ema_success_rate = 0.0  # Initialize EMA success rate

    obs = envs.reset(random_start_init=False)
    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    start_time = time.time()
    print("🚀 Starting DCLP training...")

    while global_step < args.total_timesteps:
        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            if global_step < args.learning_starts:
                actions = torch.rand(args.num_envs, n_act, device=device) * 2 - 1
            else:
                if args.obs_normalization:
                    norm_obs = normalize_obs(obs, update=False)
                else:
                    norm_obs = obs
                actions = dclp.get_action(norm_obs, deterministic=False)

        next_obs, rewards, dones, infos = envs.step(actions)
        truncations = infos["time_outs"]
        successes = infos["successes"]
        eps = int(dones.sum().item())
        succ = int(successes.sum().item())
        total_eps += eps
        total_success += succ
        window_eps_acc += eps
        window_succ_acc += succ
        batch_rate = succ / max(1, eps) if eps > 0 else None
        if batch_rate is not None:
            ema_success_rate = ema_success_rate * 0.95 + 0.05 * batch_rate
        
        if args.reward_normalization:
            update_stats(rewards, dones.float())     
        # if envs.asymmetric_obs:
        #     next_critic_obs = infos["observations"]["critic"]
        # # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        # if envs.asymmetric_obs:
        #     true_next_critic_obs = torch.where(
        #         dones[:, None] > 0,
        #         infos["observations"]["raw"]["critic_obs"],
        #         next_critic_obs,
        #     )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        
        rb.extend(transition)
        # rb.extend(transition.to(device))
        # Update normalizers


        obs = next_obs
        # Training updates
        logs_dict = {}
        if global_step >= args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                # if rb.device is not None and getattr(rb.device, "type", None) == "cpu":
                #     data = data.to(device)
                # print("debug data device,338",data.device)
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)

                # Calculate whether to update actor this step
                if args.num_updates > 1:
                    do_actor_update = (i % args.policy_frequency == 1) 
                else:
                    do_actor_update = (global_step % args.policy_frequency == 0)
                
                # Train DCLP 并获取指标
                result = dclp.train_step(data, do_actor_update)
                # Unpack tuple and extract scalars immediately
                actor_loss, critic_loss, actor_grad_norm, critic_grad_norm, qf1, qf2, min_qf_next_target_value, qf_next_target_dist, policy_qf_value, log_probs = result
                
                # Create logs dict with extracted values
                # Note: actor values will be zero when actor is not updated
                logs_dict = {
                    'actor_loss': actor_loss.detach(),
                    'qf_loss': critic_loss.detach(),
                    'actor_grad_norm': actor_grad_norm.detach(),
                    'critic_grad_norm': critic_grad_norm.detach(),
                    'qf_max': min_qf_next_target_value.max().detach(),
                    'qf_min': min_qf_next_target_value.min().detach(),
                    'q_next_dist_numpy': qf_next_target_dist.mean(dim=0).detach().cpu().numpy(),
                    'q_dist_entropy': -(qf_next_target_dist.mean(dim=0) * torch.log(qf_next_target_dist.mean(dim=0) + 1e-8)).sum().item(),
                    'policy_q_mean': policy_qf_value.mean().detach(),
                    'log_probs_mean': log_probs.mean().detach(),
                }
            
            # Update learning rate schedulers after optimizer steps
            q_scheduler.step()
            actor_scheduler.step()
            
            # --- 新增: 每 100 步记录一次日志 ---
            if global_step % args.log_interval == 0 and args.use_wandb:
                elapsed_time = time.time() - start_time
                speed = global_step / elapsed_time if elapsed_time > 0 else 0
                with torch.no_grad():
                    wandb_logs = {
                        "Training/speed": speed,
                        "Training/frame": global_step * args.num_envs,
                        "Training/global_success_rate": total_success / max(1, total_eps),
                        "Training/ema_success_rate": ema_success_rate,
                        "Training/critic_lr": q_scheduler.get_last_lr()[0],
                        "Training/actor_lr": actor_scheduler.get_last_lr()[0],
                        "Training/env_rewards": rewards.mean(),
                        "Training/actor_loss": logs_dict.get('actor_loss'),
                        "Training/qf_loss": logs_dict.get('qf_loss'),
                        "Training/actor_grad_norm": logs_dict.get('actor_grad_norm'),
                        "Training/critic_grad_norm": logs_dict.get('critic_grad_norm'),
                        "Training/buffer_rewards": raw_rewards.mean(),
                        "Training/qf_max": logs_dict.get('qf_max'),
                        "Training/qf_min": logs_dict.get('qf_min'),
                        "Training/q_dist_entropy": logs_dict.get('q_dist_entropy'),
                        "Training/q_dist_hist": wandb.Histogram(logs_dict.get('q_next_dist_numpy')) if logs_dict.get('q_next_dist_numpy') is not None else None,
                        "Training/policy_q_mean": logs_dict.get('policy_q_mean'),
                        "Training/log_probs_mean": logs_dict.get('log_probs_mean'),
                        
                        # --- 动作日志 ---
                        "Training/env0_linear_action": actions[0][0] * 10,
                        "Training/env0_angular_action": actions[0][1] * 10,
                    }

                # 过滤掉 None 值
                wandb_logs = {k: v for k, v in wandb_logs.items() if v is not None}
                if args.use_wandb:
                    wandb.log(wandb_logs, step=global_step)

            # --- Evaluation Logging ---
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                print(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length, eval_success_rate = evaluate()
                
                if args.use_wandb:
                    wandb.log({
                        "Evaluation/eval_avg_return": eval_avg_return,
                        "Evaluation/eval_avg_length": eval_avg_length,
                        "Evaluation/eval_success_rate": eval_success_rate
                    }, step=global_step)

            if window_eps_acc > window_eps_log:
                if args.use_wandb:
                    wandb.log({"Training/window_success_rate": window_succ_acc / max(1, window_eps_acc)}, step=global_step)
                else:
                    print("window_success_rate", window_succ_acc / max(1, window_eps_acc))
                window_eps_acc = 0
                window_succ_acc = 0


        # Periodic model saving
        if global_step % args.save_interval == 0 and global_step > 0:
            os.makedirs("models", exist_ok=True)
            save_path = f"./models/{run_name}_{global_step}.pt"
            dclp.save(save_path)
            # Log model checkpoint to W&B
            if args.use_wandb:
                artifact = wandb.Artifact(run_name, type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
        # Update progress bar
        rewards = rewards.mean().item()
        if global_step % 100 == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
            pbar.set_description(
                f"step: {global_step},"
                f"buffer: {(rb.ptr % rb.size)}/{rb.size},"
                f"reward: {rewards:.4f},"
                f"SPS: {steps_per_sec:.2f}" # --- 添加 SPS 到 pbar ---
            )
        pbar.update(1)
        global_step += 1

    # Final evaluation and save
    print("🏁 Training completed!")
    eval_avg_return, eval_avg_length, eval_success_rate = evaluate()
    os.makedirs("models", exist_ok=True)
    final_save_path = f"models/{run_name}_final.pt"
    dclp.save(final_save_path)
    print(f"✅ Final evaluation return: {eval_avg_return:.2f}")
    print(f"✅ Final evaluation length: {eval_avg_length:.2f}")
    print(f"✅ Final evaluation success rate: {eval_success_rate:.2f}")
    print(f"💾 Model saved to: {final_save_path}")
    if args.use_wandb:
        wandb.log({"Evaluation/eval_avg_return": eval_avg_return, "Evaluation/eval_avg_length": eval_avg_length, "Evaluation/eval_success_rate": eval_success_rate}, step=global_step+1)
        wandb.finish()


if __name__ == "__main__":
    main()