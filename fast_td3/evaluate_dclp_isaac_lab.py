#!/usr/bin/env python3

"""usage:
python ./fast_td3/evaluate_dclp_isaac_lab.py --env_name Isaac-Navigation-Flat-Jackal-v0 --num_envs 256 --eval_episodes 10 --model_path ./models/Isaac-Navigation-Flat-Jackal-v0__try_training_with_potential_reward__42_final.pt --e
xp_name post_eval_try_training_with_potential_reward  --no-headless --cuda --device_rank 0 --out_dir e
val_results  --use_wandb
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro
import tqdm
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_td3_utils import EmpiricalNormalization
from environments.isaaclab_env import IsaacLabEnv
from dclp import DCLP
from dclp_utils import DCLPArgs

torch.set_float32_matmul_precision("high")


@dataclass
class DCLPEvalArgs(DCLPArgs):
    model_path: str = "models/Isaac-Navigation-Flat-Turtlebot2-v0__dclp_training__42_2800000.pt"
    eval_episodes: int = 5
    max_eval_steps: Optional[int] = None
    deterministic: bool = True
    out_dir: str = "eval_results"
    exp_name: str = "dclp_evaluation"


def main():
    args = tyro.cli(DCLPEvalArgs)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = False
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if args.use_wandb:
        wandb.init(project=args.project, name=run_name, config=vars(args), save_code=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = IsaacLabEnv(
        task_name=args.env_name,
        device=str(device),
        num_envs=args.num_envs,
        seed=args.seed,
        action_bounds=args.action_bounds,
        render_mode=args.render_mode,
        headless=args.headless,
    )

    n_obs = envs.num_obs
    n_act = envs.num_actions

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = torch.nn.Identity()

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
        device=device,
    )

    dclp.load(args.model_path)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    eval_returns = []
    eval_lengths = []
    episode_rewards_series = []
    episode_actions_series = []
    eval_successes = 0

    num_eval_episodes = max(1, args.eval_episodes)
    eval_pbar = tqdm.tqdm(range(num_eval_episodes), desc="Evaluation", leave=False)
    for ep in eval_pbar:
        obs = envs.reset(random_start_init=False)
        ep_return = 0.0
        ep_len = 0
        ep_success = False
        rewards_trace = []
        actions_trace = {"linear": [], "angular": []}
        max_steps = args.max_eval_steps if args.max_eval_steps is not None else min(1000, envs.max_episode_steps)
        with torch.no_grad():
            for step in range(max_steps):
                if args.obs_normalization:
                    norm_obs = obs_normalizer(obs, update=False)
                else:
                    norm_obs = obs
                actions = dclp.get_action(norm_obs, deterministic=args.deterministic)
                obs, reward, done, info = envs.step(actions)
                r = reward.mean().item()
                ep_return += r
                ep_len += 1
                rewards_trace.append(r)
                actions_trace["linear"].append((actions[0][0] * 10).item())
                actions_trace["angular"].append((actions[0][1] * 10).item())
                success_signal = False
                if isinstance(info, dict):
                    if "successes" in info:
                        v = info["successes"]
                        success_signal = bool(v.any().item()) if torch.is_tensor(v) else bool(v)
                    elif "is_success" in info:
                        v = info["is_success"]
                        success_signal = bool(v.any().item()) if torch.is_tensor(v) else bool(v)
                    elif "extras" in info and isinstance(info["extras"], dict):
                        ex = info["extras"]
                        if "successes" in ex:
                            v = ex["successes"]
                            success_signal = bool(v.any().item()) if torch.is_tensor(v) else bool(v)
                        elif "is_success" in ex:
                            v = ex["is_success"]
                            success_signal = bool(v.any().item()) if torch.is_tensor(v) else bool(v)
                if done[0] and (success_signal or r > 0.0):
                    ep_success = True
                if done[0]:
                    break
        eval_returns.append(ep_return)
        eval_lengths.append(ep_len)
        eval_successes += int(ep_success)
        episode_rewards_series.append(rewards_trace)
        episode_actions_series.append(actions_trace)
        sr = eval_successes / (ep + 1)
        eval_pbar.set_postfix({"Return": f"{ep_return:.2f}", "Len": ep_len, "SR": f"{sr:.2f}"})
    eval_pbar.close()

    mean_return = float(np.mean(eval_returns)) if len(eval_returns) > 0 else 0.0
    mean_length = float(np.mean(eval_lengths)) if len(eval_lengths) > 0 else 0.0
    success_rate = eval_successes / max(1, num_eval_episodes)

    if args.use_wandb:
        wandb.log({"eval/mean_return": mean_return, "eval/mean_length": mean_length, "eval/success_rate": success_rate})

    returns_fig = plt.figure(figsize=(6, 4))
    plt.plot(eval_returns, marker="o")
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    returns_path = os.path.join(out_dir, "returns.png")
    plt.tight_layout()
    plt.savefig(returns_path)
    plt.close(returns_fig)

    lengths_fig = plt.figure(figsize=(6, 4))
    plt.plot(eval_lengths, marker="o")
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    lengths_path = os.path.join(out_dir, "lengths.png")
    plt.tight_layout()
    plt.savefig(lengths_path)
    plt.close(lengths_fig)

    for i, rewards_trace in enumerate(episode_rewards_series):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(rewards_trace)
        plt.title(f"Rewards Episode {i+1}")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        p = os.path.join(out_dir, f"rewards_episode_{i+1}.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close(fig)

    for i, actions_trace in enumerate(episode_actions_series):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(actions_trace["linear"], label="linear")
        plt.plot(actions_trace["angular"], label="angular")
        plt.legend()
        plt.title(f"Actions Episode {i+1}")
        plt.xlabel("Step")
        plt.ylabel("Scaled Action")
        p = os.path.join(out_dir, f"actions_episode_{i+1}.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close(fig)

    if args.use_wandb:
        wandb.log({
            "eval/returns_image": wandb.Image(returns_path),
            "eval/lengths_image": wandb.Image(lengths_path),
        })
        for i in range(len(episode_rewards_series)):
            wandb.log({
                f"eval/rewards_episode_{i+1}": wandb.Image(os.path.join(out_dir, f"rewards_episode_{i+1}.png")),
                f"eval/actions_episode_{i+1}": wandb.Image(os.path.join(out_dir, f"actions_episode_{i+1}.png")),
            })

    print(f"Mean Return: {mean_return:.2f}, Mean Length: {mean_length:.1f}, Success Rate: {success_rate:.2f}")
    print(f"Saved figures to: {out_dir}")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()