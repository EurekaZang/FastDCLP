#!/usr/bin/env python3
"""
train_dclp_isaac_simple.py - Simplified training script for DCLP with IsaacLab

A simplified version focused on DCLP training with Isaac-Navigation-Flat-Turtlebot2-v0
"""

import os
import sys
import random
import time
import argparse
from collections import deque

import numpy as np
import torch
import gymnasium as gym

# Environment setup
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'environments'))

from dclp import DCLP
from fast_td3_utils import SimpleReplayBuffer


class SimpleReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        return {
            'state': np.array(states),
            'action': np.array(actions),
            'reward': np.array(rewards),
            'next_state': np.array(next_states),
            'done': np.array(dones)
        }
        
    def __len__(self):
        return len(self.buffer)


def train_dclp():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Isaac-Navigation-Flat-Turtlebot2-v0')
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learning_starts', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    try:
        env = gym.make(args.env_name, render_mode=None)
        print(f"Created environment: {args.env_name}")
    except Exception as e:
        print(f"Failed to create environment {args.env_name}: {e}")
        print("Using mock environment for demonstration")
        # 模拟环境用于测试
        class MockEnv:
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(280,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
                
            def reset(self, **kwargs):
                return np.random.randn(280), {}
                
            def step(self, action):
                obs = np.random.randn(280)
                reward = np.random.randn()
                done = np.random.rand() < 0.01  # 1% chance of episode end
                truncated = False
                return obs, reward, done, truncated, {}
                
        env = MockEnv()
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize DCLP algorithm
    dclp = DCLP(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.learning_rate,
        device=args.device
    )
    
    # Initialize replay buffer
    replay_buffer = SimpleReplayBuffer(capacity=args.buffer_size)
    
    # Training statistics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("Starting training...")
    for timestep in range(args.total_timesteps):
        # Select action
        if timestep < args.learning_starts:
            # Random action during initial exploration
            action = env.action_space.sample()
        else:
            action = dclp.get_action(state, deterministic=False)
            
        # Take step in environment
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done or truncated)
        
        state = next_state
        
        # Reset environment if episode is done
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
        # Train the model
        if (timestep >= args.learning_starts and 
            len(replay_buffer) >= args.batch_size and 
            timestep % args.train_freq == 0):
            
            batch = replay_buffer.sample(args.batch_size)
            train_metrics = dclp.train_step(batch)
            
            # Log training metrics periodically
            if timestep % 1000 == 0:
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0
                mean_length = np.mean(episode_lengths) if episode_lengths else 0
                print(f"Timestep {timestep:8d} | "
                      f"Mean Reward: {mean_reward:7.2f} | "
                      f"Mean Length: {mean_length:6.1f} | "
                      f"Actor Loss: {train_metrics['actor_loss']:.4f} | "
                      f"Critic Loss: {train_metrics['critic_loss']:.4f}")
                
        # Save model periodically
        if timestep > 0 and timestep % args.save_freq == 0:
            save_path = f"dclp_model_{timestep}.pt"
            dclp.save(save_path)
            print(f"Model saved to {save_path}")
            
        # Evaluation
        if timestep > 0 and timestep % args.eval_freq == 0:
            eval_rewards = []
            for _ in range(10):  # 10 evaluation episodes
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    eval_action = dclp.get_action(eval_state, deterministic=True)
                    eval_state, reward, eval_done, truncated, _ = env.step(eval_action)
                    eval_reward += reward
                    if truncated:
                        eval_done = True
                        
                eval_rewards.append(eval_reward)
                
            mean_eval_reward = np.mean(eval_rewards)
            print(f"Evaluation at timestep {timestep}: Mean reward = {mean_eval_reward:.2f}")
    
    # Final save
    final_save_path = "dclp_model_final.pt"
    dclp.save(final_save_path)
    print(f"Final model saved to {final_save_path}")
    print("Training completed!")


if __name__ == "__main__":
    train_dclp()
