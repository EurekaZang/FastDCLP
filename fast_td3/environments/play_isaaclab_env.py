from typing import Optional, Union
import torch
import numpy as np
from .isaaclab_env import IsaacLabEnv


class PlayIsaacLabEnv(IsaacLabEnv):
    """
    Visualization wrapper for IsaacLab environments to evaluate and visualize trained FastTD3 models.
    This class extends IsaacLabEnv with additional functionality for model evaluation and performance visualization.
    """

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int = 1,  # Default to 1 for visualization
        seed: int = 42,
        action_bounds: Optional[float] = None,
        enable_viewport: bool = True,
        record_video: bool = False,
        video_path: str = "./videos/",
        max_episode_length: Optional[int] = None,
    ):
        """
        Initialize PlayIsaacLabEnv for model visualization.
        
        Args:
            task_name: Name of the IsaacLab task
            device: Device to run on ('cuda' or 'cpu')
            num_envs: Number of parallel environments (default: 1 for visualization)
            seed: Random seed for reproducibility
            action_bounds: Action bounds for clipping
            enable_viewport: Enable visual rendering
            record_video: Whether to record video of the episodes
            video_path: Path to save recorded videos
            max_episode_length: Override default episode length
        """
        # Initialize parent class with headless=False for visualization
        super().__init__(task_name, device, num_envs, seed, action_bounds)
        
        self.enable_viewport = enable_viewport
        self.record_video = record_video
        self.video_path = video_path
        
        if max_episode_length is not None:
            self.max_episode_steps = max_episode_length
            
        # Tracking variables for visualization
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Video recording setup
        if record_video:
            import os
            os.makedirs(video_path, exist_ok=True)
            self.video_frames = []
            self.episode_count = 0

    def reset(self, random_start_init: bool = False) -> torch.Tensor:
        """
        Reset environment and initialize episode tracking.
        
        Args:
            random_start_init: Whether to randomize episode start (default: False for consistent evaluation)
            
        Returns:
            Initial observation tensor
        """
        # Save previous episode statistics
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Save video if recording
            if self.record_video and self.video_frames:
                self._save_video()
                
        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        if self.record_video:
            self.video_frames = []
            
        return super().reset(random_start_init)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step environment and track episode statistics.
        
        Args:
            actions: Action tensor from the policy
            
        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        obs, rewards, dones, info = super().step(actions)
        
        # Update episode tracking
        self.current_episode_reward += rewards.sum().item()
        self.current_episode_length += 1
        
        # Capture frame for video recording
        if self.record_video and self.enable_viewport:
            try:
                frame = self.capture_frame()
                if frame is not None:
                    self.video_frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not capture frame: {e}")
        
        return obs, rewards, dones, info

    def play_episode(
        self, 
        actor_model, 
        deterministic: bool = True,
        obs_normalizer=None,
        verbose: bool = True
    ) -> dict:
        """
        Play a complete episode using the trained actor model.
        
        Args:
            actor_model: Trained FastTD3 actor model
            deterministic: Whether to use deterministic actions
            obs_normalizer: Observation normalizer (if used during training)
            verbose: Whether to print episode statistics
            
        Returns:
            Dictionary containing episode statistics
        """
        obs = self.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        actor_model.eval()
        
        with torch.no_grad():
            while not done and episode_length < self.max_episode_steps:
                # Normalize observation if normalizer is provided
                if obs_normalizer is not None:
                    normalized_obs = obs_normalizer(obs)
                else:
                    normalized_obs = obs
                
                # Get action from actor
                action = actor_model.explore(normalized_obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, dones, info = self.step(action)
                
                episode_reward += reward.sum().item()
                episode_length += 1
                done = dones.any().item()
                
                if verbose and episode_length % 100 == 0:
                    print(f"Step {episode_length}, Reward: {episode_reward:.2f}")
        
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': episode_reward > 0,  # Adjust success criteria as needed
        }
        
        if verbose:
            print(f"Episode completed: Reward={episode_reward:.2f}, Length={episode_length}")
            
        return episode_stats

    def evaluate_model(
        self, 
        actor_model, 
        num_episodes: int = 10,
        obs_normalizer=None,
        deterministic: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the trained model over multiple episodes.
        
        Args:
            actor_model: Trained FastTD3 actor model
            num_episodes: Number of episodes to evaluate
            obs_normalizer: Observation normalizer (if used during training)
            deterministic: Whether to use deterministic actions
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            if verbose:
                print(f"Playing episode {episode + 1}/{num_episodes}")
                
            stats = self.play_episode(
                actor_model, 
                deterministic=deterministic,
                obs_normalizer=obs_normalizer,
                verbose=False
            )
            
            episode_rewards.append(stats['episode_reward'])
            episode_lengths.append(stats['episode_length'])
            
            if stats['success']:
                success_count += 1
                
        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Episodes: {num_episodes}")
            print(f"Mean Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
            print(f"Mean Length: {evaluation_results['mean_length']:.1f} ± {evaluation_results['std_length']:.1f}")
            print(f"Success Rate: {evaluation_results['success_rate']:.2%}")
            print("="*50)
            
        return evaluation_results

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the environment for video recording.
        
        Returns:
            RGB frame as numpy array or None if capture fails
        """
        try:
            # This method depends on IsaacLab's rendering capabilities
            # You may need to adjust based on the specific IsaacLab version
            if hasattr(self.envs.unwrapped, 'render'):
                frame = self.envs.unwrapped.render(mode='rgb_array')
                return frame
            else:
                # Alternative method - try to get frame from simulator
                return None
        except Exception as e:
            print(f"Frame capture failed: {e}")
            return None

    def _save_video(self):
        """Save recorded frames as video."""
        if not self.video_frames:
            return
            
        try:
            import cv2
            video_filename = f"{self.video_path}/episode_{self.episode_count:03d}.mp4"
            
            # Get frame dimensions
            height, width = self.video_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Adjust FPS as needed
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            
            # Write frames
            for frame in self.video_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                
            video_writer.release()
            print(f"Video saved: {video_filename}")
            
        except ImportError:
            print("OpenCV not available for video recording. Install with: pip install opencv-python")
        except Exception as e:
            print(f"Video saving failed: {e}")
            
        finally:
            self.episode_count += 1

    def get_statistics(self) -> dict:
        """
        Get accumulated episode statistics.
        
        Returns:
            Dictionary containing episode statistics
        """
        if not self.episode_rewards:
            return {"message": "No episodes completed yet"}
            
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
        }

    def render(self):
        """Enable rendering for visualization."""
        if not self.enable_viewport:
            print("Warning: Viewport rendering is disabled")
            return
            
        try:
            # This depends on IsaacLab's rendering implementation
            if hasattr(self.envs.unwrapped, 'render'):
                self.envs.unwrapped.render()
        except Exception as e:
            print(f"Rendering error: {e}")


def load_trained_model(model_path: str, actor_class, device: str = 'cuda'):
    """
    Load a trained FastTD3 model for evaluation.
    
    Args:
        model_path: Path to the saved model file
        actor_class: Actor class (from fast_td3.py)
        device: Device to load model on
        
    Returns:
        Loaded actor model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters from checkpoint
    actor_state_dict = checkpoint.get('actor_state_dict')
    if actor_state_dict is None:
        # Try alternative key names
        for key in ['actor', 'policy', 'model']:
            if key in checkpoint:
                actor_state_dict = checkpoint[key]
                break
    
    if actor_state_dict is None:
        raise ValueError(f"Could not find actor state dict in checkpoint: {list(checkpoint.keys())}")
    
    # You'll need to recreate the actor with the same parameters used during training
    # This information should be stored in the checkpoint
    actor_kwargs = checkpoint.get('actor_kwargs', {})
    
    actor = actor_class(**actor_kwargs)
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    
    return actor


# Example usage function
def example_usage():
    """
    Example of how to use PlayIsaacLabEnv for model evaluation.
    """
    # Import necessary classes
    from fast_td3.fast_td3 import Actor
    
    # Create visualization environment
    play_env = PlayIsaacLabEnv(
        task_name="Isaac-Cartpole-v0",  # Replace with your task
        device="cuda",
        num_envs=1,
        enable_viewport=True,
        record_video=True,
        video_path="./evaluation_videos/"
    )
    
    # Load trained model
    model_path = "models/your_trained_model.pt"  # Replace with your model path
    actor = load_trained_model(model_path, Actor, device="cuda")
    
    # Evaluate model
    results = play_env.evaluate_model(
        actor_model=actor,
        num_episodes=5,
        deterministic=True,
        verbose=True
    )
    
    # Play single episode with visualization
    episode_stats = play_env.play_episode(
        actor_model=actor,
        deterministic=True,
        verbose=True
    )
    
    print("Evaluation completed!")
    print(f"Results: {results}")
