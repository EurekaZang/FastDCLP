# PlayIsaacLabEnv - Model Evaluation Guide

This guide explains how to use the `PlayIsaacLabEnv` class to evaluate and visualize your trained FastTD3 models.

## Features

- **Model Evaluation**: Test trained FastTD3 models over multiple episodes
- **Performance Visualization**: Real-time rendering and performance metrics
- **Video Recording**: Capture episodes as MP4 videos
- **Statistical Analysis**: Comprehensive performance statistics
- **Comparison Tools**: Compare trained models with random policies

## Quick Start

### 1. Basic Usage

```python
from fast_td3.environments.play_isaaclab_env import PlayIsaacLabEnv
from fast_td3.fast_td3 import Actor

# Create evaluation environment
play_env = PlayIsaacLabEnv(
    task_name="Isaac-Cartpole-v0",
    device="cuda",
    num_envs=1,
    enable_viewport=True,
    record_video=True
)

# Load your trained model (you'll need to implement this based on your checkpoint format)
actor = load_your_trained_model("path/to/model.pt")

# Evaluate model
results = play_env.evaluate_model(
    actor_model=actor,
    num_episodes=10,
    deterministic=True
)

print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Success rate: {results['success_rate']:.2%}")
```

### 2. Using the Command Line Script

```bash
# Basic evaluation
python fast_td3/environments/evaluate_model.py \
    --model_path models/your_model.pt \
    --task_name Isaac-Cartpole-v0 \
    --num_episodes 10

# With video recording
python fast_td3/environments/evaluate_model.py \
    --model_path models/your_model.pt \
    --task_name Isaac-Cartpole-v0 \
    --num_episodes 5 \
    --record_video \
    --video_path ./videos/

# Headless evaluation (no GUI)
python fast_td3/environments/evaluate_model.py \
    --model_path models/your_model.pt \
    --task_name Isaac-Cartpole-v0 \
    --num_episodes 10 \
    --headless
```

### 3. Using Jupyter Notebook

Open `model_evaluation_notebook.ipynb` for an interactive evaluation experience with visualization plots.

## Class Reference

### PlayIsaacLabEnv

#### Constructor Parameters

- `task_name` (str): IsaacLab task name (e.g., "Isaac-Cartpole-v0")
- `device` (str): Device to run on ("cuda" or "cpu")
- `num_envs` (int): Number of parallel environments (default: 1)
- `seed` (int): Random seed for reproducibility
- `action_bounds` (Optional[float]): Action bounds for clipping
- `enable_viewport` (bool): Enable visual rendering (default: True)
- `record_video` (bool): Record episodes as videos (default: False)
- `video_path` (str): Path to save videos (default: "./videos/")
- `max_episode_length` (Optional[int]): Override default episode length

#### Key Methods

##### `play_episode(actor_model, deterministic=True, obs_normalizer=None, verbose=True)`

Play a single episode and return statistics.

**Parameters:**
- `actor_model`: Trained FastTD3 actor
- `deterministic`: Use deterministic policy (no exploration noise)
- `obs_normalizer`: Observation normalizer from training
- `verbose`: Print progress information

**Returns:** Dictionary with episode statistics

##### `evaluate_model(actor_model, num_episodes=10, obs_normalizer=None, deterministic=True, verbose=True)`

Evaluate model over multiple episodes.

**Returns:** Dictionary with comprehensive evaluation statistics:
- `mean_reward`: Average episode reward
- `std_reward`: Standard deviation of rewards
- `mean_length`: Average episode length
- `success_rate`: Percentage of successful episodes
- `episode_rewards`: List of all episode rewards
- `episode_lengths`: List of all episode lengths

##### `get_statistics()`

Get accumulated statistics from all played episodes.

## Model Loading

You'll need to implement model loading based on your checkpoint format. Here's a template:

```python
def load_trained_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract actor parameters (adjust based on your checkpoint structure)
    actor_kwargs = {
        'n_obs': 1090,  # Your observation space size
        'n_act': 2,     # Your action space size
        'num_envs': 1,
        'init_scale': 0.1,
        'hidden_dim': 256,
        'device': device,
    }
    
    actor = Actor(**actor_kwargs)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.to(device)
    actor.eval()
    
    # Load normalizers if available
    obs_normalizer = None
    if 'obs_normalizer' in checkpoint:
        obs_normalizer = EmpiricalNormalization(shape=actor_kwargs['n_obs'], device=device)
        obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
        obs_normalizer.eval()
    
    return actor, obs_normalizer
```

## Common Issues and Solutions

### 1. Import Errors

If you get import errors for IsaacLab:
```bash
# Make sure IsaacLab is properly installed
pip install isaaclab

# Or if using conda:
conda install -c conda-forge isaaclab
```

### 2. CUDA Issues

If CUDA is not available:
```python
# The environment will automatically fall back to CPU
play_env = PlayIsaacLabEnv(
    task_name="your-task",
    device="cpu",  # Explicitly use CPU
    # ... other parameters
)
```

### 3. Model Loading Issues

Make sure your model checkpoint contains the necessary components:
- Actor state dict
- Observation normalizer (if used during training)
- Model architecture parameters

### 4. Video Recording Issues

If video recording fails:
```bash
# Install OpenCV
pip install opencv-python

# Make sure the video directory exists and is writable
mkdir -p ./videos/
chmod 755 ./videos/
```

## Performance Tips

1. **Use deterministic evaluation** for consistent results
2. **Set num_envs=1** for evaluation (multiple environments not needed)
3. **Enable headless mode** for faster evaluation without rendering
4. **Use GPU** if available for faster inference
5. **Record videos selectively** as it can slow down evaluation

## Example Evaluation Workflow

```python
# 1. Load model and normalizers
actor, obs_normalizer = load_trained_model("model.pt", "cuda")

# 2. Create evaluation environment
play_env = PlayIsaacLabEnv(
    task_name="Isaac-Cartpole-v0",
    device="cuda",
    record_video=True
)

# 3. Quick single episode test
episode_stats = play_env.play_episode(actor, obs_normalizer=obs_normalizer)
print(f"Test episode reward: {episode_stats['episode_reward']}")

# 4. Full evaluation
results = play_env.evaluate_model(
    actor, 
    num_episodes=20, 
    obs_normalizer=obs_normalizer
)

# 5. Analyze results
print(f"Mean performance: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Success rate: {results['success_rate']:.2%}")

# 6. Compare with random baseline
random_policy = RandomPolicy(play_env.num_actions, "cuda")
random_results = play_env.evaluate_model(random_policy, num_episodes=10)
improvement = results['mean_reward'] - random_results['mean_reward']
print(f"Improvement over random: {improvement:.2f}")
```

## Customization

You can extend `PlayIsaacLabEnv` for specific needs:

```python
class CustomPlayEnv(PlayIsaacLabEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom initialization
    
    def custom_analysis(self, results):
        # Add custom analysis methods
        pass
    
    def save_custom_metrics(self, episode_data):
        # Save additional metrics
        pass
```

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your model checkpoint format
3. Ensure IsaacLab tasks are properly configured
4. Check device compatibility (CUDA/CPU)

For more help, refer to the FastTD3 and IsaacLab documentation.
