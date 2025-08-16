#!/usr/bin/env python3
"""
Test script to check model loading and architecture detection.
"""

import sys
import os
import torch

# Add FastTD3 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_td3 import Actor, MultiTaskActor
from fast_td3_simbav2 import Actor as SimbaV2Actor, MultiTaskActor as SimbaV2MultiTaskActor
from fast_td3_utils import EmpiricalNormalization


def test_model_loading(model_path: str):
    """Test loading the model and detect architecture."""
    
    print(f"Testing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"✅ Checkpoint loaded successfully")
    print(f"Available keys: {list(checkpoint.keys())}")
    
    # Get actor state dict
    actor_state_dict = checkpoint.get('actor_state_dict')
    if actor_state_dict is None:
        print("❌ No actor_state_dict found")
        return
    
    print(f"Actor state dict keys: {list(actor_state_dict.keys())[:10]}...")  # Show first 10 keys
    
    # Detect architecture
    is_simbav2 = any('embedder' in key or 'encoder' in key or 'predictor' in key 
                     for key in actor_state_dict.keys())
    
    print(f"Architecture detected: {'SimbaV2' if is_simbav2 else 'Standard FastTD3'}")
    
    # Get args
    args_dict = checkpoint.get('args', {})
    print(f"Training args available: {list(args_dict.keys())}")
    
    # Try to determine dimensions
    n_obs = args_dict.get('n_obs', 'Unknown')
    n_act = args_dict.get('n_act', 'Unknown')
    print(f"Observation dim: {n_obs}")
    print(f"Action dim: {n_act}")
    
    # Check noise_scales shape
    noise_scales = actor_state_dict.get('noise_scales')
    if noise_scales is not None:
        print(f"Noise scales shape: {noise_scales.shape}")
        original_num_envs = noise_scales.shape[0]
        print(f"Original number of environments: {original_num_envs}")
    
    print("✅ Model analysis completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_path>")
        print("Example: python test_model.py /home/unnc/FastTD3/models/Isaac-Navigation-Flat-Turtlebot2-v0__FastTD3__1_50000.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_model_loading(model_path)
