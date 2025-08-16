#!/usr/bin/env python3
"""
Script to inspect model args in detail.
"""

import sys
import os
import torch

# Add FastTD3 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_model_args(model_path: str):
    """Inspect all model arguments."""
    
    print(f"Inspecting model: {model_path}")
    
    # Load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get args
    args_dict = checkpoint.get('args', {})
    
    print("\n=== All Training Arguments ===")
    for key, value in sorted(args_dict.items()):
        print(f"{key}: {value}")
    
    # Try to infer dimensions from state dict
    actor_state_dict = checkpoint.get('actor_state_dict', {})
    
    print("\n=== Inferred Dimensions ===")
    
    # Look for embedder weight to infer input dimension
    if 'embedder.w.w.weight' in actor_state_dict:
        embedder_weight = actor_state_dict['embedder.w.w.weight']
        n_obs = embedder_weight.shape[1]  # Input dimension
        print(f"Observation dimension (from embedder): {n_obs}")
    
    # Look for predictor to infer action dimension
    if 'predictor.mean_w2.w.weight' in actor_state_dict:
        predictor_weight = actor_state_dict['predictor.mean_w2.w.weight']
        n_act = predictor_weight.shape[0]  # Output dimension
        print(f"Action dimension (from predictor): {n_act}")
    elif 'predictor.mean_bias' in actor_state_dict:
        predictor_bias = actor_state_dict['predictor.mean_bias']
        n_act = predictor_bias.shape[0]  # Output dimension
        print(f"Action dimension (from predictor bias): {n_act}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_args.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_model_args(model_path)
