#!/usr/bin/env python3
"""
Script to analyze SimbaV2 model structure in detail.
"""

import sys
import os
import torch

# Add FastTD3 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_simbav2_structure(model_path: str):
    """Analyze SimbaV2 model structure in detail."""
    
    print(f"Analyzing SimbaV2 model: {model_path}")
    
    # Load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get args and actor state dict
    args_dict = checkpoint.get('args', {})
    actor_state_dict = checkpoint.get('actor_state_dict', {})
    
    print("\n=== Model Architecture Analysis ===")
    
    # Analyze embedder
    if 'embedder.w.w.weight' in actor_state_dict:
        embedder_weight = actor_state_dict['embedder.w.w.weight']
        n_obs = embedder_weight.shape[1]  # Input dimension
        hidden_dim = embedder_weight.shape[0]  # Hidden dimension
        print(f"Embedder: {n_obs} -> {hidden_dim}")
    
    # Analyze encoder blocks
    encoder_keys = [k for k in actor_state_dict.keys() if k.startswith('encoder.')]
    num_blocks = len(set(k.split('.')[1] for k in encoder_keys if k.split('.')[1].isdigit()))
    print(f"Number of encoder blocks: {num_blocks}")
    
    # Analyze first encoder block
    if 'encoder.0.mlp.w1.w.weight' in actor_state_dict:
        w1_weight = actor_state_dict['encoder.0.mlp.w1.w.weight']
        w1_input = w1_weight.shape[1]  # Input to w1
        w1_output = w1_weight.shape[0]  # Output from w1
        print(f"Encoder block 0 MLP w1: {w1_input} -> {w1_output}")
        
        # This tells us the expansion factor
        expansion = w1_output // w1_input
        print(f"Expansion factor: {expansion}")
    
    if 'encoder.0.mlp.w2.w.weight' in actor_state_dict:
        w2_weight = actor_state_dict['encoder.0.mlp.w2.w.weight']
        w2_input = w2_weight.shape[1]  # Input to w2
        w2_output = w2_weight.shape[0]  # Output from w2
        print(f"Encoder block 0 MLP w2: {w2_input} -> {w2_output}")
    
    # Analyze predictor
    if 'predictor.mean_bias' in actor_state_dict:
        predictor_bias = actor_state_dict['predictor.mean_bias']
        n_act = predictor_bias.shape[0]  # Output dimension
        print(f"Predictor output dimension (actions): {n_act}")
    
    # Extract all relevant hyperparameters
    print("\n=== Inferred Hyperparameters ===")
    print(f"n_obs: {n_obs}")
    print(f"n_act: {n_act}")
    print(f"hidden_dim: {hidden_dim}")
    print(f"expansion: {expansion}")
    print(f"num_blocks: {num_blocks}")
    
    # From args
    print(f"actor_hidden_dim (from args): {args_dict.get('actor_hidden_dim', 'Not found')}")
    print(f"actor_num_blocks (from args): {args_dict.get('actor_num_blocks', 'Not found')}")
    print(f"scaler_init (from args): {args_dict.get('scaler_init', 'Not found')}")
    print(f"scaler_scale (from args): {args_dict.get('scaler_scale', 'Not found')}")
    print(f"alpha_init (from args): {args_dict.get('alpha_init', 'Not found')}")
    print(f"alpha_scale (from args): {args_dict.get('alpha_scale', 'Not found')}")
    print(f"c_shift (from args): {args_dict.get('c_shift', 'Not found')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_simbav2.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    analyze_simbav2_structure(model_path)
