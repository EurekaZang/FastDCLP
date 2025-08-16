#!/usr/bin/env python3
"""
test_dclp.py - Test script to verify DCLP module functionality
"""

import os
import sys
import numpy as np
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '../..', 'DCLP'))

try:
    # Test basic PyTorch functionality
    print("Testing PyTorch...")
    x = torch.randn(2, 3)
    print(f"PyTorch tensor created: {x.shape}")
    
    # Test DCLP module imports
    print("\nTesting DCLP imports...")
    
    # First try importing the utility functions
    from dclp_utils import CNNNet, CNNDense, MLP
    print("✓ Imported utility classes successfully")
    
    # Now try the main DCLP module
    from dclp import MLPGaussianPolicy, MLPActorCritic, DCLP
    print("✓ Imported DCLP classes successfully")
    
    # Test creating DCLP instance
    print("\nTesting DCLP instantiation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dclp = DCLP(state_dim=280, action_dim=2, device='cpu')
    print("✓ DCLP instance created successfully")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_state = np.random.randn(280)
    action = dclp.get_action(test_state, deterministic=False)
    print(f"✓ Action generated: {action}")
    
    print("\nAll tests passed! ✅")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
