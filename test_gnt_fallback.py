
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.algos.gnt import ConvGnT_for_ConvNet
from types import SimpleNamespace

def test_fallback_behavior():
    print("Testing Fallback Behavior...")
    
    # --- Scenario 1: Standard Mismatch (e.g. 3 vs 32) ---
    # Layer 1: Conv(3->32)
    # Layer 2: Conv(3->64) -- INCORRECT: Should be In=32
    # This simulates the "3 vs 32" mismatch comment
    
    l1 = nn.Conv2d(3, 32, 3)
    l2 = nn.Conv2d(3, 64, 3) # Mismatch!
    
    net = nn.Sequential(l1, l2)
    
    # Mock optimizer
    opt = SimpleNamespace(state={})
    
    # Initialize Algo
    # ConvGnT_for_ConvNet usually iterates over pairs (i*2, i*2+2) if sequential
    # But here we have 2 layers directly.
    # ConvGnT_for_ConvNet expects `net[i*2]` and `net[i*2+2]`.
    # Let's use a dummy list to simulate structure: [conv1, act, conv2]
    net_list = [l1, nn.ReLU(), l2]
    
    print("\nInitializing Algo with mismatched layers...")
    try:
        algo = ConvGnT_for_ConvNet(net_list, 'relu', opt, device='cpu', util_type='contribution')
    except Exception as e:
        print(f"Init failed: {e}")
        return

    # Check initial util size
    print(f"Initial Util shape for layer 0: {algo.util[0].shape}") # Should be 32 (l1.out_channels)
    
    # Create dummy feature
    # Feature from l1: [Batch, 32, H, W]
    feature = torch.randn(1, 32, 10, 10)
    
    print("\nRunning update_utility(0, feature)...")
    try:
        algo.update_utility(0, feature)
        print(">>> SUCCESS: update_utility ran without crashing.")
        print(f">>> Resulting Util shape: {algo.util[0].shape}")
        
        # Check if fallback logic was used
        # We can't easily check internal variable, but if it didn't crash 
        # with 32 vs 3, it means fallback triggered AND 3 broadcast to 32?
        # l2 weight shape: (64, 3, 3, 3).
        # output_weight_mag = l2.weight.abs().mean(dim=(0,2,3)) -> shape (3,)
        # diff_mean = (feature - ...).mean -> shape (32,)
        # 3 vs 32.
        # Fallback: new_util = output_weight_mag (shape 3)
        # util[0] += new_util.
        # 32 += 3.
        # This should crash unless PyTorch broadcasts 3 to 32? (No, 3 doesn't broadcast to 32).
        
    except RuntimeError as e:
        print(f">>> CRASHED as expected: {e}")
    except Exception as e:
        print(f">>> CRASHED with unexpected error: {e}")

if __name__ == "__main__":
    test_fallback_behavior()
