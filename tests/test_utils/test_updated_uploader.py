#!/usr/bin/env python3
"""
Quick test script to verify the updated wandb_rank_dynamics_uploader functions work correctly.
This tests the new input-aware rank dynamics features.
"""

import sys
from pathlib import Path

# Add the project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.wandb_rank_dynamics_uploader import (
    compute_theoretical_max_rank_fast,
    compute_dynamics_for_step
)
from src.utils.zeroth_order_features import compute_rank_decay_dynamics

def test_theoretical_max_rank_computation():
    """Test the fast theoretical max rank computation."""
    print("🧪 Testing theoretical max rank computation...")
    
    # Test cases
    test_cases = [
        (512, 32, 32),    # first_feature_dim=512, batch_size=32, expected=32
        (128, 64, 64),    # first_feature_dim=128, batch_size=64, expected=64  
        (256, 512, 256),  # first_feature_dim=256, batch_size=512, expected=256
        (1024, 128, 128), # first_feature_dim=1024, batch_size=128, expected=128
    ]
    
    for first_feature_dim, batch_size, expected in test_cases:
        result = compute_theoretical_max_rank_fast(first_feature_dim, batch_size)
        print(f"  📊 first_feature_dim={first_feature_dim}, batch_size={batch_size} -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("  ✅ All theoretical max rank tests passed!")

def test_dynamics_computation():
    """Test the updated dynamics computation function."""
    print("🧪 Testing updated dynamics computation...")
    
    # Create mock rank data (simulating rank decay across layers)
    ranks_by_type = {
        'effective_rank': [32.0, 28.5, 24.2, 19.8, 15.3, 12.1, 8.7, 5.2],
        'numerical_rank': [32.0, 30.1, 27.8, 25.2, 22.1, 18.9, 15.4, 11.8]
    }
    
    # Test with input-aware dynamics
    first_feature_dim = 512
    batch_size = 32
    
    print(f"  📊 Testing with first_feature_dim={first_feature_dim}, batch_size={batch_size}")
    
    dynamics = compute_dynamics_for_step(
        ranks_by_type=ranks_by_type,
        analysis_mode='difference',
        first_feature_dim=first_feature_dim,
        batch_size=batch_size
    )
    
    print(f"  📊 Computed dynamics keys: {list(dynamics.keys())}")
    
    # Check expected keys are present
    expected_keys = [
        'effective_rank_rank_drop_gini',
        'effective_rank_rank_decay_centroid', 
        'effective_rank_normalized_aurc',
        'numerical_rank_rank_drop_gini',
        'numerical_rank_rank_decay_centroid',
        'numerical_rank_normalized_aurc'
    ]
    
    for key in expected_keys:
        assert key in dynamics, f"Missing expected key: {key}"
        print(f"    ✅ {key}: {dynamics[key]:.4f}")
    
    print("  ✅ Input-aware dynamics computation test passed!")
    
    # Test fallback to original behavior
    print(f"  📊 Testing fallback to original dynamics (no input info)")
    
    dynamics_fallback = compute_dynamics_for_step(
        ranks_by_type=ranks_by_type,
        analysis_mode='difference',
        first_feature_dim=None,
        batch_size=None
    )
    
    print(f"  📊 Fallback dynamics keys: {list(dynamics_fallback.keys())}")
    assert len(dynamics_fallback) == len(dynamics), "Fallback should produce same number of metrics"
    
    print("  ✅ Fallback dynamics computation test passed!")

def test_direct_rank_dynamics():
    """Test the updated compute_rank_decay_dynamics function directly."""
    print("🧪 Testing updated compute_rank_decay_dynamics function...")
    
    # Test rank sequence
    ranks = [32.0, 28.5, 24.2, 19.8, 15.3, 12.1, 8.7, 5.2]
    
    # Test with input-aware dynamics (new functionality)
    print("  📊 Testing input-aware dynamics (decay_from_input=True)")
    
    dynamics_input_aware = compute_rank_decay_dynamics(
        ranks,
        mode='difference',
        decay_from_input=True,
        assume_full_rank_input=True,
        input_rank=32
    )
    
    print(f"    📊 Input-aware results:")
    for key, value in dynamics_input_aware.items():
        print(f"      {key}: {value:.4f}")
    
    # Test original behavior (for comparison)
    print("  📊 Testing original dynamics (decay_from_input=False)")
    
    dynamics_original = compute_rank_decay_dynamics(
        ranks,
        mode='difference',
        decay_from_input=False
    )
    
    print(f"    📊 Original results:")
    for key, value in dynamics_original.items():
        print(f"      {key}: {value:.4f}")
    
    print("  ✅ Direct rank dynamics tests passed!")

def main():
    """Run all tests."""
    print("🚀 Testing updated wandb_rank_dynamics_uploader functions...")
    print("="*60)
    
    try:
        test_theoretical_max_rank_computation()
        print()
        
        test_dynamics_computation()
        print()
        
        test_direct_rank_dynamics()
        print()
        
        print("🎉 All tests passed! The updated functions are working correctly.")
        print("✅ Ready to use input-aware rank dynamics with global_epoch approach!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"📊 Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
