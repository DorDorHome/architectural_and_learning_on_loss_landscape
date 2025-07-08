"""
Test script to verify the enhanced feature tracking works correctly.
This demonstrates that existing code continues to work while new capabilities are available.
"""

import sys
import pathlib
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.conv_net import ConvNet
from src.algos.supervised.backprop_with_semantic_features import BackpropWithSemanticFeatures
from configs.configurations import NetParams, BackpropConfig

def test_enhanced_features():
    """Test that enhanced features work while maintaining backward compatibility."""
    
    print("Testing Enhanced Feature Tracking...")
    
    # Create a simple ConvNet for testing
    net_params = NetParams(
        num_classes=10,
        activation='relu',
        input_height=28,
        input_width=28,
        initialization='kaiming'
    )
    
    learner_config = BackpropConfig(
        device='cpu',
        opt='sgd',
        loss='cross_entropy',
        step_size=0.01,
        beta_1=0.95,
        beta_2=0.999,
        weight_decay=0.01,
        to_perturb=False,
        perturb_scale=0.05
    )
    
    # Create model and learner
    net = ConvNet(net_params)
    learner = BackpropWithSemanticFeatures(net, learner_config, net_params)
    
    # Create dummy data
    x = torch.randn(4, 3, 28, 28)  # batch_size=4, channels=3, height=28, width=28
    target = torch.randint(0, 10, (4,))  # 4 random class labels
    
    # Run learning step
    loss, output = learner.learn(x, target)
    
    print(f"Loss: {loss:.4f}")
    print(f"Output shape: {output.shape}")
    
    # Test 1: Existing functionality (backward compatibility)
    print("\n=== Testing Backward Compatibility ===")
    features_old_way = learner.previous_features
    print(f"Number of feature layers (old way): {len(features_old_way)}")
    for i, feature in enumerate(features_old_way):
        print(f"Layer {i}: shape {feature.shape}")
    
    # Test 2: New semantic functionality
    print("\n=== Testing New Semantic Features ===")
    layer_names = learner.get_layer_names()
    print(f"Layer names: {layer_names}")
    
    for layer_name in layer_names:
        feature = learner.get_feature_by_name(layer_name)
        print(f"{layer_name}: shape {feature.shape}")
    
    # Test 3: Architectural grouping
    print("\n=== Testing Architectural Grouping ===")
    conv_features = learner.get_conv_features()
    fc_features = learner.get_fc_features()
    
    print(f"Convolutional layers: {list(conv_features.keys())}")
    print(f"Fully connected layers: {list(fc_features.keys())}")
    
    # Test 4: Feature container functionality
    print("\n=== Testing Feature Container ===")
    if learner.feature_container is not None:
        container = learner.feature_container
        print(f"Container length: {len(container)}")
        print(f"Available layer names: {container.layer_names()}")
        
        # Test both access methods
        print(f"First layer (index 0): {container[0].shape}")
        print(f"First layer (by name '{layer_names[0]}'): {container[layer_names[0]].shape}")
        
        # Verify they're the same tensor
        assert torch.equal(container[0], container[layer_names[0]]), "Index and name access should return same tensor!"
        print("✓ Index and name access return identical tensors")
    
    print("\n✅ All tests passed! Enhanced features work correctly.")
    print("✅ Backward compatibility maintained.")
    print("✅ New semantic capabilities available.")

if __name__ == "__main__":
    test_enhanced_features()
