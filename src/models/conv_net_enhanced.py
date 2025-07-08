"""
Example showing how to modify ConvNet to use FeatureContainer
while maintaining backward compatibility.
"""

import torch.nn as nn
import torch
from configs.configurations import NetParams
from src.utils.feature_container import FeatureContainer

activation_dict = {
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
    'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU
}

class ConvNetWithFeatureNames(nn.Module):
    """
    Enhanced ConvNet that returns features with semantic names.
    Backward compatible with existing list-based code.
    """
    
    def __init__(self, config: NetParams):
        super().__init__()
        num_classes = config.num_classes
        self.activation = config.activation
        
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute flattened size
        # by using a dummy input to test run
        dummy = torch.zeros(1, 3, config.input_height, config.input_width)
        x = self.pool(nn.ReLU()(self.conv1(dummy)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        flattened_size = x.view(1, -1).shape[1]
        
        # FC layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.activation_fn = activation_dict.get(self.activation, nn.ReLU)()
        
        # Define feature names for better semantics
        self.feature_names = [
            'conv1_pooled',      # After conv1 + activation + pooling
            'conv2_pooled',      # After conv2 + activation + pooling  
            'conv3_flattened',   # After conv3 + activation + pooling + flattening
            'fc1_output',        # After fc1 + activation
            'fc2_output'         # After fc2 + activation
        ]
        
        # Layers for compatibility with existing perturb() method
        self.layers = nn.ModuleList([
            self.conv1, self.activation_fn,
            self.conv2, self.activation_fn,
            self.conv3, self.activation_fn,
            self.fc1, self.activation_fn,
            self.fc2, self.activation_fn,
            self.fc3
        ])

    def predict(self, x, return_feature_container=True):
        """
        Forward pass that returns both output and intermediate features.
        
        Args:
            x: Input tensor
            return_feature_container: If True, return FeatureContainer; if False, return list
        
        Returns:
            output: Final prediction
            features: FeatureContainer (or list if return_feature_container=False)
        """
        batch_size = x.size(0)
        
        # Forward pass with feature collection
        conv1_out = self.pool(self.activation_fn(self.conv1(x)))
        conv2_out = self.pool(self.activation_fn(self.conv2(conv1_out)))
        conv3_out = self.pool(self.activation_fn(self.conv3(conv2_out)))
        conv3_flat = conv3_out.view(batch_size, -1)
        fc1_out = self.activation_fn(self.fc1(conv3_flat))
        fc2_out = self.activation_fn(self.fc2(fc1_out))
        output = self.fc3(fc2_out)
        
        # Collect features
        feature_tensors = [conv1_out, conv2_out, conv3_flat, fc1_out, fc2_out]
        
        if return_feature_container:
            features = FeatureContainer(feature_tensors, self.feature_names)
            return output, features
        else:
            # Backward compatibility: return raw list
            return output, feature_tensors

    def get_layer_names(self):
        """Return the semantic names of feature layers."""
        return self.feature_names.copy()
