"""
Enhanced Backprop learner that provides semantic layer names
while maintaining 100% backward compatibility.
"""

from src.algos.supervised.basic_backprop import Backprop
from src.utils.feature_container import FeatureContainer
from configs.configurations import BackpropConfig, NetParams
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List


class BackpropWithSemanticFeatures(Backprop):
    """
    Enhanced Backprop that adds semantic layer names to features
    without breaking existing functionality.
    
    Usage:
        # Existing code continues to work unchanged:
        features = learner.previous_features
        rank_summary = compute_all_rank_measures_list(features=features)
        
        # New semantic access:
        conv1_features = learner.get_feature_by_name('conv1_pooled')
        layer_names = learner.get_layer_names()
    """
    
    def __init__(self, net: nn.Module, config: BackpropConfig, netconfig: Optional[Union[NetParams, None]] = None):
        super().__init__(net, config, netconfig)
        
        # Get layer names from the model if available
        self.layer_names = self._get_layer_names()
        
        # Storage for enhanced features (optional, doesn't break existing code)
        self.previous_features_by_name = None
        self.feature_container = None
    
    def _get_layer_names(self) -> List[str]:
        """
        Get semantic layer names from the model.
        """
        if hasattr(self.net, 'get_layer_names'):
            return self.net.get_layer_names()
        else:
            # Fallback for models that don't have semantic names yet
            model_type = type(self.net).__name__
            if 'ConvNet' in model_type:
                return ['conv1_pooled', 'conv2_pooled', 'conv3_flattened', 'fc1_output', 'fc2_output']
            elif 'ResNet' in model_type:
                # You can customize this based on your ResNet architecture
                return [f'layer_{i}' for i in range(10)]  # Adjust based on actual architecture
            elif 'VGG' in model_type:
                # You can customize this based on your VGG architecture
                return [f'layer_{i}' for i in range(15)]  # Adjust based on actual architecture
            else:
                # Generic fallback
                return [f'layer_{i}' for i in range(10)]
    
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """
        Enhanced learn method that provides both list and dict access to features.
        ALL EXISTING CODE CONTINUES TO WORK UNCHANGED!
        """
        # Call parent method (unchanged)
        loss, output = super().learn(x, target)
        
        # Create enhanced features if we have layer names and features
        if self.previous_features is not None and self.layer_names:
            # Create feature container that provides both interfaces
            
            
            
            self.feature_container = FeatureContainer(self.previous_features, self.layer_names)
            
            # Also create a simple dict for easy access
            self.previous_features_by_name = {
                name: feature for name, feature in zip(self.layer_names, self.previous_features)
            }
        
        return loss, output
    
    # New utility methods for semantic access
    def get_feature_by_name(self, layer_name: str) -> torch.Tensor:
        """
        Get feature by semantic layer name.
        
        Args:
            layer_name: Name of the layer (e.g., 'conv1_pooled', 'fc2_output')
            
        Returns:
            Feature tensor for the specified layer
        """
        if self.previous_features_by_name is None:
            raise ValueError("Enhanced features not available. Run learn() first.")
        
        if layer_name not in self.previous_features_by_name:
            available = list(self.previous_features_by_name.keys())
            raise KeyError(f"Layer '{layer_name}' not found. Available layers: {available}")
        
        return self.previous_features_by_name[layer_name]
    
    def get_layer_names(self) -> List[str]:
        """
        Get list of available semantic layer names.
        """
        return self.layer_names.copy() if self.layer_names else []
    
    def get_features_by_names(self, layer_names: List[str]) -> List[torch.Tensor]:
        """
        Get multiple features by their semantic names.
        
        Args:
            layer_names: List of layer names
            
        Returns:
            List of feature tensors in the same order as layer_names
        """
        return [self.get_feature_by_name(name) for name in layer_names]
    
    def get_conv_features(self) -> Dict[str, torch.Tensor]:
        """
        Get all convolutional layer features.
        """
        if self.previous_features_by_name is None:
            raise ValueError("Enhanced features not available. Run learn() first.")
        
        conv_features = {
            name: feature for name, feature in self.previous_features_by_name.items()
            if 'conv' in name.lower()
        }
        return conv_features
    
    def get_fc_features(self) -> Dict[str, torch.Tensor]:
        """
        Get all fully connected layer features.
        """
        if self.previous_features_by_name is None:
            raise ValueError("Enhanced features not available. Run learn() first.")
        
        fc_features = {
            name: feature for name, feature in self.previous_features_by_name.items()
            if 'fc' in name.lower()
        }
        return fc_features
