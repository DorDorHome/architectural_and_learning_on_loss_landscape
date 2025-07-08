"""
Feature container that provides both list and dictionary interfaces
for intermediate neural network features.
"""

from typing import Dict, List, Union, Any, Optional
import torch

class FeatureContainer:
    """
    A container that provides both list-like and dict-like access to features.
    This allows backward compatibility with existing list-based feature while 
    enabling semantic access via layer names.
    """
    
    def __init__(self, features: Union[List[torch.Tensor], Dict[str, torch.Tensor]], 
                 layer_names: Optional[List[str]] = None):
        """
        Initialize the feature container.
        
        Args:
            features: Either a list of tensors or dict mapping layer names to tensors
            layer_names: If features is a list, provide corresponding layer names
        """
        if isinstance(features, dict):
            self._features_dict = features
            self._features_list = list(features.values())
            self._layer_names = list(features.keys())
        elif isinstance(features, list):
            self._features_list = features
            if layer_names is None:
                # Generate default names
                self._layer_names = [f"layer_{i}" for i in range(len(features))]
            else:
                if len(layer_names) != len(features):
                    raise ValueError("Number of layer names must match number of features")
                self._layer_names = layer_names
            self._features_dict = dict(zip(self._layer_names, features))
        else:
            raise TypeError("Features must be either a list or dict")
    
    # List-like interface (for backward compatibility)
    def __getitem__(self, key: Union[int, str]) -> torch.Tensor:
        """Support both integer indexing and string key access."""
        if isinstance(key, int):
            return self._features_list[key]
        elif isinstance(key, str):
            return self._features_dict[key]
        else:
            raise TypeError("Key must be int or str")
    
    def __len__(self) -> int:
        """Return number of features."""
        return len(self._features_list)
    
    def __iter__(self):
        """Iterate over features in order."""
        return iter(self._features_list)
    
    # Dictionary-like interface
    def keys(self):
        """Return layer names."""
        return self._features_dict.keys()
    
    def values(self):
        """Return feature tensors."""
        return self._features_dict.values()
    
    def items(self):
        """Return (layer_name, feature) pairs."""
        return self._features_dict.items()
    
    def get(self, key: str, default=None):
        """Get feature by layer name with default."""
        return self._features_dict.get(key, default)
    
    # Utility methods
    def as_list(self) -> List[torch.Tensor]:
        """Return features as a list (current format)."""
        return self._features_list.copy()
    
    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return features as a dictionary."""
        return self._features_dict.copy()
    
    def layer_names(self) -> List[str]:
        """Return ordered list of layer names."""
        return self._layer_names.copy()
    
    def select_layers(self, layer_names: List[str]) -> 'FeatureContainer':
        """Return a new container with only specified layers."""
        selected_features = {name: self._features_dict[name] for name in layer_names}
        return FeatureContainer(selected_features)
    
    def __repr__(self) -> str:
        return f"FeatureContainer({len(self)} layers: {self._layer_names})"


# Backward compatibility: make FeatureContainer behave like a list by default
class BackwardCompatibleFeatureContainer(FeatureContainer):
    """
    A version that prioritizes list-like behavior for maximum backward compatibility.
    """
    
    def __init__(self, features: Union[List[torch.Tensor], Dict[str, torch.Tensor]], 
                 layer_names: Optional[List[str]] = None):
        super().__init__(features, layer_names)
        # Make it even more list-like
        self.__dict__.update({f'_item_{i}': feat for i, feat in enumerate(self._features_list)})
