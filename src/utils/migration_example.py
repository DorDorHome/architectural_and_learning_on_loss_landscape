"""
Practical example showing step-by-step migration to FeatureContainer
while maintaining backward compatibility.
"""

# import backprop module
from src.algos.supervised.basic_backprop import Backprop
from src.utils.feature_container import FeatureContainer
from configs.configurations import BackpropConfig, NetParams
import torch


# Step 1: Add this utility to your base learner
class FeatureAdapter:
    """
    Simple adapter that can wrap existing list-based features
    to provide dictionary access without breaking existing code.
    
    # support both list and dict-like access to features
    
    """
    
    def __init__(self, features_list, layer_names=None):
        self._features = features_list
        if layer_names:
            # name to index mapping for dictionary-like access
            self._name_map = {name: i for i, name in enumerate(layer_names)}
        else:
            self._name_map = {}
    
    # List interface (existing code continues to work)
    def __getitem__(self, index):
        
        # index support for int   
        if isinstance(index, int):
            return self._features[index]
        
        # string support for layer names
        elif isinstance(index, str) and index in self._name_map:
            return self._features[self._name_map[index]]
        else:
            raise KeyError(f"Unknown layer: {index}")
    
    def __len__(self):
        return len(self._features)
    
    def __iter__(self):
        return iter(self._features)
    
    # Dictionary-like access (new functionality)
    def get_by_name(self, name):
        """Get feature by semantic name."""
        return self[name]
    
    def get_layer_names(self):
        """Get available layer names."""
        return list(self._name_map.keys())


# Step 2: Example of how to modify existing basic_backprop.py
# This requires ZERO changes to existing rank tracking code.

class BackpropWithFeatureNames(Backprop):
    """
    Enhanced Backprop that adds semantic layer names to features
    without breaking existing functionality.
    """
    
    def __init__(self, net, config, netconfig=None):
        super().__init__(net, config, netconfig)
        
        # Define layer names based on your model architecture
        # This can be model-specific or auto-detected
        self.layer_names = self._get_layer_names()
    
    def _get_layer_names(self):
        """
        Get semantic layer names. This can be customized per model type.
        """
        if hasattr(self.net, 'get_layer_names'):
            return self.net.get_layer_names()
        elif hasattr(self.net, 'feature_names'):
            return self.net.feature_names
        else:
            # Fallback: generate generic names
            # You can customize this based on your model types
            model_type = type(self.net).__name__
            if 'ConvNet' in model_type:
                return ['conv1_pooled', 'conv2_pooled', 'conv3_flattened', 'fc1_output', 'fc2_output']
            elif 'DeepFFNN' in model_type:
                return [f'hidden_{i}' for i in range(len(self.net.hidden_layers) + 1)]
            else:
                return [f'layer_{i}' for i in range(10)]  # Generic fallback
    
    def learn(self, x, target):
        """
        Enhanced learn method that provides both list and dict access to features.
        ALL EXISTING CODE CONTINUES TO WORK UNCHANGED!
        """
        # Call parent method (unchanged)
        loss, output = super().learn(x, target)
        
        # Enhance features with semantic names (optional, non-breaking)
        if self.previous_features is not None and self.layer_names:
            # Only enhance if we have layer names
            # This creates a wrapper that provides both interfaces
            enhanced_features = FeatureAdapter(self.previous_features, self.layer_names)
            
            # Store enhanced version alongside original
            self.previous_features_enhanced = enhanced_features
            # Note: self.previous_features remains a list for backward compatibility
        
        return loss, output
    
    def get_feature_by_name(self, layer_name):
        """
        New utility method to get features by semantic name.
        """
        if hasattr(self, 'previous_features_enhanced'):
            return self.previous_features_enhanced.get_by_name(layer_name)
        else:
            raise ValueError("Enhanced features not available. Run learn() first.")
    
    def get_available_layer_names(self):
        """
        Get list of available semantic layer names.
        """
        return self.layer_names.copy() if self.layer_names else []


# Step 3: Example usage showing how existing code continues to work
# while new code can use semantic names

def example_usage():
    """
    Demonstrate that existing code works unchanged while enabling new capabilities.
    """
    
    # Assume you have a learner instance
    learner = BackpropWithFeatureNames(net, config)

    # Existing rank tracking code continues to work UNCHANGED:
    list_of_features_for_every_layers = learner.previous_features
    rank_summary_list = compute_all_rank_measures_list(
        features=list_of_features_for_every_layers,
        # ... other parameters
    )
    
    # NEW: You can now also access features by semantic names
    # if hasattr(learner, 'previous_features_enhanced'):
    #     conv1_features = learner.get_feature_by_name('conv1_pooled')
    #     fc2_features = learner.get_feature_by_name('fc2_output')
    #     
    #     # Or get specific layers for analysis
    #     important_layers = ['conv1_pooled', 'fc2_output']
    #     selected_features = [learner.get_feature_by_name(name) for name in important_layers]
    
    pass


# Step 4: Minimal model modification (optional)
# Add this method to your existing models:

def add_layer_names_to_existing_model():
    """
    Example of minimal change to existing ConvNet model.
    Just add this method - no other changes needed!
    """
    
    # In your ConvNet class, add this method:
    def get_layer_names(self):
        return ['conv1_pooled', 'conv2_pooled', 'conv3_flattened', 'fc1_output', 'fc2_output']
    
    # That's it! Your model now supports semantic layer names
    # while all existing code continues to work unchanged.


# Step 5: Advanced usage for new experiments
def advanced_feature_analysis_example():
    """
    Example of how enhanced features enable more sophisticated analysis.
    """
    
    # This kind of analysis becomes much easier with semantic names:
    
    # Track rank evolution of specific architectural components
    conv_layers = ['conv1_pooled', 'conv2_pooled', 'conv3_flattened']
    fc_layers = ['fc1_output', 'fc2_output']
    
    # Compare rank dynamics between conv and FC layers
    # Log features with meaningful names to wandb
    # Create targeted visualizations
    # etc.
    
    pass
