"""
version of ConvNet with layer norm incorporated.

FeatureContainer
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

class ConvNetWithFCLayerNorm(nn.Module):
    """
    Similar to ConvNet, but with layer normalization
    applied to fully connected layers.
    
    """

    
    def __init__(self, config: NetParams):
        super().__init__()

    
        num_classes = config.num_classes if config.num_classes is not None else 10
        self.activation = config.activation if config.activation is not None else 'relu'
        self.activation_fn = activation_dict.get(self.activation, nn.ReLU)()
        self.elementwise_affine = config.norm_param.layer_norm.elementwise_affine
        
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute flattened size
        # by using a dummy input to test run
        input_height = config.input_height if config.input_height is not None else 32
        input_width = config.input_width if config.input_width is not None else 32
        dummy = torch.zeros(1, 3, input_height, input_width)
        x = self.pool(self.activation_fn(self.conv1(dummy)))
        x = self.pool(self.activation_fn(self.conv2(x)))
        x = self.pool(self.activation_fn(self.conv3(x)))
        flattened_size = x.view(1, -1).shape[1]
        
        # FC layers with Layer Normalization
        
        ## fc1 + layer norm
        self.fc1 = nn.Linear(flattened_size, 128)
        self.ln1 = nn.LayerNorm(128, elementwise_affine=self.elementwise_affine)  # Layer norm after first FC layer
         
         
        ## fc2 + layer norm
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128, elementwise_affine=self.elementwise_affine)  # Layer norm after second FC layer
        
        # FC3
        self.fc3 = nn.Linear(128, num_classes)
        
        
        # Define feature names for better semantics
        self.feature_names = [
            'conv1_pooled',      # After conv1 + activation + pooling
            'conv2_pooled',      # After conv2 + activation + pooling  
            'conv3_flattened',   # After conv3 + activation + pooling + flattening
            'fc1_output(with ln)',        # After fc1 + ln1 + activation
            'fc2_output(with ln)'         # After fc2 + ln2+  activation
        ]
        
        # Layers for compatibility with existing perturb() method
        self.layers = nn.ModuleList([
            self.conv1, self.activation_fn,
            self.conv2, self.activation_fn,
            self.conv3, self.activation_fn,
            self.fc1, self.ln1,
            self.activation_fn,
            self.fc2, self.ln2,
            self.activation_fn,
            self.fc3
        ])

    def predict(self, x, return_feature_container=False):
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
        
        # FC layers with layer normalization
        fc1_raw = self.fc1(conv3_flat)
        fc_norm = self.ln1(fc1_raw)
        fc1_out = self.activation_fn(fc_norm)   
        
        
        fc2_raw = self.fc2(fc1_out)
        fc2_norm = self.ln2(fc2_raw)   
        fc2_out = self.activation_fn(fc2_norm)
        output = self.fc3(fc2_out)
        
        # Collect features
        feature_tensors = [conv1_out, conv2_out, conv3_flat, fc1_out, fc2_out]
        
        if return_feature_container:
            features = FeatureContainer(feature_tensors, self.feature_names)
            return output, features
        else:
            # Backward compatibility: return raw list
            return output, feature_tensors

    def get_plasticity_map(self):
        """
        Returns a list of dictionaries describing the topology for plasticity algorithms.
        """
        return [
            {
                'name': 'conv1',
                'weight_module': self.conv1,
                'outgoing_module': self.conv2,
                'outgoing_feeds_into_norm': True  # conv2 output IS normalized (ln_conv2)
            },
            {
                'name': 'conv2',
                'weight_module': self.conv2,
                'outgoing_module': self.conv3,
                'outgoing_feeds_into_norm': True  # conv3 output IS normalized (ln_conv3)
            },
            {
                'name': 'conv3',
                'weight_module': self.conv3,
                'outgoing_module': self.fc1,
                'outgoing_feeds_into_norm': True  # fc1 output IS normalized (ln_fc1)
            },
            {
                'name': 'fc1',
                'weight_module': self.fc1,
                'outgoing_module': self.fc2,
                'outgoing_feeds_into_norm': True  # fc2 output IS normalized (ln_fc2)
            },
            {
                'name': 'fc2',
                'weight_module': self.fc2,
                'outgoing_module': self.fc3,
                'outgoing_feeds_into_norm': False # fc3 is not normalized
            }
        ]
    def get_layer_names(self):
        """Return the semantic names of feature layers."""
        return self.feature_names.copy()


class ConvNet_conv_and_FC_LayerNorm(nn.Module):
    """
    Similar to ConvNet, but with layer normalization
    applied to both convolutional and fully connected layers.
    
    """

    
    def __init__(self, config: NetParams):
        super().__init__()
        
        num_classes = config.num_classes if config.num_classes is not None else 10
        self.activation = config.activation if config.activation is not None else 'relu'
        self.activation_fn = activation_dict.get(self.activation, nn.ReLU)()
        self.elementwise_affine = config.norm_param.layer_norm.elementwise_affine
        
        # Conv layers with Layer Normalization
        self.conv1 = nn.Conv2d(3, 32, 5)
        # For conv layers, LayerNorm needs to normalize over [C, H, W] dimensions
        # We'll apply it after each conv layer but before activation
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Layer norms for conv layers will be created dynamically based on output shapes
        # since we need to know the spatial dimensions after convolution
        
        # Compute flattened size using dummy input
        input_height = config.input_height if config.input_height is not None else 32
        input_width = config.input_width if config.input_width is not None else 32
        dummy = torch.zeros(1, 3, input_height, input_width)
        
        # Forward through conv layers to get shapes for layer norm
        conv1_out = self.conv1(dummy)
        self.ln_conv1 = nn.LayerNorm(conv1_out.shape[1:], elementwise_affine=self.elementwise_affine)  # Normalize over [C, H, W]
        
        x = self.pool(self.activation_fn(self.ln_conv1(conv1_out)))
        conv2_out = self.conv2(x)
        self.ln_conv2 = nn.LayerNorm(conv2_out.shape[1:], elementwise_affine=self.elementwise_affine)  # Normalize over [C, H, W]
        
        x = self.pool(self.activation_fn(self.ln_conv2(conv2_out)))
        conv3_out = self.conv3(x)
        self.ln_conv3 = nn.LayerNorm(conv3_out.shape[1:], elementwise_affine=self.elementwise_affine)  # Normalize over [C, H, W]
        
        x = self.pool(self.activation_fn(self.ln_conv3(conv3_out)))
        flattened_size = x.view(1, -1).shape[1]
        
        # FC layers with Layer Normalization
        self.fc1 = nn.Linear(flattened_size, 128)
        self.ln_fc1 = nn.LayerNorm(128, elementwise_affine=self.elementwise_affine)  # Layer norm after first FC layer
        
        self.fc2 = nn.Linear(128, 128)
        self.ln_fc2 = nn.LayerNorm(128, elementwise_affine=self.elementwise_affine)  # Layer norm after second FC layer
        
        self.fc3 = nn.Linear(128, num_classes)
        
        
        # Define feature names for better semantics
        self.feature_names = [
            'conv1_pooled(with ln)',      # After conv1 + ln + activation + pooling
            'conv2_pooled(with ln)',      # After conv2 + ln + activation + pooling  
            'conv3_flattened(with ln)',   # After conv3 + ln + activation + pooling + flattening
            'fc1_output(with ln)',        # After fc1 + ln + activation
            'fc2_output(with ln)'         # After fc2 + ln + activation
        ]
        
        # Layers for compatibility with existing perturb() method
        self.layers = nn.ModuleList([
            self.conv1, self.ln_conv1, self.activation_fn,
            self.conv2, self.ln_conv2, self.activation_fn,
            self.conv3, self.ln_conv3, self.activation_fn,
            self.fc1, self.ln_fc1, self.activation_fn,
            self.fc2, self.ln_fc2, self.activation_fn,
            self.fc3
        ])

    def predict(self, x, return_feature_container=False):
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
        
        # Conv layers with layer normalization
        conv1_raw = self.conv1(x)
        conv1_norm = self.ln_conv1(conv1_raw)
        conv1_activated = self.activation_fn(conv1_norm)
        conv1_out = self.pool(conv1_activated)
        
        conv2_raw = self.conv2(conv1_out)
        conv2_norm = self.ln_conv2(conv2_raw)
        conv2_activated = self.activation_fn(conv2_norm)
        conv2_out = self.pool(conv2_activated)
        
        conv3_raw = self.conv3(conv2_out)
        conv3_norm = self.ln_conv3(conv3_raw)
        conv3_activated = self.activation_fn(conv3_norm)
        conv3_pooled = self.pool(conv3_activated)
        conv3_flat = conv3_pooled.view(batch_size, -1)
        
        # FC layers with layer normalization
        fc1_raw = self.fc1(conv3_flat)
        fc1_norm = self.ln_fc1(fc1_raw)
        fc1_out = self.activation_fn(fc1_norm)
        
        fc2_raw = self.fc2(fc1_out)
        fc2_norm = self.ln_fc2(fc2_raw)
        fc2_out = self.activation_fn(fc2_norm)
        
        output = self.fc3(fc2_out)
        
        # Collect features at the same points as the original ConvNet
        feature_tensors = [conv1_out, conv2_out, conv3_flat, fc1_out, fc2_out]
        
        if return_feature_container:
            features = FeatureContainer(feature_tensors, self.feature_names)
            return output, features
        else:
            # Backward compatibility: return raw list
            return output, feature_tensors

    def get_plasticity_map(self):
        """
        Returns a list of dictionaries describing the topology for plasticity algorithms.
        """
        return [
            {
                'name': 'conv1',
                'weight_module': self.conv1,
                'outgoing_module': self.conv2,
                'outgoing_feeds_into_norm': True  # conv2 output IS normalized (ln_conv2)
            },
            {
                'name': 'conv2',
                'weight_module': self.conv2,
                'outgoing_module': self.conv3,
                'outgoing_feeds_into_norm': True  # conv3 output IS normalized (ln_conv3)
            },
            {
                'name': 'conv3',
                'weight_module': self.conv3,
                'outgoing_module': self.fc1,
                'outgoing_feeds_into_norm': True  # fc1 output IS normalized (ln_fc1)
            },
            {
                'name': 'fc1',
                'weight_module': self.fc1,
                'outgoing_module': self.fc2,
                'outgoing_feeds_into_norm': True  # fc2 output IS normalized (ln_fc2)
            },
            {
                'name': 'fc2',
                'weight_module': self.fc2,
                'outgoing_module': self.fc3,
                'outgoing_feeds_into_norm': False # fc3 is not normalized
            }
        ]
    def get_layer_names(self):
        """Return the semantic names of feature layers."""
        return self.feature_names.copy()


if __name__ == "__main__":
    # Create a dummy config
    config = NetParams(
        num_classes=10,
        input_height=32,
        input_width=32,
        activation='relu',
        norm_param={'layer_norm': {'elementwise_affine': True}}
    )
    
    # Test ConvNetWithFCLayerNorm
    print("Testing ConvNetWithFCLayerNorm")
    model_fc_ln = ConvNetWithFCLayerNorm(config)
    dummy_input = torch.randn(1, 3, 32, 32)
    output, features = model_fc_ln.predict(dummy_input)
    print("Output shape:", output.shape)
    print("Number of features:", len(features))
    print("-" * 30)
    
    # Test ConvNet_conv_and_FC_LayerNorm
    print("Testing ConvNet_conv_and_FC_LayerNorm")
    model_conv_fc_ln = ConvNet_conv_and_FC_LayerNorm(config)
    output, features = model_conv_fc_ln.predict(dummy_input)
    print("Output shape:", output.shape)
    print("Number of features:", len(features))
    print("-" * 30)

