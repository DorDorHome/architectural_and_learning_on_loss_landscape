"""
Version of ConvNet with batch normalization.
"""

import torch.nn as nn
import torch
from configs.configurations import NetParams
from src.utils.feature_container import FeatureContainer

activation_dict = {
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
    'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU
}

class ConvNetWithBatchNorm(nn.Module):
    """
    ConvNet with batch normalization applied to both convolutional and
    fully connected layers.
    """

    def __init__(self, config: NetParams):
        super().__init__()

        num_classes = config.num_classes if config.num_classes is not None else 10
        self.activation = config.activation if config.activation is not None else 'relu'
        self.activation_fn = activation_dict.get(self.activation, nn.ReLU)()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn_conv3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened size
        input_height = config.input_height if config.input_height is not None else 32
        input_width = config.input_width if config.input_width is not None else 32
        dummy = torch.zeros(1, 3, input_height, input_width)
        x = self.pool(self.activation_fn(self.bn_conv1(self.conv1(dummy))))
        x = self.pool(self.activation_fn(self.bn_conv2(self.conv2(x))))
        x = self.pool(self.activation_fn(self.bn_conv3(self.conv3(x))))
        flattened_size = x.view(1, -1).shape[1]

        # FC layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

        self.feature_names = [
            'conv1_pooled(with bn)',
            'conv2_pooled(with bn)',
            'conv3_flattened(with bn)',
            'fc1_output(with bn)',
            'fc2_output(with bn)'
        ]


    def predict(self, x: torch.Tensor, return_feature_container=False):
        """
        Forward pass that returns both output and intermediate features.
        """
        batch_size = x.size(0)

        # Conv layers
        conv1_out = self.pool(self.activation_fn(self.bn_conv1(self.conv1(x))))
        conv2_out = self.pool(self.activation_fn(self.bn_conv2(self.conv2(conv1_out))))
        conv3_out = self.pool(self.activation_fn(self.bn_conv3(self.conv3(conv2_out))))
        conv3_flat = conv3_out.view(batch_size, -1)

        # FC layers
        fc1_out = self.activation_fn(self.bn_fc1(self.fc1(conv3_flat)))
        fc2_out = self.activation_fn(self.bn_fc2(self.fc2(fc1_out)))
        output = self.fc3(fc2_out)

        feature_tensors = [conv1_out, conv2_out, conv3_flat, fc1_out, fc2_out]

        if return_feature_container:
            features = FeatureContainer(feature_tensors, self.feature_names)
            return output, features
        else:
            return output, feature_tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward compatible with PyTorch modules expecting `forward`.
        Returns only the network logits, matching `nn.Module` conventions.
        """
        output, _ = self.predict(x)
        return output
    

    def get_plasticity_map(self):
        """
        Returns a list of dictionaries describing the topology for plasticity algorithms.
        """
        return [
            {
                'name': 'conv1',
                'weight_module': self.conv1,
                'outgoing_module': self.conv2,
                'outgoing_feeds_into_norm': True  # conv2 is followed by bn_conv2
            },
            {
                'name': 'conv2',
                'weight_module': self.conv2,
                'outgoing_module': self.conv3,
                'outgoing_feeds_into_norm': True  # conv3 is followed by bn_conv3
            },
            {
                'name': 'conv3',
                'weight_module': self.conv3,
                'outgoing_module': self.fc1,
                'outgoing_feeds_into_norm': True  # fc1 is followed by bn_fc1
            },
            {
                'name': 'fc1',
                'weight_module': self.fc1,
                'outgoing_module': self.fc2,
                'outgoing_feeds_into_norm': True  # fc2 is followed by bn_fc2
            },
            {
                'name': 'fc2',
                'weight_module': self.fc2,
                'outgoing_module': self.fc3,
                'outgoing_feeds_into_norm': False # fc3 is NOT followed by BN
            }
        ]


    def get_layer_names(self):
        return self.feature_names.copy()

if __name__ == "__main__":
    config = NetParams(
        num_classes=10,
        input_height=32,
        input_width=32,
        activation='relu'
    )

    print("Testing ConvNetWithBatchNorm")
    model = ConvNetWithBatchNorm(config)
    model.eval()  # Switch to evaluation mode for single-input inference
    dummy_input = torch.randn(1, 3, 32, 32)
    output, features = model.predict(dummy_input)
    print("Output shape:", output.shape)
    print("Number of features:", len(features))
    print("-" * 30)
