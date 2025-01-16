# comparable to the conv_net.ConvNet class,
# except that the weights are normalized.


import torch.nn as nn

# original implementation:
from omegaconf import DictConfig
# use NetParams dataclass from configurations.py:
from configs.configurations import NetParams

from src.models.normalized_weights_conv_layer import NormConv2d
from src.models.normalized_weights_FC import NormalizedWeightsLinear

class ConvNet_normalized(nn.Module): 
    def __init__(self, config: NetParams ):
        """'''
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        num_classes = config.num_classes
        #in_channels, out_channels, kernel_size are not implemented
        
        
        self.conv1 = NormConv2d(3, 32, 5)
        # self.conv1_to_skip_scalar = nn.Parameter(torch.ones(32))
        
        self.conv2 = NormConv2d(32, 64, 3)
        self.conv3 = NormConv2d(64, 128, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.fc1 = NormalizedWeightsLinear(self.num_conv_outputs, 128)
        self.fc2 = NormalizedWeightsLinear(128, 128)
        self.fc3 = NormalizedWeightsLinear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.conv2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.conv3)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc3)

        self.act_type = 'relu'

    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]
