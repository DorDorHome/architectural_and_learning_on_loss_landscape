# comparable to the conv_net.ConvNet class,
# except that the weights are normalized.


import torch.nn as nn
import torch
# original implementation:
from omegaconf import DictConfig
# use NetParams dataclass from configurations.py:
from configs.configurations import NetParams

from src.models.normalized_weights_conv_layer import NormConv2d
from src.models.normalized_weights_FC import NormalizedWeightsLinear

activation_dict = {
    # 'relu': F.relu,
    # 'sigmoid': torch.sigmoid,
    # 'tanh': torch.tanh,
    # # Add more activations as needed
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
    'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}



class ConvNet_normalized(nn.Module): 
    def __init__(self, config: NetParams ):
        """'''
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        num_classes = config.num_classes
        self.conv_layer_bias = config.conv_layer_bias
        if config.weight_correction_scale == "relu_output_correction": 
            self.weight_correction_scale = 2**0.5
        else: 
            self.weight_correction_scale = config.weight_correction_scale
        self.fan_in_correction = config.fan_in_correction
        self.linear_layer_bias = config.linear_layer_bias
        #in_channels, out_channels, kernel_size are not implemented
        
        self.activation = config.activation
        
        self.conv1 = NormConv2d(3, 32, 5, bias=self.conv_layer_bias, 
                                weight_correction_scale= self.weight_correction_scale,
                                fan_in_correction =self.fan_in_correction)
        # self.conv1_to_skip_scalar = nn.Parameter(torch.ones(32))
        
        self.conv2 = NormConv2d(32, 64, 3, bias=self.conv_layer_bias ,
                                weight_correction_scale= self.weight_correction_scale ,
                                fan_in_correction =self.fan_in_correction)
        self.conv3 = NormConv2d(64, 128, 3, bias=self.conv_layer_bias ,
                                weight_correction_scale= self.weight_correction_scale,
                                fan_in_correction =self.fan_in_correction) 
        
        self.pool = nn.MaxPool2d(2, 2)

        
        # self.last_filter_output = 2 * 2
        # self.num_conv_outputs = 128 * self.last_filter_output
        dummy = torch.zeros(1, 3, config.input_height, config.input_width)
        x = self.pool(nn.ReLU()(self.conv1(dummy)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        flattened_size = x.view(1, -1).shape[1]
        
        self.num_conv_outputs = flattened_size        
        
        
        self.fc1 = NormalizedWeightsLinear(self.num_conv_outputs, 128, bias = self.linear_layer_bias)
        self.fc2 = NormalizedWeightsLinear(128, 128 , bias = self.linear_layer_bias)
        self.fc3 = NormalizedWeightsLinear(128, num_classes, bias = self.linear_layer_bias)

        self.act_type = self.activation
        self.activation = activation_dict.get(self.activation, None)
        
    

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(self.activation())
        self.layers.append(self.conv2)
        self.layers.append(self.activation())
        self.layers.append(self.conv3)
        self.layers.append(self.activation())
        self.layers.append(self.fc1)
        self.layers.append(self.activation())
        self.layers.append(self.fc2)
        self.layers.append(self.activation())
        self.layers.append(self.fc3)


    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]
