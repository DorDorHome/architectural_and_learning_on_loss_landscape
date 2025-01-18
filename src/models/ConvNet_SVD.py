import torch.nn as nn

# original implementation:
from omegaconf import DictConfig
# use NetParams dataclass from configurations.py:
from configs.configurations import NetParams

from src.models.svd_decomposed_single_conv import SVD_Conv2d
from src.models.svd_decomposed_FC import SVD_Linear 

class ConvNet_SVD(nn.Module):
    def __init__(self, config: NetParams ):
    
        super().__init__()
        # shared parameters:
        num_classes = config.num_classes
        self.conv_layer_bias = config.conv_layer_bias
        if config.weight_correction_scale == "relu_output_correction": 
            self.weight_correction_scale = 2**0.5
        else: 
            self.weight_correction_scale = config.weight_correction_scale
        self.fan_in_correction = config.fan_in_correction
        self.linear_layer_bias = config.linear_layer_bias
        
        # unique parameters for SVD options:
        self.SVD_only_stride_1 = config.SVD_only_stride_1
        self.allow_svd_values_negative = config.allow_svd_values_negative

        self.conv1 = SVD_Conv2d(3, 32, 5, bias=self.conv_layer_bias, 
                                weight_correction_scale= self.weight_correction_scale,
                                fan_in_correction =self.fan_in_correction,
                                SVD_only_stride_1 = self.SVD_only_stride_1,
                                allow_svd_values_negative = self.allow_svd_values_negative)
        # self.conv1_to_skip_scalar = nn.Parameter(torch.ones(32))
        
        self.conv2 = SVD_Conv2d(32, 64, 3, bias=self.conv_layer_bias ,
                                weight_correction_scale= self.weight_correction_scale ,
                                fan_in_correction =self.fan_in_correction,
                                SVD_only_stride_1 = self.SVD_only_stride_1,
                                allow_svd_values_negative = self.allow_svd_values_negative)
        self.conv3 = SVD_Conv2d(64, 128, 3, bias=self.conv_layer_bias ,
                                weight_correction_scale= self.weight_correction_scale,
                                fan_in_correction =self.fan_in_correction,
                                SVD_only_stride_1 = self.SVD_only_stride_1,
                                allow_svd_values_negative = self.allow_svd_values_negative) 
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        
        self.fc1 = SVD_Linear(self.num_conv_outputs, 128, bias = self.linear_layer_bias,
                            allow_svd_values_negative = self.allow_svd_values_negative)
        self.fc2 = SVD_Linear(128, 128 , bias = self.linear_layer_bias,
                            allow_svd_values_negative = self.allow_svd_values_negative)
        self.fc3 = SVD_Linear(128, num_classes, bias = self.linear_layer_bias,
                            allow_svd_values_negative = self.allow_svd_values_negative)
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

    

