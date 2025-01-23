
import torch.nn as nn
import torch
import math
from torch.nn import functional as F
# import the types:
from typing import Tuple, List, Dict, Any, Union

class NormConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        weight_correction_scale: float = 2**0.5,
        fan_in_correction: bool = True
        ):
        super(NormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Handle kernel_size being int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode


        
        # for calculating correct weight correction scale:
        # making sure self.kernel_size is a tuple of length 2:
        assert len(self.kernel_size) == 2, "kernel_size must be a tuple of length 2"
        if fan_in_correction:
            self.in_fans = in_channels * self.kernel_size[0] * self.kernel_size[1]
        else:
            self.in_fans = 1
        
        self.weight_correction_scale_total = weight_correction_scale/ math.sqrt(self.in_fans)
        
        

        # Initialize raw weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        # Initialize scalar parameters (one per output channel)
        self.scalar = nn.Parameter(torch.ones(out_channels))  # Start with scalars = 1

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            #first implementation:
            #self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        # Initialize scalar parameters to 1
        nn.init.constant_(self.scalar, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        #     # Initialize bias similar to nn.Conv2d
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, bias: bool = True) -> torch.Tensor:
        # Flatten the weight to compute norms per output channel
        weight_flat = self.weight.view(self.out_channels, -1)  # Shape: [out_channels, ...]
        # Compute L2 norm for each output channel
        norm = weight_flat.norm(p=2, dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        # Normalize the weights
        weight_normalized = self.weight / norm.view(self.out_channels, 1, 1, 1)
        # Scale the normalized weights with the scalar parameter
        scalar = self.scalar.view(self.out_channels, 1, 1, 1)
        weight_scaled = weight_normalized * scalar *self.weight_correction_scale_total
        # Perform the convolution operation
        if bias is False:
            out= F.conv2d(
                x,
                weight_scaled,
                 None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
        elif bias is True:
            out = F.conv2d(
                x,
                weight_scaled,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
            
        return out
    
    
class Norm_output_Conv2d(nn.Module):
    """
    Instead of normalizing the weights, this layer normalizes the output of the convolutional layer.
    
    
    """
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int or tuple,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                
                # not required:
                # weight_correction_scale: float = 2**0.5,
                # fan_in_correction: bool = True,
                
                #unique for this class:
                norm_pre_activation: bool = True# control whether to normalize before or after activation
                weight_decay_for_norm: float = 0.0 # weight decay for the normalization parameters
                ):
        
        super(Norm_output_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Handle kernel_size being int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # hyperparameters for the normalization:
        self.norm_pre_activation = norm_pre_activation
        self.weight_decay_for_norm = weight_decay_for_norm
        
        # weight correction is not needed. 
        # correction is done by acculumating the weight decay in the normalization parameters.
        
        # Initialize raw weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            #first implementation:
            #self.register_parameter('bias', None)
            
        # Initialize scalar parameters (one per output channel)
        self.scalar_parameter = nn.Parameter(torch.ones(out_channels))
        
        # weight decay factor for the normalization parameters:
        self.weight_decay_for_norm = weight_decay_for_norm
        
        self.norm_moving_average = torch.ones(out_channels)
        
        self.last_output_norm = 1.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # update final scalar_to_use, to represent the norm 
        self.norm_moving_average =  self.weight_decay_for_norm* self.last_output_norm +
                        (1 -self.weight_decay_for_norm) * self.norm_moving_average 
        
        weight_to_use = self.weight/ self.norm_moving_average.view(self.out_channels, 1, 1, 1)
        
        
        # Perform the convolution operation
        out = F.conv2d(
            x,
            weight_to_use
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        
        # update the last output norm:
        self.last_oupout_norm = out.norm(p=2, dim=1, keepdim=True) + 1e-8
        
        
        # normalize the output:
        if self.norm_pre_activation:
            # normalize before activation
            norm = out.norm(p=2, dim=1, keepdim=True) + 1e-8
            out = out / norm
        else:
            # normalize after activation
            norm = out.norm(p=2, dim=1, keepdim=True) + 1e-8
            out = out / norm
        
        # scale output by the scalar parameter:
        out = out * self.scalar_to_use.view(self.out_channels, 1, 1, 1)
        
        
        
        


        
        
        
        
        
                 
    
    
            
# Example Usage
if __name__ == "__main__":
    # Define input tensor of shape [batch_size, in_channels, height, width]
    x = torch.randn(8, 3, 32, 32)  # Example input

    # Create the custom normalized convolutional layer
    norm_conv = NormConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    )

    # Forward pass
    output = norm_conv(x)
    print(output.shape)  # Should be [8, 16, 32, 32]
    print(output[0, 0, 0, :5])  # Print first 5 elements of the first output channel
    output2 = norm_conv(x, bias = True)
    print(output2[0, 0, 0, :5])  # Print first 5 elements of the first output channel