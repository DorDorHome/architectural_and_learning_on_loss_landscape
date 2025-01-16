
import torch.nn as nn
import torch
import math
from torch.nn import functional as F

class NormConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int or tuple,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
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

        # Initialize raw weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        # Initialize scalar parameters (one per output channel)
        self.scalar = nn.Parameter(torch.ones(out_channels))  # Start with scalars = 1

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize scalar parameters to 1
        nn.init.constant_(self.scalar, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        #     # Initialize bias similar to nn.Conv2d
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, bias = None):
        # Flatten the weight to compute norms per output channel
        weight_flat = self.weight.view(self.out_channels, -1)  # Shape: [out_channels, ...]
        # Compute L2 norm for each output channel
        norm = weight_flat.norm(p=2, dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        # Normalize the weights
        weight_normalized = self.weight / norm.view(self.out_channels, 1, 1, 1)
        # Scale the normalized weights with the scalar parameter
        scalar = self.scalar.view(self.out_channels, 1, 1, 1)
        weight_scaled = weight_normalized * scalar
        # Perform the convolution operation
        if bias is None:
            out= F.conv2d(
                x,
                weight_scaled,
                 None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
        elif bias:
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