
import torch.nn as nn
import torch
from torch.nn import functional as F
import math


# def orthogonal_init(tensor, rows, cols):
#     """
#     Initializes a tensor with orthogonal columns using QR decomposition.

#     Parameters:
#     - tensor (nn.Parameter): The tensor to be initialized.
#     - rows (int): Number of rows.
#     - cols (int): Number of columns.
#     """
#     # Generate a random matrix
#     random_tensor = torch.randn(rows, cols)
#     # Perform QR decomposition
#     q, r = torch.linalg.qr(random_tensor)
#     # Assign the orthogonal matrix to tensor
#     with torch.no_grad():
#         tensor.copy_(q[:, :cols])


class SVD_Conv2d(torch.nn.Module):
    def __init__(self,
                input_channel:int,
                output_channel:int,
                kernel_size:int,
                stride:int = 1,
                padding:int = 0,
                dilation:int = 1,
                groups:int =1, 
                bias:bool = False,
                SVD_only_stride_1:bool = False,# if true, means that, apply svd only when stride is 1
                weight_correction_scale: float = 2**0.5,
                fan_in_correction: bool = True,
                allow_svd_values_negative: bool = False,# for svd only
                decompose_type:str = 'channel'):
        """
        stride is fixed to 1 in this module
        """
        super(SVD_Conv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        # Handle kernel_size being int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # unique parameters for SVD options:
        self.only_stride_1 = SVD_only_stride_1  # if true, means that, apply svd only when stride is 1
        # unique parameters for SVD options:
        self.decompose_type = decompose_type # need to set
        self.output_size = None
        self.groups = groups
        # unique parameters for SVD options:
        self.allow_svd_values_negative = allow_svd_values_negative
        
        assert len(self.kernel_size) == 2, "kernel_size must be a tuple of length 2"
        if fan_in_correction:
            self.in_fans = input_channel * self.kernel_size[0] * self.kernel_size[1]
        else:
            self.in_fans = 1
        
        self.weight_correction_scale_total = weight_correction_scale/ math.sqrt(self.in_fans)
        
        
    
        # Determine if SVD decomposition should be applied
        self.apply_svd = not self.only_stride_1 or self.stride == 1

        if self.apply_svd:
            if self.decompose_type == 'channel':
                # calculate the rank of the decomposed matrix
                self.r = min(output_channel,input_channel*kernel_size*kernel_size)
                
                self.N = torch.nn.Parameter(torch.empty(output_channel,self.r))#Nxr
                self.C = torch.nn.Parameter(torch.empty(self.r,input_channel*kernel_size*kernel_size))#rxCHW
                self.Sigma = torch.nn.Parameter(torch.empty(self.r))#rank = r
            else:#spatial decompose--VH-decompose
                self.r = min(input_channel*kernel_size,output_channel*kernel_size)
                self.N = torch.nn.Parameter(torch.empty(input_channel*kernel_size,self.r))#CHxr
                self.C = torch.nn.Parameter(torch.empty(self.r,output_channel*kernel_size))#rxNW
                self.Sigma = torch.nn.Parameter(torch.empty(self.r))#rank = r
            self.bias = None
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(output_channel))
                # self.register_parameter('bias',self.bias)
                torch.nn.init.constant_(self.bias,0.0)
            # self.register_parameter('N',self.N)
            # self.register_parameter('C',self.C)
            # self.register_parameter('Sigma',self.Sigma)
            
            # original code:
            # torch.nn.init.kaiming_normal_(self.N)
            # torch.nn.init.kaiming_normal_(self.C)
            # torch.nn.init.normal_(self.Sigma)
            
            nn.init.orthogonal_(self.N)#, self.N.size(0), self.N.size(1))
            nn.init.orthogonal_(self.C)#, self.C.size(0), self.C.size(1))
            
            # check orthogonality:
            with torch.no_grad():
                if self.decompose_type == 'channel':
                    assert torch.allclose(torch.matmul(self.N.t(), self.N), torch.eye(self.r).to(self.N.device), atol=1e-6), "N is not orthogonal"
                    assert torch.allclose(torch.matmul(self.C, self.C.t()), torch.eye(self.r).to(self.C.device), atol=1e-6), "C is not orthogonal"
                else:
                    assert torch.allclose(torch.matmul(self.N.t(), self.N), torch.eye(self.r).to(self.N.device), atol=1e-6), "N is not orthogonal"
                    assert torch.allclose(torch.matmul(self.C, self.C.t()), torch.eye(self.r).to(self.C.device), atol=1e-6), "C is not orthogonal"
                        
            
            torch.nn.init.normal_(self.Sigma)       
            
        else:
            self.conv2d = nn.Conv2d(in_channels =input_channel,
                                    out_channels=output_channel,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding = padding, 
                                    dilation = dilation, groups = groups ,
                                    bias = bias)


    def forward(self,x):
        if self.apply_svd:
            
            conv_filter_scaled = None
            # training mode:
            if self.training:
                # r = self.Sigma.size()[0]#r = min(N,CHW)
  
                if not self.allow_svd_values_negative:
                    Sigma = self.Sigma.abs()
                else:
                    Sigma = self.Sigma
                if self.decompose_type == 'channel':
                    conv_filter = torch.mm(torch.mm(self.N, torch.diag(Sigma)), self.C).view(self.output_channel, self.input_channel, self.kernel_size[0], self.kernel_size[1])
                else:
                    raise NotImplementedError('Only channel decompose is implemented')
            else:
                valid_idx = torch.arange(self.Sigma.size(0))[self.Sigma!=0]
                N = self.N[:,valid_idx].contiguous()
                C = self.C[valid_idx,:]
                if not self.allow_svd_values_negative:
                    Sigma = self.Sigma[valid_idx].abs()
                else:
                    Sigma = self.Sigma[valid_idx]
                r = Sigma.size(0)
                if self.decompose_type == 'channel':
                    conv_filter = torch.mm(torch.mm(N, torch.diag(Sigma)), C).view(self.output_channel, self.input_channel, self.kernel_size[0], self.kernel_size[1])
                else:
                    raise NotImplementedError('Only channel decompose is implemented')
            
            conv_filter_scaled = conv_filter * self.weight_correction_scale_total
  
            out = torch.nn.functional.conv2d(input = x,
                                         weight = conv_filter_scaled,
                                         bias = self.bias,
                                         stride = self.stride,
                                         padding = self.padding,
                                         dilation = self.dilation,
                                         groups = self.groups)

        else:
            out = self.conv2d(x)
        self.output_size = out.size()
        return out
    
    @property
    def ParamN(self):
        if not self.only_stride_1 or self.stride==1:
            return self.N
        else:
            return None
    
    @property
    def ParamC(self):
        if not self.only_stride_1 or self.stride==1:
            return self.C
        else:
            return None

    @property
    def ParamSigma(self):
        if not self.only_stride_1 or self.stride==1:
            return self.Sigma
        else:
            return None


def conv3x3(in_planes, out_planes, stride=1,SVD_only_stride_1 = False,decompose_type = 'channel'):
    "3x3 convolution with padding"
    return SVD_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding = 1,bias=False,SVD_only_stride_1=SVD_only_stride_1,decompose_type = decompose_type)

if __name__ == "__main__":
    # Define input tensor of shape [batch_size, in_channels, height, width]
    x = torch.randn(8, 3, 32, 32)  # Example input

    # Create the custom SVD decomposed convolutional layer
    svd_conv = SVD_Conv2d(
        input_channel=3,
        output_channel=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # check that svd_conv.N is orthogonal:
    print(torch.mm(svd_conv.N,svd_conv.N.t()).shape)
    print(torch.mm(svd_conv.N.t(),svd_conv.N).shape)
    
    
    # check that svd_conv.C is orthogonal:
    print(torch.mm(svd_conv.C,svd_conv.C.t()).shape)
    
    print(svd_conv.r)
    y = svd_conv(x)
    print(y.size())
    
    # set to eval mode
    svd_conv.eval()
    y2 = svd_conv(x)
    print(y2.size())