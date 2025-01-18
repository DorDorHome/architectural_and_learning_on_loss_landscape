
import torch.nn as nn
import torch
import math



class SVD_Conv2d(torch.nn.Module):
    def __init__(self,
                input_channel:int,
                output_channel:int,
                kernel_size:int,
                stride:int = 1,
                padding:int = 0,
                bias:bool = False,
                SVD_only_stride_1:bool = False,# if true, means that, apply svd only when stride is 1
                decompose_type:str = 'channel'):
        """
        stride is fixed to 1 in this module
        """
        super(SVD_Conv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.only_stride_1 = SVD_only_stride_1
        self.decompose_type = decompose_type
        self.output_size = None
        if not SVD_only_stride_1 or self.stride==1:
            if self.decompose_type == 'channel':
                r = min(output_channel,input_channel*kernel_size*kernel_size)
                self.N = torch.nn.Parameter(torch.empty(output_channel,r))#Nxr
                self.C = torch.nn.Parameter(torch.empty(r,input_channel*kernel_size*kernel_size))#rxCHW
                self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
            else:#spatial decompose--VH-decompose
                r = min(input_channel*kernel_size,output_channel*kernel_size)
                self.N = torch.nn.Parameter(torch.empty(input_channel*kernel_size,r))#CHxr
                self.C = torch.nn.Parameter(torch.empty(r,output_channel*kernel_size))#rxNW
                self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
            self.bias = None
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(output_channel))
                self.register_parameter('bias',self.bias)
                torch.nn.init.constant_(self.bias,0.0)
            self.register_parameter('N',self.N)
            self.register_parameter('C',self.C)
            self.register_parameter('Sigma',self.Sigma)
            torch.nn.init.kaiming_normal_(self.N)
            torch.nn.init.kaiming_normal_(self.C)
            torch.nn.init.normal_(self.Sigma)
        else:
            self.conv2d = nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding,bias = bias)


    def forward(self,x):
        if not self.only_stride_1 or self.stride==1:
            if self.training:
                r = self.Sigma.size()[0]#r = min(N,CHW)
                # C = self.C[:r, :]#rxCHW
                # N = self.N[:, :r].contiguous()#Nxr
                Sigma = self.Sigma.abs()
                C = torch.mm(torch.diag(torch.sqrt(Sigma)), self.C)
                N = torch.mm(self.N,torch.diag(torch.sqrt(Sigma)))

            
            else:
                valid_idx = torch.arange(self.Sigma.size(0))[self.Sigma!=0]
                N = self.N[:,valid_idx].contiguous()
                C = self.C[valid_idx,:]
                Sigma = self.Sigma[valid_idx].abs()
                r = Sigma.size(0)
                C = torch.mm(torch.diag(torch.sqrt(Sigma)), C)
                N = torch.mm(N,torch.diag(torch.sqrt(Sigma)))
            if self.decompose_type == 'channel':
                C = C.view(r,self.input_channel,self.kernel_size,self.kernel_size)
                N = N.view(self.output_channel,r,1,1)
                y = torch.nn.functional.conv2d(input = x,weight = C,bias = None,stride = self.stride,padding = self.padding)
                y = torch.nn.functional.conv2d(input = y,weight = N,bias = self.bias,stride = 1,padding = 0)
            else:#spatial decompose
                N = N.view(self.input_channel,1,self.kernel_size,r).permute(3,0,2,1)#V:rxcxHx1
                C = C.view(r,self.output_channel,self.kernel_size,1).permute(1,0,3,2)#H:Nxrx1xW
                y = torch.nn.functional.conv2d(input = x,weight = N,bias = None,stride = [self.stride,1],padding = [self.padding,0])
                y = torch.nn.functional.conv2d(input = y,weight = C,bias = self.bias,stride = [1,self.stride],padding = [0,self.padding])

            

        else:
            y = self.conv2d(x)
        self.output_size = y.size()
        return y
    
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
