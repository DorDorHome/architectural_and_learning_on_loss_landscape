
# implementation of SVD decomposed fully connected layer
# to add: options not to use fan_in correction
from locale import normalize
import torch.nn as nn
import torch
import math
from torch.nn import functional as F

activation_dict = {
    # 'relu': F.relu,
    # 'sigmoid': torch.sigmoid,
    # 'tanh': torch.tanh,
    # # Add more activations as needed
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
    'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}




class SVD_Linear(nn.Module):
    def __init__(self, in_dim:int, 
                out_dim:int,
                bias:bool = True, bias_init:float =0,
                activation: str = None,
                weight_correction_scale:float = 2**0.5,
                # unique parameters for SVD options:
                allow_svd_values_negative: bool = False,
                debug= False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.activation = activation
        
        # unique parameters for SVD options:
        self.allow_svd_values_negative = allow_svd_values_negative
        

        
        self.debug = debug
        
        self.r = min(in_dim, out_dim)
        self.N = torch.nn.Parameter(torch.empty(self.out_dim, self.r))  # Nxr    
        self.C = torch.nn.Parameter(torch.empty(self.r, self.in_dim)) 
        self.Sigma = nn.Parameter(torch.empty(self.r))
        
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim))
            nn.init.constant_(self.bias, bias_init)
        else:
            self.bias = None
            
        # should add fan_in correction here:
        #self.weight_correction_scale = weight_correction_scale/(in_dim**0.5)# 
        self.weight_correction_scale = weight_correction_scale#*self.out_dim/self.in_dim #weight_correction_scale/(in_dim**0.5)
        
        if (activation is not None) and (activation != 'linear'):
            self.activation = activation_dict.get(activation, None)   
            self.activation = self.activation()
        if (activation is None) or (activation == 'linear'):
            self.activation = None
            
        self.debug = debug
        
        nn.init.orthogonal_(self.N)#, self.N.size(0), self.N.size(1))
        nn.init.orthogonal_(self.C)#, self.C.size(0), self.C.size(1))
            
        nn.init.normal_(self.Sigma, mean=1, std=0.1)
        
        #check orthogonality of N and C:
        with torch.no_grad():
            assert torch.allclose(torch.matmul(self.N.t(), self.N), torch.eye(self.r).to(self.N.device), atol=1e-6), "N is not orthogonal"
            assert torch.allclose(torch.matmul(self.C, self.C.t()), torch.eye(self.r).to(self.C.device), atol=1e-6), "C is not orthogonal"
            if self.debug:
                print('orthogonality of N:', torch.mm(self.N, self.N.t()))
                print('orthogonality of C:', torch.mm(self.C, self.C.t()))
                
    def forward(self, x):
        if not self.allow_svd_values_negative:
            Sigma = self.Sigma.abs()
        else:
            Sigma = self.Sigma
            
        weights = torch.mm(self.N, torch.mm(torch.diag(Sigma), self.C))
        
        normalized_weights = weights*self.weight_correction_scale
        
        out = F.linear(x, normalized_weights, self.bias)
        
        if self.activation is not None:
            out = self.activation(out)
        
        return out
    
if __name__ == '__main__':
    # test SVD_Linear:
    in_dim = 10
    out_dim = 5
    batch_size = 3
    x = torch.randn(batch_size, in_dim)
    svd_linear = SVD_Linear(in_dim, out_dim)
    out = svd_linear(x)
    print(out.shape)
    print(out)
    print(svd_linear.N)
    print(svd_linear.C)
    print(svd_linear.Sigma)
    print(svd_linear.bias)
    print(svd_linear.weight_correction_scale)
    print(svd_linear.activation)
    print(svd_linear.allow_svd_values_negative)
    print(svd_linear.debug)
    print(svd_linear.r)
    
    # test SVD_Linear with activation:
    in_dim = 10
    out_dim = 5
    batch_size = 3
    x = torch.randn(batch_size, in_dim)
    svd_linear = SVD_Linear(in_dim, out_dim, activation='relu')
    out = svd_linear(x)
    print(out.shape)
    print(out)
    print(svd_linear.N)
    print(svd_linear.C)
    print(svd_linear.Sigma)
    print(svd_linear.bias)
    print(svd_linear.weight_correction_scale)
    print(svd_linear.activation)
    print(svd_linear.allow_svd_values_negative)
    print(svd_linear.debug)
    print(svd_linear.r)
    
    # test SVD_Linear with activation:
    in_dim = 10
    out_dim = 5
    batch_size = 3
    x = torch.randn(batch_size, in_dim)
    svd_linear = SVD_Linear(in_dim, out_dim, activation='relu', allow_svd_values_negative=True)
    out = svd_linear(x)
    print(out.shape)
    print(out)
    print(svd_linear.N)
    print(svd_linear.C)
    print('Sigm:', svd_linear.Sigma)
    print('Bias:', svd_linear.bias)
    print('weight correction: ', svd_linear.weight_correction_scale)
    print(svd_linear.activation)
    print(svd_linear.allow_svd_values_negative)
    print(svd_linear.debug)
    print(svd_linear.r)
    
    # test SVD_Linear with activation:
    in_dim = 10
    out_dim = 5
    batch_size = 3
    x = torch.randn(batch_size, in_dim)
    
        