# this implements several versions of single layer of FC layer, with weights normalized by the input activations

import torch.nn as nn
import torch
import math
from torch.nn import functional as F

class NormalizedWeightsLinear(nn.Module):
    """
    the weights are 
    compared to the norm Linear layer, this implementations normalise the
    weights before mulitplying with a constant
    
    Usage:
        When used to replace nn.Linear, activation should be set to False
        Bias should be False.
    """
    
    def __init__(self, in_dim, out_dim, bias = True, bias_init =0, activaton = None, test= False):
        super().__init__()
        
        self.weight=  nn.Parameter(torch.randn(out_dim, in_dim))#.div_(lr_mul)) options to multiply a user-defined constant in the future
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        
        else:
            self.bias = None
        
        self.weight_scale = nn.Parameter(2**0.5)
        
        self.activation = activation
        self.test = test
        

    def forward(self, input):
        #update weights of each layer 
        # only in training mode:
        if self.training:
            self.weight = self.weight*torch.rsqrt(torch.mean(self.weight**2, dim=1, keepdim = True)+ 1e-8)
        if self.test:
            #verify the norm of each layer is one:
            raise NotImplementedError
            
        # save memory:
        # self.scaled_weight = self.weight*self.weight_scale
        
        out = F.linear(input, self.weight*self.weight_scale, bias = self.bias)
        
        if self.activation =='relu':
            raise NotImplementedError
            
        return out
        

class BatchNormedWeightsLinear(nn.Module):
    """
    With an exponential decay, track the input activations.
    Based on the moving average of activations, the weights of the 
    FC layer will be scaled to have a certain norm, depending on the weight_correction_scale
    Usage:
        weight_correction_scale should be 2**0.5 when relu is being used afterwards
        
    
    """
    def __init__(self, in_dim, out_dim, bias = True, bias_init =0, alpha = 0.01 ,weight_correction_scale = 2**0.5, activaton = None, test= False):
        super().__init__()
        
        self.weight=  nn.Parameter(torch.randn(out_dim, in_dim))#.div_(lr_mul)) options to multiply a user-defined constant in the future
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        
        else:
            self.bias = None
        
        self.weight_scale = nn.Parameter(1)
        
        self.weight_correction_scale = weight_correction_scale
        
        self.scaled_weight = self.weight*self.weight_scale
        
        self.input_variance = torch.tensor(in_dim) #start with one, will get updated in forward pass
        self.alpha = alpha
        
        
    def forward(self, input):
        if self.test:
        # verify the shape of each item
            raise NotImplementedError
        # update the input_variance:
        if self.training:
            self.input_variance = (1-self.alpha)*self.input_variance + self.alpha*torch.norm(input, dim =0 )
            
            # update weight with input_variance and weight_correction_scale
            self.weight =(self.weight*self.weight_correction_scale)/self.input_variance
            
            
        self.scaled_weight = self.weight*self.weight_scale
        
        out = F.linear(input, self.scaled_weight, bias = self.bias)
        
        if self.activation =='relu':
            raise NotImplementedError
            
        return out
        
