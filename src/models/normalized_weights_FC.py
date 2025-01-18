# this implements several versions of single layer of FC layer, with weights normalized by the input activations
# to add: options not to use fan_in correction
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



class NormalizedWeightsLinear(nn.Module):
    """
    the weights are 
    compared to the norm Linear layer, this implementations normalise the
    weights before mulitplying with a constant
    
    Usage:
        When used to replace nn.Linear, activation should be set to False
        Bias should be None.
    """
    
    def __init__(self, in_dim:int, out_dim:int, bias:bool = True, bias_init:float =0, activation: str = None, weight_correction_scale:float = 2**0.5,  debug= False):
        super().__init__()
        
        self.weight=  nn.Parameter(torch.randn(out_dim, in_dim))#.div_(lr_mul)) options to multiply a user-defined constant in the future
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        
        else:
            self.bias = None
        
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_correction_scale = weight_correction_scale/(in_dim**0.5)# check
        
        
        if (activation is not None) and (activation != 'linear'):
            self.activation = activation_dict.get(activation, None)   
            self.activation = self.activation()
        if (activation is None) or (activation == 'linear'):
            self.activation = None
        self.debug = debug
        
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #update weights of each layer 
            
            # # in_place version, only in training mode:
            # self.weight = self.weight*torch.rsqrt(torch.mean(self.weight**2, dim=1, keepdim = True)+ 1e-8)
            
        # Compute the norm over the input dimension (dim=1)
        weight_norm_inverse = torch.rsqrt(torch.mean(self.weight ** 2, dim=1, keepdim=True) + 1e-8)
        normalized_weight = self.weight * weight_norm_inverse


        if self.debug:
            #verify the norm of each layer is one:
            raise NotImplementedError
            
        # save memory:
        # self.scaled_weight = self.weight*self.weight_scale
        
        out = F.linear(input, normalized_weight*self.weight_scale*self.weight_correction_scale, bias = self.bias)
        
        # Apply activation if specified
        if self.activation:
            out = self.activation(out)
        return out
        
class NormalizedRescalePerChannelWeightsLinear(nn.Module):
    """
    This is an implementation that not just normalizes the weights per channel,
    But has a separate learning scaling paremeter for each channel.
    
    """
    def __init__(self, in_dim:int, out_dim:int, bias:bool = True, bias_init:float =0, activation: str = None, weight_correction_scale:float = 2**0.5,  debug= False):
        super().__init__()
        
        self.weight=  nn.Parameter(torch.randn(out_dim, in_dim))#.div_(lr_mul)) options to multiply a user-defined constant in the future
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        
        else:
            self.bias = None
        
        self.weight_scale = nn.Parameter(torch.ones(out_dim))
        self.weight_correction_scale = weight_correction_scale# /(in_dim**0.5)# checked, doesn't work. Why? 
        
        
        if (activation is not None) and (activation != 'linear'):
            self.activation = activation_dict.get(activation, None)   
            self.activation = self.activation()
        if (activation is None) or (activation == 'linear'):
            self.activation = None
        self.debug = debug
        
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #update weights of each layer 
            
            # # in_place version, only in training mode:
            # self.weight = self.weight*torch.rsqrt(torch.mean(self.weight**2, dim=1, keepdim = True)+ 1e-8)
            
        # Compute the norm over the input dimension (dim=1)
        weight_norm_inverse = torch.rsqrt(torch.sum(self.weight ** 2, dim=1, keepdim=True) + 1e-8)
        #1/(torch.norm(self.weight, dim=1, keepdim=True) + 1e-8)
        #torch.rsqrt(torch.mean(self.weight ** 2, dim=1, keepdim=True) + 1e-8)
        normalized_weight = self.weight * weight_norm_inverse


        if self.debug:
            #verify the norm of each layer is one:
            raise NotImplementedError
            
        # save memory:
        # self.scaled_weight = self.weight*self.weight_scale
        
        out = F.linear(input, self.weight_scale.view(-1, 1)*normalized_weight*self.weight_correction_scale, bias = self.bias)
        
        # Apply activation if specified
        if self.activation:
            out = self.activation(out)
        return out


class BatchNormedWeightsLinear(nn.Module):
    """
    With an exponential decay, track the input activations.
    Based on the moving average of activations, the weights of the 
    FC layer will be scaled to have a certain norm, depending on the weight_correction_scale
    Usage:
        weight_correction_scale should be 2**0.5 when relu is being used afterwards
        
    
    """
    def __init__(self, in_dim:int, out_dim:int, bias:bool = True,
                 bias_init:float =0, alpha:float = 0.01 ,
                 weight_correction_scale:float = 2**0.5, 
                 activation:str = None, debug:bool= False):
        super().__init__()
        
        self.weight=  nn.Parameter(torch.randn(out_dim, in_dim))#.div_(lr_mul)) options to multiply a user-defined constant in the future
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        
        else:
            self.bias = None
        
        self.weight_scale = nn.Parameter(torch.ones(out_dim))# check dimension
        
        self.weight_correction_scale = weight_correction_scale/(in_dim**0.5)# check

        # compute on the fly instead, not needed to be stored
        # self.scaled_weight = self.weight*self.weight_scale
        
        # Initialize moving average of input variance
        self.register_buffer('input_variance_mean_across_dim', torch.ones(1)) #start with one, will get updated in forward pass
        self.register_buffer('input_mean_mean_across_dim', torch.ones(1))
        self.alpha = alpha
        if (activation is not None) and (activation != 'linear'):
            self.activation = activation_dict.get(activation, None)   
            self.activation = self.activation()
        if (activation is None) or (activation == 'linear'):
            self.activation = None
        self.debug = debug
        
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.debug:
        # verify the shape of each item
            raise NotImplementedError
        # update the input_variance:
        if self.training:
            # # Compute variance across the batch for each input dimension
            # input_batch_variance = torch.var(input, dim=0, unbiased=False)  # Shape: (in_dim,)
            # # Compute the mean variance across input dimensions
            # input_batch_variance_mean_across_dim = batch_variance.mean()
            
            # mean across batch
            input_batch_mean = torch.mean(input, dim=0) # shape (in_dim)
            
            input_batch_mean_mean_across_dim = torch.mean(input_batch_mean) 
            #self.input_variance_mean_across_dim = (1-self.alpha)*self.input_variance_mean_across_dim + self.alpha*input_batch_variance_mean_across_dim
            with torch.no_grad():
                self.input_mean_mean_across_dim = (1-self.alpha)*self.input_mean_mean_across_dim + self.alpha*input_batch_mean_mean_across_dim
                # self.input_variance.copy_((1 - self.alpha) * self.input_variance + self.alpha * batch_variance_mean)
            # perhaps compared to inplace version in the future:
            # update weight with input_variance and weight_correction_scale
            # self.weight =self.weight*self.weight_correction_scale / std_dev)
                # self.
            
            
        # Normalize the weights based on the moving average of input variance
        # std_dev = torch.sqrt(self.input_variance + 1e-8)
        
        # if correct by mean:
        # scaled_corrected_weight = self.weight*self.weight_scale*(self.weight_correction_scale / std_dev)
        # if correct by mean:
        scaled_corrected_weight = self.weight*self.weight_scale.view(-1, 1)*(self.weight_correction_scale / self.input_mean_mean_across_dim)
        
        out = F.linear(input, scaled_corrected_weight, bias = self.bias)
        
        # Apply activation if specified
        if self.activation:
            out = self.activation(out)
        return out
        
def test_normalized_weights_linear():
    device = 'cuda'
    layer = NormalizedWeightsLinear(in_dim=10, out_dim=5, activation='relu').to(device)
    input = torch.randn(3, 10, requires_grad=True).to(device)
    output = layer(input)
    assert output.shape == (3, 5), "Output shape mismatch."
    
    # Check weight normalization
    with torch.no_grad():
        weight_norm = torch.rsqrt(torch.mean(layer.weight ** 2, dim=1, keepdim=True) + 1e-8)
        normalized_weight = layer.weight * weight_norm
        # Tolerance level can be adjusted
        assert torch.allclose(layer.weight * weight_norm, normalized_weight, atol=1e-6), "Weights are not normalized correctly."
    
    # Check gradient flow
    loss = output.sum()
    loss.backward()
    assert layer.weight.grad is not None, "Gradients not flowing to weights."

def test_batch_normed_weights_linear():
    in_dim = 100
    out_dim = 50
    batch_size = 1024  # Large enough batch size
    model = BatchNormedWeightsLinear(in_dim, out_dim, activation='relu')
    
    # Generate input z with unit variance Gaussian with zero mean
    z = torch.randn(batch_size, in_dim)
    
    # Apply ReLU to z
    input = F.relu(z)
    
    # Forward pass through the model
    output = model(input)
    
    # Calculate mean and variance of the output
    output_mean = output.mean().item()
    output_var = output.var().item()
    
    print(f"Output Mean: {output_mean}")
    print(f"Output Variance: {output_var}")

    # Check if the output has zero mean and unit variance
    assert abs(output_mean) < 1e-2, f"Output mean is not close to zero: {output_mean}"
    assert abs(output_var - 1.0) < 1e-2, f"Output variance is not close to one: {output_var}"

# Run the test



if __name__ == "__main__":
    test_normalized_weights_linear()
    test_batch_normed_weights_linear()
    print("All tests passed")