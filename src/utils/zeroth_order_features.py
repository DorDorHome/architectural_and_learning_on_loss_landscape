# implement the following:
# features based only on zeroth order statistics of features.
import math
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch

def compute_effective_rank(input: torch.Tensor, input_is_svd:bool = False, use_pytorch_entropy=True):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    input : (float torch Tensor) an array of singular values or the tensor of shape (batch, m, n), batch of matrices.
    input_is_svd : (bool) whether the input is the svd of the matrix    
    :return: (float torch Tensor) the effective rank
    """
    
    if input_is_svd:
        assert input.dim() in [1,2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be a matrix or a batch of matrices"
        sv  = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))
    
    assert sv.dim() in [1, 2], "Singular values must be 1d, or 2d for batch of singular values"
    
    sum_sv = torch.sum(torch.abs(sv), dim=-1, keepdim=True) + 1e-6
    norm_sv = sv / sum_sv
    
    if use_pytorch_entropy:
        # usually more efficient
        entropy = -torch.sum(norm_sv * torch.log(norm_sv + 1e-10), dim=-1)
        effective_rank = torch.exp(entropy)
    
    else:
        batch_size = sv.shape[0] if sv.dim() == 2 else 1
        
        entropy = torch.zeros(batch_size, dtype=torch.float32, device=sv.device)
        
        for i in range(batch_size):
            for p in norm_sv[i]:
                if p > 0.0:
                    entropy[i] -= p * torch.log(p)

        effective_rank = torch.tensor(np.e) ** entropy
    
    # Squeeze output for non-batched input
    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        effective_rank = effective_rank.squeeze(0)  # Scalar
        
        
    return effective_rank.to(input.dtype)


