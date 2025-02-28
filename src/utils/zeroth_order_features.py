# implement the following:
# features based only on zeroth order statistics of features.
import math
import itertools
from unicodedata import numeric
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

def compute_approximate_rank(input: torch.Tensor, input_is_svd:bool = False, prop:float=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    
    
    input: (torch.Tensor) an array of singular values or the tensor of shape (batch, m, n), batch of matrices.
    input_is_svd : (bool) whether the input is the svd of the matrix
    param prop: (float) proportion of the variance captured by the approximate rank
    return: (torch.Tensor) integer tensor the approximate rank
    
    """
    assert 0 < prop <= 1, "prop must be between 0 and 1"
    
    if input_is_svd:
        assert input.dim() in [1,2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be a matrix or a batch of matrices"
        sv  = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))
    
    assert sv.dim() in [1, 2], "Singular values must be 1d, or 2d for batch of singular values"
    if sv.dim() == 1:
        sv = sv.unsqueeze(0)# shape: (1, min(m, n))
    
    sqrd_sv = sv ** 2 # in batch form, shape (batch, min(m, n)) or (min(m, n))
    
    # normalization is done for each member in the batch.
    # this is different from the implementation in the implementation in https://github.com/shibhansh/loss-of-plasticity
    normed_sqrd_sv = sqrd_sv / (torch.sum(sqrd_sv, dim=-1, keepdim=True) + 1e-8)
    
    sorted_normed_sqrd_sv = torch.sort(normed_sqrd_sv,
                                dim= -1,
                                descending=True)[0]  # descending order
    
    cumulative_sums = torch.cumsum(sorted_normed_sqrd_sv, dim=-1)  # Shape: (batch, k)
    # Create a tensor of prop values matching the batch size
    batch_size = cumulative_sums.shape[0]
    prop_tensor = torch.full((batch_size,1), prop, device=cumulative_sums.device, dtype=cumulative_sums.dtype)
        
    
    approximate_rank = torch.searchsorted(cumulative_sums, prop_tensor, right=True) + 1
    # while cumulative_ns_sv_sum < prop:
    #     cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
    #     approximate_rank += 1
    
    # Squeeze output for non-batched input
    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        approximate_rank = approximate_rank.squeeze(0)  # integer
        
    return approximate_rank.to(torch.int32)
 
 
def compute_l1_distribution_rank(input: torch.Tensor, input_is_svd:bool = False, prop:float=0.99):
 
    """
    Computes the l1_distribution_rank as inspired by this paper, but : https://arxiv.org/pdf/1909.12255.pdf
    param:
        input: (torch.Tensor) an array of singular values or the tensor of shape (batch, m, n), batch of matrices.
        input_is_svd : (bool) whether the input is the svd of the matrix
        param prop: (float) proportion of the variance captured by the approximate rank
    return: (torch.Tensor) integer tensor the approximate rank
    
    """
    assert 0 < prop <= 1, "prop must be between 0 and 1"
    if input_is_svd:
        assert input.dim() in [1,2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be a matrix or a batch of matrices"
        sv  = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))
    
    assert sv.dim() in [1, 2], "Singular values must be 1d, or 2d for batch of singular values"
    if sv.dim() == 1:
        sv = sv.unsqueeze(0)# shape: (1, min(m, n))
    
    abs_sv = sv  # in batch form, shape (batch, min(m, n)) or (min(m, n))
    
    # normalization is done for each member in the batch.
    # this is different from the implementation in the implementation in https://github.com/shibhansh/loss-of-plasticity
    normed_abs_sv = abs_sv / (torch.sum(abs_sv, dim=-1, keepdim=True) + 1e-8)
    
    sorted_normed_abs_sv = torch.sort(normed_abs_sv,
                                dim= -1,
                                descending=True)[0]  # descending order
    
    cumulative_sums = torch.cumsum(sorted_normed_abs_sv, dim=-1)  # Shape: (batch, k)
    # Create a tensor of prop values matching the batch size
    batch_size = cumulative_sums.shape[0]
    prop_tensor = torch.full((batch_size,1), prop, device=cumulative_sums.device, dtype=cumulative_sums.dtype)
        
    
    abs_approximate_rank = torch.searchsorted(cumulative_sums, prop_tensor, right=True) + 1

    
    # Squeeze output for non-batched input
    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        abs_approximate_rank = abs_approximate_rank.squeeze(0)  # integer
        
    return abs_approximate_rank.to(torch.int32)


def compute_numerical_rank(input: torch.Tensor, input_is_svd: bool = False, epsilon: float = 1e-2) -> torch.Tensor:
    """
    Computes the numerical rank of a matrix or batch of matrices.

    The numerical rank is defined as:
    Rank(W)_ε = #{i ∈ ℕ₊ : i ≤ min(n, d), σ_i ≥ ε * ||W||₂},
    where ||W||₂ is the spectral norm (largest singular value) of the matrix W,
    and σ_i are the singular values in descending order.

    Args:
        input (torch.Tensor): 
            - If input_is_svd=True: Singular values of shape (k,) for a single matrix or (batch, k) for a batch.
            - If input_is_svd=False: Matrix of shape (m, n) or batch of matrices of shape (batch, m, n).
        input_is_svd (bool): 
            Whether the input is precomputed singular values (True) or a matrix/matrices (False). Default: False.
        epsilon (float): 
            Tolerance parameter for numerical rank computation. Must be positive. Default: 1e-6.

    Returns:
        torch.Tensor: 
            Numerical rank(s) as an int32 tensor. Scalar for non-batched input (1D singular values or 2D matrix),
            otherwise a 1D tensor of shape (batch,) for batched input.

    Raises:
        AssertionError: If epsilon ≤ 0, or if input dimensions are invalid.
    """
    # Ensure epsilon is positive
    assert epsilon > 0, "epsilon must be positive"

    # Handle input based on whether it's singular values or matrices
    if input_is_svd:
        # Input is singular values: must be 1D (single set) or 2D (batch)
        assert input.dim() in [1, 2], "Singular values must be 1D or 2D"
        sv = input
    else:
        # Input is matrices: must be 2D (single matrix) or 3D (batch)
        assert input.dim() in [2, 3], "Input must be 2D or 3D"
        sv = torch.linalg.svdvals(input)  # Compute singular values

    # Ensure sv is 2D: (batch_size, k)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0)  # Shape: (1, k)

    # Compute spectral norm (largest singular value for each matrix)
    # sv is sorted in descending order by svdvals, so first column is the largest
    spectral_norm = sv[:, 0]  # Shape: (batch_size,)

    # Compute threshold: ε * ||W||₂
    threshold = epsilon * spectral_norm  # Shape: (batch_size,)

    # Compare all singular values to the threshold (broadcasting)
    mask = sv >= threshold.unsqueeze(1)  # Shape: (batch_size, k)

    # Numerical rank is the count of singular values >= threshold per matrix
    numerical_rank = torch.sum(mask, dim=1)  # Shape: (batch_size,)

    # For non-batched input, return a scalar; otherwise keep as 1D tensor
    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        numerical_rank = numerical_rank.squeeze(0)  # Scalar tensor

    # Return as int32 tensor for consistency
    return numerical_rank.to(torch.int32)

    
if __name__ == "__main__":
    m = torch.randn(100, 100)
    sv = torch.linalg.svdvals(m)
    rank = torch.linalg.matrix_rank(m)
    effective_rank = compute_effective_rank(m)
    approximate_rank = compute_approximate_rank(m)
    abs_approximate_rank = compute_l1_distribution_rank(m)
    numerical_rank = compute_numerical_rank(m, epsilon= 0.1)
    
    print(rank, effective_rank, approximate_rank, abs_approximate_rank, numerical_rank)

    
    batched_m = torch.randn(32, 100, 100)
    sv = torch.linalg.svdvals(m)
    rank = torch.linalg.matrix_rank(m)
    effective_rank = compute_effective_rank(input=batched_m)
    approximate_rank = compute_approximate_rank(batched_m)
    abs_approximate_rank = compute_l1_distribution_rank(batched_m)
    numerical_rank = compute_numerical_rank(batched_m, epsilon= 0.1)
    print(rank, effective_rank, approximate_rank, abs_approximate_rank, numerical_rank)
