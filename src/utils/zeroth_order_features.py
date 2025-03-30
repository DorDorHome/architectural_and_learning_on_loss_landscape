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

def compute_effective_rank(input: torch.Tensor, input_is_svd: bool = False, use_pytorch_entropy: bool = True, 
                          use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the effective rank as defined in: https://ieeexplore.ieee.org/document/7098875/
    When computing the Shannon entropy, 0 * log 0 is defined as 0.

    Args:
        input (torch.Tensor):
            - If input_is_svd=True: Singular values of shape (k,) for a single matrix.
            - If input_is_svd=False: Single matrix of shape (m, n).
        input_is_svd (bool): Whether the input is precomputed singular values. Default: False.
        use_pytorch_entropy (bool): Use vectorized entropy computation. Default: True.
        use_randomized_svd (bool): Use randomized SVD (torch.svd_lowrank) instead of full SVD. Default: True.
        q (int): Number of singular values to approximate with randomized SVD. Default: 20.
        niter (int): Number of subspace iterations for randomized SVD. Default: 2.
    
    Returns:
        torch.Tensor: The effective rank (scalar tensor).
    """
    if input_is_svd:
        assert input.dim() == 1, "Singular values must be 1d for a single matrix"
        sv = input
    else:
        assert input.dim() ==2 , "Input must be a matrix"

        m, n = input.shape
        q_adjusted = min(q, min(m, n))  # Cap q at min(m, n)        
        
    
        if use_randomized_svd:
            _, S, _ = torch.svd_lowrank(input, q=q_adjusted, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (min(m, n),)

    # compute relative epsilon to avoid numerical issues
    max_sv = torch.max(sv) if len(sv) > 0 else 0# largest singular value
    
    # return the rank as zero if max_sv is zero:    
    if max_sv == 0:
        return torch.tensor(0, dtype=input.dtype)
    
    #epsilon for division by zero:
    epsilon = max(1e-12, max_sv * 1e-8)  # avoid division by zero

    sum_sv = torch.sum(sv) + epsilon  # sum of singular values
    norm_sv = sv / sum_sv
    

    if use_pytorch_entropy:
        # usually more efficient
        log_epsilon = epsilon / sum_sv
        entropy = -torch.sum(norm_sv * torch.log(norm_sv + log_epsilon))
        effective_rank = torch.exp(entropy)
    
    else:
        mask = norm_sv > 0.0
        entropy = -torch.sum(norm_sv[mask] * torch.log(norm_sv[mask]))
        effective_rank = torch.exp(entropy)

    return effective_rank.to(input.dtype)



def compute_approximate_rank(input: torch.Tensor, input_is_svd: bool = False, prop: float = 0.99, 
                             use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    
    
    Args:
        input (torch.Tensor):
            - If input_is_svd=True: Singular values of shape (k,) for a single matrix.
            - If input_is_svd=False: Single matrix of shape (m, n).
        input_is_svd (bool): Whether the input is precomputed singular values. Default: False.
        prop (float): Proportion of variance captured by the approximate rank. Default: 0.99.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor: Approximate rank (scalar integer tensor).
    """
    
    assert 0 < prop <= 1, "prop must be between 0 and 1"
    if input_is_svd:
        assert input.dim() == 1, "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() ==2, "Input must be a matrix or a batch of matrices"
        # compute the svd of the input:
        m, n = input.shape
        q_adjusted = min(q, min(m, n))  # Cap q at min(m, n)        
        if use_randomized_svd:
            _, S, _ = torch.svd_lowrank(input, q=q_adjusted, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))
    

    
    sqrd_sv = sv ** 2
    
    # compute relative epsilon to avoid numerical issues
    max_sqrd_sv = torch.max(sqrd_sv) if len(sqrd_sv) > 0 else 0# largest singular value
    epsilon = max(1e-12, max_sqrd_sv * 1e-8)  # avoid division by zero
    sum_sqrd_sv = torch.sum(sqrd_sv) + epsilon  # sum of singular values
    # this is different from the implementation in the implementation in https://github.com/shibhansh/loss-of-plasticity

    normed_sqrd_sv = sqrd_sv / sum_sqrd_sv  # normalize the singular values
    
    sorted_normed_sqrd_sv = torch.sort(normed_sqrd_sv,
                                descending=True)[0]  # descending order
    
    cumulative_sums = torch.cumsum(sorted_normed_sqrd_sv, dim=0)

    approximate_rank = torch.searchsorted(cumulative_sums, prop, right=True) + 1
    # while cumulative_ns_sv_sum < prop:
    #     cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
    #     approximate_rank += 1
    
    return approximate_rank.to(input.dtype)
 
 
def compute_l1_distribution_rank(input: torch.Tensor, input_is_svd: bool = False, prop: float = 0.99,
                                 use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:

    """
    Computes the l1_distribution_rank as inspired by this paper, but : https://arxiv.org/pdf/1909.12255.pdf
    The difference is that we use the l1 norm of the singular values instead of the l2 norm when 
    computing the number of singular values needed to capture a certain proportion of the total singular value sum.
    
    Args:
        input (torch.Tensor):
            - If input_is_svd=True: Singular values of shape (k,) for a single matrix.
            - If input_is_svd=False: Single matrix of shape (m, n).
        input_is_svd (bool): Whether input is precomputed singular values. Default: False.
        prop (float): Proportion of total singular value sum to capture. Default: 0.99.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor: Integer tensor of rank(s).
    """
    assert 0 < prop <= 1, "prop must be between 0 and 1"
    if input_is_svd:
        assert input.dim() ==1, "Singular values must be 1d"
        sv = input
    else:
        assert input.dim() == 2, "Input must be a matrix"
        # compute the svd of the input:
        m, n = input.shape
        q_adjusted = min(q, min(m, n))  # Cap q at min(m, n)              
        if use_randomized_svd:
            U, S, V = torch.svd_lowrank(input, q=q_adjusted, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))
    
    # compute relative epsilon to avoid numerical issues
    max_sv = torch.max(sv) if len(sv) > 0 else 0# largest singular value
    epsilon = max(1e-12, max_sv * 1e-8)  # avoid division by zero
    sum_sv = torch.sum(sv) + epsilon  # sum of singular values
    
    

    # this is different from the implementation in the implementation in https://github.com/shibhansh/loss-of-plasticity
    normed_sv = sv / sum_sv  # normalize the singular values
    
    sorted_normed_sv = torch.sort(normed_sv,
                                descending=True)[0]  # descending order
    
    cumulative_sums = torch.cumsum(sorted_normed_sv, dim=0)

    abs_approximate_rank = torch.searchsorted(cumulative_sums, prop, right=True) + 1

    
    return abs_approximate_rank.to(input.dtype)


def compute_numerical_rank(input: torch.Tensor, input_is_svd: bool = False, epsilon: float = 1e-2, 
                           use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the numerical rank of a matrix or batch of matrices.

    The numerical rank is defined as:
    Rank(W)_ε = #{i ∈ ℕ₊ : i ≤ min(n, d), σ_i ≥ ε * ||W||₂},
    where ||W||₂ is the spectral norm (largest singular value) of the matrix W,
    and σ_i are the singular values in descending order.

    Args:
        input (torch.Tensor): 
            - If input_is_svd=True: Singular values of shape (k,) for a single matrix .
            - If input_is_svd=False: Matrix of shape (m, n).
        input_is_svd (bool): 
            Whether the input is precomputed singular values (True) or a matrix/matrices (False). Default: False.
        epsilon (float): Tolerance parameter. Default: 1e-2.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor:
            Numerical rank as an integer tensor, with the same dtype as the input. 
    Raises:
        AssertionError: If epsilon ≤ 0, or if input dimensions are invalid.
    """
    # Ensure epsilon is positive
    assert epsilon > 0, "epsilon must be positive"

    # Handle input based on whether it's singular values or matrices
    if input_is_svd:
        # Input is singular values: must be 1D (single set) or 2D (batch)
        assert input.dim() ==1, "Singular values must be 1d"
        sv = input#input is already the singular values
    else:
        # Input is matrices: must be 2D
        assert input.dim() ==2, "Input must be 2D"
        m, n = input.shape

        q_adjusted = min(q, min(m, n))  # Cap q at min(m, n)        
        
        if use_randomized_svd:
            _, S, _ = torch.svd_lowrank(input, q=q_adjusted, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)  # Compute singular values


    # Compute spectral norm (largest singular value for the matrix)
    # sv is sorted in descending order by svdvals, so first element is the largest
    spectral_norm = sv[0]  # a scalar

    # Compute threshold: ε * ||W||₂
    threshold = epsilon * spectral_norm  # a scalar

    # Compare all singular values to the threshold (broadcasting)
    mask = (sv >= threshold)

    # Numerical rank is the count of singular values >= threshold per matrix
    numerical_rank = torch.sum(mask) # A scalar

    # Return as int32 tensor for consistency
    return numerical_rank.to(input.dtype)



def compute_nuclear_norm(input: torch.Tensor, input_is_svd: bool = False, 
                         use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the nuclear norm as the sum of singular values.

    Args:
        input (torch.Tensor): Singular values (1D or 2D) if input_is_svd=True, or matrices (2D or 3D) otherwise.
        input_is_svd (bool): Whether input is precomputed singular values. Default: False.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor: Nuclear norm (scalar or batched).
    """
    if input_is_svd:
        assert input.dim() == 1, "Singular values must be 1d"
        sv = input
    else:
        assert input.dim() == 2,  "Input must be a matrix (2D)"
        m, n = input.shape
        q_adjusted = min(q, min(m, n))  # Cap q at min(m, n)        
        
        if use_randomized_svd:

            U, S, V = torch.svd_lowrank(input, q=q_adjusted, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)
    return sv.sum()# Sum over singular values, preserving batch dimension if present




    
    

    
if __name__ == "__main__":
    m = torch.randn(32, 100)
    # zero out the first row:
    m[0, :] = 0
    sv = torch.linalg.svdvals(m)
    rank = torch.linalg.matrix_rank(m)
    effective_rank = compute_effective_rank(m)
    approximate_rank = compute_approximate_rank(m)
    abs_approximate_rank = compute_l1_distribution_rank(m)
    numerical_rank = compute_numerical_rank(m, epsilon= 0.1)
    
    print(rank, effective_rank, approximate_rank, abs_approximate_rank, numerical_rank)

    
    # batched_m = torch.randn(32, 100, 100)
    # sv = torch.linalg.svdvals(batched_m)
    # rank = torch.linalg.matrix_rank(batched_m)
    # effective_rank = compute_effective_rank(input=batched_m)
    # approximate_rank = compute_approximate_rank(batched_m)
    # abs_approximate_rank = compute_l1_distribution_rank(batched_m)
    # numerical_rank = compute_numerical_rank(batched_m, epsilon= 0.1)
    # print("Rank:", rank)
    # print("\nEffective Rank:", effective_rank)
    # print("\nApproximate Rank:", approximate_rank)
    # print("\nL1 Distribution Rank:", abs_approximate_rank)
    # print("\nNumerical Rank:", numerical_rank)
  