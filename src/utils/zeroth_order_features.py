# implement the following:
# features based only on zeroth order statistics of features.
import math
import itertools
from unicodedata import numeric
from typing import List, Union, Dict
import numpy as np
from torch import nn
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch

def count_saturated_units_list(features_list, activation_type, threshold = 0.01):
    """
    Counts the number of saturated units for each feature tensor in a list.
        
    Args:
        features_list (list of torch.tensor): List of feature tensors, each of shape (batch_size, feature_dim)
        activation_type (str): Type of activation ('relu', 'leaky_relu', 'tanh', 'sigmoid', 'selu', 'elu', 'swish')
        threshold (float): Threshold for determining saturation
        
    Returns:
        list of int: Number of saturated units for each feature tensor
    """
    return [count_saturated_units(feature, activation_type, threshold = threshold) for feature in features_list]


def count_saturated_units(feature, activation_type, threshold = 0.01):
    """
    Counts the number of saturated units in a feature tensor based on activation type.
    A unit is saturated if all its outputs across the batch are in the saturation region.
    
    Args:
        feature (torch.tensor): Feature tensor of shape (batch_size, feature_dim)
        activation_type (str): Type of activation ('relu', 'sigmoid', 'tanh')
        
    Returns:
        int: Number of saturated units
    """
    if activation_type == 'relu':
        # Saturated when all outputs are 0
        saturated = torch.all((feature == 0), axis=0)
    elif activation_type == 'sigmoid':
        # Saturated when all outputs are < threshold or > 1-threshold
        saturated = torch.all((feature < threshold) | (feature > (1 - threshold)), axis=0)
    elif activation_type == 'tanh':
        # Saturated when all outputs are < -(1 - threshold) or > (1 - threshold)
        saturated = torch.all((feature < -1*(1-threshold)) | (feature > (1 -threshold)), axis=0)
    elif activation_type == 'leaky_relu':
    # For Leaky ReLU, no true saturation, but we can consider very small outputs
    # Here, we might not define saturation or use a small threshold, e.g., |activation| < 1e-5
    # For this example, we'll assume no saturation for simplicity
        saturated = torch.zeros(feature.shape[1], dtype=bool)
    elif activation_type == 'selu':
        # SELU approaches -λ*α ≈ -1.758 for large negative inputs
        # Consider saturation when all outputs are < -1.7
        saturated = torch.all(feature < -1.7, axis=0)
    elif activation_type == 'elu':
        # ELU approaches -α for large negative inputs
        # Assuming α=1, consider saturation when all outputs are < -0.99
        saturated = torch.all(feature < -1*(1-threshold), axis=0)
    elif activation_type == 'swish':
        # Swish: f(x) = x * sigmoid(x)
        # For large |x|, swish(x) ≈ x if x>0, ≈0 if x<0
        # Saturation might be considered for large positive x where gradient approaches 0, but it's not straightforward
        # For simplicity, we might not define saturation or use a threshold for very large positive values
        # Here, we'll assume no saturation for swish
        saturated = torch.zeros(feature.shape[1], dtype=bool)      
    else:
        raise ValueError("Unsupported activation type. Use 'relu', 'sigmoid', or 'tanh'.")
    
    return torch.sum(saturated)


def compute_all_rank_measures_list(features: list[torch.Tensor], use_pytorch_entropy_for_effective_rank: bool = True, 
                                   prop_for_approx_or_l1_rank: float = 0.99, numerical_rank_epsilon: float = 1e-2) -> list[dict]:
    """
    Computes rank measures for a list of feature matrices.
    
    Args:
        features (list[torch.Tensor]): List of matrices, each [batch_size, feature_size].
        use_pytorch_entropy (bool): Use PyTorch entropy for effective rank.
        prop (float): Proportion for approximate and L1 ranks.
        epsilon (float): Threshold for numerical rank.
    
    Returns:
        list[dict]: List of rank measure dictionaries, one per matrix.
    """
    return [compute_all_rank_measures(f, use_pytorch_entropy_for_effective_rank, prop_for_approx_or_l1_rank, numerical_rank_epsilon) for f in features]




def compute_all_rank_measures(input: torch.Tensor, use_pytorch_entropy: bool = True, 
                              prop_for_approx_or_l1_rank: float = 0.99, numerical_rank_epsilon: float = 1e-2) -> dict:
    """
    Computes four rank measures for the input matrix using full SVD:
    - Effective rank
    - Approximate rank
    - L1 distribution rank
    - Numerical rank

    The singular values are computed once using full SVD, then passed to each rank measure function.

    Args:
        input (torch.Tensor): Input matrix of shape (m, n). (expected usage: [batch_size, feature_dim])
        use_pytorch_entropy (bool): Use PyTorch's vectorized entropy computation for effective rank. Default: True.
        prop (float): Proportion of variance or sum to capture for approximate and L1 distribution ranks. Default: 0.99.
        epsilon (float): Tolerance for numerical rank. Default: 1e-2.

    Returns:
        dict: A dictionary containing the four rank measures:
            - "effective_rank": Effective rank (float tensor)
            - "approximate_rank": Approximate rank (tensor with input.dtype)
            - "l1_distribution_rank": L1 distribution rank (tensor with input.dtype)
            - "numerical_rank": Numerical rank (tensor with input.dtype)

    Raises:
        AssertionError: If input is not a 2D matrix or if prop is not between 0 and 1.
    """
    # Validate input
    assert input.dim() == 2, "Input must be a 2D matrix"
    
    # Check for non-finite values (NaN or Inf) which can cause SVD to fail
    if not torch.isfinite(input).all():
        print(f"Warning: Matrix contains non-finite values (NaN or Inf). Returning zero ranks.")
        print(f"Matrix shape: {input.shape}")
        print(f"Non-finite elements: {(~torch.isfinite(input)).sum().item()}")
        return {
            "effective_rank": torch.tensor(0.0, dtype=input.dtype),
            "approximate_rank": torch.tensor(0, dtype=input.dtype),
            "l1_distribution_rank": torch.tensor(0, dtype=input.dtype),
            "numerical_rank": torch.tensor(0, dtype=input.dtype)
        }
    
    # Check if the matrix is effectively zero
    if torch.allclose(input, torch.zeros_like(input), atol=1e-8):
        return {
            "effective_rank": torch.tensor(0.0, dtype=input.dtype),
            "approximate_rank": torch.tensor(0, dtype=input.dtype),
            "l1_distribution_rank": torch.tensor(0, dtype=input.dtype),
            "numerical_rank": torch.tensor(0, dtype=input.dtype)
        }
    
    # Try SVD with error handling for numerical issues
    try:
        # First try regular SVD
        sv = torch.linalg.svdvals(input)  # Shape: (min(m, n),)
    except RuntimeError as e:
        if "svd_cuda" in str(e) and "singular" in str(e):
            print(f"SVD failed for matrix with shape {input.shape}. This indicates structural numerical issues.")
            print(f"Matrix statistics: min={input.min():.2e}, max={input.max():.2e}, mean={input.mean():.2e}, std={input.std():.2e}")
            print(f"Matrix Frobenius norm: {torch.linalg.norm(input):.2e}")
            # Check if any row or column is all zeros (which can cause SVD issues)
            zero_rows = (input == 0).all(dim=1).sum()
            zero_cols = (input == 0).all(dim=0).sum()
            print(f"Zero rows: {zero_rows}, Zero columns: {zero_cols}")
            
            # Try alternative: add small regularization and retry
            print("Attempting regularized SVD...")
            try:
                # Add tiny regularization to diagonal
                reg_matrix = input.clone()
                min_dim = min(input.shape)
                regularization = 1e-8 * torch.eye(min_dim, device=input.device, dtype=input.dtype)
                if input.shape[0] >= input.shape[1]:
                    # Add to A^T A
                    reg_matrix = input + torch.zeros_like(input)
                    # Try computing SVD of regularized version
                    sv = torch.linalg.svdvals(reg_matrix + 1e-10)
                else:
                    # For wide matrices, add to AA^T
                    reg_matrix = input + torch.zeros_like(input)
                    sv = torch.linalg.svdvals(reg_matrix + 1e-10)
                print("Regularized SVD succeeded!")
            except:
                print("Regularized SVD also failed. Returning zero ranks.")
                return {
                    "effective_rank": torch.tensor(0.0, dtype=input.dtype),
                    "approximate_rank": torch.tensor(0, dtype=input.dtype),
                    "l1_distribution_rank": torch.tensor(0, dtype=input.dtype),
                    "numerical_rank": torch.tensor(0, dtype=input.dtype)
                }
        else:
            raise  # Re-raise if it's a different error

    # Compute each rank measure using the precomputed singular values
    effective_rank = compute_effective_rank(sv, input_is_svd=True, use_pytorch_entropy=use_pytorch_entropy)
    approximate_rank = compute_approximate_rank(sv, input_is_svd=True, prop=prop_for_approx_or_l1_rank)
    l1_distribution_rank = compute_l1_distribution_rank(sv, input_is_svd=True, prop=prop_for_approx_or_l1_rank)
    numerical_rank = compute_numerical_rank(sv, input_is_svd=True, epsilon=numerical_rank_epsilon)

    # Return results in a dictionary
    return {
        "effective_rank": effective_rank,
        "approximate_rank": approximate_rank,
        "l1_distribution_rank": l1_distribution_rank,
        "numerical_rank": numerical_rank
    }

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




def compute_rank_decay_dynamics(ranks: Union[List[Union[float, int]], torch.Tensor], 
                              mode='difference',
                              decay_from_input=True,
                              assume_full_rank_input=True,
                              input_rank=None) -> Dict:
    """
    Computes metrics to characterize the dynamics of rank decay across layers.

    Args:
        ranks (list[float | int] | torch.Tensor): A sequence of rank values for L layers.
        mode (str): 'difference' or 'ratio' to specify how drops are calculated.
        decay_from_input (bool): Whether to include input rank in the analysis. Default: True.
        assume_full_rank_input (bool): If decay_from_input=True, whether to assume input has full rank
                                     (i.e., rank = batch_size). Default: True.
        input_rank (float | int | None): Explicit input rank value. If provided, overrides assume_full_rank_input.

    Returns:
        dict: A dictionary containing the rank decay metrics:
            - "rank_drop_gini": Gini coefficient of the rank drops. 0 is perfectly even, 1 is max inequality.
            - "rank_decay_centroid": The "center of mass" of the rank drop, from 1 to L (or L+1).
            - "normalized_aurc": Normalized Area Under the Rank Curve. Close to 1 means rank stays high.
    """
    if isinstance(ranks, torch.Tensor):
        ranks = ranks.tolist()
    
    # Handle the input rank
    if decay_from_input:
        if input_rank is not None:
            # Use explicit input rank
            ranks_from_input = [float(input_rank)] + list(ranks)
        elif assume_full_rank_input:
            # This is problematic - we need to know the batch size to assume full rank
            # The function should require input_rank when assume_full_rank_input=True
            raise ValueError("When decay_from_input=True and assume_full_rank_input=True, "
                           "you must provide input_rank explicitly. "
                           "We cannot assume full rank without knowing the batch size.")
        else:
            # decay_from_input=True but assume_full_rank_input=False and no explicit input_rank
            raise ValueError("When decay_from_input=True and assume_full_rank_input=False, "
                           "you must provide input_rank explicitly.")
    else:
        # Use original behavior - start from first intermediate features
        ranks_from_input = list(ranks)
    
    if len(ranks_from_input) < 2:
        return {
            "rank_drop_gini": 0.0,
            "rank_decay_centroid": 0.0,
            "normalized_aurc": 0.0
        }

    ranks_from_input = np.array(ranks_from_input, dtype=np.float64)
    
    # Calculate drops based on the chosen mode
    if mode == 'difference':
        drops = ranks_from_input[:-1] - ranks_from_input[1:]
    elif mode == 'ratio':
        # Add a small epsilon to avoid division by zero for ranks
        safe_ranks = ranks_from_input + 1e-9
        drops = (safe_ranks[:-1] - safe_ranks[1:]) / safe_ranks[:-1]
    else:
        raise ValueError("Mode must be either 'difference' or 'ratio'")

    total_drop = np.sum(drops)

    if total_drop < 1e-9:  # No drop or negligible drop
        return {
            "rank_drop_gini": 0.0,
            "rank_decay_centroid": (len(ranks_from_input) - 1) / 2.0,  # Centered if no drop
            "normalized_aurc": ranks_from_input[0] / (ranks_from_input[0] + 1e-9) if ranks_from_input[0] > 0 else 0.0
        }

    # 2. Rank Drop Gini Coefficient
    n = len(drops)
    # Using a more efficient formula for Gini: (Σ (2i - n - 1) * x_i) / (n * Σ x_i) for sorted x
    sorted_drops = np.sort(drops)
    gini_numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_drops)
    gini_denominator = n * total_drop
    gini_coefficient = gini_numerator / gini_denominator

    # 3. Rank Decay Centroid
    # Indices now represent transitions: 1 = input→layer1, 2 = layer1→layer2, etc.
    indices = np.arange(1, n + 1)
    centroid = np.sum(indices * drops) / total_drop

    # 4. Normalized Area Under the Rank Curve (AURC)
    raw_aurc = np.sum(0.5 * (ranks_from_input[:-1] + ranks_from_input[1:]))
    norm_aurc = raw_aurc / ((len(ranks_from_input) - 1) * (ranks_from_input[0] + 1e-9)) if ranks_from_input[0] > 0 else 0.0

    return {
        "rank_drop_gini": gini_coefficient,
        "rank_decay_centroid": centroid,
        "normalized_aurc": norm_aurc
    }
    
if __name__ == "__main__":
    # m = torch.randn(100, 100)
    # # zero out the first row:
    # m[0, :] = 0
    # sv = torch.linalg.svdvals(m)
    # rank = torch.linalg.matrix_rank(m)
    # effective_rank = compute_effective_rank(m, input_is_svd=False, use_randomized_svd=False)
    # approximate_rank = compute_approximate_rank(m, input_is_svd=False, use_randomized_svd=False)
    # abs_approximate_rank = compute_l1_distribution_rank(m, input_is_svd=False, use_randomized_svd=False)
    # numerical_rank = compute_numerical_rank(m, epsilon= 0.1, input_is_svd=False, use_randomized_svd=False)
    
    # print(rank, effective_rank, approximate_rank, abs_approximate_rank, numerical_rank)

    # # test the compute_all_rank_measures function:
    # rank_measures = compute_all_rank_measures(m)
    # #print results:
    # print("Rank measures using computea_all_rank_measures function:")
    # for key, value in rank_measures.items():
    #     print(f"{key}: {value}")
    # # test the compute_nuclear_norm function:
    # nuclear_norm = compute_nuclear_norm(m)
    # print("Nuclear norm:", nuclear_norm)
    
    
    #test the compute_all_rank_measures_list function:
    # create a list of matrices:
    features = [torch.randn(100, 100) for _ in range(10)]
    # compute the rank measures for each matrix:
    rank_measures_list = compute_all_rank_measures_list(features)
    #print results:
    print('\n')

    print('comparing results:')
    m = features[0]
    rank_measures = compute_all_rank_measures(m)
    #print results:
    print("Rank measures using compute_all_rank_measures function:")
    for key, value in rank_measures.items():
        print(f"{key}: {value}")
    
    effective_rank = compute_effective_rank(m, input_is_svd=False, use_randomized_svd=False)
    approximate_rank = compute_approximate_rank(m, input_is_svd=False, use_randomized_svd=False)
    abs_approximate_rank = compute_l1_distribution_rank(m, input_is_svd=False, use_randomized_svd=False)
    numerical_rank = compute_numerical_rank(m, epsilon= 0.1, input_is_svd=False, use_randomized_svd=False)
    
    print( effective_rank, approximate_rank, abs_approximate_rank, numerical_rank)

    
    # print("Rank measures using compute_all_rank_measures_list function:")
    # for i, rank_measures in enumerate(rank_measures_list):
    #     print(f"Matrix {i}:")
    #     for key, value in rank_measures.items():
    #         print(f"{key}: {value}")

    
    
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
  
    # test the count_saturated_units function:
    # Example feature tensor (batch_size=3, feature_dim=4)
    # Example feature tensor (batch_size=3, feature_dim=4)
    feature = torch.tensor([[0, 1, 0, 0],
                        [0, 2, 0, 1],
                        [0, 3, 0, 0]])

    # For ReLU
    print(count_saturated_units(feature, 'relu'))  # Output: 2 (columns 0 and 2 are all zeros)

    # For sigmoid (assuming feature is between 0 and 1)
    sigmoid_feature = torch.tensor([[0.005, 0.5, 0.995, 0.002],
                                [0.003, 0.6, 0.999, 0.001],
                                [0.007, 0.4, 0.998, 0.004]])
    print(count_saturated_units(sigmoid_feature, 'sigmoid'))  # Counts units all < 0.01 or all > 0.99

    # For tanh (assuming feature is between -1 and 1)
    tanh_feature = torch.tensor([[-0.995, 0.5, 0.999, -0.8],
                            [-0.999, -0.6, 0.995, 0.7],
                            [-0.998, 0.4, 0.997, -0.9]])
    print(count_saturated_units(tanh_feature, 'tanh'))  # Counts units all < -0.99 or all > 0.99