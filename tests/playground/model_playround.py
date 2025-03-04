import torch
import numpy as np

def compute_effective_rank(input: torch.Tensor, input_is_svd: bool = False, use_pytorch_entropy: bool = True, 
                          use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the effective rank as defined in https://ieeexplore.ieee.org/document/7098875/.
    When computing the Shannon entropy, 0 * log 0 is defined as 0.

    Args:
        input (torch.Tensor): Singular values (1D or 2D) if input_is_svd=True, or matrices (2D or 3D) otherwise.
        input_is_svd (bool): Whether input is precomputed singular values. Default: False.
        use_pytorch_entropy (bool): Use vectorized entropy computation. Default: True.
        use_randomized_svd (bool): Use randomized SVD (torch.svd_lowrank) instead of full SVD. Default: True.
        q (int): Number of singular values to approximate with randomized SVD. Default: 20.
        niter (int): Number of subspace iterations for randomized SVD. Default: 2.

    Returns:
        torch.Tensor: Effective rank (scalar for non-batched, 1D for batched).
    """
    if input_is_svd:
        assert input.dim() in [1,2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be 2D or 3D"
        if use_randomized_svd:
            U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))

    if sv.dim() == 1:
        sv = sv.unsqueeze(0)

    sum_sv = torch.sum(sv, dim=-1, keepdim=True) + 1e-6
    norm_sv = sv / sum_sv

    if use_pytorch_entropy:
        entropy = -torch.sum(norm_sv * torch.log(norm_sv + 1e-10), dim=-1)
        effective_rank = torch.exp(entropy)
    else:
        batch_size = sv.shape[0]
        entropy = torch.zeros(batch_size, dtype=torch.float32, device=sv.device)
        for i in range(batch_size):
            for p in norm_sv[i]:
                if p > 0.0:
                    entropy[i] -= p * torch.log(p)
        effective_rank = torch.tensor(np.e) ** entropy

    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        effective_rank = effective_rank.squeeze(0)  # Scalar

    return effective_rank.to(input.dtype)

def compute_approximate_rank(input: torch.Tensor, input_is_svd: bool = False, prop: float = 0.99, 
                             use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the approximate rank as defined in https://arxiv.org/pdf/1909.12255.pdf.

    Args:
        input (torch.Tensor): Singular values (1D or 2D) if input_is_svd=True, or matrices (2D or 3D) otherwise.
        input_is_svd (bool): Whether input is precomputed singular values. Default: False.
        prop (float): Proportion of variance to capture. Default: 0.99.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor: Integer tensor of approximate rank(s).
    """
    assert 0 < prop <= 1, "prop must be between 0 and 1"
    if input_is_svd:
        assert input.dim() in [1,2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be 2D or 3D"
        if use_randomized_svd:
            _, S, _ = torch.svd_lowrank(input, q=q, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))

    if sv.dim() == 1:
        sv = sv.unsqueeze(0)# shape: (1, min(m, n))

    sqrd_sv = sv ** 2  # in batch form, shape (batch, min(m, n)) or (min(m, n))
    
    # normalization is done for each member in the batch.
    # this is different from the implementation in the implementation in https://github.com/shibhansh/loss-of-plasticity
    normed_sqrd_sv = sqrd_sv / (torch.sum(sqrd_sv, dim=-1, keepdim=True) + 1e-8)

    sorted_normed_sqrd_sv = torch.sort(normed_sqrd_sv,
                                       dim= -1,
                                       descending=True)[0]  # descending order
    

    cumulative_sums = torch.cumsum(sorted_normed_sqrd_sv, dim=-1)
    batch_size = cumulative_sums.shape[0]
    prop_tensor = torch.full((batch_size, 1), prop, device=cumulative_sums.device, dtype=cumulative_sums.dtype)
    approximate_rank = torch.searchsorted(cumulative_sums, prop_tensor, right=True) + 1

    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        approximate_rank = approximate_rank.squeeze(0)  # integer

    return approximate_rank.to(torch.int32)

def compute_l1_distribution_rank(input: torch.Tensor, input_is_svd: bool = False, prop: float = 0.99,
                                 use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the L1 distribution rank, inspired by but distinct from https://arxiv.org/pdf/1909.12255.pdf.

    Args:
        input (torch.Tensor): Singular values (1D or 2D) if input_is_svd=True, or matrices (2D or 3D) otherwise.
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
        assert input.dim() in [1, 2], "Singular values must be 1d, or 2d for batch of singular values"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be a matrix or a batch of matrices"
        if use_randomized_svd:
            U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)# shape: (batch, min(m, n)) or (min(m, n))

    if sv.dim() == 1:
        sv = sv.unsqueeze(0)

    normed_abs_sv = sv / (torch.sum(sv, dim=-1, keepdim=True) + 1e-8)
    sorted_normed_abs_sv = torch.sort(normed_abs_sv,
                                      dim=-1,
                                      descending=True)[0]  # descending order

    cumulative_sums = torch.cumsum(sorted_normed_abs_sv, dim=-1)  # Shape: (batch, k)
    # Create a tensor of prop values matching the batch size
    batch_size = cumulative_sums.shape[0]
    prop_tensor = torch.full((batch_size, 1), prop, device=cumulative_sums.device, dtype=cumulative_sums.dtype)
    abs_approximate_rank = torch.searchsorted(cumulative_sums, prop_tensor, right=True) + 1
    # Squeeze output for non-batched input
    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        abs_approximate_rank = abs_approximate_rank.squeeze(0)  # integer

    return abs_approximate_rank.to(torch.int32)

def compute_numerical_rank(input: torch.Tensor, input_is_svd: bool = False, epsilon: float = 1e-2, 
                           use_randomized_svd: bool = True, q: int = 20, niter: int = 2) -> torch.Tensor:
    """
    Computes the numerical rank as Rank(W)_ε = #{i : σ_i ≥ ε * ||W||₂}.

    Args:
        input (torch.Tensor): Singular values (1D or 2D) if input_is_svd=True, or matrices (2D or 3D) otherwise.
        input_is_svd (bool): Whether input is precomputed singular values. Default: False.
        epsilon (float): Tolerance parameter. Default: 1e-2.
        use_randomized_svd (bool): Use randomized SVD. Default: True.
        q (int): Number of singular values to approximate. Default: 20.
        niter (int): Number of subspace iterations. Default: 2.

    Returns:
        torch.Tensor: Integer tensor of rank(s).
    """
    assert epsilon > 0, "epsilon must be positive"
    if input_is_svd:
        assert input.dim() in [1, 2], "Singular values must be 1D or 2D"
        sv = input
    else:
        assert input.dim() in [2, 3], "Input must be 2D or 3D"
        if use_randomized_svd:
            U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)


    # Ensure sv is 2D: (batch_size, k)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0)  # Shape: (1, k)

    spectral_norm = sv[:, 0]
    threshold = epsilon * spectral_norm  # Shape: (batch_size,)
    mask = sv >= threshold.unsqueeze(1)  # Shape: (batch_size, k)
    numerical_rank = torch.sum(mask, dim=1)  # Shape: (batch_size,)


    if input.dim() == 1 or (not input_is_svd and input.dim() == 2):
        numerical_rank = numerical_rank.squeeze(0)  # Scalar tensor
    # Return as int32 tensor for consistency
    return numerical_rank.to(torch.int32)

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
        sv = input
    else:
        if use_randomized_svd:
            U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
            sv = S
        else:
            sv = torch.linalg.svdvals(input)
    return sv.sum(dim = -1)# Sum over singular values, preserving batch dimension if present