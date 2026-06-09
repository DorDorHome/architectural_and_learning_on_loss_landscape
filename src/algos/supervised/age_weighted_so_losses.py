"""Pure-function age-weighted soft-orthogonality losses (ASO and FASO).

These subroutines mirror the pseudocode in
`.cursor/specs/ssr_cbp_aso_and_faso_algorithms.md` (sections 3.4 and 3.2.1).

The caller is responsible for the autograd-isolation step described in
the spec: `A_reg` MUST be computed as `W @ stop_gradient(H)` (i.e.,
``F.linear(H.detach(), W, bias)`` for FC layers or
``F.conv2d(H.detach(), W, bias, ...)`` reshaped to ``[N, m]`` for
conv layers). With that contract these losses produce gradients
strictly w.r.t. the current layer's parameters and nothing else.

`A_reg` is laid out as ``[N, m]`` where ``N`` is the number of units in
the layer and ``m`` is the batch dimension (for conv layers, the spatial
positions are flattened into ``m`` as well).
"""

from typing import Tuple

import torch


_VALID_NORMALIZATION_MODES = (
    "no correction",
    "correct by input size",
    "naive mse sum correction",
)


def _canonical_normalization_mode(mode: str) -> str:
    """Normalize legacy aliases for the normalization-mode string."""
    if mode == "naive mse_sum correction":
        return "naive mse sum correction"
    return mode


def _validate_normalization_mode(mode: str) -> str:
    canonical = _canonical_normalization_mode(mode)
    if canonical not in _VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown normalization_mode: {mode!r}. "
            f"Expected one of {_VALID_NORMALIZATION_MODES}."
        )
    return canonical


def compute_aso_loss(
    A_reg: torch.Tensor,
    ages: torch.Tensor,
    maturity_threshold: int,
    age_decay_rate: float,
    normalization_mode: str,
) -> torch.Tensor:
    """Compute the Asymmetric Soft Orthogonality (ASO) loss.

    Args:
        A_reg: ``[N, m]`` preactivations recomputed via ``W @ stop_gradient(H)``.
        ages: ``[N]`` integer tensor of per-unit ages.
        maturity_threshold: Units with ``age >= maturity_threshold`` count as mature.
        age_decay_rate: ``gamma`` in ``lambda(a_i) = gamma ** a_i``.
        normalization_mode: One of "no correction", "correct by input size",
            "naive mse sum correction".

    Returns:
        Scalar tensor. Zero if there are no young or no mature units.
    """
    mode = _validate_normalization_mode(normalization_mode)

    if A_reg.dim() != 2:
        raise ValueError(f"A_reg must be 2D [N, m], got shape {tuple(A_reg.shape)}")
    if ages.dim() != 1 or ages.shape[0] != A_reg.shape[0]:
        raise ValueError(
            f"ages must be 1D with length N={A_reg.shape[0]}, "
            f"got shape {tuple(ages.shape)}"
        )

    N, m = A_reg.shape
    if N == 0 or m == 0:
        return A_reg.new_zeros(())

    mature_idx = torch.where(ages >= maturity_threshold)[0]
    young_idx = torch.where(ages < maturity_threshold)[0]
    if mature_idx.numel() == 0 or young_idx.numel() == 0:
        return A_reg.new_zeros(())

    # Spec: stop_gradient on mature rows so the optimizer cannot drag the
    # mature geometry toward the young units.
    A_mature = A_reg[mature_idx, :].detach()
    A_young = A_reg[young_idx, :]

    # cov[y, k] = <A_young[y], A_mature[k]>   shape [Y, K]
    cov = A_young @ A_mature.t()
    row_norms_sq = (cov * cov).sum(dim=1)  # [Y]

    weights = (age_decay_rate ** ages[young_idx].to(A_reg.dtype)) / (2.0 * m * m)
    loss = (weights * row_norms_sq).sum()

    Y = young_idx.numel()
    K = mature_idx.numel()
    if mode == "naive mse sum correction":
        loss = loss / (Y * K)
    elif mode == "correct by input size":
        loss = loss / Y
    # "no correction": leave as-is.

    return loss


def _faso_NxN_branch(
    A_reg: torch.Tensor,
    A_tilde: torch.Tensor,
    lambda_w: torch.Tensor,
) -> torch.Tensor:
    """Approach A from the spec: explicit ``N x N`` covariance.

    Returns ``sum_i lambda_w[i] * sum_{j != i} (A_reg[i] @ A_tilde[j])^2``.
    Caller is responsible for the trailing ``1 / (2 m^2)`` normalization.
    """
    C = A_reg @ A_tilde.t()          # [N, N]
    C_sq = C * C
    row_sum = C_sq.sum(dim=1)        # sum_j (a_i . a_j_tilde)^2
    diag = torch.diagonal(C_sq)      # (a_i . a_i_tilde)^2
    per_unit = row_sum - diag        # off-diagonal sum
    return (lambda_w * per_unit).sum()


def _faso_mxm_branch(
    A_reg: torch.Tensor,
    A_tilde: torch.Tensor,
    lambda_w: torch.Tensor,
) -> torch.Tensor:
    """Approach B from the spec: ``m x m`` batch-Gram trick.

    Uses ``a_i M_batch a_i^T = sum_j (a_i a_j_tilde^T)^2`` where
    ``M_batch = A_tilde^T A_tilde``; the self-term ``(a_i a_i_tilde^T)^2``
    is subtracted to yield the off-diagonal sum.

    Returns the same quantity as `_faso_NxN_branch`, modulo floating
    point.
    """
    M_batch = A_tilde.t() @ A_tilde            # [m, m]
    # (A_reg @ M_batch * A_reg).sum(dim=1)[i] = a_i M_batch a_i^T
    term1 = (A_reg @ M_batch * A_reg).sum(dim=1)
    term2 = ((A_reg * A_tilde).sum(dim=1)) ** 2
    per_unit = term1 - term2
    return (lambda_w * per_unit).sum()


def compute_faso_loss(
    A_reg: torch.Tensor,
    ages: torch.Tensor,
    age_decay_rate: float,
    normalization_mode: str,
) -> torch.Tensor:
    """Compute the Fully Age-weighted Soft Orthogonality (FASO) loss.

    Dynamic routing per the spec: ``N < m`` uses Approach A (explicit
    ``N x N`` covariance), else Approach B (``m x m`` batch Gram).

    Args:
        A_reg: ``[N, m]`` preactivations from ``W @ stop_gradient(H)``.
        ages: ``[N]`` integer tensor of per-unit ages.
        age_decay_rate: ``gamma`` in ``lambda(a_i) = gamma ** a_i``.
        normalization_mode: One of "no correction", "correct by input size",
            "naive mse sum correction".

    Returns:
        Scalar tensor; zero if ``N == 0`` or ``m == 0``.
    """
    mode = _validate_normalization_mode(normalization_mode)

    if A_reg.dim() != 2:
        raise ValueError(f"A_reg must be 2D [N, m], got shape {tuple(A_reg.shape)}")
    if ages.dim() != 1 or ages.shape[0] != A_reg.shape[0]:
        raise ValueError(
            f"ages must be 1D with length N={A_reg.shape[0]}, "
            f"got shape {tuple(ages.shape)}"
        )

    N, m = A_reg.shape
    if N == 0 or m == 0:
        return A_reg.new_zeros(())

    A_tilde = A_reg.detach()
    lambda_w = age_decay_rate ** ages.to(A_reg.dtype)

    if N < m:
        loss = _faso_NxN_branch(A_reg, A_tilde, lambda_w)
    else:
        loss = _faso_mxm_branch(A_reg, A_tilde, lambda_w)

    loss = loss / (2.0 * m * m)

    if mode == "naive mse sum correction":
        loss = loss / (N * N)
    elif mode == "correct by input size":
        loss = loss / N
    # "no correction": leave as-is.

    return loss


def faso_routing_branch(N: int, m: int) -> str:
    """Return the branch (`'NxN'` or `'mxm'`) that ``compute_faso_loss``
    would dispatch to for the given shape. Exposed for tests / logging.
    """
    return "NxN" if N < m else "mxm"


__all__: Tuple[str, ...] = (
    "compute_aso_loss",
    "compute_faso_loss",
    "faso_routing_branch",
    "_canonical_normalization_mode",
    "_faso_NxN_branch",
    "_faso_mxm_branch",
)
