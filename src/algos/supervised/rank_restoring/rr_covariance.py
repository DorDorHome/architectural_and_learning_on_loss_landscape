import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


def _resolve_dtype(dtype_name: Optional[str], reference: Tensor) -> torch.dtype:
    if dtype_name is None:
        return reference.dtype
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported covariance dtype '{dtype_name}'") from exc


@dataclass
class CovarianceState:
    """
    CovarianceState is a dataclass that stores the (snapshot of) Sigma, the the empirical feature covariance matrix.
    the 
    and the beta and ridge parameters for the exponential moving average covariance for the estimated covariance.
    
    """
    
    ema: Tensor
    beta: float
    ridge: float
    diag_only: bool

    def update(self, h: Tensor, dtype: Optional[str] = None) -> Tensor:
        with torch.no_grad():
            if h.dim() != 2:
                raise ValueError(f"Expected 2D tensor (features x batch), got {h.shape}")
            batch = h.shape[1]
            device = h.device
            target_dtype = _resolve_dtype(dtype, h)
            ema = self.ema.to(device=device, dtype=target_dtype)
            h = h.to(dtype=target_dtype)
            
            # compute the empirical feature covariance matrix
            if self.diag_only:
                cov = torch.mean(h * h, dim=1)
            else:
                cov = h @ h.t() / float(batch)
                
            # update the exponential moving average covariance

            ema.mul_(self.beta).add_(cov, alpha=1 - self.beta)
            self.ema.copy_(ema)
            if self.diag_only:
                result = ema + self.ridge
            else:
                eye = torch.eye(ema.size(0), device=device, dtype=ema.dtype)
                result = ema + self.ridge * eye
            return result.detach().clone()


def initialize_covariance(d_dim: int, device: torch.device, beta: float, ridge: float, diag_only: bool,
                           dtype: torch.dtype) -> CovarianceState:
    """
    Create a `CovarianceState` for tracking the EMA feature covariance Σ.

    This allocates the internal EMA buffer (`state.ema`) on the requested device/dtype:
    - If `diag_only=True`, `ema` has shape (d_dim,) and tracks only diag(Σ).
    - If `diag_only=False`, `ema` has shape (d_dim, d_dim) and tracks the full Σ.

    The returned state can be updated each step via `state.update(H_prev)` to produce the
    current regularized covariance estimate:
        Σ_t = EMA[(1/m) H H^T] + ridge * I    (full)   or   + ridge               (diag-only)

    Args:
        d_dim: Input/feature dimension d (number of rows in H_prev).
        device: Target device for the EMA buffer (e.g., CUDA device of the layer weights).
        beta: EMA decay factor in [0, 1). Larger means slower updates.
        ridge: Nonnegative ridge added for numerical stability.
        diag_only: If True, track only the diagonal of Σ to save memory/compute.
        dtype: Data type for the EMA buffer (typically matches layer weight dtype).

    Returns:
        A `CovarianceState` instance initialized with a zero EMA buffer.

    Raises:
        RuntimeError: If allocation fails; CUDA-related failures are routed through
            `raise_gpu_corruption_error` for clearer diagnostics.
    """
    
    try:
        if diag_only:
            ema = torch.zeros(d_dim, device=device, dtype=dtype)
        else:
            ema = torch.zeros(d_dim, d_dim, device=device, dtype=dtype)
        return CovarianceState(ema=ema, beta=beta, ridge=ridge, diag_only=diag_only)
    except RuntimeError as e:
        if 'CUDA' in str(e) or 'cuda' in str(e):
            from src.utils.gpu_health_check import raise_gpu_corruption_error
            raise_gpu_corruption_error(device, e)
        else:
            raise
