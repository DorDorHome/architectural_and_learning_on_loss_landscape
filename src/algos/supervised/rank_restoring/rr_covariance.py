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
            if self.diag_only:
                cov = torch.mean(h * h, dim=1)
            else:
                cov = h @ h.t() / float(batch)
            ema.mul_(self.beta).add_(cov, alpha=1 - self.beta)
            if self.diag_only:
                return ema + self.ridge
            else:
                eye = torch.eye(ema.size(0), device=device, dtype=ema.dtype)
                return ema + self.ridge * eye


def initialize_covariance(d_dim: int, device: torch.device, beta: float, ridge: float, diag_only: bool,
                           dtype: torch.dtype) -> CovarianceState:
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
