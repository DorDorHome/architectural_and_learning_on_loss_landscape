import torch
from torch import Tensor
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SigmaOrthonormalBasis:
    basis: Tensor
    sigma: Tensor
    diag_only: bool

    @classmethod
    def from_columns(cls, V: Tensor, sigma: Tensor, diag_only: bool, eps: float) -> "SigmaOrthonormalBasis":
        if V.numel() == 0:
            return cls(basis=V.new_zeros(V.shape[0], 0), sigma=sigma, diag_only=diag_only)
        cols: List[Tensor] = []
        for j in range(V.shape[1]):
            v = V[:, j]
            for q in cols:
                proj = sigma_inner(q, v, sigma, diag_only)
                v = v - proj * q
            norm = sigma_norm(v, sigma, diag_only)
            if norm < eps:
                continue
            cols.append(v / norm)
        if len(cols) == 0:
            return cls(basis=V.new_zeros(V.shape[0], 0), sigma=sigma, diag_only=diag_only)
        return cls(basis=torch.stack(cols, dim=1), sigma=sigma, diag_only=diag_only)

    def project(self, x: Tensor) -> Tensor:
        if self.basis.numel() == 0:
            return x
        coeffs = torch.stack(
            [sigma_inner(self.basis[:, j], x, self.sigma, self.diag_only) for j in range(self.basis.shape[1])],
            dim=0,
        )
        return self.basis @ coeffs


def sigma_inner(u: Tensor, v: Tensor, sigma: Tensor, diag_only: bool) -> Tensor:
    if diag_only:
        return torch.sum(u * sigma * v)
    else:
        return torch.dot(u, sigma @ v)


def sigma_norm(v: Tensor, sigma: Tensor, diag_only: bool) -> Tensor:
    inner = sigma_inner(v, v, sigma, diag_only)
    return torch.sqrt(torch.clamp(inner, min=0.0))


def project_complement(u: Tensor, basis: SigmaOrthonormalBasis) -> Tuple[Tensor, Tensor]:
    if basis.basis.numel() == 0:
        return u, sigma_norm(u, basis.sigma, basis.diag_only)
    proj = basis.project(u)
    residual = u - proj
    return residual, sigma_norm(residual, basis.sigma, basis.diag_only)


def draw_sigma_unit(
    basis: SigmaOrthonormalBasis,
    trials: int,
    eps: float,
    dtype: torch.dtype,
    nullspace_epsilon: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, bool]:
    best_residual: Optional[Tensor] = None
    best_norm = torch.tensor(-1.0, device=basis.sigma.device, dtype=basis.sigma.dtype)
    dim = basis.sigma.shape[0]
    for _ in range(trials):
        seed = torch.randn(dim, device=basis.sigma.device, dtype=dtype, generator=generator)
        residual, norm = project_complement(seed, basis)
        if norm > best_norm:
            best_norm = norm
            best_residual = residual
    if best_residual is None or best_norm < eps:
        return compute_fallback_direction(basis, dtype), True
    w = best_residual / torch.clamp(best_norm, min=eps)
    if nullspace_epsilon > 0.0 and basis.basis.numel() > 0:
        w = w + nullspace_epsilon * torch.randn_like(w)
        residual, norm = project_complement(w, basis)
        w = residual / torch.clamp(norm, min=eps)
    return w, False


def compute_fallback_direction(basis: SigmaOrthonormalBasis, dtype: torch.dtype) -> Tensor:
    dim = basis.sigma.shape[0]
    if basis.diag_only:
        inv_sigma = torch.reciprocal(torch.clamp(basis.sigma, min=1e-9))
        idx = torch.argmin(inv_sigma)
        direction = torch.zeros(dim, device=basis.sigma.device, dtype=dtype)
        direction[idx] = 1.0
        return direction / torch.sqrt(torch.clamp(basis.sigma[idx], min=1e-9))
    else:
        q = torch.randn(dim, device=basis.sigma.device, dtype=dtype)
        for _ in range(32):
            q = q - basis.project(q)
            norm = torch.norm(q)
            if norm < 1e-9:
                q = torch.randn(dim, device=basis.sigma.device, dtype=dtype)
                continue
            q = q / norm
        return q
