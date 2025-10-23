from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SigmaGeometry:
    """Utility class for operations in the covariance induced geometry."""

    sigma: Tensor
    diag_only: bool
    eps: float

    def __post_init__(self) -> None:
        if self.diag_only:
            if self.sigma.dim() != 1:
                raise ValueError("Diagonal sigma expected to be a 1-D tensor")
            self._sigma_clamped = torch.clamp(self.sigma, min=self.eps)
            self._trace = float(self._sigma_clamped.sum().item())
            self._lambda_min = float(self._sigma_clamped.min().item())
            self._sqrt = torch.sqrt(self._sigma_clamped)
            self._inv_sqrt = torch.reciprocal(self._sqrt)
            self.dim = int(self.sigma.numel())
        else:
            if self.sigma.dim() != 2:
                raise ValueError("Full sigma expected to be a 2-D tensor")
            try:
                eigvals, eigvecs = torch.linalg.eigh(self.sigma)
            except RuntimeError:
                # Fallback to CPU computation for numerical stability
                sigma_cpu = self.sigma.detach().cpu().double()
                eigvals, eigvecs = torch.linalg.eigh(sigma_cpu)
                eigvals = eigvals.to(self.sigma.device, dtype=self.sigma.dtype)
                eigvecs = eigvecs.to(self.sigma.device, dtype=self.sigma.dtype)
            eigvals = torch.clamp(eigvals, min=self.eps)
            self._eigvals = eigvals
            self._eigvecs = eigvecs
            diag_sqrt = torch.diag(torch.sqrt(eigvals))
            diag_inv_sqrt = torch.diag(torch.reciprocal(torch.sqrt(eigvals)))
            self._sqrt = eigvecs @ diag_sqrt @ eigvecs.t()
            self._inv_sqrt = eigvecs @ diag_inv_sqrt @ eigvecs.t()
            self._trace = float(eigvals.sum().item())
            self._lambda_min = float(eigvals.min().item())
            self.dim = int(self.sigma.size(0))

    @property
    def trace(self) -> float:
        return self._trace

    @property
    def lambda_min(self) -> float:
        return self._lambda_min

    def apply(self, vec: Tensor) -> Tensor:
        if self.diag_only:
            return self.sigma * vec
        return self.sigma @ vec

    def apply_matrix(self, mat: Tensor) -> Tensor:
        if self.diag_only:
            return self.sigma.unsqueeze(0) * mat
        return self.sigma @ mat

    def inner(self, u: Tensor, v: Tensor) -> Tensor:
        if self.diag_only:
            return torch.dot(u, self.sigma * v)
        return torch.dot(u, self.sigma @ v)

    def norm(self, vec: Tensor) -> Tensor:
        value = self.inner(vec, vec)
        return torch.sqrt(torch.clamp(value, min=0.0))

    def vector_energy(self, vec: Tensor) -> float:
        return float(self.inner(vec, vec).item())

    def matrix_energy(self, weight: Tensor) -> float:
        if self.diag_only:
            sigma = self.sigma.view(1, -1)
            energy = torch.sum(weight * sigma * weight)
        else:
            energy = torch.einsum('ij,jk,ik->', weight, self.sigma, weight)
        return float(energy.item())

    def whiten_columns(self, cols: Tensor) -> Tensor:
        if cols.numel() == 0:
            return cols
        if self.diag_only:
            return self._sqrt.unsqueeze(1) * cols
        return self._sqrt @ cols

    def unwhiten_vector(self, vec: Tensor) -> Tensor:
        if self.diag_only:
            return self._inv_sqrt * vec
        return self._inv_sqrt @ vec

    def random_unit(self, dtype: torch.dtype) -> Tensor:
        vec = torch.randn(self.dim, device=self.sigma.device, dtype=dtype)
        norm = torch.clamp(self.norm(vec), min=self.eps)
        return vec / norm

    def lambda_min_whitened(self, weight: Tensor) -> float:
        if weight.numel() == 0:
            return 0.0
        if self.diag_only:
            whitened = self._sqrt.unsqueeze(0) * weight
        else:
            whitened = weight @ self._sqrt.t()
        gram = whitened @ whitened.t()
        try:
            eigvals = torch.linalg.eigvalsh(gram)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(gram.double()).to(gram.dtype)
        return float(torch.clamp_min(eigvals.min(), 0.0).item())


class SigmaProjector:
    """Σ-orthogonal projector for kept directions."""

    def __init__(self, geometry: SigmaGeometry, basis: Optional[Tensor], reg_epsilon: float) -> None:
        self.geometry = geometry
        self.reg_epsilon = reg_epsilon
        if basis is None or basis.numel() == 0:
            self.basis = geometry.sigma.new_zeros((geometry.dim, 0))
        else:
            if basis.dim() != 2 or basis.size(0) != geometry.dim:
                raise ValueError("Basis columns must match sigma dimensionality")
            self.basis = basis.clone()
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        if self.basis.numel() == 0:
            self._G_inv = None
            return
        if self.geometry.diag_only:
            sigma_basis = self.geometry.sigma.unsqueeze(1) * self.basis
        else:
            sigma_basis = self.geometry.sigma @ self.basis
        gram = self.basis.t() @ sigma_basis
        dim = gram.size(0)
        trace = torch.trace(gram)
        reg = self.reg_epsilon * trace / max(dim, 1)
        if reg <= 0:
            reg = self.reg_epsilon
        gram = gram + reg * torch.eye(dim, device=gram.device, dtype=gram.dtype)
        self._G_inv = torch.linalg.solve(gram, torch.eye(dim, device=gram.device, dtype=gram.dtype))

    def apply(self, vec: Tensor) -> Tensor:
        if self.basis.numel() == 0:
            return torch.zeros_like(vec)
        if self.geometry.diag_only:
            sigma_vec = self.geometry.sigma * vec
        else:
            sigma_vec = self.geometry.sigma @ vec
        coeff = self.basis.t() @ sigma_vec
        proj_coeff = self._G_inv @ coeff
        return self.basis @ proj_coeff

    def project_complement(self, vec: Tensor) -> Tensor:
        return vec - self.apply(vec)

    def add_vector(self, vec: Tensor) -> None:
        if self.basis.numel() == 0:
            self.basis = vec.unsqueeze(1)
        else:
            self.basis = torch.cat([self.basis, vec.unsqueeze(1)], dim=1)
        self._refresh_cache()

    @property
    def num_vectors(self) -> int:
        return 0 if self.basis.numel() == 0 else int(self.basis.size(1))

    def least_covered_direction(self, dtype: torch.dtype) -> Tensor:
        if self.num_vectors == 0:
            return self.geometry.random_unit(dtype=dtype)
        whitened = self.geometry.whiten_columns(self.basis)
        gram = whitened @ whitened.t()
        try:
            eigvals, eigvecs = torch.linalg.eigh(gram)
        except RuntimeError:
            gram_cpu = gram.detach().cpu().double()
            eigvals, eigvecs = torch.linalg.eigh(gram_cpu)
            eigvals = eigvals.to(gram.device, dtype=gram.dtype)
            eigvecs = eigvecs.to(gram.device, dtype=gram.dtype)
        idx = torch.argmin(eigvals)
        eigvec = eigvecs[:, idx]
        candidate = self.geometry.unwhiten_vector(eigvec)
        residual = self.project_complement(candidate)
        norm = self.geometry.norm(residual)
        if norm < self.geometry.eps:
            return self.geometry.random_unit(dtype=dtype)
        return residual / norm


class EnergyAllocator:
    """Allocates Σ-energy for replacement units under a layer budget."""

    def __init__(
        self,
        q_target: float,
        layer_size: int,
        used_energy: float,
        tau: float,
        lambda_min_sigma: float,
        lambda_star: Optional[float],
        replacements: int,
    ) -> None:
        self.q_target = q_target
        self.layer_size = layer_size
        self.total_target = q_target * layer_size
        self.used_energy = used_energy
        self.remaining = replacements
        self.lambda_min_sigma = lambda_min_sigma
        self.lambda_star = lambda_star
        self.q_min = tau * lambda_min_sigma
        self.residual = max(0.0, self.total_target - self.used_energy)

    def allocate(self) -> Tuple[float, bool]:
        saturated = self.residual <= 0.0
        if self.remaining <= 0:
            q_alloc = max(self.q_min, self.lambda_star or 0.0)
            return q_alloc, True
        if not saturated:
            q_alloc = min(self.q_target, self.residual / self.remaining)
            self.residual = max(0.0, self.residual - q_alloc)
        else:
            floor = self.q_min
            if self.lambda_star is not None:
                floor = max(floor, self.lambda_star)
            q_alloc = min(self.q_target, floor)
        self.remaining -= 1
        self.used_energy += q_alloc
        return q_alloc, saturated


def chi0_for_activation(activation: Optional[str], leaky_negative_slope: float = 0.01) -> float:
    if activation is None:
        return 1.0
    key = activation.lower()
    if key == 'relu':
        return 0.5
    if key == 'leaky_relu':
        alpha = leaky_negative_slope
        return (1.0 + alpha * alpha) / 2.0
    if key in {'gelu', 'selu', 'swish'}:
        return 0.5
    if key in {'sigmoid', 'tanh'}:
        return 1.0
    return 1.0
