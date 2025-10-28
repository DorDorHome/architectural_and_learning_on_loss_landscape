from __future__ import annotations

import logging
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Module

from configs.configurations import RRContinuousBackpropConfig
from src.algos.AdamGnT import AdamGnT
from src.algos.gnt import GnT_for_FC
from src.algos.supervised.rank_restoring.rr_covariance import CovarianceState, initialize_covariance
from src.algos.supervised.rank_restoring.sigma_geometry import (
    EnergyAllocator,
    SigmaGeometry,
    SigmaProjector,
    chi0_for_activation,
)


@dataclass
class LayerReplacementStats:
    successful: int = 0
    fallbacks: int = 0
    saturated: int = 0
    allocated_q: List[float] = field(default_factory=list)
    last_allocations: List[float] = field(default_factory=list)
    last_saturated: int = 0
    last_Q_target: float = float("nan")
    last_Q_used: float = float("nan")
    last_lambda_min_white: float = float("nan")
    last_metrics: Optional[Dict[str, float]] = None


class RR_GnT_for_FC(GnT_for_FC):
    """Rank-restoring Generate-and-Test for fully-connected networks."""

    def __init__(
        self,
        net: Sequence[Module],
        hidden_activation: str,
        opt: AdamGnT,
        config: RRContinuousBackpropConfig,
        loss_func,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            net=net,
            hidden_activation=hidden_activation,
            opt=opt,
            decay_rate=config.decay_rate_utility_track,
            replacement_rate=config.neurons_replacement_rate,
            init=config.init,
            device=device,
            maturity_threshold=config.maturity_threshold,
            util_type=config.util_type,
            loss_func=loss_func,
            accumulate=config.accumulate,
        )

        self.rr_config = config
        self.layer_covariances: List[Optional[CovarianceState]] = [None for _ in range(self.num_hidden_layers)]
        self.layer_stats: Dict[int, LayerReplacementStats] = {}
        self._layer_log_counters: List[int] = [0 for _ in range(self.num_hidden_layers)]

        self.hidden_activation = hidden_activation or "linear"
        self._leaky_slope = 0.01
        for module in net:
            if isinstance(module, nn.LeakyReLU):
                self._leaky_slope = float(module.negative_slope)
                break

        self._chi0_constant = chi0_for_activation(self.hidden_activation, self._leaky_slope)
        self._logger = logging.getLogger(__name__)

    def gen_and_test(self, features: List[Tensor], batch_input: Optional[Tensor] = None) -> None:
        if not isinstance(features, list):
            raise TypeError("features passed to generate-and-test should be a list")
        if batch_input is None:
            raise ValueError("batch_input must be provided for FC RR-CBP")
        if not self.rr_config.rrcbp_enabled:
            return

        features_to_replace, num_features_to_replace = self.test_features(features=features)
        if all(count == 0 for count in num_features_to_replace):
            return

        with torch.no_grad():
            for layer_idx, count in enumerate(num_features_to_replace):
                if count == 0:
                    continue

                layer = self.net[layer_idx * 2]
                next_layer = self.net[layer_idx * 2 + 2]
                self._ensure_covariance(layer_idx, layer)
                sigma_state = self.layer_covariances[layer_idx]
                assert sigma_state is not None

                H_prev = self._compute_layer_inputs(layer_idx, features, batch_input, layer)
                sigma = sigma_state.update(H_prev, dtype=self.rr_config.covariance_dtype)
                keep_idx = self._kept_indices(layer_idx, features_to_replace[layer_idx])

                stats = self._replace_units(
                    layer_idx=layer_idx,
                    keep_idx=keep_idx,
                    replace_idx=features_to_replace[layer_idx],
                    sigma=sigma,
                    H_prev=H_prev,
                    activations=features[layer_idx],
                )

                self._post_replacement_housekeeping(layer_idx, features_to_replace[layer_idx])

                if self.rr_config.log_rank_metrics_every > 0 and count > 0:
                    self._layer_log_counters[layer_idx] += 1
                    if self._layer_log_counters[layer_idx] >= self.rr_config.log_rank_metrics_every:
                        metrics = self._compute_rank_metrics(
                            layer_idx=layer_idx,
                            sigma=sigma,
                            activations=features[layer_idx],
                            stats=stats,
                        )
                        stats.last_metrics = metrics
                        self._emit_rank_metrics(layer_idx, metrics)
                        self._layer_log_counters[layer_idx] = 0

    def _ensure_covariance(self, layer_idx: int, layer: Module) -> None:
        if self.layer_covariances[layer_idx] is not None:
            return
        if not isinstance(layer, Linear):
            raise TypeError("RR-CBP for FC networks expects Linear layers")
        device = layer.weight.device
        dtype = layer.weight.dtype
        self.layer_covariances[layer_idx] = initialize_covariance(
            d_dim=layer.in_features,
            device=device,
            beta=self.rr_config.sigma_ema_beta,
            ridge=self.rr_config.sigma_ridge,
            diag_only=self.rr_config.diag_sigma_only,
            dtype=dtype,
        )

    def _compute_layer_inputs(
        self,
        layer_idx: int,
        features: List[Tensor],
        batch_input: Tensor,
        layer: Module,
    ) -> Tensor:
        if layer_idx == 0:
            inputs = batch_input
        else:
            inputs = features[layer_idx - 1]
        if inputs.dim() != 2:
            raise ValueError("Expected 2-D inputs for FC RR-CBP")
        return inputs.t().to(layer.weight.dtype).contiguous()

    def _kept_indices(self, layer_idx: int, replace_idx: Tensor) -> Tensor:
        layer = self.net[layer_idx * 2]
        if not isinstance(layer, Linear):
            raise TypeError("Unsupported layer type for keep index computation")
        total = layer.out_features
        mask = torch.ones(total, dtype=torch.bool, device=replace_idx.device)
        mask[replace_idx] = False
        return torch.arange(total, device=replace_idx.device)[mask]

    def _replace_units(
        self,
        layer_idx: int,
        keep_idx: Tensor,
        replace_idx: Tensor,
        sigma: Tensor,
        H_prev: Tensor,
        activations: Tensor,
    ) -> LayerReplacementStats:
        layer = self.net[layer_idx * 2]
        next_layer = self.net[layer_idx * 2 + 2]
        config = self.rr_config

        weight_matrix = layer.weight.data
        geometry = SigmaGeometry(
            sigma=sigma,
            diag_only=config.diag_sigma_only,
            eps=config.sigma_eig_floor,
        )
        kept_vectors = weight_matrix[keep_idx, :].t()
        projector = SigmaProjector(
            geometry=geometry,
            basis=kept_vectors,
            reg_epsilon=config.projector_reg_epsilon,
        )
        stats = self.layer_stats.setdefault(layer_idx, LayerReplacementStats())

        chi0 = self._resolve_chi0(activations)
        v_target = geometry.trace / max(geometry.dim, 1)
        q_target = v_target / max(chi0, config.proj_eps)

        dtype = weight_matrix.dtype

        used_energy = 0.0
        if kept_vectors.numel() > 0:
            for col in range(kept_vectors.size(1)):
                used_energy += geometry.vector_energy(kept_vectors[:, col])

        lambda_star = None
        if config.use_lambda_star:
            if config.lambda_star is not None:
                lambda_star = float(config.lambda_star)
            elif kept_vectors.numel() > 0:
                whitened = geometry.whiten_columns(kept_vectors)
                gram_white = whitened.t() @ whitened
                force_cpu_eigh = os.environ.get('SIGMA_FORCE_CPU_EIGH', '0') == '1'
                if force_cpu_eigh and gram_white.device.type == 'cuda':
                    eigvals = torch.linalg.eigvalsh(gram_white.detach().cpu())
                    eigvals = eigvals.to(gram_white.device, dtype=gram_white.dtype)
                else:
                    try:
                        eigvals = torch.linalg.eigvalsh(gram_white)
                    except RuntimeError:
                        eigvals = torch.linalg.eigvalsh(gram_white.cpu().double()).to(gram_white.dtype)
                if eigvals.numel() > 0:
                    lambda_star = float(min(1.0, 2.0 * torch.clamp_min(eigvals.min(), 0.0).item()))

        allocator = EnergyAllocator(
            q_target=q_target,
            layer_size=weight_matrix.size(0),
            used_energy=used_energy,
            tau=config.tau,
            lambda_min_sigma=geometry.lambda_min,
            lambda_star=lambda_star,
            replacements=int(replace_idx.numel()),
        )

        recent_allocations: List[float] = []
        recent_saturated = 0
        new_vectors: List[Tensor] = []

        for unit_idx in replace_idx.tolist():
            if self.ages[layer_idx][unit_idx] > 0:
                bias_corrected_act = self.mean_feature_act[layer_idx][unit_idx] / (
                    1 - self.decay_rate ** self.ages[layer_idx][unit_idx]
                )
                self._transfer_bias(
                    next_layer=next_layer,
                    unit_idx=unit_idx,
                    bias_corrected_act=bias_corrected_act,
                )

            direction, used_fallback = self._sample_sigma_direction(projector, geometry, dtype)
            if config.orthonormalize_batch and new_vectors:
                direction = self._orthonormalize_direction(direction, new_vectors, geometry)

            q_alloc, saturated = allocator.allocate()
            if saturated:
                stats.saturated += 1
                recent_saturated += 1
                if config.improve_conditioning_if_saturated:
                    direction = self._maybe_nullspace_seed(direction, geometry, dtype)

            norm_dir = torch.clamp(geometry.norm(direction), min=config.proj_eps)
            scale = math.sqrt(max(q_alloc, config.proj_eps)) / norm_dir
            w_in = direction * scale

            projector.add_vector(w_in)
            new_vectors.append(w_in / torch.clamp(geometry.norm(w_in), min=config.proj_eps))
            self._assign_weight(layer, unit_idx, w_in)

            acts = (w_in.unsqueeze(0) @ H_prev).squeeze(0)
            bias_value = self._center_bias(acts)
            layer.bias.data[unit_idx] = bias_value.to(device=layer.bias.device, dtype=layer.bias.dtype)

            self._zero_and_seed_outgoing(next_layer, unit_idx)

            if used_fallback:
                stats.fallbacks += 1
            else:
                stats.successful += 1

            stats.allocated_q.append(float(q_alloc))
            recent_allocations.append(float(q_alloc))

        if replace_idx.numel() > 0:
            self.mean_feature_act[layer_idx][replace_idx] = 0.0
            self.util[layer_idx][replace_idx] = 0.0
            self.ages[layer_idx][replace_idx] = 0.0

        stats.last_allocations = recent_allocations
        stats.last_saturated = recent_saturated
        stats.last_Q_target = q_target * weight_matrix.size(0)
        stats.last_Q_used = geometry.matrix_energy(weight_matrix)
        stats.last_lambda_min_white = geometry.lambda_min_whitened(weight_matrix)
        return stats

    def _sample_sigma_direction(
        self,
        projector: SigmaProjector,
        geometry: SigmaGeometry,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, bool]:
        attempts = max(1, int(self.rr_config.max_proj_trials))
        last: Optional[Tensor] = None
        for _ in range(attempts):
            candidate = geometry.random_unit(dtype=dtype)
            residual = projector.project_complement(candidate)
            norm = geometry.norm(residual)
            if norm > self.rr_config.proj_eps:
                return residual / norm, False
            last = residual
        fallback = projector.least_covered_direction(dtype=dtype)
        norm = geometry.norm(fallback)
        if norm <= self.rr_config.proj_eps and last is not None:
            fallback = last
            norm = geometry.norm(fallback)
        norm = torch.clamp(norm, min=self.rr_config.proj_eps)
        return fallback / norm, True

    def _orthonormalize_direction(
        self,
        direction: Tensor,
        existing: List[Tensor],
        geometry: SigmaGeometry,
    ) -> Tensor:
        adjusted = direction.clone()
        for vec in existing:
            coeff = geometry.inner(vec, adjusted)
            adjusted = adjusted - coeff * vec
        norm = geometry.norm(adjusted)
        if norm <= self.rr_config.proj_eps:
            return direction
        return adjusted / norm

    def _maybe_nullspace_seed(
        self,
        direction: Tensor,
        geometry: SigmaGeometry,
        dtype: torch.dtype,
    ) -> Tensor:
        epsilon = getattr(self.rr_config, "nullspace_seed_epsilon", 0.0)
        if epsilon <= 0.0:
            return direction
        noise = geometry.random_unit(dtype=dtype)
        candidate = direction + epsilon * noise
        norm = geometry.norm(candidate)
        if norm <= self.rr_config.proj_eps:
            return direction
        return candidate / norm

    def _assign_weight(self, layer: Linear, unit_idx: int, vector: Tensor) -> None:
        layer.weight.data[unit_idx, :] = vector

    def _transfer_bias(self, next_layer: Module, unit_idx: int, bias_corrected_act: Tensor) -> None:
        contribution = next_layer.weight.data[:, unit_idx] * bias_corrected_act
        next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

    def _zero_and_seed_outgoing(self, next_layer: Module, unit_idx: int) -> None:
        epsilon = self.rr_config.epsilon_micro_seed
        use_seed = self.rr_config.use_micro_seed and epsilon > 0.0
        next_layer.weight.data[:, unit_idx] = 0.0
        if use_seed:
            noise = torch.randn_like(next_layer.weight.data[:, unit_idx])
            noise = noise - noise.mean()
            norm = torch.clamp(noise.norm(), min=self.rr_config.proj_eps)
            next_layer.weight.data[:, unit_idx] = epsilon * (noise / norm)

    def _resolve_chi0(self, activations: Tensor) -> float:
        if self.rr_config.chi0_override is not None:
            return float(self.rr_config.chi0_override)
        if (
            self.rr_config.estimate_chi0_from_batch
            and activations is not None
            and activations.numel() > 0
        ):
            value = torch.clamp(activations.reshape(-1).pow(2).mean(), min=self.rr_config.proj_eps)
            return float(value.item())
        return float(self._chi0_constant)

    def _center_bias(self, activations: Tensor) -> Tensor:
        if activations.numel() == 0:
            return torch.zeros(1, device=activations.device, dtype=activations.dtype).squeeze(0)
        if self.rr_config.center_bias == "median":
            return -torch.median(activations)
        return -activations.mean()

    def _compute_rank_metrics(
        self,
        layer_idx: int,
        sigma: Tensor,
        activations: Tensor,
        stats: LayerReplacementStats,
    ) -> Dict[str, float]:
        layer = self.net[layer_idx * 2]
        weight = layer.weight.data
        if self.rr_config.diag_sigma_only:
            gram = (weight * sigma.unsqueeze(0)) @ weight.t()
        else:
            gram = weight @ sigma @ weight.t()

        try:
            rank_val = float(torch.linalg.matrix_rank(gram, tol=self.rr_config.proj_eps).item())
        except RuntimeError:
            rank_val = float(torch.linalg.matrix_rank(gram.cpu().double(), tol=self.rr_config.proj_eps).item())

        lambda_min = float("nan")
        force_cpu_eigh = os.environ.get('SIGMA_FORCE_CPU_EIGH', '0') == '1'
        if force_cpu_eigh and gram.device.type == 'cuda':
            eigvals = torch.linalg.eigvalsh(gram.detach().cpu())
            eigvals = eigvals.to(gram.device, dtype=gram.dtype)
        else:
            try:
                eigvals = torch.linalg.eigvalsh(gram)
            except RuntimeError:
                eigvals = torch.linalg.eigvalsh(gram.cpu().double()).to(gram.dtype)
        if eigvals.numel() > 0:
            lambda_min = float(torch.clamp_min(eigvals.min(), 0.0).item())

        if activations.numel() > 0:
            active_fraction = float((activations > 0).float().mean().item())
        else:
            active_fraction = 0.0

        total_attempts = max(stats.successful + stats.fallbacks, 1)
        success_ratio = float(stats.successful / total_attempts)
        fallback_ratio = float(stats.fallbacks / total_attempts)

        allocations = stats.last_allocations
        q_alloc_mean = float(sum(allocations) / len(allocations)) if allocations else 0.0
        saturated_fraction = (
            float(stats.last_saturated / max(len(allocations), 1)) if allocations else 0.0
        )

        energy_ratio = float("nan")
        if math.isfinite(stats.last_Q_target) and stats.last_Q_target != 0.0:
            energy_ratio = float(stats.last_Q_used / stats.last_Q_target)

        return {
            "rank_WSigmaWT": rank_val,
            "lambda_min": lambda_min,
            "lambda_min_whitened": stats.last_lambda_min_white,
            "active_fraction": active_fraction,
            "successful_replacements": float(stats.successful),
            "fallback_replacements": float(stats.fallbacks),
            "success_ratio": success_ratio,
            "fallback_ratio": fallback_ratio,
            "q_alloc_mean": q_alloc_mean,
            "energy_ratio": energy_ratio,
            "saturated_fraction": saturated_fraction,
        }

    def _emit_rank_metrics(self, layer_idx: int, metrics: Dict[str, float]) -> None:
        if self._logger is None:
            return
        self._logger.info(
            (
                "RR-CBP FC layer %d | rank %.2f | lambda_min %.3e | lambda_min_white %.3e "
                "| energy %.3f | success %.2f | fallback %.2f | saturated %.2f"
            ),
            layer_idx,
            metrics.get("rank_WSigmaWT", float("nan")),
            metrics.get("lambda_min", float("nan")),
            metrics.get("lambda_min_whitened", float("nan")),
            metrics.get("energy_ratio", float("nan")),
            metrics.get("success_ratio", float("nan")),
            metrics.get("fallback_ratio", float("nan")),
            metrics.get("saturated_fraction", float("nan")),
        )

    def _post_replacement_housekeeping(self, layer_idx: int, replace_idx: Tensor) -> None:
        if replace_idx.numel() == 0:
            return

        layer = self.net[layer_idx * 2]
        next_layer = self.net[layer_idx * 2 + 2]

        if isinstance(self.opt, AdamGnT):
            self.opt.state[layer.weight]["exp_avg"][replace_idx, :] = 0.0
            self.opt.state[layer.weight]["exp_avg_sq"][replace_idx, :] = 0.0
            self.opt.state[layer.weight]["step"][replace_idx, :] = 0
            self.opt.state[layer.bias]["exp_avg"][replace_idx] = 0.0
            self.opt.state[layer.bias]["exp_avg_sq"][replace_idx] = 0.0
            self.opt.state[layer.bias]["step"][replace_idx] = 0

            self.opt.state[next_layer.weight]["exp_avg"][:, replace_idx] = 0.0
            self.opt.state[next_layer.weight]["exp_avg_sq"][:, replace_idx] = 0.0
            self.opt.state[next_layer.weight]["step"][:, replace_idx] = 0

        self.accumulated_num_features_to_replace[layer_idx] = 0.0

    def get_layer_stats(self) -> Dict[int, LayerReplacementStats]:
        return self.layer_stats
