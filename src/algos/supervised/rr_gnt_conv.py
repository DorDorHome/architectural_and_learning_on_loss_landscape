from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module

from configs.configurations import RRContinuousBackpropConfig
from src.algos.AdamGnT import AdamGnT
from src.algos.gnt import ConvGnT_for_ConvNet
from src.algos.supervised.rank_restoring.rr_covariance import CovarianceState, initialize_covariance
from src.algos.supervised.rank_restoring.sigma_geometry import (
    EnergyAllocator,
    SigmaGeometry,
    SigmaProjector,
    chi0_for_activation,
)
from src.algos.supervised.rr_gnt_fc import LayerReplacementStats


class RR_GnT_for_ConvNet(ConvGnT_for_ConvNet):
    """Rank-restoring Generate-and-Test for convolutional networks."""

    def __init__(
        self,
        net: Sequence[Module],
        hidden_activation: str,
        opt: AdamGnT,
        config: RRContinuousBackpropConfig,
        loss_func,
        device: str = "cpu",
        num_last_filter_outputs: int = 1,
    ) -> None:
        super().__init__(
            net=net,
            hidden_activation=hidden_activation,
            opt=opt,
            replacement_rate=config.neurons_replacement_rate,
            decay_rate=config.decay_rate_utility_track,
            init=config.init,
            num_last_filter_outputs=num_last_filter_outputs,
            util_type=config.util_type,
            maturity_threshold=config.maturity_threshold,
            device=device,
        )
        self.rr_config = config
        self.layer_covariances: List[Optional[CovarianceState]] = [None for _ in range(self.num_hidden_layers)]
        self.layer_stats: Dict[int, LayerReplacementStats] = {}
        self._layer_log_counters: List[int] = [0 for _ in range(self.num_hidden_layers)]
        self._logger = logging.getLogger(__name__)
        self._conv_output_multipliers: List[int] = []
        for layer_idx in range(self.num_hidden_layers):
            if self.use_map:
                map_item = self.plasticity_map[layer_idx]
                current_layer = map_item['weight_module']
                next_layer = map_item['outgoing_module']
            else:
                current_layer = self.net[layer_idx * 2]
                next_layer = self.net[layer_idx * 2 + 2]
                
            if isinstance(current_layer, Conv2d) and isinstance(next_layer, Linear):
                self._conv_output_multipliers.append(num_last_filter_outputs)
            else:
                self._conv_output_multipliers.append(1)
        
                
        self.hidden_activation = hidden_activation or "linear"
        self._leaky_slope = 0.01
        
        
        # FIX: Safer iteration for LeakyReLU parameter detection
        network_modules = net if isinstance(net, list) else net.modules()
        for module in network_modules:
            if isinstance(module, torch.nn.LeakyReLU):
                self._leaky_slope = float(module.negative_slope)
                break
        self._chi0_constant = chi0_for_activation(self.hidden_activation, self._leaky_slope)

    def gen_and_test(self, features: List[Tensor], batch_input: Optional[Tensor] = None) -> None:
        if not isinstance(features, list):
            raise TypeError("features passed to generate-and-test should be a list")
        if batch_input is None:
            raise ValueError("batch_input must be provided for convolutional RR-CBP")
        if not self.rr_config.rrcbp_enabled:
            return

        (
            features_to_replace_input,
            features_to_replace_output,
            num_features_to_replace,
        ) = self.test_features(features=features)

        if all(count == 0 for count in num_features_to_replace):
            return

        with torch.no_grad():
            for layer_idx, count in enumerate(num_features_to_replace):
                if count == 0:
                    continue
                
                # Use Map Logic or legacy
                if self.use_map:
                    current_layer = self.plasticity_map[layer_idx]['weight_module']
                else:
                    current_layer = self.net[layer_idx * 2]

                self._ensure_covariance(layer_idx, current_layer)
                sigma_state = self.layer_covariances[layer_idx]
                assert sigma_state is not None

                H_prev = self._compute_layer_inputs(layer_idx=layer_idx, features=features, batch_input=batch_input)
                sigma = sigma_state.update(H_prev, dtype=self.rr_config.covariance_dtype)

                keep_idx = self._kept_indices(layer_idx, features_to_replace_input[layer_idx])
                stats = self._replace_units(
                    layer_idx=layer_idx,
                    keep_idx=keep_idx,
                    replace_input_idx=features_to_replace_input[layer_idx],
                    replace_output_idx=features_to_replace_output[layer_idx],
                    sigma=sigma,
                    H_prev=H_prev,
                    activations=features[layer_idx],
                )

                self._post_replacement_housekeeping(
                    layer_idx,
                    replace_input_idx=features_to_replace_input[layer_idx],
                    replace_output_idx=features_to_replace_output[layer_idx],
                )

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
        if isinstance(layer, Conv2d):
            in_dim = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, Linear):
            in_dim = layer.in_features
        else:
            raise TypeError("RR-CBP for ConvNet supports Conv2d and Linear layers only")
        device = layer.weight.device
        dtype = layer.weight.dtype
        self.layer_covariances[layer_idx] = initialize_covariance(
            d_dim=in_dim,
            device=device,
            beta=self.rr_config.sigma_ema_beta,
            ridge=self.rr_config.sigma_ridge,
            diag_only=self.rr_config.diag_sigma_only,
            dtype=dtype,
        )

    def _compute_layer_inputs(self, layer_idx: int, features: List[Tensor], batch_input: Tensor) -> Tensor:
        # use Map Logic or legacy
        if self.use_map:
            current_layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            current_layer = self.net[layer_idx * 2]

        if isinstance(current_layer, Conv2d):
            layer_input = batch_input if layer_idx == 0 else features[layer_idx - 1]
            patches = F.unfold(
                layer_input,
                kernel_size=current_layer.kernel_size,
                dilation=current_layer.dilation,
                padding=current_layer.padding,
                stride=current_layer.stride,
            )
            batch_size, dim, spatial = patches.shape
            return patches.permute(1, 0, 2).reshape(dim, batch_size * spatial)
        if isinstance(current_layer, Linear):
            if layer_idx == 0:
                raise ValueError("Linear layer cannot be the first hidden layer in ConvNet RR-CBP")
            prev = features[layer_idx - 1]
            batch_size = prev.shape[0]
            return prev.view(batch_size, -1).t()
        raise TypeError("Unsupported layer type in ConvNet RR-CBP")

    def _kept_indices(self, layer_idx: int, replace_idx: Tensor) -> Tensor:
        # Use Map Logic or legacy
        if self.use_map:
            layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            layer = self.net[layer_idx * 2]

        if isinstance(layer, Conv2d):
            total = layer.out_channels
        elif isinstance(layer, Linear):
            total = layer.out_features
        else:
            raise TypeError("Unsupported layer type for keep index computation")
        mask = torch.ones(total, dtype=torch.bool, device=replace_idx.device)
        mask[replace_idx] = False
        return torch.arange(total, device=replace_idx.device)[mask]

    def _replace_units(
        self,
        layer_idx: int,
        keep_idx: Tensor,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
        sigma: Tensor,
        H_prev: Tensor,
        activations: Tensor,
    ) -> LayerReplacementStats:
        # Use Map Logic or legacy
        if self.use_map:
            map_item = self.plasticity_map[layer_idx]
            layer = map_item['weight_module']
            next_layer = map_item['outgoing_module']
            # Correct bias compensation check for norms
            should_compensate = not map_item.get('outgoing_feeds_into_norm', False)
        else:
            layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]
            should_compensate = True

        config = self.rr_config

        weight_matrix = self._flatten_weight(layer)
        geometry = SigmaGeometry(sigma=sigma, diag_only=config.diag_sigma_only, eps=config.sigma_eig_floor)
        
        # Detect if sigma_geometry fell back to CPU due to GPU corruption
        # If so, move all tensors to CPU to avoid device mismatch errors
        sigma_device = geometry._sqrt.device if hasattr(geometry, '_sqrt') else sigma.device
        force_cpu = sigma.is_cuda and sigma_device.type == 'cpu'
        
        if force_cpu:
            import warnings
            warnings.warn(f"GPU corruption detected in layer {layer_idx}. Skipping neuron replacement for this batch.")
            # GPU is too corrupted to even transfer data - skip replacement entirely
            # Return early with zero replacements
            stats = self.layer_stats.setdefault(layer_idx, LayerReplacementStats())
            return stats
        
        kept_vectors = weight_matrix[keep_idx, :].t()
        projector = SigmaProjector(geometry=geometry, basis=kept_vectors, reg_epsilon=config.projector_reg_epsilon)
        stats = self.layer_stats.setdefault(layer_idx, LayerReplacementStats())

        chi0 = self._resolve_chi0(activations)
        v_target = geometry.trace / max(geometry.dim, 1)
        q_target = v_target / max(chi0, config.proj_eps)

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
            replacements=int(replace_input_idx.numel()),
        )

        recent_allocations: List[float] = []
        recent_saturated = 0

        for idx_pos, unit_idx in enumerate(replace_input_idx.tolist()):
            # Only compensate bias if appropriate
            if should_compensate and self.ages[layer_idx][unit_idx] > 0:
                bias_corrected_act = self.mean_feature_act[layer_idx][unit_idx] / (
                    1 - self.decay_rate ** self.ages[layer_idx][unit_idx]
                )
                self._transfer_bias(
                    layer_idx=layer_idx,
                    next_layer=next_layer,
                    current_layer=layer,
                    unit_idx=unit_idx,
                    idx_pos=idx_pos,
                    replace_input_idx=replace_input_idx,
                    replace_output_idx=replace_output_idx,
                    bias_corrected_act=bias_corrected_act,
                )

            direction, used_fallback = self._sample_sigma_direction(projector, geometry, weight_matrix.dtype)
            q_alloc, saturated = allocator.allocate()
            if saturated:
                stats.saturated += 1
                recent_saturated += 1

            norm_dir = torch.clamp(geometry.norm(direction), min=config.proj_eps)
            scale = math.sqrt(max(q_alloc, config.proj_eps)) / norm_dir
            w_in = direction * scale

            projector.add_vector(w_in)
            self._assign_weight(layer, unit_idx, w_in)

            acts = (w_in.unsqueeze(0) @ H_prev).squeeze(0)
            bias_value = self._center_bias(acts)
            layer.bias.data[unit_idx] = bias_value.to(device=layer.bias.device, dtype=layer.bias.dtype)

            self._zero_and_seed_outgoing(
                layer_idx=layer_idx,
                next_layer=next_layer,
                unit_idx=unit_idx,
                idx_pos=idx_pos,
                replace_input_idx=replace_input_idx,
                replace_output_idx=replace_output_idx,
            )

            if used_fallback:
                stats.fallbacks += 1
            else:
                stats.successful += 1

            stats.allocated_q.append(q_alloc)
            recent_allocations.append(q_alloc)

        if replace_input_idx.numel() > 0:
            self.mean_feature_act[layer_idx][replace_input_idx] = 0.0
            self.mean_abs_feature_act[layer_idx][replace_input_idx] = 0.0
            self.util[layer_idx][replace_input_idx] = 0.0
            self.ages[layer_idx][replace_input_idx] = 0.0

        stats.last_allocations = recent_allocations
        stats.last_saturated = recent_saturated
        stats.last_Q_target = q_target * weight_matrix.size(0)
        stats.last_Q_used = geometry.matrix_energy(self._flatten_weight(layer))
        stats.last_lambda_min_white = geometry.lambda_min_whitened(self._flatten_weight(layer))
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

    def _flatten_weight(self, layer: Module) -> Tensor:
        if isinstance(layer, Conv2d):
            return layer.weight.data.view(layer.out_channels, -1)
        if isinstance(layer, Linear):
            return layer.weight.data
        raise TypeError("Unsupported layer type for flattening")

    def _assign_weight(self, layer: Module, unit_idx: int, vector: Tensor) -> None:
        if isinstance(layer, Conv2d):
            layer.weight.data[unit_idx] = vector.view_as(layer.weight.data[unit_idx])
        elif isinstance(layer, Linear):
            layer.weight.data[unit_idx, :] = vector
        else:
            raise TypeError("Unsupported layer type during assignment")

    def _transfer_bias(
        self,
        layer_idx: int,
        next_layer: Module,
        current_layer: Module,
        unit_idx: int,
        idx_pos: int,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
        bias_corrected_act: Tensor,
    ) -> None:
        if isinstance(next_layer, Conv2d):
            # Ensure proper broadcasting for Conv bias compensation
            if isinstance(current_layer, Conv2d):
               # Conv->Conv usually (In, H, W) vs (Out, In, k, k)
               # Standard logic
               contribution = (next_layer.weight.data[:, unit_idx, ...] * bias_corrected_act).sum(dim=(1, 2))
            else:
                contribution = (next_layer.weight.data[:, unit_idx, ...] * bias_corrected_act).sum(dim=(1, 2))
           
            next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)
        elif isinstance(next_layer, Linear):
            cols = self._resolve_output_columns(
                layer_idx=layer_idx,
                idx_pos=idx_pos,
                replace_input_idx=replace_input_idx,
                replace_output_idx=replace_output_idx,
                current_layer=current_layer,
            )
            if cols.numel() > 0:
                contribution = (next_layer.weight.data[:, cols] * bias_corrected_act).sum(dim=1)
                next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

    def _resolve_output_columns(
        self,
        layer_idx: int,
        idx_pos: int,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
        current_layer: Module,
    ) -> Tensor:
        if replace_output_idx.numel() == 0:
            return torch.empty(0, dtype=replace_input_idx.dtype, device=replace_input_idx.device)
        if isinstance(current_layer, Conv2d):
            multiplier = self._conv_output_multipliers[layer_idx]
            if multiplier > 1:
                start = idx_pos * multiplier
                end = start + multiplier
                return replace_output_idx[start:end]
        if replace_output_idx.numel() == replace_input_idx.numel():
            return replace_output_idx[idx_pos : idx_pos + 1]
        return replace_output_idx

    def _zero_and_seed_outgoing(
        self,
        layer_idx: int,
        next_layer: Module,
        unit_idx: int,
        idx_pos: int,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
    ) -> None:
        epsilon = self.rr_config.epsilon_micro_seed
        use_seed = self.rr_config.use_micro_seed and epsilon > 0.0
        if isinstance(next_layer, Conv2d):
            next_layer.weight.data[:, unit_idx, ...] = 0.0
            if use_seed:
                noise = torch.randn_like(next_layer.weight.data[:, unit_idx, ...])
                noise = noise - noise.mean(dim=(1, 2), keepdim=True)
                norm = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)
                norm = torch.clamp(norm, min=self.rr_config.proj_eps)
                noise = noise / norm.view(-1, 1, 1)
                next_layer.weight.data[:, unit_idx, ...] = epsilon * noise
        elif isinstance(next_layer, Linear):
            
            if self.use_map:
                current_layer = self.plasticity_map[layer_idx]['weight_module']
            else:
                current_layer = self.net[layer_idx * 2]
                  
            
            cols = self._resolve_output_columns(layer_idx, idx_pos, replace_input_idx, replace_output_idx, current_layer)
            if cols.numel() > 0:
                next_layer.weight.data[:, cols] = 0.0
                if use_seed:
                    noise = torch.randn(next_layer.weight.size(0), cols.numel(), device=next_layer.weight.device, dtype=next_layer.weight.dtype)
                    noise = noise - noise.mean(dim=0, keepdim=True)
                    norm = torch.norm(noise, dim=0, keepdim=True)
                    norm = torch.clamp(norm, min=self.rr_config.proj_eps)
                    next_layer.weight.data[:, cols] = epsilon * (noise / norm)
            else:
                next_layer.weight.data[:, unit_idx] = 0.0
                if use_seed:
                    noise = torch.randn(next_layer.weight.size(0), device=next_layer.weight.device, dtype=next_layer.weight.dtype)
                    noise = noise - noise.mean()
                    norm = torch.clamp(noise.norm(), min=self.rr_config.proj_eps)
                    next_layer.weight.data[:, unit_idx] = epsilon * (noise / norm)
        else:
            next_layer.weight.data[:, unit_idx] = 0.0

    def _resolve_chi0(self, activations: Tensor) -> float:
        if self.rr_config.chi0_override is not None:
            return float(self.rr_config.chi0_override)
        if self.rr_config.estimate_chi0_from_batch and activations is not None and activations.numel() > 0:
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
        # Use Map Logic or legacy
        if self.use_map:
            layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            layer = self.net[layer_idx * 2]
            
        weight = self._flatten_weight(layer)
        
        # Try to compute gram matrix, but if GPU is corrupted, return placeholder metrics
        try:
            if self.rr_config.diag_sigma_only:
                gram = (weight * sigma.unsqueeze(0)) @ weight.t()
            else:
                gram = weight @ sigma @ weight.t()
        except RuntimeError as e:
            # GPU corruption detected during gram matrix computation
            import warnings
            warnings.warn(f"GPU corruption detected in rank metrics computation for layer {layer_idx}. "
                         f"Returning placeholder metrics. Error: {e}")
            # Return placeholder metrics to allow training to continue
            return {
                f'layer_{layer_idx}_rank': float('nan'),
                f'layer_{layer_idx}_lambda_min': float('nan'),
                f'layer_{layer_idx}_active_frac': 0.0,
                f'layer_{layer_idx}_fallback': 1.0,
                f'layer_{layer_idx}_success': 0.0,
                f'layer_{layer_idx}_saturated': float(stats.saturated),
                f'layer_{layer_idx}_Q_mean': float('nan'),
                f'layer_{layer_idx}_Q_std': float('nan'),
            }

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
        saturated_fraction = float(stats.last_saturated / max(len(allocations), 1)) if allocations else 0.0
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
                "RR-CBP Conv layer %d | rank %.2f | lambda_min %.3e | lambda_min_white %.3e "
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

    def _post_replacement_housekeeping(
        self,
        layer_idx: int,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
    ) -> None:
        
        #  Use Map Logic or legacy
        if self.use_map:
            map_item = self.plasticity_map[layer_idx]
            layer = map_item['weight_module']
            next_layer = map_item['outgoing_module']
        else:
            layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]
            
        if isinstance(self.opt, AdamGnT) and replace_input_idx.numel() > 0:
            self.opt.state[layer.weight]["exp_avg"][replace_input_idx, ...] = 0.0
            self.opt.state[layer.weight]["exp_avg_sq"][replace_input_idx, ...] = 0.0
            self.opt.state[layer.weight]["step"][replace_input_idx, ...] = 0
            self.opt.state[layer.bias]["exp_avg"][replace_input_idx] = 0.0
            self.opt.state[layer.bias]["exp_avg_sq"][replace_input_idx] = 0.0
            self.opt.state[layer.bias]["step"][replace_input_idx] = 0
            if isinstance(next_layer, Conv2d):
                self.opt.state[next_layer.weight]["exp_avg"][:, replace_input_idx, ...] = 0.0
                self.opt.state[next_layer.weight]["exp_avg_sq"][:, replace_input_idx, ...] = 0.0
                self.opt.state[next_layer.weight]["step"][:, replace_input_idx, ...] = 0
            elif isinstance(next_layer, Linear):
                cols = replace_output_idx if replace_output_idx.numel() > 0 else replace_input_idx
                self.opt.state[next_layer.weight]["exp_avg"][:, cols] = 0.0
                self.opt.state[next_layer.weight]["exp_avg_sq"][:, cols] = 0.0
                self.opt.state[next_layer.weight]["step"][:, cols] = 0
        if replace_input_idx.numel() > 0:
            self.accumulated_num_features_to_replace[layer_idx] = 0.0

    def get_layer_stats(self) -> Dict[int, LayerReplacementStats]:
        return self.layer_stats
