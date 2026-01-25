"""
Rank-Restoring Generate-and-Test for Convolutional Networks (Version 2).

This implementation follows the RR_CBP_2_algorithm_guide.md precisely.
It inherits from ConvGnT_for_ConvNet to reuse age tracking, utility computation, and test_features,
but overrides the feature generation to use Σ-orthogonal weight initialization.

Two modes:
- RR-CBP2: Σ-orthogonal directions with unit Σ-norm
- RR-CBP-E2: Σ-orthogonal directions with energy budget control
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module

from configs.configurations import RRCBP2Config
from src.algos.AdamGnT import AdamGnT
from src.algos.gnt import ConvGnT_for_ConvNet
from src.algos.supervised.rank_restoring.rr_covariance import CovarianceState, initialize_covariance
from src.algos.supervised.rank_restoring.sigma_geometry import (
    EnergyAllocator,
    SigmaGeometry,
    SigmaProjector,
    chi0_for_activation,
)
from src.algos.supervised.rr_gnt2_fc import LayerReplacementStats2


class RR_GnT2_for_ConvNet(ConvGnT_for_ConvNet):
    """
    Rank-Restoring Generate-and-Test for convolutional networks (Version 2).
    
    This class inherits from ConvGnT_for_ConvNet to reuse:
    - Age tracking (self.ages)
    - Utility computation (update_utility, test_features)
    - Mean feature activation tracking (self.mean_feature_act)
    
    It overrides the gen_and_test method to implement Σ-orthogonal weight initialization
    following the algorithm guide in RR_CBP_2_algorithm_guide.md.
    """

    def __init__(
        self,
        net: Union[List[Module], nn.Module], 
        hidden_activation: str,
        opt: AdamGnT,
        config: RRCBP2Config,
        loss_func,
        device: str = "cpu",
        num_last_filter_outputs: int = 1,
    ) -> None:
        # Initialize base ConvGnT class
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

        # Store configuration
        self.config = config
        
        # Initialize covariance states for each hidden layer
        self.layer_covariances: List[Optional[CovarianceState]] = [
            None for _ in range(self.num_hidden_layers)
        ]
        
        # Statistics tracking
        self.layer_stats: Dict[int, LayerReplacementStats2] = {}
        self._layer_log_counters: List[int] = [0 for _ in range(self.num_hidden_layers)]
        
        # Track conv output multipliers for handling Conv2d -> Linear transitions
        self._conv_output_multipliers: List[int] = []
        
        
        for layer_idx in range(self.num_hidden_layers):
            
            #Use Map or Legacy Logic
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
       
        # Activation function for chi0 computation
        self.hidden_activation = hidden_activation or "linear"
        self._leaky_slope = 0.01
        # FIX: Handle both List and Module types for iteration
        iterable_net = net if isinstance(net, list) else net.modules()
        
        for module in iterable_net:
            if isinstance(module, nn.LeakyReLU):
                self._leaky_slope = float(module.negative_slope)
                break
        
        self._chi0_constant = chi0_for_activation(self.hidden_activation, self._leaky_slope)
        self._logger = logging.getLogger(__name__)

    def gen_and_test(self, features: List[Tensor], batch_input: Optional[Tensor] = None) -> None:
        """
        Perform generate-and-test with Σ-orthogonal weight initialization.
        
        Args:
            features: List of activation tensors for each hidden layer
            batch_input: Input batch tensor (required for computing layer inputs)
        """
        if not isinstance(features, list):
            raise TypeError("features passed to generate-and-test should be a list")
        if batch_input is None:
            raise ValueError("batch_input must be provided for ConvNet RR-CBP2")
        if not self.config.rrcbp_enabled:
            return

        # Use inherited test_features to determine which neurons to replace
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

                # REFACTOR: Map or Legacy
                if self.use_map:
                    map_item = self.plasticity_map[layer_idx]
                    current_layer = map_item['weight_module']
                else:
                    current_layer = self.net[layer_idx * 2]
                
                
                # Ensure covariance state is initialized
                self._ensure_covariance(layer_idx, current_layer)
                sigma_state = self.layer_covariances[layer_idx]
                assert sigma_state is not None

                # Compute upstream features H_prev
                H_prev = self._compute_layer_inputs(layer_idx, features, batch_input)
                
                # Update covariance EMA and get current Σ
                sigma = sigma_state.update(H_prev, dtype=self.config.covariance_dtype)

                # Get kept indices
                keep_idx = self._kept_indices(layer_idx, features_to_replace_input[layer_idx])

                # Replace units with Σ-orthogonal directions
                stats = self._replace_units(
                    layer_idx=layer_idx,
                    keep_idx=keep_idx,
                    replace_input_idx=features_to_replace_input[layer_idx],
                    replace_output_idx=features_to_replace_output[layer_idx],
                    sigma=sigma,
                    H_prev=H_prev,
                    activations=features[layer_idx],
                )

                # Reset optimizer state and statistics
                self._post_replacement_housekeeping(
                    layer_idx,
                    replace_input_idx=features_to_replace_input[layer_idx],
                    replace_output_idx=features_to_replace_output[layer_idx],
                )

                # Optionally log rank metrics
                if self.config.log_rank_metrics_every > 0 and count > 0:
                    self._layer_log_counters[layer_idx] += 1
                    if self._layer_log_counters[layer_idx] >= self.config.log_rank_metrics_every:
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
        """Initialize covariance state for a layer if not already done."""
        if self.layer_covariances[layer_idx] is not None:
            return
        
        if isinstance(layer, Conv2d):
            in_dim = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, Linear):
            in_dim = layer.in_features
        else:
            raise TypeError("RR-CBP2 for ConvNet supports Conv2d and Linear layers only")
        
        device = layer.weight.device
        dtype = layer.weight.dtype
        self.layer_covariances[layer_idx] = initialize_covariance(
            d_dim=in_dim,
            device=device,
            beta=self.config.sigma_ema_beta,
            ridge=self.config.sigma_ridge,
            diag_only=self.config.diag_sigma_only,
            dtype=dtype,
        )

    def _compute_layer_inputs(
        self,
        layer_idx: int,
        features: List[Tensor],
        batch_input: Tensor,
    ) -> Tensor:
        """
        Compute upstream features H_prev for the given layer.
        
        For Conv2d layers, uses unfold to extract patches.
        For Linear layers, flattens the input.
        
        Returns:
            H_prev: Tensor of shape (d, m) where d is input dimension
        """
        # REFACTOR: Map or Legacy
        if self.use_map:
            current_layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            current_layer = self.net[layer_idx * 2]
        
        
        if isinstance(current_layer, Conv2d):
            # Get layer input
            layer_input = batch_input if layer_idx == 0 else features[layer_idx - 1]
            
            # Extract patches using unfold
            patches = F.unfold(
                layer_input,
                kernel_size=current_layer.kernel_size,
                dilation=current_layer.dilation,
                padding=current_layer.padding,
                stride=current_layer.stride,
            )
            # patches shape: (batch, d, spatial)
            batch_size, dim, spatial = patches.shape
            # Reshape to (d, batch * spatial)
            return patches.permute(1, 0, 2).reshape(dim, batch_size * spatial)
        
        elif isinstance(current_layer, Linear):
            if layer_idx == 0:
                raise ValueError("Linear layer cannot be the first hidden layer in ConvNet RR-CBP2")
            prev = features[layer_idx - 1]
            batch_size = prev.shape[0]
            # Flatten and transpose to (d, batch)
            return prev.view(batch_size, -1).t()
        
        raise TypeError("Unsupported layer type in ConvNet RR-CBP2")

    def _kept_indices(self, layer_idx: int, replace_idx: Tensor) -> Tensor:
        """Get indices of units that are kept (not replaced)."""
        
        # REFACTOR: Map or Legacy
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

    def _flatten_weight(self, layer: Module) -> Tensor:
        """Flatten weight tensor to 2D (out_features, in_features)."""
        if isinstance(layer, Conv2d):
            # (out_channels, in_channels, kH, kW) -> (out_channels, in_channels * kH * kW)
            return layer.weight.data.view(layer.out_channels, -1)
        elif isinstance(layer, Linear):
            return layer.weight.data
        raise TypeError("Unsupported layer type")

    def _replace_units(
        self,
        layer_idx: int,
        keep_idx: Tensor,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
        sigma: Tensor,
        H_prev: Tensor,
        activations: Tensor,
    ) -> LayerReplacementStats2:
        """
        Core replacement logic following the algorithm guide.
        """
        # REFACTOR: Map or Legacy
        should_compensate = True
        if self.use_map:
            map_item = self.plasticity_map[layer_idx]
            layer = map_item['weight_module']
            next_layer = map_item['outgoing_module']
            should_compensate = not map_item.get('outgoing_feeds_into_norm', False)
        else:
            layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]
        
        config = self.config

        # Get flattened weight matrix
        weight_matrix = self._flatten_weight(layer)
        
        # Build SigmaGeometry
        geometry = SigmaGeometry(
            sigma=sigma,
            diag_only=config.diag_sigma_only,
            eps=config.sigma_eig_floor,
        )
        
        # Get kept weight vectors as columns
        kept_vectors = weight_matrix[keep_idx, :].t()
        
        # Build SigmaProjector
        projector = SigmaProjector(
            geometry=geometry,
            basis=kept_vectors,
            reg_epsilon=config.projector_reg_epsilon,
        )
        
        stats = self.layer_stats.setdefault(layer_idx, LayerReplacementStats2())

        # Energy allocation setup (for RR-CBP-E2)
        allocator = None
        q_target = 1.0
        if config.use_energy_budget:
            chi0 = self._resolve_chi0(activations)
            v_target = geometry.trace / max(geometry.dim, 1)
            q_target = v_target / max(chi0, config.proj_eps)
            
            # Compute used energy
            used_energy = 0.0
            if kept_vectors.numel() > 0:
                for col in range(kept_vectors.size(1)):
                    used_energy += geometry.vector_energy(kept_vectors[:, col])
            
            # Optional lambda_star
            lambda_star = None
            if config.use_lambda_star:
                if config.lambda_star is not None:
                    lambda_star = float(config.lambda_star)
                elif kept_vectors.numel() > 0:
                    whitened = geometry.whiten_columns(kept_vectors)
                    gram_white = whitened.t() @ whitened
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

        dtype = weight_matrix.dtype
        recent_allocations: List[float] = []
        recent_saturated = 0
        new_vectors: List[Tensor] = []

        for idx_pos, unit_idx in enumerate(replace_input_idx.tolist()):
            # (1) Bias transfer
            # LOGIC FIX: Check if we should compensate for bias shift 
            # (i.e. don't if next layer goes to BatchNorm)
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

            # (2) Sample Σ-orthogonal direction
            direction, used_fallback = self._sample_sigma_direction(projector, geometry, dtype)
            
            # (2b) Optionally orthonormalize
            if config.orthonormalize_batch and new_vectors:
                direction = self._orthonormalize_direction(direction, new_vectors, geometry)

            # (3) Scale direction
            if config.use_energy_budget and allocator is not None:
                q_alloc, saturated = allocator.allocate()
                if saturated:
                    stats.saturated += 1
                    recent_saturated += 1
                w_scaled = self._scale_to_energy(direction, geometry, q_alloc)
                recent_allocations.append(float(q_alloc))
            else:
                w_scaled = direction
                q_alloc = 1.0

            # (4) Update projector
            projector.add_vector(w_scaled)
            new_vectors.append(w_scaled / torch.clamp(geometry.norm(w_scaled), min=config.proj_eps))

            # (5) Assign weights
            self._assign_weight(layer, unit_idx, w_scaled)

            # (6) Bias centering
            acts = (w_scaled.unsqueeze(0) @ H_prev).squeeze(0)
            bias_value = self._center_bias(acts)
            layer.bias.data[unit_idx] = bias_value.to(device=layer.bias.device, dtype=layer.bias.dtype)

            # (7) Zero outgoing weights
            self._zero_and_seed_outgoing(
                layer_idx=layer_idx,
                next_layer=next_layer,
                unit_idx=unit_idx,
                idx_pos=idx_pos,
                replace_input_idx=replace_input_idx,
                replace_output_idx=replace_output_idx,
            )

            # Track statistics
            if used_fallback:
                stats.fallbacks += 1
            else:
                stats.successful += 1
            stats.allocated_q.append(float(q_alloc))

        # Reset statistics
        if replace_input_idx.numel() > 0:
            self.mean_feature_act[layer_idx][replace_input_idx] = 0.0
            self.util[layer_idx][replace_input_idx] = 0.0
            self.ages[layer_idx][replace_input_idx] = 0.0

        stats.last_allocations = recent_allocations
        stats.last_saturated = recent_saturated
        if config.use_energy_budget:
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
        """Sample a direction in the Σ-orthogonal complement."""
        attempts = max(1, int(self.config.max_proj_trials))
        last_candidate: Optional[Tensor] = None
        
        for _ in range(attempts):
            u = torch.randn(geometry.dim, device=geometry.sigma.device, dtype=dtype)
            residual = projector.project_complement(u)
            norm = geometry.norm(residual)
            
            if norm > self.config.proj_eps:
                return residual / norm, False
            
            last_candidate = residual

        fallback = projector.least_covered_direction(dtype=dtype)
        norm = geometry.norm(fallback)
        
        if norm <= self.config.proj_eps and last_candidate is not None:
            fallback = last_candidate
            norm = geometry.norm(fallback)
        
        norm = torch.clamp(norm, min=self.config.proj_eps)
        return fallback / norm, True

    def _orthonormalize_direction(
        self,
        direction: Tensor,
        existing: List[Tensor],
        geometry: SigmaGeometry,
    ) -> Tensor:
        """Orthonormalize direction against existing vectors."""
        adjusted = direction.clone()
        for vec in existing:
            coeff = geometry.inner(vec, adjusted)
            adjusted = adjusted - coeff * vec
        norm = geometry.norm(adjusted)
        if norm <= self.config.proj_eps:
            return direction
        return adjusted / norm

    def _scale_to_energy(self, direction: Tensor, geometry: SigmaGeometry, q_alloc: float) -> Tensor:
        """Scale direction to achieve target energy."""
        norm = geometry.norm(direction)
        if norm < self.config.proj_eps or q_alloc <= 0:
            return direction
        scale = math.sqrt(q_alloc) / norm
        return direction * scale

    def _assign_weight(self, layer: Module, unit_idx: int, vector: Tensor) -> None:
        """Assign weight vector to the appropriate layer format."""
        if isinstance(layer, Conv2d):
            # Reshape flat vector to (in_channels, kH, kW)
            layer.weight.data[unit_idx] = vector.view(
                layer.in_channels,
                layer.kernel_size[0],
                layer.kernel_size[1],
            )
        elif isinstance(layer, Linear):
            layer.weight.data[unit_idx, :] = vector

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
        """Transfer bias from removed unit to next layer."""
        if isinstance(current_layer, Conv2d) and isinstance(next_layer, Linear):
            # Conv -> Linear transition
            multiplier = self._conv_output_multipliers[layer_idx]
            start_idx = unit_idx * multiplier
            end_idx = start_idx + multiplier
            contribution = (
                next_layer.weight.data[:, start_idx:end_idx].sum(dim=1) * bias_corrected_act
            )
            next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)
        elif isinstance(next_layer, Conv2d):
            # Conv -> Conv
            contribution = (
                next_layer.weight.data[:, unit_idx, :, :].sum(dim=(1, 2)) * bias_corrected_act
            )
            next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)
        elif isinstance(next_layer, Linear):
            # Linear -> Linear
            contribution = next_layer.weight.data[:, unit_idx] * bias_corrected_act
            next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

    def _zero_and_seed_outgoing(
        self,
        layer_idx: int,
        next_layer: Module,
        unit_idx: int,
        idx_pos: int,
        replace_input_idx: Tensor,
        replace_output_idx: Tensor,
    ) -> None:
        """Zero outgoing weights and optionally micro-seed."""
        epsilon = self.config.epsilon_micro_seed
        use_seed = self.config.use_micro_seed and epsilon > 0.0
        
        # Map or Legacy for current_layer resolution
        if self.use_map:
            current_layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            current_layer = self.net[layer_idx * 2]
        
        if isinstance(current_layer, Conv2d) and isinstance(next_layer, Linear):
            # Conv -> Linear transition
            multiplier = self._conv_output_multipliers[layer_idx]
            start_idx = unit_idx * multiplier
            end_idx = start_idx + multiplier
            next_layer.weight.data[:, start_idx:end_idx] = 0.0
            if use_seed:
                noise = torch.randn_like(next_layer.weight.data[:, start_idx:end_idx])
                noise = noise - noise.mean()
                norm = torch.clamp(noise.norm(), min=self.config.proj_eps)
                next_layer.weight.data[:, start_idx:end_idx] = epsilon * (noise / norm)
        elif isinstance(next_layer, Conv2d):
            # Conv -> Conv
            next_layer.weight.data[:, unit_idx, :, :] = 0.0
            if use_seed:
                noise = torch.randn_like(next_layer.weight.data[:, unit_idx, :, :])
                noise = noise - noise.mean()
                norm = torch.clamp(noise.norm(), min=self.config.proj_eps)
                next_layer.weight.data[:, unit_idx, :, :] = epsilon * (noise / norm)
        elif isinstance(next_layer, Linear):
            # Linear -> Linear
            next_layer.weight.data[:, unit_idx] = 0.0
            if use_seed:
                noise = torch.randn_like(next_layer.weight.data[:, unit_idx])
                noise = noise - noise.mean()
                norm = torch.clamp(noise.norm(), min=self.config.proj_eps)
                next_layer.weight.data[:, unit_idx] = epsilon * (noise / norm)

    def _resolve_chi0(self, activations: Tensor) -> float:
        """Resolve χ₀(φ) constant."""
        if self.config.chi0_override is not None:
            return float(self.config.chi0_override)
        if (
            self.config.estimate_chi0_from_batch
            and activations is not None
            and activations.numel() > 0
        ):
            value = torch.clamp(activations.reshape(-1).pow(2).mean(), min=self.config.proj_eps)
            return float(value.item())
        return float(self._chi0_constant)

    def _center_bias(self, activations: Tensor) -> Tensor:
        """Compute bias to center preactivations."""
        if activations.numel() == 0:
            return torch.zeros(1, device=activations.device, dtype=activations.dtype).squeeze(0)
        if self.config.center_bias == "median":
            return -torch.median(activations)
        return -activations.mean()

    def _compute_rank_metrics(
        self,
        layer_idx: int,
        sigma: Tensor,
        activations: Tensor,
        stats: LayerReplacementStats2,
    ) -> Dict[str, float]:
        """Compute rank metrics for logging."""
        # Map or Legacy
        if self.use_map:
            layer = self.plasticity_map[layer_idx]['weight_module']
        else:
            layer = self.net[layer_idx * 2]
        
        weight = self._flatten_weight(layer)
        
        if self.config.diag_sigma_only:
            gram = (weight * sigma.unsqueeze(0)) @ weight.t()
        else:
            gram = weight @ sigma @ weight.t()

        try:
            rank_val = float(torch.linalg.matrix_rank(gram, tol=self.config.proj_eps).item())
        except RuntimeError:
            rank_val = float(torch.linalg.matrix_rank(gram.cpu().double(), tol=self.config.proj_eps).item())

        lambda_min = float("nan")
        
        # Check if we should use CPU eigendecomposition
        force_cpu_eigh = os.environ.get('SIGMA_FORCE_CPU_EIGH', '0') == '1'
        
        try:
            if force_cpu_eigh and gram.device.type == 'cuda':
                # Use CPU path directly
                gram_cpu = gram.detach().cpu()
                eigvals = torch.linalg.eigvalsh(gram_cpu)
                eigvals = eigvals.to(gram.device, dtype=gram.dtype)
            else:
                eigvals = torch.linalg.eigvalsh(gram)
        except RuntimeError:
            # Fallback: try CPU with double precision and stronger regularization
            try:
                gram_cpu = gram.detach().cpu().double()
                # Add regularization to improve conditioning
                reg = 1e-6 * torch.trace(gram_cpu) / max(gram_cpu.size(0), 1)
                gram_reg = gram_cpu + reg * torch.eye(gram_cpu.size(0), dtype=torch.float64)
                eigvals = torch.linalg.eigvalsh(gram_reg)
                eigvals = eigvals.to(gram.device, dtype=gram.dtype)
            except RuntimeError:
                # Last resort: return NaN
                eigvals = torch.tensor([], device=gram.device, dtype=gram.dtype)
        
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
        """Emit rank metrics to logger."""
        if self._logger is None:
            return
        self._logger.info(
            (
                "RR-CBP2 Conv layer %d | rank %.2f | lambda_min %.3e | lambda_min_white %.3e "
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
        """Reset optimizer state for replaced units."""
        if replace_input_idx.numel() == 0:
            return

        # Map or Legacy
        if self.use_map:
            map_item = self.plasticity_map[layer_idx]
            layer = map_item['weight_module']
            next_layer = map_item['outgoing_module']
        else:
            layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]

        if isinstance(self.opt, AdamGnT):
            # Reset for current layer
            # LOGIC FIX: Check if state exists using dict .get() or `in` check 
            # to be robust against frozen layers or missing Grads
            
            
            # --- Current Weights ---
            if layer.weight in self.opt.state and 'exp_avg' in self.opt.state[layer.weight]:            
                # Reset for current layer
                if isinstance(layer, Conv2d):
                    self.opt.state[layer.weight]["exp_avg"][replace_input_idx, :, :, :] = 0.0
                    self.opt.state[layer.weight]["exp_avg_sq"][replace_input_idx, :, :, :] = 0.0
                    self.opt.state[layer.weight]["step"][replace_input_idx, :, :, :] = 0
                elif isinstance(layer, Linear):
                    self.opt.state[layer.weight]["exp_avg"][replace_input_idx, :] = 0.0
                    self.opt.state[layer.weight]["exp_avg_sq"][replace_input_idx, :] = 0.0
                    self.opt.state[layer.weight]["step"][replace_input_idx, :] = 0
            # --- Current Bias ---
            if layer.bias in self.opt.state and 'exp_avg' in self.opt.state[layer.bias]:               
                self.opt.state[layer.bias]["exp_avg"][replace_input_idx] = 0.0
                self.opt.state[layer.bias]["exp_avg_sq"][replace_input_idx] = 0.0
                self.opt.state[layer.bias]["step"][replace_input_idx] = 0

            # Reset for next layer outgoing weights
            if next_layer.weight in self.opt.state and 'exp_avg' in self.opt.state[next_layer.weight]:
                if isinstance(layer, Conv2d) and isinstance(next_layer, Linear):
                    multiplier = self._conv_output_multipliers[layer_idx]
                    for idx in replace_input_idx.tolist():
                        start = idx * multiplier
                        end = start + multiplier
                        self.opt.state[next_layer.weight]["exp_avg"][:, start:end] = 0.0
                        self.opt.state[next_layer.weight]["exp_avg_sq"][:, start:end] = 0.0
                        self.opt.state[next_layer.weight]["step"][:, start:end] = 0
                elif isinstance(next_layer, Conv2d):
                    self.opt.state[next_layer.weight]["exp_avg"][:, replace_input_idx, :, :] = 0.0
                    self.opt.state[next_layer.weight]["exp_avg_sq"][:, replace_input_idx, :, :] = 0.0
                    self.opt.state[next_layer.weight]["step"][:, replace_input_idx, :, :] = 0
                elif isinstance(next_layer, Linear):
                    self.opt.state[next_layer.weight]["exp_avg"][:, replace_input_idx] = 0.0
                    self.opt.state[next_layer.weight]["exp_avg_sq"][:, replace_input_idx] = 0.0
                    self.opt.state[next_layer.weight]["step"][:, replace_input_idx] = 0

        self.accumulated_num_features_to_replace[layer_idx] = 0.0

    def get_layer_stats(self) -> Dict[int, LayerReplacementStats2]:
        """Get replacement statistics for all layers."""
        return self.layer_stats
