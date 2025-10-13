"""
Loss landscape change during task shift

Purpose:
- Train across tasks with the task-shift modes used in improving_plasticity_via_advanced_optimizer,
  and compute LLA metrics ONLY at the end of each task (right before the shift).
- Save plots locally (planes, ESD, optional SAM), named with architecture and task number.
- Do NOT log images or checkpoints to wandb; only log numeric metrics (e.g., top-k eigenvalues,
  layer-wise rank/gini summaries, weight norms) into wandb.

Dependencies:
- Reuses dataset/model/learner factories and wrappers from this repo.
- Uses functions from experiments/Hessian_and_landscape_plot_with_plasticity_loss/lla_pipeline.py for spectrum/planes/SAM.

Notes:
- Defaults aim to keep runtime tractable per task (moderate grid size, limited probes).
- You can tweak the end-of-task LLA block via CLI flags.
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import numbers
from pathlib import Path
import hashlib
import uuid
import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Repo root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import OmegaConf

# Factories and utils from the repo
from configs.configurations import ExperimentConfig, NetConfig, NetParams, DataConfig  # type: ignore
from src.models.model_factory import model_factory  # type: ignore
from src.data_loading.dataset_factory import dataset_factory  # type: ignore
from src.data_loading.transform_factory import transform_factory  # type: ignore
from src.data_loading.shifting_dataset import (
    create_stateful_dataset_wrapper,
    create_stateless_dataset_wrapper,
)  # type: ignore
from src.algos.supervised.supervised_factory import create_learner  # type: ignore
# from src.utils.miscellaneous import compute_matrix_rank_summaries  # (legacy; removed by refactor)
# Rank / feature tracking utilities (replace prior layer_rank/gini custom code)
from src.utils.zeroth_order_features import compute_all_rank_measures_list, count_saturated_units_list  # type: ignore
from src.utils.rank_drop_dynamics import compute_rank_dynamics_from_features  # type: ignore
from src.utils.track_weights_norm import track_weight_stats  # type: ignore
from src.utils.task_shift_logging import build_logging_config_dict  # type: ignore

try:  # Feature containers provide richer metadata for intermediate activations
    from src.utils.feature_container import FeatureContainer  # type: ignore
except Exception:  # pragma: no cover - optional dependency for older checkpoints
    FeatureContainer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

# LLA submodule integration (replace previous custom pipeline usage)
# Add submodule src path
LLA_SUBMODULE_SRC = Path(__file__).parent / 'external' / 'loss-landscape-analysis' / 'src'
if LLA_SUBMODULE_SRC.exists() and str(LLA_SUBMODULE_SRC) not in sys.path:
    sys.path.insert(0, str(LLA_SUBMODULE_SRC))

try:  # Import required LLA components
    from src_lla import viz_esd, viz_lla  # type: ignore
    from src_lla.loss_landscapes.metrics.metric import Metric  # type: ignore
    from src_lla.hessian.hessian import hessian_calc  # type: ignore
    from src_lla.loss_landscapes import random_plane  # type: ignore
    from src_lla.loss_landscapes.model_interface.model_parameters import ModelParameters  # type: ignore
    LLA_AVAILABLE = True
except Exception as e:  # fallback flag
    LOGGER.warning(
        "[LLA] Submodule import failed (%s); falling back to legacy pipeline (limited features).",
        e,
    )

    class Metric:  # type: ignore[empty-body]
        """Fallback stub when LLA submodule is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            """No-op initializer to satisfy Metric interface expectations."""

        def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
            """Raise to indicate the Metric can't be used without LLA."""
            raise RuntimeError(
                "Loss Landscape Analysis submodule is unavailable; metrics cannot be computed."
            )

    LLA_AVAILABLE = False
    hessian_calc = None  # type: ignore[assignment]
    random_plane = None  # type: ignore[assignment]
    ModelParameters = None  # type: ignore[assignment]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge src into dst (in place) and return dst."""

    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _build_lla_cfg(base_cfg: ExperimentConfig, eval_batch_size: int = 128) -> Dict[str, Any]:
    """Construct a minimal LLA config dict expected by lla_pipeline functions."""

    device = str(base_cfg.device) if hasattr(base_cfg, 'device') else str(base_cfg.net.device)
    lla_cfg: Dict[str, Any] = {
        'device': device,
        'data': {
            'dataset': str(base_cfg.data.dataset),
            'use_torchvision': bool(base_cfg.data.use_torchvision),
            'data_path': str(base_cfg.data.data_path),
            'num_classes': int(base_cfg.data.num_classes),
        },
        'net': {
            'type': str(base_cfg.net.type),
            'network_class': str(base_cfg.net.network_class),
            'device': device,
            'netparams': OmegaConf.to_container(base_cfg.net.netparams, resolve=True),
        },
        'learner': {
            'loss': 'cross_entropy' if str(base_cfg.learner.loss) == '' else str(base_cfg.learner.loss),
        },
        'lla': {
            'evaluation_data': {
                'eval_batch_size': int(eval_batch_size),
                'fixed_eval_seed': 42,
            },
            'planes': {
                'grid_resolution': 31,
                'span': [-0.5, 0.5],
                'normalization': 'filter',
                'bn': {'eval_mode_only': True, 'exclude_bn_and_bias': False},
                'chunk_size': 128,
                'max_eval_points_cap': 4096,
                'loss_cap_offset': 200.0,
            },
            'spectrum': {
                'power_iters': 20,
                'top_k': 5,
                'hutchinson_probes': 6,
                'esd': {'num_probes': 6, 'lanczos_steps': 30, 'bins': 200},
            },
            'sam': {
                'rho': 0.05,
                'subsample_stride': 2,
            },
        },
    }

    try:
        user_lla = getattr(base_cfg, 'lla', None)
        if user_lla is not None:
            user_lla_dict = OmegaConf.to_container(user_lla, resolve=True)  # type: ignore[arg-type]
            if isinstance(user_lla_dict, dict):
                _deep_update(lla_cfg['lla'], user_lla_dict)
                spec = lla_cfg['lla'].get('spectrum', {})
                if isinstance(spec, dict):
                    esd = spec.get('esd', {}) if isinstance(spec.get('esd', {}), dict) else {}
                    if isinstance(esd, dict):
                        spec['slq_m'] = esd.get('lanczos_steps', spec.get('slq_m', 50))
                        spec['slq_probes'] = esd.get('num_probes', spec.get('slq_probes', 8))
                        spec['slq_grid'] = esd.get('bins', spec.get('slq_grid', 200))
                        lla_cfg['lla']['spectrum'] = spec
    except Exception:
        pass

    return lla_cfg


def _make_criterion(loss_name: str | None = 'cross_entropy') -> nn.Module:
    """Return a criterion instance aligned with the configured learner loss."""
    if loss_name == 'mse':
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


class FixedBatchMetric(Metric):
    """Metric implementation compatible with LLA's expectations for both
    - Hessian construction (called as metric(None, model=..., return_pred=True))
    - Plane evaluation (called as metric(model_wrapper))

    Captures a fixed (x,y) batch for reproducibility & efficiency.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str, criterion: nn.Module):
        """Cache a single evaluation batch on the requested device for reuse."""
        super().__init__()
        self.x = x.detach().to(device)
        self.y = y.detach().to(device)
        self.device = device
        self.criterion = criterion

    def _forward(self, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass on the cached batch and return (loss, logits)."""
        model.eval()
        out = model(self.x)
        if isinstance(self.criterion, nn.MSELoss):
            if self.y.dtype in (torch.long, torch.int64):
                y_oh = torch.nn.functional.one_hot(self.y, num_classes=out.shape[-1]).float()
            else:
                y_oh = self.y.float()
            loss = self.criterion(out, y_oh)
        else:
            loss = self.criterion(out, self.y)
        return loss, out

    # Plane / landscape API call (wrapped model)
    def __call__(self, model_wrapper, *args, **kwargs):  # type: ignore[override]
        """Expose Metric-compatible call hooks for plane and Hessian evaluations."""
        # If called with wrapped model (ModelWrapper) we extract underlying module
        if hasattr(model_wrapper, 'modules') and len(model_wrapper.modules) == 1:  # plane evaluation path
            model = model_wrapper.modules[0]
            loss, _ = self._forward(model)
            return loss.detach().cpu().item()
        # Hessian path: signature metric(None, model=..., return_pred=True)
        model = kwargs.get('model', None)
        return_pred = kwargs.get('return_pred', False)
        if model is None:
            raise ValueError("Metric called without model.")
        loss, out = self._forward(model)
        if return_pred:
            return loss, out
        return loss


def _first_batch(loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the first batch from a loader and move tensors onto the requested device."""
    batch = next(iter(loader))
    x, y = batch
    if isinstance(y, tuple):  # drifting_values case not supported for LLA Hessian; use original labels
        # y is (drifting_values, original_labels); use original_labels for supervised loss
        try:
            _, orig = y
            y = orig
        except Exception:
            y = y[0]
    return x.to(device), y.to(device)


def _coerce_feature_list(features: Any) -> List[torch.Tensor]:
    """Normalize assorted feature containers into a list of detached tensors."""

    tensors: List[torch.Tensor] = []
    if features is None:
        return tensors

    if FeatureContainer is not None and isinstance(features, FeatureContainer):
        try:
            iter_features = features.as_list()
        except Exception:
            iter_features = list(features)
    elif isinstance(features, dict):
        iter_features = list(features.values())
    elif isinstance(features, (list, tuple)):
        iter_features = list(features)
    elif hasattr(features, '__iter__') and not isinstance(features, (str, bytes)):
        try:
            iter_features = list(features)
        except TypeError:
            return tensors
    else:
        return tensors

    for feat in iter_features:
        if isinstance(feat, torch.Tensor):
            tensors.append(feat.detach())
    return tensors


def _extract_features_with_hooks(
    model: nn.Module,
    inputs_cpu: torch.Tensor,
    device: str,
    module_types: tuple[type, ...] = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Linear,
    ),
) -> tuple[List[torch.Tensor], List[str]]:
    """Capture intermediate activations via forward hooks as a fallback.

    Returns a tuple of (feature_tensors, layer_names). Each tensor is detached
    and moved to CPU. Layer names are made unique for repeated modules.
    """

    handles: List[Any] = []  # type: ignore[var-annotated]
    captured: List[torch.Tensor] = []
    layer_names: List[str] = []
    call_counts: dict[str, int] = defaultdict(int)

    def _register_hook(layer_name: str):
        def _hook(module, _inp, output):  # type: ignore[no-untyped-def]
            tensors: List[torch.Tensor] = []
            if isinstance(output, torch.Tensor):
                tensors = [output]
            elif isinstance(output, (list, tuple)):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
            if not tensors:
                return

            call_counts[layer_name] += 1
            call_index = call_counts[layer_name] - 1
            for idx, tensor in enumerate(tensors):
                suffix = f"{call_index}" if len(tensors) == 1 else f"{call_index}_{idx}"
                layer_names.append(f"{layer_name}#{suffix}")
                captured.append(tensor.detach().cpu())

        return _hook

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, module_types):
            handles.append(module.register_forward_hook(_register_hook(name)))

    prev_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(inputs_cpu.to(device))
    finally:
        for h in handles:
            h.remove()
        if prev_training:
            model.train()

    return captured, layer_names


def _save_plane_artifacts(
    plane_arr: np.ndarray,
    plane_dir: Path,
    distance: float,
    prefix: str = 'plane_hessian',
) -> Dict[str, str]:
    """Persist plane visualizations (heatmap/3D/contour) and return artifact paths."""
    artifacts: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"[LLA][plane] matplotlib unavailable: {e}")
        return artifacts

    coords = np.linspace(-distance, distance, plane_arr.shape[0])
    X, Y = np.meshgrid(coords, coords, indexing='xy')

    try:
        heatmap_png = plane_dir / f'{prefix}_heatmap.png'
        fig_h, ax_h = plt.subplots(figsize=(5.2, 4.4))
        im = ax_h.imshow(
            plane_arr,
            origin='lower',
            cmap='viridis',
            extent=[-distance, distance, -distance, distance],
            aspect='auto',
        )
        ax_h.set_xlabel('Direction 1 (center=0)')
        ax_h.set_ylabel('Direction 2 (center=0)')
        ax_h.set_title('Hessian-aligned plane (heatmap)')
        cbar = fig_h.colorbar(im, ax=ax_h)
        cbar.set_label('Loss')
        fig_h.tight_layout()
        fig_h.savefig(str(heatmap_png))
        artifacts['heatmap_png'] = str(heatmap_png)
        plt.close(fig_h)
    except Exception as e:
        print(f"[LLA][plane] heatmap plot failed: {e}")

    try:
        surface_png = plane_dir / f'{prefix}_surface3d.png'
        fig3d = plt.figure(figsize=(6, 4.8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        surf = ax3d.plot_surface(X, Y, plane_arr, cmap=cm.viridis, linewidth=0, antialiased=True)
        ax3d.set_xlabel('Dir 1')
        ax3d.set_ylabel('Dir 2')
        ax3d.set_zlabel('Loss')
        ax3d.set_title('Loss surface (Hessian plane)')
        fig3d.colorbar(surf, shrink=0.6, aspect=12, pad=0.08).set_label('Loss')
        fig3d.tight_layout()
        fig3d.savefig(str(surface_png))
        artifacts['surface3d_png'] = str(surface_png)
        plt.close(fig3d)
    except Exception as e:
        print(f"[LLA][plane] surface plot failed: {e}")

    try:
        contour_png = plane_dir / f'{prefix}_contour.png'
        fig_c, ax_c = plt.subplots(figsize=(5.2, 4.4))
        CS = ax_c.contour(X, Y, plane_arr, levels=20, cmap='viridis')
        ax_c.clabel(CS, inline=True, fontsize=7)
        ax_c.set_xlabel('Direction 1 (center=0)')
        ax_c.set_ylabel('Direction 2 (center=0)')
        ax_c.set_title('Loss contours (Hessian plane)')
        fig_c.tight_layout()
        fig_c.savefig(str(contour_png))
        artifacts['contour_png'] = str(contour_png)
        plt.close(fig_c)
    except Exception as e:
        print(f"[LLA][plane] contour plot failed: {e}")

    return artifacts


def _extract_esd_arrays(esd_payload: Dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract eigenvalue and weight arrays from a viz_esd payload, enforcing uniform shapes."""
    eigenvalues_runs = esd_payload.get('eigenvalues_runs')
    weights_runs = esd_payload.get('weights_runs')
    if not isinstance(eigenvalues_runs, (list, tuple)) or not isinstance(weights_runs, (list, tuple)):
        return None, None
    filtered_eigs: list[np.ndarray] = []
    filtered_weights: list[np.ndarray] = []
    for eigs, weights in zip(eigenvalues_runs, weights_runs):
        try:
            eig_arr = np.asarray(eigs, dtype=np.float64)
            weight_arr = np.asarray(weights, dtype=np.float64)
        except Exception:
            continue
        if eig_arr.size == 0 or weight_arr.size == 0:
            continue
        length = min(eig_arr.shape[0], weight_arr.shape[0])
        if length == 0:
            continue
        filtered_eigs.append(eig_arr[:length])
        filtered_weights.append(weight_arr[:length])
    if not filtered_eigs:
        return None, None
    try:
        eig_matrix = np.vstack(filtered_eigs)
        weight_matrix = np.vstack(filtered_weights)
        return eig_matrix, weight_matrix
    except Exception:
        return None, None


def _compute_esd_signed_stats(esd_payload: Dict[str, Any], eps: float = 1e-8) -> Dict[str, float]:
    """Compute summary statistics about signed eigenvalue mass using the viz_esd payload."""
    stats: Dict[str, float] = {}
    eig_matrix, weight_matrix = _extract_esd_arrays(esd_payload)
    if eig_matrix is None or weight_matrix is None:
        return stats
    if eig_matrix.shape != weight_matrix.shape:
        return stats

    # Boolean masks partition the spectrum into negative/positive/near-zero regions and let us
    # reuse the same broadcasting logic for both mass and count computations.
    neg_mask = eig_matrix < -eps
    pos_mask = eig_matrix > eps
    zero_mask = ~(neg_mask | pos_mask)

    neg_mass = float(np.mean(np.sum(weight_matrix * neg_mask, axis=1)))
    pos_mass = float(np.mean(np.sum(weight_matrix * pos_mask, axis=1)))
    zero_mass = float(np.mean(np.sum(weight_matrix * zero_mask, axis=1)))
    total_mass = neg_mass + pos_mass + zero_mass
    if total_mass <= 0:
        total_mass = neg_mass + pos_mass + zero_mass + eps

    stats['negative_weight_mass'] = neg_mass
    stats['positive_weight_mass'] = pos_mass
    stats['zero_weight_mass'] = zero_mass
    stats['negative_fraction'] = neg_mass / max(eps, (neg_mass + pos_mass)) if (neg_mass + pos_mass) > eps else 0.0
    stats['signed_weight_balance'] = pos_mass - neg_mass
    stats['negative_mass_ratio_total'] = neg_mass / total_mass if total_mass > eps else 0.0

    stats['negative_count'] = float(np.mean(np.sum(neg_mask, axis=1)))
    stats['positive_count'] = float(np.mean(np.sum(pos_mask, axis=1)))
    stats['zero_count'] = float(np.mean(np.sum(zero_mask, axis=1)))

    min_eig = float(np.min(eig_matrix)) if eig_matrix.size else 0.0
    max_eig = float(np.max(eig_matrix)) if eig_matrix.size else 0.0
    stats['min_eigenvalue'] = min_eig
    stats['max_eigenvalue'] = max_eig

    density = esd_payload.get('density')
    segments = esd_payload.get('segments')
    try:
        density_arr = np.asarray(density, dtype=np.float64)
        segments_arr = np.asarray(segments, dtype=np.float64)
        if density_arr.size and segments_arr.size and density_arr.shape == segments_arr.shape:
            neg_region = segments_arr < -eps
            pos_region = segments_arr > eps
            if np.any(neg_region):
                stats['negative_density_integral'] = float(np.trapz(density_arr[neg_region], segments_arr[neg_region]))
                stats['negative_density_peak'] = float(np.max(density_arr[neg_region]))
            if np.any(pos_region):
                stats['positive_density_integral'] = float(np.trapz(density_arr[pos_region], segments_arr[pos_region]))
                stats['positive_density_peak'] = float(np.max(density_arr[pos_region]))
    except Exception:
        pass

    return stats


def _save_esd_payload_npz(esd_payload: Dict[str, Any], out_path: Path) -> None:
    """Persist raw viz_esd payload components in an NPZ for downstream post-processing."""
    try:
        # eigenvalues_runs / weights_runs are ragged lists; store them with dtype=object so we don't
        # lose per-run lengths when round-tripping the NPZ archive.
        eigenvalues_runs = np.array(esd_payload.get('eigenvalues_runs', []), dtype=object)
        weights_runs = np.array(esd_payload.get('weights_runs', []), dtype=object)
        density = np.asarray(esd_payload.get('density', []), dtype=np.float64)
        segments = np.asarray(esd_payload.get('segments', []), dtype=np.float64)
        np.savez(
            out_path,
            eigenvalues_runs=eigenvalues_runs,
            weights_runs=weights_runs,
            density=density,
            segments=segments,
        )
    except Exception as e:
        print(f"[LLA][spectrum] failed to save esd payload npz: {e}")


def _save_density_plots(esd_payload: Dict[str, Any], output_dir: Path, exp_name: str = 'spectrum') -> Dict[str, str]:
    """Render log-scale density plots (raw + smoothed) from viz_esd data across the full spectrum."""
    artifacts: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[LLA][spectrum] matplotlib unavailable for density plots: {e}")
        return artifacts

    eig_matrix, weight_matrix = _extract_esd_arrays(esd_payload)
    try:
        density = np.asarray(esd_payload.get('density', []), dtype=np.float64)
        segments = np.asarray(esd_payload.get('segments', []), dtype=np.float64)
    except Exception as e:
        print(f"[LLA][spectrum] invalid density payload: {e}")
        density = np.array([])
        segments = np.array([])

    # Determine plotting range from whichever data we have available.
    x_min = float(np.min(segments)) if segments.size else None
    x_max = float(np.max(segments)) if segments.size else None
    if eig_matrix is not None and eig_matrix.size:
        eig_min = float(np.min(eig_matrix))
        eig_max = float(np.max(eig_matrix))
        x_min = eig_min if x_min is None else min(x_min, eig_min)
        x_max = eig_max if x_max is None else max(x_max, eig_max)

    if x_min is not None and x_max is not None:
        bound = max(abs(x_min), abs(x_max))
        if not np.isfinite(bound) or bound == 0.0:
            bound = 1.0
        pad = 0.02 * bound
        x_min = -bound - pad
        x_max = bound + pad

    # Build a raw (unsmoothed) weighted histogram averaged across runs, if eigenvalues are available.
    hist_centers: np.ndarray | None = None
    hist_density: np.ndarray | None = None
    if eig_matrix is not None and eig_matrix.size and weight_matrix is not None and weight_matrix.size:
        num_runs, _ = eig_matrix.shape
        # Match the number of bins to the smoothed density resolution when possible.
        num_bins = int(segments.size) if segments.size else max(512, int(np.sqrt(eig_matrix.shape[1]) * 8))
        if num_bins < 10:
            num_bins = 10
        if x_min is None or x_max is None:
            local_min = float(np.min(eig_matrix))
            local_max = float(np.max(eig_matrix))
            bound = max(abs(local_min), abs(local_max))
            if not np.isfinite(bound) or bound == 0.0:
                bound = 1.0
            pad = 0.02 * bound
            x_min = -bound - pad
            x_max = bound + pad
        bin_edges = np.linspace(x_min, x_max, num_bins + 1)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        aggregated = np.zeros_like(centers, dtype=np.float64)
        for run_idx in range(num_runs):
            try:
                counts, _ = np.histogram(
                    eig_matrix[run_idx],
                    bins=bin_edges,
                    weights=weight_matrix[run_idx],
                )
            except Exception:
                continue
            aggregated += counts
        if num_runs > 0:
            aggregated /= float(num_runs)
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        total_mass = float(np.sum(aggregated) * bin_width)
        if total_mass > 0:
            aggregated /= total_mass
            hist_centers = centers
            hist_density = aggregated

    if hist_centers is not None and hist_density is not None:
        try:
            raw_path = output_dir / f'{exp_name}_esd_raw.png'
            fig_raw, ax_raw = plt.subplots(figsize=(6.4, 4.8))
            ax_raw.semilogy(hist_centers, hist_density + 1e-12, color='#253494', linewidth=1.3, label='Raw (weighted histogram)')
            ax_raw.axvline(0.0, color='#d7301f', linestyle='--', linewidth=1.0, label='λ = 0')
            ax_raw.set_xlabel('Eigenvalue λ')
            ax_raw.set_ylabel('Density (Log Scale)')
            ax_raw.set_title('Hessian ESD (raw)')
            if x_min is not None and x_max is not None:
                ax_raw.set_xlim(x_min, x_max)
            ax_raw.legend(loc='upper right')
            ax_raw.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
            fig_raw.tight_layout()
            fig_raw.savefig(str(raw_path))
            artifacts['esd_raw_png'] = str(raw_path)
            plt.close(fig_raw)
        except Exception as e:
            print(f"[LLA][spectrum] raw density plot failed: {e}")

    if density.size and segments.size and density.shape == segments.shape:
        try:
            smooth_path = output_dir / f'{exp_name}_esd_smoothed.png'
            fig_smooth, ax_smooth = plt.subplots(figsize=(6.4, 4.8))
            ax_smooth.semilogy(segments, density + 1e-12, color='#2b8cbe', linewidth=1.5, label='Smoothed (Gaussian kernel)')
            ax_smooth.axvline(0.0, color='#d7301f', linestyle='--', linewidth=1.0, label='λ = 0')
            ax_smooth.set_xlabel('Eigenvalue λ')
            ax_smooth.set_ylabel('Density (Log Scale)')
            ax_smooth.set_title('Hessian ESD (smoothed)')
            if x_min is not None and x_max is not None:
                ax_smooth.set_xlim(x_min, x_max)
            ax_smooth.legend(loc='upper right')
            ax_smooth.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
            fig_smooth.tight_layout()
            fig_smooth.savefig(str(smooth_path))
            artifacts['esd_smoothed_png'] = str(smooth_path)
            plt.close(fig_smooth)
        except Exception as e:
            print(f"[LLA][spectrum] smoothed density plot failed: {e}")

    return artifacts


def _compute_signed_hessian_axes(
    model: nn.Module,
    metric: FixedBatchMetric,
    eigen_iter: int,
    candidate_count: int,
) -> tuple[ModelParameters | None, ModelParameters | None, Dict[str, Any]]:
    """Derive eigenvector directions for the largest positive and most negative Hessian eigenvalues."""

    info: Dict[str, Any] = {'candidate_count': int(candidate_count)}
    if hessian_calc is None or ModelParameters is None:
        info['status'] = 'hessian_unavailable'
        return None, None, info

    candidate_count = max(2, int(candidate_count))
    eigen_iter = max(1, int(eigen_iter))

    try:
        hessian = hessian_calc(model, metric)
        eigenvalues, eigenvectors = hessian.eigs_calc(top_n=candidate_count, n_iter=eigen_iter)
    except Exception as exc:
        info['status'] = 'eigs_calc_failed'
        info['error'] = str(exc)
        return None, None, info
    finally:
        try:
            hessian.reset()
        except Exception:
            pass

    # Normalize eigenvalue list and keep track of their indices for lookup.
    processed: list[tuple[int, float]] = []
    for idx, val in enumerate(eigenvalues or []):
        try:
            if val is None:
                continue
            processed.append((idx, float(val)))
        except Exception:
            continue

    info['candidate_eigenvalues'] = [v for _, v in processed]
    if not processed:
        info['status'] = 'no_eigenvalues'
        return None, None, info

    pos_idx, pos_val = None, float('-inf')
    neg_idx, neg_val = None, float('inf')
    for idx, val in processed:
        if val > pos_val:
            pos_idx, pos_val = idx, val
        if val < neg_val:
            neg_idx, neg_val = idx, val

    if pos_idx is None:
        info['status'] = 'missing_positive'
        info['positive_eigenvalue'] = None
        info['negative_eigenvalue'] = neg_val if neg_idx is not None else None
        return None, None, info
    if neg_idx is None:
        info['status'] = 'missing_negative'
        info['positive_eigenvalue'] = pos_val
        info['negative_eigenvalue'] = None
        return None, None, info

    info['positive_eigenvalue'] = pos_val
    info['negative_eigenvalue'] = neg_val
    info['status'] = 'ok'

    pos_axis = ModelParameters(eigenvectors[pos_idx])
    neg_axis = ModelParameters(eigenvectors[neg_idx])

    return pos_axis, neg_axis, info


def _compute_planes_and_spectrum_with_lla(
    model: nn.Module,
    eval_loader: DataLoader,
    task_dir: Path,
    lla_cfg: Dict[str, Any],
    do_planes: bool,
    do_spectrum: bool,
) -> Dict[str, Any]:
    """Run high-level LLA helpers (viz_lla, viz_esd) to produce plane and spectrum artifacts."""

    device = lla_cfg.get('device', 'cuda:0')
    loss_name = lla_cfg.get('learner', {}).get('loss', 'cross_entropy')
    criterion = _make_criterion(loss_name)
    x, y = _first_batch(eval_loader, device)
    metric = FixedBatchMetric(x, y, device, criterion)

    out: Dict[str, Any] = {}
    plane_dir = task_dir / 'planes'
    plane_dir.mkdir(exist_ok=True)
    spectrum_dir = task_dir

    planes_cfg = lla_cfg.get('lla', {}).get('planes', {})
    grid_res = int(planes_cfg.get('grid_resolution', 41))
    if 'distance' in planes_cfg:
        distance = float(planes_cfg.get('distance'))
    else:
        span_cfg = planes_cfg.get('span', [-0.5, 0.5])
        if isinstance(span_cfg, (list, tuple)) and len(span_cfg) == 2:
            try:
                distance = float(max(abs(float(span_cfg[0])), abs(float(span_cfg[1]))))
            except Exception:
                distance = 1.0
        else:
            distance = 1.0
    normalization = planes_cfg.get('normalization', 'filter')
    plane_order_raw = planes_cfg.get('order', 2)
    plane_order = int(plane_order_raw) if plane_order_raw is not None else 2
    plane_mode_raw = planes_cfg.get('mode', 'add')
    plane_mode = str(plane_mode_raw) if plane_mode_raw is not None else 'add'
    plane_raa = planes_cfg.get('raa', None)
    plane_b_sqrt_raw = planes_cfg.get('b_sqrt', True)
    if isinstance(plane_b_sqrt_raw, str):
        plane_b_sqrt = plane_b_sqrt_raw.strip().lower() not in {'0', 'false', 'no', 'off'}
    else:
        plane_b_sqrt = bool(plane_b_sqrt_raw)

    spectrum_cfg = lla_cfg.get('lla', {}).get('spectrum', {})
    top_k = int(spectrum_cfg.get('top_k', 5))
    power_iters = int(spectrum_cfg.get('power_iters', 100))
    trace_probes = int(spectrum_cfg.get('hutchinson_probes', 32))
    esd_cfg = spectrum_cfg.get('esd', {}) if isinstance(spectrum_cfg.get('esd', {}), dict) else {}
    lanczos_steps = int(esd_cfg.get('lanczos_steps', 100))
    esd_probes = int(esd_cfg.get('num_probes', 1))
    max_v = int(esd_cfg.get('max_v', max(10, esd_probes)))
    n_kh = float(esd_cfg.get('n_kh', 0.5))
    signed_candidate_count = int(planes_cfg.get('signed_axes_top_n', max(4, top_k * 2)))

    t_all0 = time.time()

    # -------------------- Loss landscape via viz_lla --------------------
    if do_planes:
        try:
            plane_array = viz_lla(
                model,
                metric,
                device=device,
                dist=distance,
                steps=grid_res,
                axes='hessian',
                normalization=normalization,
                cur_name='plane',
                viz_dir=str(plane_dir),
                res_dir=str(plane_dir),
                to_save=False,
                to_viz=False,
                return_loss=True,
            )
            plane_info: Dict[str, Any] = {
                'grid_resolution': grid_res,
                'distance': distance,
                'normalization': normalization,
                'axes_source': 'largest_magnitude_eigenvectors',
                'algorithm': 'viz_lla',
            }
            if plane_array is not None:
                plane_arr_np = np.asarray(plane_array, dtype=np.float64)
                if plane_arr_np.size > 0:
                    loss_min = float(np.nanmin(plane_arr_np))
                    loss_cap_offset = planes_cfg.get('loss_cap_offset', 200.0)
                    if isinstance(loss_cap_offset, (int, float)) and float(loss_cap_offset) > 0:
                        cap_threshold = loss_min + float(loss_cap_offset)
                        plane_arr_np = np.minimum(plane_arr_np, cap_threshold)
                        plane_info['loss_cap_threshold'] = cap_threshold
                    plane_info['loss_min'] = loss_min
                plane_prefix = 'plane_hessian'
                plane_path = plane_dir / f'{plane_prefix}.npy'
                np.save(plane_path, plane_arr_np)
                plane_info['npy'] = str(plane_path)
                plane_info.update(_save_plane_artifacts(plane_arr_np, plane_dir, distance, prefix=plane_prefix))
            out['plane'] = plane_info
        except Exception as e:
            print(f"[LLA] viz_lla failed: {e}")

    # -------------------- Signed extreme-eigenvector plane --------------------
    if do_planes and random_plane is not None:
        try:
            pos_axis, neg_axis, signed_axes_info = _compute_signed_hessian_axes(
                model,
                metric,
                eigen_iter=power_iters,
                candidate_count=signed_candidate_count,
            )
            if pos_axis is not None and neg_axis is not None:
                signed_plane_array = random_plane(
                    model,
                    metric,
                    distance=distance,
                    steps=grid_res,
                    normalization=normalization,
                    order=plane_order,
                    a1=pos_axis,
                    a2=neg_axis,
                    mode=plane_mode,
                    raa=plane_raa,
                    b_sqrt=plane_b_sqrt,
                )
                signed_plane_info: Dict[str, Any] = {
                    'grid_resolution': grid_res,
                    'distance': distance,
                    'normalization': normalization,
                    'axes_source': 'extreme_signed_eigenvectors',
                    'algorithm': 'random_plane',
                }
                signed_plane_info.update(signed_axes_info)
                signed_arr_np = np.asarray(signed_plane_array, dtype=np.float64)
                if signed_arr_np.size > 0:
                    loss_min_signed = float(np.nanmin(signed_arr_np))
                    loss_cap_offset = planes_cfg.get('loss_cap_offset', 200.0)
                    if isinstance(loss_cap_offset, (int, float)) and float(loss_cap_offset) > 0:
                        cap_threshold_signed = loss_min_signed + float(loss_cap_offset)
                        signed_arr_np = np.minimum(signed_arr_np, cap_threshold_signed)
                        signed_plane_info['loss_cap_threshold'] = cap_threshold_signed
                    signed_plane_info['loss_min'] = loss_min_signed
                signed_prefix = 'plane_hessian_signed'
                signed_plane_path = plane_dir / f'{signed_prefix}.npy'
                np.save(signed_plane_path, signed_arr_np)
                signed_plane_info['npy'] = str(signed_plane_path)
                signed_plane_info.update(
                    _save_plane_artifacts(signed_arr_np, plane_dir, distance, prefix=signed_prefix)
                )
                out['plane_signed'] = signed_plane_info
            else:
                status = signed_axes_info.get('status', 'missing_signed_axes')
                print(f"[LLA] Signed eigenvector plane skipped ({status}).")
                out['plane_signed_diagnostics'] = signed_axes_info
        except Exception as e:
            print(f"[LLA] signed-eigenvector plane failed: {e}")

    # -------------------- Spectrum via viz_esd --------------------
    if do_spectrum:
        try:
            viz_output = viz_esd(
                model,
                metric,
                eigs=True,
                top_n=max(2, top_k),
                eigs_n_iter=power_iters,
                trace=True,
                trace_n_iter=trace_probes,
                esd=True,
                esd_n_iter=lanczos_steps,
                n_v=max(1, esd_probes),
                max_v=max_v,
                to_save=True,
                to_viz=False,
                exp_name='spectrum',
                viz_dir=str(spectrum_dir),
                res_dir=str(spectrum_dir),
                calc_crit=True,
                n_kh=n_kh,
                return_data=True,
            )
            if isinstance(viz_output, tuple) and len(viz_output) == 2:
                viz_results, esd_payload = viz_output
            else:
                viz_results = viz_output
                esd_payload = None
            eigvals, eigvecs, trace_est, re_val, khn_val = viz_results

            # Eigenvector pickles produced by viz_esd are very large; remove them immediately to conserve disk space.
            eigvec_pickle_path = spectrum_dir / 'eigenvectors_spectrum.pickle'
            if eigvec_pickle_path.exists():
                try:
                    eigvec_pickle_path.unlink()
                except Exception as cleanup_err:
                    print(f"[LLA] Failed to delete eigenvectors pickle: {cleanup_err}")

            eigvals_list = [float(v) for v in eigvals] if eigvals is not None else []
            trace_float = float(trace_est) if trace_est is not None else None
            partial_topk_sum = float(np.sum(eigvals_list[:top_k])) if eigvals_list else None
            diagnostics: Dict[str, Any] = {}
            if partial_topk_sum is not None:
                diagnostics['partial_topk_sum'] = partial_topk_sum
            if partial_topk_sum is not None and trace_float is not None:
                diagnostics['trace_fraction_topk'] = float(partial_topk_sum / (trace_float + 1e-12))
            if re_val is not None:
                diagnostics['hesd_re'] = float(re_val)
            if khn_val is not None:
                diagnostics['hesd_khn'] = float(khn_val)

            esd_artifacts: Dict[str, str] = {}
            esd_stats: Dict[str, float] = {}
            if isinstance(esd_payload, dict):
                try:
                    esd_npz_path = spectrum_dir / 'spectrum_esd_raw.npz'
                    _save_esd_payload_npz(esd_payload, esd_npz_path)
                    esd_artifacts['esd_npz'] = str(esd_npz_path)
                except Exception as e:
                    print(f"[LLA][spectrum] failed to persist esd payload: {e}")
                esd_stats = _compute_esd_signed_stats(esd_payload)
                diagnostics.update(esd_stats)
                density_artifacts = _save_density_plots(esd_payload, spectrum_dir, exp_name='spectrum')
                esd_artifacts.update(density_artifacts)

            spec_summary: Dict[str, Any] = {
                'top_k': eigvals_list[:top_k],
                'hutchinson_trace': trace_float,
                'dim': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'k_used': top_k,
                'probes': trace_probes,
                'n_v': max(1, esd_probes),
                'algorithm': 'viz_esd',
                'diagnostics': diagnostics,
            }
            if esd_stats:
                spec_summary['signed_stats'] = esd_stats
            if esd_artifacts:
                spec_summary['esd_artifacts'] = esd_artifacts

            spectrum_json = spectrum_dir / 'spectrum.json'
            spectrum_json.write_text(json.dumps(spec_summary, indent=2))

            spectrum_paths: Dict[str, str] = {}
            hesd_png = spectrum_dir / 'spectrum_esd.png'
            if hesd_png.exists():
                spectrum_paths['esd_png'] = str(hesd_png)
            for key, path_str in esd_artifacts.items():
                spectrum_paths[key] = path_str
            eigenvalues_log = spectrum_dir / 'eigenvalues_spectrum.log'
            if eigenvalues_log.exists():
                spectrum_paths['eigenvalues_log'] = str(eigenvalues_log)
            trace_log = spectrum_dir / 'trace_spectrum.log'
            if trace_log.exists():
                spectrum_paths['trace_log'] = str(trace_log)
            criteria_log = spectrum_dir / 'hessian_criteria_spectrum.log'
            if criteria_log.exists():
                spectrum_paths['criteria_log'] = str(criteria_log)

            out['spectrum'] = spec_summary
            out['spectrum_json_path'] = str(spectrum_json)
            if spectrum_paths:
                out['spectrum_paths'] = spectrum_paths
        except Exception as e:
            print(f"[LLA] viz_esd failed: {e}")

    out['total_runtime_sec'] = time.time() - t_all0
    return out


def _rename_with_arch_task(path: Path, arch: str, task_idx: int, phase: str | None = None) -> Path:
    """Rename a file to include architecture, task index, and phase labels for traceability."""
    if not path.exists():
        return path
    stem = path.stem
    new_name = f"{stem}_{arch}_task{task_idx}"
    if phase:
        new_name += f"_{phase}"
    new_name += path.suffix
    new_path = path.with_name(new_name)
    try:
        path.rename(new_path)
        return new_path
    except Exception:
        return path


def _snapshot_eval_dataset(train_set):
    """Best-effort snapshot/freeze of the current task's dataset for evaluation.
    This aims to avoid distribution drift between end-of-task training and LLA evaluation.
    Tries a deepcopy, then probes common freeze/eval methods if they exist.
    Falls back silently if unsupported.
    """
    ds = train_set
    try:
        import copy
        ds = copy.deepcopy(train_set)
    except Exception:
        ds = train_set
    # Try common methods/flags to freeze drift
    candidates: list[tuple[str, tuple]] = [
        ('freeze', ()),
        ('set_eval_mode', (True,)),
        ('disable_drift', (True,)),
        ('set_drift_enabled', (False,)),
        ('set_training', (False,)),
        ('eval', ()),
    ]
    for name, args in candidates:
        try:
            fn = getattr(ds, name, None)
            if callable(fn):
                fn(*args)
        except Exception:
            pass
    return ds


class _SingleBatchDataset(torch.utils.data.Dataset):
    """Dataset wrapper for exactly one captured training batch.
    Supports classification labels or (drifting_values, original_labels) tuples.
    """
    def __init__(self, x: torch.Tensor, y: Any):
        """Store tensors (or tuple targets) that represent a fixed evaluation batch."""
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """Return the number of examples in the captured batch."""
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        """Return the indexed example, preserving tuple targets if present."""
        if isinstance(self.y, tuple) and len(self.y) == 2:
            dv, ol = self.y
            return self.x[idx], (dv[idx], ol[idx])
        return self.x[idx], self.y[idx]


def _make_unique_run_root(base_dir: Path, cfg: Any) -> tuple[Path, str]:
    """Create a unique per-run directory under base_dir and return (path, run_id).

    run_id format: YYYYmmdd-HHMMSS-<cfgsha8>-<rand6>
    - cfgsha8: short SHA1 of the resolved YAML for traceability across runs
    - rand6: guards against collisions when launching multiple runs within the same second
    """
    # Hash the full resolved config for reference (not for uniqueness)
    try:
        cfg_yaml = OmegaConf.to_yaml(cfg)
    except Exception:
        cfg_yaml = str(cfg)
    cfg_sha = hashlib.sha1(cfg_yaml.encode("utf-8")).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Try a few times with a fresh random suffix if a rare collision occurs
    for _ in range(5):
        rand6 = uuid.uuid4().hex[:6]
        run_id = f"{ts}-{cfg_sha}-{rand6}"
        run_root = base_dir / f"run_{run_id}"
        if not run_root.exists():
            run_root.mkdir(parents=True, exist_ok=False)
            # Save minimal metadata for traceability
            try:
                meta = {
                    "run_id": run_id,
                    "timestamp": ts,
                    "cfg_sha1_8": cfg_sha,
                    "arch": str(getattr(getattr(cfg, 'net', object()), 'type', 'unknown')),
                }
                (run_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
                (run_root / "config_snapshot.yaml").write_text(cfg_yaml)
            except Exception:
                pass
            return run_root, run_id
    # Fallback: include full uuid to avoid collision
    run_id = f"{ts}-{cfg_sha}-{uuid.uuid4().hex}"
    run_root = base_dir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root, run_id


def _run_lla_evaluation(
    model: nn.Module,
    eval_source,
    base_cfg: ExperimentConfig,
    out_root: Path,
    task_idx: int,
    arch: str,
    eval_phase: str,
    do_planes: bool = True,
    do_sam: bool = False,
    do_spectrum: bool = True,
) -> Dict[str, Any]:
    """Run LLA evaluation (planes/spectrum/SAM optional) for a specific task/phase."""

    device = str(base_cfg.net.device)
    model.eval()

    # Build eval loader (single-batch)
    eval_ds = _snapshot_eval_dataset(eval_source)
    eval_bs = min(128, len(eval_ds)) if hasattr(eval_ds, '__len__') else 128
    eval_loader = DataLoader(eval_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=True)

    # LLA cfg & output directory
    lla_cfg = _build_lla_cfg(base_cfg, eval_batch_size=eval_bs)
    phase_slug = eval_phase.strip().lower().replace(" ", "_") or "phase"
    task_root = out_root / f"task_{task_idx:04d}_{arch}"
    phase_dir = task_root / phase_slug
    task_dir = phase_dir  # Back-compat for downstream logic expecting task_dir
    phase_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        'phase': phase_slug,
        'output_dir': str(phase_dir),
    }
    if not LLA_AVAILABLE:
        print("[LLA] Submodule not available; skipping planes/spectrum (set up submodule to enable).")
        return result

    try:
        res = _compute_planes_and_spectrum_with_lla(
            model=model,
            eval_loader=eval_loader,
            task_dir=task_dir,
            lla_cfg=lla_cfg,
            do_planes=do_planes,
            do_spectrum=do_spectrum,
        )
        result.update(res)
        # Rename outputs to include arch & task for consistency
        renamed_spectrum: Dict[str, str] = {}
        spectrum_files = [
            'spectrum_esd.png',
            'spectrum_hist.png',
            'spectrum_esd_raw.png',
            'spectrum_esd_smoothed.png',
            'spectrum_esd_raw.npz',
            'spectrum.json',
            'eigenvalues_spectrum.log',
            'trace_spectrum.log',
            'hessian_criteria_spectrum.log',
        ]
        for fname in spectrum_files:
            p = task_dir / fname
            new_path = _rename_with_arch_task(p, arch, task_idx, phase_slug)
            if new_path.exists():
                renamed_spectrum[fname] = str(new_path)
        if renamed_spectrum:
            result.setdefault('spectrum_paths', {}).update(renamed_spectrum)
            if 'spectrum.json' in renamed_spectrum:
                result['spectrum_json_path'] = renamed_spectrum['spectrum.json']
        for plane_key in ['plane', 'plane_signed']:
            plane_entry = result.get(plane_key)
            if not isinstance(plane_entry, dict):
                continue
            for key in ['heatmap_png', 'surface3d_png', 'contour_png', 'npy']:
                if key in plane_entry:
                    try:
                        renamed = _rename_with_arch_task(Path(plane_entry[key]), arch, task_idx, phase_slug)
                        if renamed.exists():
                            plane_entry[key] = str(renamed)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[LLA] LLA evaluation failed: {e}")

    if do_sam:
        print("[LLA][SAM] SAM surface not yet refactored to LLA; skipping (previous implementation removed).")

    return result


def _collect_spectrum_metrics(spec_json_path: str | Path | None, do_spectrum: bool, prefix: str) -> Dict[str, float]:
    """Load spectrum summary JSON and flatten key diagnostics into scalar metrics."""
    metrics: Dict[str, float] = {}
    if not do_spectrum or spec_json_path is None:
        return metrics
    try:
        spec_path = Path(spec_json_path)
    except TypeError:
        return metrics
    if not spec_path.exists():
        return metrics
    try:
        spec_data = json.loads(spec_path.read_text())
    except Exception:
        return metrics

    try:
        topk_iter = list(spec_data.get('top_k', []))
    except Exception:
        topk_iter = []
    for idx, value in enumerate(topk_iter):
        try:
            metrics[f'{prefix}/hessian_top_eig_{idx+1}'] = float(value)
        except Exception:
            continue

    trace_estimate = spec_data.get('hutchinson_trace', None)
    if trace_estimate is not None:
        try:
            metrics[f'{prefix}/hessian_trace_estimate'] = float(trace_estimate)
        except Exception:
            pass

    diagnostics_obj = spec_data.get('diagnostics', {})
    if isinstance(diagnostics_obj, dict):
        diagnostics = cast(Dict[str, Any], diagnostics_obj)
        # Diagnostics includes a mix of scalar Hessian criteria and our enriched signed-mass stats.
        trace_fraction = diagnostics.get('trace_fraction_topk', None)
        if trace_fraction is not None:
            try:
                metrics[f'{prefix}/hessian_trace_fraction_topk'] = float(trace_fraction)
            except Exception:
                pass
        partial_topk_sum = diagnostics.get('partial_topk_sum', None)
        if partial_topk_sum is not None:
            try:
                metrics[f'{prefix}/hessian_partial_topk_sum'] = float(partial_topk_sum)
            except Exception:
                pass
        hesd_re = diagnostics.get('hesd_re', None)
        if hesd_re is not None:
            try:
                metrics[f'{prefix}/hesd_re'] = float(hesd_re)
            except Exception:
                pass
        hesd_khn = diagnostics.get('hesd_khn', None)
        if hesd_khn is not None:
            try:
                metrics[f'{prefix}/hesd_khn'] = float(hesd_khn)
            except Exception:
                pass
        neg_fraction = diagnostics.get('negative_fraction')
        if neg_fraction is not None:
            try:
                metrics[f'{prefix}/hessian_negative_fraction'] = float(neg_fraction)
            except Exception:
                pass
        signed_balance = diagnostics.get('signed_weight_balance')
        if signed_balance is not None:
            try:
                metrics[f'{prefix}/hessian_signed_mass_balance'] = float(signed_balance)
            except Exception:
                pass
        neg_mass = diagnostics.get('negative_weight_mass')
        if neg_mass is not None:
            try:
                metrics[f'{prefix}/hessian_negative_mass'] = float(neg_mass)
            except Exception:
                pass
        pos_mass = diagnostics.get('positive_weight_mass')
        if pos_mass is not None:
            try:
                metrics[f'{prefix}/hessian_positive_mass'] = float(pos_mass)
            except Exception:
                pass
        zero_mass = diagnostics.get('zero_weight_mass')
        if zero_mass is not None:
            try:
                metrics[f'{prefix}/hessian_zero_mass'] = float(zero_mass)
            except Exception:
                pass
        neg_mass_ratio_total = diagnostics.get('negative_mass_ratio_total')
        if neg_mass_ratio_total is not None:
            try:
                metrics[f'{prefix}/hessian_negative_mass_ratio_total'] = float(neg_mass_ratio_total)
            except Exception:
                pass
        neg_count = diagnostics.get('negative_count')
        if neg_count is not None:
            try:
                metrics[f'{prefix}/hessian_negative_count'] = float(neg_count)
            except Exception:
                pass
        pos_count = diagnostics.get('positive_count')
        if pos_count is not None:
            try:
                metrics[f'{prefix}/hessian_positive_count'] = float(pos_count)
            except Exception:
                pass
        zero_count = diagnostics.get('zero_count')
        if zero_count is not None:
            try:
                metrics[f'{prefix}/hessian_zero_count'] = float(zero_count)
            except Exception:
                pass
        min_eig = diagnostics.get('min_eigenvalue')
        if min_eig is not None:
            try:
                metrics[f'{prefix}/hessian_min_eigenvalue'] = float(min_eig)
            except Exception:
                pass
        max_eig = diagnostics.get('max_eigenvalue')
        if max_eig is not None:
            try:
                metrics[f'{prefix}/hessian_max_eigenvalue'] = float(max_eig)
            except Exception:
                pass
        neg_density_integral = diagnostics.get('negative_density_integral')
        if neg_density_integral is not None:
            try:
                metrics[f'{prefix}/hessian_negative_density_integral'] = float(neg_density_integral)
            except Exception:
                pass
        pos_density_integral = diagnostics.get('positive_density_integral')
        if pos_density_integral is not None:
            try:
                metrics[f'{prefix}/hessian_positive_density_integral'] = float(pos_density_integral)
            except Exception:
                pass
        neg_density_peak = diagnostics.get('negative_density_peak')
        if neg_density_peak is not None:
            try:
                metrics[f'{prefix}/hessian_negative_density_peak'] = float(neg_density_peak)
            except Exception:
                pass
        pos_density_peak = diagnostics.get('positive_density_peak')
        if pos_density_peak is not None:
            try:
                metrics[f'{prefix}/hessian_positive_density_peak'] = float(pos_density_peak)
            except Exception:
                pass

    return metrics


# Use a local copy of the optimizer experiment's config to keep this experiment self-contained
@hydra.main(config_path="cfg", config_name="task_shift_config", version_base=None)
def main(cfg: ExperimentConfig) -> Any:
    """Entry point for task-shift training that wraps training and LLA evaluations."""
    # Seed
    if cfg.seed is None or not isinstance(cfg.seed, (int, float)):
        cfg.seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {cfg.seed}")
    _seed_everything(int(cfg.seed))

    print(OmegaConf.to_yaml(cfg))

    # Transforms & dataset
    transform = transform_factory(cfg.data.dataset, cfg.net.type)
    train_set, _ = dataset_factory(cfg.data, transform=transform, with_testset=False)

    # Infer input dims if missing
    try:
        cfg.net.netparams.input_height = train_set[0][0].shape[1]
        cfg.net.netparams.input_width = train_set[0][0].shape[2]
    except Exception:
        pass

    # Model & learner
    net = model_factory(cfg.net)
    net.to(cfg.net.device)
    learner = create_learner(cfg.learner, net, cfg.net)
    net.train()

    # Out root for artifacts (local only) — make per-run unique directory
    arch = str(cfg.net.type)
    base_out = PROJECT_ROOT / 'outputs' / 'loss_landscape_at_new_and_old_tasks' / arch
    out_root, run_id = _make_unique_run_root(base_out, cfg)
    print(f"Saving artifacts under: {out_root} (run_id={run_id})")

    # wandb (numeric only)
    use_wandb = bool(getattr(cfg, 'use_wandb', False))
    if use_wandb:
        try:
            import wandb  # type: ignore
            run_cfg = {
                'seed': int(cfg.seed),
                'arch': arch,
                'task_shift_mode': str(cfg.task_shift_mode),
                'run_id': run_id,
                'artifact_dir': str(out_root),
            }
            try:
                cfg_dict = build_logging_config_dict(cfg)
            except Exception as e_sanitize:
                print(f"Warning: task shift logging sanitization failed, falling back to full config. Error: {e_sanitize}")
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            # Attach run metadata without mutating original cfg
            cfg_for_wandb: Dict[str, Any] = dict(cfg_dict)
            run_meta = dict(cfg_for_wandb.get('run_meta', {}))
            run_meta.update(run_cfg)
            cfg_for_wandb['run_meta'] = run_meta
            wandb.init(project=str(cfg.wandb.project), config=cfg_for_wandb, name=run_id)
        except Exception as e:
            print(f"[wandb] init failed: {e}. Proceeding without wandb.")
            use_wandb = False

    # Determine shift mode
    is_stateful = cfg.task_shift_mode in ["drifting_values", 'continuous_input_deformation']
    dataset_wrapper = create_stateful_dataset_wrapper(cfg, train_set) if is_stateful else None

    # num_workers logic similar to the reference script
    try:
        num_workers = int(os.cpu_count() or 0) if str(cfg.num_workers).lower() == "auto" else int(cfg.num_workers)
    except Exception:
        num_workers = 0
    if is_stateful and num_workers != 0:
        print("Info: Forcing num_workers=0 for stateful dataset wrapper.")
        num_workers = 0

    def _seed_worker(worker_id: int):
        """Per-worker seeding hook to keep dataloader shuffling deterministic."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    epochs_per_task = int(cfg.epochs)
    num_tasks = int(cfg.num_tasks)

    # Training across tasks
    # holders for the exact last training batch of the task
    last_batch_inp_cpu: torch.Tensor | None = None
    last_batch_target_cpu: Any | None = None
    if is_stateful:
        current_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
        if dataset_wrapper is not None and hasattr(dataset_wrapper, 'update_task'):
            dataset_wrapper.update_task()
        current_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
    else:
        current_train_set = create_stateless_dataset_wrapper(cfg, train_set, 0) or train_set
    next_train_set = None
    # Outer progress bar over tasks; inner over epochs per task
    with tqdm(total=num_tasks, desc='Tasks', position=0, leave=True, dynamic_ncols=True) as pbar_tasks:
        for task_idx in range(num_tasks):
            if task_idx > 0:
                if is_stateful:
                    current_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
                else:
                    current_train_set = next_train_set or train_set

            pbar_tasks.set_description(f"Tasks {task_idx+1}/{num_tasks}")

            # Reset last batch holders for this task
            last_batch_inp_cpu = None
            last_batch_target_cpu = None

            # LLA toggles are evaluated once per task to determine which analyses to run
            lla_cfg_block = getattr(cfg, 'lla', {})

            def _get_toggle(name: str, section: str, default: bool) -> bool:
                """Resolve boolean toggles from the cfg.lla block with sensible defaults."""
                try:
                    if hasattr(lla_cfg_block, name):
                        val = getattr(lla_cfg_block, name)
                        if isinstance(val, bool):
                            return val
                    sec = getattr(lla_cfg_block, section, None)
                    if sec is not None and hasattr(sec, 'enable'):
                        sec_val = getattr(sec, 'enable')
                        if isinstance(sec_val, bool):
                            return sec_val
                except Exception:
                    pass
                return default

            do_planes = _get_toggle('enable_planes', 'planes', True)
            do_spectrum = _get_toggle('enable_spectrum', 'spectrum', True)
            do_sam = _get_toggle('enable_sam', 'sam', False)

            # Optional LLA evaluation before any training on this task
            pre_eval_result: Dict[str, Any] | None = None
            pre_runtime: float | None = None
            pre_metrics: Dict[str, float] = {}
            if do_planes or do_spectrum or do_sam:
                pbar_tasks.set_postfix_str("LLA pre-training…")
                pre_start = time.time()
                pre_eval_result = _run_lla_evaluation(
                    model=net,
                    eval_source=current_train_set,
                    base_cfg=cfg,
                    out_root=out_root,
                    task_idx=task_idx,
                    arch=arch,
                    eval_phase='pre_training',
                    do_planes=do_planes,
                    do_sam=do_sam,
                    do_spectrum=do_spectrum,
                )
                pre_runtime = float(pre_eval_result.get('total_runtime_sec', time.time() - pre_start))
                pre_metrics = _collect_spectrum_metrics(
                    pre_eval_result.get('spectrum_json_path'),
                    do_spectrum,
                    prefix='pre_training',
                )
                pbar_tasks.set_postfix_str("LLA pre-training complete")

            train_loader = torch.utils.data.DataLoader(
                current_train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                worker_init_fn=_seed_worker if num_workers > 0 else None,
            )

            # Update drift if needed
            if is_stateful and hasattr(dataset_wrapper, 'update_drift'):
                dataset_wrapper.update_drift()

            # Train for epochs_per_task
            last_epoch_loss_mean: float | None = None
            last_epoch_accuracy: float | None = None
            with tqdm(total=epochs_per_task, desc=f"Epochs (Task {task_idx+1}/{num_tasks})", position=1, leave=False, dynamic_ncols=True) as pbar_epochs:
                for epoch in range(epochs_per_task):
                    net.train()
                    epoch_loss = 0.0
                    epoch_total = 0
                    epoch_correct = 0
                    for batch_idx, (inp, target) in enumerate(train_loader):
                        inp = inp.to(cfg.net.device)
                        if cfg.task_shift_mode == 'drifting_values':
                            drifting_values, original_labels = target
                            drifting_values = drifting_values.to(cfg.net.device)
                            original_labels = original_labels.to(cfg.net.device)
                            loss, output = learner.learn_from_partial_values(inp, drifting_values, original_labels)
                        else:
                            target = target.to(cfg.net.device)
                            loss, output = learner.learn(inp, target)
                        # capture the exact last training batch (store on CPU, detached)
                        try:
                            if cfg.task_shift_mode == 'drifting_values':
                                last_batch_inp_cpu = inp.detach().cpu()
                                last_batch_target_cpu = (
                                    drifting_values.detach().cpu(),
                                    original_labels.detach().cpu(),
                                )
                            else:
                                last_batch_inp_cpu = inp.detach().cpu()
                                last_batch_target_cpu = target.detach().cpu()
                        except Exception:
                            pass
                        # accumulate loss and accuracy stats for this epoch
                        epoch_loss += float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
                        if cfg.task_shift_mode != 'drifting_values':
                            with torch.no_grad():
                                preds = output.argmax(dim=1)
                                epoch_correct += int((preds == target).sum().item())
                                epoch_total += int(target.size(0))
                        else:
                            epoch_total += int(inp.size(0))
                    pbar_epochs.set_postfix(epoch=epoch+1)
                    pbar_epochs.update(1)

                    # end-of-epoch logging (numeric only) with global_epoch index
                    if use_wandb:
                        try:
                            import wandb  # type: ignore
                            global_epoch = task_idx * epochs_per_task + epoch
                            epoch_loss_mean = epoch_loss / max(1, len(train_loader))
                            last_epoch_loss_mean = float(epoch_loss_mean)
                            log_epoch: Dict[str, float | int] = {
                                'global_epoch': int(global_epoch),
                                'task_idx': int(task_idx),
                                'epoch_loss': float(epoch_loss_mean),
                            }
                            if cfg.task_shift_mode != 'drifting_values' and epoch_total > 0:
                                acc_val = float(epoch_correct / max(1, epoch_total))
                                log_epoch['epoch_accuracy'] = acc_val
                                last_epoch_accuracy = acc_val
                            wandb.log(log_epoch, step=int(global_epoch))  # type: ignore[attr-defined]
                        except Exception as e:
                            print(f"[wandb] epoch log failed: {e}")

            # End-of-task: compute numeric summaries and LLA metrics (once per task)
            pbar_tasks.set_postfix_str("LLA end-of-task…")
            # ---------------- Rank / feature based summaries (mirror reference script) ----------------
            rank_summary_list: List[Dict[str, float]] = []
            list_of_features_for_every_layers: List[torch.Tensor] = []
            feature_layer_names: List[str] | None = None
            feature_candidates: List[torch.Tensor] = []
            # Use learner.previous_features if available; else regenerate from last batch
            try:
                primary_features = getattr(learner, 'previous_features', None)
                if FeatureContainer is not None and isinstance(primary_features, FeatureContainer):
                    try:
                        feature_layer_names = list(primary_features.layer_names())
                    except Exception:
                        feature_layer_names = None
                feature_candidates = _coerce_feature_list(primary_features)
                if not feature_candidates and last_batch_inp_cpu is not None:
                    net_was_training = net.training
                    try:
                        net.eval()
                        with torch.no_grad():
                            inp_dev = last_batch_inp_cpu.to(cfg.net.device)
                            features_obj: Any | None = None
                            if hasattr(net, 'predict'):
                                predict_fn = getattr(net, 'predict')  # type: ignore[attr-defined]
                                predict_calls = [
                                    lambda: predict_fn(inp_dev),
                                    lambda: predict_fn(inp_dev, return_feature_container=True),
                                    lambda: predict_fn(inp_dev, return_feature_container=False),
                                ]
                                for call in predict_calls:
                                    try:
                                        predict_result = call()
                                    except TypeError:
                                        continue
                                    except Exception as pred_exc:
                                        LOGGER.debug("[rank] predict call failed: %s", pred_exc)
                                        continue
                                    else:
                                        if isinstance(predict_result, tuple):
                                            if len(predict_result) >= 2:
                                                features_obj = predict_result[1]
                                            elif len(predict_result) == 1:
                                                features_obj = predict_result[0]
                                        else:
                                            features_obj = predict_result
                                        break
                            if features_obj is None:
                                features_obj = getattr(net, 'previous_features', None)
                            feature_candidates = _coerce_feature_list(features_obj)
                            if not feature_candidates:
                                try:
                                    _ = net(inp_dev)
                                    feature_candidates = _coerce_feature_list(getattr(net, 'previous_features', None))
                                except Exception:
                                    pass
                    finally:
                        if net_was_training:
                            net.train()
                if not feature_candidates and last_batch_inp_cpu is not None:
                    try:
                        hook_features, hook_names = _extract_features_with_hooks(
                            net,
                            last_batch_inp_cpu,
                            device=str(cfg.net.device),
                        )
                        feature_candidates = _coerce_feature_list(hook_features)
                        if hook_names:
                            feature_layer_names = hook_names
                    except Exception as hook_exc:
                        LOGGER.debug("[rank] hook-based feature extraction failed: %s", hook_exc)
                list_of_features_for_every_layers = [
                    feat.view(feat.size(0), -1) for feat in feature_candidates if isinstance(feat, torch.Tensor)
                ]
                if list_of_features_for_every_layers:
                    rank_summary_list = compute_all_rank_measures_list(
                        features=list_of_features_for_every_layers,
                        use_pytorch_entropy_for_effective_rank=getattr(cfg, 'use_pytorch_entropy_for_effective_rank', False),
                        prop_for_approx_or_l1_rank=getattr(cfg, 'prop_for_approx_or_l1_rank', 0.99),
                        numerical_rank_epsilon=getattr(cfg, 'numerical_rank_epsilon', 1e-6),
                    )
                elif getattr(cfg, 'track_rank', False):
                    print(f"[rank] No feature snapshots captured for task {task_idx}; skipping rank metrics.")
            except Exception as e:
                print(f"[rank] end-of-task feature extraction failed: {e}")
                list_of_features_for_every_layers = []
            # Dead units
            dead_units_for_features: List[int] = []
            if getattr(cfg, 'track_dead_units', False) and list_of_features_for_every_layers:
                try:
                    dead_units_for_features = count_saturated_units_list(
                        features_list=list_of_features_for_every_layers,
                        activation_type=cfg.net.netparams.activation,
                        threshold=getattr(cfg, 'threshold_for_non_saturating_act', 0.0),
                    )
                except Exception as e:
                    print(f"[dead-units] failed: {e}")
            # Actual rank (optional)
            actual_rank_list: List[int] = []
            if getattr(cfg, 'track_actual_rank', False) and list_of_features_for_every_layers:
                try:
                    for feat in list_of_features_for_every_layers:
                        try:
                            actual_rank_list.append(int(torch.linalg.matrix_rank(feat.cpu().detach())))
                        except Exception:
                            actual_rank_list.append(0)
                except Exception as e:
                    print(f"[actual-rank] failed: {e}")
            # Rank dynamics
            rank_dynamics: Dict[str, float] = {}
            rank_dynamics_enabled = bool(getattr(cfg, 'enable_rank_dynamics', getattr(cfg, 'track_rank_drop', True)))
            if rank_dynamics_enabled and list_of_features_for_every_layers and rank_summary_list:
                try:
                    rank_dynamics = compute_rank_dynamics_from_features(
                        feature_list=list_of_features_for_every_layers,
                        rank_summary_list=rank_summary_list,
                        batch_size=getattr(cfg, 'specified_batch_size', getattr(cfg, 'batch_size', 0)),
                        mode=getattr(cfg, 'rank_drop_mode', 'ratio'),
                        use_theoretical_max_first=getattr(cfg, 'from_theoretical_max_first_feature_rank', False),
                    )
                except Exception as e:
                    print(f"[rank-dynamics] failed: {e}")

            # Weight magnitudes (naming: <layer>_mean_abs_weight)
            weight_summ: Dict[str, float] = {}
            try:
                if getattr(cfg, 'track_weight_magnitude', True):
                    layer_identifiers_cfg = getattr(cfg, 'layers_identifier', None)
                    layer_identifiers_clean: List[int | str] | None
                    if layer_identifiers_cfg is None:
                        layer_identifiers_clean = None
                    elif isinstance(layer_identifiers_cfg, str):
                        ident_str = layer_identifiers_cfg.strip()
                        if ident_str == "" or ident_str.lower() in {'none', 'null'}:
                            layer_identifiers_clean = None
                        else:
                            layer_identifiers_clean = [ident_str]
                    elif isinstance(layer_identifiers_cfg, numbers.Integral):
                        layer_identifiers_clean = [int(layer_identifiers_cfg)]
                    elif isinstance(layer_identifiers_cfg, Sequence) and not isinstance(layer_identifiers_cfg, (str, bytes)):
                        cleaned: List[int | str] = []
                        for ident in layer_identifiers_cfg:
                            if isinstance(ident, numbers.Integral):
                                cleaned.append(int(ident))
                            elif isinstance(ident, str):
                                ident_str = ident.strip()
                                if ident_str and ident_str.lower() not in {'none', 'null'}:
                                    cleaned.append(ident_str)
                        layer_identifiers_clean = cleaned or None
                    else:
                        layer_identifiers_clean = None

                    weight_stats = track_weight_stats(net, layer_identifiers=layer_identifiers_clean)
                    for name, val in weight_stats.items():
                        key = name if name.endswith('_mean_abs_weight') else f'{name}_mean_abs_weight'
                        weight_summ[key] = float(val)
            except Exception as e:
                print(f"[weights] tracking failed: {e}")

            # Build eval dataset from the exact last train batch if available; else snapshot current train set
            if last_batch_inp_cpu is not None and last_batch_target_cpu is not None:
                eval_source_ds = _SingleBatchDataset(last_batch_inp_cpu, last_batch_target_cpu)
            else:
                eval_source_ds = current_train_set

            # LLA: spectrum and/or planes/SAM (plots saved locally) — post-training on current task
            pbar_tasks.set_postfix_str("LLA post-training…")
            post_start = time.time()
            post_eval_result = _run_lla_evaluation(
                model=net,
                eval_source=eval_source_ds,
                base_cfg=cfg,
                out_root=out_root,
                task_idx=task_idx,
                arch=arch,
                eval_phase='post_training',
                do_planes=do_planes,
                do_sam=do_sam,
                do_spectrum=do_spectrum,
            )
            post_runtime = float(post_eval_result.get('total_runtime_sec', time.time() - post_start))
            post_metrics = _collect_spectrum_metrics(
                post_eval_result.get('spectrum_json_path'),
                do_spectrum,
                prefix='post_training',
            )

            # Prepare next task dataset (no evaluation here; next iteration will evaluate before training)
            if task_idx < num_tasks - 1:
                if is_stateful:
                    if dataset_wrapper is not None and hasattr(dataset_wrapper, 'update_task'):
                        dataset_wrapper.update_task()
                    next_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
                else:
                    next_train_set = create_stateless_dataset_wrapper(cfg, train_set, task_idx + 1) or train_set
                pbar_tasks.set_postfix_str("LLA post-training complete; next task prepared")
            else:
                next_train_set = None
                pbar_tasks.set_postfix_str("LLA post-training complete")

            # Log numeric metrics to wandb (images/checkpoints are NOT logged)
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    log_data: Dict[str, float | int | str] = {
                        'task_idx': task_idx,
                        'arch': arch,
                        'lla_runtime_sec': float(post_runtime),
                        'post_training/lla_runtime_sec': float(post_runtime),
                        **weight_summ,
                    }
                    log_data.update(post_metrics)
                    # Per-layer rank metrics
                    # Try semantic layer names first (mirror reference script behavior)
                    semantic_logged = False
                    try:
                        layer_names: List[str] | None = None
                        if hasattr(learner, 'get_layer_names'):
                            layer_names = learner.get_layer_names()
                        if not layer_names and 'feature_layer_names' in locals() and feature_layer_names:
                            layer_names = feature_layer_names
                        if layer_names:
                            for i, layer_name in enumerate(layer_names):
                                if i < len(rank_summary_list):
                                    rs = rank_summary_list[i]
                                    safe_name = str(layer_name).replace(' ', '_')
                                    log_data[f'{safe_name}_effective_rank'] = rs['effective_rank']
                                    log_data[f'{safe_name}_approximate_rank'] = rs['approximate_rank']
                                    log_data[f'{safe_name}_l1_distribution_rank'] = rs['l1_distribution_rank']
                                    log_data[f'{safe_name}_numerical_rank'] = rs['numerical_rank']
                            semantic_logged = True
                    except Exception as e_sem:
                        print(f"[rank] semantic layer naming failed: {e_sem}; falling back to indexed names.")
                    if not semantic_logged:
                        for i, rs in enumerate(rank_summary_list):
                            log_data[f'layer_{i}_effective_rank'] = rs['effective_rank']
                            log_data[f'layer_{i}_approximate_rank'] = rs['approximate_rank']
                            log_data[f'layer_{i}_l1_distribution_rank'] = rs['l1_distribution_rank']
                            log_data[f'layer_{i}_numerical_rank'] = rs['numerical_rank']
                    # Dead units
                    for i, dead in enumerate(dead_units_for_features):
                        log_data[f'layer_{i}_num_dead_units'] = int(dead)
                    # Actual ranks
                    for i, ar in enumerate(actual_rank_list):
                        log_data[f'layer_{i}_actual_rank'] = int(ar)
                    # Rank dynamics metrics
                    for k, v in rank_dynamics.items():
                        log_data[k] = float(v)
                    if last_epoch_loss_mean is not None:
                        log_data['task_last_epoch_loss'] = float(last_epoch_loss_mean)
                    if last_epoch_accuracy is not None:
                        log_data['task_last_epoch_accuracy'] = float(last_epoch_accuracy)
                    if pre_runtime is not None:
                        log_data['pre_training/lla_runtime_sec'] = float(pre_runtime)
                        log_data['pre_training/task_idx'] = int(task_idx)
                    if pre_metrics:
                        log_data.update(pre_metrics)
                    # Align end-of-task metrics with global epoch timeline
                    end_of_task_step = int((task_idx + 1) * epochs_per_task - 1)
                    log_data['global_epoch'] = end_of_task_step
                    wandb.log(log_data, step=end_of_task_step)  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

            pbar_tasks.update(1)
    print("Task-shift training with pre- and post-task LLA analyses complete.")


if __name__ == "__main__":
    # Optional: ensure CUDA multiprocessing safety in some environments
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("multiprocessing start method set to 'spawn'.")
    except Exception:
        pass
    main()
