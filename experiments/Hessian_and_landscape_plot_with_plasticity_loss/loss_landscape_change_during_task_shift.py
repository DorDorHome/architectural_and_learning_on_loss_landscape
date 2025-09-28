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
from pathlib import Path
import hashlib
import uuid
from typing import Any, Dict, List, Tuple

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

# LLA submodule integration (replace previous custom pipeline usage)
# Add submodule src path
LLA_SUBMODULE_SRC = Path(__file__).parent / 'external' / 'loss-landscape-analysis' / 'src'
if LLA_SUBMODULE_SRC.exists() and str(LLA_SUBMODULE_SRC) not in sys.path:
    sys.path.insert(0, str(LLA_SUBMODULE_SRC))

try:  # Import required LLA components
    from src_lla.hessian.hessian import hessian_calc  # type: ignore
    from src_lla.loss_landscapes.dev import vec_H_eigenvects  # type: ignore
    from src_lla.loss_landscapes.main import random_plane  # type: ignore
    from src_lla.loss_landscapes.metrics.metric import Metric  # type: ignore
    LLA_AVAILABLE = True
except Exception as e:  # fallback flag
    print(f"[LLA] Submodule import failed ({e}); falling back to legacy pipeline (limited features).")
    LLA_AVAILABLE = False

# Legacy SAM surface still relies on old custom implementation; we keep it optional if needed.
try:
    from experiments.Hessian_and_landscape_plot_with_plasticity_loss.lla_pipeline import (
        plot_sam_surface,  # type: ignore
    )  # Only SAM retained for now
except Exception:
    plot_sam_surface = None  # type: ignore


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
    """Construct a minimal LLA config dict expected by lla_pipeline functions.
    Keeps moderate defaults to avoid large per-task slowdowns.
    """
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
    # Allow evaluation-only overrides from YAML under cfg.lla
    try:
        user_lla = getattr(base_cfg, 'lla', None)
        if user_lla is not None:
            user_lla_dict = OmegaConf.to_container(user_lla, resolve=True)  # type: ignore[arg-type]
            if isinstance(user_lla_dict, dict):
                _deep_update(lla_cfg['lla'], user_lla_dict)
                # Map ESD keys to SLQ controls used by the analyzer if present
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
        super().__init__()
        self.x = x.detach().to(device)
        self.y = y.detach().to(device)
        self.device = device
        self.criterion = criterion

    def _forward(self, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
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


def _smooth_esd(eigs: List[List[float]], weights: List[List[float]], grid: int = 200, sigma_scale: float = 0.02) -> dict:
    all_e = np.array([e for sub in eigs for e in sub], dtype=np.float64)
    all_w = np.array([w for sub in weights for w in sub], dtype=np.float64)
    if all_e.size == 0:
        return {
            'lambda_grid': np.zeros(1),
            'density': np.zeros(1),
            'lam_min': 0.0,
            'lam_max': 0.0,
            'sigma': 0.0,
            'degenerate': True,
        }
    lam_min = float(all_e.min())
    lam_max = float(all_e.max())
    span = lam_max - lam_min
    if span <= 1e-12:
        span = 1.0
        lam_min -= 0.5
        lam_max += 0.5
    lambda_grid = np.linspace(lam_min, lam_max, grid, dtype=np.float64)
    sigma = max(sigma_scale * (lam_max - lam_min), 1e-8)
    dens = np.zeros_like(lambda_grid)
    k = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    for ev, w in zip(all_e, all_w):
        dens += w * k * np.exp(-0.5 * ((lambda_grid - ev) / sigma) ** 2)
    area = np.trapz(dens, lambda_grid)
    if area > 0:
        dens /= area
    degenerate = bool(len(np.unique(np.round(all_e, 8))) == 1)
    return {
        'lambda_grid': lambda_grid,
        'density': dens,
        'lam_min': float(lam_min),
        'lam_max': float(lam_max),
        'sigma': float(sigma),
        'degenerate': degenerate,
        'raw_eigs': all_e,
        'raw_weights': all_w,
    }


def _compute_planes_and_spectrum_with_lla(
    model: nn.Module,
    eval_loader: DataLoader,
    task_dir: Path,
    lla_cfg: Dict[str, Any],
    do_planes: bool,
    do_spectrum: bool,
) -> Dict[str, Any]:
    """Core refactored LLA evaluation: Hessian eigenvectors, plane, spectrum (top-k, trace, ESD).
    Returns dict with similar keys to legacy pipeline for downstream logging compatibility.
    """
    device = lla_cfg.get('device', 'cuda:0')
    loss_name = lla_cfg.get('learner', {}).get('loss', 'cross_entropy')
    criterion = _make_criterion(loss_name)
    x, y = _first_batch(eval_loader, device)
    metric = FixedBatchMetric(x, y, device, criterion)

    out: Dict[str, Any] = {}
    plane_dir = task_dir / 'planes'
    plane_dir.mkdir(exist_ok=True)

    spectrum_dir = task_dir  # keep in task root for backward compatibility

    # Parameters from config or sensible LLA defaults
    planes_cfg = lla_cfg.get('lla', {}).get('planes', {})
    grid_res = int(planes_cfg.get('grid_resolution', 41))
    # Derive distance: prefer explicit 'distance'; else infer symmetric half-range from 'span'
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

    top_k = int(lla_cfg.get('lla', {}).get('spectrum', {}).get('top_k', 5))
    power_iters = int(lla_cfg.get('lla', {}).get('spectrum', {}).get('power_iters', 100))
    trace_probes = int(lla_cfg.get('lla', {}).get('spectrum', {}).get('hutchinson_probes', 32))
    esd_cfg = lla_cfg.get('lla', {}).get('spectrum', {}).get('esd', {})
    lanczos_steps = int(esd_cfg.get('lanczos_steps', lla_cfg.get('lla', {}).get('spectrum', {}).get('slq_m', 80)))
    esd_probes = int(esd_cfg.get('num_probes', lla_cfg.get('lla', {}).get('spectrum', {}).get('slq_probes', 8)))
    esd_bins = int(esd_cfg.get('bins', lla_cfg.get('lla', {}).get('spectrum', {}).get('slq_grid', 200)))
    sigma_scale = float(esd_cfg.get('sigma_scale', 0.02))

    t_all0 = time.time()
    hess = hessian_calc(model, metric)

    # Top eigen-stuff
    t_eig0 = time.time()
    eigvals, eigvecs = hess.eigs_calc(top_n=max(2, top_k), n_iter=power_iters)
    t_eigs = time.time() - t_eig0

    # Trace via Hutchinson
    t_tr0 = time.time()
    trace_est = hess.tr_calc(n_iter=trace_probes)
    t_trace = time.time() - t_tr0

    # ESD via SLQ
    t_esd0 = time.time()
    esd_eigs, esd_weights = hess.esd_calc(n_iter=lanczos_steps, n_v=esd_probes)
    esd_pack = _smooth_esd(esd_eigs, esd_weights, grid=esd_bins, sigma_scale=sigma_scale)
    t_esd = time.time() - t_esd0

    # Plane (Hessian-aligned)
    plane_info: Dict[str, Any] | None = None
    if do_planes:
        try:
            a1, a2 = vec_H_eigenvects(hess)
            plane_arr = random_plane(
                model,
                metric,
                distance=distance,
                steps=grid_res,
                normalization=normalization,
                deepcopy_model=True,
                a1=a1,
                a2=a2,
                mode='add',
            )
            plane_path = plane_dir / 'plane_hessian.npy'
            np.save(plane_path, plane_arr)
            # Visualizations: heatmap (centered), 3D surface, and contour
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib import cm
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

                # Build centered coordinate grid: (-distance .. distance)
                # random_plane returns a square matrix of shape (steps, steps)
                coords = np.linspace(-distance, distance, plane_arr.shape[0])
                X, Y = np.meshgrid(coords, coords, indexing='xy')

                # 1) Heatmap (centered)
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
                heatmap_png = plane_dir / 'plane_hessian_heatmap.png'
                fig_h.tight_layout()
                fig_h.savefig(heatmap_png)
                plt.close(fig_h)

                # 2) 3D surface plot
                fig3d = plt.figure(figsize=(6, 4.8))
                ax3d = fig3d.add_subplot(111, projection='3d')
                surf = ax3d.plot_surface(X, Y, plane_arr, cmap=cm.viridis, linewidth=0, antialiased=True)
                ax3d.set_xlabel('Dir 1')
                ax3d.set_ylabel('Dir 2')
                ax3d.set_zlabel('Loss')
                ax3d.set_title('Loss surface (Hessian plane)')
                fig3d.colorbar(surf, shrink=0.6, aspect=12, pad=0.08).set_label('Loss')
                surface_png = plane_dir / 'plane_hessian_surface3d.png'
                fig3d.tight_layout()
                fig3d.savefig(surface_png)
                plt.close(fig3d)

                # 3) Contour plot (optional for clearer level sets)
                fig_c, ax_c = plt.subplots(figsize=(5.2, 4.4))
                CS = ax_c.contour(X, Y, plane_arr, levels=20, cmap='viridis')
                ax_c.clabel(CS, inline=True, fontsize=7)
                ax_c.set_xlabel('Direction 1 (center=0)')
                ax_c.set_ylabel('Direction 2 (center=0)')
                ax_c.set_title('Loss contours (Hessian plane)')
                contour_png = plane_dir / 'plane_hessian_contour.png'
                fig_c.tight_layout()
                fig_c.savefig(contour_png)
                plt.close(fig_c)

            except Exception as e:
                print(f"[LLA][plane] plot failed: {e}")
            plane_info = {
                'grid_resolution': grid_res,
                'distance': distance,
                'normalization': normalization,
                'npy': str(plane_path),
                'heatmap_png': str(plane_dir / 'plane_hessian_heatmap.png'),
                'surface3d_png': str(plane_dir / 'plane_hessian_surface3d.png'),
                'contour_png': str(plane_dir / 'plane_hessian_contour.png'),
            }
        except Exception as e:
            print(f"[LLA] Hessian plane failed: {e}")

    # Build spectrum json (compat shape)
    spec = {
        'top_k': eigvals[:top_k],
        'hutchinson_trace': float(trace_est),
        'dim': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'k_used': top_k,
        'probes': trace_probes,
        'algorithm_topk': 'LLA_hessian_power_iteration',
        'algorithm_trace': 'LLA_Hutchinson',
        'algorithm_esd': 'LLA_SLQ',
        'timing_seconds': {
            'topk': t_eigs,
            'hutchinson_trace': t_trace,
            'esd_slq': t_esd,
            'total': t_eigs + t_trace + t_esd,
        },
        'esd': {
            'n_grid': esd_bins,
            'probes': esd_probes,
            'm': lanczos_steps,
            'lam_min': float(esd_pack['lam_min']),
            'lam_max': float(esd_pack['lam_max']),
            'sigma': float(esd_pack['sigma']),
            'degenerate': bool(esd_pack['degenerate']),
        },
    }

    # -------------------- Diagnostics (top-k vs Rayleigh vs SLQ bulk) --------------------
    diagnostics: Dict[str, Any] = {}
    try:
        # Normalize eigvals to a plain python list of floats for robust downstream processing
        try:
            eigvals_list: List[float] = [float(v) for v in eigvals]
        except Exception:
            eigvals_list = []
        # Attempt to read rayleigh eigenvalues produced by plane helper if present
        axes_meta_path = task_dir / 'hessian_axes_meta.json'
        rayleigh_top1 = None
        if axes_meta_path.exists():
            try:
                axes_meta = json.loads(axes_meta_path.read_text())
                rayleigh_top1 = float(axes_meta.get('rayleigh_top1', None))
                diagnostics['rayleigh_top1'] = rayleigh_top1
                if 'rayleigh_top2' in axes_meta:
                    diagnostics['rayleigh_top2'] = float(axes_meta['rayleigh_top2'])
            except Exception:
                pass
        # r1: alignment of top eigenvalue between Rayleigh and power iteration
        if rayleigh_top1 is not None and len(eigvals_list) > 0:
            top1 = eigvals_list[0]
            diagnostics['r1_rayleigh_alignment'] = float(abs(top1 - rayleigh_top1) / (abs(top1) + 1e-12))
        else:
            diagnostics['r1_rayleigh_alignment'] = None
        # r2: SLQ capture of top eigenvalue (max raw SLQ sample vs eigvals[0])
        raw_eigs_all = esd_pack.get('raw_eigs', np.array([]))  # type: ignore[assignment]
        if isinstance(raw_eigs_all, np.ndarray) and raw_eigs_all.size > 0 and len(eigvals_list) > 0:
            top1 = eigvals_list[0]
            try:
                max_raw = float(raw_eigs_all.max())
                diagnostics['r2_slq_top_alignment'] = float(abs(max_raw - top1) / (abs(top1) + 1e-12))
                diagnostics['slq_max_raw_eig'] = max_raw
                diagnostics['slq_min_raw_eig'] = float(raw_eigs_all.min())
                diagnostics['slq_std_raw_eig'] = float(np.std(raw_eigs_all))
            except Exception:
                diagnostics['r2_slq_top_alignment'] = None
        else:
            diagnostics['r2_slq_top_alignment'] = None
        # trace consistency fraction
        partial_trace = float(np.sum(eigvals_list[:top_k])) if len(eigvals_list) > 0 else 0.0
        diagnostics['partial_topk_sum'] = partial_trace
        diagnostics['trace_fraction_topk'] = float(partial_trace / (float(trace_est) + 1e-12))
        # bulk width vs top eigenvalue
        if isinstance(raw_eigs_all, np.ndarray) and raw_eigs_all.size > 0 and len(eigvals_list) > 0:
            top1 = eigvals_list[0]
            diagnostics['bulk_std_over_top1'] = float(np.std(raw_eigs_all) / (abs(top1) + 1e-12))
        # probe variance (approximate) – per-probe mean & std of max value
        try:
            per_probe_max = [max(p) if len(p) > 0 else 0.0 for p in esd_eigs]
            per_probe_mean = [float(np.mean(p)) if len(p) > 0 else 0.0 for p in esd_eigs]
            diagnostics['probe_max_mean'] = float(np.mean(per_probe_max))
            diagnostics['probe_max_std'] = float(np.std(per_probe_max))
            diagnostics['probe_mean_mean'] = float(np.mean(per_probe_mean))
            diagnostics['probe_mean_std'] = float(np.std(per_probe_mean))
        except Exception:
            pass
    except Exception as e_diag:
        diagnostics['diagnostics_error'] = f"{e_diag}"

    spec['diagnostics'] = diagnostics

    # Human-readable printout (concise)
    try:
        print("[LLA][Diagnostics] Top-k eigenvalues (power iter):", [float(v) for v in eigvals_list[:top_k]])
        if diagnostics.get('rayleigh_top1') is not None:
            print(f"[LLA][Diagnostics] Rayleigh top1: {diagnostics.get('rayleigh_top1')}  alignment r1: {diagnostics.get('r1_rayleigh_alignment')}")
        print(f"[LLA][Diagnostics] SLQ raw eig max: {diagnostics.get('slq_max_raw_eig')}  r2 alignment: {diagnostics.get('r2_slq_top_alignment')}")
        print(f"[LLA][Diagnostics] Trace fraction (top_k / trace): {diagnostics.get('trace_fraction_topk')}")
        print(f"[LLA][Diagnostics] bulk std / top1: {diagnostics.get('bulk_std_over_top1')}")
        if 'probe_max_std' in diagnostics:
            print(f"[LLA][Diagnostics] probe max mean/std: {diagnostics.get('probe_max_mean')} / {diagnostics.get('probe_max_std')}")
    except Exception:
        pass

    # Save ESD arrays
    np.savez(
        spectrum_dir / 'spectrum_esd.npz',
        lambda_grid=esd_pack['lambda_grid'],
        density=esd_pack['density'],
        raw_eigs=esd_pack['raw_eigs'],
        raw_weights=esd_pack['raw_weights'],
    )
    # Plot ESD (smoothed density)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Linear-scale density with top-k overlay
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(esd_pack['lambda_grid'], esd_pack['density'], label='SLQ density')
        # Overlay vertical lines for top-k
        try:
            for i, v in enumerate(eigvals[:top_k]):
                ax.axvline(float(v), color='r', linestyle='--', alpha=0.6, label='top-k eigvals' if i == 0 else None)
        except Exception:
            pass
        ax.set_xlabel('eigenvalue')
        ax.set_ylabel('density')
        ax.set_title('ESD (SLQ) + top-k')
        ax.legend(loc='best', fontsize=7)
        fig.tight_layout()
        esd_png = spectrum_dir / 'spectrum_esd.png'
        fig.savefig(esd_png)
        # Log-scale variant
        try:
            ax.set_yscale('log')
            fig.savefig(spectrum_dir / 'spectrum_esd_log.png')
        except Exception:
            pass
        plt.close(fig)
    except Exception as e:
        print(f"[LLA][ESD] plotting failed: {e}")

    # Plot raw eigenvalue histogram (weighted by SLQ weights)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig_hg, ax_hg = plt.subplots(figsize=(5, 4))
        raw_eigs = esd_pack.get('raw_eigs', np.array([]))
        raw_weights = esd_pack.get('raw_weights', np.array([]))
        if isinstance(raw_eigs, np.ndarray) and raw_eigs.size > 0:
            weights = None
            if isinstance(raw_weights, np.ndarray) and raw_weights.size == raw_eigs.size and raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum()
            ax_hg.hist(raw_eigs, bins=min(50, max(10, int(np.sqrt(raw_eigs.size)))), weights=weights, color='steelblue', edgecolor='black')
            # Overlay top-k vertical lines
            try:
                for v in eigvals[:top_k]:
                    ax_hg.axvline(float(v), color='r', linestyle='--', alpha=0.6)
            except Exception:
                pass
        ax_hg.set_xlabel('eigenvalue')
        ax_hg.set_ylabel('weighted count' if (isinstance(raw_weights, np.ndarray) and raw_weights.size == raw_eigs.size) else 'count')
        ax_hg.set_title('Spectrum histogram (SLQ + top-k)')
        fig_hg.tight_layout()
        hist_png = spectrum_dir / 'spectrum_hist.png'
        fig_hg.savefig(hist_png)
        plt.close(fig_hg)
    except Exception as e:
        print(f"[LLA][ESD] histogram plotting failed: {e}")

    (spectrum_dir / 'spectrum.json').write_text(json.dumps(spec, indent=2))

    # Clean up Hessian graph
    try:
        hess.reset()
    except Exception:
        pass

    if plane_info is not None:
        out['plane'] = plane_info
    if do_spectrum:
        out['spectrum'] = spec
    out['total_runtime_sec'] = time.time() - t_all0
    return out


def _rename_with_arch_task(path: Path, arch: str, task_idx: int) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    new_name = f"{stem}_{arch}_task{task_idx}{path.suffix}"
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
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
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


def _run_lla_end_of_task(
    model: nn.Module,
    train_set,
    base_cfg: ExperimentConfig,
    out_root: Path,
    task_idx: int,
    arch: str,
    do_planes: bool = True,
    do_sam: bool = False,
    do_spectrum: bool = True,
) -> Dict[str, Any]:
    device = str(base_cfg.net.device)
    model.eval()

    # Build eval loader (single-batch)
    eval_ds = _snapshot_eval_dataset(train_set)
    eval_bs = min(128, len(eval_ds)) if hasattr(eval_ds, '__len__') else 128
    eval_loader = DataLoader(eval_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=True)

    # LLA cfg & output directory
    lla_cfg = _build_lla_cfg(base_cfg, eval_batch_size=eval_bs)
    task_dir = out_root / f"task_{task_idx:04d}_{arch}"
    task_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {}
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
        for fname in ['spectrum_esd.png', 'spectrum_hist.png', 'spectrum_esd.npz', 'spectrum.json']:
            p = task_dir / fname
            _rename_with_arch_task(p, arch, task_idx)
        if 'plane' in result:
            for key in ['heatmap_png', 'surface3d_png', 'contour_png', 'npy']:
                if key in result['plane']:
                    try:
                        _rename_with_arch_task(Path(result['plane'][key]), arch, task_idx)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[LLA] End-of-task LLA evaluation failed: {e}")

    if do_sam:
        print("[LLA][SAM] SAM surface not yet refactored to LLA; skipping (previous implementation removed).")

    return result


# Use a local copy of the optimizer experiment's config to keep this experiment self-contained
@hydra.main(config_path="cfg", config_name="task_shift_config", version_base=None)
def main(cfg: ExperimentConfig) -> Any:
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
    base_out = PROJECT_ROOT / 'outputs' / 'loss_landscape_taskshift' / arch
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
            wandb.init(project=str(cfg.wandb.project), config=run_cfg, name=run_id)
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
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    epochs_per_task = int(cfg.epochs)
    num_tasks = int(cfg.num_tasks)

    # Training across tasks
    prev_layer_gini_mean: float | None = None
    prev_layer_rank_mean: float | None = None
    # holders for the exact last training batch of the task
    last_batch_inp_cpu: torch.Tensor | None = None
    last_batch_target_cpu: Any | None = None
    # Outer progress bar over tasks; inner over epochs per task
    with tqdm(total=num_tasks, desc='Tasks', position=0, leave=True, dynamic_ncols=True) as pbar_tasks:
        for task_idx in range(num_tasks):
            pbar_tasks.set_description(f"Tasks {task_idx+1}/{num_tasks}")

            # Build task dataset
            if is_stateful:
                if hasattr(dataset_wrapper, 'update_task'):
                    dataset_wrapper.update_task()
                current_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
            else:
                current_train_set = create_stateless_dataset_wrapper(cfg, train_set, task_idx) or train_set

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
            # Use learner.previous_features if available; else regenerate from last batch
            try:
                if hasattr(learner, 'previous_features') and isinstance(learner.previous_features, list):
                    list_of_features_for_every_layers = [f.detach() for f in learner.previous_features]
                elif last_batch_inp_cpu is not None:
                    net.eval()
                    with torch.no_grad():
                        inp_dev = last_batch_inp_cpu.to(cfg.net.device)
                        if hasattr(net, 'predict'):
                            _, list_of_features_for_every_layers = net.predict(inp_dev)  # type: ignore
                        else:
                            _ = net(inp_dev)
                # Flatten to (batch, dim)
                list_of_features_for_every_layers = [feat.view(feat.size(0), -1) for feat in list_of_features_for_every_layers if isinstance(feat, torch.Tensor)]
                if list_of_features_for_every_layers:
                    rank_summary_list = compute_all_rank_measures_list(
                        features=list_of_features_for_every_layers,
                        use_pytorch_entropy_for_effective_rank=getattr(cfg, 'use_pytorch_entropy_for_effective_rank', False),
                        prop_for_approx_or_l1_rank=getattr(cfg, 'prop_for_approx_or_l1_rank', 0.99),
                        numerical_rank_epsilon=getattr(cfg, 'numerical_rank_epsilon', 1e-6),
                    )
            except Exception as e:
                print(f"[rank] end-of-task feature extraction failed: {e}")
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
                    weight_stats = track_weight_stats(net, layer_identifiers=getattr(cfg, 'layers_identifier', None))
                    for name, val in weight_stats.items():
                        key = name if name.endswith('_mean_abs_weight') else f'{name}_mean_abs_weight'
                        weight_summ[key] = float(val)
            except Exception as e:
                print(f"[weights] tracking failed: {e}")

            # LLA: spectrum/planes/SAM selection from YAML toggles
            lla_cfg_block = getattr(cfg, 'lla', {})
            # Prefer top-level toggles; fall back to per-section 'enable'
            def _get_toggle(name: str, section: str, default: bool) -> bool:
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

            # Build eval dataset from the exact last train batch if available; else snapshot current train set
            if last_batch_inp_cpu is not None and last_batch_target_cpu is not None:
                eval_source_ds = _SingleBatchDataset(last_batch_inp_cpu, last_batch_target_cpu)
            else:
                eval_source_ds = current_train_set

            # LLA: spectrum and/or planes/SAM (plots saved locally)
            t0 = time.time()
            _ = _run_lla_end_of_task(
                model=net,
                train_set=eval_source_ds,
                base_cfg=cfg,
                out_root=out_root,
                task_idx=task_idx,
                arch=arch,
                do_planes=do_planes,
                do_sam=do_sam,
                do_spectrum=do_spectrum,
            )
            dt = time.time() - t0

            # Parse top-k from saved spectrum.json if present
            topk_metrics: Dict[str, float] = {}
            spec_json = out_root / f"task_{task_idx:04d}_{arch}" / 'spectrum.json'
            if not spec_json.exists():
                # try renamed variant
                for p in (out_root / f"task_{task_idx:04d}_{arch}").glob('spectrum*.json'):
                    spec_json = p
                    break
            try:
                if spec_json.exists() and do_spectrum:
                    spec = json.loads(spec_json.read_text())
                    topk = spec.get('top_k', [])
                    for i, v in enumerate(topk):
                        topk_metrics[f'hessian_top_eig_{i+1}'] = float(v)
                    trace_estimate = spec.get('hutchinson_trace', None)
                    if trace_estimate is not None:
                        topk_metrics['hessian_trace_estimate'] = float(trace_estimate)
                    diagnostics = spec.get('diagnostics', {})
                    if isinstance(diagnostics, dict):
                        trace_fraction = diagnostics.get('trace_fraction_topk', None)
                        if trace_fraction is not None:
                            topk_metrics['hessian_trace_fraction_topk'] = float(trace_fraction)
                        partial_topk_sum = diagnostics.get('partial_topk_sum', None)
                        if partial_topk_sum is not None:
                            topk_metrics['hessian_partial_topk_sum'] = float(partial_topk_sum)
            except Exception:
                pass

            # Log numeric metrics to wandb (images/checkpoints are NOT logged)
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    log_data: Dict[str, float | int | str] = {
                        'task_idx': task_idx,
                        'arch': arch,
                        'lla_runtime_sec': dt,
                        **weight_summ,
                        **topk_metrics,
                    }
                    # Per-layer rank metrics
                    # Try semantic layer names first (mirror reference script behavior)
                    semantic_logged = False
                    try:
                        if hasattr(learner, 'get_layer_names'):
                            layer_names = learner.get_layer_names()
                            for i, layer_name in enumerate(layer_names):
                                if i < len(rank_summary_list):
                                    rs = rank_summary_list[i]
                                    log_data[f'{layer_name}_effective_rank'] = rs['effective_rank']
                                    log_data[f'{layer_name}_approximate_rank'] = rs['approximate_rank']
                                    log_data[f'{layer_name}_l1_distribution_rank'] = rs['l1_distribution_rank']
                                    log_data[f'{layer_name}_numerical_rank'] = rs['numerical_rank']
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
                    # Align end-of-task metrics with global epoch timeline
                    end_of_task_step = int((task_idx + 1) * epochs_per_task - 1)
                    log_data['global_epoch'] = end_of_task_step
                    wandb.log(log_data, step=end_of_task_step)  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

            # proceed to next task (no gini/rank mean retained anymore)
            prev_layer_gini_mean = None
            prev_layer_rank_mean = None

            pbar_tasks.update(1)
    print("Task-shift training with end-of-task LLA analyses complete.")


if __name__ == "__main__":
    # Optional: ensure CUDA multiprocessing safety in some environments
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("multiprocessing start method set to 'spawn'.")
    except Exception:
        pass
    main()
