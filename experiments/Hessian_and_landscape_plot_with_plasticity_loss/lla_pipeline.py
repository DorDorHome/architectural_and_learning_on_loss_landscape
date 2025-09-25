"""
LLA Pipeline: planes, spectrum, SAM surface, and mode connectivity.

Dependencies:
- This project (model/data factories under src/*)
- LLA (Loss Landscape Analysis) via git submodule under this experiment folder: external/loss-landscape-analysis
- dnn-mode-connectivity via git submodule: external/dnn-mode-connectivity

This module exposes pure functions that can be imported and a CLI via argparse.
It is designed to run a short fine-tune, save checkpoints, compute directions (Hessian, random, trajectory PCA),
evaluate planes and SAM surfaces, compute spectrum (top-k, trace, ESD), run mode connectivity, and couple with rank stats.

Note: Heavy computations are guarded with chunking, fixed eval batch, and configurable fallbacks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports from the main repo
# These paths assume this file is executed from repo root or the experiment folder.
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loading.dataset_factory import dataset_factory  # type: ignore
from src.data_loading.transform_factory import transform_factory  # type: ignore
from src.models.model_factory import model_factory  # type: ignore
from configs.configurations import DataConfig, NetConfig, NetParams  # type: ignore

# Optional: W&B
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

# Matplotlib only for saving figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------
# Utilities and dataclasses
# -------------------------

@dataclass
class LLARuntime:
    device: str
    fp16: bool = False
    workers: int = 0
    pbar: bool = True
    abort_if_estimated_hours_over: Optional[float] = None


def set_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def timer_block(msg: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"[START] {msg}")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            print(f"[END] {msg} in {dt:.2f}s")
    return _Timer()


# -------------------------
# Data & model preparation
# -------------------------

def prepare_data_and_model(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, nn.Module]:
    device = cfg.get('device', 'cuda:0')
    # Build transform and datasets
    transform = transform_factory(cfg['data']['dataset'], cfg['net']['type'])
    # Wrap dict into DataConfig for factory compatibility
    data_cfg = DataConfig(**cfg['data'])
    train_set, _ = dataset_factory(data_cfg, transform=transform, with_testset=False)

    # Fixed eval subset (single loader with fixed batch size)
    g = torch.Generator()
    g.manual_seed(cfg['lla']['evaluation_data'].get('fixed_eval_seed', 42))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.get('batch_size', 256),
        shuffle=True,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=True,
        generator=g,
    )

    eval_loader = DataLoader(
        train_set,
        batch_size=cfg['lla']['evaluation_data'].get('eval_batch_size', 256),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        generator=g,
    )

    # Build model (wrap into NetConfig/NetParams)
    netparams = NetParams(**cfg['net'].get('netparams', {}))
    # Auto-infer ConvNet dims/channels from a dataset sample when missing
    try:
        sample_x, _ = train_set[0]  # type: ignore[index]
        if isinstance(sample_x, torch.Tensor):
            if sample_x.ndim == 3:
                c, h, w = int(sample_x.shape[0]), int(sample_x.shape[1]), int(sample_x.shape[2])
            elif sample_x.ndim == 2:
                c, h, w = 1, int(sample_x.shape[0]), int(sample_x.shape[1])
            else:
                c, h, w = None, None, None  # type: ignore[assignment]
            # Fill only if unspecified
            if getattr(netparams, 'input_height', None) is None:
                netparams.input_height = h  # type: ignore[assignment]
            if getattr(netparams, 'input_width', None) is None:
                netparams.input_width = w  # type: ignore[assignment]
            if getattr(netparams, 'in_channels', None) is None or netparams.in_channels in (0, 1) and c == 3:
                netparams.in_channels = c  # type: ignore[assignment]
    except Exception:
        pass
    net_cfg = NetConfig(type=cfg['net']['type'], netparams=netparams, network_class=cfg['net'].get('network_class'))
    net = model_factory(net_cfg)
    net.to(device)
    return train_loader, eval_loader, net


# -------------------------
# Training and checkpoints
# -------------------------

def quick_finetune_and_checkpoint(cfg: Dict[str, Any], model: nn.Module, train_loader: DataLoader, out_dir: Path) -> List[Path]:
    device = cfg.get('device', 'cuda:0')
    epochs = int(cfg['lla']['training'].get('epochs_short', 2))
    lr = float(cfg['lla']['training'].get('lr', 0.01))
    weight_decay = float(cfg['lla']['training'].get('weight_decay', 0.0))
    save_checkpoints = bool(cfg['lla']['training'].get('save_checkpoints', True))
    max_ckpts = int(cfg['lla']['training'].get('max_checkpoints', 10))

    # Simple optimizer and loss based on top-level learner
    opt_name = cfg['learner'].get('opt', 'sgd').lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg['learner'].get('momentum', 0.0), weight_decay=weight_decay)

    loss_name = cfg['learner'].get('loss', 'cross_entropy')
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()
    steps_per_epoch = len(train_loader)
    save_every = max(1, steps_per_epoch // max(1, max_ckpts))

    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    global_step = 0
    for ep in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = _model_forward_outputs(model, x)
            if isinstance(criterion, nn.MSELoss):
                # If labels are class indices, convert to one-hot or float as needed.
                if y.dtype in (torch.long, torch.int64):
                    y_onehot = torch.nn.functional.one_hot(y, num_classes=out.shape[-1]).float()
                else:
                    y_onehot = y.float()
                loss = criterion(out, y_onehot)
            else:
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if save_checkpoints and (global_step % save_every == 0):
                p = ckpt_dir / f"ckpt_step{global_step:06d}.pt"
                torch.save({'step': global_step, 'state_dict': model.state_dict()}, p)
                saved.append(p)
            global_step += 1
    # Always save final
    p = ckpt_dir / f"ckpt_final.pt"
    torch.save({'step': global_step, 'state_dict': model.state_dict()}, p)
    saved.append(p)

    # Cap saved list to last max_ckpts
    if len(saved) > max_ckpts:
        saved = saved[-max_ckpts:]
    return saved


############################################
# Lightweight, self-contained implementations
############################################

@torch.no_grad()
def _get_eval_batch(eval_loader: DataLoader, device: str):
    batch = next(iter(eval_loader))
    x, y = batch
    return x.to(device), y.to(device)


def _criterion_from_cfg(cfg: Dict[str, Any]) -> nn.Module:
    loss_name = cfg['learner'].get('loss', 'cross_entropy')
    if loss_name == 'mse':
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def _model_forward_outputs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward helper that supports models exposing predict().
    Returns the primary output tensor suitable for computing loss.
    """
    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        out = model.predict(x)
        if isinstance(out, tuple) or isinstance(out, list):
            return out[0]
        return out  # type: ignore[return-value]
    return model(x)


def _forward_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    out = _model_forward_outputs(model, x)
    if isinstance(criterion, nn.MSELoss):
        if y.dtype in (torch.long, torch.int64):
            y = torch.nn.functional.one_hot(y, num_classes=out.shape[-1]).float()
        else:
            y = y.float()
    return criterion(out, y)


def _named_params(model: nn.Module, exclude_bn_and_bias: bool = False):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_bn_and_bias:
            lname = name.lower()
            if 'bias' in lname or 'bn' in lname or 'batchnorm' in lname:
                continue
        yield name, p


def _flatten_params(model: nn.Module, exclude_bn_and_bias: bool = False) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
    vecs = []
    shapes: List[Tuple[str, Tuple[int, ...]]] = []
    for name, p in _named_params(model, exclude_bn_and_bias):
        shapes.append((name, tuple(p.shape)))
        vecs.append(p.detach().flatten())
    if len(vecs) == 0:
        return torch.empty(0, device=next(model.parameters()).device), shapes
    return torch.cat(vecs), shapes


def _unflatten_into_state(base_state: Dict[str, torch.Tensor], shapes: List[Tuple[str, Tuple[int, ...]]], vec: torch.Tensor) -> Dict[str, torch.Tensor]:
    out_state = {k: v.clone() for k, v in base_state.items()}
    idx = 0
    for name, shape in shapes:
        numel = int(np.prod(shape))
        segment = vec[idx: idx + numel].view(shape)
        idx += numel
        out_state[name] = segment.to(out_state[name].device, dtype=out_state[name].dtype)
    return out_state


def _get_base_trainable_state(model: nn.Module, exclude_bn_and_bias: bool = False) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    if not exclude_bn_and_bias:
        return state
    # When excluding, we still keep all tensors in state; only direction vectors ignore them.
    return state


def _hvp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, criterion: nn.Module, exclude_bn_and_bias: bool = False) -> torch.Tensor:
    # Compute HVP with model in eval mode for stable curvature (avoid BN/Dropout drift)
    was_training = model.training
    model.eval()
    params = [p for _, p in _named_params(model, exclude_bn_and_bias)]
    loss = _forward_loss(model, x, y, criterion)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
    inner = (flat_grad * v).sum()
    hvps = torch.autograd.grad(inner, params, retain_graph=False)
    flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvps]).detach()
    if was_training:
        model.train()
    return flat_hvp


def _assign_displacement(model: nn.Module, shapes: List[Tuple[str, Tuple[int, ...]]], base_state: Dict[str, torch.Tensor], disp_vec: torch.Tensor) -> None:
    """Set model trainable params to base + displacement (in the order of shapes)."""
    idx = 0
    name_to_param = dict(model.named_parameters())
    for name, shape in shapes:
        numel = int(np.prod(shape))
        segment = disp_vec[idx: idx + numel].view(shape)
        idx += numel
        if name in name_to_param:
            p = name_to_param[name]
            base = base_state[name].to(p.device, dtype=p.dtype)
            p.data.copy_(base + segment.to(p.device, dtype=p.dtype))


def _load_base(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    for k in current.keys():
        if k in base_state:
            current[k].copy_(base_state[k].to(current[k].device, dtype=current[k].dtype))


def get_hessian_axes(model: nn.Module, eval_loader: DataLoader, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Approximate top-2 Hessian eigenvectors of the loss on a fixed batch via power iteration with orthogonalization."""
    device = cfg.get('device', 'cuda:0')
    exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
    criterion = _criterion_from_cfg(cfg)
    x, y = _get_eval_batch(eval_loader, device)

    v0, shapes = _flatten_params(model, exclude_bn)
    dim = v0.numel()
    if dim == 0:
        raise RuntimeError("Model has no trainable parameters for Hessian computation.")
    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)

    iters = int(cfg.get('lla', {}).get('spectrum', {}).get('power_iters', 20))
    t0 = time.time()
    eig1 = 0.0
    for _ in range(iters):
        Hv = _hvp(model, x, y, v, criterion, exclude_bn)
        eig1 = float((v @ Hv).item())
        v = Hv / (Hv.norm() + 1e-12)
    top1 = v.clone()

    # Second vector with deflation
    w = torch.randn(dim, device=device)
    w = w - (w @ top1) * top1
    w = w / (w.norm() + 1e-12)
    eig2 = 0.0
    for _ in range(iters):
        Hw = _hvp(model, x, y, w, criterion, exclude_bn)
        # Project out component along top1
        Hw = Hw - (Hw @ top1) * top1
        eig2 = float((w @ Hw).item())
        w = Hw / (Hw.norm() + 1e-12)
    top2 = w.clone()

    dt = time.time() - t0
    meta = {
        'rayleigh_top1': eig1,
        'rayleigh_top2': eig2,
        'dim': dim,
        'power_iters': iters,
        'algorithm': 'power_iteration+deflation',
        'time_seconds': dt,
    }
    return top1, top2, meta


@torch.no_grad()
def plot_plane(model: nn.Module, base_state: Dict[str, Any], axes: Tuple[torch.Tensor, torch.Tensor], eval_loader: DataLoader, cfg: Dict[str, Any], out_dir: Path, label: str) -> Dict[str, Any]:
    device = cfg.get('device', 'cuda:0')
    criterion = _criterion_from_cfg(cfg)
    x, y = _get_eval_batch(eval_loader, device)
    v1, v2 = axes
    # Normalize directions globally
    v1 = v1 / (v1.norm() + 1e-12)
    v2 = v2 - (v1 @ v2) * v1
    v2 = v2 / (v2.norm() + 1e-12)

    grid_res = int(cfg['lla']['planes'].get('grid_resolution', 41))
    span_cfg = cfg['lla']['planes'].get('span', 1.0)
    if isinstance(span_cfg, (list, tuple)) and len(span_cfg) == 2:
        a0, a1 = float(span_cfg[0]), float(span_cfg[1])
        b0, b1 = a0, a1
    else:
        s = float(span_cfg)
        a0, a1 = -s, s
        b0, b1 = -s, s
    alphas = torch.linspace(a0, a1, grid_res, device=device)
    betas = torch.linspace(b0, b1, grid_res, device=device)

    # Trainable param shapes (for displacement application)
    exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
    _, shapes = _flatten_params(model, exclude_bn)

    losses = torch.zeros(grid_res, grid_res, device=device)
    t0 = time.time()
    # Evaluate point-by-point to keep memory low
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = a * v1 + b * v2
            _load_base(model, base_state)
            _assign_displacement(model, shapes, base_state, vec)
            model.eval()
            l = _forward_loss(model, x, y, criterion)
            losses[i, j] = l.detach()
            # restore base state
            _load_base(model, base_state)
    dt = time.time() - t0
    num_points = grid_res * grid_res
    print(f"[TIMING][plane:{label}] grid_eval_forward_passes={num_points} total_time={dt:.2f}s avg_per_point={dt/num_points:.4f}s")

    losses_np = losses.detach().cpu().numpy()
    npy_path = out_dir / f"plane_{label}.npy"
    np.save(npy_path, losses_np)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(losses_np, origin='lower', extent=[a0, a1, b0, b1], cmap='viridis', aspect='auto')
    ax.set_title(f"Loss plane: {label}")
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png_path = out_dir / f"plane_{label}.png"
    fig.savefig(png_path)
    plt.close(fig)

    # Plot 3D surface
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D backend registered
    except Exception:
        pass
    X, Y = np.meshgrid(np.linspace(a0, a1, grid_res), np.linspace(b0, b1, grid_res), indexing='ij')
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(X, Y, losses_np, cmap='viridis', linewidth=0, antialiased=True)
    ax3d.set_xlabel('alpha')
    ax3d.set_ylabel('beta')
    ax3d.set_zlabel('loss')
    ax3d.set_title(f"Loss plane 3D: {label}")
    fig3d.colorbar(surf, shrink=0.6, aspect=10)
    ax3d.view_init(elev=30, azim=135)
    fig3d.tight_layout()
    png3d_path = out_dir / f"plane_{label}_3d.png"
    fig3d.savefig(png3d_path)
    plt.close(fig3d)

    return {
        'grid_resolution': grid_res,
        'span': {'alpha': [a0, a1], 'beta': [b0, b1]},
        'npy': str(npy_path),
        'png': str(png_path),
    }


def _lanczos_tridiag(hvp_fn, dim: int, m: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run m-step Lanczos to build tridiagonal T. Returns (alpha, beta, steps)."""
    v_prev = torch.zeros(dim, device=device)
    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    alpha = torch.zeros(m, device=device)
    beta = torch.zeros(m-1, device=device) if m > 1 else torch.zeros(0, device=device)
    steps = 0
    for j in range(m):
        w = hvp_fn(v)
        a = (v @ w)
        alpha[j] = a
        if j == 0:
            w = w - a * v
        else:
            w = w - a * v - beta[j-1] * v_prev
        b = w.norm()
        steps = j + 1
        if j < m - 1:
            beta[j] = b
        if b.item() < 1e-8:
            break
        v_prev, v = v, (w / b)
    return alpha[:steps], beta[:max(0, steps-1)], steps


def _esd_via_slq(hvp_fn, dim: int, device: str, m: int = 50, probes: int = 8, grid: int = 200) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stochastic Lanczos Quadrature to approximate empirical spectral density of Hessian.
    Returns (lambda_grid, density, meta)."""
    all_thetas: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []
    lanczos_steps: List[int] = []
    for _ in range(probes):
        alpha, beta, steps = _lanczos_tridiag(hvp_fn, dim, m, device)
        T = torch.diag(alpha)
        if steps > 1:
            off = torch.diag(beta, diagonal=1) + torch.diag(beta, diagonal=-1)
            T = T + off
        evals, evecs = torch.linalg.eigh(T)
        # SLQ weights: square of first row of eigenvectors
        w = (evecs[0, :].abs() ** 2)
        all_thetas.append(evals.detach())
        all_weights.append(w.detach())
        lanczos_steps.append(steps)
    thetas = torch.cat(all_thetas)
    weights = torch.cat(all_weights)
    lam_min = float(thetas.min().item())
    lam_max = float(thetas.max().item())
    span = lam_max - lam_min if lam_max > lam_min else 1.0
    lam_min -= 0.05 * span
    lam_max += 0.05 * span
    lam_grid = np.linspace(lam_min, lam_max, grid, dtype=np.float64)
    sigma = 0.02 * span + 1e-8
    dens = np.zeros_like(lam_grid)
    # Gaussian kernel smoothing
    k = (1.0 / (np.sqrt(2 * np.pi) * sigma))
    th = thetas.detach().cpu().numpy().astype(np.float64)
    wt = weights.detach().cpu().numpy().astype(np.float64)
    for t, w in zip(th, wt):
        dens += w * k * np.exp(-0.5 * ((lam_grid - t) / sigma) ** 2)
    # Normalize to unit area
    area = np.trapz(dens, lam_grid)
    if area > 0:
        dens = dens / area
    meta = {
        'probes': probes,
        'lanczos_m': m,
        'avg_steps': float(np.mean(lanczos_steps)),
        'lam_min': lam_min,
        'lam_max': lam_max,
        'sigma': sigma,
    }
    return lam_grid, dens, meta


def compute_spectrum(model: nn.Module, eval_loader: DataLoader, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    device = cfg.get('device', 'cuda:0')
    criterion = _criterion_from_cfg(cfg)
    x, y = _get_eval_batch(eval_loader, device)
    exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
    v0, _ = _flatten_params(model, exclude_bn)
    dim = v0.numel()
    top_k = int(cfg['lla']['spectrum'].get('top_k', 5))
    k = min(top_k, max(1, dim))

    def hvp_op(v: torch.Tensor) -> torch.Tensor:
        return _hvp(model, x, y, v, criterion, exclude_bn)

    t_topk0 = time.time()
    vecs: List[torch.Tensor] = []
    vals: List[float] = []
    for _ in range(k):
        v = torch.randn(dim, device=device)
        # Orthogonalize against previous
        for u in vecs:
            v = v - (v @ u) * u
        v = v / (v.norm() + 1e-12)
        for _it in range(15):
            Hv = hvp_op(v)
            # Deflate
            for u in vecs:
                Hv = Hv - (Hv @ u) * u
            v = Hv / (Hv.norm() + 1e-12)
        eig = float((v @ hvp_op(v)).item())
        vecs.append(v.detach())
        vals.append(eig)
    t_topk = time.time() - t_topk0

    # Hutchinson trace estimate
    probes = int(cfg['lla']['spectrum'].get('hutchinson_probes', 32))
    t_tr0 = time.time()
    trace_est = 0.0
    for _ in range(probes):
        z = torch.randint(0, 2, (dim,), device=device, dtype=torch.float32) * 2 - 1  # Rademacher
        Hz = hvp_op(z)
        trace_est += float((z @ Hz).item())
    trace_est /= max(1, probes)
    t_tr = time.time() - t_tr0

    # ESD via SLQ
    esd_m = int(cfg['lla']['spectrum'].get('slq_m', 50))
    esd_probes = int(cfg['lla']['spectrum'].get('slq_probes', 8))
    esd_grid = int(cfg['lla']['spectrum'].get('slq_grid', 200))
    t_esd0 = time.time()
    lam_grid, density, esd_meta = _esd_via_slq(hvp_op, dim, device=device, m=esd_m, probes=esd_probes, grid=esd_grid)
    t_esd = time.time() - t_esd0

    # Save ESD
    esd_dir = out_dir
    np.savez(esd_dir / 'spectrum_esd.npz', lambda_grid=lam_grid, density=density, **{k: v for k, v in esd_meta.items() if isinstance(v, (int, float))})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lam_grid, density)
    ax.set_xlabel('eigenvalue')
    ax.set_ylabel('density')
    ax.set_title('Empirical Spectral Density (SLQ)')
    fig.tight_layout()
    esd_png = esd_dir / 'spectrum_esd.png'
    fig.savefig(esd_png)
    plt.close(fig)

    spec = {
        'top_k': vals,
        'hutchinson_trace': trace_est,
        'dim': dim,
        'k_used': k,
        'probes': probes,
        'algorithm_topk': 'power_iteration+deflation',
        'algorithm_trace': 'Hutchinson (Rademacher probes)',
        'algorithm_esd': 'SLQ (Lanczos tridiagonalization + Gaussian kernel)',
        'timing_seconds': {
            'topk': t_topk,
            'hutchinson_trace': t_tr,
            'esd_slq': t_esd,
            'total': t_topk + t_tr + t_esd,
        },
        'esd': {
            'n_grid': esd_grid,
            'probes': esd_probes,
            'm': esd_m,
            **esd_meta,
            'png': str(esd_png),
            'npz': str(esd_dir / 'spectrum_esd.npz'),
        }
    }
    (out_dir / 'spectrum.json').write_text(json.dumps(spec, indent=2))
    # Clear console summary
    print(f"[TIMING][spectrum] dim={dim} topk(k={k})={t_topk:.2f}s hutchinson(probes={probes})={t_tr:.2f}s esd_slq(probes={esd_probes},m={esd_m})={t_esd:.2f}s total={(t_topk+t_tr+t_esd):.2f}s")
    if len(vals) >= 2:
        print(f"[METRICS][spectrum] top2_eigs≈ [{vals[0]:.4g}, {vals[1]:.4g}] (k={k})")
    elif len(vals) == 1:
        print(f"[METRICS][spectrum] top1_eig≈ {vals[0]:.4g} (k={k})")
    return spec


def plot_sam_surface(model: nn.Module, base_state: Dict[str, Any], axes: Tuple[torch.Tensor, torch.Tensor], eval_loader: DataLoader, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # First-order SAM approximation: robust loss ≈ loss(theta) + rho * ||grad_theta||
    device = cfg.get('device', 'cuda:0')
    rho = float(cfg['lla']['sam'].get('rho', 0.05))
    criterion = _criterion_from_cfg(cfg)
    x, y = _get_eval_batch(eval_loader, device)
    v1, v2 = axes
    v1 = v1 / (v1.norm() + 1e-12)
    v2 = v2 - (v1 @ v2) * v1
    v2 = v2 / (v2.norm() + 1e-12)
    grid_res = int(cfg['lla']['planes'].get('grid_resolution', 41))
    span_cfg = cfg['lla']['planes'].get('span', 1.0)
    if isinstance(span_cfg, (list, tuple)) and len(span_cfg) == 2:
        a0, a1 = float(span_cfg[0]), float(span_cfg[1])
        b0, b1 = a0, a1
    else:
        s = float(span_cfg)
        a0, a1 = -s, s
        b0, b1 = -s, s
    alphas = torch.linspace(a0, a1, grid_res, device=device)
    betas = torch.linspace(b0, b1, grid_res, device=device)
    exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
    _, shapes = _flatten_params(model, exclude_bn)

    robust = torch.zeros(grid_res, grid_res, device=device)
    t0 = time.time()
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = a * v1 + b * v2
            _load_base(model, base_state)
            _assign_displacement(model, shapes, base_state, vec)
            model.train()  # need grads for norm
            loss = _forward_loss(model, x, y, criterion)
            params = [p for _, p in _named_params(model, exclude_bn)]
            grads = torch.autograd.grad(loss, params, retain_graph=False)
            grad_norm = torch.cat([g.view(-1) for g in grads]).norm().item()
            robust[i, j] = loss.detach() + rho * grad_norm
            # restore base
            _load_base(model, base_state)
    dt = time.time() - t0
    num_points = grid_res * grid_res
    print(f"[TIMING][sam_surface] grid_eval_points={num_points} rho={rho} total_time={dt:.2f}s avg_per_point={dt/num_points:.4f}s (includes grad norm)")

    arr = robust.detach().cpu().numpy()
    npy_path = out_dir / 'sam_surface.npy'
    np.save(npy_path, arr)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, origin='lower', extent=[a0, a1, b0, b1], cmap='magma', aspect='auto')
    ax.set_title('SAM approx surface')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png_path = out_dir / 'sam_surface.png'
    fig.savefig(png_path)
    plt.close(fig)
    # 3D view for SAM surface
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass
    X, Y = np.meshgrid(np.linspace(a0, a1, grid_res), np.linspace(b0, b1, grid_res), indexing='ij')
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(X, Y, arr, cmap='magma', linewidth=0, antialiased=True)
    ax3d.set_xlabel('alpha')
    ax3d.set_ylabel('beta')
    ax3d.set_zlabel('robust loss')
    ax3d.set_title('SAM approx surface (3D)')
    fig3d.colorbar(surf, shrink=0.6, aspect=10)
    ax3d.view_init(elev=30, azim=135)
    fig3d.tight_layout()
    png3d_path = out_dir / 'sam_surface_3d.png'
    fig3d.savefig(png3d_path)
    plt.close(fig3d)
    return {'npy': str(npy_path), 'png': str(png_path), 'png3d': str(png3d_path), 'rho': rho}


def fit_mode_connectivity(model_fn, thetaA: Dict[str, Any], thetaB: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Minimal baseline: midpoint is simple average. Return as both midpoint and info.
    thetaM: Dict[str, Any] = {}
    for k in thetaA.keys():
        if k in thetaB:
            a = thetaA[k]
            b = thetaB[k]
            thetaM[k] = ((a + b) / 2.0).to(a.device)
        else:
            thetaM[k] = thetaA[k]
    info = {'strategy': 'average_midpoint'}
    (out_dir / 'mode_fit.json').write_text(json.dumps(info, indent=2))
    return thetaM, info


@torch.no_grad()
def plot_mode_curve(model_fn, thetaA: Dict[str, Any], thetaM: Dict[str, Any], thetaB: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # Evaluate quadratic Bézier curve loss using a fixed batch
    device = cfg.get('device', 'cuda:0')
    criterion = _criterion_from_cfg(cfg)
    # Build a fresh model for evaluation and a batch
    # model_fn should create a new model
    model = model_fn()
    model.to(device)
    # Need eval_loader; simplest is to rebuild data quickly
    # Assumption: same dataset config
    from src.data_loading.transform_factory import transform_factory as _tf
    from src.data_loading.dataset_factory import dataset_factory as _df
    from configs.configurations import DataConfig as _DataCfg
    transform = _tf(cfg['data']['dataset'], cfg['net']['type'])
    train_set, _ = _df(_DataCfg(**cfg['data']), transform=transform, with_testset=False)
    eval_loader = DataLoader(train_set, batch_size=cfg['lla']['evaluation_data'].get('eval_batch_size', 256), shuffle=False, num_workers=0)
    x, y = _get_eval_batch(eval_loader, device)

    t_points = int(cfg['lla']['mode_connectivity'].get('curve_points', 21))
    ts = torch.linspace(0, 1, t_points, device=device)
    losses: List[float] = []
    t0 = time.time()
    for t in ts:
        # Quadratic Bézier
        state: Dict[str, torch.Tensor] = {}
        for k in thetaA.keys():
            a = thetaA[k].to(device)
            m = thetaM.get(k, a).to(device)
            b = thetaB.get(k, a).to(device)
            state[k] = ((1 - t) ** 2) * a + 2 * (1 - t) * t * m + (t ** 2) * b
        model.load_state_dict({**model.state_dict(), **state}, strict=False)
        model.eval()
        l = _forward_loss(model, x, y, criterion)
        losses.append(float(l.item()))
    dt = time.time() - t0
    print(f"[TIMING][mode_curve] points={t_points} total_time={dt:.2f}s avg_per_point={dt/max(1,t_points):.4f}s")
    arr = np.array(losses, dtype=np.float32)
    npy_path = out_dir / 'mode_curve.npy'
    np.save(npy_path, arr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.linspace(0, 1, t_points), arr, marker='o')
    ax.set_xlabel('t')
    ax.set_ylabel('loss')
    ax.set_title('Mode connectivity (quadratic Bézier)')
    fig.tight_layout()
    png_path = out_dir / 'mode_curve.png'
    fig.savefig(png_path)
    plt.close(fig)
    return {'npy': str(npy_path), 'png': str(png_path)}


def _gini(x: torch.Tensor) -> float:
    x = x.flatten().abs().sort().values
    n = x.numel()
    if n == 0:
        return 0.0
    cumx = torch.cumsum(x, dim=0)
    g = (n + 1 - 2 * (cumx.sum() / x.sum())) / n if x.sum() > 0 else 0.0
    return float(g)


def compute_rank_stats_at_checkpoints(model_factory_cfg: Dict[str, Any], checkpoint_paths: List[Path], cfg: Dict[str, Any], eval_loader: DataLoader) -> Dict[str, Any]:
    # Uses per-layer SVD summaries (approximate rank with prop) and Gini of singular values
    device = cfg.get('device', 'cuda:0')
    # rank_tracking config moved to top-level; keep a backward-compatible fallback
    rt_cfg = cfg.get('rank_tracking', cfg.get('lla', {}).get('rank_tracking', {}))
    prop = float(rt_cfg.get('approximate_rank_prop', 0.99))
    from src.models.model_factory import model_factory as _mf
    # Wrap net dict into NetConfig
    _netparams = NetParams(**model_factory_cfg.get('netparams', {}))
    _netcfg = NetConfig(type=model_factory_cfg['type'], netparams=_netparams, network_class=model_factory_cfg.get('network_class'))
    model = _mf(_netcfg)
    model.to(device)
    criterion = _criterion_from_cfg(cfg)
    x, y = _get_eval_batch(eval_loader, device)

    from src.utils.miscellaneous import compute_matrix_rank_summaries  # type: ignore

    per_ckpt: List[Dict[str, Any]] = []
    for p in checkpoint_paths:
        state = torch.load(p, map_location=device)
        state_dict = state.get('state_dict', state)
        model.load_state_dict({**model.state_dict(), **state_dict}, strict=False)
        model.eval()
        loss_val = float(_forward_loss(model, x, y, criterion).item())
        layer_stats: List[Dict[str, Any]] = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                W = mod.weight.detach()
                M = W  # (out,in)
            elif isinstance(mod, nn.Conv2d):
                W = mod.weight.detach()
                M = W.view(W.shape[0], -1)  # out, in*k*k
            else:
                continue
            try:
                rank, eff_rank, approx_rank, approx_abs = compute_matrix_rank_summaries(M, prop=prop)
                sv = torch.linalg.svdvals(M)
                gini = _gini(sv)
                layer_stats.append({
                    'layer': name,
                    'rank': int(rank.item()),
                    'effective_rank': float(eff_rank.item()),
                    'approx_rank': int(approx_rank.item()),
                    'approx_abs': int(approx_abs.item()),
                    'gini_sv': gini,
                    'shape': list(M.shape),
                })
            except Exception as e:
                layer_stats.append({'layer': name, 'error': str(e)})
        per_ckpt.append({'checkpoint': str(p), 'loss': loss_val, 'layers': layer_stats})

    out = {'checkpoints': per_ckpt}
    return out


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="LLA pipeline")
    parser.add_argument('--config', type=str, default=str(Path(__file__).with_name('cfg').joinpath('config.yaml')))
    parser.add_argument('--task', type=str, default='all', choices=['all','planes','spectrum','sam','mode','rank'])
    parser.add_argument('--outdir', type=str, default=str(Path(__file__).with_name('results')))
    args = parser.parse_args()

    # Load config via YAML directly (keep it light to avoid adding Hydra dependency here)
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg.get('seed', None))

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare data/model
    with timer_block('prepare_data_and_model'):
        train_loader, eval_loader, model = prepare_data_and_model(cfg)

    # Short fine-tune and checkpoints
    with timer_block('quick_finetune_and_checkpoint'):
        ckpts = quick_finetune_and_checkpoint(cfg, model, train_loader, out_root)

    # Save a minimal manifest
    manifest = {
        'num_checkpoints': len(ckpts),
        'checkpoints': [str(p) for p in ckpts],
        'model': cfg['net']['type'],
        'dataset': cfg['data']['dataset'],
    }
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # Reload last checkpoint as base
    last_ckpt = ckpts[-1]
    base = torch.load(last_ckpt, map_location=cfg.get('device', 'cuda:0'))
    base_state = base.get('state_dict', base)
    model.load_state_dict({**model.state_dict(), **base_state}, strict=False)

    # Execute tasks based on selection
    tasks_to_run = [args.task] if args.task != 'all' else ['planes', 'spectrum', 'sam', 'mode', 'rank']

    if 'planes' in tasks_to_run:
        with timer_block('hessian_axes_and_plane'):
            v1, v2, meta = get_hessian_axes(model, eval_loader, cfg)
            (out_root / 'hessian_axes_meta.json').write_text(json.dumps(meta, indent=2))
            print(f"[TIMING][hessian_axes] algorithm={meta.get('algorithm')} power_iters={meta.get('power_iters')} dim={meta.get('dim')} time={meta.get('time_seconds'):.2f}s eigs≈[{meta.get('rayleigh_top1'):.4g},{meta.get('rayleigh_top2'):.4g}]")
            plane_dir = out_root / 'planes'
            plane_dir.mkdir(exist_ok=True)
            plot_plane(model, base_state, (v1, v2), eval_loader, cfg, plane_dir, label='hessian')

        # Also: random directions around the trained base
        with timer_block('random_plane_directions'):
            device = cfg.get('device', 'cuda:0')
            exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
            v0, _ = _flatten_params(model, exclude_bn)
            dim = v0.numel()
            if dim > 0:
                r1 = torch.randn(dim, device=device); r1 = r1/(r1.norm()+1e-12)
                r2 = torch.randn(dim, device=device); r2 = r2 - (r1@r2)*r1; r2 = r2/(r2.norm()+1e-12)
                plane_dir = out_root / 'planes'
                plane_dir.mkdir(exist_ok=True)
                plot_plane(model, base_state, (r1, r2), eval_loader, cfg, plane_dir, label='random_dirs')

        # And: plane around a freshly initialized ConvNet (random base)
        with timer_block('random_base_plane'):
            try:
                _netparams = NetParams(**cfg['net'].get('netparams', {}))
                _netcfg = NetConfig(type=cfg['net']['type'], netparams=_netparams, network_class=cfg['net'].get('network_class'))
                fresh = model_factory(_netcfg)
                fresh.to(cfg.get('device', 'cuda:0'))
                rand_state = fresh.state_dict()
                device = cfg.get('device', 'cuda:0')
                exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
                v0, _ = _flatten_params(model, exclude_bn)
                dim = v0.numel()
                if dim > 0:
                    r1 = torch.randn(dim, device=device); r1 = r1/(r1.norm()+1e-12)
                    r2 = torch.randn(dim, device=device); r2 = r2 - (r1@r2)*r1; r2 = r2/(r2.norm()+1e-12)
                    plane_dir = out_root / 'planes'
                    plane_dir.mkdir(exist_ok=True)
                    plot_plane(model, rand_state, (r1, r2), eval_loader, cfg, plane_dir, label='random_base')
            except Exception as e:
                print(f"[warn] random_base_plane skipped: {e}")

    if 'spectrum' in tasks_to_run:
        with timer_block('spectrum'):
            compute_spectrum(model, eval_loader, cfg, out_root)

    if 'sam' in tasks_to_run:
        with timer_block('sam_surface'):
            # reuse axes if computed, else compute random axes
            try:
                v1, v2, meta = get_hessian_axes(model, eval_loader, cfg)
                print(f"[TIMING][sam_surface] axes_source=hessian algorithm={meta.get('algorithm')} power_iters={meta.get('power_iters')} time={meta.get('time_seconds'):.2f}s")
            except Exception:
                device = cfg.get('device', 'cuda:0')
                exclude_bn = bool(cfg['lla']['planes'].get('bn', {}).get('exclude_bn_and_bias', False))
                v0, _ = _flatten_params(model, exclude_bn)
                dim = v0.numel()
                v1 = torch.randn(dim, device=device); v1 = v1 / (v1.norm()+1e-12)
                v2 = torch.randn(dim, device=device); v2 = v2 - (v1@v2)*v1; v2 = v2/(v2.norm()+1e-12)
                print("[TIMING][sam_surface] axes_source=random time≈0.00s")
            plot_sam_surface(model, base_state, (v1, v2), eval_loader, cfg, out_root)

    if 'mode' in tasks_to_run:
        with timer_block('mode_connectivity'):
            # Define a factory to rebuild the same model
            def model_fn():
                _netparams = NetParams(**cfg['net'].get('netparams', {}))
                _netcfg = NetConfig(type=cfg['net']['type'], netparams=_netparams, network_class=cfg['net'].get('network_class'))
                m = model_factory(_netcfg)
                return m
            # For A and B, we pick the first and last checkpoints
            A_state = torch.load(ckpts[0], map_location=cfg.get('device', 'cuda:0'))
            A = A_state.get('state_dict', A_state)
            B_state = torch.load(ckpts[-1], map_location=cfg.get('device', 'cuda:0'))
            B = B_state.get('state_dict', B_state)
            thetaM, _info = fit_mode_connectivity(model_fn, A, B, cfg, out_root)
            plot_mode_curve(model_fn, A, thetaM, B, cfg, out_root)

    if 'rank' in tasks_to_run:
        with timer_block('rank_tracking'):
            rank_stats = compute_rank_stats_at_checkpoints(cfg['net'], ckpts, cfg, eval_loader)
            (out_root / 'rank_stats.json').write_text(json.dumps(rank_stats, indent=2))

    print("LLA pipeline run complete.")


if __name__ == '__main__':
    main()
