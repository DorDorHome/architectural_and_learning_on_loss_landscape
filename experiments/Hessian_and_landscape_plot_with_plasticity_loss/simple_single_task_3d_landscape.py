"""
Simple single-task training + 3D loss landscape plot

Purpose:
- Run a standard training loop (no task shifts, no wandb), then compute Hessian axes
  and plot the 2D/3D loss landscape plane using the exact helper functions/imports
  used in loss_landscape_change_during_task_shift.py.
- Save all artifacts into a separate output folder.

Notes:
- Uses the same dataset/model/learner factories and lla_pipeline helpers.
- Evaluates the landscape using the exact last training batch for stable center.
"""
from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Repo root for imports (match task-shift script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import OmegaConf

# Factories and utils from the repo (match task-shift script)
from configs.configurations import ExperimentConfig  # type: ignore
from src.models.model_factory import model_factory  # type: ignore
from src.data_loading.dataset_factory import dataset_factory  # type: ignore
from src.data_loading.transform_factory import transform_factory  # type: ignore
from src.algos.supervised.supervised_factory import create_learner  # type: ignore

# Import LLA utilities (pure, no side-effects)
from experiments.Hessian_and_landscape_plot_with_plasticity_loss.lla_pipeline import (
    get_hessian_axes,
    plot_plane,
    _get_eval_batch,
    _criterion_from_cfg,
)  # type: ignore


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge src into dst (in place) and return dst. (same logic as task-shift script)"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _build_lla_cfg(base_cfg: ExperimentConfig, eval_batch_size: int = 128) -> Dict[str, Any]:
    """Construct a minimal LLA config dict expected by lla_pipeline functions.
    Keeps moderate defaults to avoid slowdowns. Mirrors task-shift script behavior.
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
        },
    }
    # Allow evaluation-only overrides from YAML under cfg.lla
    try:
        user_lla = getattr(base_cfg, 'lla', None)
        if user_lla is not None:
            user_lla_dict = OmegaConf.to_container(user_lla, resolve=True)  # type: ignore[arg-type]
            if isinstance(user_lla_dict, dict):
                _deep_update(lla_cfg['lla'], user_lla_dict)
    except Exception:
        pass
    return lla_cfg


def _snapshot_eval_dataset(train_set):
    """Best-effort snapshot/freeze of the dataset for evaluation.
    Tries deepcopy; probes common freeze/eval methods if exposed.
    """
    ds = train_set
    try:
        import copy
        ds = copy.deepcopy(train_set)
    except Exception:
        ds = train_set
    for name, args in [
        ('freeze', ()),
        ('set_eval_mode', (True,)),
        ('disable_drift', (True,)),
        ('set_drift_enabled', (False,)),
        ('set_training', (False,)),
        ('eval', ()),
    ]:
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


def _seed_worker(worker_id: int):
    """Module-level worker init fn so it can be pickled by multiprocessing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    # num_workers logic similar to reference
    try:
        num_workers = int(os.cpu_count() or 0) if str(cfg.num_workers).lower() == "auto" else int(cfg.num_workers)
    except Exception:
        num_workers = 0

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    # Train for cfg.epochs epochs; capture last batch
    last_batch_inp_cpu: torch.Tensor | None = None
    last_batch_target_cpu: Any | None = None
    with tqdm(total=int(cfg.epochs), desc="Epochs", position=0, leave=True, dynamic_ncols=True) as pbar_epochs:
        for epoch in range(int(cfg.epochs)):
            epoch_loss = 0.0
            for (inp, target) in train_loader:
                inp = inp.to(cfg.net.device)
                # If target is a tuple of two tensors (drifting-style), use partial-values learner path
                if isinstance(target, tuple) and len(target) == 2:
                    drifting_values, original_labels = target
                    drifting_values = drifting_values.to(cfg.net.device)
                    original_labels = original_labels.to(cfg.net.device)
                    loss, _ = learner.learn_from_partial_values(inp, drifting_values, original_labels)
                    last_batch_inp_cpu = inp.detach().cpu()
                    last_batch_target_cpu = (
                        drifting_values.detach().cpu(),
                        original_labels.detach().cpu(),
                    )
                else:
                    target = target.to(cfg.net.device)
                    loss, _ = learner.learn(inp, target)
                    last_batch_inp_cpu = inp.detach().cpu()
                    last_batch_target_cpu = target.detach().cpu()
                epoch_loss += float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
            pbar_epochs.set_postfix(loss=f"{epoch_loss / max(1, len(train_loader)):.4f}")
            pbar_epochs.update(1)

    # Evaluate landscape using the exact last training batch (fallback: snapshot of train_set)
    net.eval()
    if last_batch_inp_cpu is not None and last_batch_target_cpu is not None:
        eval_ds = _SingleBatchDataset(last_batch_inp_cpu, last_batch_target_cpu)
    else:
        eval_ds = _snapshot_eval_dataset(train_set)
    eval_bs = min(128, len(eval_ds)) if hasattr(eval_ds, '__len__') else 128
    eval_loader = DataLoader(eval_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=True)

    # LLA cfg and baseline center loss for debugging
    lla_cfg = _build_lla_cfg(cfg, eval_batch_size=eval_bs)
    criterion = _criterion_from_cfg(lla_cfg)
    xb, yb = _get_eval_batch(eval_loader, str(cfg.net.device))
    with torch.no_grad():
        logits = net(xb)
        try:
            base_center_loss = float(criterion(logits, yb).item())
        except Exception:
            # If yb is a tuple (drifting_values, original_labels), try classification on original labels
            if isinstance(yb, (list, tuple)) and len(yb) == 2:
                base_center_loss = float(criterion(logits, yb[1]).item())
            else:
                base_center_loss = float('nan')

    # Output root (separate folder)
    arch = str(cfg.net.type)
    out_root = PROJECT_ROOT / 'outputs' / 'loss_landscape_single_debug' / arch
    out_root.mkdir(parents=True, exist_ok=True)
    planes_dir = out_root / 'planes'
    planes_dir.mkdir(parents=True, exist_ok=True)

    # Compute Hessian axes and plot plane (2D + 3D)
    t0 = time.time()
    v1, v2, meta = get_hessian_axes(net, eval_loader, lla_cfg)
    (out_root / 'hessian_axes_meta.json').write_text(__import__('json').dumps(meta, indent=2))
    # IMPORTANT: pass a frozen copy of state_dict to avoid aliasing live params
    base_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    plane_info = plot_plane(net, base_state, (v1, v2), eval_loader, lla_cfg, planes_dir, label='hessian')
    dt = time.time() - t0

    print(f"Loss landscape (plane) computed in {dt:.2f}s.")
    if isinstance(plane_info, dict):
        png3d = plane_info.get('png3d')
        png2d = plane_info.get('png')
        npy = plane_info.get('npy')
        if png3d:
            print(f"3D plot saved to: {png3d}")
        if png2d:
            print(f"2D plot saved to: {png2d}")
        if npy:
            print(f"Raw plane data saved to: {npy}")
            # Compare center value in plane with direct baseline loss
            try:
                import numpy as _np
                arr = _np.load(npy, allow_pickle=True)
                Z = arr if isinstance(arr, _np.ndarray) else (arr.item().get('Z') if hasattr(arr, 'item') else arr)
                if isinstance(Z, _np.ndarray) and Z.ndim >= 2:
                    h, w = Z.shape
                    center_val = float(Z[h//2, w//2])
                    print(f"[DEBUG] baseline_center_loss={base_center_loss:.6f} vs plane_center={center_val:.6f}")
            except Exception as e:
                print(f"[DEBUG] failed to compare center values: {e}")


if __name__ == "__main__":
    # Optional: ensure CUDA multiprocessing safety in some environments
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("multiprocessing start method set to 'spawn'.")
    except Exception:
        pass
    main()
