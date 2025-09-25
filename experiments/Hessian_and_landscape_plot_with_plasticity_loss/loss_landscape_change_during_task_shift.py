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
from typing import Any, Dict, List, Tuple
import hashlib

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
from src.utils.miscellaneous import compute_matrix_rank_summaries  # type: ignore

# Import LLA utilities (pure, no side-effects)
from experiments.Hessian_and_landscape_plot_with_plasticity_loss.lla_pipeline import (
    get_hessian_axes,
    plot_plane,
    compute_spectrum,
    plot_sam_surface,
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


def _gini(x: torch.Tensor) -> float:
    x = x.flatten().abs().sort().values
    n = x.numel()
    if n == 0:
        return 0.0
    cumx = torch.cumsum(x, dim=0)
    g = (n + 1 - 2 * (cumx.sum() / x.sum())) / n if x.sum() > 0 else 0.0
    return float(g)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge src into dst (in place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _collect_weight_stats(model: nn.Module) -> Dict[str, float]:
    l2 = 0.0
    l1 = 0.0
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            l2 += float(p.detach().float().norm(p=2).item())
            l1 += float(p.detach().float().abs().sum().item())
            count += p.numel()
    return {
        'weight_l2_total': l2,
        'weight_l1_total': l1,
        'weight_params': float(count),
        'weight_l2_avg_per_param': l2 / max(1.0, float(count)),
        'weight_l1_avg_per_param': l1 / max(1.0, float(count)),
    }


def _collect_layer_rank_gini(model: nn.Module, prop: float = 0.99) -> Dict[str, float]:
    ranks: List[float] = []
    ginis: List[float] = []
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
            g = _gini(sv)
            ranks.append(float(approx_rank.item()))
            ginis.append(float(g))
        except Exception:
            continue
    return {
        'layer_rank_mean': float(np.mean(ranks)) if ranks else 0.0,
        'layer_rank_median': float(np.median(ranks)) if ranks else 0.0,
        'layer_gini_mean': float(np.mean(ginis)) if ginis else 0.0,
        'layer_count': float(len(ranks)),
    }


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


def _read_yaml_bytes_for_run_id() -> bytes:
    """Read the task-shift YAML to compute a stable run_id. Returns bytes; empty if read fails."""
    yaml_path = Path(__file__).with_name('cfg') / 'task_shift_config.yaml'
    try:
        return yaml_path.read_bytes()
    except Exception:
        return b''


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


def _run_lla_end_of_task(
    model: nn.Module,
    train_set,
    base_cfg: ExperimentConfig,
    out_root: Path,
    task_idx: int,
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

    # LLA cfg & output directory for this task
    lla_cfg = _build_lla_cfg(base_cfg, eval_batch_size=eval_bs)
    task_dir = out_root / f"task_{task_idx:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Criterion for losses (same as lla pipeline)
    criterion = _criterion_from_cfg(lla_cfg)
    x, y = _get_eval_batch(eval_loader, device)

    result: Dict[str, Any] = {}

    # 1) Hessian axes and plane
    if do_planes or do_sam:
        v1, v2, meta = get_hessian_axes(model, eval_loader, lla_cfg)
        (task_dir / 'hessian_axes_meta.json').write_text(json.dumps(meta, indent=2))
        # Freeze base state to avoid aliasing with live params during plane eval
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if do_planes:
            planes_dir = task_dir / 'planes'
            planes_dir.mkdir(exist_ok=True)
            plane_info = plot_plane(model, base_state, (v1, v2), eval_loader, lla_cfg, planes_dir, label='hessian')
            result['plane'] = plane_info

        if do_sam:
            sam_info = plot_sam_surface(model, base_state, (v1, v2), eval_loader, lla_cfg, task_dir)
            result['sam'] = sam_info

    # 2) Spectrum (top-k, Hutchinson, ESD)
    if do_spectrum:
        spec_info = compute_spectrum(model, eval_loader, lla_cfg, task_dir)
        result['spectrum'] = spec_info

    # Back to train mode handled by caller
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

    # Derive run_id from YAML content and set output root to outputs/loss_landscape_taskshift/<run_id>
    yaml_bytes = _read_yaml_bytes_for_run_id()
    if not yaml_bytes:
        try:
            yaml_bytes = OmegaConf.to_yaml(cfg).encode('utf-8')
        except Exception:
            yaml_bytes = b''
    h = hashlib.sha256(yaml_bytes).digest()
    run_id = str(int.from_bytes(h[:8], byteorder='big', signed=False))
    out_root = PROJECT_ROOT / 'outputs' / 'loss_landscape_taskshift' / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    # Save a copy of the YAML used
    try:
        if yaml_bytes:
            (out_root / 'task_shift_config.yaml').write_bytes(yaml_bytes)
        else:
            (out_root / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg))
    except Exception as e:
        print(f"[warn] failed to save YAML copy: {e}")

    # wandb (numeric only)
    use_wandb = bool(getattr(cfg, 'use_wandb', False))
    if use_wandb:
        try:
            import wandb  # type: ignore
            run_cfg = {
                'seed': int(cfg.seed),
                'arch': str(cfg.net.type),
                'task_shift_mode': str(cfg.task_shift_mode),
                'run_id': run_id,
            }
            wandb.init(project=str(cfg.wandb.project), config=run_cfg)
            try:
                wandb.log({'run_id': int(run_id)}, step=0)  # type: ignore[attr-defined]
            except Exception:
                pass
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
            # Layer-wise rank/gini and weight norms
            rank_cfg_prop = float(getattr(cfg, 'prop_for_approx_or_l1_rank', 0.99)) if hasattr(cfg, 'prop_for_approx_or_l1_rank') else 0.99
            layer_summ = _collect_layer_rank_gini(net, prop=rank_cfg_prop)
            weight_summ = _collect_weight_stats(net)

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
                do_planes=do_planes,
                do_sam=do_sam,
                do_spectrum=do_spectrum,
            )
            dt = time.time() - t0

            # Parse top-k from saved spectrum.json if present
            topk_metrics: Dict[str, float] = {}
            spec_json = out_root / f"task_{task_idx:04d}" / 'spectrum.json'
            if not spec_json.exists():
                # try renamed variant
                for p in (out_root / f"task_{task_idx:04d}").glob('spectrum*.json'):
                    spec_json = p
                    break
            try:
                if spec_json.exists() and do_spectrum:
                    spec = json.loads(spec_json.read_text())
                    topk = spec.get('top_k', [])
                    for i, v in enumerate(topk):
                        topk_metrics[f'hessian_top_eig_{i+1}'] = float(v)
            except Exception:
                pass

            # Log numeric metrics to wandb (images/checkpoints are NOT logged)
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    rank_drop_gini = None
                    rank_mean_delta = None
                    if prev_layer_gini_mean is not None:
                        rank_drop_gini = float(prev_layer_gini_mean - layer_summ.get('layer_gini_mean', 0.0))
                    if prev_layer_rank_mean is not None:
                        rank_mean_delta = float(layer_summ.get('layer_rank_mean', 0.0) - prev_layer_rank_mean)
                    log_data: Dict[str, float | int | str] = {
                        'task_idx': task_idx,
                        'arch': str(cfg.net.type),
                        'run_id': int(run_id),
                        'lla_runtime_sec': dt,
                        **layer_summ,
                        **weight_summ,
                        **topk_metrics,
                    }
                    if last_epoch_loss_mean is not None:
                        log_data['task_last_epoch_loss'] = float(last_epoch_loss_mean)
                    if last_epoch_accuracy is not None:
                        log_data['task_last_epoch_accuracy'] = float(last_epoch_accuracy)
                    if rank_drop_gini is not None:
                        log_data['rank_drop_gini'] = rank_drop_gini
                    if rank_mean_delta is not None:
                        log_data['layer_rank_mean_delta'] = rank_mean_delta
                    # align end-of-task metrics with the global epoch timeline
                    end_of_task_step = int((task_idx + 1) * epochs_per_task - 1)
                    log_data['global_epoch'] = end_of_task_step
                    wandb.log(log_data, step=end_of_task_step)  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

            # proceed to next task
            prev_layer_gini_mean = layer_summ.get('layer_gini_mean', None)
            prev_layer_rank_mean = layer_summ.get('layer_rank_mean', None)

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
