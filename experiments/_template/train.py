"""
Minimal experiment entrypoint template.

This script demonstrates the standard wiring pattern:
- Hydra-configured ExperimentConfig read from cfg/config.yaml
- Transform selection via transform_factory
- Dataset construction via dataset_factory (+ optional wrappers)
- Model via model_factory, learner via create_learner
- Simple train loop with loss/accuracy reporting
"""

from typing import Any
import os
import random
import pathlib

import torch
import torch.nn.functional as F
import numpy as np

import hydra
from omegaconf import OmegaConf

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Ensure project root on sys.path for absolute imports
import sys
sys.path.append(str(PROJECT_ROOT))

from configs.configurations import ExperimentConfig
from src.data_loading.transform_factory import transform_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.shifting_dataset import (
    create_stateful_dataset_wrapper,
    create_stateless_dataset_wrapper,
)
from src.models.model_factory import model_factory
from src.algos.supervised.supervised_factory import create_learner


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> Any:
    # Seed
    if cfg.seed is None or not isinstance(cfg.seed, (int, float)):
        cfg.seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {cfg.seed}")
    random.seed(cfg.seed)
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(OmegaConf.to_yaml(cfg))

    # Transforms and datasets
    transform = transform_factory(cfg.data.dataset, cfg.net.type)
    train_set, test_set = dataset_factory(cfg.data, transform=transform, with_testset=bool(getattr(cfg.evaluation, 'use_testset', False)))

    # Infer image dims for conv nets if not specified
    if hasattr(cfg.net, 'netparams') and cfg.net.netparams is not None:
        try:
            sample_x = train_set[0][0]
            if isinstance(sample_x, torch.Tensor) and sample_x.ndim >= 3:
                if cfg.net.netparams.input_height is None:
                    cfg.net.netparams.input_height = int(sample_x.shape[-2])
                if cfg.net.netparams.input_width is None:
                    cfg.net.netparams.input_width = int(sample_x.shape[-1])
        except Exception:
            pass

    # Model and learner
    net = model_factory(cfg.net)
    net.to(cfg.net.device)
    learner = create_learner(cfg.learner, net, cfg.net)

    # Task-shift wrappers
    shift_mode = getattr(cfg, 'task_shift_mode', None)
    is_stateful = shift_mode in ["drifting_values", "continuous_input_deformation"]
    dataset_wrapper = create_stateful_dataset_wrapper(cfg, train_set) if is_stateful else None

    # Dataloader workers
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

    # Loss selection
    loss_name = getattr(cfg.learner, 'loss', 'cross_entropy')
    if loss_name == 'cross_entropy':
        loss_fn = F.cross_entropy
    elif loss_name == 'mse':
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    net.train()
    total_epochs = int(cfg.epochs)
    runs = int(getattr(cfg, 'runs', 1) or 1)
    for run in range(runs):
        if is_stateful and hasattr(dataset_wrapper, 'time_step'):
            print(f"Starting run {run} at wrapper time_step={dataset_wrapper.time_step}")
        for task_idx in range(int(getattr(cfg, 'num_tasks', 1) or 1)):
            if is_stateful:
                if hasattr(dataset_wrapper, 'update_task'):
                    dataset_wrapper.update_task()
                current_train = dataset_wrapper
            else:
                current_train = create_stateless_dataset_wrapper(cfg, train_set, task_idx) or train_set

            loader = torch.utils.data.DataLoader(
                current_train,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                worker_init_fn=_seed_worker if num_workers > 0 else None,
            )

            for epoch in range(total_epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                for batch_idx, (x, y) in enumerate(loader):
                    x = x.to(cfg.net.device)

                    # Drifting-values regression path
                    if shift_mode == 'drifting_values':
                        drifting_values, original_labels = y
                        drifting_values = drifting_values.to(cfg.net.device)
                        original_labels = original_labels.to(cfg.net.device)
                        if hasattr(learner, 'learn_from_partial_values'):
                            loss, output = learner.learn_from_partial_values(x, drifting_values, original_labels)
                        else:
                            # Fallback to direct regression step
                            output = net(x)
                            loss = loss_fn(output.squeeze(), drifting_values)
                            loss.backward()
                            # If learner has optimizer, step it; otherwise skip
                            if hasattr(learner, 'optimizer'):
                                learner.optimizer.step(); learner.optimizer.zero_grad(set_to_none=True)
                    else:
                        y = y.to(cfg.net.device)
                        loss, output = learner.learn(x, y)

                        with torch.no_grad():
                            _, pred = torch.max(output, dim=1)
                            epoch_correct += pred.eq(y).sum().item()

                    epoch_total += x.size(0)
                    epoch_loss += float(loss if not torch.is_tensor(loss) else loss.item())

                if is_stateful and hasattr(dataset_wrapper, 'update_drift'):
                    dataset_wrapper.update_drift()

                metrics = {
                    'run': run,
                    'task': task_idx,
                    'epoch': epoch,
                    'loss': epoch_loss / max(1, len(loader)),
                }
                if shift_mode != 'drifting_values':
                    metrics['accuracy'] = epoch_correct / max(1, epoch_total)
                print(metrics)


if __name__ == "__main__":
    # Clear any cached CUDA memory before starting
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    main()
