"""Quick validation for Step 8 rank tracking integration.
Runs a 1-epoch tiny training loop with track_rank=True on a small subset of CIFAR10.
"""
import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf
from copy import deepcopy
from configs.configurations import ExperimentConfig, NetConfig, NetParams, BackpropConfig, EvaluationConfig, DataConfig
from src.models.model_factory import model_factory
from src.algos.supervised.basic_backprop import Backprop
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory


def main():
    exp = ExperimentConfig(
        epochs=1,
        batch_size=32,
        data=DataConfig(dataset='cifar10', data_path='/hdda/datasets', use_torchvision=True, shuffle=True),
        net=NetConfig(type='ConvNet', netparams=NetParams(num_classes=10, in_channels=3, input_height=None, input_width=None, activation='relu')),
        learner=BackpropConfig(type='backprop'),
        evaluation=EvaluationConfig(use_testset=False, eval_freq_epoch=1),
        track_rank=True,
    )
    # Convert dataclass experiment config to OmegaConf for pretty print
    exp_cfg = OmegaConf.create({
        'epochs': exp.epochs,
        'batch_size': exp.batch_size,
        'track_rank': exp.track_rank,
        'data': exp.data.__dict__,
        'net': {
            'type': exp.net.type,
            'netparams': exp.net.netparams.__dict__ if exp.net.netparams else None
        },
        'learner': exp.learner.__dict__,
        'evaluation': exp.evaluation.__dict__ if exp.evaluation else None
    })
    print("Config:\n", OmegaConf.to_yaml(exp_cfg))
    net = model_factory(exp.net)
    learner = Backprop(net, exp.learner)
    transform = transform_factory(exp.data.dataset, exp.net.type)
    trainset, _ = dataset_factory(exp.data, transform)
    # reduce dataset for speed
    subset_indices = list(range(0, 256))
    train_subset = torch.utils.data.Subset(trainset, subset_indices)
    loader = torch.utils.data.DataLoader(train_subset, batch_size=exp.batch_size, shuffle=True)
    net.train()
    for epoch in range(exp.epochs):
        for inp, lab in loader:
            inp, lab = inp.to(exp.device), lab.to(exp.device)
            loss, out = learner.learn(inp, lab)
        # emulate evaluation block from single_run
        if epoch % exp.evaluation.eval_freq_epoch == 0:
            with torch.no_grad():
                pred, feats = net.predict(inp)
                print(f"Collected {len(feats)} feature tensors for rank tracking test.")
    print("Rank tracking validation script completed (check for no exceptions).")

if __name__ == '__main__':
    main()
