import pytest

# from dataclasses import is_dataclass



import sys
from pathlib import Path
from tests.conftest import PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# cs is an instance of hydras ConfigStore:
from configs.configurations import ExperimentConfig, cs

from src.models.conv_net import ConvNet
from src.models.model_factory import model_factory

# import the dataclass for model params:
#from configs.configurations import NetParams

import torch
import torch.nn as nn

# hydra imports:
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore


def test_conv_net():
    with initialize(config_path='../../experiments/basic_training/cfg', version_base="1.1"):
        cfg = compose(config_name="basic_config")
    print(OmegaConf.to_yaml(cfg))
    netparams = cfg.experiments.net.netparams
    model = ConvNet(netparams)
    assert isinstance(model, nn.Module)
    print('convnet is of instance torch.nn.Module')
    # model = ConvNet()
    # assert is_dataclass(model)

def test_function_factory():
    with initialize(config_path='../../experiments/basic_training/cfg', version_base="1.1"):
        cfg = compose(config_name="basic_config")
    print(OmegaConf.to_yaml(cfg))
    netconfig = cfg.experiments.net
    
    model2 = model_factory(netconfig)
    assert isinstance(model2, nn.Module)
    print('function factory returns a torch.nn.Module instance')

if __name__ == "__main__":
    test_conv_net()
    test_function_factory()
