#import importlib

from typing import Any

# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

print(PROJECT_ROOT)

from configs.configurations import DataConfig

import hydra
from omegaconf import DictConfig, OmegaConf

def dataset_factory(config: DataConfig) -> Any:
    """
    function to create dataset instances based on the configuration.
    
    """
    if config.data_path is not None:
        dataset_path = cfg.data.data_path
    else:
        print("provide data path, either for reading from disk or downloading")
        raise ValueError("data path not provided")
    
    if config.use_torchvision:
        # for CIFAR10:
        if config.dataset == 'CIFAR10':
                # Define transformations for the training and test sets
            trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        
        # for mnist:
        elif cfg.data.dataset == 'MNIST':
            trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

        elif cfg.data.dataset == 'IMAGENET':
            trainset = torchvision.datasets.ImageNet(root=dataset_path, train=True, download=True, transform=transform)
            
        else:
            # raise error not implemented
            raise NotImplementedError("dataset not implemented")
    
    # handling the case where torchvision is not used:
    # custom data formats can be used:
    elif not config.use_torchvision:
        pass # implement custom data loading here
    
    return trainset
        
        