#import importlib

from typing import Any

import test
import torchvision.datasets
import torchvision.transforms as transforms

# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

print(PROJECT_ROOT)

from configs.configurations import DataConfig
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data_loading.transform_factory import transform_factory


def dataset_factory(config: DataConfig, transform) -> Any:
    """
    Factory function to create dataset instances based on the configuration and model.

    Args:
        config (DataConfig): Configuration object containing dataset parameters.
        model_name (str): The name of the model (used to determine transforms).

    Returns:
        Any: A dataset instance.
    """
    if not config.data_path:
        raise ValueError("Data path not provided. Please specify 'data_path' in the configuration for reading from disk or downloading."
        )
    dataset_path = config.data_path
    
    if config.use_torchvision:
        # dynamically load datasets from torchvision:
        try: 
            dataset_class = getattr(torchvision.datasets, config.dataset)
            trainset = dataset_class(root=dataset_path, train=True, download=True, transform=transform)
            
        except AttributeError:
            raise AttributeError(f"dataset{config.dataset} not found in torchvision.datasets")
            
        
        # # for CIFAR10:
        # if config.dataset == 'CIFAR10':
        #         # Define transformations for the training and test sets
        #     trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        
        # # for mnist:
        # elif config.dataset == 'MNIST':
        #     trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

        # elif config.dataset == 'IMAGENET':
        #     trainset = torchvision.datasets.ImageNet(root=dataset_path, train=True, download=True, transform=transform)
            

    # custom data formats can be used:
    elif not config.use_torchvision:
        raise NotImplementedError("custom dataset not implemented")
        
    
    return trainset
        
if __name__=="__main__":
    # test the dataset_factory function
    
    test_dataconfig = DataConfig(dataset='CIFAR10',
                                 data_path= "/hdda/datasets",
                                 use_torchvision=True)
    model_name = 'ResNet18'
    dataset_factory(test_dataconfig, model_name)