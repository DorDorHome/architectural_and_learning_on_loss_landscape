#import importlib

from typing import Any
from functools import partial as partial_fn
import test
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import os
import pickle
# get parent directory of project root
import sys
import pathlib
import numpy as np
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

print(PROJECT_ROOT)

from configs.configurations import DataConfig
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data_loading.transform_factory import transform_factory



def load_imagenet_by_classes(path_to_classes_folder_of_Imagnet:str  , train_images_per_class , test_images_per_class, classes=[], randomize: bool = True, test= False):
    """
    when given the classes of Imagenet, this function loads the data of the classes from the folder.
    Returns:
    x_train (of size train_images_per_class )
    y_train (of size train_images_per_class )
    x_test  ( of size 700 - train_images_per_class)
    y_test
    
    """
    if test:
        # pick one class, and load the data for that class
        data_file = os.path.join( path_to_classes_folder_of_Imagnet, str(0) +'.npy')
        new_x = np.load(data_file)
        # verify that the number of images is 700 and that it is the same as 
        # the sum of train_images_per_class and test_images_per_class
        assert new_x.shape[0] == 700
        assert new_x.shape[0] == train_images_per_class + test_images_per_class
        
        
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = os.path.join( path_to_classes_folder_of_Imagnet, str(_class) +'.npy')
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
        

    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    
    # shuffle the data:
    if randomize:
        idxs = np.random.permutation(x_train.size(0))
        x_train = x_train[idxs]
        y_train = y_train[idxs]
        
        idxs = np.random.permutation(x_test.size(0))
        x_test = x_test[idxs]
        y_test = y_test[idxs]
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    
    return x_train, y_train, x_test, y_test

def Imagnet_dataset_by_classes_factory(path_to_classes_folder_of_Imagnet: str, train_images_per_class: int, test_images_per_class: int, test: bool = False):
    
    factory_function_by_classes = partial_fn(load_imagenet_by_classes,
                                            path_to_classes_folder_of_Imagnet ,
                                            train_images_per_class ,
                                            test_images_per_class,
                                            test = test)
    return factory_function_by_classes


def Imagnet_dataset_generator(path_to_class_directory:str, transform = None):
    """
    Factory function to build a dataset generator for Imagenet.

    input:
        : str
        
    output:
        An Imagenet dataset generator that can be used to load Imagenet data by classes.
    
    """
    _dataset_by_classes = partial_fn(Imagnet_dataset_by_classes_factory, path_to_classes_folder_of_Imagnet = path_to_class_directory)
    if transform is not None:
      # not implemented yet, assert not implemented error:
        raise NotImplementedError("Transforms not implemented yet.")
      

    return _dataset_by_classes


def dataset_factory(config: DataConfig, transform, with_testset= False) -> Any:
    """
    Factory function to create dataset instances based on the configuration and model.

    Args:
        config (DataConfig): Configuration object containing dataset parameters.
        transform (Any): Transform pipeline for the dataset. 
        
    Returns:
        Any: A trainset and testset if with_testset is True, otherwise only a trainset.
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
            if with_testset:
                testset = dataset_class(root=dataset_path, train=False, download=True, transform=transform)
            # if the attribute .classes exists for the trainset, make sure it agress with config.num_classes:
            try:
                if len(trainset.classes) != config.num_classes:
                    raise ValueError(f"Number of classes in the train dataset ({len(trainset.classes)}) does not match the number of classes specified in the configuration ({config.num_classes}).")
                if len(test.classes) != config.num_classes:
                    raise ValueError(f"Number of classes in the test dataset ({len(testset.classes)}) does not match the number of classes specified in the configuration ({config.num_classes}).")
            except AttributeError:
                pass
            
            
        except AttributeError:
            raise AttributeError(f"dataset{config.dataset} not found in torchvision.datasets. Try setting use_torchvision to False.")
            
        
        # for CIFAR10:
        if config.dataset == 'CIFAR10':
                # Define transformations for the training and test sets
            trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        
        # # for mnist:
        # elif config.dataset == 'MNIST':
        #     trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

        # elif config.dataset == 'IMAGENET':
        #     trainset = torchvision.datasets.ImageNet(root=dataset_path, train=True, download=True, transform=transform)
            

    # custom data formats can be used:
    elif not config.use_torchvision:
        if config.dataset == 'imagenet_for_plasticity':
            assert config.num_classes == 1000, "Number of classes for Imagenet_plasicity_shifting_tasks should be 1000"
            # return a custom dataset object that allows for plasticity tasks
            path_to_imagenet =  os.path.join(config.data_path, 'imagenet_for_plasticity')
            path_to_class_directory = os.path.join(path_to_imagenet, 'data', 'classes')
            imagenet_dataset_factory_by_classes = Imagnet_dataset_generator(path_to_class_directory=path_to_class_directory,
                                                    transform = transform)
            
            # return a factory of Imagenet datasets that, based on 
            return imagenet_dataset_factory_by_classes
            
        else:
            raise NotImplementedError("custom dataset not implemented")

    if with_testset:
        return trainset, testset
    else:
        return trainset, None
        
if __name__=="__main__":
    # test the dataset_factory function
    
    test_dataconfig = DataConfig(dataset='CIFAR10',
                                 data_path= "/hdda/datasets",
                                 use_torchvision=True)
    model_name = 'ResNet18'
    dataset_factory(test_dataconfig, model_name)
    
    
    # verify that when Imagenet is loaded, 
    # 1. the number of images for each class is 700
    # 2. the classes object consists of random mutation of the classes of Imagenet, repeating
    # 3. the number of classes is 10
    
    
