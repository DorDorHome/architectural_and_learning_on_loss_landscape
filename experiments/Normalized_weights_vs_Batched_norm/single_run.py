from typing import Any
import sys
import pathlib
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from configs.configurations import ExperimentConfig

import os
# import pickle
# import argparse
# import numpy as np
from tqdm import tqdm
import hydra 
from omegaconf import OmegaConf # , DictConfig
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
# import model factory:

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
import torchvision.transforms as transforms
import torchvision
import torch

@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg :ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    
    net = model_factory(cfg.net)
    
        #verifty cfg.runs ==1:
    if cfg.runs != 1:
        raise ValueError("This script is only for single runs. Please set runs to 1 or use another script.")

    # if the run_id attribute is not set, set it to 0
    if cfg.run_id is None:
        cfg.run_id = 0
        
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
    if cfg.learner.type == 'cbp' and cfg.net.type == 'conv_net':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinualBackprop_for_ConvNet
        learner = ContinualBackprop_for_ConvNet(net, cfg.learner)
    
    # setup data:
    
    # setup data:
    # load the transfrom based on the dataset and model:
    # combination of dataset and model determines the transform
    if cfg.net.class != 'fc':
        transform = transform_factory(cfg.data.dataset, cfg.net.type)
    else:
        transform = None
        
    #trainset with the transform:
    if cfg.evaluation.with_testset:
        trainset, testset = dataset_factory(cfg.data, transform, with_testset= cfg.evaluation.with_testset)
    else
        trainset = dataset_factory(cfg.data, transform, with_testset= cfg.evaluation.use_testset)

    #trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= cfg.batch_size, shuffle=True, num_workers=2, pin_memory = True)
    
    # setup evaluation:
    
