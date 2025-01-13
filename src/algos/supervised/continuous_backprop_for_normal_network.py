# An implementation of continuous backprop for any network

from src.algos.supervised.base_learner import Learner
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod

from torch.optim import optimizer
from omegaconf import DictConfig

# parent directory of the project:
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



class ContinousBackprop_with_hooks(Learner):
    """
    An implementation of the continuous backprop 
    The network is assumed to have forward hooks implemented
    """
    
    def __init__(self, net:nn.Module, config: DictConfig ):
        super().__init__()