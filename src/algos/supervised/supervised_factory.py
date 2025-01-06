# factory function for supervised learning algorithms
# this should include all implemented SL typed learning method, including all backprop like methods.

# from typing import Union # for future implementation of configurations file that support dataclass for different objects.

from omegaconf import DictConfig
import sys
# import the Backprop class in basic_backprop.py file, contained in the supervised folder:
from src.algos.supervised.basic_backprop import Backprop


def create_learner(config: DictConfig):
    if config.type == 'backprop':
        return Backprop(config)
    
    # for support of torchvision models:
    
    


    
    
    