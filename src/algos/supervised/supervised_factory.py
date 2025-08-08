# factory function for supervised learning algorithms
# this should include all implemented SL typed learning method, including all backprop like methods.

# from typing import Union # for future implementation of configurations file that support dataclass for different objects.

from omegaconf import DictConfig
import sys
# import the Backprop class in basic_backprop.py file, contained in the supervised folder:
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet, ContinualBackprop_for_FC

def create_learner(config: DictConfig, net):
    if config.type == 'backprop':
        return Backprop(net, config)
    elif config.type == 'basic_continous_backprop':
        if config.network_class == 'conv':
            return ContinuousBackprop_for_ConvNet(net, config)
        elif config.network_class == 'fc':
            return ContinualBackprop_for_FC(net, config)
        else:
            raise ValueError(f"Unsupported network type for basic_continous_backprop: {net.type}")
    
    # for support of torchvision models: