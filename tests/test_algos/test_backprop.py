# test basic_backprop.py
# together with the factory function in supervised_factory.py


import pytest
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.supervised_factory import create_learner
from src.models.conv_net import ConvNet
import torch.nn as nn

def test_backprop_initialization():
    netconfig = {'num_classes': 10}
    net = ConvNet(config = netconfig) # default num_classes = 10

    # define config for learner:
    backprop_config = {'to_perturb': False,
                       'perturb_scale': 0.1}
