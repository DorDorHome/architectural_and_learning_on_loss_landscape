# test basic_backprop.py
# together with the factory function in supervised_factory.py


import pytest
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.supervised_factory import create_learner
from src.models.conv_net import ConvNet
import torch.nn as nn

from configs.configurations import NetParams, BackpropConfig

def test_backprop_initialization():
    net_params = NetParams(num_classes=10, in_channels=1, input_height=28, input_width=28)
    net = ConvNet(config=net_params)

    # define config for learner:
    backprop_config = BackpropConfig(
        type='backprop',
        to_perturb=False,
        device='cpu'
    )

    learner = create_learner(backprop_config, net)
    assert isinstance(learner, Backprop)
    assert learner.net == net
