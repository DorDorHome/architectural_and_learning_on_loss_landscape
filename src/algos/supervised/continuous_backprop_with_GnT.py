# continuous backprop via generate and test method:
# this mainly uses code snippets from loss of plasticity paper

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod

# for choices of optimizer
from torch.optim import optimizer

# for configuration:
from omegaconf import DictConfig

# import the BackpropConfig class from the config file in configs folder in the
# parent directory of the project:
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# base learner class:
from src.algos.supervised.base_learner import Learner

# import the gnt class from the gnt.py file in the same directory:
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC
from src.algos.AdamGnT import AdamGnT
from configs.configurations import ContinuousBackpropConfig

_all_loss_functions = {'cross_entropy': F.cross_entropy, 'mse': F.mse_loss}


class ContinualBackprop_for_FC(object):
    """
    The Continual Backprop algorithm, used in https://arxiv.org/abs/2108.06325v3
    
    Only works for ConvNet model, as it uses the predict method of the ConvNet class
    """
    def __init__(
            self,
            net: nn.Module, 
            learnerconfig: ContinuousBackpropConfig
            # step_size=0.001, # in BaseLearnerConfig
            # loss='mse', # in BaseLearnerConfig
            # opt='sgd', # in BaseLearnerConfig
            # beta_1=0.9, # in BaseLearnerConfig
            # beta_2=0.999,  # in BaseLearnerConfig
            # neurons_replacement_rate=0.001, # in ContinuousBackpropConfig
            # decay_rate_utility_track=0.9,  # in ContinuousBackpropConfig
            # device='cpu', # in BaseLearnerConfig
            # maturity_threshold=100, # in ContinuousBackpropConfig
            # util_type='contribution', # in ContinuousBackpropConfig
            # init='kaiming',  # in ContinuousBackpropConfig
            # accumulate=False,  # in ContinuousBackpropConfig
            # momentum=0, # in BaseLearnerConfig
            # outgoing_random=False, # in ContinuousBackpropConfig
            # weight_decay=0 # in BaseLearnerConfig
    ):
        self.net = net
        self.opt = learnerconfig.opt
        self.loss = learnerconfig.loss
        self.step_size = learnerconfig.step_size
        self.beta_1 = learnerconfig.beta_1
        self.beta_2 = learnerconfig.beta_2
        self.neurons_replacement_rate = learnerconfig.neurons_replacement_rate
        self.decay_rate_utility_track = learnerconfig.decay_rate_utility_track
        self.device = learnerconfig.device
        self.maturity_threshold = learnerconfig.maturity_threshold
        self.util_type = learnerconfig.util_type
        self.init = learnerconfig.init
        self.accumulate = learnerconfig.accumulate
        self.momentum = learnerconfig.momentum
        self.outgoing_random = learnerconfig.outgoing_random
        self.weight_decay = learnerconfig.weight_decay
        
        
        

        # define the optimizer
        if self.opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=self.step_size, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=self.step_size, betas=(self.beta_1, self.beta_2), weight_decay=self.weight_decay)

        # define the loss function
        self.loss_func = _all_loss_functions[self.loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = None
        
        if self.net.type == 'FC':
            self.gnt = GnT_for_FC(
                net=self.net.layers,
                hidden_activation=self.net.act_type,
                opt=self.opt,
                replacement_rate=self.neurons_replacement_rate,
                decay_rate=self.decay_rate_utility_track,
                maturity_threshold=self.maturity_threshold,
                util_type=self.util_type,
                device=self.device,
                loss_func=self.loss_func,
                init=self.init,
                accumulate=self.accumulate,
            )

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT_for_FC: # original: GnT:
            self.gnt.gen_and_test(features=self.previous_features)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()

        return loss.detach()

class ContinuousBackprop_for_ConvNet(object):
    """
    The Continual Backprop algorithm
    """
    def __init__(self,
                 net,
                 learnerconfig: ContinuousBackpropConfig
                #  step_size=0.001,
                #  loss='mse',
                #  opt='sgd',
                #  beta_1=0.9,
                #  beta_2=0.999,
                #  neurons_replacement_rate=0.0001,
                #  decay_rate=0.9,
                #  init='kaiming',
                #  util_type='contribution',
                #  maturity_threshold=100,
                #  device='cpu',
                #  momentum=0,
                #  weight_decay=0
                 ):
        self.net = net
        self.opt = learnerconfig.opt
        self.loss = learnerconfig.loss
        self.step_size = learnerconfig.step_size
        self.beta_1 = learnerconfig.beta_1
        self.beta_2 = learnerconfig.beta_2
        self.neurons_replacement_rate = learnerconfig.neurons_replacement_rate
        self.decay_rate_utility_track = learnerconfig.decay_rate_utility_track
        self.init = learnerconfig.init
        self.util_type = learnerconfig.util_type
        self.maturity_threshold = learnerconfig.maturity_threshold
        self.device = learnerconfig.device
        self.momentum = learnerconfig.momentum
        self.weight_decay = learnerconfig.weight_decay
        


        # define the optimizer
        if self.opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=self.step_size, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=self.step_size, betas=(self.beta_1, self.beta_2), weight_decay=self.weight_decay)

        # define the loss function
        self.loss_func = _all_loss_functions[self.loss]
        #{'cross_entropy': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = ConvGnT_for_ConvNet(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            init=self.init,
            num_last_filter_outputs=net.last_filter_output,
            util_type=self.util_type,
            maturity_threshold=self.maturity_threshold,
            device=self.device,
        )

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # take a generate-and-test step
        self.gnt.gen_and_test(features=self.previous_features)

        return loss.detach(), output
