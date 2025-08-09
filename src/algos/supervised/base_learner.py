# Objectives for Learner:
# Initialization: Handle the setup of the neural network, optimizer, and loss function.
# Learning Process: Define a generic learn method that can be extended or overridden by subclasses.
# Utility Functions: Include any shared utility methods that multiple learners might use.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod
from typing import Callable
from torch.optim import optimizer
from omegaconf import DictConfig
from configs.configurations import *
from src.loss_for_regularization.regularized_loss_for_svd_conv import RegularizedLoss_SVD_conv

class Learner(ABC):
    """
    abstract base class for different learning algorithms
    """

    def __init__(self, net: nn.Module, config: BaseLearnerConfig, netconfig: Optional[Union[NetParams, None]] = None):

        """handle the setup of networks(agents), optimizer, and loss function."""
        # network/agent:
        self.config = config
        self.netconfig = netconfig
        
        self.device = config.device
        self.net = net.to(self.device)

        # intialize optimizer 
        self.opt = self._init_optimizer(config)
        self.loss= config.loss
        # initialize loss function
        self.loss_func = self._init_loss(config.loss)
        
        # for more complicated implementations that need to keep track of previous features
        self.previous_features = None
        
        # for algorithms that need to keep track of previous gradients
        self.latest_gradients = None
        
        

    def _init_optimizer(self, config: BaseLearnerConfig):
        
        opt = config.opt# need to override with the optimizer of choice
        
        step_size = config.step_size
        beta_1 = config.beta_1
        beta_2 = config.beta_2
        weight_decay = config.weight_decay
        momentum = config.momentum if hasattr(config, 'momentum') else 0.0
        

        # initialize loss function was moved to the __init__ method
        # self.loss_func = self._init_loss(self.loss)
        
        if opt == 'sgd':
            optimizer = optim.SGD(self.net.parameters(),
                                  lr=step_size,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        elif opt == 'adam':
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=step_size,
                                   betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)
        
        else:
            raise ValueError(f'Optimizer {opt} not implemented')
        
        return optimizer
    
    def _init_loss(self, loss: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        loss_funcs: dict[str,  Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            'cross_entropy': F.cross_entropy,
            'mse': F.mse_loss
        }
        
        if loss not in loss_funcs:
            raise ValueError(f"Unsupported loss type: {loss}")
        
        # Check if regularization is enabled
        main_loss_func = loss_funcs[loss]
                
        if hasattr(self.config, 'additional_regularization') and self.config.additional_regularization:
            lambda_orth = self.config.lambda_orth if hasattr(self.config, 'lambda_orth') and self.config.lambda_orth is not None else 1e-4
            return RegularizedLoss_SVD_conv(
                main_loss_func=main_loss_func,
                model=self.net,
                lambda_orth=lambda_orth,
                allow_svd_values_negative=self.netconfig.netparams.allow_svd_values_negative
            )
        else:
            return main_loss_func



    @abstractmethod
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """learn from a batch of data"""
        pass
    
    def _forward(self, x: torch.Tensor):# -> Any | Any:
        return self.net.predict(x)