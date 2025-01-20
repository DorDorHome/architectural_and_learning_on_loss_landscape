# a basic backpropagation algorithm, with the option to perturb
from src.algos.supervised.base_learner import Learner
from configs.configurations import BaseLearnerConfig,  BackpropConfig, NetParams
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod

from torch.optim import optimizer
from omegaconf import DictConfig
from typing import Optional, Union

# import the BackpropConfig class from the config file in configs folder in the
# parent directory of the project:
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



class Backprop(Learner):
    def __init__(self, net: nn.Module, config: BackpropConfig, netconfig: Optional[Union[NetParams, None]] = None):
        super().__init__(net, config, netconfig)
        self.to_perturb = config.to_perturb
        self.perturb_scale = config.perturb_scale
        
        
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """implement the basic backpropagation algorithm"""
        # move data to device:
        x, target = x.to(self.device), target.to(self.device)
        self.opt.zero_grad()
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features
        
        # backpropagate
        loss.backward()
        self.opt.step()
        
        if self.to_perturb:
            self.perturb()
            
        return loss.item(), output.detach()
     
    
    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)