# a basic backpropagation algorithm, with the option to perturb
from src.algos.supervised.base_learner import Learner

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod

from torch.optim import optimizer
from omegaconf import DictConfig
class Backprop(Learner):
    def __init__(self, net: nn.Module, config: DictConfig):
        super().__init__(net, config)
        self.to_perturb = config.to_perturb
        self.perturb_scale = config.perturb_scale
        
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """implement the basic backpropagation algorithm"""
        self.opt.zero_grad()
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features
        
        # backpropagate
        loss.backward()
        self.opt.step()
        
        if self.to_perturb:
            self._perturb()
            
        if self.loss == 'nll':
            return loss.item(), output.detact()
        
        return loss.detach()#.item()?
    
    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)