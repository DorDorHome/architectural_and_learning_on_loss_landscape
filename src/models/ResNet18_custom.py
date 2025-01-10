# from types import NoneType
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from typing import List
from torchvision.models import resnet18 
import numpy as np
import os
#import networkx as nx
import uuid
from collections import deque
#import the NetParams dataclass from configurations.py
from configs.configurations import NetParams

def kaiming_init(model, nonlinearity='relu', mode='fan_out', a=0, gain=None):
    """
    Initialize the weights of the model using Kaiming initialization.

    Parameters:
    - model (nn.Module): The neural network model to initialize.
    - nonlinearity (str): The non-linear activation function used after this layer ('relu' is default).
    - mode (str): Either 'fan_in' or 'fan_out'. Choose 'fan_out' to preserve the magnitude of the variance of the weights in the forward pass.
    - a (float): The negative slope of the rectifier used after this layer (only used with 'leaky_relu').
    - gain (float, optional): An optional scaling factor. Default is calculated from `nonlinearity` and `a`.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ResNet18_with_custom_classifier(nn.Module):
    model_name = "ResNet18_with_custom_classifier"  # Class attribute
    def __init__(self, config: NetParams):
        
        # get the configuration parameters for initializing the model
        pretrained = getattr(config, 'pretrained', False)
        num_classes = getattr(config, 'num_classes', None)
        initialization = getattr(config, 'initialization', "kaiming")
        loss_window_size = getattr(config, 'loss_window_size', 10)  
        track_performance_internally = getattr(config, 'track_performance_internally', False)
        super(ResNet18_with_custom_classifier, self).__init__()
        
        # Load the ResNet18 model
        self.model: nn.Module = resnet18(pretrained=pretrained)

        # set unique model id:
        self.model_id = str(uuid.uuid4())

        self.model_count = None
        self.generation = None
        self.survival_count = None

        # Modify the classifier to match the number of classes
        # if not specified, keep the original number of classes
        if num_classes is not None:
            self.model.fc = nn.Linear(512, num_classes)
        if initialization == "kaiming":
            self.model.apply(kaiming_init)
    
    def forward(self, x):
        return self.model(x)
        
        
    def predict(self, x):
        """
        A method to return both the final output and the intermediate features
        """
        return self.model(x), None

            
    