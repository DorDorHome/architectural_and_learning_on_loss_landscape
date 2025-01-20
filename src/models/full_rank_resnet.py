# from types import NoneType
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from typing import List
#from torchvision.models import resnet18 
from src.models.resnet_normalized_conv import resnet18 as resnet18_norm
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

class ResNet18_skip_to_last_with_custom_classifier(nn.Module):
    model_name = "ResNet18_with_custom_classifier"  # Class attribute
    def __init__(self, config: NetParams):
        
        # get the configuration parameters for initializing the model
        pretrained = getattr(config, 'pretrained', False)
        num_classes = getattr(config, 'num_classes', None)
        initialization = getattr(config, 'initialization', "kaiming")
        loss_window_size = getattr(config, 'loss_window_size', 10)  
        track_performance_internally = getattr(config, 'track_performance_internally', False)
        super(ResNet18_skip_to_last_with_custom_classifier, self).__init__()
        
        # Load the ResNet18 model
        self.model: nn.Module = resnet18_norm(pretrained=pretrained)
        
        # change forward method:
        
        

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
        return self.model(x) #x shape: torch.Size([1, 3, 224, 224])
        # x1 = self.model.conv1(x) #x1.shape torch.Size([1, 64, 112, 112])
        # x2 = self.model.bn1(x1)   #x2.shape torch.Size([1, 64, 112, 112])
        # x3 = self.model.relu(x2) #x3.shape torch.Size([1, 64, 112, 112])
        # x4 = self.model.maxpool(x3) #x4.shape torch.Size([1, 64, 56, 56])
        
        # x5 = self.model.layer1(x4) #x5.shape torch.Size([1, 64, 56, 56])
        # x6 = self.model.layer2(x5)  # torch.Size([1, 128, 28, 28])
        # x7 = self.model.layer3(x6) # torch.Size([1, 256, 14, 14])
        # x8 = self.model.layer4(x7) # torch.Size([1, 512, 7, 7])
        
        # x9 = self.model.avgpool(x8) # torch.Size([1, 512, 1, 1])
        # x10 = torch.flatten(x9, 1) # torch.Size([1, 512])
        # x11 = self.model.fc(x10)   # torch.Size([1, num_classes])
        # return x11
        
        
    def predict(self, x):
        """
        A method to return both the final output and the intermediate features
        """
        return self.model(x), None


if __name__ == "__main__":
    # test the model:
    net = ResNet18_skip_to_last_with_custom_classifier(NetParams())
    print(net)
    # test the forward method:
    x = torch.randn(1, 3, 332, 332)
    y = net(x)
    print(y.shape)
    # test the predict method:
    y, features = net.predict(x)
    print(y.shape)
    print(features)
    print("Test passed")
            
    