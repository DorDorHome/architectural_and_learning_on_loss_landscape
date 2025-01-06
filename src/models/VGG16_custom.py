# a version of VGG16, mainly by importing from torchvision, 
# with a few extra features:
# 1. in forward method, the intermediate features are returned
# 2. methods to serialize and deserialize the model
# 3. an option to change the final fully connected layer based on the number of classes

from types import NoneType
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from typing import List
from torchvision.models import vgg16
import numpy as np
import os
#import networkx as nx
import uuid
from collections import deque
#import the NetParams dataclass from configurations.py
from configs.configurations import NetParams

# Function to initialize model weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

def change_model_last_FC_layer(model: torch.nn.Module, second_last_dim:int  = 4096, n_classes:int = 10) -> torch.nn.Module:
    """
    """
    model.classifier[6] = nn.Linear( second_last_dim, out_features= n_classes)
    return model

class vgg_with_internal_performance_track_custom_classifier(nn.Module):
    model_name = "VGG_with_Custom_Classifier"  # Class attribute
    def __init__(self, config: NetParams):
        
    
        # get the configuration parameters for initializing the model
        pretrained = getattr(config, 'pretrained', False)
        num_classes = getattr(config, 'num_classes', None)
        initialization = getattr(config, 'initialization', "kaim")
        loss_window_size = getattr(config, 'loss_window_size', 10)  
        track_performance_internally = getattr(config, 'track_performance_internally', False)
        super(vgg_with_internal_performance_track_custom_classifier, self).__init__()
        
        # Load the VGG16 model
        self.model: nn.Module = vgg16(pretrained=pretrained)

        # set unique model id:
        self.model_id = str(uuid.uuid4())

        self.model_count = None
        self.generation = None
        self.survival_count = None

        # Modify the classifier to match the number of classes
        # if not specified, keep the original number of classes
        if num_classes is not None:
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        if initialization == "kaiming":
            self.model.apply(initialize_weights)
        
        # setting attributes for genealogy:
        self.parents = None

        
        # internal performance track:
        # everything related to performance of latest training epoch:
        if track_performance_internally:
            self.latest_epoch_loss = None
            self.latest_accuracy = None
            self.loss_window_size = loss_window_size
            self.loss_history = deque[float](maxlen=loss_window_size)
            self.latest_mean_loss_reduction = None

            
            # number generations is not needed, as it will be tracked by the genealogy
            # total number of epoch trained:
            self.total_epochs_trained_as_individual = 0 

            self.total_epochs_including_parents = 0 # actually also tracked by genealogy



    def forward(self, x):
        self.model(x)
        
        
    def predict(self, x):
        """
        A method to return both the final output and the intermediate features
        """
        pass #TO-DO, to be implemented


    def serialize(self):
        return {
            'state_dict': self.state_dict(),
            'custom_attributes': {
                'model_id': self.model_id,
                'generation': self.generation,
                'latest_loss': self.latest_loss,
                'latest_accuracy': self.latest_accuracy,
                'parents': self.parents,
                'total_epochs_trained_as_individual': self.total_epochs_trained_as_individual,
                'total_epochs_including_parents': self.total_epochs_including_parents,

            }
        }

    @classmethod
    def deserialize(cls, data):
        model = cls()
        model.load_state_dict(data['state_dict'])

        # more general implementation:
        # for attr_name, attr_value in state['custom_attributes'].items():
        #     setattr(model, attr_name, attr_value)
        model.model_id = data['custom_attributes']['model_id']
        model.generation = data['custom_attributes']['generation']
        model.latest_loss = data['custom_attributes']['latest_loss']
        model.latest_accuracy = data['custom_attributes']['latest_accuracy']
        model.parents = data['custom_attributes']['parents']
        model.total_epochs_trained_as_individual = data['custom_attributes']['total_epochs_trained_as_individual']
        model.total_epochs_including_parents = data['custom_attributes']['total_epochs_including_parents']


        return model


