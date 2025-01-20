## This file contains 3 versions of VGG:
# 1. vanila version
# 2. with Batch Normalization
# 3. with normalized weights, but no Batch Normalization:

import torch
from torch import nn
#import the NetParams dataclass from configurations.py
from configs.configurations import NetParams
from src.models.normalized_weights_FC import NormalizedWeightsLinear
from src.models.vgg_normalized_conv import vgg16 as vgg16_normalized

# from ..utils.nn import init_weights_


# Function to initialize model weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, NormalizedWeightsLinear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
    

def change_model_last_normalized_FC_layer(model: torch.nn.Module, second_last_dim:int  = 4096, n_classes:int = 10) -> torch.nn.Module:
    """
    """
    model.classifier[6] = NormalizedWeightsLinear( second_last_dim, out_dim= n_classes)
    return model

class vgg_normalized_custom(nn.Module):
    model_name = "VGG_with_Custom_Classifier"  # Class attribute
    def __init__(self, config: NetParams):
        
        # get the configuration parameters for initializing the model
        pretrained = getattr(config, 'pretrained', False)
        num_classes = getattr(config, 'num_classes', None)
        initialization = getattr(config, 'initialization', "kaiming")
        loss_window_size = getattr(config, 'loss_window_size', 10)  
        track_performance_internally = getattr(config, 'track_performance_internally', False)
        super(vgg_normalized_custom, self).__init__()
        
        # Load the VGG16 model
        self.model: nn.Module = vgg16_normalized(pretrained=pretrained)

        # set unique model id:
        # self.model_id = str(uuid.uuid4())

        self.model_count = None
        self.generation = None
        self.survival_count = None

        # Modify the classifier to match the number of classes
        # if not specified, keep the original number of classes
        if num_classes is not None:
            self.model.classifier[6] = NormalizedWeightsLinear(4096, num_classes)
        # if initialization == "kaiming":
        #     self.model.apply(initialize_weights)
        
        # setting attributes for genealogy:
        # self.parents = None

        
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
        return self.model(x)
        
        
    def predict(self, x):
        """
        A method to return both the final output and the intermediate features
        """
        return self.model(x), None


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

if __name__=="__main__":
    # test the vgg_normalized_custom class
    config = NetParams()
    model = vgg_normalized_custom(config)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output)
    print(output.shape)
    print("Model loaded successfully")
    # test the change_model_last_normalized_FC_layer function
    # model = change_model_last_normalized_FC_layer(model)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output)
    print(output.shape)
    print("Model loaded successfully")




# class vgg_normalized(nn.Module):
#     """VGG_A model

#     size of Linear layers is smaller since input assumed to be 32x32x3, instead of
#     224x224x3
#     """

#     def __init__(self, 
#                 #  input_channels=3,
#                 #  num_classes=10, 
#                 #  init_weights=True, 
#                 #  initialization="kaiming"):
#                 config: NetParams):
        
#         input_channels= getattr(config, 'input_channels', 3)
#         num_classes = getattr(config, 'num_classes', None)
#         initialization = getattr(config, 'initialization', "kaiming")
#         # loss_window_size = getattr(config, 'loss_window_size', 10)  
#         # track_performance_internally = getattr(config, 'track_performance_internally', False)
                

#         super().__init__()

#         self.features = nn.Sequential(
#             # stage 1
#             nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage 4
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage5
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 1 * 1, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes))

#         if initialization == "kaiming":
#             self.apply(initialize_weights)
        

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(-1, 512 * 1 * 1))
#         return x
    
#     def predict(self, x):
#         """
#         A method to return both the final output and the intermediate features
#         """
#         return self.forward(x), None

#     # def _init_weights(self):
#     #     for m in self.modules():
#     #         (m)


# class VGGABatchNorm(nn.Module):
#     """VGG_A model with BatchNorm after each layer

#     size of Linear layers ismaller since input assumed to be 32x32x3, instead of
#     224x224x3
#     """

#     def __init__(self,
#                 config: NetParams):
                
#         input_channels= getattr(config, 'input_channels', 3)
#         num_classes = getattr(config, 'num_classes', None)
#         initialization = getattr(config, 'initialization', "kaiming")
#         # loss_window_size = getattr(config, 'loss_window_size', 10)  
#         # track_performance_internally = getattr(config, 'track_performance_internally', False)
                
                 
                 
                 
                 
#         super().__init__()

#         self.features = nn.Sequential(
#             # stage 1
#             nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             #stage 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage 4
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # stage5
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 1 * 1, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes))

#         if initialization == "kaiming":
#             self.apply(initialize_weights)
        

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(-1, 512 * 1 * 1))
#         return x

#     def predict(self, x):
#         """
#         A method to return both the final output and the intermediate features
#         """
#         return self.forward(x), None

#     # def _init_weights(self):
#     #     for m in self.modules():
#     #         init_weights_(m)
            
            
# if __name__ == "__main__":
#     # set config:
#     config = NetParams(num_classes=10, initialization="kaiming")
    
#     # test the model
#     model = VGGA(config)
#     print(model)
#     x = torch.randn(2, 3, 32, 32)
#     print(model(x).shape)
#     model = VGGABatchNorm(config)
#     print(model)
#     x = torch.randn(2, 3, 32, 32)
#     print(model(x).shape)

