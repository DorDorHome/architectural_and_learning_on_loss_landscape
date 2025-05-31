# identical to the conv_net implementation in loss of plasiticity paper, for easy comparison
# notice the predict method is different from the usual forward method


import torch.nn as nn
import torch
# original implementation:
# from omegaconf import DictConfig

# use NetParams dataclass from configurations.py:
from configs.configurations import NetParams


activation_dict = {
    # 'relu': F.relu,
    # 'sigmoid': torch.sigmoid,
    # 'tanh': torch.tanh,
    # # Add more activations as needed
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
    'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}




class ConvNet(nn.Module):
    def __init__(self, config: NetParams ):
        """
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        num_classes = config.num_classes
        #in_channels, out_channels, kernel_size are not implemented
        
        self.activation = config.activation
        
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        self.pool = nn.MaxPool2d(2, 2)

        
        # self.last_filter_output = 2 * 2
        # self.num_conv_outputs = 128 * self.last_filter_output
            # Compute the size of the conv feature map.
        dummy = torch.zeros(1, 3, config.input_height, config.input_width)
        x = self.pool(nn.ReLU()(self.conv1(dummy)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        flattened_size = x.view(1, -1).shape[1]
    
        
        #self.fc1 = nn.Linear(self.num_conv_outputs, 128)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # self.act_type = self.activation
        self.activation = activation_dict.get(self.activation, None)
        

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(self.activation())
        self.layers.append(self.conv2)
        self.layers.append(self.activation())
        self.layers.append(self.conv3)
        self.layers.append(self.activation())
        self.layers.append(self.fc1)
        self.layers.append(self.activation())
        self.layers.append(self.fc2)
        self.layers.append(self.activation())
        self.layers.append(self.fc3)



    def predict(self, x):
        batch_size = x.size(0)
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(batch_size, -1)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]
