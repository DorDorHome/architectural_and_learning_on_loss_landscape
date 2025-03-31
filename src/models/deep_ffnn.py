import torch.nn as nn
import torch
from configs.configurations import LinearNetParams

class Layer(nn.Module):
    def __init__(self, in_shape, out_shape, act_type='relu' , initialization = 'kaiming'):
        super(Layer, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.act_type = act_type

        self.layers = nn.ModuleList()

        bias = True
        self.fc = nn.Linear(in_shape, out_shape, bias=bias)
        self.layers.append(self.fc)

        if self.act_type == 'linear':
            self.act_layer = None
        else:
            self.act_layer = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                      'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]
            self.act_layer = self.act_layer()
            self.layers.append(self.act_layer)

        # Initialize the weights
        if bias:
            self.fc.bias.data.fill_(0.0)
        if initialization == "kaiming":
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity=self.act_type)
        # otherwise raise not supported initialization error:
        else:
            raise ValueError(f"Initialization {initialization} not supported")
        
        
        
    def forward(self, x):
        x = self.fc(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class DeepFFNN(nn.Module):
    def __init__(self, 
                 config: LinearNetParams):
                 
                #  input_size,
                #  num_features=2000,
                #  num_outputs=1,
                #  num_hidden_layers=2,
                #  act_type='relu'):
    
        super(DeepFFNN, self).__init__()
        self.input_size = config.input_size
        self.num_features = config.num_features
        self.num_outputs = config.num_outputs
        self.num_hidden_layers = config.num_hidden_layers
        self.act_type = config.act_type
        self.initialization = config.initialization
        
        self.layers_to_log = [-(i * 2 + 1) for i in range(self.num_hidden_layers + 1)]

        # define the architecture
        self.layers = nn.ModuleList()

        self.in_layer = Layer(in_shape=self.input_size, out_shape=self.num_features, act_type=self.act_type, initialization=self.initialization)
        self.layers.extend(self.in_layer.layers)

        self.hidden_layers = []
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(Layer(in_shape=self.num_features, out_shape=self.num_features, act_type=self.act_type))
            self.layers.extend(self.hidden_layers[i].layers)

        self.out_layer = Layer(in_shape=self.num_features, out_shape=self.num_outputs, act_type='linear')
        self.layers.extend(self.out_layer.layers)

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        activations = []
        out = self.in_layer.forward(x=x)
        activations.append(out)
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(x=out)
            activations.append(out)
        out = self.out_layer.forward(x=out)
        return out, activations

