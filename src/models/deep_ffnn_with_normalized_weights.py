from src.models.normalized_weights_FC import NormalizedWeightsLinear
from src.models.normalized_weights_FC import NormalizedRescalePerChannelWeightsLinear
from src.models.normalized_weights_FC import  BatchNormedWeightsLinear
from configs.configurations import LinearNetParams
import torch
import torch.nn as nn   


class DeepFFNN_weight_norm_single_recale(nn.Module):
    def __init__(self, config: LinearNetParams):
        super(DeepFFNN_weight_norm_single_recale, self).__init__()
        self.input_size = config.input_size
        self.num_features = config.num_features
        self.num_outputs = config.num_outputs
        self.num_hidden_layers = config.num_hidden_layers
        self.act_type = config.act_type
        self.layers_to_log = [-(i * 2 + 1) for i in range(self.num_hidden_layers + 1)]

        
        # get the configuration parameters for initializing the model
        
        
        self.layers = nn.ModuleList()
        
        self.in_layer = NormalizedWeightsLinear(in_dim=self.input_size, out_dim=self.num_features, activation=self.act_type)
        
        self.layers.append(self.in_layer)
        
        self.hidden_layers = [] # save hidden layers, to access internal features.
        
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(NormalizedWeightsLinear(in_dim=self.num_features, out_dim=self.num_features, activation=self.act_type))
            self.layers.append(self.hidden_layers[i])
        
        self.out_layer = NormalizedWeightsLinear(in_dim=self.num_features, out_dim=self.num_outputs, activation='linear')
        
        self.layers.append(self.out_layer)
        
    def predict(self, input):
        """
        """
        
        inner_features = [ ] # also called activations
        
        out = self.in_layer.forward(input)
        inner_features.append(out)
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(out)
            inner_features.append(out)
        out = self.out_layer.forward(out)
        return out, inner_features
        
class DeepFFNN_weight_norm_multi_channel_recale(nn.Module):
    def __init__(self, config: LinearNetParams):
        super(DeepFFNN_weight_norm_multi_channel_recale, self).__init__()
        self.input_size = config.input_size
        self.num_features = config.num_features
        self.num_outputs = config.num_outputs
        self.num_hidden_layers = config.num_hidden_layers
        self.act_type = config.act_type
        self.layers_to_log = [-(i * 2 + 1) for i in range(self.num_hidden_layers + 1)]

        
        # get the configuration parameters for initializing the model
        self.layers = nn.ModuleList()
        
        self.in_layer = NormalizedRescalePerChannelWeightsLinear(in_dim=self.input_size, out_dim=self.num_features, activation=self.act_type)
        
        self.layers.append(self.in_layer)
        
        self.hidden_layers = [] # save hidden layers, to access internal features.
        
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(NormalizedRescalePerChannelWeightsLinear(in_dim=self.num_features, out_dim=self.num_features, activation=self.act_type))
            self.layers.append(self.hidden_layers[i])
        
        self.out_layer = NormalizedRescalePerChannelWeightsLinear(in_dim=self.num_features, out_dim=self.num_outputs, activation='linear')
        
        self.layers.append(self.out_layer)
        
    def predict(self, input):
        """
        """
        
        inner_features = [ ] # also called activations
        
        out = self.in_layer.forward(input)
        inner_features.append(out)
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(out)
            inner_features.append(out)
        out = self.out_layer.forward(out)
        return out, inner_features
        
        

class DeepFFNN_EMA_batch_weight_norm(nn.Module):
    def __init__(self, config: LinearNetParams):
        super(DeepFFNN_EMA_batch_weight_norm, self).__init__()
        self.input_size = config.input_size
        self.num_features = config.num_features
        self.num_outputs = config.num_outputs
        self.num_hidden_layers = config.num_hidden_layers
        self.act_type = config.act_type
        self.layers_to_log = [-(i * 2 + 1) for i in range(self.num_hidden_layers + 1)]

        
        # get the configuration parameters for initializing the model
        
        
        self.layers = nn.ModuleList()
        
        self.in_layer = BatchNormedWeightsLinear(in_dim=self.input_size, out_dim=self.num_features, activation=self.act_type)
        
        self.layers.append(self.in_layer)
        
        self.hidden_layers = [] # save hidden layers, to access internal features.
        
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(BatchNormedWeightsLinear(in_dim=self.num_features, out_dim=self.num_features, activation=self.act_type))
            self.layers.append(self.hidden_layers[i])
        
        self.out_layer = BatchNormedWeightsLinear(in_dim=self.num_features, out_dim=self.num_outputs, activation='linear')
        
        self.layers.append(self.out_layer)
        
    def predict(self, input):
        """
        """
        
        inner_features = [ ] # also called activations
        
        out = self.in_layer.forward(input)
        inner_features.append(out)
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(out)
            inner_features.append(out)
        out = self.out_layer.forward(out)
        return out, inner_features
        

if __name__ == "main":
    # Example configuration
    config = LinearNetParams(
        input_size=10,
        num_features=20,
        num_outputs=1,
        num_hidden_layers=3,
        act_type='relu'
    )

    # Initialize the model
    model = DeepFFNN_weight_norm(config)

    # Create some example input tensors
    example_input = torch.randn(5, config.input_size)  # Batch size of 5

    # Perform a forward pass
    output, inner_features = model.predict(example_input)

    print("Output:", output)
    print("Inner Features:", inner_features)
    
    
    
    
        
        
