import importlib# for dynamic import
# for support of all models available in torchvision:
from torchvision import models as torchvision_models
from typing import Any
# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from conv_net import ConvNet
from VGG16_custom import vgg_with_internal_performance_track_custom_classifier

from configs.configurations import NetConfig, NetParams  # Assuming NetParams is defined here

def model_factory(config: NetConfig) -> Any:
    """
    Factory function to create model instances based on the configuration.

    Args:
        config (NetParams): Configuration object containing model parameters.

    Returns:
        Any: An instance of the requested PyTorch model.
    
    Raises:
        ValueError: If the specified model type is unsupported.
    """
    # Get the model type from the configuration. expect a string
    model_type = config.type
    
    # if model_type = 'convnet', return an instance of ConvNet:
    if model_type == 'convnet':
        if config.params is None:
            raise ValueError("config.params cannot be None for ConvNet")
        return ConvNet(config.params)
    
    if model_type =='vgg_custom':
        if config.params is None:
            raise ValueError("config.params cannot be None for VGG16_custom")
        return vgg_with_internal_performance_track_custom_classifier(config.params)
    