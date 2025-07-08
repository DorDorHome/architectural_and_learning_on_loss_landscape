import importlib# for dynamic import
# for support of all models available in torchvision:
from torchvision import models as torchvision_models
from typing import Any
# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "experiments/basic_training/cfg")
sys.path.append(str(PROJECT_ROOT))

from src.models.conv_net import ConvNet

from configs.configurations import NetConfig, NetParams  # Assuming NetParams is defined here

import hydra 
from omegaconf import DictConfig, OmegaConf

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
    if model_type == 'ConvNet':
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet")
        return ConvNet(config.netparams)
    if model_type == "ConvNet_norm":
        from src.models.conv_net_weights_normalized import ConvNet_normalized
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_norm")
        return ConvNet_normalized(config.netparams)
    if model_type == "ConvNet_SVD":  
        from src.models.ConvNet_SVD import ConvNet_SVD
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_SVD")
        return ConvNet_SVD(config.netparams)
    
    if model_type == "ConvNet_FC_layer_norm":
        from src.models.layer_norm_conv_net import ConvNetWithFCLayerNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_layer_norm")
        return ConvNetWithFCLayerNorm(config.netparams)
    
    if model_type == "ConvNet_conv_and_FC_layer_norm":
        from src.models.layer_norm_conv_net import ConvNet_conv_and_FC_LayerNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_conv_and_FC_layer_norm")
        return ConvNet_conv_and_FC_LayerNorm(config.netparams)
        
    if model_type == 'ConvNet_batch_norm':
        from src.models.conv_net_batch_norm import ConvNetWithBatchNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_batch_norm")
        return ConvNetWithBatchNorm(config.netparams)
    
    if model_type == 'vgg_custom':
        from src.models.VGG16_custom import vgg_with_internal_performance_track_custom_classifier as VGG_custom

        if config.netparams is None:
            raise ValueError("config.params cannot be None for VGG16_custom")
        return VGG_custom(config.netparams)
        
    if model_type == 'vgg_custom_norm':   
        from src.models.VGG_normalized_conv import vgg_normalized_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for vgg_custom_norm")
        return vgg_normalized_custom(config.netparams)
        
    if model_type == 'resnet_custom':
        from src.models.ResNet18_custom import ResNet18_with_custom_classifier as ResNet_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for resnet_custom")
        return ResNet_custom(config.netparams)
    
    if model_type == 'full_rank_resnet_custom':
        from src.models.full_rank_resnet import ResNet18_skip_to_last_with_custom_classifier as full_rank_ResNet_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for resnet_custom")
        return full_rank_ResNet_custom(config.netparams)
     
    
    if model_type == "deep_ffnn":
        from src.models.deep_ffnn import DeepFFNN
        if config.netparams is None:
            raise ValueError("config.params cannot be None for deep_ffnn")
        return DeepFFNN(config.netparams)
    if model_type == "deep_ffnn_weight_norm_single_rescale":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_weight_norm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for weight_norm_deep_ffnn")
        return DeepFFNN_weight_norm(config.netparams)
    if model_type== "deep_ffnn_weight_norm_multi_channel_rescale":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_weight_norm_multi_channel_recale
        if config.netparams is None:
            raise ValueError("config.params cannot be None for weight_norm_deep_ffnn")
        return DeepFFNN_weight_norm_multi_channel_recale(config.netparams)
    if model_type == "deep_ffnn_weight_batch_norm":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_EMA_batch_weight_norm
        return DeepFFNN_EMA_batch_weight_norm(config.netparams)
    if model_type == "ffnn_normal_BN":
        from src.models.deep_ffnn import FFNN_with_BN
        return FFNN_with_BN(config.netparams)
    
    else:
        # If the model type is not supported, raise a ValueError
        raise ValueError(f"Unsupported model type: {model_type}")
    
    
    # if model_type =='vgg_custom':
    #     if config.params is None:
    #         raise ValueError("config.params cannot be None for VGG16_custom")
    #     return vgg_with_internal_performance_track_custom_classifier(config.params)
    
config_name = "basic_config"
@hydra.main(config_path=CONFIG_PATH, config_name=config_name)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    # get the network configuration:
    net_config = cfg.net
    
    # use the factory function to create the model instance
    
    model = model_factory(net_config)
    print('Model created successfully')
    print(model)

if __name__ == "__main__":
    
    # changing the config path if needed:
    
    # experiments/rank_tracking_in_shifting_task/cfg/rank_tracking_in_shifting_tasks_config_conv.yaml
    CONFIG_PATH = str(PROJECT_ROOT / "experiments/rank_tracking_in_shifting_task/cfg")
    config_name = "rank_tracking_in_shifting_tasks_config_conv"
    main()
    