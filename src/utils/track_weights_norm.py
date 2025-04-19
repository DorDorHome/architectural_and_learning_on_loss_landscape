import torch
import torch.nn as nn
import numpy as np

def get_module_weight_magnitude(module: nn.Module):
    """
    Given a module, if it has a 'weight' attribute with at least 2 dimensions,
    compute and return the mean of the absolute values of the weight matrix.
    Biases (or 1D tensors) are ignored.
    """
    if not hasattr(module, "weight") or module.weight is None:
        return None
    weight_tensor = module.weight.detach()
    if weight_tensor.ndim < 2:
        # Likely this is a bias-like tensor or something we want to ignore.
        return None
    # Compute mean of absolute weight values
    return float(torch.abs(weight_tensor).mean().item())


def track_weight_stats(model, layer_identifiers=None):
    """
    Traverse the model and compute mean absolute weight statistics on layers of interest.
    
    Parameters:
      model (nn.Module): The PyTorch model (custom or torchvision's ResNet, etc.)
      layer_identifiers (list): Optional. A list of identifiers specifying which layers to track.
          The list can contain integers (which refer to the ordering of eligible sub-modules)
          or strings (to be matched to the module names as given by model.named_modules()).
          If omitted, all sub-modules with a weight parameter (having at least 2 dimensions) are tracked.
          
    Returns:
      dict: A dictionary mapping an identifier (either the module name or an integer index) 
            to the mean absolute value of its weight matrix.
    """
    
    tracked_stats = {}
    
    # Build a list of candidate modules along with their names.
    # Using .named_modules() we traverse recursively all submodules.
    candidate_modules = []
    for name, module in model.named_modules():
        # Skip the top-level module (with empty string name) if desired
        if name == "":
            continue
        stat = get_module_weight_magnitude(module)
        if stat is not None:
            candidate_modules.append((name, module))
    
    # If layer_identifiers is not provided track all candidate modules.
    if layer_identifiers is None:
        # Save with the module name as identifier.
        for name, module in candidate_modules:
            stat = get_module_weight_magnitude(module)
            tracked_stats[name] = stat
        return tracked_stats

    # Otherwise, allow the identifiers to refer either to indices or to module names.
    # Build mapping from index and name for convenient lookup.
    for idx, (name, module) in enumerate(candidate_modules):
        stat = get_module_weight_magnitude(module)
        # If the identifier list includes the integer index, track it.
        if idx in layer_identifiers:
            tracked_stats[f"idx_{idx}::{name}"+"_mean_abs_weight"] = stat
        # Or if the module name is in the list, track it.
        elif name in layer_identifiers:
            tracked_stats[name+"_mean_abs_weight"] = stat

    return tracked_stats


# ============================================================
# Example usage with your custom DeepFFNN and torchvision ResNet.
# ============================================================

# Example 1: With your custom model.

if __name__ == "__main__":
    from typing import Any
    import sys
    import pathlib
    import torch.utils

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    print(PROJECT_ROOT)
    sys.path.append(str(PROJECT_ROOT))
    from configs.configurations import ExperimentConfig
    from src.models.deep_ffnn import DeepFFNN

    # Dummy configuration object
    class Config:
        def __init__(self):
            self.input_size = 10
            self.num_features = 20
            self.num_outputs = 5
            self.num_hidden_layers = 2
            self.act_type = 'relu'
            self.initialization = 'kaiming'

    config = Config()
    custom_model = DeepFFNN(config)

    # Now, suppose you want to track only the layers at specific indices.
    # For instance, here we select indices 0 and 2 among the eligible modules.
    custom_stats = track_weight_stats(custom_model)#, layer_identifiers=[0, 2])
    print("Custom Model Weight Stats:")
    print(custom_stats)


    # Example 2: With a torchvision ResNet (or any standard model)
    # For demonstration, we build a small ResNet-like model.
    from torchvision.models import resnet18

    resnet_model = resnet18(pretrained=False)

    # Suppose you want to track all eligible layers in ResNet by name.
    resnet_stats = track_weight_stats(resnet_model)
    print("\nResNet Weight Stats (tracking all eligible layers):")
    for name, stat in resnet_stats.items():
        print(f"{name}: {stat}")
        
