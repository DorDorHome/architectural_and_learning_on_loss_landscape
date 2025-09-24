# factory function for supervised learning algorithms
# this should include all implemented SL typed learning method, including all backprop like methods.

# from typing import Union # for future implementation of configurations file that support dataclass for different objects.

from omegaconf import DictConfig
import sys
# import the Backprop class in basic_backprop.py file, contained in the supervised folder:
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet, ContinualBackprop_for_FC
from typing import Optional
import warnings

def create_learner(config: DictConfig, net, netconfig=None):
    """
    Create a learner instance based on config.

    Backwards compatibility notes:
    - Supports legacy type 'basic_continous_backprop'.
    - Accepts aliases: 'cbp', 'continuous_backprop'.
    - Infers network_class if missing using net.type heuristics.
    """

    learner_type = getattr(config, 'type', None)
    if learner_type is None:
        raise ValueError("Learner config must include a 'type' field.")

    # Normalize learner type aliases
    normalized_type = learner_type
    if learner_type in {'basic_continous_backprop', 'basic_continuous_backprop'}:
        warnings.warn(
            "Learner type 'basic_continous_backprop' is deprecated. Use 'continuous_backprop' or 'cbp' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized_type = 'continuous_backprop'
    elif learner_type in {'cbp', 'continuous_backprop'}:
        normalized_type = 'continuous_backprop'

    # Infer network_class if missing
    net_cls: Optional[str] = getattr(config, 'network_class', None)
    if net_cls is None:
        net_type = getattr(net, 'type', getattr(net, '__class__', type('x',(object,),{})).__name__)
        net_type_str = str(net_type).lower()
        if any(token in net_type_str for token in ['convnet', 'resnet', 'vgg', 'cnn']):
            net_cls = 'conv'
        elif any(token in net_type_str for token in ['ffnn', 'mlp', 'fc']):
            net_cls = 'fc'
        else:
            net_cls = 'other'
        # Reflect inference back into config if it's a mutable OmegaConf object
        try:
            config.network_class = net_cls
        except Exception:
            pass

    if normalized_type == 'backprop':
        return Backprop(net, config, netconfig)
    elif normalized_type == 'continuous_backprop':
        if net_cls == 'conv':
            return ContinuousBackprop_for_ConvNet(net, config, netconfig)
        elif net_cls == 'fc':
            return ContinualBackprop_for_FC(net, config, netconfig)
        else:
            raise ValueError(f"Unsupported network_class '{net_cls}' for continuous_backprop (net.type={getattr(net,'type',None)})")
    else:
        raise ValueError(f"Unsupported learner type: {learner_type}")
    # for support of torchvision models: