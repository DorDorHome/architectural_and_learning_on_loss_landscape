# factory function for supervised learning algorithms
# this should include all implemented SL typed learning method, including all backprop like methods.

# from typing import Union # for future implementation of configurations file that support dataclass for different objects.
from omegaconf import DictConfig, OmegaConf
import sys
# import the Backprop class in basic_backprop.py file, contained in the supervised folder:
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet, ContinualBackprop_for_FC
from src.algos.supervised.rr_cbp_conv import RankRestoringCBP_for_ConvNet
from src.algos.supervised.rr_cbp_fc import RankRestoringCBP_for_FC
from src.algos.supervised.rr_cbp2_fc import RankRestoringCBP2_for_FC
from src.algos.supervised.rr_cbp2_conv import RankRestoringCBP2_for_ConvNet
from configs.configurations import RRContinuousBackpropConfig, RRCBP2Config
from typing import Optional
import warnings

__all__ = ["create_learner"]

def create_learner(config: DictConfig, net, netconfig=None):
    """
    Create a learner instance based on config.

    Backwards compatibility notes:
    - Supports legacy type 'basic_continous_backprop'.
    - Accepts aliases: 'cbp', 'continuous_backprop'.
    - Infers network_class if missing using net.type heuristics.
    
    RR-CBP2 variants:
    - 'rr_cbp2': Rank-Restoring CBP v2 without energy budget (unit Σ-norm)
    - 'rr_cbp_e_2': Rank-Restoring CBP v2 with energy budget control
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
    elif normalized_type == 'rr_cbp':
        if not isinstance(config, RRContinuousBackpropConfig):
            if isinstance(config, DictConfig):
                config = RRContinuousBackpropConfig(**OmegaConf.to_container(config, resolve=True))
            elif isinstance(config, dict):
                config = RRContinuousBackpropConfig(**config)
            else:
                raise TypeError("RR-CBP requires RRContinuousBackpropConfig-compatible config")
            try:
                config.network_class = net_cls
            except Exception:
                pass
        if net_cls == 'fc':
            return RankRestoringCBP_for_FC(net, config, netconfig)
        if net_cls == 'conv':
            return RankRestoringCBP_for_ConvNet(net, config, netconfig)
        raise ValueError(
            f"Unsupported network_class '{net_cls}' for rank-restoring CBP (net.type={getattr(net, 'type', None)})"
        )
    elif normalized_type in {'rr_cbp2', 'rr_cbp_e_2'}:
        # Rank-Restoring CBP Version 2
        # - 'rr_cbp2': Without energy budget (use_energy_budget=False)
        # - 'rr_cbp_e_2': With energy budget (use_energy_budget=True)
        if not isinstance(config, RRCBP2Config):
            if isinstance(config, DictConfig):
                config_dict = OmegaConf.to_container(config, resolve=True)
            elif isinstance(config, dict):
                config_dict = config.copy()
            else:
                raise TypeError("RR-CBP2 requires RRCBP2Config-compatible config")
            
            # Set use_energy_budget based on learner type
            if normalized_type == 'rr_cbp_e_2':
                config_dict['use_energy_budget'] = True
            elif normalized_type == 'rr_cbp2':
                config_dict['use_energy_budget'] = False
            
            config = RRCBP2Config(**config_dict)
            
            try:
                config.network_class = net_cls
            except Exception:
                pass
        else:
            # Config is already RRCBP2Config, but ensure use_energy_budget matches learner type
            if normalized_type == 'rr_cbp_e_2' and not config.use_energy_budget:
                config.use_energy_budget = True
            elif normalized_type == 'rr_cbp2' and config.use_energy_budget:
                config.use_energy_budget = False
        
        if net_cls == 'fc':
            return RankRestoringCBP2_for_FC(net, config, netconfig)
        if net_cls == 'conv':
            return RankRestoringCBP2_for_ConvNet(net, config, netconfig)
        raise ValueError(
            f"Unsupported network_class '{net_cls}' for rank-restoring CBP2 (net.type={getattr(net, 'type', None)})"
        )
    else:
        raise ValueError(f"Unsupported learner type: {learner_type}")
