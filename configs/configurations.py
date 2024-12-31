# dataclasses for configurations objects:

from torch._C import device
from dataclasses import dataclass, field 
from typing import Optional, Union
from hydra.core.config_store import ConfigStore
import torch

@dataclass
class DataConfig:
    dataset: str= 'mnist'
    data_path: Optional[Union[None, str]] = None
    num_classes: Optional[Union[None, int]] = 10
    shuffle: Optional[Union[None, bool]] = False
    transform: Optional[Union[None, str]] = None

@dataclass
class NetParams:
    num_classes: int = 10

@dataclass
class BaseLearnerConfig:
    type: str 
    device: str = 'cuda'
    opt: str = 'adam' #or 'sgd'
    step_size: float= 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    weight_decay: float = 0.0
    momentum: Optional[float] = 0.0
    loss: str = 'nll'
    to_perturb: Optional[bool] = False
    perturb_scale: Optional[float] = 0.1
    previous_features: Optional[Union[None, torch.Tensor]] = None
    latest_gradients: Optional[Union[None, torch.Tensor]] = None
   
@dataclass
class BackpropConfig(BaseLearnerConfig):
    type: str = 'backprop'
    to_perturb: bool = False
 
# @dataclass
# class ContinualBackpropConfig(BaseLearnerConfig):
#     replacement_rate: float = 0.001
#     decay_rate: float = 0.9
#     maturity_threshold: int = 100
#     util_type: str = 'contribution'
#     init: str = 'kaiming'
#     accumulate: bool = False
#     outgoing_random: bool = False



@dataclass
class ExperimentConfig:
    runs: int = 1
    seed: Optional[int] = None
    device: str = 'cuda'
    data: DataConfig = DataConfig()
    net: NetParams = NetParams()
    learner: Union[BackpropConfig] = BackpropConfig()
    
    def __post_init__(self):
        # Ensure that network.num_classes matches data.num_classes
        if self.net.num_classes != self.data.num_classes:
            raise ValueError("network.params.num_classes must match data.num_classes")
        
# Register the configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)