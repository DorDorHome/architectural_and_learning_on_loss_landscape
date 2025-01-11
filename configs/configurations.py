# dataclasses for configurations objects:

from torch._C import device
from dataclasses import dataclass, field 
from typing import Optional, Union
from hydra.core.config_store import ConfigStore
import torch

@dataclass
class DataConfig:
    dataset: str= 'mnist'
    use_torchvision: Optional[Union[None, bool]] = True
    data_path: Optional[Union[None, str]] = None
    num_classes: Optional[Union[None, int]] = 10
    shuffle: Optional[Union[None, bool]] = False
    transform: Optional[Union[None, str]] = None
    class Config:
        version_base = "1.1"


@dataclass
class NetParams:
    pretrained: Optional[Union[None, bool]] = False
    num_classes: Optional[Union[None, int]] = 10
    initialization: Optional[Union[None, str]] = 'kaiming'
    in_channels: Optional[Union[None, int]] = 1
    out_channels: Optional[Union[None, int]] = 10
    kernel_size: Optional[Union[None, int]] = 5
    class Config:
        version_base = "1.1"

#dataclass for network configurations
@dataclass
class NetConfig:
    type: str
    params: Optional[Union[None, NetParams]] = None
    class Config:
        version_base = "1.1"

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
    loss: str = 'cross_entropy'
    to_perturb: Optional[bool] = False
    perturb_scale: Optional[float] = 0.1
    # previous_features: Optional[Union[None, torch.Tensor]] = None
    # latest_gradients: Optional[Union[None, torch.Tensor]] = None
    class Config:
        version_base = "1.1"

@dataclass
class BackpropConfig(BaseLearnerConfig):
    type: str = 'backprop'
    to_perturb: bool = False
    class Config:
        version_base = "1.1"

@dataclass
class EvaluationConfig:
    eval_freq_epoch: int = 1
    eval_metrics:  list = field(default_factory=lambda: ['accuracy', 'loss'])  # Correct
    save_dir: str ='results_raw'
    # save_name: str = 'basic_training'


 
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
class logConfig:
    save_dir: str = 'results_raw'
    save_name: str = 'basic_training'
    class Config:
        version_base = "1.1"

@dataclass
class ExperimentConfig:
    use_wandb: bool = False
    runs: int = 1
    seed: Optional[int] = None
    device: str = 'cuda'
    epochs: int = 10
    batch_size: int = 128
    data: DataConfig = DataConfig()
    net: NetConfig = NetConfig(type='ConvNet')
    learner: Union[BackpropConfig] = BackpropConfig()
    evaluation: Union[EvaluationConfig, None] = EvaluationConfig()
    
    
    def __post_init__(self):
        if self.net.num_classes != self.data.num_classes:
            raise ValueError("net.num_classes must match data.num_classes")
    class Config:
        version_base = "1.1"

# Register the configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)