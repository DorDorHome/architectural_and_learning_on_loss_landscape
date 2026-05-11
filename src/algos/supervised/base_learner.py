# Objectives for Learner:
# Initialization: Handle the setup of the neural network, optimizer, and loss function.
# Learning Process: Define a generic learn method that can be extended or overridden by subclasses.
# Utility Functions: Include any shared utility methods that multiple learners might use.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Union, Any
from torch.optim import optimizer
from omegaconf import DictConfig
from configs.configurations import *
from src.losses.orthogonality import RegularizedLoss_SVD_conv, KernelSORegularizer

class Learner(ABC):
    """
    abstract base class for different learning algorithms
    """

    def __init__(self, net: nn.Module, config: BaseLearnerConfig, netconfig: Optional[Union[NetParams, LinearNetParams, None]] = None):

        """handle the setup of networks(agents), optimizer, and loss function."""
        # network/agent:
        self.config = config
        self.netconfig = netconfig
        
        self.device = config.device
        self.net = net.to(self.device)

        # intialize optimizer 
        self.opt = self._init_optimizer(config)
        self.loss= config.loss
        # initialize loss function
        self.loss_func = self._init_loss(config.loss)
        
        # for more complicated implementations that need to keep track of previous features
        self.previous_features = None
        
        # for algorithms that need to keep track of previous gradients
        self.latest_gradients = None

        # Perturbation defaults; subclasses overwrite these from their config when used.
        # Declared here so the shared `perturb()` method has well-defined attributes.
        self.to_perturb: bool = bool(getattr(config, 'to_perturb', False))
        self.perturb_scale: float = float(getattr(config, 'perturb_scale', 0.0))
        
        

    def _init_optimizer(self, config: BaseLearnerConfig):
        
        opt = config.opt# need to override with the optimizer of choice
        
        step_size = config.step_size
        beta_1 = config.beta_1
        beta_2 = config.beta_2
        weight_decay = config.weight_decay
        momentum = config.momentum if hasattr(config, 'momentum') and config.momentum is not None else 0.0
        

        # initialize loss function was moved to the __init__ method
        # self.loss_func = self._init_loss(self.loss)
        
        if opt == 'sgd':
            optimizer = optim.SGD(self.net.parameters(),
                                  lr=step_size,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        elif opt == 'adam':
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=step_size,
                                   betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)
        elif opt == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(),
                                   lr=step_size,
                                   betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)
        
        else:
            raise ValueError(f'Optimizer {opt} not implemented')
        
        return optimizer
    
    def _init_loss(self, loss: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        loss_funcs: dict[str,  Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            'cross_entropy': F.cross_entropy,
            'mse': F.mse_loss
        }
        
        if loss not in loss_funcs:
            raise ValueError(f"Unsupported loss type: {loss}")
        
        # Check if regularization is enabled
        main_loss_func = loss_funcs[loss]
                
        if hasattr(self.config, 'additional_regularization') and self.config.additional_regularization:
            lambda_orth = getattr(self.config, 'lambda_orth', 1e-4)
            
            if self.config.additional_regularization == 'SVD_Orthogonal':
                return RegularizedLoss_SVD_conv(
                    main_loss_func=main_loss_func,
                    model=self.net,
                    lambda_orth=lambda_orth,
                    allow_svd_values_negative=False  # Default to False for now
                )
            elif self.config.additional_regularization == 'Kernel SO':
                normalization_mode = getattr(self.config, 'normalization_mode', "naive mse sum correction")
                return KernelSORegularizer(
                    main_loss_func=main_loss_func,
                    model=self.net,
                    lambda_orth=lambda_orth,
                    normalization_mode=normalization_mode
                )
            else:
                raise ValueError(f"Unsupported additional_regularization: {self.config.additional_regularization}")
        else:
            return main_loss_func



    @abstractmethod
    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Any:
        """learn from a batch of data"""
        pass
    
    def _forward(self, x: torch.Tensor):# -> Any | Any:
        return self.net.predict(x)

    @staticmethod
    def _iter_perturbable_modules(net: nn.Module) -> Iterator[nn.Module]:
        """
        Yield the weight-bearing modules whose parameters perturb() should noise.

        Prefers `net.get_plasticity_map()` when available (the source of truth for
        layer topology). Each map entry contributes its `weight_module`, and the
        final entry's `outgoing_module` is also included so the output layer is
        perturbed too. Norm layers (BN/LN) are intentionally excluded because
        `get_plasticity_map` does not list them as `weight_module`.

        Falls back to the legacy `net.layers[i*2]` walk when no plasticity map
        is exposed, to preserve behavior for models that haven't been migrated
        to the map-based topology yet.
        """
        if hasattr(net, "get_plasticity_map"):
            try:
                pmap = net.get_plasticity_map()
            except Exception:
                pmap = None
            if pmap:
                seen: set[int] = set()
                for item in pmap:
                    module = item.get('weight_module')
                    if module is not None and id(module) not in seen:
                        seen.add(id(module))
                        yield module
                final_module = pmap[-1].get('outgoing_module')
                if final_module is not None and id(final_module) not in seen:
                    yield final_module
                return

        layers = getattr(net, "layers", None)
        if layers is None:
            return
        for i in range(0, len(layers), 2):
            yield layers[i]

    def perturb(self) -> None:
        """
        Add Gaussian noise (std = `self.perturb_scale`) to the weight and bias
        of every perturbable module in `self.net`.

        Module discovery is delegated to `_iter_perturbable_modules`, which
        prefers `get_plasticity_map` over the legacy even-index walk so the
        method behaves correctly for models with interleaved norm layers.
        """
        with torch.no_grad():
            for module in self._iter_perturbable_modules(self.net):
                weight = getattr(module, 'weight', None)
                if weight is not None:
                    weight.add_(
                        torch.empty_like(weight).normal_(
                            mean=0.0, std=self.perturb_scale
                        )
                    )
                bias = getattr(module, 'bias', None)
                if bias is not None:
                    bias.add_(
                        torch.empty_like(bias).normal_(
                            mean=0.0, std=self.perturb_scale
                        )
                    )