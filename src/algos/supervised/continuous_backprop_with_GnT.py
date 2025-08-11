# continuous backprop via generate and test method:
# this mainly uses code snippets from loss of plasticity paper

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union, Tuple, Sequence, cast  # noqa: F401

# import the BackpropConfig class from the config file in configs folder in the
# parent directory of the project:
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# base learner class:
from src.algos.supervised.base_learner import Learner

# import the gnt class from the gnt.py file in the same directory:
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC
from src.algos.AdamGnT import AdamGnT
from configs.configurations import ContinuousBackpropConfig, NetConfig


class ContinualBackprop_for_FC(Learner):
    """
    The Continual Backprop algorithm, used in https://arxiv.org/abs/2108.06325v3
    
    Only works for FC model, as it uses the predict method of the FC class.
    
    For easy tracking of features, the features are stored in the `previous_features` attribute
    after each forward pass. 

    This allows retrieval of the features during the generate-and-test step, and possibly later use in other methods, 
    such as tracking rank of features of other analyses.
    
    """
    def __init__(
            self,
            net: nn.Module, 
            config: ContinuousBackpropConfig,
            netconfig: Optional[Union[NetConfig, None]] = None
    ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)
        
        self.neurons_replacement_rate = config.neurons_replacement_rate
        self.decay_rate_utility_track = config.decay_rate_utility_track
        self.maturity_threshold = config.maturity_threshold
        self.util_type = config.util_type
        self.init = config.init
        self.accumulate = config.accumulate
        self.outgoing_random = config.outgoing_random  # NOTE: currently unused in GnT init

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size, 
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay)  # type: ignore[arg-type]
            )

        hidden_activation = None
        if netparams is not None:
            for attr_name in ('activation', 'act_type'):
                if hasattr(netparams, attr_name):
                    hidden_activation = getattr(netparams, attr_name)
                    break
        if hidden_activation is None:
            raise ValueError(f"hidden_activation must be specified in netparams.activation or netparams.act_type (netparams={netparams})")

        if getattr(self.net, 'type', None) != 'FC':
            raise TypeError(f"ContinualBackprop_for_FC requires net.type == 'FC', got {getattr(self.net,'type',None)}")

        self.gnt: Optional[GnT_for_FC] = GnT_for_FC(
            net=self.net.layers,
            hidden_activation=hidden_activation,
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            maturity_threshold=self.maturity_threshold,
            util_type=self.util_type,
            device=self.device,
            loss_func=self.loss_func,
            init=self.init,
            accumulate=self.accumulate,
        )

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)
        
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # take a generate-and-test step
        # Clear grads before structural adaptation; GnT may inspect or create params expecting clean .grad buffers
       
        self.opt.zero_grad()
        if type(self.gnt) is GnT_for_FC: # original: GnT:
            self.gnt.gen_and_test(features=self.previous_features)

        # if self.loss_func == F.cross_entropy:
        return loss.detach(), output.detach()

        # return loss.detach()

class ContinuousBackprop_for_ConvNet(Learner):
    """Continual Backprop algorithm for ConvNets."""
    def __init__(self,
                 net: nn.Module,
                 config: ContinuousBackpropConfig,
                 netconfig: Optional[Union[NetConfig, None]] = None
                 ):
        # Extract netparams from netconfig if provided
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)
        
        # Store config-specific parameters
        self.neurons_replacement_rate = config.neurons_replacement_rate
        self.decay_rate_utility_track = config.decay_rate_utility_track
        self.init = config.init
        self.util_type = config.util_type
        self.maturity_threshold = config.maturity_threshold

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size, 
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay)  # type: ignore[arg-type]
            )

        hidden_activation = None
        if netparams is not None:
            for attr_name in ('activation', 'act_type'):
                if hasattr(netparams, attr_name):
                    hidden_activation = getattr(netparams, attr_name)
                    break
        if hidden_activation is None:
            raise ValueError(f"hidden_activation must be specified in netparams.activation or netparams.act_type (netparams={netparams})")

        # Calculate num_last_filter_outputs from the network structure
        # This represents the spatial dimensions of the last conv layer before flattening
        num_last_filter_outputs = self._calculate_last_filter_outputs()

        # define the generate-and-test object for the given network
        self.gnt = ConvGnT_for_ConvNet(
            net=self.net.layers,
            hidden_activation=hidden_activation,
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            init=self.init,
            num_last_filter_outputs=num_last_filter_outputs,
            util_type=self.util_type,
            maturity_threshold=self.maturity_threshold,
            device=self.device,
        )

    def _calculate_last_filter_outputs(self) -> int:
        """
        Calculate the spatial dimensions (H x W) of the last convolutional layer
        before it gets flattened into a linear layer.
        
        This value represents how many spatial locations each filter in the last
        conv layer produces, which is needed for proper feature replacement when
        transitioning from Conv2d to Linear layers.
        """
        layers = cast(Sequence[nn.Module], self.net.layers)  # type: ignore[assignment]
        last_conv_idx = -1
        first_linear_idx = -1
        
        # Find the last Conv2d layer and first Linear layer
        for i in range(0, len(layers), 2):  # step by 2 since layers alternate conv/activation
            if isinstance(layers[i], nn.Conv2d):
                last_conv_idx = i
            elif isinstance(layers[i], nn.Linear):
                if first_linear_idx == -1:
                    first_linear_idx = i
                break
        
        if last_conv_idx == -1 or first_linear_idx == -1:
            # Fallback: if we can't determine the structure, use a default value
            # This handles cases where the network doesn't have the expected Conv->Linear transition
            return 1
        last_conv = cast(nn.Conv2d, layers[last_conv_idx])
        first_linear = cast(nn.Linear, layers[first_linear_idx])
        try:
            num_last_filter_outputs: int = first_linear.in_features // last_conv.out_channels  # type: ignore[attr-defined]
            return max(1, int(num_last_filter_outputs))
        except Exception:
            return 1

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        # move data to device:
        x, target = x.to(self.device), target.to(self.device)
        
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        loss.backward()
        self.opt.step()
        
        # Clear grads before structural adaptation; GnT may inspect or create params expecting clean .grad buffers
        self.opt.zero_grad()

        # take a generate-and-test step
        self.gnt.gen_and_test(features=self.previous_features)

        return loss.detach(), output
