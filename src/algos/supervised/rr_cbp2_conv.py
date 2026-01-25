"""
Rank-Restoring Continual Backpropagation Learner for ConvNets (Version 2).

This learner implements the RR-CBP2 and RR-CBP-E2 algorithms for convolutional networks.
It follows the algorithm guide in RR_CBP_2_algorithm_guide.md.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, Module

from configs.configurations import NetConfig, RRCBP2Config
from src.algos.AdamGnT import AdamGnT
from src.algos.supervised.base_learner import Learner
from src.algos.supervised.rr_gnt2_conv import RR_GnT2_for_ConvNet


class RankRestoringCBP2_for_ConvNet(Learner):
    """
    Rank-Restoring Continual Backpropagation for ConvNets (Version 2).
    
    This learner combines standard backpropagation with the RR-CBP2/RR-CBP-E2 
    generate-and-test mechanism for neuron replacement.
    
    The main difference from standard CBP is that replaced neurons are initialized
    with Σ-orthogonal weights instead of random weights, which helps maintain
    full rank in the feature representations.
    
    Two modes:
    - RR-CBP2 (use_energy_budget=False): Unit Σ-norm for new weights
    - RR-CBP-E2 (use_energy_budget=True): Energy-budget controlled Σ-norm
    """
    
    def __init__(
        self,
        net: nn.Module,
        config: RRCBP2Config,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)

        # Initialize AdamGnT optimizer (required for proper state management)
        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay),
            )

        # Get hidden activation from netparams
        hidden_activation = None
        if netparams is not None:
            for attr_name in ('activation', 'act_type'):
                if hasattr(netparams, attr_name):
                    hidden_activation = getattr(netparams, attr_name)
                    break
        if hidden_activation is None:
            raise ValueError(
                "hidden_activation must be specified in netparams.activation or netparams.act_type"
            )

        # Calculate num_last_filter_outputs (spatial dimensions before flattening)
        num_last_filter_outputs = self._calculate_last_filter_outputs()

        # Initialize the RR-GnT2 module for generate-and-test
        self.rr_gnt = RR_GnT2_for_ConvNet(
            net=self.net,
            hidden_activation=hidden_activation,
            opt=self.opt,
            config=config,
            loss_func=self.loss_func,
            device=self.device,
            num_last_filter_outputs=num_last_filter_outputs,
        )

    def _calculate_last_filter_outputs(self) -> int:
        """
        Calculate the spatial dimensions (H x W) of the last convolutional layer
        before it gets flattened into a linear layer.
        """
        layers = cast(Sequence[Module], self.net.layers)
        last_conv_idx = -1
        first_linear_idx = -1
        
        # Find the last Conv2d layer and first Linear layer
        for i in range(0, len(layers), 2):
            if isinstance(layers[i], Conv2d):
                last_conv_idx = i
            elif isinstance(layers[i], Linear):
                if first_linear_idx == -1:
                    first_linear_idx = i
                break
        
        if last_conv_idx == -1 or first_linear_idx == -1:
            return 1
        
        last_conv = cast(Conv2d, layers[last_conv_idx])
        first_linear = cast(Linear, layers[first_linear_idx])
        
        try:
            num_last_filter_outputs: int = first_linear.in_features // last_conv.out_channels
            return max(1, int(num_last_filter_outputs))
        except Exception:
            return 1

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn from a batch of data.
        
        This method:
        1. Performs forward pass to get predictions and features
        2. Computes loss and performs backward pass
        3. Updates weights with optimizer step
        4. Performs generate-and-test with Σ-orthogonal replacement
        
        Args:
            x: Input batch tensor
            target: Target labels/values
            
        Returns:
            Tuple of (loss, output) tensors
        """
        x, target = x.to(self.device), target.to(self.device)

        # Forward pass to get predictions and hidden features
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # Backward pass and optimizer step
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # Generate-and-test with Σ-orthogonal replacement
        if self.rr_gnt.config.rrcbp_enabled:
            self.rr_gnt.gen_and_test(features=self.previous_features, batch_input=x)

            # Re-run forward pass to get fresh features after network modifications
            with torch.no_grad():
                _, fresh_features = self.net.predict(x)
                self.previous_features = fresh_features

        return loss.detach(), output.detach()

    def get_replacement_stats(self):
        """Get replacement statistics from the GnT module."""
        return self.rr_gnt.get_layer_stats()
