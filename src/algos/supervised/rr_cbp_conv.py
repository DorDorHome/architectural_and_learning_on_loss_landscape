from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from configs.configurations import NetConfig, RRContinuousBackpropConfig
from src.algos.AdamGnT import AdamGnT
from src.algos.supervised.base_learner import Learner
from src.algos.supervised.rr_gnt_conv import RR_GnT_for_ConvNet


class RankRestoringCBP_for_ConvNet(Learner):
    """Rank-restoring Continual Backprop learner for convolutional networks."""

    def __init__(
        self,
        net: nn.Module,
        config: RRContinuousBackpropConfig,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)

        if config.opt == "adam":
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay),
            )

        hidden_activation = None
        if netparams is not None:
            for attr_name in ("activation", "act_type"):
                if hasattr(netparams, attr_name):
                    hidden_activation = getattr(netparams, attr_name)
                    break
        if hidden_activation is None:
            raise ValueError(
                "hidden_activation must be specified in netparams.activation or netparams.act_type"
            )

        num_last_filter_outputs = self._calculate_last_filter_outputs()

        self.rr_gnt = RR_GnT_for_ConvNet(
            net=self.net.layers,
            hidden_activation=hidden_activation,
            opt=self.opt,
            config=config,
            loss_func=self.loss_func,
            device=self.device,
            num_last_filter_outputs=num_last_filter_outputs,
        )

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)

        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.opt.zero_grad()
        self.rr_gnt.gen_and_test(features=self.previous_features, batch_input=x)

        # Re-run forward pass to get fresh features after potential network modifications
        with torch.no_grad():
            _, fresh_features = self.net.predict(x)
            self.previous_features = fresh_features

        return loss.detach(), output.detach()

    def _calculate_last_filter_outputs(self) -> int:
        layers = getattr(self.net, "layers", None)
        if layers is None:
            return 1
        last_conv_idx = -1
        first_linear_idx = -1
        for idx, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                last_conv_idx = idx
            elif isinstance(layer, nn.Linear):
                first_linear_idx = idx
                break
        if last_conv_idx == -1 or first_linear_idx == -1:
            return 1
        conv_layer = layers[last_conv_idx]
        linear_layer = layers[first_linear_idx]
        try:
            outputs = linear_layer.in_features // conv_layer.out_channels
            return max(1, int(outputs))
        except Exception:
            return 1
