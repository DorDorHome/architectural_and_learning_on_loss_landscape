from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from configs.configurations import NetConfig, RRContinuousBackpropConfig
from src.algos.AdamGnT import AdamGnT
from src.algos.supervised.base_learner import Learner
from src.algos.supervised.rr_gnt_fc import RR_GnT_for_FC


class RankRestoringCBP_for_FC(Learner):
    def __init__(
        self,
        net: nn.Module,
        config: RRContinuousBackpropConfig,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)

        if getattr(self.net, 'type', None) != 'FC':
            raise TypeError(
                f"RankRestoringCBP_for_FC requires net.type == 'FC', got {getattr(self.net, 'type', None)}"
            )

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay),
            )

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

        self.rr_gnt = RR_GnT_for_FC(
            net=self.net.layers,
            hidden_activation=hidden_activation,
            opt=self.opt,
            config=config,
            loss_func=self.loss_func,
            device=self.device,
        )

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)

        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.rr_gnt.rr_config.rrcbp_enabled:
            self.opt.zero_grad()
            self.rr_gnt.gen_and_test(features=self.previous_features, batch_input=x)

            # Re-run forward pass to get fresh features after potential network modifications
            with torch.no_grad():
                _, fresh_features = self.net.predict(x)
                self.previous_features = fresh_features

        return loss.detach(), output.detach()
