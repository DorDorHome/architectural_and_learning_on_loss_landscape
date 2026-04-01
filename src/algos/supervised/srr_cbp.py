import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, cast, Sequence

from src.algos.supervised.base_learner import Learner
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC
from src.algos.AdamGnT import AdamGnT
from configs.configurations import SRRCBPConfig, NetConfig

class SRR_CBP_for_FC(Learner):
    def __init__(
            self,
            net: nn.Module, 
            config: SRRCBPConfig,
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
        self.outgoing_random = config.outgoing_random
        
        self.lambda_0 = config.lambda_0
        self.gamma = config.gamma

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size, 
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay)
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
            raise TypeError(f"SRR_CBP_for_FC requires net.type == 'FC', got {getattr(self.net,'type',None)}")

        self.gnt = GnT_for_FC(
            net=self.net,
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

        self.pre_activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.pre_activations[layer_idx] = output
            return hook

        if hasattr(self.net, "get_plasticity_map"):
            plasticity_map = self.net.get_plasticity_map()
            for i, item in enumerate(plasticity_map):
                # We only track hidden layers, so we skip the last layer (which outputs to classes)
                if i < len(plasticity_map) - 1:
                    item['weight_module'].register_forward_hook(get_hook(i))
        else:
            # Fallback to legacy layers
            layers = cast(Sequence[nn.Module], self.net.layers)
            layer_idx = 0
            for i in range(0, len(layers), 2):
                if isinstance(layers[i], nn.Linear):
                    if layer_idx < len(self.gnt.ages):
                        layers[i].register_forward_hook(get_hook(layer_idx))
                    layer_idx += 1

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)
        
        output, features = self.net.predict(x)
        task_loss = self.loss_func(output, target)
        
        aso_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(self.gnt.ages)):
            if i not in self.pre_activations:
                continue
                
            ages = self.gnt.ages[i]
            mature_idx = torch.where(ages >= self.maturity_threshold)[0]
            young_idx = torch.where(ages < self.maturity_threshold)[0]
            
            if len(mature_idx) > 0 and len(young_idx) > 0:
                A = self.pre_activations[i]
                # For FC, A is [batch_size, num_units]
                # We want [num_units, batch_size]
                A = A.t()
                
                m = A.shape[1]
                
                A_mature = A[mature_idx, :].detach()
                A_young = A[young_idx, :]
                
                cov = torch.matmul(A_young, A_mature.t())
                
                weights = (self.gamma ** ages[young_idx]) / (2 * m**2)
                
                # Squared L2 norm of each row in cov
                row_norms_sq = torch.sum(cov ** 2, dim=1)
                
                layer_aso_loss = torch.sum(weights * row_norms_sq)
                aso_loss = aso_loss + layer_aso_loss
                
        aso_loss = self.lambda_0 * aso_loss
        total_loss = task_loss + aso_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        self.opt.zero_grad()
        self.gnt.gen_and_test(features=features)

        return task_loss.detach(), output.detach()


class SRR_CBP_for_ConvNet(Learner):
    def __init__(self,
                 net: nn.Module,
                 config: SRRCBPConfig,
                 netconfig: Optional[Union[NetConfig, None]] = None
                 ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)
        
        self.neurons_replacement_rate = config.neurons_replacement_rate
        self.decay_rate_utility_track = config.decay_rate_utility_track
        self.init = config.init
        self.util_type = config.util_type
        self.maturity_threshold = config.maturity_threshold
        
        self.lambda_0 = config.lambda_0
        self.gamma = config.gamma

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size, 
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay)
            )

        hidden_activation = None
        if netparams is not None:
            for attr_name in ('activation', 'act_type'):
                if hasattr(netparams, attr_name):
                    hidden_activation = getattr(netparams, attr_name)
                    break
        if hidden_activation is None:
            raise ValueError(f"hidden_activation must be specified in netparams.activation or netparams.act_type (netparams={netparams})")

        num_last_filter_outputs = self._calculate_last_filter_outputs()

        self.gnt = ConvGnT_for_ConvNet(
            net=self.net,
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

        self.pre_activations = {}
        self._register_hooks()

    def _calculate_last_filter_outputs(self) -> int:
        if hasattr(self.net, "get_plasticity_map"):
            try:
                plasticity_map = self.net.get_plasticity_map()
                for item in plasticity_map:
                    current_layer = item['weight_module']
                    outgoing_module = item['outgoing_module']
                    
                    if isinstance(current_layer, nn.Conv2d) and isinstance(outgoing_module, nn.Linear):
                        num_last_filter_outputs = outgoing_module.in_features // current_layer.out_channels
                        return max(1, int(num_last_filter_outputs))
                return 1
            except Exception:
                pass
        
        layers = cast(Sequence[nn.Module], self.net.layers)
        last_conv_idx = -1
        first_linear_idx = -1
        
        for i in range(0, len(layers), 2):
            if isinstance(layers[i], nn.Conv2d):
                last_conv_idx = i
            elif isinstance(layers[i], nn.Linear):
                if first_linear_idx == -1:
                    first_linear_idx = i
                break
        
        if last_conv_idx == -1 or first_linear_idx == -1:
            return 1
        last_conv = cast(nn.Conv2d, layers[last_conv_idx])
        first_linear = cast(nn.Linear, layers[first_linear_idx])
        try:
            num_last_filter_outputs = first_linear.in_features // last_conv.out_channels
            return max(1, int(num_last_filter_outputs))
        except Exception:
            return 1

    def _register_hooks(self):
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.pre_activations[layer_idx] = output
            return hook

        if hasattr(self.net, "get_plasticity_map"):
            plasticity_map = self.net.get_plasticity_map()
            for i, item in enumerate(plasticity_map):
                if i < len(plasticity_map) - 1:
                    item['weight_module'].register_forward_hook(get_hook(i))
        else:
            layers = cast(Sequence[nn.Module], self.net.layers)
            layer_idx = 0
            for i in range(0, len(layers), 2):
                if isinstance(layers[i], (nn.Conv2d, nn.Linear)):
                    if layer_idx < len(self.gnt.ages):
                        layers[i].register_forward_hook(get_hook(layer_idx))
                    layer_idx += 1

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)
        
        output, features = self.net.predict(x=x)
        task_loss = self.loss_func(output, target)
        
        aso_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(self.gnt.ages)):
            if i not in self.pre_activations:
                continue
                
            ages = self.gnt.ages[i]
            mature_idx = torch.where(ages >= self.maturity_threshold)[0]
            young_idx = torch.where(ages < self.maturity_threshold)[0]
            
            if len(mature_idx) > 0 and len(young_idx) > 0:
                A = self.pre_activations[i]
                
                # Reshape A to (num_units, m)
                if A.dim() == 4:
                    # Conv2d: A is [batch_size, channels, H, W]
                    # We want [channels, batch_size * H * W]
                    batch_size, channels, H, W = A.shape
                    A = A.transpose(0, 1).reshape(channels, -1)
                elif A.dim() == 2:
                    # Linear: A is [batch_size, num_units]
                    # We want [num_units, batch_size]
                    A = A.t()
                else:
                    continue
                    
                m = A.shape[1]
                
                A_mature = A[mature_idx, :].detach()
                A_young = A[young_idx, :]
                
                cov = torch.matmul(A_young, A_mature.t())
                
                weights = (self.gamma ** ages[young_idx]) / (2 * m**2)
                
                row_norms_sq = torch.sum(cov ** 2, dim=1)
                
                layer_aso_loss = torch.sum(weights * row_norms_sq)
                aso_loss = aso_loss + layer_aso_loss
                
        aso_loss = self.lambda_0 * aso_loss
        total_loss = task_loss + aso_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        
        self.opt.zero_grad()
        self.gnt.gen_and_test(features=features)

        return task_loss.detach(), output.detach()
