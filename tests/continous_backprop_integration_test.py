import sys
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configuration Mocks
@dataclass
class MockConfig:
    neurons_replacement_rate: float = 0.5 # Aggressive to ensure events happen
    decay_rate_utility_track: float = 0.9
    maturity_threshold: int = 0
    util_type: str = 'contribution'
    init: str = 'kaiming'
    opt: str = 'adam'
    step_size: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    weight_decay: float = 0.0
    device: str = 'cpu'
    
    # Check configurations.py BaseLearnerConfig & ContinuousBackpropConfig for missing fields
    loss: str = 'cross_entropy'
    accumulate: bool = False
    outgoing_random: bool = False
    use_grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    enable_cuda1_workarounds: bool = False
    additional_regularization: str = None
    lambda_orth: float = None
    to_perturb: bool = False
    perturb_scale: float = 0.1
    momentum: float = 0.0
@dataclass
class MockNetParams:
    activation: str = 'relu'

@dataclass
class MockNetConfig:
    netparams: MockNetParams = MockNetParams()

from src.models.conv_net import ConvNet
from configs.configurations import NetParams
from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet
from src.algos.gnt import ConvGnT_for_ConvNet

# --- Subclass to force the "New" behavior without editing source yet ---
class ModernContinuousBackprop(ContinuousBackprop_for_ConvNet):
    def __init__(self, net, config, netconfig):
        # We manually init this to replicate the class structure but pass different args to GnT
        # This is a bit hacker-ish but allows testing the logic change in isolation
        
        # 1. Standard super init (which will unfortunately create the Legacy GnT)
        super().__init__(net, config, netconfig)
        
        # 2. OVERWRITE with the "Modern" GnT (The change we want to make)
        # We assume self.gnt was created by super using net.layers.
        # We explicitly recreate it using 'net' (the module)
        
        num_last_filter_outputs = self._calculate_last_filter_outputs()
        
        self.gnt = ConvGnT_for_ConvNet(
            net=self.net, # <--- THIS IS THE CHANGE TO BE TEstED
            hidden_activation='relu',
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            init=self.init,
            num_last_filter_outputs=num_last_filter_outputs,
            util_type=self.util_type,
            maturity_threshold=self.maturity_threshold,
            device=self.device,
        )

# --- Subclass to enforce "Old" behavior explicitly ---
class LegacyContinuousBackprop(ContinuousBackprop_for_ConvNet):
    def __init__(self, net, config, netconfig):
        super().__init__(net, config, netconfig)
        # Assuming current source code uses self.net.layers, this wrapper is just identity
        # IF source code changes, this wrapper would need to force self.net.layers
        pass 

def run_simulation(learner_cls, name):
    print(f"Running simulation for: {name}")
    torch.manual_seed(100)
    
    # 1. Setup Net and Learner
    net_config = NetParams(num_classes=10, input_height=32, input_width=32, activation='relu')
    net = ConvNet(net_config)
    
    cb_config = MockConfig()
    net_wrapper_config = MockNetConfig()
    
    learner = learner_cls(net, cb_config, net_wrapper_config)
    
    # 2. Check if GnT detected Map
    detected_map = learner.gnt.use_map
    print(f"  [{name}] Internal GnT use_map: {detected_map}")
    
    # 3. Training Loop
    metrics = []
    
    # Data batch
    x = torch.randn(5, 3, 32, 32)
    y = torch.randint(0, 10, (5,))
    
    for i in range(5):
        loss, _ = learner.learn(x, y)
        
        # Record Model State hash
        w_sum = sum(p.sum().item() for p in net.parameters())
        if hasattr(learner.opt, 'state'):
            opt_sum = 0.0
            for group in learner.opt.param_groups:
                for p in group['params']:
                    if p in learner.opt.state:
                        s = learner.opt.state[p]
                        if 'exp_avg' in s: opt_sum += s['exp_avg'].sum().item()
        
        metrics.append((loss.item(), w_sum, opt_sum))
        
    return metrics

if __name__ == "__main__":
    
    # Run Legacy (Passing List)
    print("--- 1. Legacy Run ---")
    legacy_metrics = run_simulation(ContinuousBackprop_for_ConvNet, "Legacy(Current)")

    # Run Modern (Passing Module)
    # This mocks the edit 'net=self.net'
    print("\n--- 2. Modern Run (Proposed Edit) ---")
    modern_metrics = run_simulation(ModernContinuousBackprop, "Modern(Proposed)")
    
    print("\n--- COMPARISON ---")
    all_match = True
    for i, (leg, mod) in enumerate(zip(legacy_metrics, modern_metrics)):
        if leg != mod:
            all_match = False
            print(f"Step {i} MISMATCH:\n  Leg: {leg}\n  Mod: {mod}")
        else:
            print(f"Step {i}: Match")
            
    if all_match:
        print("\n✅ INTEGRATION SUCCESS: Current 'ContinuousBackprop' can be upgraded to pass 'self.net' safely.")
        print("   The logic remains identical for ConvNets without norms.")
    else:
        print("\n❌ INTEGRATION FAIL: The proposed change alters training physics.")