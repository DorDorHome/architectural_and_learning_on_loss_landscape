import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
from src.models.conv_net import ConvNet
from src.models.deep_ffnn import DeepFFNN
from configs.configurations import NetParams, LinearNetParams
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC
from src.algos.AdamGnT import AdamGnT  # Required to trigger opt state clearing logic

def get_stats_from_run(mode):
    logger.info(f"\n[TEST ENGINE] Starting Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'
    
    config = NetParams(num_classes=10, input_height=32, input_width=32, activation='relu')
    net = ConvNet(config)
    
    # Use AdamGnT to verify state clearing logic works
    optimizer = AdamGnT(net.parameters(), lr=0.01)
    
    # --- MODE SELECTION ---
    if mode == 'legacy':
        gnt_input = net.layers
        logger.info("[Setup] Passing 'net.layers' (List) to GnT (Simulating current ContinuousBackprop)")
    elif mode == 'module':
        gnt_input = net
        logger.info("[Setup] Passing 'net' (ConvNet Object) to GnT (Expects get_plasticity_map)")
    
    try:
        gnt = ConvGnT_for_ConvNet(
            net=gnt_input,  
            hidden_activation='relu',
            opt=optimizer,
            replacement_rate=0.5,
            decay_rate=0.9,
            maturity_threshold=0,
            init='kaiming',
            device=device,
            # CRITICAL: 32x32 input -> 2x2 spatial output after 3 convs & pools = 4 outputs
            num_last_filter_outputs=4 
        )
        mode_detected = "Map" if gnt.use_map else "Legacy"
        logger.info(f"[Init] Success. Detected {gnt.num_hidden_layers} hidden layers. Mode={mode_detected}")
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(10, 3, 32, 32) 
    
    # Run a few steps to populate optimizer state
    for step in range(3):
        # 1. Forward
        _, features_list = net.predict(dummy_input)
        
        # 2. Fake a gradient step so optimizer state exists
        loss = features_list[-1].sum()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 3. GnT Step (Modifies weights and Opt state)
        try:
            gnt.gen_and_test(features_list)
        except Exception as e:
            logger.error(f"[Step {step}] GnT CRASH: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Snapshot Stats
        # A. Utility values
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        
        # B. Checksum of Weights
        weights_sum = 0.0
        for name, param in net.named_parameters():
             if 'weight' in name: weights_sum += param.data.sum().item()
        
        # C. Checksum of Optimizer State (Critical for Adam)
        opt_state_sum = 0.0
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    opt_state_sum += state['exp_avg'].sum().item()
                if 'exp_avg_sq' in state:
                    opt_state_sum += state['exp_avg_sq'].sum().item()

        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4), 'opt_sum': round(opt_state_sum, 4)})
        
    logger.info(f"[Result] Mode='{mode}' finished successfully.")
    return stats

def get_stats_from_run_fc(mode):
    logger.info(f"\n[TEST ENGINE] Starting FC Test Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'

    config = LinearNetParams(
        input_size=10, num_features=20, num_outputs=5, num_hidden_layers=2, act_type='relu'
    )
    net = DeepFFNN(config)
    optimizer = AdamGnT(net.parameters(), lr=0.01)
    
    if mode == 'legacy':
        gnt_input = net.layers
    elif mode == 'module':
        # Should fallback to unwrapping layers internally
        gnt_input = net
    
    try:
        # FIXED: Using GnT_for_FC now
        gnt = GnT_for_FC(
            net=gnt_input,  
            hidden_activation='relu',
            opt=optimizer,
            replacement_rate=0.5,
            decay_rate=0.9,
            maturity_threshold=0,
            init='kaiming',
            device=device,
        )
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(5, 10) 
    
    for step in range(2):
        _, features = net.predict(dummy_input)
        
        loss = features[-1].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
            gnt.gen_and_test(features)
        except Exception as e:
            logger.error(f"[Step {step}] CRASH: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        weights_sum = sum([p.data.sum().item() for n, p in net.named_parameters() if 'weight' in n])
        
        opt_state_sum = 0.0
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state: opt_state_sum += state['exp_avg'].sum().item()

        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4), 'opt_sum': round(opt_state_sum, 4)})
        
    return stats

if __name__ == "__main__":
    print("====== CONVNET TEST (Map vs Legacy) ======")
    conv_leg = get_stats_from_run('legacy')
    conv_mod = get_stats_from_run('module')
    
    if conv_leg and conv_mod:
        match = True
        for i, (l, m) in enumerate(zip(conv_leg, conv_mod)):
            # Compare dicts
            if l != m:
                print(f"Step {i} MISMATCH\nL: {l}\nM: {m}")
                match = False
        if match:
            print("✅ CONV SUCCESS: Utils, Weights, and Optimizer States are identical.")
        else:
            print("❌ CONV FAIL: Logic mismatch.")
    else:
        print("❌ CONV FAIL: Crash encountered.")

    print("\n====== FC TEST (Legacy vs Fallback) ======")
    fc_leg = get_stats_from_run_fc('legacy')
    fc_mod = get_stats_from_run_fc('module') 
    
    if fc_leg and fc_mod and fc_leg == fc_mod:
        print("✅ FC SUCCESS: Fallback wrapper works correctly.")
    else:
        print("❌ FC FAIL: Mismatch or crash.")