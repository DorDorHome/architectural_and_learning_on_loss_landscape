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

from src.models.conv_net import ConvNet
from src.models.deep_ffnn import DeepFFNN
from configs.configurations import NetParams, LinearNetParams
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC

def get_stats_from_run(mode):
    logger.info(f"\n[TEST ENGINE] Starting Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'
    
    config = NetParams(num_classes=10, input_height=32, input_width=32, activation='relu')
    net = ConvNet(config)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    # --- MODE SELECTION ---
    if mode == 'legacy':
        gnt_input = net.layers
        logger.info("[Setup] Passing 'net.layers' (List) to GnT")
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
            device=device
        )
        mode_detected = "Map" if gnt.use_map else "Legacy"
        logger.info(f"[Init] Success. Detected {gnt.num_hidden_layers} hidden layers. Mode={mode_detected}")
        
        # Verify mode activated correctly
        if mode == 'module' and not gnt.use_map:
            logger.error("FAIL: Passed module but GnT did not detect map!")
            return None
        if mode == 'legacy' and gnt.use_map:
            logger.error("FAIL: Passed list but GnT tried to use map!")
            return None
            
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(10, 3, 32, 32) 
    
    for step in range(3):
        # Forward pass
        _, features_list = net.predict(dummy_input)
        
        # GnT Step
        try:
            gnt.gen_and_test(features_list)
        except Exception as e:
            logger.error(f"[Step {step}] CRASH: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Snapshot Stats
        # 1. Utility values
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        
        # 2. Checksum of Weights (Prove modification happened and is identical)
        weights_sum = 0.0
        for name, param in net.named_parameters():
             if 'weight' in name: weights_sum += param.data.sum().item()
                
        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4)})
        
    logger.info(f"[Result] Mode='{mode}' finished successfully.")
    return stats

def get_stats_from_run_fc(mode):
    # This test remains simplistic for now as DeepFFNN hasn't been modified with get_plasticity_map yet
    # It serves to ensure we didn't break legacy FC functionality with refactors.
    logger.info(f"\n[TEST ENGINE] Starting FC Test Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'

    config = LinearNetParams(
        input_size=10, num_features=20, num_outputs=5, num_hidden_layers=2, act_type='relu'
    )
    net = DeepFFNN(config)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    if mode == 'legacy':
        gnt_input = net.layers
    elif mode == 'module':
        # DeepFFNN doesn't have get_plasticity_map yet, so this checks the robust fallback
        # gnt.py logic: if has layers but no map, unwraps layers -> behaves like legacy
        gnt_input = net
    
    try:
        gnt = GnT_for_FC(
            net=gnt_input,  
            hidden_activation='relu',
            opt=optimizer,
            replacement_rate=0.5,
            decay_rate=0.9,
            maturity_threshold=0,
            init='kaiming',
            device=device
        )
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(5, 10) 
    
    for step in range(2):
        _, features = net.predict(dummy_input)
        try:
             # DeepFFNN.predict returns features as list, GnT expects that list
            gnt.gen_and_test(features)
        except Exception as e:
            logger.error(f"[Step {step}] CRASH: {e}")
            return None
        
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        weights_sum = sum([p.data.sum().item() for n, p in net.named_parameters() if 'weight' in n])
        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4)})
        
    return stats

if __name__ == "__main__":
    print("====== CONVNET TEST (Map vs Legacy) ======")
    conv_leg = get_stats_from_run('legacy')
    conv_mod = get_stats_from_run('module')
    
    if conv_leg and conv_mod:
        match = True
        for i, (l, m) in enumerate(zip(conv_leg, conv_mod)):
            if l != m:
                print(f"Step {i} MISMATCH\nL: {l}\nM: {m}")
                match = False
        if match:
            print("✅ CONV SUCCESS: Explicit Map produces identical physics to Legacy List.")
        else:
            print("❌ CONV FAIL: Logic mismatch.")
    else:
        print("❌ CONV FAIL: Crash encountered.")

    print("\n====== FC TEST (Legacy vs Fallback) ======")
    fc_leg = get_stats_from_run_fc('legacy')
    fc_mod = get_stats_from_run_fc('module') # Tests if passing object unwraps layers correctly
    
    if fc_leg and fc_mod and fc_leg == fc_mod:
        print("✅ FC SUCCESS: Fallback wrapper works correctly.")
    else:
        print("❌ FC FAIL: Mismatch or crash.")