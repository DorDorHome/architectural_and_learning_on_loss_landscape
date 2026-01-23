import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import specific project classes
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
        # Simulate old code: passing the list directly
        gnt_input = net.layers
        logger.info("[Setup] Passing 'net.layers' (List) to GnT")
    elif mode == 'module':
        # Simulate new code: passing the object
        gnt_input = net
        logger.info("[Setup] Passing 'net' (ConvNet Object) to GnT")
    
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
        logger.info(f"[Init] Success. Detected {gnt.num_hidden_layers} hidden layers.")
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(10, 3, 32, 32) 
    
    for step in range(3):
        _, features_list = net.predict(dummy_input)
        try:
            gnt.gen_and_test(features_list)
        except Exception as e:
            logger.error(f"[Step {step}] CRASH: {e}")
            return None
        
        # Snapshot Stats
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        weights_sum = 0.0
        # Calculate weight sum safely (Module vs List agnostic for checking)
        for name, param in net.named_parameters():
             if 'weight' in name: weights_sum += param.data.sum().item()
                
        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4)})
        
    logger.info(f"[Result] Mode='{mode}' finished successfully.")
    return stats
            
            

def get_stats_from_run_fc(mode):
    logger.info(f"\n[TEST ENGINE] Starting FC Test Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'
    
    # Configure DeepFFNN
    # Layers: In(Linear), Act, Hidden(Linear), Act, Out(Linear)
    # 2 Hidden layers requested -> 2 iterations in GnT
    config = LinearNetParams(
        input_size=10, 
        num_features=20, 
        num_outputs=5, 
        num_hidden_layers=2, # Creates In + 1 Hidden Block
        act_type='relu'
    )
    net = DeepFFNN(config)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    if mode == 'legacy':
        gnt_input = net.layers
    elif mode == 'module':
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
            device=device,
            util_type='contribution'
        )
        logger.info(f"[Init] Success. Detected {gnt.num_hidden_layers} hidden layers.")
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        return None

    stats = []
    dummy_input = torch.randn(5, 10) 
    
    for step in range(3):
        # DeepFFNN returns (out, [features...])
        _, activations = net.predict(dummy_input)
        
        # GnT for FC expects features [h1, h2, ...] 
        # DeepFFNN activations includes output at end? 
        # Let's check: in_layer(x), hidden_layers(x)
        # We need to pass the list of features.
        try:
            gnt.gen_and_test(activations)
        except Exception as e:
            logger.error(f"[Step {step}] CRASH: {e}")
            return None
        
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        weights_sum = 0.0
        for name, param in net.named_parameters():
             if 'weight' in name: weights_sum += param.data.sum().item()
        stats.append({'utils': utils, 'w_sum': round(weights_sum, 4)})
        
    return stats

if __name__ == "__main__":
    print("====== CONVNET TEST ======")
    conv_leg = get_stats_from_run('legacy')
    conv_mod = get_stats_from_run('module')
    
    if conv_leg and conv_mod and conv_leg == conv_mod:
        print("✅ CONV SUCCESS: Both modes match.")
    else:
        print("❌ CONV FAIL: Legacy/Module mismatch or crash.")

    print("\n====== FC TEST ======")
    fc_leg = get_stats_from_run_fc('legacy')
    fc_mod = get_stats_from_run_fc('module')
    
    if fc_leg and fc_mod and fc_leg == fc_mod:
        print("✅ FC SUCCESS: Both modes match.")
    else:
        print("❌ FC FAIL: Legacy/Module mismatch or crash.")