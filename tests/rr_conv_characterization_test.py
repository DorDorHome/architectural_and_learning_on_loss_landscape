import sys
from pathlib import Path
import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
from src.models.conv_net import ConvNet
from configs.configurations import NetParams, RRCBP2Config
from src.algos.supervised.rr_gnt2_conv import RR_GnT2_for_ConvNet
from src.algos.AdamGnT import AdamGnT

def get_stats_from_run(mode):
    logger.info(f"\n[TEST ENGINE] Starting Mode='{mode}'")
    torch.manual_seed(42)
    device = 'cpu'
    
    # 1. Setup Network and Optimizer
    config = NetParams(num_classes=10, input_height=32, input_width=32, activation='relu')
    net = ConvNet(config)
    optimizer = AdamGnT(net.parameters(), lr=0.01)
    
    # 2. Setup RR-CBP2 Config
    # Check RRCBP2Config fields match your project definition. 
    # Providing assumed defaults based on rr_gnt2_conv.py usage.
    rr_config = RRCBP2Config(
        neurons_replacement_rate=0.5,
        decay_rate_utility_track=0.9,
        maturity_threshold=0,
        init='kaiming',
        util_type='contribution',
        rrcbp_enabled=True,
        # RR Specifics
        sigma_ema_beta=0.99,
        sigma_ridge=1e-4,
        diag_sigma_only=False,
        use_energy_budget=True,
        proj_eps=1e-6,
        max_proj_trials=5,
        covariance_dtype='float32',  # <--- FIXED: Must be string, not torch.float32
        log_rank_metrics_every=0 
    )

    # 3. Mode Selection
    if mode == 'legacy':
        gnt_input = net.layers
        logger.info("[Setup] Passing 'net.layers' (List)")
    elif mode == 'module':
        gnt_input = net
        logger.info("[Setup] Passing 'net' (Object)")
    
    # 4. Initialize Learner
    try:
        gnt = RR_GnT2_for_ConvNet(
            net=gnt_input,
            hidden_activation='relu',
            opt=optimizer,
            config=rr_config,
            loss_func=nn.CrossEntropyLoss(),
            device=device,
            # CRITICAL: 32x32 input -> 2x2 spatial output after 3 convs & pools = 4 outputs
            num_last_filter_outputs=4 
        )
        mode_detected = "Map" if gnt.use_map else "Legacy"
        logger.info(f"[Init] Success. Detected {gnt.num_hidden_layers} hidden layers. Mode={mode_detected}")
    except Exception as e:
        logger.error(f"[Init] CRASH: {e}")
        import traceback
        traceback.print_exc()
        return None

    stats = []
    dummy_input = torch.randn(10, 3, 32, 32) 
    
    # 5. Simulation Loop
    for step in range(3):
        # A. Forward
        logits, features_list = net.predict(dummy_input)
        
        # B. Backward (Populate Grads & Opt State)
        # We use logits to ensure gradients reach Output Layer weights
        loss = logits.sum() 
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # C. GnT Step
        try:
            # RR-Conv requires batch_input explicitly
            gnt.gen_and_test(features_list, batch_input=dummy_input)
        except Exception as e:
            logger.error(f"[Step {step}] GnT CRASH: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # D. Snapshot Stats
        
        # 1. Util Values
        utils = [[round(x, 6) for x in layer[:2].tolist()] for layer in gnt.util]
        
        # 2. Weights Checksum
        weights_sum = 0.0
        for name, param in net.named_parameters():
             if 'weight' in name: weights_sum += param.data.sum().item()
        
        # 3. Optimizer State Checksum
        opt_state_sum = 0.0
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    opt_state_sum += state['exp_avg'].sum().item()
                if 'exp_avg_sq' in state:
                    opt_state_sum += state['exp_avg_sq'].sum().item()
                    
        # 4. Covariance State Checksum (Unique to RR)
        # This confirms the RR logic/input computation is identical
        cov_sum = 0.0
        for cov in gnt.layer_covariances:
            if cov is not None:
                cov_sum += cov.ema.sum().item() 
                
        stats.append({
            'utils': utils, 
            'w_sum': round(weights_sum, 4), 
            'opt_sum': round(opt_state_sum, 4),
            'cov_sum': round(cov_sum, 4)
        })
        
    logger.info(f"[Result] Mode='{mode}' finished successfully.")
    return stats

if __name__ == "__main__":
    print("====== RR-CONV TEST (Map vs Legacy) ======")
    conv_leg = get_stats_from_run('legacy')
    conv_mod = get_stats_from_run('module')
    
    if conv_leg and conv_mod:
        match = True
        for i, (l, m) in enumerate(zip(conv_leg, conv_mod)):
            # Compare stats dicts
            if l != m:
                print(f"Step {i} MISMATCH")
                print(f"Legacy: {l}")
                print(f"Module: {m}")
                match = False
        if match:
            print("✅ RR-CONV SUCCESS: Utils, Weights, Opt States, and Covariance Matrices are identical.")
        else:
            print("❌ RR-CONV FAIL: Logic mismatch.")
    else:
        print("❌ RR-CONV FAIL: Crash encountered.")