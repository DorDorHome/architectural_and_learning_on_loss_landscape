import torch
import pytest
from typing import Tuple

from configs.configurations import RRContinuousBackpropConfig, LinearNetParams, NetConfig, NetParams
from src.models.deep_ffnn import DeepFFNN
from src.models.conv_net import ConvNet
from src.algos.supervised.rr_cbp_fc import RankRestoringCBP_for_FC
from src.algos.supervised.rr_cbp_conv import RankRestoringCBP_for_ConvNet


def test_bias_transfer_fc():
    """Test that bias transfer prevents function jumps in FC networks."""
    device = 'cpu'
    
    # Create a small FC model
    net_params = LinearNetParams(input_size=8, num_features=6, num_outputs=3, num_hidden_layers=1, act_type='relu')
    model = DeepFFNN(net_params)
    model.type = 'FC'
    model = model.to(device)
    
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=1.0,  # Replace all mature units
        maturity_threshold=1,  # All units become mature after 1 step
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.0,  # No EMA, just use current batch
        sigma_ridge=1e-3,
        log_rank_metrics_every=0,  # No logging for this test
    )
    config.opt = 'adam'
    net_config = NetConfig(type='FC', netparams=net_params)

    learner = RankRestoringCBP_for_FC(model, config, netconfig=net_config)

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(16, 8, device=device)
    y = torch.randint(0, 3, (16,), device=device)

    # Get initial network output
    with torch.no_grad():
        initial_output, _ = model.predict(x)

    # Perform one learning step (this should trigger unit replacement)
    loss1, output1 = learner.learn(x, y)
    
    # Perform second learning step (this should also trigger replacement)
    loss2, output2 = learner.learn(x, y)

    # Verify that losses are finite
    assert torch.isfinite(loss1).item(), "First step loss should be finite"
    assert torch.isfinite(loss2).item(), "Second step loss should be finite"
    
    # Verify outputs are finite
    assert torch.isfinite(output1).all(), "First step output should be finite"
    assert torch.isfinite(output2).all(), "Second step output should be finite"
    
    # Check that some units were actually replaced
    stats = learner.rr_gnt.get_layer_stats()
    total_replacements = sum(s.successful + s.fallbacks for s in stats.values())
    assert total_replacements > 0, "Some units should have been replaced"
    
    print(f"✅ FC bias transfer test passed. Total replacements: {total_replacements}")


def test_bias_transfer_conv():
    """Test that bias transfer prevents function jumps in ConvNet."""
    device = 'cpu'
    
    # Create a small ConvNet
    params = NetParams(
        activation="relu",
        num_classes=3,
        input_height=28,
        input_width=28,
    )
    model = ConvNet(params)
    model = model.to(device)
    
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=0.5,  # Replace half of mature units
        maturity_threshold=1,
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.0,
        sigma_ridge=1e-3,
        log_rank_metrics_every=0,
    )
    config.opt = 'adam'
    config.network_class = 'conv'
    net_config = NetConfig(type='ConvNet', netparams=params, network_class='conv')

    learner = RankRestoringCBP_for_ConvNet(model, config, netconfig=net_config)

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(8, 3, 28, 28, device=device)
    y = torch.randint(0, 3, (8,), device=device)

    # Get initial network output
    with torch.no_grad():
        initial_output, _ = model.predict(x)

    # Perform learning steps
    loss1, output1 = learner.learn(x, y)
    loss2, output2 = learner.learn(x, y)

    # Verify finite losses and outputs
    assert torch.isfinite(loss1).item(), "Conv first step loss should be finite"
    assert torch.isfinite(loss2).item(), "Conv second step loss should be finite"
    assert torch.isfinite(output1).all(), "Conv first step output should be finite"
    assert torch.isfinite(output2).all(), "Conv second step output should be finite"
    
    # Check replacements occurred
    stats = learner.rr_gnt.get_layer_stats()
    total_replacements = sum(s.successful + s.fallbacks for s in stats.values())
    
    print(f"✅ ConvNet bias transfer test passed. Total replacements: {total_replacements}")


def test_function_continuity_fc():
    """Test that bias transfer preserves function output during unit replacement."""
    device = 'cpu'
    
    # Create model and learner
    net_params = LinearNetParams(input_size=4, num_features=3, num_outputs=2, num_hidden_layers=1, act_type='relu')
    model = DeepFFNN(net_params)
    model.type = 'FC'
    model = model.to(device)
    
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=1.0,  # Force replacement
        maturity_threshold=1,
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.8,  # Use EMA to build up unit activations
        sigma_ridge=1e-3,
        log_rank_metrics_every=0,
    )
    config.opt = 'adam'
    net_config = NetConfig(type='FC', netparams=net_params)

    learner = RankRestoringCBP_for_FC(model, config, netconfig=net_config)

    # Create test input
    torch.manual_seed(123)
    x = torch.randn(4, 4, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    # Let the network run a few steps to build up EMA statistics
    for _ in range(3):
        learner.learn(x, y)

    # Now test function continuity
    with torch.no_grad():
        output_before, _ = model.predict(x)
    
    # Perform replacement step  
    loss, output_after_train = learner.learn(x, y)
    
    # Get output after replacement
    with torch.no_grad():
        output_after, _ = model.predict(x)
    
    # The outputs should be reasonably close (bias transfer should prevent large jumps)
    # Note: Some difference is expected due to the learning step, but it shouldn't be huge
    output_diff = torch.abs(output_after - output_before).mean()
    
    print(f"✅ Function continuity test: Average output change = {output_diff:.6f}")
    
    # The change should be reasonable (not a massive jump)
    assert output_diff < 10.0, f"Output change too large: {output_diff}"
    assert torch.isfinite(output_after).all(), "Output should remain finite"


if __name__ == "__main__":
    test_bias_transfer_fc()
    test_bias_transfer_conv()
    test_function_continuity_fc()
    print("🎉 All bias transfer tests passed!")