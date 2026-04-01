import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from configs.configurations import NetParams, SRRCBPConfig
from src.models.conv_net import ConvNet
from src.models.layer_norm_conv_net import ConvNetWithFCLayerNorm, ConvNet_conv_and_FC_LayerNorm
from src.algos.supervised.supervised_factory import create_learner

def get_dummy_conv_net_config():
    return OmegaConf.create({
        'num_classes': 10,
        'input_height': 32,
        'input_width': 32,
        'activation': 'relu',
    })

def get_dummy_ln_conv_net_config():
    return OmegaConf.create({
        'num_classes': 10,
        'input_height': 32,
        'input_width': 32,
        'activation': 'relu',
        'norm_param': {'layer_norm': {'elementwise_affine': True}}
    })

def get_srr_cbp_config():
    return SRRCBPConfig(
        type='srr_cbp',
        device='cpu',
        step_size=0.01,
        lambda_0=0.1,
        gamma=0.99,
        maturity_threshold=10,
        neurons_replacement_rate=0.1,
        util_type='contribution'
    )

@pytest.fixture
def dummy_input():
    return torch.randn(4, 3, 32, 32)

@pytest.fixture
def dummy_target():
    return torch.randint(0, 10, (4,))

@pytest.mark.parametrize("model_class, config_func", [
    (ConvNet, get_dummy_conv_net_config),
    (ConvNetWithFCLayerNorm, get_dummy_ln_conv_net_config),
    (ConvNet_conv_and_FC_LayerNorm, get_dummy_ln_conv_net_config)
])
def test_srr_cbp_initialization_and_forward(model_class, config_func, dummy_input, dummy_target):
    net_params = config_func()
    model = model_class(net_params)
    
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    learner_config = get_srr_cbp_config()
    learner = create_learner(learner_config, model, net_config)
    
    # Check that pre_activations hooks are registered
    assert hasattr(learner, 'pre_activations')
    assert isinstance(learner.pre_activations, dict)
    
    # Run a learn step
    loss, output = learner.learn(dummy_input, dummy_target)
    
    assert loss is not None
    assert output.shape == (4, 10)
    
    # Check that pre_activations are populated
    assert len(learner.pre_activations) > 0
    for i, pre_act in learner.pre_activations.items():
        assert pre_act is not None
        assert pre_act.shape[0] == 4  # batch size

@pytest.mark.parametrize("model_class, config_func", [
    (ConvNet, get_dummy_conv_net_config),
    (ConvNetWithFCLayerNorm, get_dummy_ln_conv_net_config)
])
def test_srr_cbp_weight_update(model_class, config_func, dummy_input, dummy_target):
    net_params = config_func()
    model = model_class(net_params)
    
    # Save original weights
    original_weights = {name: param.clone() for name, param in model.named_parameters()}

    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    learner_config = get_srr_cbp_config()
    learner = create_learner(learner_config, model, net_config)

    # Run a learn step
    loss, _ = learner.learn(dummy_input, dummy_target)

    # Check that weights have been updated
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param, original_weights[name]), f"Weights for {name} did not change"

def test_srr_cbp_aso_loss_computation():
    """
    Test that ASO loss is actually computed and affects gradients.
    We do this by running two identical models, one with lambda_0 = 0 (no ASO loss)
    and one with lambda_0 > 0. The gradients should be different.
    """
    torch.manual_seed(42)
    net_params = get_dummy_conv_net_config()
    model1 = ConvNet(net_params)
    
    torch.manual_seed(42)
    model2 = ConvNet(net_params)
    
    # Ensure models are identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data.copy_(p1.data)
        
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))
    
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    
    config1 = get_srr_cbp_config()
    config1.lambda_0 = 0.0  # No ASO loss
    learner1 = create_learner(config1, model1, net_config)
    
    config2 = get_srr_cbp_config()
    config2.lambda_0 = 10.0  # High ASO loss to make difference obvious
    learner2 = create_learner(config2, model2, net_config)
    
    # We need to artificially age some units to trigger ASO loss
    # ASO loss only computes if there are both young and mature units
    for i in range(len(learner2.gnt.ages)):
        num_units = learner2.gnt.ages[i].shape[0]
        # Make half of them mature, half young
        learner2.gnt.ages[i][:num_units//2] = config2.maturity_threshold + 1
        learner2.gnt.ages[i][num_units//2:] = 0
        
        learner1.gnt.ages[i][:num_units//2] = config1.maturity_threshold + 1
        learner1.gnt.ages[i][num_units//2:] = 0
        
    loss1, _ = learner1.learn(dummy_input, dummy_target)
    loss2, _ = learner2.learn(dummy_input, dummy_target)

    # Weights of at least one parameter should be different
    weights_different = False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2):
            weights_different = True
            break

    assert weights_different, "Weights are identical, ASO loss might not be affecting the backward pass"
    
    # Loss2 should be higher due to the penalty
    # Wait, the returned loss might just be the task loss, not total loss.
    # We should check if the implementation returns task_loss or total_loss.
    # The plan says "Return task_loss.detach() and output.detach()".
    # So loss1 and loss2 (returned) should be identical!
    assert torch.allclose(loss1, loss2), "Returned loss should be task_loss only"
