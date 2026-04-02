import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from configs.configurations import NetParams, SRRCBPConfig, LinearNetParams
from src.models.conv_net import ConvNet
from src.models.layer_norm_conv_net import ConvNetWithFCLayerNorm, ConvNet_conv_and_FC_LayerNorm
from src.models.deep_ffnn import DeepFFNN
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

def get_dummy_fc_net_config():
    return LinearNetParams(
        input_size=32 * 32 * 3,
        num_features=64,
        num_outputs=10,
        num_hidden_layers=2,
        act_type='relu',
        initialization='kaiming'
    )

def get_srr_cbp_config():
    return SRRCBPConfig(
        type='srr_cbp',
        device='cpu',
        step_size=0.01,
        SO_reg_lambda=0.1,
        aso_age_decay_rate=0.99,
        maturity_threshold=10,
        neurons_replacement_rate=0.1,
        util_type='contribution'
    )

@pytest.fixture
def dummy_input():
    return torch.randn(4, 3, 32, 32)

@pytest.fixture
def dummy_input_fc():
    return torch.randn(4, 32 * 32 * 3)

@pytest.fixture
def dummy_target():
    return torch.randint(0, 10, (4,))

@pytest.mark.parametrize("model_class, config_func, is_fc", [
    (ConvNet, get_dummy_conv_net_config, False),
    (ConvNetWithFCLayerNorm, get_dummy_ln_conv_net_config, False),
    (ConvNet_conv_and_FC_LayerNorm, get_dummy_ln_conv_net_config, False),
    (DeepFFNN, get_dummy_fc_net_config, True)
])
def test_srr_cbp_initialization_and_forward(model_class, config_func, is_fc, dummy_input, dummy_input_fc, dummy_target):
    net_params = config_func()
    model = model_class(net_params)
    
    net_type = 'FC' if is_fc else 'ConvNet'
    if is_fc:
        model.type = 'FC'
    net_config = OmegaConf.create({'type': net_type, 'netparams': net_params})
    learner_config = get_srr_cbp_config()
    learner = create_learner(learner_config, model, net_config)
    
    # Check that pre_activations hooks are registered
    assert hasattr(learner, 'pre_activations')
    assert isinstance(learner.pre_activations, dict)
    
    # Run a learn step
    inp = dummy_input_fc if is_fc else dummy_input
    loss, output = learner.learn(inp, dummy_target)
    
    assert loss is not None
    assert output.shape == (4, 10)
    
    # Check that pre_activations are populated
    assert len(learner.pre_activations) > 0
    for i, pre_act in learner.pre_activations.items():
        assert pre_act is not None
        assert pre_act.shape[0] == 4  # batch size

@pytest.mark.parametrize("model_class, config_func, is_fc", [
    (ConvNet, get_dummy_conv_net_config, False),
    (ConvNetWithFCLayerNorm, get_dummy_ln_conv_net_config, False),
    (DeepFFNN, get_dummy_fc_net_config, True)
])
def test_srr_cbp_weight_update(model_class, config_func, is_fc, dummy_input, dummy_input_fc, dummy_target):
    net_params = config_func()
    model = model_class(net_params)
    
    # Save original weights
    original_weights = {name: param.clone() for name, param in model.named_parameters()}

    net_type = 'FC' if is_fc else 'ConvNet'
    if is_fc:
        model.type = 'FC'
    net_config = OmegaConf.create({'type': net_type, 'netparams': net_params})
    learner_config = get_srr_cbp_config()
    learner = create_learner(learner_config, model, net_config)

    # Run a learn step
    inp = dummy_input_fc if is_fc else dummy_input
    loss, _ = learner.learn(inp, dummy_target)

    # Check that weights have been updated
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param, original_weights[name]), f"Weights for {name} did not change"

def test_srr_cbp_aso_loss_computation():
    """
    Test that ASO loss is actually computed and affects gradients.
    We do this by running two identical models, one with SO_reg_lambda = 0 (no ASO loss)
    and one with SO_reg_lambda > 0. The gradients should be different.
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
    config1.SO_reg_lambda = 0.0  # No ASO loss
    learner1 = create_learner(config1, model1, net_config)
    
    config2 = get_srr_cbp_config()
    config2.SO_reg_lambda = 10.0  # High ASO loss to make difference obvious
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
    
    assert torch.allclose(loss1, loss2), "Returned loss should be task_loss only"

def test_srr_cbp_mature_units_stop_gradient():
    """
    Verify that ASO loss does NOT send gradients back through mature units.
    """
    torch.manual_seed(42)
    net_params = get_dummy_conv_net_config()
    model = ConvNet(net_params)
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    
    config = get_srr_cbp_config()
    config.SO_reg_lambda = 10.0 # High penalty
    learner = create_learner(config, model, net_config)
    
    # Force ages: first half mature, second half young
    for i in range(len(learner.gnt.ages)):
        num_units = learner.gnt.ages[i].shape[0]
        learner.gnt.ages[i][:num_units//2] = config.maturity_threshold + 1
        learner.gnt.ages[i][num_units//2:] = 0
        
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))
    
    # We will temporarily override the learner's loss_func to return 0
    # This ensures ONLY the ASO loss drives gradients
    learner.loss_func = lambda x, y: torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # Run the learn step
    learner.learn(dummy_input, dummy_target)
    
    # Verify gradients
    layer_idx = 0
    for item in model.get_plasticity_map():
        if layer_idx >= len(learner.gnt.ages):
            break
        weight_module = item['weight_module']
        if weight_module.weight.grad is not None:
            num_units = learner.gnt.ages[layer_idx].shape[0]
            mature_grads = weight_module.weight.grad[:num_units//2]
            young_grads = weight_module.weight.grad[num_units//2:]
            
            # Gradients for mature units should be exactly zero
            assert torch.all(mature_grads == 0.0), f"Mature units in layer {layer_idx} received gradients from ASO loss!"
            
            # Gradients for young units should be non-zero
            assert not torch.all(young_grads == 0.0), f"Young units in layer {layer_idx} did not receive gradients from ASO loss!"
            
        layer_idx += 1

def test_srr_cbp_edge_cases_ages():
    net_params = get_dummy_conv_net_config()
    model = ConvNet(net_params)
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    config = get_srr_cbp_config()
    learner = create_learner(config, model, net_config)
    
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))

    # Case 1: All units are young (age 0)
    for i in range(len(learner.gnt.ages)):
        learner.gnt.ages[i].fill_(0)
    
    # Run learn, ensure no crash
    learner.learn(dummy_input, dummy_target)
    
    # Case 2: All units are mature
    for i in range(len(learner.gnt.ages)):
        learner.gnt.ages[i].fill_(config.maturity_threshold + 5)
        
    # Run learn, ensure no crash
    learner.learn(dummy_input, dummy_target)

def test_srr_cbp_aso_normalization_mode():
    """
    Verify that the aso_normalization_mode correctly scales the ASO penalty.
    """
    torch.manual_seed(42)
    net_params = get_dummy_conv_net_config()
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))
    
    def get_gradients_for_mode(mode):
        torch.manual_seed(42)
        model = ConvNet(net_params)
        config = get_srr_cbp_config()
        config.SO_reg_lambda = 10.0
        config.aso_normalization_mode = mode
        learner = create_learner(config, model, net_config)
        
        # We will temporarily override the learner's loss_func to return 0
        # This ensures ONLY the ASO loss drives gradients
        learner.loss_func = lambda x, y: torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Set up ages: half young, half mature
        for i in range(len(learner.gnt.ages)):
            num_units = learner.gnt.ages[i].shape[0]
            learner.gnt.ages[i][:num_units//2] = config.maturity_threshold + 1
            learner.gnt.ages[i][num_units//2:] = 0
            
        learner.learn(dummy_input, dummy_target)
        
        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.clone()
        return grads, learner

    grads_no_corr, learner_no_corr = get_gradients_for_mode("no correction")
    grads_input_size, _ = get_gradients_for_mode("correct by input size")
    grads_mse_sum, _ = get_gradients_for_mode("naive mse sum correction")
    
    layer_idx = 0
    model_dummy = ConvNet(net_params)
    for item in model_dummy.get_plasticity_map():
        if layer_idx >= len(learner_no_corr.gnt.ages):
            break
            
        weight_name = None
        for name, module in model_dummy.named_modules():
            if module is item['weight_module']:
                weight_name = name + '.weight'
                break
                
        if weight_name and weight_name in grads_no_corr:
            num_units = learner_no_corr.gnt.ages[layer_idx].shape[0]
            n_m = num_units // 2
            n_y = num_units - n_m
            
            grad_no_corr = grads_no_corr[weight_name]
            grad_input_size = grads_input_size[weight_name]
            grad_mse_sum = grads_mse_sum[weight_name]
            
            # Check scaling
            assert torch.allclose(grad_no_corr / n_y, grad_input_size, atol=1e-5), f"Failed 'correct by input size' scaling for {weight_name}"
            assert torch.allclose(grad_no_corr / (n_y * n_m), grad_mse_sum, atol=1e-5), f"Failed 'naive mse sum correction' scaling for {weight_name}"
            
        layer_idx += 1

def test_srr_cbp_gamma_decay():
    """
    Ensure that a young unit with age=9 receives a smaller gradient penalty 
    than a young unit with age=0.
    """
    torch.manual_seed(42)
    net_params = get_dummy_conv_net_config()
    model = ConvNet(net_params)
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': net_params})
    
    config = get_srr_cbp_config()
    config.SO_reg_lambda = 10.0
    config.gamma = 0.5 # use a small gamma to make differences obvious
    learner = create_learner(config, model, net_config)
    
    # Set up ages
    for i in range(len(learner.gnt.ages)):
        num_units = learner.gnt.ages[i].shape[0]
        # Unit 0: age 0 (young)
        # Unit 1: age 9 (young)
        # Rest: mature
        learner.gnt.ages[i].fill_(config.maturity_threshold + 1)
        learner.gnt.ages[i][0] = 0
        learner.gnt.ages[i][1] = config.maturity_threshold - 1
        
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))
    
    learner.loss_func = lambda x, y: torch.tensor(0.0, device=x.device, requires_grad=True)
    learner.learn(dummy_input, dummy_target)
    
    layer_idx = 0
    for item in model.get_plasticity_map():
        if layer_idx >= len(learner.gnt.ages):
            break
        weight_module = item['weight_module']
        if weight_module.weight.grad is not None:
            grad_0 = weight_module.weight.grad[0].norm()
            grad_1 = weight_module.weight.grad[1].norm()
            
            # The gradient penalty for age 0 should be larger than for age 9
            assert grad_0 > grad_1, f"Gamma decay failed in layer {layer_idx}: grad_0 ({grad_0}) <= grad_1 ({grad_1})"
            
        layer_idx += 1
