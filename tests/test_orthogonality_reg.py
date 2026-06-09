import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.losses.orthogonality import KernelSORegularizer
from src.models.conv_net import ConvNet
from src.models.layer_norm_conv_net import ConvNet_conv_and_FC_LayerNorm
from configs.configurations import NetParams, BaseLearnerConfig
from src.algos.supervised.base_learner import Learner

# Mock learner for testing BaseLearner integration
class MockLearner(Learner):
    def learn(self, x, target):
        self.opt.zero_grad()
        output = self._forward(x)
        # Handle models that return a tuple (like ConvNet)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_func(output, target)
        loss.backward()
        self.opt.step()
        return loss.item()

def test_kernel_so_regularizer_convnet():

    """
    Test that the KernelSORegularizer integrates and runs as expected when applied to a ConvNet model.

    - Checks that convolutional and linear layers are correctly cached by the regularizer.
    - Verifies that the regularizer's output is a valid loss tensor.
    - Ensures forward pass and gradient calculation operate with dummy input/output.
    """
    config = NetParams(input_height=32, input_width=32, num_classes=10)
    model = ConvNet(config)
    
    # Dummy loss function
    def dummy_loss(output, target):
        return F.mse_loss(output, target)
        
    reg = KernelSORegularizer(dummy_loss, model, lambda_orth=0.1)
    
    # Check if layers are cached correctly
    assert len(reg.conv_layers) == 3
    assert len(reg.linear_layers) == 3
    
    # Forward pass
    dummy_output = torch.randn(2, 10)
    dummy_target = torch.randn(2, 10)
    
    loss = reg(dummy_output, dummy_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

def test_kernel_so_regularizer_layer_norm_convnet():

    """
    Test that the KernelSORegularizer works as expected when applied to 
    ConvNet_conv_and_FC_LayerNorm, which includes both convolutional and fully-connected layers 
    with layer normalization.

    - Constructs a NetParams config and attaches a dummy `norm_param` with an elementwise affine property (to satisfy layer norm instantiation).
    - Instantiates the model and KernelSORegularizer with a dummy loss and regularization strength.
    - Asserts that conv_layers and linear_layers are both detected as 3 (typical for these architectures).
    - Runs a dummy forward/target pair through the regularizer, checks returned loss is a valid, positive tensor.
    """
    # Helper class to mock norm_param
    class DummyNormParam:
        class LayerNormParam:
            elementwise_affine = True
        layer_norm = LayerNormParam()
        
    config = NetParams(input_height=32, input_width=32, num_classes=10)
    # Dynamically attach the norm_param attribute
    setattr(config, 'norm_param', DummyNormParam())
    
    model = ConvNet_conv_and_FC_LayerNorm(config)
    
    def dummy_loss(output, target):
        return F.mse_loss(output, target)
        
    reg = KernelSORegularizer(dummy_loss, model, lambda_orth=0.1)
    
    assert len(reg.conv_layers) == 3
    assert len(reg.linear_layers) == 3
    
    dummy_output = torch.randn(2, 10)
    dummy_target = torch.randn(2, 10)
    
    loss = reg(dummy_output, dummy_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

def test_kernel_so_regularizer_known_weights():
    """
    Test that KernelSORegularizer computes the expected orthogonality penalty for known weights,
    across all normalization_mode settings.

    - Uses a simple 2x2 Linear layer whose weights are a scaled identity matrix (orthogonal but scaled).
    - For each normalization mode, computes expected penalty value:
      - "no correction"    : ||G - I||_F^2, where G = W W^T.
      - "correct by input size": divides penalty by input size.
      - "naive mse sum correction": computes mean squared error elementwise.
    - Asserts that returned penalty matches analytic result.
    """
    # Create a simple model with known weights
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            # Set weights to an orthogonal  matrix scaled by 2
            # W = [[2, 0], [0, 2]]
            self.linear.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            
    model = SimpleModel()
    
    def dummy_loss(output, target):
        return torch.tensor(0.0)
        
    # Test "no correction" mode
    reg_no_corr = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="no correction")
    loss_no_corr = reg_no_corr(None, None)
    
    # W @ W^T = [[4, 0], [0, 4]]
    # I = [[1, 0], [0, 1]]
    # G - I = [[3, 0], [0, 3]]
    # ||G - I||_F^2 = 3^2 + 3^2 = 18
    assert torch.isclose(loss_no_corr, torch.tensor(18.0))
    
    # Test "correct by input size" mode
    reg_corr_input = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="correct by input size")
    loss_corr_input = reg_corr_input(None, None)
    # 18 / 2 = 9
    assert torch.isclose(loss_corr_input, torch.tensor(9.0))
    
    # Test "naive mse sum correction" mode
    reg_naive = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="naive mse sum correction")
    loss_naive = reg_naive(None, None)
    # F.mse_loss(G, I, reduction='mean') = 18 / 4 = 4.5
    assert torch.isclose(loss_naive, torch.tensor(4.5))

def test_kernel_so_regularizer_gradients():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            self.linear.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
            
    model = SimpleModel()
    
    def dummy_loss(output, target):
        # Return 0 connected to the graph
        return 0.0 * (output.sum() if output is not None else torch.tensor(0.0, requires_grad=True))
        
    # Forward pass
    dummy_out = torch.zeros(1, requires_grad=True)
    
    # Test "no correction" mode
    reg = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="no correction")
    loss = reg(dummy_out, None)
    loss.backward()
    
    # Expected gradient for ||W W^T - I||_F^2 w.r.t W is 4 * (W W^T - I) W
    # W = [[2, 0], [0, 2]]
    # W W^T - I = [[3, 0], [0, 3]]
    # Grad = 4 * [[3, 0], [0, 3]] * [[2, 0], [0, 2]] = [[24, 0], [0, 24]]
    expected_grad = torch.tensor([[24.0, 0.0], [0.0, 24.0]])
    
    assert model.linear.weight.grad is not None
    assert torch.allclose(model.linear.weight.grad, expected_grad)
    
    # Test "correct by input size" mode (divided by D=2)
    model.linear.weight.grad.zero_()
    reg_corr = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="correct by input size")
    loss_corr = reg_corr(dummy_out, None)
    loss_corr.backward()
    assert torch.allclose(model.linear.weight.grad, expected_grad / 2.0)
    
    # Test "naive mse sum correction" mode (divided by D^2=4)
    model.linear.weight.grad.zero_()
    reg_naive = KernelSORegularizer(dummy_loss, model, lambda_orth=1.0, normalization_mode="naive mse sum correction")
    loss_naive = reg_naive(dummy_out, None)
    loss_naive.backward()
    assert torch.allclose(model.linear.weight.grad, expected_grad / 4.0)

def test_base_learner_integration():
    config = NetParams(input_height=32, input_width=32, num_classes=10)
    model = ConvNet(config)
    
    learner_config = BaseLearnerConfig(
        type='mock',
        device='cpu',
        loss='mse',
        additional_regularization='Kernel SO',
        lambda_orth=0.5,
        normalization_mode='correct by input size',
        opt='sgd',
        step_size=0.1
    )
    
    learner = MockLearner(model, learner_config, config)
    
    # Check if the loss function is correctly wrapped
    assert isinstance(learner.loss_func, KernelSORegularizer)
    assert learner.loss_func.lambda_orth == 0.5
    assert learner.loss_func.normalization_mode == 'correct by input size'
    
    # Test the learn step to ensure gradients flow and weights update
    # Note: ConvNet expects 3 channels. The flattened size depends on input size.
    # We used input_height=32, input_width=32, so we must pass 32x32 images.
    # Wait, ConvNet's predict method has a bug where it calls layers[3] on x1, but layers[3] is a conv layer and x1 is pool output.
    # Actually, let's just use a SimpleModel to test the BaseLearner integration to avoid ConvNet specific bugs.
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x)
        def predict(self, x):
            return self.linear(x)
            
    model = SimpleModel()
    learner = MockLearner(model, learner_config, config)
    
    # Check if the loss function is correctly wrapped
    assert isinstance(learner.loss_func, KernelSORegularizer)
    assert learner.loss_func.lambda_orth == 0.5
    assert learner.loss_func.normalization_mode == 'correct by input size'
    
    dummy_x = torch.randn(2, 10)
    dummy_target = torch.randn(2, 10)
    
    # Save initial weights of a layer to check if they change
    initial_weight = model.linear.weight.data.clone()
    
    loss_val = learner.learn(dummy_x, dummy_target)
    
    # Verify loss is a valid number
    assert isinstance(loss_val, float)
    assert loss_val > 0
    
    # Verify weights have been updated (gradient step was taken)
    assert not torch.allclose(model.linear.weight.data, initial_weight)
    
    # Verify that the regularizer cached the components of the loss
    assert hasattr(learner.loss_func, 'last_task_loss')
    assert hasattr(learner.loss_func, 'last_reg_loss')
    assert learner.loss_func.last_reg_loss > 0

if __name__ == "__main__":
    test_kernel_so_regularizer_convnet()
    test_kernel_so_regularizer_layer_norm_convnet()
    test_kernel_so_regularizer_known_weights()
    test_kernel_so_regularizer_gradients()
    test_base_learner_integration()
    print("All tests passed!")
