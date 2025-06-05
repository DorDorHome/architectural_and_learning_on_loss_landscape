# implementation of loss landscape smoothness measures
## includes:
### the loss itself as it changes with parameters 

import torch
import torch.nn as nn
import math


def estimate_effective_beta(
                            model: nn.Module,
                            batch: tuple[torch.Tensor, torch.Tensor],  # (inputs, targets)
                            criterion: nn.Module,
                            step_size_min: float = 1e-4,
                            step_size_max: float = 1e-2,
                            num_step_sizes: int = 3
                            ):
    inputs, targets = batch

    # Original forward/backward
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    orig_grad = [p.grad.detach().clone() for p in model.parameters()]
    orig_grad_norm = torch.norm(torch.cat([g.flatten() for g in orig_grad]), p=2)

    if orig_grad_norm == 0:  # Zero gradient edge case
        return 0.0

    # Preserve original parameters
    orig_params = [p.detach().clone() for p in model.parameters()]

    # Generate logarithmic step sizes
    if step_size_min == step_size_max:
        step_sizes = [step_size_min]
    else:
        assert num_step_sizes >= 2, "Require ≥2 steps when min ≠ max"
        # Convert to logarithmic exponents
        log_min = math.log10(step_size_min)
        log_max = math.log10(step_size_max)
        step_sizes = torch.logspace(log_min, log_max, steps=num_step_sizes).tolist()

    max_beta = 0
    for eta in step_sizes:
        # Perturb parameters
        with torch.no_grad():
            for p, g in zip(model.parameters(), orig_grad):
                p.add_(eta * g)  # θ_k + η∇L(θ_k)

        # Compute perturbed gradient
        model.zero_grad()
        outputs_p = model(inputs)
        loss_p = criterion(outputs_p, targets)
        loss_p.backward()
        perturb_grad = [p.grad.detach().clone() for p in model.parameters()]

        # Calculate gradient difference
        grad_diff = torch.cat([(pg - og).flatten()
                               for pg, og in zip(perturb_grad, orig_grad)])
        grad_diff_norm = torch.norm(grad_diff, p=2)

        # Compute effective beta
        step_size = eta * orig_grad_norm
        beta_eff = (grad_diff_norm / step_size).item() if step_size > 0 else 0
        max_beta = max(max_beta, beta_eff)

        # Restore original parameters
        with torch.no_grad():
            for p, orig in zip(model.parameters(), orig_params):
                p.copy_(orig)

    return max_beta



def print_comparison(name, estimated, theoretical):
    abs_error = abs(estimated - theoretical)
    rel_error = abs_error / theoretical * 100
    print(f"\n{name}:")
    print(f"  Estimated β: {estimated:.6f}")
    print(f"  Theoretical β: {theoretical:.6f}")
    print(f"  Absolute error: {abs_error:.6f}")
    print(f"  Relative error: {rel_error:.4f}%")
    print("  Status: ", "Acceptable (<1%)" if rel_error < 1 else "Check required")

if __name__ == "__main__":
    
    torch.manual_seed(42)
    
    # ===== Test 1: Perfect alignment case =====
    # Quadratic function f(w) = 0.5*(5w1² + 3w2²)
    # Hessian = [[5, 0], [0, 3]], max eigenvalue = 5
    model1 = nn.Linear(2, 1, bias=False)
    model1.weight.data.copy_(torch.tensor([[1.0, 0.0]]))  # Aligned with max eigenvector
    X1 = torch.diag(torch.sqrt(torch.tensor([5.0, 3.0])))
    y1 = torch.zeros(2)
    criterion1 = nn.MSELoss()
    
    # Theoretical β = max eigenvalue of Hessian (5.0)
    hessian1 = 2 * X1.T @ X1 / X1.shape[0]
    L_theoretical1 = torch.linalg.eigvalsh(hessian1).max().item()
    
    # Estimate β
    effective_beta1 = estimate_effective_beta(
        model1, (X1, y1), criterion1,
        step_size_min=1e-6, step_size_max=1e-3, num_step_sizes=20
    )
    print_comparison("Test 1: Perfect alignment", effective_beta1, L_theoretical1)

    # ===== Test 2: Non-aligned case =====
    # Quadratic function f(w) = 0.5*(w1² + 2w2²)
    # Hessian = [[1, 0], [0, 2]], max eigenvalue = 2
    model2 = nn.Linear(2, 1, bias=False)
    model2.weight.data.copy_(torch.tensor([[1.0, 1.0]]))  # Not aligned with max eigenvector
    X2 = torch.diag(torch.sqrt(torch.tensor([1.0, 2.0])))
    y2 = torch.zeros(2)
    criterion2 = nn.MSELoss()
    
    # Theoretical β = 2.0
    hessian2 = 2 * X2.T @ X2 / X2.shape[0]
    L_theoretical2 = torch.linalg.eigvalsh(hessian2).max().item()
    
    # Estimate β
    effective_beta2 = estimate_effective_beta(
        model2, (X2, y2), criterion2,
        step_size_min=1e-6, step_size_max=1e-3, num_step_sizes=20
    )
    print_comparison("\nTest 2: Non-aligned case", effective_beta2, L_theoretical2)


    import matplotlib.pyplot as plt
    # ===== Test 3: Time-series analysis =====
    def train_and_monitor(model, X, y, criterion):
        beta_history = []
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for epoch in range(100):
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            
            beta = estimate_effective_beta(
                model, (X, y), criterion,
                step_size_min=1e-6, step_size_max=1e-3, num_step_sizes=10
            )
            beta_history.append(beta)
            
            optimizer.step()
        
        plt.plot(beta_history)
        plt.title('Effective β-smoothness During Training')
        plt.xlabel('Epoch')
        plt.ylabel('β estimate')
        folder_to_save = 'testing_plots'
        # create the folder if it does not exist
        import os
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        plt.savefig(f'{folder_to_save}/effective_beta_smoothness_training.png')

    print("\n=== Training dynamics visual test ===")
    train_and_monitor(model1, X1, y1, criterion1)

    
    
    
    # torch.manual_seed(42)
    
    # # Simple linear regression: y = 2x + 1 + noise
    # X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    # y = 2*X.squeeze() + 1 + torch.randn(4)*0.1
    
    # # Simple neural network (linear regression)
    # model = nn.Sequential(nn.Linear(1, 1))
    # criterion = nn.MSELoss()
    
    # # Theoretical β-smoothness for MSE loss:
    # # For f(w) = ||Xw - y||^2/n, ∇²f(w) = 2XᵀX/n
    # X_matrix = torch.cat([X, torch.ones(4,1)], dim=1)
    # hessian = 2 * X_matrix.t() @ X_matrix / 4
    # L_theoretical = torch.linalg.eigvalsh(hessian).max().item()
    
    # print(f"Theoretical β-smoothness: {L_theoretical:.4f}")
    
    # # Training loop with verification
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # for epoch in range(5):
    #     inputs, targets = X, y
        
    #     # Forward + backward
    #     outputs = model(inputs).squeeze()
    #     loss = criterion(outputs, targets)
    #     optimizer.zero_grad()
    #     loss.backward()
        
    #     # Estimate effective β
    #     effective_beta = estimate_effective_beta(
    #         model, (inputs, targets), criterion,
    #         step_size_min=1e-4, step_size_max=1e-2, num_step_sizes=5
    #     )
        
    #     # Update parameters
    #     optimizer.step()
        
    #     # Verify β estimate
    #     print(f"Epoch {epoch+1}:")
    #     print(f"  Estimated β: {effective_beta:.4f}")
    #     print(f"  Theoretical β: {L_theoretical:.4f}")
    #     assert abs(effective_beta - L_theoretical) < 1e-5, \
    #         "Estimated β deviates from theoretical value"
    #     assert effective_beta <= L_theoretical * 1.1, \
    #         "Estimated β exceeds theoretical maximum"
    
    # print("All tests passed! β estimates match theoretical values")
    
    
    
    # # from torchvision.models import resnet18

    # # # Example usage
    # # model = resnet18(pretrained=False)
    # # criterion = nn.CrossEntropyLoss()
    
    # # # Dummy batch
    # # inputs = torch.randn(8, 3, 224, 224)  # Batch of 8 images
    # # targets = torch.randint(0, 1000, (8,))  # Random targets for 1000 classes
    # # batch = (inputs, targets)
    
    # # beta_eff = estimate_effective_beta(model, batch, criterion)
    # # print(f"Estimated effective beta: {beta_eff}")