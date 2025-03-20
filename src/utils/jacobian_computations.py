import torch
import torch.nn as nn
import torch.autograd as autograd
import time

# Define a simple network that collects intermediate features
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            features[f'fc{i+1}'] = x  # store each fc output
            x = self.relu(x)
            features[f'relu{i+1}'] = x  # store each relu output
        out = self.output_layer(x)
        # Return the final output and the features dictionary
        return out, features

device = torch.device('cuda:0')


# Instantiate the network
input_dim = 200000
hidden_dim = [10000, 5000, 100]
output_dim = 100
model = SimpleNet(input_dim, hidden_dim, output_dim).to(device)
model.train()

# Create a dummy input with requires_grad=True
x = torch.randn(1, input_dim, requires_grad=True).to(device)

# Forward pass
final_output, features = model(x)

# Let's define a function that gets a specific intermediate feature by name.
def feature_func(x, feature_name):
    # Run the forward pass through the network and extract a specific feature
    _, features = model(x)
    return features[feature_name]

### Approach A: Using torch.autograd.grad to compute Jacobian for a feature
def compute_jacobian_grad(feature_name):
    # Compute the feature with a new input that requires grad
    x = torch.randn(1, input_dim, requires_grad=True).to(device)
    feat = feature_func(x, feature_name)
    
    # To compute the full Jacobian, we need to loop over the output dimension of the feature
    jacobian = []
    for i in range(feat.shape[1]):  # assuming feature shape is [1, feature_dim]
        # Create grad_output vector: zeros everywhere except 1 at position i
        grad_out = torch.zeros_like(feat)
        grad_out[0, i] = 1.0
        # Get gradient of that component wrt x
        grad_x = autograd.grad(feat, x, grad_outputs=grad_out, retain_graph=True, create_graph=False)[0]
        jacobian.append(grad_x.detach().clone())
    jacobian = torch.stack(jacobian, dim=0)  # shape: [feature_dim, x.shape...]
    return jacobian

### Approach B: Using torch.autograd.functional.jacobian directly
def compute_jacobian_functional(feature_name):
    # Define a lambda that only returns the feature of interest
    func = lambda inp: feature_func(inp, feature_name)
    jac = autograd.functional.jacobian(func, x)
    return jac

# Timing helper
def time_function(func, n_iters=10):
    start = time.time()
    for _ in range(n_iters):
        _ = func()
    end = time.time()
    return (end - start) / n_iters

if __name__ == '__main__':
    # Compare for the feature 'relu'
    
    # compare the results for both approaches:
    jac_grad = compute_jacobian_grad('relu3')
    jac_func = compute_jacobian_functional('relu3')
    print("Jacobian computed using torch.autograd.grad:")
    print(jac_grad.shape)
    print("Jacobian computed using torch.autograd.functional.jacobian:")
    #reshape
    jac_func = jac_func.reshape(jac_grad.shape)
    print(jac_func.shape)
    print("Difference between the two approaches:")
    difference = torch.norm(jac_grad - jac_func)
    print(difference)
    
    
    
    
    
    # compare the computation time for both approaches
    
    n_iters = 20
    print("Comparing computational efficiency of the two jacobian computation for the 'relu' feature") 

    time_grad = time_function(lambda: compute_jacobian_grad('relu3'), n_iters)
    time_functional = time_function(lambda: compute_jacobian_functional('relu3'), n_iters)

    print("Average time using torch.autograd.grad: {:.6f} sec".format(time_grad))
    print("Average time using torch.autograd.functional.jacobian: {:.6f} sec".format(time_functional))

    # Note: For a single sample with relatively low dimensions, timing differences may be small.
    # In higher dimension cases or batched inputs, the differences may become more pronounced.