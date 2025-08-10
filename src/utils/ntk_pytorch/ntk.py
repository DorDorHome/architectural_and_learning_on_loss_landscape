import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev

def get_ntk(model, x):
    """
    Computes the Neural Tangent Kernel for a PyTorch model using torch.func.

    Args:
        model: A PyTorch model (nn.Module).
        x: A batch of input data (PyTorch tensor).

    Returns:
        The NTK matrix as a PyTorch tensor.
    """
    model.eval()
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}

    def f(params, x):
        return functional_call(model, params, (x,))

    jac_fn = vmap(jacrev(f, argnums=0), in_dims=(None, 0))

    jacobian = jac_fn(params, x)
    jacobian = [j.flatten(start_dim=2) for j in jacobian.values()]
    jacobian = torch.cat(jacobian, dim=2) # (N, C, P)

    ntk = torch.einsum('icp,jcp->ij', jacobian, jacobian)

    return ntk

def get_ntk_eigenvalues(ntk_matrix):
    """
    Computes the eigenvalues of an NTK matrix.
    """
    return torch.linalg.eigh(ntk_matrix).eigenvalues
