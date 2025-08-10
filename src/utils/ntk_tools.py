import torch
import jax
import jax.numpy as jnp
import numpy as np
import neural_tangents as nt
from jax.experimental.host_callback import call

def _torch_to_jax_params_and_shapes(torch_params):
    """Converts a PyTorch parameter dictionary to JAX parameters and shapes."""
    jax_params = []
    shapes = []
    names = []
    for name, param in torch_params.items():
        jax_params.append(jnp.array(param.cpu().detach().numpy()))
        shapes.append(param.shape)
        names.append(name)
    return jax_params, shapes, names

def _torch_model_to_jax_apply_fn(model, shapes, names, sample_input):
    """
    Creates a JAX-compatible apply function for a PyTorch model.
    """
    # Perform a dummy forward pass to get the output shape
    with torch.no_grad():
        sample_output = model(sample_input)

    output_shape = sample_output.shape

    def apply_fn(params, x):
        # This function will be called by JAX, so its inputs are JAX arrays.

        def torch_forward_pass(args):
            # This function runs on the host (CPU) and can execute arbitrary Python code.
            params_flat, x_numpy = args

            # 1. Reconstruct the state_dict from the flattened parameters.
            state_dict = {}
            current_pos = 0
            for i in range(len(shapes)):
                shape = shapes[i]
                name = names[i]
                num_elements = np.prod(shape)
                param_numpy = params_flat[current_pos : current_pos + num_elements].reshape(shape)
                state_dict[name] = torch.from_numpy(param_numpy)
                current_pos += num_elements

            # 2. Load the state_dict into the model.
            model.load_state_dict(state_dict)

            # 3. Perform the forward pass.
            x_torch = torch.from_numpy(x_numpy)
            with torch.no_grad():
                output = model(x_torch)

            return output.cpu().numpy()

        # We need to flatten the parameters to pass them to the host callback.
        params_flat = jnp.concatenate([p.flatten() for p in params])

        # The output shape and dtype must be specified for the host callback.
        result = call(
            torch_forward_pass,
            (params_flat, x),
            result_shape=jax.ShapeDtypeStruct(shape=output_shape, dtype=x.dtype)
        )
        return result

    return apply_fn


def get_ntk_fn(model, params_torch, sample_input):
    """
    Returns a function that computes the NTK for a PyTorch model.
    """
    jax_params, shapes, names = _torch_to_jax_params_and_shapes(params_torch)
    apply_fn = _torch_model_to_jax_apply_fn(model, shapes, names, sample_input)

    kernel_fn = nt.empirical_kernel_fn(apply_fn)

    def ntk_fn(x_batch):
        x_batch_jax = jnp.array(x_batch.cpu().detach().numpy())
        ntk_matrix_jax = kernel_fn(x_batch_jax, None, 'ntk', jax_params)
        return torch.from_numpy(np.asarray(ntk_matrix_jax))

    return ntk_fn


def get_ntk_eigenvalues(ntk_matrix):
    """
    Computes the eigenvalues of an NTK matrix.
    """
    return torch.linalg.eigh(ntk_matrix).eigenvalues
