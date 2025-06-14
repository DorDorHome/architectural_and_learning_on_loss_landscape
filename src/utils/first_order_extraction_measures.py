# implementation of jacobian computation of features with respect to the input:
# unfinished





import os
import os.path as osp
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd.functional import jacobian
# from configs import get_args # Assuming this is your arg parser

from typing import Dict, Any


# --- Helper for Input Patch Extraction & Preprocessing ---
from typing import Callable, Tuple

def extract_altered_input_patch_and_preprocessor(
    original_images: torch.Tensor, # Full original images (B, C, H, W)
    input_patch_config: Dict[str, Any] # Dict: e.g., {'method': 'zero_padding', 'patch_size': 16, ...}
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """"
    Given a batch of original images and a configuration dict,
    extracts a patch and returns the altered input tensor
    and a preprocessing function to get the altered input, which has the same size as the original input.
    The preprocessing function can be used to prepare the input for the model.
    Parameters:
        original_images: Batch of input images (B, C, H, W).
        input_patch_config: Configuration dict for patch extraction.
        method: Method for patch extraction ('interpolate', 'zero_padding', 'strided_subsample', or 'none').
        patch_size: Size of the square patch to extract (if applicable).
        row_idx, col_idx: Starting indices for patch extraction (if applicable).
    
    
    
    """

    
    method: str = input_patch_config.get('method', 'zero_padding')
    B, C, H_orig, W_orig = original_images.shape
    
    altered_input_patch_tensor: torch.Tensor = None  # type: ignore
    def _preprocess_fn_internal(x: torch.Tensor) -> torch.Tensor:
        return x  # Identity by default

    if method == 'interpolate':
        patch_size = input_patch_config.get('patch_size', 16)
        altered_input_patch_tensor = functional.interpolate(original_images, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
        def interpolate_preprocessor(patch_tensor_arg):
            return functional.interpolate(patch_tensor_arg, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        _preprocess_fn_internal = interpolate_preprocessor

    elif method == 'zero_padding':
        patch_size = input_patch_config.get('patch_size', 16)
        row_idx = input_patch_config.get('row_idx', (H_orig - patch_size) // 2)
        col_idx = input_patch_config.get('col_idx', (W_orig - patch_size) // 2)
        
        r_start = max(0, row_idx)
        r_end = min(H_orig, row_idx + patch_size)
        c_start = max(0, col_idx)
        c_end = min(W_orig, col_idx + patch_size)
        actual_patch_h = r_end - r_start
        actual_patch_w = c_end - c_start

        if actual_patch_h <= 0 or actual_patch_w <=0:
            raise ValueError(f"Invalid patch dimensions for zero_padding: {actual_patch_h}x{actual_patch_w}")

        altered_input_patch_tensor_base = torch.zeros(B, C, patch_size, patch_size, device=original_images.device, dtype=original_images.dtype)
        altered_input_patch_tensor_base[:,:, 0:actual_patch_h, 0:actual_patch_w] = original_images[:, :, r_start:r_end, c_start:c_end]
        altered_input_patch_tensor = altered_input_patch_tensor_base.clone()

        def pad_preprocessor(patch_tensor_arg):
            padded_tensor = torch.zeros(B, C, H_orig, W_orig, device=patch_tensor_arg.device, dtype=patch_tensor_arg.dtype)
            target_r_start, target_r_end = row_idx, row_idx + patch_size
            target_c_start, target_c_end = col_idx, col_idx + patch_size
            source_r_start, source_r_end = 0, patch_size
            source_c_start, source_c_end = 0, patch_size
            
            final_target_r_start = max(0, target_r_start)
            final_target_r_end = min(H_orig, target_r_end)
            final_target_c_start = max(0, target_c_start)
            final_target_c_end = min(W_orig, target_c_end)

            final_source_r_start = source_r_start + (final_target_r_start - target_r_start)
            final_source_r_end = source_r_end - (target_r_end - final_target_r_end)
            final_source_c_start = source_c_start + (final_target_c_start - target_c_start)
            final_source_c_end = source_c_end - (target_c_end - final_target_c_end)
            
            if (final_target_r_end > final_target_r_start and
                final_target_c_end > final_target_c_start and
                final_source_r_end > final_source_r_start and
                final_source_c_end > final_source_c_start):
                padded_tensor[:,:, final_target_r_start:final_target_r_end, final_target_c_start:final_target_c_end] = \
                    patch_tensor_arg[:,:, final_source_r_start:final_source_r_end, final_source_c_start:final_source_c_end]
            return padded_tensor
        _preprocess_fn_internal = pad_preprocessor

    elif method == 'strided_subsample':
        downsize_factor = input_patch_config.get('downsize_factor')
        if downsize_factor is None or not isinstance(downsize_factor, int) or downsize_factor < 1:
            raise ValueError("'strided_subsample' method requires an integer 'downsize_factor' >= 1.")
        d = downsize_factor
        H_new = max(1, H_orig // d)
        W_new = max(1, W_orig // d)
        sampled_extent_h = (H_new - 1) * d + 1
        sampled_extent_w = (W_new - 1) * d + 1
        offset_h = (H_orig - sampled_extent_h) // 2
        offset_w = (W_orig - sampled_extent_w) // 2

        altered_input_patch_tensor = torch.zeros(B, C, H_new, W_new, device=original_images.device, dtype=original_images.dtype)
        for i_new in range(H_new):
            h_orig_idx = offset_h + i_new * d
            if not (0 <= h_orig_idx < H_orig): continue 
            for j_new in range(W_new):
                w_orig_idx = offset_w + j_new * d
                if not (0 <= w_orig_idx < W_orig): continue
                altered_input_patch_tensor[:, :, i_new, j_new] = original_images[:, :, h_orig_idx, w_orig_idx]

        def strided_subsample_preprocessor(patch_tensor_arg):
            reconstructed_image = torch.zeros(B, C, H_orig, W_orig, device=patch_tensor_arg.device, dtype=patch_tensor_arg.dtype)
            _, _, H_new_arg, W_new_arg = patch_tensor_arg.shape 
            for i_arg in range(H_new_arg):
                h_orig_idx_place = offset_h + i_arg * d
                if not (0 <= h_orig_idx_place < H_orig): continue
                for j_arg in range(W_new_arg):
                    w_orig_idx_place = offset_w + j_arg * d
                    if not (0 <= w_orig_idx_place < W_orig): continue
                    reconstructed_image[:, :, h_orig_idx_place, w_orig_idx_place] = patch_tensor_arg[:, :, i_arg, j_arg]
            return reconstructed_image
        _preprocess_fn_internal = strided_subsample_preprocessor
        
    elif method == 'none' or method is None:
        altered_input_patch_tensor = original_images.clone()
    else:
        raise ValueError(f"Unknown input_patch_config method: {method}")

    if altered_input_patch_tensor is None:
        raise ValueError("altered_input_patch_tensor could not be created.")
    return altered_input_patch_tensor, _preprocess_fn_internal

# --- Helper for AD Strategy__
def _get_jacobian_params(ad_strategy_str: str) -> Dict[str, Any]:
    """"
    Returns the parameters for the jacobian function based on the AD strategy.
    
    The output is to be used in
    torch.autograd.functional.jacobian function.
    
    
    """

    if ad_strategy_str == 'vectorized':
         # a specialized internal implementation to compute the Jacobian. This vectorized approach attempts to compute multiple columns (or rows, depending on the perspective) of the Jacobian matrix in parallel
        return {'vectorize': True} # strategy is ignored if vectorize=True
    elif ad_strategy_str == 'forward':
        return {'vectorize': False, 'strategy': 'forward-mode'}
    # Default to 'reverse' or if unknown strategy is passed
    return {'vectorize': False, 'strategy': 'reverse-mode'}



# --- Function 1: Jacobian of Full Output w.r.t. Input Patch ---
def compute_jacobian_rank_wrt_altered_input_patch(
    model: nn.Module,
    original_full_images: torch.Tensor,  # Used to derive the input patch
    input_patch_config: Dict[str, Any], # Dict for extract_altered_input_patch_and_preprocessor
    # ranks_list_ref,
    log_print_fn,
    sample_idx,
    ad_strategy='reverse', # 'reverse', 'forward', 'vectorized'
    precomputed_layer_outputs_for_shapes=None, # To get num_layers and shapes
    save_jacob=False, verbose=False, 
    #args_for_saving=None,
    specify_device: str = None # option: None, or 'cuda'/'cpu' to specify device
):
    """ 
    Computes the rank of the partial Jacobian matrix of a model
    respect to a selection of its input features.
    
    i.e. 
    output rank of d(FullLayerOutput) / d(InputPatch).
    

    """
    

    if original_full_images.shape[0] != 1:
        original_full_images = original_full_images[0:1] # Process one sample

    if specify_device is None:
        # use the device of the original_full_images tensor
        specify_device = original_full_images.device

    altered_input_patch_tensor, preprocess_fn_to_alter_input = extract_altered_input_patch_and_preprocessor(
            original_full_images, input_patch_config
        )
    
    
    altered_input_patch_tensor = altered_input_patch_tensor.detach().requires_grad_(True)

    if use_cuda_flag and not altered_input_patch_tensor.is_cuda: # Should be on cuda if original_full_images was
         altered_input_patch_tensor = altered_input_patch_tensor.cuda()


    # Determine number of layers from precomputed_outputs or a test run
    if precomputed_layer_outputs_for_shapes is None:
        model.eval()
        with torch.no_grad():
            # The model will receive the preprocessed patch
            test_model_input = preprocess_fn_to_alter_input(altered_input_patch_tensor.clone().detach())
            layer_outputs_for_shapes_raw = model(test_model_input)
            if not isinstance(layer_outputs_for_shapes_raw, (list, tuple)):
                layer_outputs_for_shapes_raw = [layer_outputs_for_shapes_raw]
            layer_outputs_for_shapes = layer_outputs_for_shapes_raw
    else:
        layer_outputs_for_shapes = precomputed_layer_outputs_for_shapes
        if not isinstance(layer_outputs_for_shapes, (list, tuple)):
            layer_outputs_for_shapes = [layer_outputs_for_shapes]
    
    num_layers = len(layer_outputs_for_shapes)
    
    
    # while len(ranks_list_ref) < num_layers: ranks_list_ref.append([])
    
    # container for ranks of each layer
    rank_for_this_sample = [None] * num_layers
    
    current_sample_ranks_for_log = []
    jacobian_backend_params = _get_jacobian_params(ad_strategy)

    for layer_idx in range(num_layers):
        def get_full_output_for_layer_fn(current_input_patch) -> torch.Tensor:
            model_input = preprocess_fn_to_alter_input(current_input_patch)
            all_outputs = model(model_input)
            if not isinstance(all_outputs, (list, tuple)):
                return all_outputs # Assume single output if not list/tuple
            return all_outputs[layer_idx]

        # Jacobian: d(FullLayerOutput) / d(InputPatchTensor)
        jacobian_tensor = torch.autograd.functional.jacobian(
            get_full_output_for_layer_fn,
            altered_input_patch_tensor,
            strict=True, create_graph=False, **jacobian_backend_params
        )
        # Expected shape: (B_out, C_out, H_out, W_out, B_in_patch, C_in_patch, H_in_patch, W_in_patch)
        # Assuming B_out=1 (from model output for single sample) and B_in_patch=1 (altered_input_patch_tensor is (1,C,H,W))
        jacobian_squeezed = jacobian_tensor.squeeze(0).squeeze(3) # Squeeze B_out and B_in_patch

        num_output_elements = jacobian_squeezed.shape[0] * jacobian_squeezed.shape[1] * jacobian_squeezed.shape[2]
        num_input_patch_elements = jacobian_squeezed.shape[3] * jacobian_squeezed.shape[4] * jacobian_squeezed.shape[5]
        jacobian_matrix = jacobian_squeezed.reshape(num_output_elements, num_input_patch_elements)
        
        jacob_rank = 0.0
        if jacobian_matrix.numel() > 0:
            try:
                jacob_rank = torch.matrix_rank(torch.mm(jacobian_matrix.T, jacobian_matrix)).item()
            except RuntimeError as e:
                log_print_fn(f"RuntimeError rank (InputPatch) L{layer_idx} S{sample_idx}: {e}. Shape: {jacobian_matrix.shape}. Rank=-1.\n")
                jacob_rank = -1.0
        
        ranks_list_ref[layer_idx].append(jacob_rank)
        current_sample_ranks_for_log.append(jacob_rank)

        if verbose:
            log_print_fn(f'[S{sample_idx} L{layer_idx} InPatch]: rank={jacob_rank:.2f}, J_shape={list(jacobian_matrix.shape)}\n')
        if save_jacob and jacobian_matrix.numel() > 0 and args_for_saving:
            # ... (saving logic as before, adapt filename)
            try:
                log_dir = 'jaco_logs_input_patch'
                os.makedirs(log_dir, exist_ok=True)
                model_name = getattr(args_for_saving, 'model', 'unk_model')
                saved_name = osp.join(log_dir, f'jin_{model_name}_s{sample_idx}_l{layer_idx}.pt')
                torch.save(jacobian_matrix.detach().cpu(), saved_name, pickle_protocol=4)
            except Exception as e:
                log_print_fn(f"Error saving Jin L{layer_idx} S{sample_idx}: {e}\n")


    log_print_fn(f'S{sample_idx:03d} (InPatchRanks {input_patch_config.get("method","N/A")}-{input_patch_config.get("patch_size","N/A")}): ' + 
                 ', '.join([f'{r:.2f}' for r in current_sample_ranks_for_log]) + '\n')

# ... (imports and _get_jacobian_params, extract_altered_input_patch_and_preprocessor remain the same) ...

# --- Function 1b: Jacobian of Full Output w.r.t. Input Patch ---
def compute_partial_jacobian_rank_wrt_original_input_patch(
    model: nn.Module,
    original_full_images: torch.Tensor, # Shape (1, C, H, W)
    input_patch_config: Dict[str, Any],
    log_print_fn: Callable[[str], None],
    sample_idx: Any,
    ad_strategy: str = 'reverse',
    precomputed_layer_outputs_for_shapes: Optional[List[torch.Tensor]] = None, # For shape info and num_layers
    save_jacob_fn: Optional[Callable[[torch.Tensor, int, Any, str], None]] = None,
    verbose: bool = False,
) -> List[Optional[float]]:
    """ 
    Computes ranks of Jacobians related to an input patch.

    If evaluate_at_original_image_point is True:
        Computes d(model(original_full_images)) / d(selected_patch_pixels_from_original).
        The 'selected_patch_pixels_from_original' are defined by input_patch_config.
        The model is evaluated at the original_full_images point.
    
    If evaluate_at_original_image_point is False:
        Computes d(model(preprocess_fn(input_patch))) / d(input_patch).
        The 'input_patch' is extracted via input_patch_config.
        The 'preprocess_fn' reconstructs a full image from the 'input_patch'.
        The model is evaluated at this reconstructed image point.

    Returns a list of ranks, one for each layer output of the model.
    """
    if original_full_images.shape[0] != 1:
        log_print_fn(f"Warning: Jacobian (InputPatch) expects batch size 1. Using first sample from batch of {original_full_images.shape[0]}.\n")
        original_full_images = original_full_images[0:1]

    device = next(model.parameters()).device
    original_full_images = original_full_images.to(device)
    model.eval()

    # This tensor will be what we differentiate with respect to.
    tensor_for_jacobian_input: torch.Tensor
    # This function defines how the model's actual input is constructed from tensor_for_jacobian_input
    construct_model_input_fn: Callable[[torch.Tensor], torch.Tensor]

    if evaluate_at_original_image_point:
        # Case B: Model sees original_full_images. Derivative w.r.t. selected patch pixels.
        # We extract the patch pixels to be the 'inputs' for the jacobian function.
        # The 'func' for jacobian will embed these patch pixels back into the original_full_images context.
        
        # Use extract_altered_input_patch_and_preprocessor to get the patch pixels
        # The 'preprocess_fn' returned here is NOT used to construct model input directly,
        # but helps define which pixels constitute the patch.
        patch_pixels_tensor, _ = extract_altered_input_patch_and_preprocessor(
            original_full_images.clone(), # Use a clone to be safe
            input_patch_config 
        )
        tensor_for_jacobian_input = patch_pixels_tensor.detach().requires_grad_(True)

        # Create a mask for the patch pixels within the full image dimensions
        # This requires knowing how 'extract_altered_input_patch_and_preprocessor' maps patch pixels
        # back to original image coordinates. This is implicitly defined by its 'method'.
        # For simplicity, let's assume 'input_patch_config' gives us enough info
        # (e.g., for 'zero_padding', it's row_idx, col_idx, patch_size).
        # For 'strided_subsample', it's downsize_factor and offsets.
        
        # We need a way to place `tensor_for_jacobian_input` (the patch values)
        # into the `original_full_images` structure, keeping other pixels fixed.
        
        # Create a version of original_full_images that does not require grad for non-patch parts.
        fixed_background = original_full_images.clone().detach()

        # The construct_model_input_fn will take the `tensor_for_jacobian_input` (patch pixels)
        # and combine them with `fixed_background`.
        # This is the tricky part: how to map patch_pixels_tensor back.
        # We can reuse the logic from the preprocessor of extract_altered_input_patch_and_preprocessor.
        
        # Get the "placement" function (similar to the preprocessor but for original image)
        # This is essentially what the preprocessor from extract_altered_input_patch_and_preprocessor does
        # if its input is the patch and it places it into a zero tensor.
        # Here, we want to place it into the `fixed_background`.

        # Let's get the "placement" preprocessor from a dummy call
        # This preprocessor knows how to take a patch-shaped tensor and put it into a full-sized image.
        _, placement_fn_for_patch = extract_altered_input_patch_and_preprocessor(
            torch.zeros_like(original_full_images), # Dummy, only need the function
            input_patch_config
        )
        
        # Create a mask of where the patch pixels go
        # This mask will be 1 where patch_pixels are, 0 elsewhere.
        # We can generate this by passing a tensor of ones through the placement_fn.
        patch_mask_for_placement = placement_fn_for_patch(
            torch.ones_like(tensor_for_jacobian_input, device=device)
        ).bool()


        def _construct_model_input_for_case_b(current_patch_pixels: torch.Tensor) -> torch.Tensor:
            # current_patch_pixels is tensor_for_jacobian_input
            # Place these pixels into the fixed_background using the mask
            # This assumes current_patch_pixels has the same shape as tensor_for_jacobian_input
            
            # Create an image that has the current_patch_pixels in their designated spots
            # and zeros elsewhere.
            image_with_patch_values = placement_fn_for_patch(current_patch_pixels)
            
            # Combine with fixed_background: take patch values where mask is true, background elsewhere.
            model_input_tensor = torch.where(patch_mask_for_placement, image_with_patch_values, fixed_background)
            return model_input_tensor
        
        construct_model_input_fn = _construct_model_input_for_case_b
        log_suffix_info = f"OrigImgPt-PatchDef:{input_patch_config.get('method','N/A')}"

    else: # evaluate_at_original_image_point is False (original Case A)
        # Model sees preprocessed(patch). Derivative w.r.t. patch.
        _altered_patch, _preprocess_fn = \
            extract_altered_input_patch_and_preprocessor(
                original_full_images, input_patch_config
            )
        tensor_for_jacobian_input = _altered_patch.to(device).detach().requires_grad_(True)
        construct_model_input_fn = lambda current_patch: _preprocess_fn(current_patch.to(device))
        log_suffix_info = f"AlteredImgPt-PatchDef:{input_patch_config.get('method','N/A')}"


    # --- Define the function for Jacobian computation (consistent for both cases) ---
    def model_outputs_all_layers_fn(current_input_to_diff: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # current_input_to_diff is tensor_for_jacobian_input
        model_input = construct_model_input_fn(current_input_to_diff)
        all_outputs = model(model_input)
        return tuple(all_outputs) if isinstance(all_outputs, (list, tuple)) else (all_outputs,)

    # --- Compute Jacobians ---
    jacobian_backend_params = _get_jacobian_params(ad_strategy)
    jacobians_tuple: Tuple[torch.Tensor, ...] = torch.autograd.functional.jacobian(
        model_outputs_all_layers_fn,
        tensor_for_jacobian_input, # This is what we differentiate w.r.t.
        strict=True, create_graph=False, **jacobian_backend_params
    )

    # --- Determine num_layers and precompute shapes if needed for robust reshape ---
    _layer_outputs_shapes_local: List[torch.Size]
    if precomputed_layer_outputs_for_shapes:
        _layer_outputs_shapes_local = [out.shape for out in precomputed_layer_outputs_for_shapes]
    else:
        with torch.no_grad():
            # Need to run the model once to get output shapes if not provided
            # Use a detached version of tensor_for_jacobian_input for this
            temp_model_input = construct_model_input_fn(tensor_for_jacobian_input.clone().detach())
            temp_outputs = model(temp_model_input)
            if not isinstance(temp_outputs, (list, tuple)): temp_outputs = [temp_outputs]
            _layer_outputs_shapes_local = [out.shape for out in temp_outputs]
    
    num_layers = len(jacobians_tuple)
    if num_layers != len(_layer_outputs_shapes_local):
        log_print_fn(f"Warning: Mismatch between num_layers from Jacobian ({num_layers}) and shape determination ({len(_layer_outputs_shapes_local))}).\n")
        # Fallback or error, for now, assume jacobians_tuple is the source of truth for num_layers
        # This might happen if precomputed_layer_outputs_for_shapes was from a different context.

    ranks_for_this_sample: List[Optional[float]] = [None] * num_layers
    
    for layer_idx, jacobian_tensor_for_layer in enumerate(jacobians_tuple):
        current_layer_output_shape: torch.Size = _layer_outputs_shapes_local[layer_idx]
        num_output_elements = int(torch.prod(torch.tensor(current_layer_output_shape[1:])))
        num_input_elements = int(torch.prod(torch.tensor(tensor_for_jacobian_input.shape[1:])))

        # Jacobian tensor shape: (*output_shape, *input_shape_for_jacobian)
        # We need to handle the batch dimensions carefully.
        # Output batch is dim 0 of current_layer_output_shape.
        # Input batch is dim 0 of tensor_for_jacobian_input.
        # jacobian_tensor_for_layer shape: (B_out, C_out, H_out, W_out, B_in, C_in, H_in, W_in)
        # We assume B_out=1 and B_in=1 from original_full_images and tensor_for_jacobian_input prep.
        
        # Squeeze B_out (dim 0) and B_in (dim len(current_layer_output_shape)-1, as it's after output dims)
        if jacobian_tensor_for_layer.shape[0] == 1 and \
           len(jacobian_tensor_for_layer.shape) > (len(current_layer_output_shape) -1) and \
           jacobian_tensor_for_layer.shape[len(current_layer_output_shape)-1] == 1:
            jacobian_squeezed = jacobian_tensor_for_layer.squeeze(0).squeeze(len(current_layer_output_shape)-1)
        else:
            log_print_fn(f"Warning: Jacobian squeeze assumption failed for L{layer_idx}. J shape: {jacobian_tensor_for_layer.shape}, Layer out shape: {current_layer_output_shape}, Input for J shape: {tensor_for_jacobian_input.shape}\n")
            jacobian_squeezed = jacobian_tensor_for_layer # Fallback, reshape might be incorrect

        try:
            jacobian_matrix = jacobian_squeezed.reshape(num_output_elements, num_input_elements)
        except RuntimeError as reshape_error:
            log_print_fn(f"Error reshaping Jacobian L{layer_idx} S{sample_idx}: {reshape_error}.\n"
                         f"  J_squeezed: {jacobian_squeezed.shape}, target_out_elem: {num_output_elements}, target_in_elem: {num_input_elements}.\n"
                         f"  Layer_out_shape: {current_layer_output_shape}, tensor_for_J_input_shape: {tensor_for_jacobian_input.shape}.\n")
            ranks_for_this_sample[layer_idx] = -2.0 # Reshape error
            continue

        jacob_rank = 0.0
        if jacobian_matrix.numel() > 0:
            try:
                jacob_rank = torch.matrix_rank(torch.mm(jacobian_matrix.T, jacobian_matrix)).item()
            except RuntimeError as e:
                log_print_fn(f"RuntimeError rank L{layer_idx} S{sample_idx}: {e}. J_matrix shape: {jacobian_matrix.shape}. Rank=-1.\n")
                jacob_rank = -1.0
        
        ranks_for_this_sample[layer_idx] = jacob_rank

        if verbose:
            log_print_fn(f'[S{sample_idx} L{layer_idx} {("OrigPt" if evaluate_at_original_image_point else "AlteredPt")}]: rank={jacob_rank:.2f}, J_shape={list(jacobian_matrix.shape)}\n')
        if save_jacob_fn and jacobian_matrix.numel() > 0 and jacob_rank != -2.0:
            save_jacob_fn(jacobian_matrix.detach().cpu(), layer_idx, sample_idx, 
                          f"input_patch_{'orig_eval' if evaluate_at_original_image_point else 'altered_eval'}")
            
    log_print_fn(f'S{sample_idx:03d} (Ranks {log_suffix_info}): ' +
                 ', '.join([f'{r:.2f}' if r is not None else 'NaN' for r in ranks_for_this_sample]) + '\n')

    return ranks_for_this_sample


# --- Function 2: Jacobian of Output Patch w.r.t. Full Input ---


# def compute_jacobian_rank_wrt_full_input_for_output_patch(
#     model,
#     original_full_images,
#     output_patch_config, # Dict: e.g. {'patch_h_factor': 0.5, ...} or {'y':10, 'h':16 ...}
#     ranks_list_ref,
#     log_print_fn,
#     sample_idx,
#     ad_strategy='reverse', # 'reverse', 'forward', 'vectorized'
#     precomputed_layer_outputs_for_shapes=None,
#     save_jacob=False, verbose=False, args_for_saving=None, use_cuda_flag=False
# ):
#     if original_full_images.shape[0] != 1:
#         original_full_images = original_full_images[0:1]

#     input_for_jacobian = original_full_images.clone().detach().requires_grad_(True)

#     if use_cuda_flag:
#         model = model.cuda()
#         input_for_jacobian = input_for_jacobian.cuda()
    
#     if precomputed_layer_outputs_for_shapes is None:
#         model.eval()
#         with torch.no_grad():
#             layer_outputs_for_shapes_raw = model(input_for_jacobian.clone().detach()) # Use a detached clone for this pass
#             if not isinstance(layer_outputs_for_shapes_raw, (list, tuple)):
#                 layer_outputs_for_shapes_raw = [layer_outputs_for_shapes_raw]
#             layer_outputs_for_shapes = layer_outputs_for_shapes_raw
#     else:
#         layer_outputs_for_shapes = precomputed_layer_outputs_for_shapes
#         if not isinstance(layer_outputs_for_shapes, (list, tuple)):
#             layer_outputs_for_shapes = [layer_outputs_for_shapes]

#     num_layers = len(layer_outputs_for_shapes)
#     while len(ranks_list_ref) < num_layers: ranks_list_ref.append([])

#     current_sample_ranks_for_log = []
#     jacobian_backend_params = _get_jacobian_params(ad_strategy)

#     for layer_idx, layer_shape_info in enumerate(layer_outputs_for_shapes):
#         B, C_out, H_out, W_out = layer_shape_info.shape

#         # --- Output patch definition logic (same as before, simplified here) ---
#         patch_h_eff, patch_w_eff = H_out, W_out # Default to full output
#         y_start, x_start = 0, 0
#         if 'patch_h' in output_patch_config and 'patch_w' in output_patch_config : # Fixed size
#             patch_h_cfg = min(output_patch_config['patch_h'], H_out)
#             patch_w_cfg = min(output_patch_config['patch_w'], W_out)
#             y_start = output_patch_config.get('y', (H_out - patch_h_cfg) // 2)
#             x_start = output_patch_config.get('x', (W_out - patch_w_cfg) // 2)
#             patch_h_eff, patch_w_eff = patch_h_cfg, patch_w_cfg
#         elif 'patch_h_factor' in output_patch_config: # Relative size
#             patch_h_f = max(1, int(H_out * output_patch_config['patch_h_factor']))
#             patch_w_f = max(1, int(W_out * output_patch_config['patch_w_factor']))
#             y_start = int(H_out * output_patch_config.get('y_offset_factor', (1-output_patch_config['patch_h_factor'])/2))
#             x_start = int(W_out * output_patch_config.get('x_offset_factor', (1-output_patch_config['patch_w_factor'])/2))
#             patch_h_eff, patch_w_eff = patch_h_f, patch_w_f
        
#         y_end = min(y_start + patch_h_eff, H_out)
#         x_end = min(x_start + patch_w_eff, W_out)
#         actual_patch_h = y_end - y_start
#         actual_patch_w = x_end - x_start

#         if actual_patch_h <= 0 or actual_patch_w <= 0:
#             # ... (logging for invalid patch) ...
#             ranks_list_ref[layer_idx].append(float('nan'))
#             current_sample_ranks_for_log.append(float('nan'))
#             continue
#         # --- End of output patch definition ---

#         def get_patched_output_for_layer_fn(current_full_input):
#             all_outputs = model(current_full_input)
#             if not isinstance(all_outputs, (list, tuple)):
#                 all_outputs = [all_outputs]
#             target_layer_output = all_outputs[layer_idx]
#             return target_layer_output[:, :, y_start:y_end, x_start:x_end]

#         jacobian_tensor = torch.autograd.functional.jacobian(
#             get_patched_output_for_layer_fn,
#             input_for_jacobian, # Full original input
#             strict=True, create_graph=False, **jacobian_backend_params
#         )
#         # Expected: (B_out_patch, C_out_patch, H_out_patch, W_out_patch, B_in_full, C_in_full, H_in_full, W_in_full)
#         # B_out_patch=1 (since it's from one layer's output for one sample), B_in_full=1
#         jacobian_squeezed = jacobian_tensor.squeeze(0).squeeze(3)

#         num_output_patch_elements = jacobian_squeezed.shape[0] * jacobian_squeezed.shape[1] * jacobian_squeezed.shape[2]
#         num_input_full_elements = jacobian_squeezed.shape[3] * jacobian_squeezed.shape[4] * jacobian_squeezed.shape[5]
#         jacobian_matrix = jacobian_squeezed.reshape(num_output_patch_elements, num_input_full_elements)

#         jacob_rank = 0.0
#         if jacobian_matrix.numel() > 0:
#             try:
#                 jacob_rank = torch.matrix_rank(torch.mm(jacobian_matrix.T, jacobian_matrix)).item()
#             except RuntimeError as e:
#                 log_print_fn(f"RuntimeError rank (OutPatch) L{layer_idx} S{sample_idx}: {e}. Shape: {jacobian_matrix.shape}. Rank=-1.\n")
#                 jacob_rank = -1.0
        
#         ranks_list_ref[layer_idx].append(jacob_rank)
#         current_sample_ranks_for_log.append(jacob_rank)

#         if verbose:
#             log_print_fn(f'[S{sample_idx} L{layer_idx} OutPatch({actual_patch_h}x{actual_patch_w})]: rank={jacob_rank:.2f}, J_shape={list(jacobian_matrix.shape)}\n')
#         if save_jacob and jacobian_matrix.numel() > 0 and args_for_saving:
#             # ... (saving logic, adapt filename)
#             try:
#                 log_dir = 'jaco_logs_output_patch'
#                 os.makedirs(log_dir, exist_ok=True)
#                 model_name = getattr(args_for_saving, 'model', 'unk_model')
#                 saved_name = osp.join(log_dir, f'jout_{model_name}_s{sample_idx}_l{layer_idx}.pt')
#                 torch.save(jacobian_matrix.detach().cpu(), saved_name, pickle_protocol=4)
#             except Exception as e:
#                 log_print_fn(f"Error saving Jout L{layer_idx} S{sample_idx}: {e}\n")

#     log_print_fn(f'S{sample_idx:03d} (OutPatchRanks Hf={output_patch_config.get("patch_h_factor","N/A")}): ' + 
#                  ', '.join([f'{r:.2f}' for r in current_sample_ranks_for_log]) + '\n')



# --- Main script structure (example of how to use) ---
if __name__ == '__main__':
    args = get_args()
    use_cuda = args.use_cuda if hasattr(args, 'use_cuda') else torch.cuda.is_available()

    # Data
    data_loader = build_data_loader(args, args.data, args.imagenet_dir, shuffle=False, # Usually False for specific sample analysis
                                    batch_size=1, num_workers=args.num_workers)
    if args.sample_idx is not None:
        selected_sample_indices = [int(item) for item in args.sample_idx.split(',')]
    else:
        selected_sample_indices = [1, 1] # Default to one sample for example
        args.sample_idx = ','.join([str(i) for i in selected_sample_indices])

    # Model
    net = build_model(args.model, args.method, no_epoch=args.epoch_num, use_cuda=use_cuda,
                      pretrained=not args.wo_pretrained, args=args)
    net.eval() # IMPORTANT: Set model to evaluation mode

    # Determine num_layers and layer names (once)
    num_layers = 0
    layer_names = []
    precomputed_shapes_for_all_samples = None # To store shapes if needed across calls
    with torch.no_grad():
        # Use a sample image from loader to get shapes
        try:
            sample_img_for_shape, _ = next(iter(data_loader))
            if use_cuda: sample_img_for_shape = sample_img_for_shape.cuda()
            
            # Get shapes for the "wrt_input_patch" case first, as model input might change
            # This part is a bit tricky ifpreprocess_fn_to_alter_input changes dimensions fed to model.
            # For simplicity, let's assume we can get a representative set of output shapes
            # from the full image input.
            outputs_for_shape_determination = net(sample_img_for_shape)
            if not isinstance(outputs_for_shape_determination, (list, tuple)):
                outputs_for_shape_determination = [outputs_for_shape_determination]
            precomputed_shapes_for_all_samples = outputs_for_shape_determination # Save these
            num_layers = len(outputs_for_shape_determination)
            layer_names = [f'Layer{i}' for i in range(num_layers)]
        except StopIteration:
            print("Data loader is empty. Cannot determine layer shapes.")
            exit()


    # Ranks storage (can have one for each method if comparing)
    ranks_input_patch_method = [[] for _ in range(num_layers)]
    ranks_output_patch_method = [[] for _ in range(num_layers)]

    # Logging
    # You might want separate log files or distinct sections in one log file
    log_suffix = f"{args.model}_{args.sample_idx}"
    if args.wo_pretrained: log_suffix += "_wo_pretrained"
    
    log_print_main = Logger(f'logs_jacob/jacobian_ranks_{log_suffix}.txt')
    log_print_main.write(f'Script args: {str(args)}\n')
    log_print_main.write(f'Layers ({num_layers}): {str(layer_names)}\n')
    log_print_main.write(f'AD Strategy for Jacobian: {args.ad_strategy if hasattr(args, "ad_strategy") else "reverse (default)"}\n\n')


    # --- Configuration for the two methods ---
    # Example: args.jacobian_mode = 'input_patch' or 'output_patch' or 'both'
    # Example: args.ad_strategy = 'reverse' / 'forward' / 'vectorized'

    input_patch_cfg = {
        'method': getattr(args, 'input_patch_method', 'zero_padding'), # 'zero_padding', 'interpolate', 'none'
        'patch_size': getattr(args, 'input_patch_size', 32), # e.g., 16, 32
        'row_idx': getattr(args, 'input_patch_row', (224 - getattr(args, 'input_patch_size', 32)) // 2), # Centered
        'col_idx': getattr(args, 'input_patch_col', (224 - getattr(args, 'input_patch_size', 32)) // 2)  # Centered
    }
    log_print_main.write(f'Input Patch Config: {input_patch_cfg}\n')

    output_patch_cfg = {
        # Factors relative to layer's output H, W
        'patch_h_factor': getattr(args, 'output_patch_h_factor', 0.5), 
        'patch_w_factor': getattr(args, 'output_patch_w_factor', 0.5),
        'y_offset_factor': getattr(args, 'output_patch_y_offset_factor', 0.25), # Centered by default
        'x_offset_factor': getattr(args, 'output_patch_x_offset_factor', 0.25)
        # Or fixed size: 'patch_h': 16, 'patch_w': 16, 'y': 10, 'x': 10
    }
    log_print_main.write(f'Output Patch Config: {output_patch_cfg}\n\n')
    
    ad_strat = getattr(args, 'ad_strategy', 'reverse')
    verbose_flag = getattr(args, 'verbose_jacobian', False)
    save_jacob_flag = getattr(args, 'save_jacobian_tensors', False)


    for i, (images, _) in enumerate(data_loader, start=1):
        if not (selected_sample_indices[0] <= i <= selected_sample_indices[1]):
            continue
        
        log_print_main.write(f"--- Processing Sample {i} ---\n")
        current_images = images # Should be (1, C, H, W) from loader

        # Option 1: Jacobian w.r.t. Input Patch
        if getattr(args, 'run_input_patch_jacobian', False): # Add this arg to your parser
            log_print_main.write(f"Running Jacobian w.r.t. Input Patch for Sample {i}\n")
            compute_partial_jacobian_rank_wrt_original_input_patch(
                model=net,
                original_full_images=current_images,
                input_patch_config=input_patch_cfg,
                ranks_list_ref=ranks_input_patch_method,
                log_print_fn=log_print_main.write,
                sample_idx=i,
                ad_strategy=ad_strat,
                precomputed_layer_outputs_for_shapes=precomputed_shapes_for_all_samples, # Pass cached shapes
                save_jacob=save_jacob_flag, verbose=verbose_flag, 
                args_for_saving=args, use_cuda_flag=use_cuda
            )

        # Option 2: Jacobian of Output Patch w.r.t. Full Input
        if getattr(args, 'run_output_patch_jacobian', False): # Add this arg to your parser
            log_print_main.write(f"Running Jacobian of Output Patch w.r.t. Full Input for Sample {i}\n")
            compute_jacobian_rank_wrt_full_input_for_output_patch(
                model=net,
                original_full_images=current_images,
                output_patch_config=output_patch_cfg,
                ranks_list_ref=ranks_output_patch_method,
                log_print_fn=log_print_main.write,
                sample_idx=i,
                ad_strategy=ad_strat,
                precomputed_layer_outputs_for_shapes=precomputed_shapes_for_all_samples, # Pass cached shapes
                save_jacob=save_jacob_flag, verbose=verbose_flag, 
                args_for_saving=args, use_cuda_flag=use_cuda
            )
        log_print_main.write(f"--- Finished Sample {i} ---\n\n")


    # Final summary logging
    if getattr(args, 'run_input_patch_jacobian', False) and any(len(r) > 0 for r in ranks_input_patch_method):
        log_print_main.write('\nSummary Ranks (Input Patch Method):\n')
        log_print_main.write('n_samples is {}: '.format(len(ranks_input_patch_method[0])) + ', '.join(
            ['{:.2f}'.format(np.mean(rank) if len(rank)>0 else np.nan) for rank in ranks_input_patch_method]) + '\n')

    if getattr(args, 'run_output_patch_jacobian', False) and any(len(r) > 0 for r in ranks_output_patch_method):
        log_print_main.write('\nSummary Ranks (Output Patch Method):\n')
        log_print_main.write('n_samples is {}: '.format(len(ranks_output_patch_method[0])) + ', '.join(
            ['{:.2f}'.format(np.mean(rank) if len(rank)>0 else np.nan) for rank in ranks_output_patch_method]) + '\n')

    log_print_main.close()
    print("Processing complete. Check log file.")
