# during training, sometimes the loss can become non-finite (NaN or Inf).
# This can happen due to various reasons, such as numerical instability, exploding gradients, or issues with the data.
# This function is designed to log detailed information about the training state when a non-finite loss is detected.

import torch

def _log_and_raise_non_finite_error(task_idx, epoch,
                                    batch_idx, loss_val,
                                    input_data, label_data,
                                    output_from_network, learner, net):
    """Logs detailed information when a non-finite loss is detected and raises a ValueError to STOP training."""  
    print("\n" + "="*80)
    print(f"❌ NON-FINITE LOSS DETECTED - DETAILED DIAGNOSTIC REPORT")
    print(f"Location: task {task_idx}, epoch {epoch}, batch {batch_idx}")
    print(f"Loss value: {loss_val}")
    print("="*80)
    
    # 1. Check input and output data
    if input_data is not None and torch.is_tensor(input_data):
        input_finite = torch.all(torch.isfinite(input_data))
        print(f"Input data: shape={input_data.shape}, min={input_data.min().item():.6f}, max={input_data.max().item():.6f}, finite={input_finite}")
        if not input_finite:
            print("  ❌ Input data contains NaN/Inf - DATA CORRUPTION DETECTED")
    
    if label_data is not None and torch.is_tensor(label_data):
        label_finite = torch.all(torch.isfinite(label_data))
        print(f"Label data: shape={label_data.shape}, min={label_data.min().item():.6f}, max={label_data.max().item():.6f}, finite={label_finite}")
    
    if output_from_network is not None and torch.is_tensor(output_from_network):
        output_finite = torch.all(torch.isfinite(output_from_network))
        print(f"Network output: shape={output_from_network.shape}, min={output_from_network.min().item():.6f}, max={output_from_network.max().item():.6f}, finite={output_finite}")
        if not output_finite:
            print("  ❌ Network output contains NaN/Inf - MODEL IS BROKEN")
    
    # 2. Check intermediate layer features
    if hasattr(learner, 'previous_features') and learner.previous_features:
        print(f"\n📊 LAYER-BY-LAYER FEATURE ANALYSIS:")
        for i, feature in enumerate(learner.previous_features):
            if feature is not None and torch.is_tensor(feature):
                f_min, f_max = feature.min().item(), feature.max().item()
                f_mean = feature.mean().item()
                f_finite = torch.all(torch.isfinite(feature))
                print(f"  Layer {i}: shape={feature.shape}, min={f_min:.6f}, max={f_max:.6f}, mean={f_mean:.6f}, finite={f_finite}")
                if abs(f_max) > 1e10 or abs(f_min) > 1e10:
                    print(f"    ⚠️  EXTREME VALUES in layer {i}")
                if not f_finite:
                    print(f"    ❌ NaN/Inf in layer {i}")
    
    # 3. Detailed weight normalization diagnostics
    print(f"\n🔍 DETAILED WEIGHT NORMALIZATION ANALYSIS:")
    if net is not None:
        for name, module in net.named_modules():
            module_type = type(module).__name__
            if any(norm_type in module_type for norm_type in ['Normalized', 'normalized', 'Norm', 'norm', 'NormConv2d']):
                print(f"\n  📋 Layer: {name} (type: {module_type})")
                
                # Check weight parameter
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight
                    w_min, w_max = w.min().item(), w.max().item()
                    w_mean = w.mean().item()
                    w_finite = torch.all(torch.isfinite(w))
                    print(f"    Weight: shape={w.shape}, min={w_min:.6f}, max={w_max:.6f}, mean={w_mean:.6f}, finite={w_finite}")
                    if not w_finite:
                        nan_count = torch.isnan(w).sum().item()
                        inf_count = torch.isinf(w).sum().item()
                        print(f"    ❌ Weight has {nan_count} NaNs, {inf_count} Infs")
                
                # Check weight_scale parameter
                if hasattr(module, 'weight_scale') and module.weight_scale is not None:
                    ws = module.weight_scale
                    ws_min, ws_max = ws.min().item(), ws.max().item()
                    ws_mean = ws.mean().item()
                    ws_finite = torch.all(torch.isfinite(ws))
                    print(f"    Weight_scale: shape={ws.shape}, min={ws_min:.6f}, max={ws_max:.6f}, mean={ws_mean:.6f}, finite={ws_finite}")
                    if not ws_finite:
                        nan_count = torch.isnan(ws).sum().item()
                        inf_count = torch.isinf(ws).sum().item()
                        print(f"    ❌ Weight_scale has {nan_count} NaNs, {inf_count} Infs")
                    elif ws_max > 1e6:
                        print(f"    ⚠️  EXPLOSIVE weight_scale: max={ws_max:.2e}")
                
                # Check bias parameter
                if hasattr(module, 'bias') and module.bias is not None:
                    b = module.bias
                    b_min, b_max = b.min().item(), b.max().item()
                    b_mean = b.mean().item()
                    b_finite = torch.all(torch.isfinite(b))
                    print(f"    Bias: shape={b.shape}, min={b_min:.6f}, max={b_max:.6f}, mean={b_mean:.6f}, finite={b_finite}")
                    if not b_finite:
                        nan_count = torch.isnan(b).sum().item()
                        inf_count = torch.isinf(b).sum().item()
                        print(f"    ❌ Bias has {nan_count} NaNs, {inf_count} Infs")
                
                # Check weight normalization intermediate values if possible
                try:
                    if hasattr(module, 'weight') and module.weight is not None:
                        # Simulate the weight normalization computation to check intermediate values
                        w = module.weight
                        if 'NormalizedWeightsLinear' in module_type:
                            weight_norm_inverse = torch.rsqrt(torch.mean(w ** 2, dim=1, keepdim=True) + 1e-8)
                            wni_min, wni_max = weight_norm_inverse.min().item(), weight_norm_inverse.max().item()
                            wni_finite = torch.all(torch.isfinite(weight_norm_inverse))
                            print(f"    Weight_norm_inverse: min={wni_min:.6f}, max={wni_max:.6f}, finite={wni_finite}")
                            if not wni_finite:
                                print(f"    ❌ Weight_norm_inverse contains NaN/Inf")
                            if wni_max > 1e6:
                                print(f"    ⚠️  EXPLOSIVE weight_norm_inverse: max={wni_max:.2e}")
                        elif 'NormalizedRescalePerChannelWeightsLinear' in module_type:
                            weight_norm_inverse = torch.rsqrt(torch.sum(w ** 2, dim=1, keepdim=True) + 1e-8)
                            wni_min, wni_max = weight_norm_inverse.min().item(), weight_norm_inverse.max().item()
                            wni_finite = torch.all(torch.isfinite(weight_norm_inverse))
                            print(f"    Weight_norm_inverse: min={wni_min:.6f}, max={wni_max:.6f}, finite={wni_finite}")
                            if not wni_finite:
                                print(f"    ❌ Weight_norm_inverse contains NaN/Inf")
                            if wni_max > 1e6:
                                print(f"    ⚠️  EXPLOSIVE weight_norm_inverse: max={wni_max:.2e}")
                        elif 'NormConv2d' in module_type:
                            # For NormConv2d, check the L2 norm computation
                            weight_flat = w.view(w.shape[0], -1)  # Shape: [out_channels, ...]
                            norm = weight_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
                            norm_min, norm_max = norm.min().item(), norm.max().item()
                            norm_finite = torch.all(torch.isfinite(norm))
                            print(f"    Conv weight L2_norm per channel: min={norm_min:.6f}, max={norm_max:.6f}, finite={norm_finite}")
                            if not norm_finite:
                                print(f"    ❌ Conv weight norms contain NaN/Inf")
                            if norm_max > 1e6 or norm_min < 1e-6:
                                print(f"    ⚠️  EXTREME conv weight norms: min={norm_min:.2e}, max={norm_max:.2e}")
                            
                            # Check scalar parameter for NormConv2d
                            if hasattr(module, 'scalar') and module.scalar is not None:
                                scalar = module.scalar
                                scalar_min, scalar_max = scalar.min().item(), scalar.max().item()
                                scalar_finite = torch.all(torch.isfinite(scalar))
                                print(f"    Conv scalar parameter: min={scalar_min:.6f}, max={scalar_max:.6f}, finite={scalar_finite}")
                                if not scalar_finite:
                                    print(f"    ❌ Conv scalar contains NaN/Inf")
                                if scalar_max > 1e6:
                                    print(f"    ⚠️  EXPLOSIVE conv scalar: max={scalar_max:.2e}")
                except Exception as e:
                    print(f"    Warning: Could not compute normalization diagnostics for {name}: {e}")
                
                # Check gradients if available
                if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                    wg = module.weight.grad
                    wg_min, wg_max = wg.min().item(), wg.max().item()
                    wg_finite = torch.all(torch.isfinite(wg))
                    print(f"    Weight.grad: min={wg_min:.6f}, max={wg_max:.6f}, finite={wg_finite}")
                    if not wg_finite:
                        print(f"    ❌ Weight gradient contains NaN/Inf")
                
                if hasattr(module, 'weight_scale') and module.weight_scale is not None and module.weight_scale.grad is not None:
                    wsg = module.weight_scale.grad
                    wsg_min, wsg_max = wsg.min().item(), wsg.max().item()
                    wsg_finite = torch.all(torch.isfinite(wsg))
                    print(f"    Weight_scale.grad: min={wsg_min:.6f}, max={wsg_max:.6f}, finite={wsg_finite}")
                    if not wsg_finite:
                        print(f"    ❌ Weight_scale gradient contains NaN/Inf")
    
    # 4. Summary and recommendations
    print(f"\n💡 DEBUGGING RECOMMENDATIONS:")
    print("  1. Look for the first layer showing extreme values or NaN/Inf")
    print("  2. Check if weight_norm_inverse is exploding (indicates tiny weights)")
    print("  3. Check if weight_scale is exploding (unstable learning)")
    print("  4. Reduce learning rate by 10x")
    print("  5. Increase weight decay by 10x")
    print("  6. Add gradient clipping (max_norm=1.0)")
    print("  7. Consider different activation function (avoid ELU with weight norm)")
    
    print("="*80)
    print("TRAINING STOPPED - Use the above diagnostic info to fix the model")
    print("="*80)
    
    # CRITICAL: Actually stop the training by raising an exception
    raise ValueError(f"Non-finite loss detected at task {task_idx}, epoch {epoch}, batch {batch_idx}. "
                    f"Training stopped. Check the detailed diagnostic report above.")