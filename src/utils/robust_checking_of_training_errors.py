# during training, sometimes the loss can become non-finite (NaN or Inf).
# This can happen due to various reasons, such as numerical instability, exploding gradients, or issues with the data.
# This function is designed to log detailed information about the training state when a non-finite loss is detected.

import torch

def _log_and_raise_non_finite_error(task_idx, epoch,
                                    batch_idx, loss_val,
                                    input_data, label_data,
                                    output_from_network, learner, net):
    """Logs detailed information when a non-finite loss is detected and raises a ValueError."""  
    # check loss from loss function
    print(f"Non-finite loss detected at task {task_idx}, epoch {epoch}, batch {batch_idx}")
    
    # Check loss_val
    if torch.is_tensor(loss_val):
        print(f"Loss value: {loss_val.item()}")
    else:
        print(f"Loss value: {loss_val} (already a float or non-tensor)")

    
    # Check input_data
    if input_data is not None and torch.is_tensor(input_data):
        print(f"Input data stats: min={input_data.min().item()}, max={input_data.max().item()}, mean={input_data.mean().item()}, isfinite={torch.all(torch.isfinite(input_data))}")
    elif input_data is None:
        print("Input data is None.")
    else:
        print("Input data is not a tensor or is None.")

    # Check label_data
    if label_data is not None and torch.is_tensor(label_data):
        # Add a check for label range if it's for classification
        print(f"Label stats: min={label_data.min().item()}, max={label_data.max().item()}, isfinite={torch.all(torch.isfinite(label_data))}")
    elif label_data is None:
        print("Label data is None.")
    else:
        print("Label data is not a tensor or is None.")
    
    # check output from network
    if output_from_network is not None and torch.is_tensor(output_from_network):
        print(f"Output from network stats: min={output_from_network.min().item()},max={output_from_network.max().item()},mean={output_from_network.mean().item()},isfinite={torch.all(torch.isfinite(output_from_network))},shape={output_from_network.shape}, dtype={output_from_network.dtype}")
    elif output_from_network is None:
        print("Output from network is None.")
    else:
        print(f"Output from network is not a tensor or is None. Type: {type(output_from_network)}")
    
    
    # Check features from learner
    if hasattr(learner, 'previous_features') and learner.previous_features is not None:
        if not learner.previous_features: # Check if the list is empty
            print("learner.previous_features is an empty list.")
        else:
            for i, feature in enumerate(learner.previous_features):
                if feature is not None and torch.is_tensor(feature):
                    print(f"Feature layer {i} stats: min={feature.min().item()}, max={feature.max().item()}, mean={feature.mean().item()}, isfinite={torch.all(torch.isfinite(feature))}")
                elif feature is None:
                    print(f"Feature layer {i} is None.")
                else:
                    print(f"Feature layer {i} is not a tensor or is None.")
    elif hasattr(learner, 'previous_features') and learner.previous_features is None:
         print("Learner's 'previous_features' attribute is None.")
    else:
        print("Learner does not have 'previous_features' attribute or it's not accessible.")

    # Check model parameters from net
    print("Model parameter statistics (at point of non-finite loss detection):")
    if net is not None:
        for name, param in net.named_parameters():
            print(f"Parameter {name}:")
            if param.data is not None and torch.is_tensor(param.data):
                print(f"  Data stats: min={param.data.min().item()}, max={param.data.max().item()}, mean={param.data.mean().item()}, isfinite={torch.all(torch.isfinite(param.data))}")
            else:
                print("  Data is None or not a tensor.")
            
            if param.grad is not None and torch.is_tensor(param.grad):
                print(f"  Grad stats: min={param.grad.min().item()}, max={param.grad.max().item()}, mean={param.grad.mean().item()}, isfinite={torch.all(torch.isfinite(param.grad))}")
            else:
                print(f"  Grad: None or not a tensor (This is expected if optimizer.zero_grad() has been called or if error occurred before .backward())")
    else:
        print("Network object (net) is None.")
        
    print(f"--- END OF NON-FINITE LOSS DIAGNOSTICS ---")
    raise ValueError(f"Non-finite loss detected at task {task_idx}, epoch {epoch}, batch {batch_idx}. Stopping training. Check logs above for details.")