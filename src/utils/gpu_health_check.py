"""
GPU Health Check Utility

Provides functions to detect GPU memory corruption and handle it gracefully.
"""

import torch
import warnings


class GPUCorruptionError(RuntimeError):
    """Raised when GPU memory corruption is detected and training cannot continue."""
    pass


def check_gpu_health(device: torch.device) -> bool:
    """
    Check if GPU is healthy by attempting a simple operation.
    
    Args:
        device: The GPU device to check
        
    Returns:
        True if GPU is healthy, False if corrupted
    """
    if not device.type == 'cuda':
        return True  # CPU is always "healthy"
    
    try:
        # Try a simple CUDA operation
        test_tensor = torch.zeros(10, 10, device=device)
        _ = test_tensor @ test_tensor
        return True
    except RuntimeError:
        return False


def raise_gpu_corruption_error(device: torch.device, original_error: Exception):
    """
    Raise a GPUCorruptionError with helpful diagnostic information.
    
    Args:
        device: The corrupted GPU device
        original_error: The original error that triggered detection
    """
    error_msg = f"""
================================================================================
GPU MEMORY CORRUPTION DETECTED ON {device}
================================================================================

The GPU has entered a corrupted state and cannot continue training.
This is often caused by:
  1. Hardware issues (GPU memory errors, overheating)
  2. Driver instability
  3. CUDA library bugs (especially with older PyTorch/CUDA versions)

Original error: {original_error}

RECOMMENDED ACTIONS:
-------------------
1. Check GPU health:
   nvidia-smi -q | grep -A 10 "ECC Errors"
   
2. Try a different GPU if available:
   device=cuda:0  (or cuda:1, cuda:2, etc.)
   
3. Reset the GPU (may require reboot):
   sudo nvidia-smi --gpu-reset -i {device.index if hasattr(device, 'index') else 'N/A'}
   
4. Check GPU temperature and memory:
   nvidia-smi
   
5. Update PyTorch/CUDA if feasible (current: PyTorch {torch.__version__})

6. As a last resort, force CPU eigendecomposition:
   export SIGMA_FORCE_CPU_EIGH=1  # For sigma geometry (rank-restoring)
   export LLA_PREFER_GPU_EIGH=0   # For loss landscape analysis
   
================================================================================
Training cannot continue. Please address GPU issues and restart.
================================================================================
"""
    raise GPUCorruptionError(error_msg)
