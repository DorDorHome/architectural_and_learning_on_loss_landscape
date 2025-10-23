# GPU Selection Quick Reference

## Default (Recommended)
```bash
python train_with_improved_optimizer.py
# Uses device=cuda:0 (from config.yaml)
# 100% performance, fully reliable
```

## Use cuda:1 (with fix)
```bash
# Method 1: Source the setup script
source setup_cuda1.sh
python train_with_improved_optimizer.py device=cuda:1

# Method 2: Inline environment variables
SIGMA_FORCE_CPU_EIGH=1 LLA_PREFER_GPU_EIGH=0 \
  python train_with_improved_optimizer.py device=cuda:1

# Method 3: Export then run
export SIGMA_FORCE_CPU_EIGH=1
export LLA_PREFER_GPU_EIGH=0
python train_with_improved_optimizer.py device=cuda:1
```

## Override device from command line
```bash
# Use cuda:0
python train_with_improved_optimizer.py device=cuda:0

# Use cuda:1 (remember to set environment variables!)
source setup_cuda1.sh
python train_with_improved_optimizer.py device=cuda:1

# Use CPU
python train_with_improved_optimizer.py device=cpu
```

## Test GPU before long runs
```bash
# Test cuda:0 (should pass)
python test_cuda1_fix.py  # uses cuda:1 by default
# Edit script to use cuda:0 if needed

# Test cuda:1 (will show cuSOLVER error without env vars)
python test_cuda1_fix.py

# Test cuda:1 with CPU fallback (should pass)
SIGMA_FORCE_CPU_EIGH=1 python test_cuda1_fix.py
```

## When to Use Which GPU?

### Use cuda:0 when:
- ✅ You want maximum reliability
- ✅ You want 100% performance
- ✅ You're running long experiments
- ✅ **This is the default and recommended choice**

### Use cuda:1 when:
- ⚠️ cuda:0 is busy with other work
- ⚠️ You specifically need to balance load
- ⚠️ You accept ~3% performance overhead
- ⚠️ **Remember to source setup_cuda1.sh first!**

## Troubleshooting

### "cuSOLVER error" on cuda:1
```bash
# Solution: Use CPU eigendecomposition
source setup_cuda1.sh
# Then re-run your command
```

### "GPU corruption detected"
```bash
# Solution 1: Use cuda:0 instead
python train_with_improved_optimizer.py device=cuda:0

# Solution 2: Reset GPU (if no other processes using it)
sudo nvidia-smi --gpu-reset -i 1

# Solution 3: Reboot system
sudo reboot
```

### Check GPU status
```bash
# Monitor GPUs
nvidia-smi -l 1

# Check for errors
nvidia-smi -q -i 1 | grep -i error

# Check what's using GPUs
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Files to Read for More Info

- `CUDA1_FIX_SUMMARY.md` - Complete summary of the issue and fixes
- `GPU_CUDA1_ISSUE.md` - Detailed technical documentation
- `setup_cuda1.sh` - Convenience script for cuda:1
- `test_cuda1_fix.py` - Test script to verify GPU health
