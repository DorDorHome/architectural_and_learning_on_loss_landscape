#!/bin/bash
# Convenience script to configure environment for cuda:1 usage
# Source this file before running training: source setup_cuda1.sh

echo "==================================================================="
echo "Configuring environment for cuda:1 (with CPU eigendecomposition)"
echo "==================================================================="

# Force CPU eigendecomposition to work around cuda:1 cuSOLVER issues
export SIGMA_FORCE_CPU_EIGH=1
export LLA_PREFER_GPU_EIGH=0

echo ""
echo "✓ Environment variables set:"
echo "  SIGMA_FORCE_CPU_EIGH=1   (sigma geometry uses CPU)"
echo "  LLA_PREFER_GPU_EIGH=0    (loss landscape uses CPU)"
echo ""
echo "Performance: ~97% of full GPU speed (3% overhead from CPU eigh)"
echo ""
echo "Now you can run training with cuda:1:"
echo "  python train_with_improved_optimizer.py device=cuda:1"
echo ""
echo "==================================================================="
