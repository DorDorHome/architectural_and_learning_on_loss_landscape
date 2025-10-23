"""
Quick test to verify cuda:1 works with improved matrix conditioning.
This script starts with a fresh CUDA context.
"""

import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import warnings

print("Testing cuda:1 with improved sigma_geometry conditioning...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

device = torch.device('cuda:1')
print(f"\nUsing device: {device}")

# Test basic operations
print("\n1. Testing basic tensor operations...")
try:
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = x @ y
    print(f"   ✓ Matrix multiplication works: {z.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test eigendecomposition with conditioning
print("\n2. Testing eigendecomposition with ill-conditioned matrix...")
try:
    # Create an ill-conditioned symmetric matrix (like sigma often is)
    A = torch.randn(128, 128, device=device)
    sigma = A @ A.t()  # Positive semi-definite
    
    # Add tiny noise that might cause issues
    sigma = sigma + 1e-8 * torch.randn_like(sigma)
    
    # Apply conditioning (like our fix does)
    sigma_sym = 0.5 * (sigma + sigma.t())
    eps = 1e-5
    n = sigma_sym.size(0)
    regularization = eps * 10.0
    sigma_reg = sigma_sym + regularization * torch.eye(n, device=device, dtype=sigma.dtype)
    
    # Try eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(sigma_reg)
    print(f"   ✓ Eigendecomposition successful")
    print(f"   Eigenvalues range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
    print(f"   Condition number: {(eigvals.max() / eigvals.min()):.2e}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test the actual SigmaGeometry class
print("\n3. Testing SigmaGeometry class...")
try:
    from src.algos.supervised.rank_restoring.sigma_geometry import SigmaGeometry
    
    # Create a covariance-like matrix
    A = torch.randn(64, 128, device=device)
    sigma = (A.t() @ A) / 64  # Empirical covariance
    
    geometry = SigmaGeometry(sigma=sigma, diag_only=False, eps=1e-5)
    print(f"   ✓ SigmaGeometry initialized successfully")
    print(f"   Trace: {geometry.trace:.4f}")
    print(f"   Lambda_min: {geometry.lambda_min:.2e}")
    
    # Test operations
    v = torch.randn(128, device=device)
    v_norm = geometry.norm(v)
    print(f"   ✓ Norm computation works: {v_norm:.4f}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with very ill-conditioned matrix
print("\n4. Testing with extremely ill-conditioned matrix...")
try:
    # Create a matrix with huge condition number
    U = torch.randn(64, 64, device=device)
    U, _ = torch.linalg.qr(U)  # Orthonormal
    
    # Eigenvalues spanning many orders of magnitude
    eigvals_test = torch.logspace(-8, 2, 64, device=device)
    sigma_ill = U @ torch.diag(eigvals_test) @ U.t()
    
    print(f"   Original condition number: {(eigvals_test.max() / eigvals_test.min()):.2e}")
    
    geometry_ill = SigmaGeometry(sigma=sigma_ill, diag_only=False, eps=1e-5)
    print(f"   ✓ SigmaGeometry handled ill-conditioned matrix")
    print(f"   Regularized lambda_min: {geometry_ill.lambda_min:.2e}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed! cuda:1 is working properly with conditioning fixes.")
print("="*70)
