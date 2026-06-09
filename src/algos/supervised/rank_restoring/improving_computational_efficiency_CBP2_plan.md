# Improving Computational Efficiency of RR-CBP2

This document outlines the computational bottlenecks in the RR-CBP2 implementation and proposed optimizations.

---

## 1. Analysis of Computational Bottlenecks

### 1.1 Overview of Key Variables

| Variable | Meaning | Typical Values |
|----------|---------|----------------|
| `d` | Input dimension to layer | Conv: 128×3×3 = 1152, FC: 512-4096 |
| `k` | Number of kept neurons | Layer width - replacements (e.g., 251 for 256-width layer replacing 5) |
| `r` | Number of replacements per batch | 1-10 typically |
| `m` | Number of samples for covariance | batch × spatial (e.g., 64×1024 = 65536 for conv) |
| `N` | Layer width | 64-512 typically |

### 1.2 Bottleneck Details

#### **CRITICAL: SigmaProjector._refresh_cache()** 
- **Location**: `sigma_geometry.py:241-283`
- **Operation**: Computes Gram matrix G = V^T Σ V and solves G^{-1}
- **Cost**: O(k³) per call
- **Frequency**: Called **every time** `add_vector()` is invoked (once per replacement)
- **Total cost per replacement batch**: O(r × k³)
- **Example**: 256-width layer, 5 replacements → 5 × 251³ ≈ 79 million operations

```python
# Current code (lines 249, 278)
gram = self.basis.t() @ sigma_basis       # O(d × k²)
self._G_inv = torch.linalg.solve(gram_reg, torch.eye(dim, ...))  # O(k³)
```

#### **HIGH: SigmaGeometry eigendecomposition**
- **Location**: `sigma_geometry.py:84`
- **Operation**: Full eigendecomposition of Σ to compute Σ^{1/2} and Σ^{-1/2}
- **Cost**: O(d³)
- **Frequency**: Once per layer per replacement batch (when SigmaGeometry is created)
- **Example**: d=1152 (conv layer with 128 input channels, 3×3 kernel) → 1.5 billion operations

```python
# Current code (line 84)
eigvals, eigvecs = torch.linalg.eigh(sigma_reg)
```

#### **HIGH: Covariance EMA update for Conv layers**
- **Location**: `rr_covariance.py:35`
- **Operation**: Computes H H^T / batch
- **Cost**: O(d² × m)
- **Frequency**: Every replacement batch
- **Example**: d=1152, m=65536 → 87 billion multiplications

```python
# Current code (line 35)
cov = h @ h.t() / float(batch)
```

#### **MEDIUM-HIGH: Used energy computation loop**
- **Location**: `rr_gnt2_conv.py:364-367`
- **Operation**: Computes Σ-energy for each kept vector individually
- **Cost**: O(k × d²) with Python loop overhead
- **Frequency**: Once per replacement batch

```python
# Current code
for col in range(kept_vectors.size(1)):
    used_energy += geometry.vector_energy(kept_vectors[:, col])
```

#### **MEDIUM-HIGH: lambda_star computation**
- **Location**: `rr_gnt2_conv.py:375-382`
- **Operation**: Whitens kept vectors, computes Gram, finds min eigenvalue
- **Cost**: O(d² × k + k³)
- **Frequency**: Once per layer if `use_lambda_star=True`

```python
# Current code
whitened = geometry.whiten_columns(kept_vectors)  # O(d² × k)
gram_white = whitened.t() @ whitened              # O(k² × d)
eigvals = torch.linalg.eigvalsh(gram_white)       # O(k³)
```

#### **MEDIUM: Rank metrics computation**
- **Location**: `rr_gnt2_conv.py:666-700`
- **Operation**: Computes W Σ W^T and its eigenvalues for logging
- **Cost**: O(N × d² + N³)
- **Frequency**: Every `log_rank_metrics_every` steps

---

## 2. Proposed Optimizations

### 2.1 Incremental Gram Inverse Update (Priority: CRITICAL)

**Problem**: `SigmaProjector._refresh_cache()` does O(k³) work every time a single vector is added.

**Solution**: Use Sherman-Morrison-Woodbury formula to update G^{-1} incrementally in O(k²).

**Mathematical basis**: When adding a vector v to basis V, the new Gram matrix is:
```
G_new = [G,        V^T Σ v  ]
        [v^T Σ V,  v^T Σ v  ]
```

Using block matrix inversion:
```
G_new^{-1} can be computed from G^{-1} in O(k²)
```

**Files to modify**: `sigma_geometry.py`
- Modify `SigmaProjector.add_vector()` method
- Add `_init_single_vector_cache()` helper
- Add `_incremental_update()` helper
- Keep `_refresh_cache()` for initial construction

**Expected speedup**: 10-30× for the replacement phase

---

### 2.2 Diagonal Covariance Option (Priority: HIGH)

**Problem**: Full covariance Σ ∈ ℝ^{d×d} requires O(d³) eigendecomposition.

**Solution**: Already implemented via `diag_sigma_only` config flag. When enabled:
- Eigendecomposition: O(d) instead of O(d³)
- Σ^{1/2} computation: O(d) instead of O(d³)
- Memory: O(d) instead of O(d²)

**Recommendation**: Use `diag_sigma_only=True` for layers where d > 512.

**Trade-off**: Assumes features are decorrelated. Works well after batch/layer normalization.

**Files to modify**: None (config change only)

**Expected speedup**: 100-1000× for eigendecomposition step

---

### 2.3 Subsample Covariance for Conv Layers (Priority: HIGH)

**Problem**: For conv layers, m = batch × spatial can be very large (e.g., 65536), making covariance computation O(d² × m) expensive.

**Solution**: Subsample spatial positions when m exceeds a threshold.

**Files to modify**: `rr_covariance.py`
- Modify `CovarianceState.update()` to accept `max_samples` parameter
- Add random subsampling when m > max_samples

**Example change**:
```python
def update(self, h: Tensor, dtype: Optional[str] = None, max_samples: int = 8192) -> Tensor:
    d, m = h.shape
    if m > max_samples:
        indices = torch.randperm(m, device=h.device)[:max_samples]
        h = h[:, indices]
        m = max_samples
    # ... rest unchanged
```

**Expected speedup**: Linear with subsampling ratio (e.g., 8× for 65536 → 8192)

---

### 2.4 Vectorize Used Energy Computation (Priority: MEDIUM)

**Problem**: Python loop over k vectors, each doing O(d²) work.

**Solution**: Replace loop with batched matrix operation.

**Files to modify**: `rr_gnt2_conv.py` (and `rr_gnt_conv.py` for v1)

**Current code**:
```python
used_energy = 0.0
for col in range(kept_vectors.size(1)):
    used_energy += geometry.vector_energy(kept_vectors[:, col])
```

**Proposed code**:
```python
if geometry.diag_only:
    used_energy = (kept_vectors * geometry.sigma.unsqueeze(1) * kept_vectors).sum().item()
else:
    used_energy = torch.einsum('id,dj,ij->', kept_vectors, geometry.sigma, kept_vectors).item()
```

**Expected speedup**: 2-5× (eliminates Python loop overhead, better memory access patterns)

---

## 3. Implementation Plan

### Phase 1: Incremental Gram Inverse (Highest Impact)

1. **Modify `SigmaProjector` class in `sigma_geometry.py`**:
   - Add `_init_single_vector_cache(vec)` method for base case
   - Add `_incremental_update(vec)` method using Sherman-Morrison
   - Modify `add_vector()` to use incremental update
   - Keep `_refresh_cache()` for `__init__` (initial k vectors)

2. **Add numerical stability safeguards**:
   - Check Schur complement for near-zero values
   - Periodic full recompute option (every N additions) to prevent error accumulation
   - Fallback to `_refresh_cache()` if incremental update fails

3. **Test thoroughly**:
   - Unit tests comparing incremental vs full recompute results
   - Numerical precision tests with ill-conditioned matrices

### Phase 2: Covariance Subsampling

1. **Modify `CovarianceState` in `rr_covariance.py`**:
   - Add `max_samples` parameter to `update()` method
   - Implement random subsampling logic

2. **Update config system**:
   - Add `covariance_max_samples` to `RRCBP2Config`

### Phase 3: Vectorized Energy Computation

1. **Add `batch_vector_energy()` method to `SigmaGeometry`**:
   - Efficient batched computation for multiple vectors

2. **Update `rr_gnt2_conv.py` and `rr_gnt_conv.py`**:
   - Replace loop with batched call

---

## 4. Summary Table

| Optimization | Target | Current Cost | New Cost | Speedup | Priority |
|--------------|--------|--------------|----------|---------|----------|
| Incremental Gram Inverse | `SigmaProjector.add_vector()` | O(r × k³) | O(r × k²) | 10-30× | CRITICAL |
| Diagonal Covariance | `SigmaGeometry.__init__()` | O(d³) | O(d) | 100-1000× | HIGH (config) |
| Covariance Subsampling | `CovarianceState.update()` | O(d² × m) | O(d² × m') | m/m' | HIGH |
| Vectorized Energy | `_replace_units()` loop | O(k × d²) + loop | O(k × d²) batched | 2-5× | MEDIUM |

---

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical instability in incremental inverse | Add Schur complement floor; periodic full recompute |
| Covariance subsampling loses information | Make subsampling optional with reasonable default threshold |
| Sherman-Morrison accumulates error over many updates | Recompute from scratch every N additions or when condition number degrades |
| Breaking existing functionality | Comprehensive unit tests comparing old vs new implementations |
