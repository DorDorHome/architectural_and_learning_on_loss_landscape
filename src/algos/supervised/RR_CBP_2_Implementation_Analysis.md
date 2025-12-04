# RR-CBP2 Implementation Analysis

This document provides a comprehensive mapping between the mathematical algorithms in `RR_CBP_2_algorithm_guide.md` and their implementation in the codebase. It serves as a reference for understanding how each algorithmic step is realized in code.

---

## Table of Contents

1. [Algorithm Recap](#1-algorithm-recap)
   - [1.1 CBP (Continual Backpropagation)](#11-cbp-continual-backpropagation)
   - [1.2 RR-CBP2 (Rank-Restoring CBP)](#12-rr-cbp2-rank-restoring-cbp)
   - [1.3 RR-CBP-E2 (RR-CBP with Energy Control)](#13-rr-cbp-e2-rr-cbp-with-energy-control)
2. [Class Architecture Overview](#2-class-architecture-overview)
3. [Algorithm-to-Code Mapping](#3-algorithm-to-code-mapping)
   - [3.1 Training Loop and Per-Step Updates](#31-training-loop-and-per-step-updates)
   - [3.2 Utility Computation and Maturity Testing](#32-utility-computation-and-maturity-testing)
   - [3.3 Covariance Tracking](#33-covariance-tracking)
   - [3.4 Σ-Geometry Operations](#34-σ-geometry-operations)
   - [3.5 Direction Sampling (RR-CBP2)](#35-direction-sampling-rr-cbp2)
   - [3.6 Energy Budget Allocation (RR-CBP-E2)](#36-energy-budget-allocation-rr-cbp-e2)
   - [3.7 Bias Transfer and Outgoing Weight Reset](#37-bias-transfer-and-outgoing-weight-reset)
4. [Design Philosophy](#4-design-philosophy)
5. [Configuration Parameters](#5-configuration-parameters)

---

## 1. Algorithm Recap

### 1.1 CBP (Continual Backpropagation)

**Purpose:** Periodically replace low-utility, mature neurons to maintain network plasticity.

**Key Components:**
- **Age tracking** ($a_{\ell,i}$): Steps since last reset
- **EMA of activations** ($f_{\ell,i}$): Exponential moving average of neuron activations
- **Utility EMA** ($u_{\ell,i}$): Combines activation magnitude with outgoing weight importance
- **Maturity threshold** (M): Neurons must be mature before replacement
- **Replacement rate** ($\rho$): Fraction of neurons replaced per step

**Per-Step Logic:**
```
for each step:
    1. Standard training step (forward, loss, backward, optimizer)
    2. Update per-neuron statistics (age, utility, mean activations)
    3. Identify mature neurons with lowest utility
    4. Replace selected neurons:
       - Transfer bias to next layer
       - Reinitialize incoming weights
       - Zero outgoing weights
       - Reset statistics
```

### 1.2 RR-CBP2 (Rank-Restoring CBP)

**Purpose:** Same as CBP but replaces neurons using **$\Sigma$-orthogonal directions** instead of random initialization. This restores rank in the feature covariance.

**Key Difference from CBP:**
The incoming weight vector w for a new neuron is chosen to be **orthogonal to all kept neurons in the $\Sigma$-geometry**:

    $\langle w, v_j \rangle_\Sigma = w^T \Sigma v_j = 0$   for all j $\in$ K (kept set)

**Direction Sampling (Section 5.1):**
1. Draw $u \sim \mathcal{N}(0, I_d)$
2. Project to $\Sigma$-orthogonal complement: $\hat{w} = (I - P_\Sigma) u$
3. If $|\hat{w}|_\Sigma$ > 0: normalize to unit $\Sigma$-norm
4. Else: use **least-covered direction** (Section 5.2)

**Scaling:** Unit $\Sigma$-norm ($|w|_\Sigma = 1$)

### 1.3 RR-CBP-E2 (RR-CBP with Energy Control)

**Purpose:** Same as RR-CBP2 but controls the **$\Sigma$-norm** of new weights to match a per-layer energy budget.

**Key Extensions (Section 6):**

1. **Per-unit target variance:**
   $q_{\text{tar}} = v_{\text{tar}} / \chi_0(\phi)$ = $(\text{tr}(\Sigma) / d) / \chi_0(\phi)$

2. **Layer energy budget:**
   $Q_{\text{tar}} = N \cdot q_{\text{tar}}$
   $Q_{\text{res}} = \max(Q_{\text{tar}} - Q_{\text{used}}, 0)$

3. **Allocation per new unit:**
   - **Underbudget:** $q_{\text{alloc}} = \min(q_{\text{tar}}, Q_{\text{res}} / r)$
   - **Overbudget (saturated):** $q_{\text{alloc}} = \min(q_{\text{tar}}, \max(q_{\min}, \lambda_*))$
     - where $q_{\min} = \tau \cdot \lambda_{\min}(\Sigma)$ is the rank-restoring floor

4. **Scaling step:**
   $w \leftarrow w \cdot \sqrt{q_{\text{alloc}}} / |w|_\Sigma$

---

## 2. Class Architecture Overview

### Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LEARNER LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  RankRestoringCBP2_for_FC        RankRestoringCBP2_for_ConvNet          │
│  (rr_cbp2_fc.py)                 (rr_cbp2_conv.py)                      │
│       │                                │                                │
│       │ owns                           │ owns                           │
│       ▼                                ▼                                │
├─────────────────────────────────────────────────────────────────────────┤
│                        GENERATE-AND-TEST LAYER                          │
├─────────────────────────────────────────────────────────────────────────┤
│  RR_GnT2_for_FC                  RR_GnT2_for_ConvNet                    │
│  (rr_gnt2_fc.py)                 (rr_gnt2_conv.py)                      │
│       │                                │                                │
│       │ inherits from                  │ inherits from                  │
│       ▼                                ▼                                │
│  GnT_for_FC                      ConvGnT_for_ConvNet                    │
│  (gnt.py)                        (gnt.py)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                          HELPER LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  CovarianceState      SigmaGeometry      SigmaProjector    EnergyAllocator│
│  (rr_covariance.py)   (sigma_geometry.py)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Responsibilities

| Class | File | Responsibility |
|-------|------|----------------|
| `RankRestoringCBP2_for_FC` | `rr_cbp2_fc.py` | High-level learner: orchestrates training loop, forward/backward, calls GnT |
| `RankRestoringCBP2_for_ConvNet` | `rr_cbp2_conv.py` | Same as above for ConvNets |
| `RR_GnT2_for_FC` | `rr_gnt2_fc.py` | Core algorithm: $\Sigma$-orthogonal replacement for FC layers |
| `RR_GnT2_for_ConvNet` | `rr_gnt2_conv.py` | Core algorithm: $\Sigma$-orthogonal replacement for Conv layers |
| `GnT_for_FC` | `gnt.py` | Base class: age tracking, utility computation, test_features |
| `ConvGnT_for_ConvNet` | `gnt.py` | Base class for ConvNets |
| `CovarianceState` | `rr_covariance.py` | Maintains EMA of feature covariance $\Sigma$ |
| `SigmaGeometry` | `sigma_geometry.py` | $\Sigma$-inner products, norms, whitening |
| `SigmaProjector` | `sigma_geometry.py` | $\Sigma$-orthogonal projection onto kept subspace |
| `EnergyAllocator` | `sigma_geometry.py` | Energy budget allocation (RR-CBP-E2 only) |
| `RRCBP2Config` | `configurations.py` | Configuration dataclass with all hyperparameters |

---

## 3. Algorithm-to-Code Mapping

### 3.1 Training Loop and Per-Step Updates

**Algorithm (Section 4.3 CBP pseudocode):**
```pseudo
for t = 1 to T do
  $\hat{y}_t \leftarrow f_\theta(x_t)$
  $L_t \leftarrow L(\theta; x_t, y_t)$
  $g_t \leftarrow \nabla_\theta L_t$
  $\theta \leftarrow OptimizerStep(\theta, g_t; \alpha)$
  ...
```

**Implementation:**

| Step | Class | Method | Code Location |
|------|-------|--------|---------------|
| Forward pass | `RankRestoringCBP2_for_FC` | `learn()` | `output, features = self.net.predict(x)` |
| Loss computation | `RankRestoringCBP2_for_FC` | `learn()` | `loss = self.loss_func(output, target)` |
| Backward pass | `RankRestoringCBP2_for_FC` | `learn()` | `loss.backward()` |
| Optimizer step | `RankRestoringCBP2_for_FC` | `learn()` | `self.opt.step()` |
| Generate-and-test | `RankRestoringCBP2_for_FC` | `learn()` | `self.rr_gnt.gen_and_test(...)` |

**Code excerpt from `rr_cbp2_fc.py`:**
```python
def learn(self, x: torch.Tensor, target: torch.Tensor):
    x, target = x.to(self.device), target.to(self.device)
    
    # Forward pass
    output, features = self.net.predict(x)
    loss = self.loss_func(output, target)
    
    # Backward pass and optimizer step
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    
    # Generate-and-test with $\Sigma$-orthogonal replacement
    if self.rr_gnt.config.rrcbp_enabled:
        self.opt.zero_grad()
        self.rr_gnt.gen_and_test(features=self.previous_features, batch_input=x)
```

---

### 3.2 Utility Computation and Maturity Testing

**Algorithm (Section 4.1-4.2):**
```pseudo
for i = 1 to N_ℓ do
  a_{ℓ,i} ← a_{ℓ,i} + 1                    # Increment age
  f_{ℓ,i} ← η f_{ℓ,i} + (1-η) h_{ℓ,i,t}   # Update activation EMA
  u_{ℓ,i} ← η u_{ℓ,i} + (1-η) y_{ℓ,i}     # Update utility EMA
end for

E_ℓ ← { i | a_{ℓ,i} ≥ M }                  # Mature set
S_ℓ ← indices of r_ℓ smallest u_hat_{ℓ,i}  # Replacement set
```

**Implementation:**

These steps are inherited from the base classes `GnT_for_FC` and `ConvGnT_for_ConvNet` in `gnt.py`. The RR-CBP2 classes reuse this logic completely.

| Step | Class | Method |
|------|-------|--------|
| Age increment | `GnT_for_FC` | `test_features()` → `self.ages[i] += 1` |
| Activation EMA | `GnT_for_FC` | `update_utility()` → `self.mean_feature_act[layer_idx] += (1-η) * features.mean(dim=0)` |
| Utility EMA | `GnT_for_FC` | `update_utility()` → `self.util[layer_idx] += (1-η) * new_util` |
| Maturity check | `GnT_for_FC` | `test_features()` → `self.ages[i] >= self.maturity_threshold` |
| Select lowest utility | `GnT_for_FC` | `test_features()` → `topk(smallest=True)` on bias-corrected utility |

**Key code from `gnt.py` (base class):**
```python
def update_utility(self, layer_idx=0, features=None):
    with torch.no_grad():
        self.util[layer_idx] *= self.decay_rate
        bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
        
        # Compute output weight magnitude (how important this neuron is to next layer)
        output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
        
        # Contribution utility: importance × activation magnitude
        if self.util_type == 'contribution':
            new_util = output_wight_mag * features.abs().mean(dim=0)
        
        self.util[layer_idx] += (1 - self.decay_rate) * new_util
```

---

### 3.3 Covariance Tracking

**Algorithm (Section 2.1):**

    $\Sigma := (1/m) H H^T$

with EMA update.

**Implementation:**

| Step | Class | Method | Algorithm Reference |
|------|-------|--------|---------------------|
| Initialize $\Sigma$ state | `RR_GnT2_for_FC` | `_ensure_covariance()` | Section 2.1 |
| Update $\Sigma$ EMA | `CovarianceState` | `update()` | EMA version of Eq. (2.1) |

**Code from `rr_covariance.py`:**
```python
def update(self, h: Tensor, dtype: Optional[str] = None) -> Tensor:
    with torch.no_grad():
        batch = h.shape[1]
        # Compute covariance: Σ = (1/m) H H^T
        if self.diag_only:
            cov = torch.mean(h * h, dim=1)  # Diagonal only
        else:
            cov = h @ h.t() / float(batch)  # Full covariance
        
        # EMA update: Σ ← β Σ + (1-β) cov
        ema.mul_(self.beta).add_(cov, alpha=1 - self.beta)
        
        # Add ridge regularization
        return ema + self.ridge * eye
```

**Code from `rr_gnt2_fc.py` showing where H_prev is computed:**
```python
def _compute_layer_inputs(self, layer_idx, features, batch_input, layer):
    """Compute H_prev (d × m matrix) for the layer."""
    if layer_idx == 0:
        inputs = batch_input
    else:
        inputs = features[layer_idx - 1]
    
    # Transpose to get (d, m) shape as per algorithm guide notation
    return inputs.t().to(layer.weight.dtype).contiguous()
```

---

### 3.4 $\Sigma$-Geometry Operations

**Algorithm (Section 2.2):**

    $\langle u, v \rangle_\Sigma = u^T \Sigma v$
    $|u|_\Sigma = \sqrt{u^T \Sigma u}$

**Algorithm (Section 2.4 - $\Sigma$-orthogonal projector):**

    $P_\Sigma = V (V^T \Sigma V)^{-1} V^T \Sigma$

**Implementation:**

| Operation | Class | Method | Algorithm Equation |
|-----------|-------|--------|-------------------|
| $\Sigma$-inner product | `SigmaGeometry` | `inner(u, v)` | Eq. (2.2) |
| $\Sigma$-norm | `SigmaGeometry` | `norm(vec)` | $\lvert u \rvert_\Sigma$ |
| Vector energy | `SigmaGeometry` | `vector_energy(vec)` | $\lvert u \rvert^2_\Sigma$ |
| Matrix energy | `SigmaGeometry` | `matrix_energy(W)` | $Q(W;\Sigma) = \text{tr}(W \Sigma W^T)$ |
| Whitening | `SigmaGeometry` | `whiten_columns(V)` | $\tilde{V} = \Sigma^{1/2} V$ |
| Unwhitening | `SigmaGeometry` | `unwhiten_vector(v)` | $\Sigma^{-1/2} v$ |
| Build projector | `SigmaProjector` | `__init__()` | Eq. (2.1) |
| Project to complement | `SigmaProjector` | `project_complement(u)` | $(I - P_\Sigma) u$ |
| Least-covered direction | `SigmaProjector` | `least_covered_direction()` | Section 5.2 |

**Code from `sigma_geometry.py`:**
```python
class SigmaGeometry:
    def inner(self, u: Tensor, v: Tensor) -> Tensor:
        """Σ-inner product: u^T Σ v"""
        if self.diag_only:
            return torch.dot(u, self.sigma * v)
        return torch.dot(u, self.sigma @ v)
    
    def norm(self, vec: Tensor) -> Tensor:
        """Σ-norm: sqrt(v^T Σ v)"""
        value = self.inner(vec, vec)
        return torch.sqrt(torch.clamp(value, min=0.0))

class SigmaProjector:
    def project_complement(self, vec: Tensor) -> Tensor:
        """Project to Σ-orthogonal complement: (I - P_Σ) vec"""
        return vec - self.apply(vec)
    
    def apply(self, vec: Tensor) -> Tensor:
        """Apply projector P_Σ v = V (V^T Σ V)^{-1} V^T Σ v"""
        sigma_vec = self.geometry.sigma @ vec
        coeff = self.basis.t() @ sigma_vec       # V^T Σ v
        proj_coeff = self._G_inv @ coeff         # (V^T Σ V)^{-1} V^T Σ v
        return self.basis @ proj_coeff           # V (...)
```

---

### 3.5 Direction Sampling (RR-CBP2)

**Algorithm (Section 5.1 and 5.4 pseudocode):**
```pseudo
# (2) Draw direction and project into $\Sigma$-orthogonal complement
$u      \leftarrow GaussianSample(d_\ell)$
w_hat  ← $(I - P_\Sigma) u$

if ||$\hat{w}$||_$\Sigma$ > 0 then
  $w_{\text{dir}} \leftarrow \hat{w} / ||\hat{w}||_\Sigma$
else
  $w_{\text{dir}} \leftarrow LeastCoveredDirection(W_{\text{keep}}, \Sigma_\ell)$
end if
```

**Implementation:**

| Step | Class | Method |
|------|-------|--------|
| Sample u ~ N(0,I) | `RR_GnT2_for_FC` | `_sample_sigma_direction()` |
| Project to complement | `SigmaProjector` | `project_complement()` |
| Normalize | `SigmaGeometry` | `norm()` then divide |
| Fallback direction | `SigmaProjector` | `least_covered_direction()` |

**Code from `rr_gnt2_fc.py`:**
```python
def _sample_sigma_direction(self, projector, geometry, dtype):
    """
    Sample a direction in the Σ-orthogonal complement of kept vectors.
    Returns: (direction, used_fallback)
    """
    attempts = max(1, int(self.config.max_proj_trials))
    
    for _ in range(attempts):
        # Draw u ~ N(0, I)
        u = torch.randn(geometry.dim, device=geometry.sigma.device, dtype=dtype)
        
        # Project into Σ-orthogonal complement: $(I - P_\Sigma) u$
        residual = projector.project_complement(u)
        norm = geometry.norm(residual)
        
        if norm > self.config.proj_eps:
            # Normalize to unit $\Sigma$-norm
            return residual / norm, False
    
    # Fallback: least-covered direction (Section 5.2)
    fallback = projector.least_covered_direction(dtype=dtype)
    norm = geometry.norm(fallback)
    return fallback / norm, True
```

**Least-covered direction (Section 5.2):**
```python
def least_covered_direction(self, dtype):
    """
    Find direction with \text{smallest} eigenvalue in M' = Σ^{1/2} V V^T Σ^{1/2}.
    Return: Σ^{-1/2} u_min, normalized.
    """
    # Whiten the basis: V~ = Σ^{1/2} V
    whitened = self.geometry.whiten_columns(self.basis)
    
    # Form M' = V~ V~^T = Σ^{1/2} V V^T Σ^{1/2}
    gram = whitened @ whitened.t()
    
    # Find eigenvector with \text{smallest} eigenvalue
    eigvals, eigvecs = torch.linalg.eigh(gram)
    idx = torch.argmin(eigvals)
    eigvec = eigvecs[:, idx]
    
    # Unwhiten: w = Σ^{-1/2} u_min
    candidate = self.geometry.unwhiten_vector(eigvec)
    
    # Project to complement and normalize
    residual = self.project_complement(candidate)
    return residual / self.geometry.norm(residual)
```

---

### 3.6 Energy Budget Allocation (RR-CBP-E2)

**Algorithm (Section 6.1-6.4):**

```pseudo
# (1) Compute layer targets
$v_{\text{tar}} \leftarrow (1 / d_\ell) * \text{trace}(\Sigma_\ell)$
$q_{\text{tar}} \leftarrow v_{\text{tar}} / \chi_0$

$Q_{\text{tar}} \leftarrow N_\ell * q_{\text{tar}}$
$Q_{\text{res}} \leftarrow \max(Q_{\text{tar}} - Q_{\text{used}}, 0)$

if $Q_{\text{res}}$ > 0 then
  $q_{\text{alloc}} \leftarrow \min(q_{\text{tar}}, Q_{\text{res}} / r_\ell)$           # Underbudget
else
  $\lambda_min_\Sigma \leftarrow \text{smallest}_eigenvalue(\Sigma_\ell)$
  $q_{\min} \leftarrow \tau * \lambda_min_\Sigma$                          # Rank-restoring floor
  $\lambda_* \leftarrow conditioning target (optional)
  $q_{\text{alloc}} \leftarrow \min(q_{\text{tar}}, \max(q_{\min}, \lambda_*))$     # Overbudget
end if

# (3) Scale direction
$w_{\text{dir}} \leftarrow w_{\text{dir}} * sqrt(q_{\text{alloc}}) / ||w_{\text{dir}}||_\Sigma$
```

**Implementation:**

| Step | Class | Method | Algorithm Reference |
|------|-------|--------|---------------------|
| Compute $\chi_0(\phi)$ | `chi0_for_activation()` | function | Section 6.1 |
| Compute $q_{\text{tar}}$ | `RR_GnT2_for_FC._replace_units()` | inline | Eq. in Section 6.1 |
| Compute $Q_{\text{used}}$ | `SigmaGeometry` | `vector_energy()` | Section 6.2 |
| Allocate q | `EnergyAllocator` | `allocate()` | Section 6.2-6.3 |
| Scale direction | `RR_GnT2_for_FC` | `_scale_to_energy()` | Section 6.4 step (3) |

**Code from `rr_gnt2_fc.py` (`_replace_units()`):**
```python
# Energy allocation setup (for RR-CBP-E2)
if config.use_energy_budget:
    chi0 = self._resolve_chi0(activations)
    
    # v_tar = (1/d) tr(Σ)
    v_target = geometry.trace / \max(geometry.dim, 1)
    
    # $q_{\text{tar}} = v_{\text{tar}} / \chi_0(\phi)$
    q_target = v_target / \max(chi0, config.proj_eps)
    
    # Q_used = $\Sigma_{i \in K}$ ||w_i||²_Σ
    used_energy = 0.0
    for col in range(kept_vectors.size(1)):
        used_energy += geometry.vector_energy(kept_vectors[:, col])
    
    # Create allocator
    allocator = EnergyAllocator(
        q_target=q_target,
        layer_size=weight_matrix.size(0),  # N_ℓ
        used_energy=used_energy,
        tau=config.tau,
        lambda_min_sigma=geometry.lambda_min,
        lambda_star=lambda_star,
        replacements=int(replace_idx.numel()),  # r_ℓ
    )
```

**Code from `sigma_geometry.py` (`EnergyAllocator`):**
```python
class EnergyAllocator:
    def __init__(self, q_target, layer_size, used_energy, tau, 
                 lambda_min_sigma, lambda_star, replacements):
        self.q_target = q_target
        self.total_target = q_target * layer_size           # Q_tar = N * q_tar
        self.residual = max(0.0, self.total_target - used_energy)  # Q_res
        self.q_min = tau * lambda_min_sigma                 # Rank-restoring floor
    
    def allocate(self):
        saturated = self.residual <= 0.0
        
        if not saturated:
            # Underbudget: fair share of remaining budget
            q_alloc = min(self.q_target, self.residual / self.remaining)
            self.residual -= q_alloc
        else:
            # Overbudget: use rank-restoring floor
            floor = self.q_min
            if self.lambda_star is not None:
                floor = max(floor, self.lambda_star)
            q_alloc = min(self.q_target, floor)
        
        self.remaining -= 1
        return q_alloc, saturated
```

**Code for scaling (`_scale_to_energy()`):**
```python
def _scale_to_energy(self, direction, geometry, q_alloc):
    """Scale direction to achieve ||w||²_Σ = q_alloc"""
    norm = geometry.norm(direction)
    scale = math.sqrt(q_alloc) / norm
    return direction * scale
```

---

### 3.7 Bias Transfer and Outgoing Weight Reset

**Algorithm (Section 4.2 and 5.3):**
```pseudo
# (1) Bias transfer
TransferBiasFromUnit(ℓ, i, f_hat_{ℓ,i}, outgoing_weights)

# (4) Bias centering
$a \leftarrow w_{\text{dir}}^T H_{\text{prev}}$
$b \leftarrow -mean(a)$
SetBias(ℓ, i, b)

# (5) Zero outgoing weights
ZeroOutgoingWeights(ℓ, i)
```

**Implementation:**

| Step | Class | Method |
|------|-------|--------|
| Bias transfer | `RR_GnT2_for_FC` | `_transfer_bias()` |
| Bias centering | `RR_GnT2_for_FC` | `_center_bias()` |
| Zero outgoing | `RR_GnT2_for_FC` | `_zero_and_seed_outgoing()` |

**Code from `rr_gnt2_fc.py`:**
```python
def _transfer_bias(self, next_layer, unit_idx, bias_corrected_act):
    """Transfer bias from removed unit to next layer."""
    # Contribution to next layer = outgoing_weight * mean_activation
    contribution = next_layer.weight.data[:, unit_idx] * bias_corrected_act
    next_layer.bias.data += contribution

def _center_bias(self, activations):
    """Compute bias to center preactivations: b = -mean(a)"""
    if self.config.center_bias == "median":
        return -torch.median(activations)
    return -activations.mean()

def _zero_and_seed_outgoing(self, next_layer, unit_idx):
    """Zero outgoing weights (optionally add micro-seed)."""
    next_layer.weight.data[:, unit_idx] = 0.0
    if self.config.use_micro_seed:
        noise = torch.randn_like(next_layer.weight.data[:, unit_idx])
        next_layer.weight.data[:, unit_idx] = epsilon * (noise / noise.norm())
```

---

## 4. Design Philosophy

### 4.1 Separation of Concerns

The implementation follows a clean separation:

1. **Learner Layer** (`rr_cbp2_fc.py`, `rr_cbp2_conv.py`):
   - Orchestrates the training loop
   - Owns the optimizer and network
   - Calls GnT after each training step
   - *Philosophy:* The learner is the user-facing API; it should be simple and hide complexity.

2. **GnT Layer** (`rr_gnt2_fc.py`, `rr_gnt2_conv.py`):
   - Implements the core replacement algorithm
   - Owns covariance states and statistics
   - Inherits utility computation from base classes
   - *Philosophy:* The GnT class encapsulates all $\Sigma$-geometry logic, separate from training.

3. **Helper Layer** (`sigma_geometry.py`, `rr_covariance.py`):
   - Provides mathematical primitives (Σ-inner product, projection, etc.)
   - Stateless except for EMA tracking
   - *Philosophy:* Mathematical operations should be reusable and testable in isolation.

### 4.2 Inheritance for Code Reuse

The `RR_GnT2_for_FC` class inherits from `GnT_for_FC` to reuse:
- Age tracking infrastructure (`self.ages`)
- Utility computation (`update_utility()`)
- Maturity testing logic (`test_features()`)
- Mean activation tracking (`self.mean_feature_act`)

This avoids duplicating ~200 lines of CBP utility logic while allowing the RR-CBP2 to override only the weight initialization method.

### 4.3 Configuration-Driven Behavior

The `RRCBP2Config` dataclass controls the algorithm variant:
- `use_energy_budget=False` → **RR-CBP2** (unit $\Sigma$-norm)
- `use_energy_budget=True` → **RR-CBP-E2** (energy budget)

This single flag switches between algorithms without code changes:
```python
if config.use_energy_budget and allocator is not None:
    q_alloc, saturated = allocator.allocate()
    w_scaled = self._scale_to_energy(direction, geometry, q_alloc)
else:
    w_scaled = direction  # Already unit $\Sigma$-norm
```

### 4.4 Incremental Projector Updates

When replacing multiple neurons in one step, the projector is updated incrementally:
```python
for unit_idx in replace_idx:
    direction = _sample_sigma_direction(projector, ...)
    projector.add_vector(w_scaled)  # Add to basis for next iteration
```

This ensures each new neuron is orthogonal to all previously added neurons in the same batch.

---

## 5. Configuration Parameters

The `RRCBP2Config` class contains all hyperparameters. Key ones grouped by purpose:

### CBP Parameters (inherited)
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `neurons_replacement_rate` | 0.001 | Fraction $\rho$ of neurons to replace |
| `maturity_threshold` | 100 | Age M before eligible for replacement |
| `decay_rate_utility_track` | 0.9 | EMA decay rate $\eta$ for utility |
| `util_type` | "contribution" | Utility formula |

### Covariance Tracking
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sigma_ema_beta` | 0.99 | EMA decay for covariance $\Sigma$ |
| `sigma_ridge` | 0.0001 | Ridge regularization for $\Sigma$ |
| `diag_sigma_only` | False | Use diagonal approximation |

### Σ-Geometry
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_proj_trials` | 4 | Attempts before fallback to least-covered |
| `proj_eps` | 1e-8 | Numerical floor for norms |
| `sigma_eig_floor` | 1e-6 | Minimum eigenvalue for $\Sigma$ |
| `orthonormalize_batch` | True | Orthonormalize new vectors within batch |

### Energy Budget (RR-CBP-E2)
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_energy_budget` | False | Enable RR-CBP-E2 mode |
| `tau` | 0.01 | Rank-restoring floor multiplier |
| `use_lambda_star` | False | Use conditioning target |
| `lambda_star` | None | Override conditioning target |
| `chi0_override` | None | Override $\chi_0$($\phi$) constant |

### Bias and Outgoing Weights
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `center_bias` | "mean" | Bias centering method ("mean" or "median") |
| `use_micro_seed` | False | Add small noise to zeroed outgoing |
| `epsilon_micro_seed` | 0.0001 | Micro-seed magnitude |

---

## Summary

The RR-CBP2 implementation cleanly separates:
1. **What to replace** (CBP utility logic in base classes)
2. **How to replace** (Σ-orthogonal initialization in RR_GnT2_*)
3. **How much energy** (EnergyAllocator for RR-CBP-E2)

The algorithm guide's pseudocode maps directly to methods:
- `CBP per-step updates` → `GnT_for_FC.update_utility()`, `GnT_for_FC.test_features()`
- `RR_ReinitUnit()` → `RR_GnT2_for_FC._replace_units()`
- `SigmaProjector()` → `SigmaProjector.__init__()`, `project_complement()`
- `LeastCoveredDirection()` → `SigmaProjector.least_covered_direction()`
- `RR_EnergyAwareReinitUnit()` → `_replace_units()` with `EnergyAllocator`

This design allows easy extension (e.g., new utility types, new energy allocation schemes) while maintaining fidelity to the mathematical formulation.
