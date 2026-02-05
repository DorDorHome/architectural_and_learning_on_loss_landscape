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
│  (src/algos/gnt.py)              (src/algos/gnt.py)                     │
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
| `GnT_for_FC` | `src/algos/gnt.py` | Base class: age tracking, utility computation, test_features |
| `ConvGnT_for_ConvNet` | `src/algos/gnt.py` | Base class for ConvNets |
| `CovarianceState` | `rr_covariance.py` | Maintains EMA of feature covariance $\Sigma$ |
| `SigmaGeometry` | `sigma_geometry.py` | $\Sigma$-inner products, norms, whitening |
| `SigmaProjector` | `sigma_geometry.py` | $\Sigma$-orthogonal projection onto kept subspace |
| `EnergyAllocator` | `sigma_geometry.py` | Energy budget allocation (RR-CBP-E2 only) |
| `RRCBP2Config` | `configurations.py` | Configuration dataclass with all hyperparameters |

---

## 3. Algorithm-to-Code Mapping

### 3.1 Training Loop and Per-Step Updates

**Algorithm (Section 4.3 CBP pseudocode):**
- **for** $t = 1$ **to** $T$ **do**
  - $\hat{y}_t \leftarrow f_\theta(x_t)$
  - $L_t \leftarrow L(\theta; x_t, y_t)$
  - $g_t \leftarrow \nabla_\theta L_t$
  - $\theta \leftarrow \mathrm{OptimizerStep}(\theta, g_t; \alpha)$
  - $\ldots$

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
    
    # Forward pass to get predictions and hidden features
    output, features = self.net.predict(x)
    loss = self.loss_func(output, target)
    self.previous_features = features
    
    # Backward pass and optimizer step
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    
    # Generate-and-test with $\Sigma$-orthogonal replacement
    if self.rr_gnt.config.rrcbp_enabled:
        self.opt.zero_grad()
        self.rr_gnt.gen_and_test(features=self.previous_features, batch_input=x)

        # Re-run forward pass to get fresh features after network modifications
        with torch.no_grad():
            _, fresh_features = self.net.predict(x)
            self.previous_features = fresh_features

    return loss.detach(), output.detach()
```

---

### 3.2 Utility Computation and Maturity Testing

**Algorithm (Section 4.1-4.2):**
- **for** $i = 1$ **to** $N_\ell$ **do**
  - $a_{\ell,i} \leftarrow a_{\ell,i} + 1$  (increment age)
  - $f_{\ell,i} \leftarrow \eta f_{\ell,i} + (1-\eta)\, h_{\ell,i,t}$  (activation EMA)
  - $u_{\ell,i} \leftarrow \eta u_{\ell,i} + (1-\eta)\, y_{\ell,i}$  (utility EMA)
- $E_\ell \leftarrow \{\, i \mid a_{\ell,i} \ge M \,\}$  (mature set)
- $S_\ell \leftarrow$ indices of $r_\ell$ smallest $\hat{u}_{\ell,i}$  (replacement set)

**Implementation:**

These steps are inherited from the base classes `GnT_for_FC` and `ConvGnT_for_ConvNet` in `src/algos/gnt.py`. The RR-CBP2 classes reuse this logic completely.

| Step | Class | Method |
|------|-------|--------|
| Age increment | `GnT_for_FC` | `test_features()` → `self.ages[i] += 1` |
| Activation EMA (+ bias correction) | `GnT_for_FC` | `update_utility()` → EMA of `features.mean(dim=0)` and `bias_correction = 1 - decay_rate**age` |
| Utility EMA (+ bias correction) | `GnT_for_FC` | `update_utility()` → EMA of `new_util`, then `bias_corrected_util = util / bias_correction` |
| Maturity check | `GnT_for_FC` | `test_features()` → `eligible = where(self.ages[i] > self.maturity_threshold)` |
| Select lowest utility | `GnT_for_FC` | `test_features()` → `topk(-bias_corrected_util[eligible], k)` (equivalently “k smallest”) |

**Key code from `src/algos/gnt.py` (base class):**
```python
def update_utility(self, layer_idx=0, features=None, next_features=None):
    with torch.no_grad():
        self.util[layer_idx] *= self.decay_rate
        # Adam-style bias correction
        bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

        # Activation EMA (with bias correction via bias_correction)
        self.mean_feature_act[layer_idx] *= self.decay_rate
        self.mean_feature_act[layer_idx] -= - (1 - self.decay_rate) * features.mean(dim=0)
        bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

        # Resolve current/next modules (map-based or legacy)
        if self.use_map:
            map_item = self.plasticity_map[layer_idx]
            current_layer = map_item['weight_module']
            next_layer = map_item['outgoing_module']
        else:
            current_layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]

        # Output weight magnitude (importance to next layer)
        output_weight_mag = next_layer.weight.data.abs().mean(dim=0)

        if self.util_type == 'contribution':
            new_util = output_weight_mag * features.abs().mean(dim=0)
        # ... other util_type branches omitted ...

        self.util[layer_idx] += (1 - self.decay_rate) * new_util
        self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction
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
        if self.basis.numel() == 0:
            return torch.zeros_like(vec)
        if self.geometry.diag_only:
            sigma_vec = self.geometry.sigma * vec
        else:
            sigma_vec = self.geometry.sigma @ vec
        coeff = self.basis.t() @ sigma_vec       # V^T Σ v
        proj_coeff = self._G_inv @ coeff         # (V^T Σ V)^{-1} V^T Σ v
        return self.basis @ proj_coeff           # V (...)
```

---

### 3.5 Direction Sampling (RR-CBP2)

**Algorithm (Section 5.1 and 5.4 pseudocode):**
- **(2) Draw direction and project into $\Sigma$-orthogonal complement**
  - $u \leftarrow \mathrm{GaussianSample}(d_\ell)$
  - $\hat{w} \leftarrow (I - P_\Sigma)\, u$
- **if** $\|\hat{w}\|_\Sigma > 0$ **then**
  - $w_{\mathrm{dir}} \leftarrow \hat{w} / \|\hat{w}\|_\Sigma$
- **else**
  - $w_{\mathrm{dir}} \leftarrow \mathrm{LeastCoveredDirection}(W_{\mathrm{keep}}, \Sigma_\ell)$

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

- **(1) Compute layer targets**
  - $v_{\mathrm{tar}} \leftarrow (1 / d_\ell)\, \mathrm{trace}(\Sigma_\ell)$
  - $q_{\mathrm{tar}} \leftarrow v_{\mathrm{tar}} / \chi_0$
  - $Q_{\mathrm{tar}} \leftarrow N_\ell\, q_{\mathrm{tar}}$
  - $Q_{\mathrm{res}} \leftarrow \max(Q_{\mathrm{tar}} - Q_{\mathrm{used}}, 0)$
- **if** $Q_{\mathrm{res}} > 0$ **then** (underbudget)
  - $q_{\mathrm{alloc}} \leftarrow \min(q_{\mathrm{tar}}, Q_{\mathrm{res}} / r_\ell)$
- **else** (overbudget / saturated)
  - $\lambda_{\min,\Sigma} \leftarrow \mathrm{smallest\_eigenvalue}(\Sigma_\ell)$
  - $q_{\min} \leftarrow \tau\, \lambda_{\min,\Sigma}$  (rank-restoring floor)
  - $\lambda_* \leftarrow$ conditioning target (optional)
  - $q_{\mathrm{alloc}} \leftarrow \min\!\bigl(q_{\mathrm{tar}}, \max(q_{\min}, \lambda_*)\bigr)$
- **(3) Scale direction**
  - $w_{\mathrm{dir}} \leftarrow w_{\mathrm{dir}} \cdot \sqrt{q_{\mathrm{alloc}}} \,/\, \|w_{\mathrm{dir}}\|_\Sigma$

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
    v_target = geometry.trace / max(geometry.dim, 1)
    
    # $q_{\text{tar}} = v_{\text{tar}} / \chi_0(\phi)$
    q_target = v_target / max(chi0, config.proj_eps)
    
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
    def __init__(
        self,
        q_target: float,
        layer_size: int,
        used_energy: float,
        tau: float,
        lambda_min_sigma: float,
        lambda_star: Optional[float],
        replacements: int,
    ) -> None:
        self.q_target = q_target
        self.layer_size = layer_size
        self.total_target = q_target * layer_size
        self.used_energy = used_energy
        self.remaining = replacements
        self.lambda_min_sigma = lambda_min_sigma
        self.lambda_star = lambda_star
        self.q_min = tau * lambda_min_sigma
        self.residual = max(0.0, self.total_target - self.used_energy)

    def allocate(self) -> Tuple[float, bool]:
        saturated = self.residual <= 0.0
        if self.remaining <= 0:
            q_alloc = max(self.q_min, self.lambda_star or 0.0)
            return q_alloc, True
        if not saturated:
            q_alloc = min(self.q_target, self.residual / self.remaining)
            self.residual = max(0.0, self.residual - q_alloc)
        else:
            floor = self.q_min
            if self.lambda_star is not None:
                floor = max(floor, self.lambda_star)
            q_alloc = min(self.q_target, floor)
        self.remaining -= 1
        self.used_energy += q_alloc
        return q_alloc, saturated
```

**Code for scaling (`_scale_to_energy()`):**
```python
def _scale_to_energy(self, direction, geometry, q_alloc):
    """Scale direction to achieve ||w||²_Σ = q_alloc"""
    norm = geometry.norm(direction)
    if norm < self.config.proj_eps or q_alloc <= 0:
        return direction
    scale = math.sqrt(q_alloc) / norm
    return direction * scale
```

---

### 3.7 Bias Transfer and Outgoing Weight Reset

**Algorithm (Section 4.2 and 5.3):**
- **(1) Bias transfer**
  - $\mathrm{TransferBiasFromUnit}(\ell, i, \hat{f}_{\ell,i}, \mathrm{outgoing\_weights})$
- **(4) Bias centering**
  - $a \leftarrow w_{\mathrm{dir}}^T H_{\mathrm{prev}}$
  - $b \leftarrow -\mathrm{mean}(a)$
  - $\mathrm{SetBias}(\ell, i, b)$
- **(5) Zero outgoing weights**
  - $\mathrm{ZeroOutgoingWeights}(\ell, i)$

**Implementation note:** In `RR_GnT2_for_FC._replace_units()`, bias transfer is **conditional**:
- It is skipped when the removed unit has age 0.
- It is also skipped when the model's plasticity map indicates the outgoing module feeds into a normalization layer (`outgoing_feeds_into_norm=True`), since bias compensation would be canceled/undesirable.

**Implementation:**

| Step | Class | Method |
|------|-------|--------|
| Bias transfer | `RR_GnT2_for_FC` | `_transfer_bias()` |
| Bias centering | `RR_GnT2_for_FC` | `_center_bias()` |
| Zero outgoing | `RR_GnT2_for_FC` | `_zero_and_seed_outgoing()` |

**Code from `rr_gnt2_fc.py`:**
```python
def _transfer_bias(self, next_layer: Module, unit_idx: int, bias_corrected_act: Tensor) -> None:
    """Transfer bias from removed unit to next layer (Section 4.2)."""
    contribution = next_layer.weight.data[:, unit_idx] * bias_corrected_act
    next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

def _center_bias(self, activations: Tensor) -> Tensor:
    """Compute bias to center preactivations (Section 5.3)."""
    if activations.numel() == 0:
        return torch.zeros(1, device=activations.device, dtype=activations.dtype).squeeze(0)
    if self.config.center_bias == "median":
        return -torch.median(activations)
    return -activations.mean()

def _zero_and_seed_outgoing(self, next_layer: Module, unit_idx: int) -> None:
    """Zero outgoing weights and optionally micro-seed (Section 5.3)."""
    epsilon = self.config.epsilon_micro_seed
    use_seed = self.config.use_micro_seed and epsilon > 0.0
    next_layer.weight.data[:, unit_idx] = 0.0
    if use_seed:
        noise = torch.randn_like(next_layer.weight.data[:, unit_idx])
        noise = noise - noise.mean()
        norm = torch.clamp(noise.norm(), min=self.config.proj_eps)
        next_layer.weight.data[:, unit_idx] = epsilon * (noise / norm)
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

### 4.5 Numerical Stability for Eigendecomposition

The implementation includes multiple layers of safeguards for eigendecomposition in ill-conditioned cases:

**Problem:** When energy budget is exhausted and λ_min(Σ) → 0, weights receive near-zero Σ-energy, creating degenerate gram matrices that fail standard eigendecomposition.

**Solutions implemented in `lambda_min_whitened()` (sigma_geometry.py:186-223):**

1. **Symmetrization:** Force gram matrix symmetry to avoid numerical artifacts
   ```python
   gram = 0.5 * (gram + gram.t())
   ```

2. **Adaptive Regularization:** Add small regularization proportional to matrix trace
   ```python
   base_reg = max(eps * 100.0, trace / n * 1e-6)
   gram = gram + base_reg * eye
   ```

3. **Double Precision with Fallback:** Use float64 for eigendecomposition with stronger regularization on failure
   ```python
   try:
       gram_cpu = gram.detach().cpu().double()
       eigvals = torch.linalg.eigvalsh(gram_cpu)
   except RuntimeError:
       # Add 100x stronger regularization and retry
       gram_cpu = gram_cpu + (base_reg * 100.0) * eye
       eigvals = torch.linalg.eigvalsh(gram_cpu)
   ```

4. **CPU Path Workaround:** Force CPU eigendecomposition when `enable_cuda1_workarounds=True`
   ```python
   force_cpu_eigh = os.environ.get('SIGMA_FORCE_CPU_EIGH', '0') == '1'
   if force_cpu_eigh and gram.device.type == 'cuda':
       gram_cpu = gram.detach().cpu().double()
   ```

**Why This Matters:**
- Saturated regime (Q_res ≤ 0) is **algorithmically valid** per Section 6.3
- New units receive q_alloc = τ · λ_min(Σ)
- When λ_min(Σ) ≈ 0, weights have tiny Σ-energy
- Gram matrix W Σ W^T ≈ 0 → requires high precision + regularization
- These safeguards allow monitoring even in extreme saturation

**Note:** The `lambda_min_whitened()` computation is **diagnostic only** - not required by core RR-CBP-E2 algorithm. It monitors weight conditioning after replacement.

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
