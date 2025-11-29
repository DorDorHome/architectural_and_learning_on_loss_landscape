# RR-CBP2 and RR-CBP-E2 Implementation Plan

## Overview

This document outlines the implementation plan for **RR-CBP2** (Rank-Restoring Continual Backprop version 2) and **RR-CBP-E2** (RR-CBP with Σ-aware energy control version 2), based on the algorithm guide in `RR_CBP_2_algorithm_guide.md`.

### Key Differences from Existing RR-CBP

The existing `rr_cbp` implementation has suspected implementation errors. The new version (`rr_cbp2`, `rr_cbp_e_2`) will:

1. **Keep existing files untouched** - `rr_cbp_fc.py`, `rr_cbp_conv.py`, `rr_gnt_fc.py`, `rr_gnt_conv.py` remain unchanged
2. **Create fresh implementations** following the algorithm guide precisely
3. **Reuse working components** - `sigma_geometry.py`, `rr_covariance.py`, base `GnT_for_FC` and `ConvGnT_for_ConvNet` classes

---

## Files to Create

### 1. Configuration Class
**File:** `configs/configurations.py` (ADD new config class - do NOT modify existing)

```python
@dataclass
class RRCBP2Config(ContinuousBackpropConfig):
    """Configuration for RR-CBP2 and RR-CBP-E2 algorithms."""
    type: str = 'rr_cbp2'
    
    # Core RR-CBP2 settings
    rrcbp_enabled: bool = True
    
    # Covariance EMA settings
    sigma_ema_beta: float = 0.99          # EMA decay for covariance tracking
    sigma_ridge: float = 1e-4              # Ridge regularization for Σ
    
    # Projection settings
    max_proj_trials: int = 4               # Max attempts to find orthogonal direction
    proj_eps: float = 1e-8                 # Numerical floor for projection norms
    projector_reg_epsilon: float = 1e-6    # Regularization for Gram inverse
    
    # Bias centering
    center_bias: str = 'mean'              # 'mean' or 'median'
    
    # Σ-geometry settings
    diag_sigma_only: bool = False          # Use diagonal Σ approximation
    sigma_eig_floor: float = 1e-6          # Eigenvalue floor for Σ
    covariance_dtype: Optional[str] = None # Override dtype for covariance
    
    # Orthonormalization
    orthonormalize_batch: bool = True      # Orthonormalize new vectors within batch
    
    # Logging
    log_rank_metrics_every: int = 0        # Log rank metrics every N replacements (0 = disabled)
    
    # ===== RR-CBP-E2 (Energy-Aware) Settings =====
    # Set use_energy_budget=True to switch from rr_cbp2 to rr_cbp_e_2 behavior
    use_energy_budget: bool = False
    
    # Energy budget parameters (Section 6 of algorithm guide)
    tau: float = 1e-2                      # Scale factor for q_min = τ * λ_min(Σ)
    use_lambda_star: bool = False          # Use λ* conditioning target
    lambda_star: Optional[float] = None    # Override λ* (if None, computed from whitened Gram)
    
    # Chi-squared constant estimation
    chi0_override: Optional[float] = None  # Override χ₀(φ) constant
    estimate_chi0_from_batch: bool = False # Estimate χ₀ from activations
    
    # Micro-seeding outgoing weights (optional)
    use_micro_seed: bool = False           # Seed outgoing weights with small noise
    epsilon_micro_seed: float = 1e-4       # Magnitude for micro-seeding
    
    # Saturated regime handling
    improve_conditioning_if_saturated: bool = True
    nullspace_seed_epsilon: float = 0.0    # Add noise in nullspace direction
```

### 2. New GnT Classes for FC Networks
**File:** `src/algos/supervised/rr_gnt2_fc.py`

This file contains `RR_GnT2_for_FC` which:
- Inherits from `GnT_for_FC` (reuses age tracking, utility computation, test_features)
- Overrides `gen_new_features` to implement Σ-orthogonal weight initialization
- Overrides `gen_and_test` to pass additional context (H_prev, batch_input)

**Key Methods:**
```python
class RR_GnT2_for_FC(GnT_for_FC):
    def __init__(self, net, hidden_activation, opt, config, loss_func, device):
        # Initialize base GnT (inherits ages, utility tracking, test_features)
        # Initialize covariance state per layer
        # Store config for RR-CBP2 parameters
    
    def gen_and_test(self, features, batch_input):
        """Override: includes batch_input for covariance computation."""
        features_to_replace, num_features_to_replace = self.test_features(features)
        if all(count == 0 for count in num_features_to_replace):
            return
        
        for layer_idx, count in enumerate(num_features_to_replace):
            if count == 0:
                continue
            
            # Compute upstream features H_prev
            H_prev = self._compute_layer_inputs(layer_idx, features, batch_input)
            
            # Update covariance EMA
            sigma = self._update_covariance(layer_idx, H_prev)
            
            # Get kept indices
            keep_idx = self._kept_indices(layer_idx, features_to_replace[layer_idx])
            
            # Replace units with Σ-orthogonal directions
            self._replace_units(
                layer_idx=layer_idx,
                keep_idx=keep_idx,
                replace_idx=features_to_replace[layer_idx],
                sigma=sigma,
                H_prev=H_prev,
            )
            
            # Reset optimizer state and statistics
            self._post_replacement_housekeeping(layer_idx, features_to_replace[layer_idx])
    
    def _replace_units(self, layer_idx, keep_idx, replace_idx, sigma, H_prev):
        """Core replacement logic following Algorithm Section 5 (RR-CBP) or Section 6 (RR-CBP-E)."""
        layer = self.net[layer_idx * 2]
        next_layer = self.net[layer_idx * 2 + 2]
        
        # Build SigmaGeometry and SigmaProjector
        geometry = SigmaGeometry(sigma, self.config.diag_sigma_only, self.config.sigma_eig_floor)
        kept_vectors = layer.weight.data[keep_idx, :].t()  # d × k
        projector = SigmaProjector(geometry, kept_vectors, self.config.projector_reg_epsilon)
        
        # Energy allocation (for RR-CBP-E2)
        if self.config.use_energy_budget:
            allocator = self._create_energy_allocator(layer_idx, geometry, kept_vectors)
        
        for unit_idx in replace_idx.tolist():
            # (1) Bias transfer from old unit
            if self.ages[layer_idx][unit_idx] > 0:
                self._transfer_bias(next_layer, unit_idx)
            
            # (2) Sample Σ-orthogonal direction
            w_dir, used_fallback = self._sample_sigma_direction(projector, geometry)
            
            # (3) Scale direction
            if self.config.use_energy_budget:
                q_alloc, saturated = allocator.allocate()
                w_scaled = self._scale_to_energy(w_dir, geometry, q_alloc)
            else:
                # RR-CBP2: use unit Σ-norm
                w_scaled = w_dir  # Already normalized to ||w||_Σ = 1
            
            # (4) Update projector with new vector
            projector.add_vector(w_scaled)
            
            # (5) Assign incoming weights
            layer.weight.data[unit_idx, :] = w_scaled
            
            # (6) Bias centering
            acts = (w_scaled.unsqueeze(0) @ H_prev).squeeze(0)
            layer.bias.data[unit_idx] = -acts.mean()  # or median
            
            # (7) Zero outgoing weights
            next_layer.weight.data[:, unit_idx] = 0.0
            
            # (8) Reset statistics
            self.ages[layer_idx][unit_idx] = 0
            self.util[layer_idx][unit_idx] = 0
            self.mean_feature_act[layer_idx][unit_idx] = 0
    
    def _sample_sigma_direction(self, projector, geometry):
        """Sample direction in Σ-orthogonal complement of kept vectors."""
        for _ in range(self.config.max_proj_trials):
            u = torch.randn(geometry.dim, device=geometry.sigma.device, dtype=geometry.sigma.dtype)
            w_hat = projector.project_complement(u)
            norm = geometry.norm(w_hat)
            if norm > self.config.proj_eps:
                return w_hat / norm, False
        
        # Fallback: least-covered direction
        return projector.least_covered_direction(dtype=geometry.sigma.dtype), True
    
    def _scale_to_energy(self, w_dir, geometry, q_alloc):
        """Scale direction to achieve ||w||_Σ² = q_alloc."""
        norm = geometry.norm(w_dir)
        if norm < self.config.proj_eps:
            return w_dir
        scale = math.sqrt(q_alloc) / norm
        return w_dir * scale
```

### 3. New GnT Classes for ConvNets
**File:** `src/algos/supervised/rr_gnt2_conv.py`

Similar structure to FC but handles:
- Conv2d layers with patch unfolding for H_prev
- Proper weight flattening and reshaping
- Conv-to-Linear transitions

### 4. New Learner Classes
**Files:** 
- `src/algos/supervised/rr_cbp2_fc.py`
- `src/algos/supervised/rr_cbp2_conv.py`

These follow the same pattern as existing `rr_cbp_fc.py` and `rr_cbp_conv.py`:

```python
class RankRestoringCBP2_for_FC(Learner):
    def __init__(self, net, config, netconfig):
        super().__init__(net, config, netparams)
        
        # Initialize AdamGnT optimizer
        self.opt = AdamGnT(...)
        
        # Initialize RR_GnT2_for_FC
        self.rr_gnt = RR_GnT2_for_FC(
            net=self.net.layers,
            hidden_activation=hidden_activation,
            opt=self.opt,
            config=config,
            loss_func=self.loss_func,
            device=self.device,
        )
    
    def learn(self, x, target):
        # Forward pass
        output, features = self.net.predict(x)
        loss = self.loss_func(output, target)
        
        # Backward pass
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        # Generate-and-test with Σ-orthogonal replacement
        self.opt.zero_grad()
        self.rr_gnt.gen_and_test(features=features, batch_input=x)
        
        return loss.detach(), output.detach()
```

### 5. Factory Update
**File:** `src/algos/supervised/supervised_factory.py` (APPEND new cases)

```python
# Add imports
from src.algos.supervised.rr_cbp2_fc import RankRestoringCBP2_for_FC
from src.algos.supervised.rr_cbp2_conv import RankRestoringCBP2_for_ConvNet
from configs.configurations import RRCBP2Config

# Add in create_learner function:
elif normalized_type == 'rr_cbp2':
    if not isinstance(config, RRCBP2Config):
        if isinstance(config, DictConfig):
            config = RRCBP2Config(**OmegaConf.to_container(config, resolve=True))
        elif isinstance(config, dict):
            config = RRCBP2Config(**config)
        else:
            raise TypeError("RR-CBP2 requires RRCBP2Config-compatible config")
    if net_cls == 'fc':
        return RankRestoringCBP2_for_FC(net, config, netconfig)
    if net_cls == 'conv':
        return RankRestoringCBP2_for_ConvNet(net, config, netconfig)
    raise ValueError(f"Unsupported network_class '{net_cls}' for RR-CBP2")

elif normalized_type == 'rr_cbp_e_2':
    if not isinstance(config, RRCBP2Config):
        config = RRCBP2Config(**OmegaConf.to_container(config, resolve=True))
    config.use_energy_budget = True  # Enable energy-aware mode
    if net_cls == 'fc':
        return RankRestoringCBP2_for_FC(net, config, netconfig)
    if net_cls == 'conv':
        return RankRestoringCBP2_for_ConvNet(net, config, netconfig)
    raise ValueError(f"Unsupported network_class '{net_cls}' for RR-CBP-E2")
```

---

## Reusable Components (DO NOT MODIFY)

The following files are **reused as-is** and should NOT be modified:

1. **`src/algos/supervised/rank_restoring/sigma_geometry.py`**
   - `SigmaGeometry` - Σ-inner products, norms, whitening
   - `SigmaProjector` - Σ-orthogonal projection
   - `EnergyAllocator` - Energy budget allocation
   - `chi0_for_activation` - χ₀(φ) constants

2. **`src/algos/supervised/rank_restoring/rr_covariance.py`**
   - `CovarianceState` - EMA covariance tracking
   - `initialize_covariance` - Factory function

3. **`src/algos/gnt.py`**
   - `GnT_for_FC` - Base class with age/utility tracking and `test_features`
   - `ConvGnT_for_ConvNet` - Base class for ConvNets

4. **`src/algos/AdamGnT.py`**
   - Modified Adam optimizer for GnT

---

## Implementation Order

1. **Phase 1: Configuration**
   - Add `RRCBP2Config` class to `configurations.py`

2. **Phase 2: FC Implementation**
   - Create `rr_gnt2_fc.py` with `RR_GnT2_for_FC`
   - Create `rr_cbp2_fc.py` with `RankRestoringCBP2_for_FC`

3. **Phase 3: ConvNet Implementation**
   - Create `rr_gnt2_conv.py` with `RR_GnT2_for_ConvNet`
   - Create `rr_cbp2_conv.py` with `RankRestoringCBP2_for_ConvNet`

4. **Phase 4: Factory Integration**
   - Update `supervised_factory.py` with `rr_cbp2` and `rr_cbp_e_2` types

5. **Phase 5: Testing**
   - Create unit tests in `tests/test_rr_cbp2.py`
   - Validate against algorithm guide equations

---

## Key Algorithm Equations (Reference)

### RR-CBP2 Core (Section 5)

1. **Σ-Orthogonal Projection:**
   $$P_Σ = V (V^T Σ V)^{-1} V^T Σ$$

2. **Direction Sampling:**
   $$\hat{w} = (I - P_Σ) u, \quad u \sim \mathcal{N}(0, I)$$
   $$w_{dir} = \hat{w} / ||\hat{w}||_Σ$$

3. **Least-Covered Direction (fallback):**
   $$M' = Σ^{1/2} V V^T Σ^{1/2}$$
   $$w_{dir} = Σ^{-1/2} u_{min} / ||Σ^{-1/2} u_{min}||_Σ$$

### RR-CBP-E2 Energy Control (Section 6)

4. **Target Variance:**
   $$v_{tar} = \frac{1}{d} tr(Σ)$$
   $$q_{tar} = v_{tar} / χ_0(φ)$$

5. **Energy Budget:**
   $$Q_{tar} = N · q_{tar}$$
   $$Q_{used} = tr(W_{keep} Σ W_{keep}^T)$$
   $$Q_{res} = max(Q_{tar} - Q_{used}, 0)$$

6. **Allocation:**
   - Underbudget: $$q_{alloc} = min(q_{tar}, Q_{res}/r)$$
   - Overbudget: $$q_{alloc} = min(q_{tar}, max(q_{min}, λ_*))$$
   - Floor: $$q_{min} = τ · λ_{min}(Σ)$$

7. **Scaling:**
   $$w ← w · \sqrt{q_{alloc}} / ||w||_Σ$$

---

## Validation Checklist

- [ ] Ages increment correctly each step
- [ ] Utility EMA matches algorithm guide formula
- [ ] Bias-corrected utility used for selection
- [ ] Maturity threshold respected
- [ ] Covariance EMA updates correctly
- [ ] Σ-orthogonal projection computed correctly
- [ ] Least-covered direction fallback works
- [ ] Bias centering uses mean/median
- [ ] Outgoing weights zeroed
- [ ] Optimizer state reset for replaced units
- [ ] Energy budget computed correctly (RR-CBP-E2)
- [ ] Scaling matches target q_alloc (RR-CBP-E2)

---

## Usage Example

```yaml
# Config for RR-CBP2
learner:
  type: rr_cbp2
  step_size: 0.001
  neurons_replacement_rate: 0.001
  maturity_threshold: 100
  decay_rate_utility_track: 0.99
  sigma_ema_beta: 0.99
  sigma_ridge: 1e-4

# Config for RR-CBP-E2 (energy-aware)
learner:
  type: rr_cbp_e_2
  step_size: 0.001
  neurons_replacement_rate: 0.001
  maturity_threshold: 100
  tau: 0.01
  use_lambda_star: true
```
