# Continuous (Continual) Backprop with Generate‑and‑Test (GnT)

## 1. High‑Level Overview
Continuous / Continual Backprop extends standard training by periodically replacing low‑utility hidden units (neurons or conv filters) with newly initialized ones while preserving functional stability.  
Core cycle per training batch:
1. Forward + loss
2. Backward + optimizer step
3. Structural adaptation (Generate‑and‑Test):
   - Update per-feature utility statistics (EMA + bias correction)
   - Select mature, low‑utility features
   - Re-initialize their incoming weights (and zero outgoing weights)
   - Optionally compensate downstream bias (FC variant)
   - Reset optimizer state for replaced parameter slices

Two learner front-ends:
- ContinualBackprop_for_FC (Fully Connected)
- ContinuousBackprop_for_ConvNet (Conv + Linear hybrids)

Two GnT back-ends:
- GnT_for_FC (neuron-level replacement)
- ConvGnT_for_ConvNet (filter-level + flattened mapping handling)

Custom optimizer:
- AdamGnT (Adam variant with per‑element step tensor enabling selective state reset)

---

## 2. File Roles
| File | Role |
|------|------|
| continuous_backprop_with_GnT.py | Orchestrates gradient step + GnT call |
| gnt.py | Implements GnT_for_FC and ConvGnT_for_ConvNet |
| AdamGnT.py | Optimizer with per-parameter-element state needed for partial reset |

---

## 3. ContinualBackprop_for_FC (Learner Subclass)

### 3.1 Responsibilities
- Validate model type (expects attribute net.type == 'FC')
- Initialize optimizer (AdamGnT if configured)
- Extract hidden activation type for initialization bounds
- Own GnT_for_FC instance
- Provide learn(x, target) method combining gradient descent + structural step

### 3.2 Training Flow (learn)
```
forward -> loss -> zero_grad -> backward -> step -> zero_grad -> gnt.gen_and_test
```
Second zero_grad is maintained to ensure a clean gradient state before structural mutation (some GnT variants may inspect/assume cleared gradients).

### 3.3 Key Attributes
- neurons_replacement_rate: expected fraction of mature units replaced per call
- decay_rate_utility_track: EMA decay (β)
- maturity_threshold: min age before unit eligible
- util_type: metric variant (weight, contribution, etc.)
- init: initialization scheme ('kaiming', 'xavier', 'lecun', 'default')
- accumulate: deterministic fractional accumulation vs stochastic rounding
- outgoing_random (currently unused, candidate extension flag)
- gnt: GnT_for_FC instance

---

## 4. ContinuousBackprop_for_ConvNet

### 4.1 Differences from FC Variant
- Must infer num_last_filter_outputs = (#flattened positions per conv filter) for conv→linear transition
- Uses ConvGnT_for_ConvNet (filter indexing + output index expansion)
- Similar training loop; current code misses an initial zero_grad before backward (inconsistency with FC)

### 4.2 Structural Assumptions
- Alternating pattern: Conv/Activation or Linear/Activation
- Last conv followed by linear classifier
- _calculate_last_filter_outputs divides first_linear.in_features by last_conv.out_channels (assumes full flatten; fragile if extra layers inserted)

---

## 5. GnT_for_FC (Generate‑and‑Test Core)

### 5.1 Data Structures (per hidden layer i)
- util[i]: EMA accumulator of raw utility
- bias_corrected_util[i]: util[i] / (1 - decay_rate ** age_i)
- ages[i]: step counters since (re)generation
- mean_feature_act[i]: EMA of mean activation (centering baseline)
- accumulated_num_features_to_replace[i]: fractional remainder (if accumulate=True)
- bounds[i]: uniform init bound per layer (activation-aware)
- opt.state[...] slices: exp_avg, exp_avg_sq, step (reset per replaced unit if AdamGnT)

### 5.2 Utility Update (update_utility)
Formula pattern:
- util_i ← decay * util_i + (1 - decay) * new_util
- bias_correction_i = 1 - decay ** age_vector
- bias_corrected_util_i = util_i / bias_correction_i

new_util depends on util_type:
| util_type | Formula (per neuron j) | Intuition |
|-----------|------------------------|-----------|
| weight | mean |W_out(:, j)| | Outgoing connectivity strength |
| contribution | mean |W_out(:, j)| * mean |a_j| | Magnitude-weighted activity |
| adaptation | 1 / mean |W_in(j, :)| | Prefer under-adapted (small input weights) |
| zero_contribution | mean |W_out(:, j)| * mean |a_j - μ_j| | Activity novelty |
| adaptable_contribution | zero_contribution / mean |W_in(j,:)| | Hybrid exploration |
| feature_by_input | mean |a_j - μ_j| / mean |W_in(j,:)| | Activity variability normalized |
| random | Uniform[0,1] | Baseline exploration |

Bias-corrected mean activations used when centering is required.

### 5.3 Feature Selection (test_features)
Steps per layer:
1. ages[i] += 1
2. update_utility(i, features[i])
3. eligible = { j | ages[i][j] > maturity_threshold }
4. expected = replacement_rate * |eligible|
5. Convert to integer:
   - accumulate=True:
     - accumulator += expected
     - k = floor(accumulator); accumulator -= k
   - else:
     - if expected < 1: k = Bernoulli(expected)
     - else: k = floor(expected)
6. Rank k lowest bias_corrected_util via topk on negative tensor
7. Zero util + mean_feature_act for chosen indices
8. Return index tensors + counts

### 5.4 Generation (gen_new_features)
For each replaced neuron j:
- Zero and reinitialize incoming weights uniformly in [-bound_i, bound_i]
- Zero its bias
- Bias compensation (unique to FC):
  downstream_bias += Σ_k ( W_out[k, j] * mean_act_j / (1 - decay ** age_j) )
- Zero outgoing weights W_out[:, j]
- Reset age to 0

Compensation attempts to preserve aggregate output by adding back expected removed contribution.

### 5.5 Optimizer State Reset (update_optim_params)
Only if opt_type == 'adam':
- Zero exp_avg, exp_avg_sq, step entries for:
  - Reinitialized input row and bias elements
  - Corresponding outgoing column weights
Purpose: prevent legacy momentum / variance statistics from biasing new random weights.

AdamGnT uses per-element step tensor (state['step'] same shape as parameter) enabling selective reset.

---

## 6. ConvGnT_for_ConvNet Differences

| Aspect | FC | Conv |
|--------|----|------|
| Unit definition | Neuron (Linear row) | Filter (Conv out channel) or Linear neuron |
| Output index mapping | 1:1 | conv→linear expands each filter into num_last_filter_outputs flattened positions |
| Utility centering | 1D means | Spatial reduction (mean over H,W) then optional reshape |
| Bias compensation | Present | Not implemented (outgoing bias unchanged) |
| Expected replacements | Computed each call | Precomputed per layer (num_new_features_to_replace) |
| Optim state reset | Resets in/out rows | Similar but (currently) omits exp_avg reset for some conv weights (review needed) |

---

## 7. AdamGnT Specifics

### 7.1 State Layout
Each parameter p has:
- step: tensor same shape as p (unconventional; standard Adam stores scalar step)
- exp_avg
- exp_avg_sq
- (optional) max_exp_avg_sq (if amsgrad)

### 7.2 Update Differences from Standard Adam
Sequence:
- state['step'] += 1 (elementwise)
- exp_avg, exp_avg_sq updated normally
- bias_correction1, bias_correction (beta2 branch) computed
- bias_correction.sqrt().div_(bias_correction1) mutates tensor
- denom = sqrt(second moment) + eps; then denom.div_(exp_avg)
- p.data.addcdiv_(bias_correction, denom, value=-lr)
This algebra differs from canonical Adam expression; it effectively applies:
p ← p - lr * (bias_correction * exp_avg / denom_raw)
(denom was divided by exp_avg, so addcdiv uses (bias_correction / (denom_raw/exp_avg)) = bias_correction * exp_avg / denom_raw)

Behavior: stable but not identical to published Adam; acceptable if empirically validated.

### 7.3 Justification for Per-Element step
Allows resetting subset of steps to 0 for replaced units so their bias corrections restart.

---

## 8. Interactions & Invariants

| Invariant | Purpose |
|-----------|---------|
| len(net.layers) even for hidden pairs | Assumed indexing (layer_idx*2) |
| next_layer exists for each hidden layer | Required for outgoing weight magnitude & compensation |
| features passed as list aligned with hidden layers | Utility indexing |
| Replacement never touches output layer directly | GnT loops over hidden only |
| ages reset only after generation, not during test | Ensures bias_correction uses pre-replacement age |

---

## 9. Edge Cases & Risks

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Missing zero_grad before backward (Conv learner) | Gradient accumulation bug | Add self.opt.zero_grad() |
| type(self.gnt) is GnT_for_FC | Blocks subclassing | Use isinstance |
| _calculate_last_filter_outputs heuristic fragile | Wrong num_last_filter_outputs → misaligned index expansion | Use dynamic dummy forward |
| Mean update uses -= - | Readability | Replace with += |
| util_type else: new_util = 0 | Silent misconfiguration | Raise ValueError |
| Conv optimizer reset skips exp_avg for weights | Asymmetrical state → drift | Add reset for exp_avg conv weights |
| Bias compensation only FC | Inconsistent behavior | Document or unify |
| Outgoing_random unused | Config drift | Remove or integrate |
| Potential zero division if input_wight_mag == 0 | Inf utility | Clamp or add epsilon |
| AdamGnT per-element step non-standard | Harder to swap optimizer | Document clearly |

---

## 10. Rationale for Two GnT Classes
Differences in:
- Shape semantics (4D conv vs 2D linear activations)
- Mapping of filter → flattened linear inputs
- Utility aggregation axes (dim=(0,2,3) vs dim=0)
- Index expansion logic
Keeping separate avoids dense conditional branching and improves clarity. Refactor into a shared base only if additional architectures added or logic duplication grows (> ~70% overlap).

---

## 11. Suggested Improvements (Optional Patches)
1. Add zero_grad before backward in ContinuousBackprop_for_ConvNet.learn
2. Replace type(...) check with isinstance in FC learn
3. Explicit ValueError for unknown util_type
4. Reset exp_avg for conv weights in ConvGnT update_optim_params
5. Introduce BaseGnT protocol for typing clarity
6. Dummy forward method to compute num_last_filter_outputs robustly
7. Clarify comments around bias compensation ordering (uses pre-reset ages)

---

## 12. Example Annotated FC Learn Loop
```python
# Forward (captures features for utility update)
output, features = net.predict(x)
loss = loss_func(output, target)

# Gradient update
opt.zero_grad()
loss.backward()
opt.step()

# Structural adaptation (stateless w.r.t autograd)
opt.zero_grad()  # ensure clean grads if GnT inspects .grad
gnt.gen_and_test(features)
```

---

## 13. Utility Ranking Logic Summary
Let:
- u_t^j = utility EMA for unit j at step t
- v_t^j = instantaneous metric (new_util)
- β = decay_rate
- a_t^j = age (steps since creation)
Then:
u_t^j = β u_{t-1}^j + (1-β) v_t^j  
û_t^j = u_t^j / (1 - β^{a_t^j}) (bias-corrected)  
Ranking: pick k units with smallest û_t^j among eligible (a_t^j > maturity_threshold)

---

## 14. Bias Compensation (FC Only)
For replaced indices R in layer L:
bias_{L+1} ← bias_{L+1} + Σ_{j∈R}  W_out[:, j] * ( mean_act_j / (1 - β^{age_j}) )  
Interpretation: add back expected contribution removed by zeroing outgoing weights.

---

## 15. Why no_grad for Structural Step
- Prevents autograd graph bloat
- Ensures in-place weight resets do not retroactively affect gradient history
- Safe because subsequent forward pass rebuilds graph with new parameters
- Optimizer references same Parameter objects; changes are immediately effective

---

## 16. Validation Checklist
| Test | Expected |
|------|----------|
| Replace all utilities with large values | No replacements (low utility needed) |
| Set replacement_rate=0 | Pure gradient descent |
| Force maturity_threshold=0 | Immediate eligibility; higher churn |
| Switch util_type to 'random' | Near-uniform replacement sampling |
| After replacement, optimizer state for rows/cols zeroed | exp_avg, exp_avg_sq, step = 0 at indices |

---

## 17. Minimal Unit Test Ideas (Conceptual)
```python
# 1. Utility monotonic decay test
# 2. Replacement count expectation over N steps
# 3. Optimizer state reset verification
# 4. Bias compensation numerical effect (pre vs post layer output shift)
```

---

## 18. Known Assumptions to Document
- Layers array indexing pattern: [Linear/Conv, Activation, ..., Linear_out]
- predict returns (output, list_of_hidden_activations) aligned with GnT expectations
- No mixed precision logic (would require dtype-safe replacements)

---

## 19. Summary
The implementation cleanly separates:
- Learner: gradient-based optimization + call site
- GnT: statistical tracking + structural adaptation
- Optimizer: state supports partial resets

Design is modular and readable; refinements mainly concern consistency, edge-case handling, and improved robustness in conv shape inference and optimizer state symmetry.

---

## 20. Quick Reference (Core Hyperparameters)
| Name | Effect |
|------|--------|
| replacement_rate | Turnover speed (exploration) |
| decay_rate | Smoothing of utility signal |
| maturity_threshold | Stabilization period for new units |
| util_type | Exploration/exploitation flavor |
| accumulate | Deterministic vs stochastic rounding |
| init | Initialization scaling (activation-aware) |

---