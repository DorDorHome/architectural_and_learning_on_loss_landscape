# Alternative Architecture Outline: Utility-Based Continuous Backprop (Generate-and-Test Redesign)

## 1. Philosophy & High-Level Intent
A utility-based continual backprop system maintains model plasticity by *surgically refreshing* underperforming hidden units while preserving overall function. Instead of adding/removing entire layers or performing global architecture search, it:
- Monitors per-unit contribution ("utility") online.
- Protects immature (newly spawned) units until sufficient exposure.
- Re-initializes the lowest-utility mature units.
- Compensates downstream statistics to avoid output shocks.
- Resets optimizer state for refreshed parameters to allow fair adaptation.

Goal: Decouple structural adaptation (exploration) from gradient learning (exploitation) with minimal disruption and maximal modularity.

## 2. Core Conceptual Axes
| Axis | Question | Design Target |
|------|----------|---------------|
| Observation | How to collect activations efficiently? | Passive hooks, batched stats |
| Utility Signal | What defines unit usefulness? | Pluggable metrics (weight * act, centered variance, adaptation) |
| Maturity | When is a unit eligible for replacement? | Age > threshold (configurable) |
| Selection | Which units to refresh this step? | Ranked + expected turnover -> discrete plan |
| Replacement | How to reinitialize? | Init policy (activation-aware bounds) |
| Continuity | How to reduce sudden function drift? | Bias / output compensation layer-wise |
| Optimizer Sync | How to avoid stale momentum? | Selective moment reset via adapter |
| Scheduling | How often to adapt? | Step scheduler / probabilistic trigger |
| Extensibility | Add new layer types? | Registry + LayerAdapter abstraction |
| Observability | How to debug evolution? | Structured event logs + metrics exports |

## 3. Problems in Current Monolithic Implementation
- Tight coupling to sequential (Linear/Activation) layout.
- Two parallel GnT classes (FC / Conv) with duplicated logic.
- Manual feature list passing (net.predict) instead of generic hooks.
- Hard-coded index arithmetic for conv→linear flattening.
- Bias compensation only applied in FC path (inconsistent semantics).
- AdamGnT optimizer diverges from canonical Adam; partial state resets rely on custom per-element step.
- Utility variants baked into large conditional block (difficult to extend).
- Lack of explicit failure handling (NaNs, exploding stats, degenerate utility).

## 4. Design Goals
1. Separation of concerns (each responsibility isolated).
2. Layer-agnostic core engine; layer-specific logic via adapters.
3. Policy-driven utility / selection / replacement.
4. Non-invasive integration: wrap any nn.Module without modifying forward signature.
5. Stable & testable bias-compensation semantics.
6. Optional soft / staged replacement (future extension) without redesign.
7. Optimizer-agnostic (fall back gracefully if state reset unsupported).
8. Rich instrumentation and reproducibility.

## 5. Component Overview
| Component | Responsibility | Interface Sketch |
|-----------|----------------|------------------|
| ActivationCollector | Capture per-layer activations via forward hooks | register(module) -> handle; get(layer_id) |
| LayerRegistry | Enumerate & describe units (axis, parameter refs, downstream links) | register(layer, adapter); iter_hidden() |
| UtilityPolicy | Compute instantaneous per-unit metric | compute(layer_id, activations, params, stats) -> tensor |
| StatStore | Maintain EMA stats (util, bias-correction denom, ages, mean activations) | update(layer_id, batch) |
| MaturityFilter | Mask ineligible units | eligible(layer_id, ages) -> index tensor |
| SelectionPolicy | Convert expected turnover to concrete indices | select(util_vec, eligible, expected) |
| TurnoverScheduler | Decide expected fraction for this step | expected(step, layer_id) -> float |
| ReplacementPolicy | Reinitialize selected units | reinit(layer_id, indices, registry, bounds) |
| CompensationPolicy | Compute downstream adjustment (bias/output) | compensate(layer_id, indices, stats, registry) |
| OptimizerAdapter | Reset optimizer state selectively | reset(layer_id, indices, param_refs) |
| BoundsCalculator | Provide init bounds per layer/activation | bound(layer_id) -> float |
| EventLogger | Persist adaptation events | log(event_dict) |
| Engine | Orchestrate end-to-end cycle | step(step_idx) |

## 6. Data Flow (Per Training Step)
1. Forward: hooks record activations per registered layer.
2. After optimizer.step(): Engine invoked.
3. For each layer: update EMA stats (util, mean act, ages).
4. Compute bias-corrected utility.
5. Determine eligibility (age > threshold).
6. TurnoverScheduler returns expected_rate; SelectionPolicy yields indices.
7. ReplacementPolicy reinitializes weights.
8. CompensationPolicy adjusts downstream bias (if configured).
9. OptimizerAdapter resets relevant moments.
10. StatStore resets util/mean/ages for replaced units.
11. EventLogger records action.

## 7. Detailed Component Specifications
### 7.1 StatStore
State per layer:
- util (EMA accumulator)
- bias_denom (1 - decay^age) implicit via age
- mean_act (EMA of mean activation)
- age (tensor)
- (optional) running_abs_act
API:
- update(layer_id, activations) -> updates all stats
- reset_units(layer_id, indices)

### 7.2 UtilityPolicy Examples
Let β = decay_rate.
Generic EMA update:
```
util = β * util + (1-β) * new_util
bias_corrected = util / (1 - β ** age)
```
Policies:
- Contribution: new_util = mean_out_weight * mean(|a|)
- CenteredContribution: mean_out_weight * mean(|a - mean_act|)
- AdaptationInverse: 1 / mean(|incoming_weights| + ε)
- Hybrid: CenteredContribution / mean(|incoming_weights| + ε)
- RandomBaseline: U(0,1)

### 7.3 SelectionPolicy
Inputs: bias_corrected_util (u), eligible indices E, expected_count λ.
Modes:
- Accumulate: accumulator += λ; k = floor(accumulator); accumulator -= k
- Stochastic: k ~ Bernoulli(λ) if λ < 1 else k = floor(λ)
- Hybrid: temperature-scaled softmax sampling from 1/u
Return: smallest-k utilities (deterministic) or sampled.

### 7.4 ReplacementPolicy
Steps:
1. Zero incoming weight rows.
2. Sample new rows from Uniform(-bound, bound).
3. Zero bias at indices.
4. (Optional soft mode): keep old row in shadow buffer for interpolation.

### 7.5 CompensationPolicy (Bias Preservation)
For FC layer L with replaced set R and downstream Linear L+1:
```
Δb_{L+1} = Σ_{j∈R} W_out[:, j] * E[a_j]
```
Where E[a_j] approximated by bias-corrected mean_act[j]. Apply only if continuity flag enabled.
Conv→Linear: reshape filter activations into flattened positions before aggregation.

### 7.6 OptimizerAdapter
Supports: Adam, SGD (momentum), RMSprop.
Interface:
```
reset(layer_id, indices):
  for each param tensor T in layer:
     if shape matches unit axis -> zero selected slices in exp_avg, exp_avg_sq, momentum, etc.
```
Fallback: if state shape incompatible, skip with warning.

### 7.7 BoundsCalculator
Activation-aware bound selection:
| init | bound formula |
|------|---------------|
| kaiming | gain * sqrt(3 / fan_in) |
| lecun | sqrt(3 / fan_in) |
| xavier | gain * sqrt(6 / (fan_in + fan_out)) |
| default | sqrt(1 / fan_in) |

### 7.8 TurnoverScheduler
Variants:
- Constant: λ = base_rate
- Warmup: λ = base_rate * sigmoid((step - s0)/τ)
- Annealed: λ = base_rate * (1 - step / max_steps)^γ
- Adaptive: scale by moving variance of utility (higher variance → lower refresh).

## 8. Pseudocode (Engine Step)
```python
def engine_step(step):
    for layer_id in registry.hidden_layers():
        acts = activations.pop(layer_id)
        stats.update(layer_id, acts)
        util_vec = utility_policy.compute(layer_id, acts, stats, registry)
        bias_corrected = utility_policy.bias_correct(layer_id, util_vec, stats)
        eligible = maturity_filter(layer_id, stats)
        if eligible.numel() == 0: continue
        expected = scheduler.expected(step, layer_id, eligible.numel())
        indices = selection_policy.select(layer_id, bias_corrected, eligible, expected)
        if indices.numel() == 0: continue
        if continuity.enabled:
            compensation_policy.compensate(layer_id, indices, stats, registry)
        replacement_policy.reinit(layer_id, indices, registry, bounds)
        optimizer_adapter.reset(layer_id, indices)
        stats.reset_units(layer_id, indices)
        logger.log({...})
```

## 9. Bias Compensation Notes
- Applied *before* wiping outgoing weights if formula depends on existing W_out.
- Uses bias-corrected mean to avoid early-age underestimation.
- Optional; disable to allow exploration noise.
- For conv→linear, expand filter mean to flattened index grouping.

## 10. Edge Cases & Safeguards
| Condition | Safeguard |
|-----------|-----------|
| All utilities identical | Random tie-break |
| New_util NaN/Inf | Replace with 0 and flag warning |
| input_weight_mag ≈ 0 | Add ε denominator |
| Replace > 50% layer (burst) | Cap fraction per step |
| Layer too small (<=2 units) | Skip structural adaptation |
| Optimizer unsupported | Skip reset with log |

## 11. Extensibility Path
Add LayerAdapter subclasses:
- LinearAdapter
- Conv2dAdapter
- AttentionHeadAdapter (maps heads)
- MLPBlockAdapter (Transformer feedforward expansion dim)
Each declares: unit_axis, incoming_params, outgoing_params, downstream_bias_param (optional).

## 12. Configuration Schema (YAML Example)
```yaml
utility_based_adaptation:
  enabled: true
  decay_rate: 0.98
  maturity_threshold: 50
  base_replacement_rate: 0.002
  accumulate: true
  continuity: { enabled: true, bias_compensation: true }
  utility_policy: contribution_centered
  selection_policy: lowest_k
  scheduler: constant
  init: kaiming
  layers:
    include: [Linear, Conv2d]
    exclude_names: [classifier]
  logging:
    every_n_steps: 50
    dump_jsonl: runs/structural_events.jsonl
```

## 13. Migration Plan
| Phase | Action | Risk |
|-------|--------|------|
| 1 | Introduce ActivationCollector + Registry (read-only) | Low |
| 2 | Extract utility & selection policies; keep old GnT call site | Low |
| 3 | Replace GnT_for_FC / ConvGnT with Engine; parity tests | Medium |
| 4 | Introduce OptimizerAdapter (support Adam & SGD) | Medium |
| 5 | Add bias compensation policy abstraction | Low |
| 6 | Add new layer adapters (optional) | Medium |

## 14. Testing Strategy
| Test | Assertion |
|------|-----------|
| Deterministic replacement (accumulate mode) | Sum(replaced) over N steps ≈ N * rate * eligible_count |
| Utility monotonic smoothing | Var(util_t - util_{t-1}) constrained |
| Bias compensation correctness | Output delta small after replacement |
| Optimizer reset | exp_avg slice = 0 post-step |
| Maturity gating | No replacements age <= threshold |
| Config toggles | Disable module -> no structural events |

## 15. Metrics & Instrumentation
- Per-layer: mean utility, std utility, replacement count, age histogram.
- Global: cumulative replaced fraction, stability score (output drift norm), training loss impact window (pre/post replacement).
- Optional: KL divergence between logits pre/post adaptation (continuity quality).

## 16. Failure Modes & Mitigations
| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Over-refresh | Training destabilizes | Lower rate / raise maturity_threshold |
| Under-refresh | Stagnation of utility distribution | Increase rate or diversify utility metric |
| Utility collapse (all ~0) | Flat distribution | Switch policy or inject random utility noise |
| Drift spike | Large loss jump post-step | Enable / refine compensation |
| Memory overhead (hooks) | GPU usage spike | Lazy detach activations, fp16 stats |

## 17. Optional Enhancements (Future)
- Soft replacement interpolation (fade-in new units).
- Diversity regularizer (penalize similarity between active units; reduce redundant replacements).
- Meta-schedule learning (learn replacement_rate over time).
- Adaptive maturity (per-unit threshold scaled by recent variance).

## 18. Glossary
| Term | Definition |
|------|------------|
| Utility | Scalar measure of unit contribution (EMA) |
| Maturity | Age threshold gating replacement eligibility |
| Replacement | Re-initialization of selected unit’s incoming (and outgoing) connectivity |
| Compensation | Downstream bias correction to preserve function |
| Accumulate Mode | Deterministic fractional expectation integration over steps |

## 19. Summary
This redesign decomposes utility-based continuous backprop into orthogonal, testable policies and adapters. It removes architecture assumptions, improves extensibility (new layer types via adapters), and clarifies invariants. Structural adaptation becomes a pluggable post-optimizer event rather than embedded control logic. Migration can proceed incrementally while preserving current empirical behavior.

---
**Next Step (if adopted):** Implement Phase 1 (ActivationCollector + Registry) and create parity tests comparing old vs new replacement counts & loss trajectories.
