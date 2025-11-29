# Rank-Restoring Continual Backpropagation (RR-CBP) Instructions

## Purpose
Rank-Restoring Continual Backpropagation (RR-CBP) keeps the Continual Backprop (CBP) replacement schedule (low-utility, mature units) while rebuilding incoming weights so that refreshed units inject well-conditioned directions under the layer covariance geometry. The current implementation supports fully connected networks and convolutional networks through `RankRestoringCBP_for_FC` and `RankRestoringCBP_for_ConvNet` together with their `RR_GnT_*` helpers.

## Core Data Flow
- During `learn`, the learner mirrors CBP: forward pass, optimizer update, then `gen_and_test` on the RR-specific Generate-and-Test (GnT) helper.
- `gen_and_test` receives the latest feature activations and, for convolutional layers, the original batch input. For each hidden layer with pending replacements it:
  1. Ensures a covariance tracker is initialised (`CovarianceState`).
  2. Updates the EMA covariance Sigma_l from layer inputs `H_prev` (supports full or diagonal mode).
  3. Builds a sigma-aware geometry abstraction (`SigmaGeometry`) and a projector onto the kept incoming weights (`SigmaProjector`).
  4. Runs the sigma-aware replacement and bookkeeping routine.

## Covariance Tracking
- `CovarianceState` gathers layer inputs shaped `(d_l, batch)` and maintains an exponential moving average controlled by `sigma_ema_beta`; `sigma_ridge` adds numerical stability.
- `covariance_dtype` optionally promotes precision (for example `float64`).
- `diag_sigma_only=True` keeps only the diagonal of Sigma, allowing inexpensive elementwise sigma operations while using the same API as the dense path.

## Geometry Primitives
- `SigmaGeometry` wraps Sigma and exposes sigma inner products, norms, whitening, random unit sampling, energy measurements, and the smallest eigenvalue of the whitened weight matrix. Dense and diagonal cases share this interface.
- `SigmaProjector` forms the sigma-orthogonal projector onto the kept incoming vectors, caching a regularised Gram inverse and supporting incremental updates as new directions are added during the same step.
- `EnergyAllocator` enforces a per-layer sigma-energy budget derived from Sigma, activation statistics, `tau`, and (optionally) `lambda_star`.

## Replacement Procedure Per Layer
1. **Gather sets.** Compute `keep_idx` (units retained) and `replace_idx` (units selected by CBP utility plus maturity logic).
2. **Prepare geometry.**
   - Collect the kept incoming rows into `V = W_kept^T` and initialise `SigmaProjector`.
   - Estimate `chi0` via `chi0_for_activation`, override with `chi0_override`, or derive from the current batch if `estimate_chi0_from_batch` is true.
   - Derive the target sigma-energy: `v_target = trace(Sigma) / max(dim, 1)` and `q_target = v_target / max(chi0, proj_eps)`.
   - Measure energy already spent by kept vectors to seed the allocator.
   - Optionally determine `lambda_star` (user-specified or derived from the minimum eigenvalue of the whitened kept Gram) and pass `tau` so the allocator can clamp the per-unit floor.
3. **Iterate replacements.** For each unit in `replace_idx`:
   - Transfer bias mass to downstream consumers if the unit’s age is non-zero.
   - Sample sigma-random directions, project onto the complement, and normalise. Retry up to `max_proj_trials`; fall back to `least_covered_direction` when all trials collapse. `nullspace_seed_epsilon` can nudge the fallback if saturation is detected.
   - When `orthonormalize_batch` is enabled, perform sigma Gram-Schmidt against any new vectors generated earlier in the same iteration.
   - Call `EnergyAllocator.allocate()` to obtain the sigma-energy `q_alloc`; saturation events are recorded and can trigger optional nullspace seeding when `improve_conditioning_if_saturated` is set.
   - Rescale the sigma-unit vector to match the allocated energy, update the projector, and write the reshaped weights back to the layer.
   - Form activations `a = w @ H_prev`, centre the bias according to `center_bias` (`mean` or `median`), and zero the outgoing weight slice. If `use_micro_seed` with `epsilon_micro_seed > 0`, inject a zero-mean sigma-normalised perturbation so the refreshed unit participates immediately without altering the batch output.
   - Reset utility trackers (`ages`, running means, absolute means) and log success, fallback, and saturation statistics.
4. **Optimizer reset.** Clear AdamGnT moment vectors (`exp_avg`, `exp_avg_sq`, `step`) for both the refreshed incoming weights and biases plus their outgoing connectors so the optimiser restarts cold.
5. **Metrics logging.** Every `log_rank_metrics_every` replacement cycles, record diagnostics such as sigma-aware rank, minimum eigenvalues, activation fraction, energy ratios, and success statistics.

## Bias and Activation Statistics
- `chi0_for_activation` supplies a default `chi0` constant based on the activation family (ReLU -> 0.5, LeakyReLU accounts for negative slope, sigmoid/tanh -> 1.0, etc.).
- `estimate_chi0_from_batch=True` replaces the constant with the clamped mean squared activation for the current batch.
- `chi0_override` forces a manual value for experimentation.

## Configuration Reference (`RRContinuousBackpropConfig`)
Extends `ContinuousBackpropConfig` with:
- `rrcbp_enabled`: master toggle to fall back to vanilla CBP without code changes.
- Covariance controls: `sigma_ema_beta`, `sigma_ridge`, `covariance_dtype`, `diag_sigma_only`, `sigma_eig_floor`.
- Projection controls: `max_proj_trials`, `proj_eps`, `projector_reg_epsilon`, `orthonormalize_batch`, `improve_conditioning_if_saturated`, `nullspace_seed_epsilon`.
- Energy policy: `tau`, `lambda_star`, `use_lambda_star`.
- Activation heuristics: `center_bias`, `estimate_chi0_from_batch`, `chi0_override`.
- Micro-seeding: `use_micro_seed`, `epsilon_micro_seed`.
- Telemetry cadence: `log_rank_metrics_every`.

## FC vs Conv Specifics
- **FC (`RR_GnT_for_FC`)**: Layer inputs come directly from cached features (transposed to `(d, batch)`), and the downstream weights are dense matrices. Optimiser resets operate on rows and columns as in vanilla CBP.
- **Conv (`RR_GnT_for_ConvNet`)**: Layer inputs are extracted with `torch.nn.functional.unfold`, producing spatial patches for sigma computation. The helper tracks Conv-to-Linear transitions through `_conv_output_multipliers` so outgoing index mapping respects flattening. Bias transfer and outgoing resets cover convolution kernels with optional micro-seeding of filter blocks, and optimiser states are cleared along the input-channel dimension (`[:, replace_idx, ...]`).

## Logging and Diagnostics
- Each layer maintains `LayerReplacementStats` summarising successes, fallbacks, saturation counts, recent energy allocations, sigma-energy budget, and whitened minimum eigenvalues.
- `_emit_rank_metrics` writes a concise log line with rank, sigma-spectrum statistics, success ratios, and energy ratios for regression tracking.

## Testing Checklist
- PyTest suites (`tests/test_rr_gnt_fc.py`, `tests/test_rr_gnt_conv.py`) should confirm that two consecutive `learn` steps keep the loss finite and that modules import correctly with the sigma-aware path enabled.
- Targeted tests should cover diagonal sigma mode, fallback logic, energy allocator behaviour, optimiser resets, and micro-seed toggles.
- Analytical validation can compare `rank(W Sigma W^T)` or `lambda_min_whitened` before and after replacement to ensure conditioning does not regress relative to random reinitialisation.

## Implementation Notes
- Wrap weight edits in `torch.no_grad()` and keep tensor dtypes and devices aligned with the owning layer to avoid autograd pollution or host-device copies.
- Guard against empty replacement sets and empty kept sets.
- The replacement logic still relies on CBP maturity and utility schedules; features like `accumulate` remain compatible without modification.
- When `diag_sigma_only=True`, every sigma interaction reduces to elementwise scaling; ensure the simplified code path stays vectorised.
