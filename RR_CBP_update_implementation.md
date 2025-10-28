# Implementation Brief: Upgrade CBP to RR-CBP with Σ-aware, Variance-Preserving Rank Restoration

## Objective
Replace CBP’s random reinit of low-utility units with a **rank- and conditioning-aware** reconstruction of incoming weights that:
1) is **Σ-orthogonal** to kept units,
2) preserves **post-activation second moment** (He-like) in the **empirical covariance geometry** of the layer,
3) respects a **layer energy budget**, and
4) still restores rank under **overbudget** conditions via a controlled minimum Σ-norm.

This improves the preactivation Gram spectrum and, via NTK factorization, the learnability spectrum.

---

## Interfaces and Data

- Per-layer state
  - `Sigma[l]` — EMA of upstream covariance: `Sigma[l] ≈ (1/m) H^{l-1} H^{l-1}^T`
  - `age[l,i], util[l,i], meanEMA[l,i]` — existing CBP stats
  - Incoming matrix `Win[l]` (rows = units, cols = d_l); Outgoing `Wout[l]`
  - Biases `b[l]`, next-layer biases `b[l+1]`

- Hyperparameters (new)
  - `beta` — EMA decay for `Sigma` (suggest 0.99–0.999)
  - `tau` — rank-restoring floor scale (e.g., 1e-2)
  - `lambda_star` — optional conditioning target (default: disabled or use `min(1, 2*lambda_min(WhitenedGramKeep))`)
  - `epsilon` — optional micro-seed amplitude (e.g., 1e-4)
  - `act` — activation type per layer (for `chi0(act)`)

- Existing CBP hyperparameters retained
  - `rho` (replacement rate), `M` (maturity), utility EMA, etc.

---

## Algorithm Changes (Step-by-Step)

### 1) Maintain Σ via EMA
At each training step and for each layer `l`:
- Extract batch upstream features `H = H^{(l-1)} ∈ R^{d_l × m}`.
- Update: `Sigma[l] = beta*Sigma[l] + (1-beta)*(H H^T / m)`.

> Numerical stability: store `Sigma_eps = Sigma + eps*I` with `eps ≈ 1e-6` when inverting or taking square-roots.

### 2) Choose replacement set
- As in CBP: select `r = ceil(rho * N_l)` **mature** units with smallest utility.

### 3) Build Σ-projector onto kept span
- Let `V` be incoming columns of **kept** units (`d × k`).
- Compute `G = V^T Sigma V` and use `G_reg = G + gamma*I` with small `gamma` (e.g., `1e-6*trace(G)/k`) to stabilize.
- Projector (right-acting on vectors): `P = V G_reg^{-1} V^T Sigma`.

> Notes:
> - If `V` is empty, `P = 0`.
> - Cache and reuse `G_reg^{-1}` while processing all replacements in this layer.

### 4) Direction for each replaced unit
- Sample `u ~ N(0, I_d)` and set `w_hat = (I - P) u`.
- If `||w_hat||_Sigma == 0` (saturated complement), use **least-covered** direction:
  - Compute `M' = Sigma^{1/2} V V^T Sigma^{1/2}` (via eigendecomp of `Sigma` or Cholesky).
  - Let `u_min` be the smallest-eigenvalue eigenvector of `M'`.
  - Set `w_dir = Sigma^{-1/2} u_min`.
- Else `w_dir = w_hat`.

> Implementation: obtain `Sigma^{±1/2}` from eigendecomposition of `Sigma_eps` (SPD); cache per layer.

### 5) Variance-preserving Σ-norm allocation (He-like but Σ-aware)
- Let `v_tar = trace(Sigma)/d` (average upstream second moment per coordinate).
- Let `chi0 = chi0(act)`; for ReLU, `chi0 = 1/2`.
- Set **target preactivation variance**: `q_tar = v_tar / chi0`.
- Energy budget for layer: `Q_tar = N * q_tar`.
- Current used energy by kept units: `Q_used = trace(Win_keep * Sigma * Win_keep^T)`.
- Residual: `Q_res = max(0, Q_tar - Q_used)`.

For each replaced unit (process sequentially):
- If `Q_res > 0`: `q_alloc = min(q_tar, Q_res / remaining_units)` and then reduce `Q_res`.
- Else (overbudget): define floor `q_min = tau * lambda_min(Sigma)` and optional target `lambda_star`; set `q_alloc = min(q_tar, max(q_min, lambda_star))`.

Scale: `w_in = w_dir * sqrt(q_alloc) / ||w_dir||_Sigma`.

> Rationale: preserves post-activation second moment (under Gaussian approx) and keeps the **preactivation Gram** spectrum well conditioned. In whitened coordinates, the new preactivation eigenvalue equals `q_alloc`.

### 6) Bias centering and outgoing weights
- Center preactivations on the batch: `b[l,i] = -mean((w_in^T) H)`.
- Zero outgoing row to preserve function; optionally **micro-seed**:
  - Build `X_i = phi(w_in^T H + b[l,i])`.
  - Choose small vector `v` with `X_i^T v ≈ 0` and set `Wout[l][i,:] = epsilon * v`.

### 7) Finalize bookkeeping
- Transfer mean activation to next-layer biases **before** replacing (as in CBP).
- Reset age, utility, and mean EMA for the replaced unit.

---

## Numerical & Engineering Notes

- **Regularization:** always invert `G + gamma*I`. Choose `gamma` via relative scale (e.g., `gamma = 1e-6 * (trace(G)/k)`).
- **Root and inverse root of Σ:** use eigendecomposition (`Sigma = U diag(λ) U^T`), then `Sigma^{1/2} = U diag(√λ) U^T`, `Sigma^{-1/2} = U diag(1/√λ) U^T` with clipping at `λ_min_clip = eps`.
- **Batch sizing:** if `m < d`, `Sigma` is rank-deficient; the `eps*I` regularization ensures SPD.
- **Complexity:** per layer, projector build costs `O(d k^2 + k^3)`; usually `k ≈ N_l - r` and `r` is small (`rho` small). Eigen for `Sigma` is `O(d^3)` but amortized by EMA cadence (do not recompute every step; reuse for `K` steps).
- **Caching:** cache `U, √λ, 1/√λ` for `Sigma`, and `G^{-1}` across all replacements in the layer for the current step.

---

## Hyperparameters and Defaults

- `beta = 0.99` (covariance EMA); utility EMA unchanged.
- `tau = 1e-2` (rank-restoring floor).
- `lambda_star`: default disabled; or `min(1, 2*lambda_min(WhitenedGramKeep))`.
- `epsilon = 1e-4` (micro-seed); can be `0` to disable.
- `chi0(act)`: ReLU `1/2`, LeakyReLU(`α`) `(1+α^2)/2`. Otherwise estimate on batch.

---

## Telemetry / Diagnostics

- Log per layer: `lambda_min` of whitened preactivation Gram (kept vs after), `Q_used/Q_tar`, average `q_alloc`, fraction of saturated replacements.
- Check orthogonality: `||V^T Sigma w_in||_2` should be small.
- Check moment matching: empirical `mean(phi(w_in^T H + b)^2)` vs `v_tar`.

---

## Backward Compatibility and Feature Flags

- Feature flag `rrcbp_enabled`: if off, revert to original CBP random reinit.
- Sub-flags: `use_micro_seed`, `use_lambda_star`, `estimate_chi0_from_batch`.

---

## Tests and Acceptance Criteria

1. **Projection correctness:** for random tests, ensure `||V^T Sigma w_in||_2 / ||w_in||_Σ < 1e-6`.
2. **Moment preservation:** for ReLU, verify on batch that `mean(phi(a)^2)` is within `±10%` of `v_tar`.
3. **Rank restoration:** in whitened space, eigen-count increases by the number of replacements unless saturated.
4. **Stability:** training loss unaffected or improved vs CBP baseline on a sanity task; no gradient explosions.
5. **Ablations:** compare random reinit vs Σ-orthogonal unit-norm vs Σ-orthogonal with variance-preserving scaling (this spec); demonstrate improved `lambda_min` and similar or better accuracy.

---

## Pseudocode Reference
See the **“RR-CBP with Σ-aware scaling”** algorithm block above; implement as written. Avoid inserting LaTeX spacing commands inside the algorithmic environment in code comments (keeps Obsidian rendering clean).
