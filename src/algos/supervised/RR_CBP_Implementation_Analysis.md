# Rank-Restoring CBP Implementation Analysis

This note matches each equation or pseudocode block from the RR-CBP specification with the concrete code paths currently living in `src/algos/supervised`. For every step, the relevant formula appears first, followed by the pseudocode fragment and the exact function that realises it.

## 1. Covariance Tracking

**Specification equation**

$$
\Sigma_l \leftarrow \beta \Sigma_l + (1-\beta)\frac{1}{m} H^{(l-1)} H^{(l-1)\top} + \lambda I
$$

**Implementation**

- `src/algos/supervised/rank_restoring/rr_covariance.py::CovarianceState.update` executes this update (dense path lines 24-41, diagonal path lines 33-35). Ridge term is appended using either `ema + ridge` (diag) or `ema + ridge * eye` (dense). Dtype promotion flows through `_resolve_dtype`.

```python
def update(self, h: Tensor, dtype: Optional[str] = None) -> Tensor:
    with torch.no_grad():
        batch = h.shape[1]
        ema = self.ema.to(device=device, dtype=target_dtype)
        h = h.to(dtype=target_dtype)
        if self.diag_only:
            cov = torch.mean(h * h, dim=1)
        else:
            cov = h @ h.t() / float(batch)
        ema.mul_(self.beta).add_(cov, alpha=1 - self.beta)
        if self.diag_only:
            return ema + self.ridge
        eye = torch.eye(ema.size(0), device=device, dtype=ema.dtype)
        return ema + self.ridge * eye
```

- `src/algos/supervised/rr_gnt_fc.py::_compute_layer_inputs` (lines 119-134) produces `H_prev = inputs.T` so that columns are batch samples, matching the formula.

```python
if layer_idx == 0:
    inputs = batch_input
else:
    inputs = features[layer_idx - 1]
if inputs.dim() != 2:
    raise ValueError("Expected 2-D inputs for FC RR-CBP")
return inputs.t().to(layer.weight.dtype).contiguous()
```

- `src/algos/supervised/rr_gnt_conv.py::_compute_layer_inputs` (lines 86-118) performs the analogous transformation for convolutional layers by unfolding feature maps into the matrix `H` required by the covariance update.

```python
patches = F.unfold(
    layer_input,
    kernel_size=current_layer.kernel_size,
    dilation=current_layer.dilation,
    padding=current_layer.padding,
    stride=current_layer.stride,
)
batch_size, dim, spatial = patches.shape
return patches.permute(1, 0, 2).reshape(dim, batch_size * spatial)
```

## 2. Sigma-Orthogonal Projection

**Specification equations**

$$
P_\Sigma = V (V^\top \Sigma V)^{-1} V^\top \Sigma, \qquad \tilde w = (I - P_\Sigma) u
$$

**Pseudocode**

$$
\begin{aligned}
u &\gets \Sigma v \\[2pt]
c &\gets V^{\top} u \\[2pt]
\tilde{c} &\gets (V^{\top} \Sigma V + \varepsilon I)^{-1} c \\[2pt]
\text{proj} &\gets V\,\tilde{c} \\[2pt]
\text{residual} &\gets v - \text{proj}
\end{aligned}
$$**Implementation**

- `src/algos/supervised/rank_restoring/sigma_geometry.py::SigmaProjector._refresh_cache` builds `Gram = basis.T @ (Sigma @ basis)` and solves for its inverse with regularisation (`projector_reg_epsilon`).
- `SigmaProjector.apply` and `project_complement` execute the matrix times vector logic above; all calls stay in sigma geometry instead of materialising full matrices.

```python
def apply(self, vec: Tensor) -> Tensor:
    if self.basis.numel() == 0:
        return torch.zeros_like(vec)
    if self.geometry.diag_only:
        sigma_vec = self.geometry.sigma * vec
    else:
        sigma_vec = self.geometry.sigma @ vec
    coeff = self.basis.t() @ sigma_vec
    proj_coeff = self._G_inv @ coeff
    return self.basis @ proj_coeff

def project_complement(self, vec: Tensor) -> Tensor:
    return vec - self.apply(vec)
```

- `_sample_sigma_direction` in `rr_gnt_fc.py` (lines 205-234) draws `u ~ N(0, I)` via `geometry.random_unit`, runs `projector.project_complement`, and returns `residual / ‖residual‖_Σ` once the Σ-norm exceeds `proj_eps`.

```python
for _ in range(attempts):
    candidate = geometry.random_unit(dtype=dtype)
    residual = projector.project_complement(candidate)
    norm = geometry.norm(residual)
    if norm > self.rr_config.proj_eps:
        return residual / norm, False
fallback = projector.least_covered_direction(dtype=dtype)
norm = geometry.norm(fallback)
norm = torch.clamp(norm, min=self.rr_config.proj_eps)
return fallback / norm, True
```

## 3. Fallback Direction

**Specification equation**

$$
\arg\min_{w\neq 0} \frac{w^\top \Sigma V V^\top \Sigma w}{w^\top \Sigma w}
$$

**Pseudocode**

$$
\begin{aligned}
W &\gets \Sigma^{1/2} V \\[2pt]
G &\gets W W^{\top} \\[2pt]
(\lambda, q) &\gets \text{eigmin}(G) \\[2pt]
\hat{v} &\gets \Sigma^{-1/2} q \\[2pt]
r &\gets (I - P_{\Sigma})\,\hat{v} \\[2pt]
\text{return} & r / \lVert r \rVert_{\Sigma}
\end{aligned}
$$**Implementation**

- `SigmaProjector.least_covered_direction` (lines 147-171) performs the whitening, eigen-decomposition, back transformation, and complement projection exactly as in the pseudocode. The helper is used when `_sample_sigma_direction` exhausts all seeds without a valid complement.

```python
whitened = self.geometry.whiten_columns(self.basis)
gram = whitened @ whitened.t()
eigvals, eigvecs = torch.linalg.eigh(gram)
idx = torch.argmin(eigvals)
eigvec = eigvecs[:, idx]
candidate = self.geometry.unwhiten_vector(eigvec)
residual = self.project_complement(candidate)
norm = self.geometry.norm(residual)
if norm < self.geometry.eps:
    return self.geometry.random_unit(dtype=dtype)
return residual / norm
```

## 4. Energy Allocation and Kaiming-Style Scaling

**Specification equations**

$$
q_\text{target} = \frac{\operatorname{trace}(\Sigma)}{d \cdot \chi_0}, \qquad \|w\|_\Sigma^2 = q_\text{alloc}
$$

**Pseudocode**

$$
\begin{aligned}
\chi_0 &\gets \text{resolve\_chi0}(a) \\[2pt]
v_{\text{target}} &\gets \tfrac{\operatorname{trace}(\Sigma)}{\max(d, 1)} \\[2pt]
q_{\text{target}} &\gets v_{\text{target}} / \max(\chi_0, \varepsilon) \\[2pt]
(q_{\text{alloc}}, s) &\gets \text{allocate}(q_{\text{target}}) \\[2pt]
w_{\text{in}} &\gets z \cdot \sqrt{q_{\text{alloc}}} / \lVert z \rVert_{\Sigma}
\end{aligned}
$$

**Implementation**

- `rr_gnt_fc.py::_resolve_chi0` (lines 267-280) maps the activation family to χ₀, optionally estimating from batch statistics.
- `rr_gnt_fc.py::_replace_units` (lines 151-254) computes `v_target`, `q_target`, initialises `EnergyAllocator`, and rescales directions to achieve `‖w_in‖_Σ^2 = q_alloc`.
- `src/algos/supervised/rank_restoring/sigma_geometry.py::EnergyAllocator` encapsulates the budget, residual tracking, and saturation floor (`tau * lambda_min_sigma` and optional `lambda_star`).

```python
def _resolve_chi0(self, activations: Tensor) -> float:
    if self.rr_config.chi0_override is not None:
        return float(self.rr_config.chi0_override)
    if (
        self.rr_config.estimate_chi0_from_batch
        and activations is not None
        and activations.numel() > 0
    ):
        value = torch.clamp(activations.reshape(-1).pow(2).mean(), min=self.rr_config.proj_eps)
        return float(value.item())
    return float(self._chi0_constant)

q_alloc, saturated = allocator.allocate()
norm_dir = torch.clamp(geometry.norm(direction), min=config.proj_eps)
scale = math.sqrt(max(q_alloc, config.proj_eps)) / norm_dir
w_in = direction * scale

def allocate(self) -> Tuple[float, bool]:
    saturated = self.residual <= 0.0
    if self.remaining <= 0:
        q_alloc = max(self.q_min, self.lambda_star or 0.0)
        return q_alloc, True
    if not saturated:
        q_alloc = min(self.q_target, self.residual / self.remaining)
        self.residual = max(0.0, self.residual - q_alloc)
    else:
        floor = max(self.q_min, self.lambda_star or 0.0)
        q_alloc = min(self.q_target, floor)
    self.remaining -= 1
    self.used_energy += q_alloc
    return q_alloc, saturated
```

## 5. Per-Unit Replacement (FC Path)

The spec’s per-unit step reads:

1. Bias transfer: \( b_{l+1} \leftarrow b_{l+1} + \hat f_i W_{:, i} \).
2. Draw Σ-orthogonal direction, project, normalise.
3. Orthonormalise against other new vectors (if enabled).
4. Apply Kaiming-style scaling using `q_alloc`.
5. Write weights, centre bias \( b_i = -\operatorname{stat}(a) \).
6. Zero outgoing weights; optional micro-seed.
7. Reset unit bookkeeping.

**Implementation mapping** (all in `src/algos/supervised/rr_gnt_fc.py`):

- Bias transfer: `_transfer_bias` lines 237-244.
- Direction sampling and fallback: `_sample_sigma_direction` lines 205-234 plus fallback handling.
- Batch orthonormalisation: `_orthonormalize_direction` lines 236-250 executes Σ Gram-Schmidt when `orthonormalize_batch=True`.
- Scaling and weight write-back: `w_in = direction * scale` lines 221-230 followed by `_assign_weight` lines 250-252.
- Bias centering: `_center_bias` lines 282-290; call site lines 223-228 assign the centred value directly.
- Outgoing zeroing and micro-seed: `_zero_and_seed_outgoing` lines 244-263.
- Bookkeeping reset: lines 232-241 zero `mean_feature_act`, `util`, and `ages`. Optimiser state reset lives in `_post_replacement_housekeeping` lines 312-336.

```python
if self.ages[layer_idx][unit_idx] > 0:
    bias_corrected_act = self.mean_feature_act[layer_idx][unit_idx] / (
        1 - self.decay_rate ** self.ages[layer_idx][unit_idx]
    )
    contribution = next_layer.weight.data[:, unit_idx] * bias_corrected_act
    next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

if config.orthonormalize_batch and new_vectors:
    direction = self._orthonormalize_direction(direction, new_vectors, geometry)

acts = (w_in.unsqueeze(0) @ H_prev).squeeze(0)
bias_value = self._center_bias(acts)
layer.bias.data[unit_idx] = bias_value.to(device=layer.bias.device, dtype=layer.bias.dtype)

next_layer.weight.data[:, unit_idx] = 0.0
if use_seed:
    noise = torch.randn_like(next_layer.weight.data[:, unit_idx])
    noise = noise - noise.mean()
    norm = torch.clamp(noise.norm(), min=self.rr_config.proj_eps)
    next_layer.weight.data[:, unit_idx] = epsilon * (noise / norm)

self.mean_feature_act[layer_idx][replace_idx] = 0.0
self.util[layer_idx][replace_idx] = 0.0
self.ages[layer_idx][replace_idx] = 0.0

self.opt.state[layer.weight]["exp_avg"][replace_idx, :] = 0.0
self.opt.state[next_layer.weight]["exp_avg"][:, replace_idx] = 0.0
```

## 6. Per-Unit Replacement (Conv Path)

The conv helper mirrors the same seven steps but adds tensor reshaping:

- Bias transfer across Conv→Conv and Conv→Linear is implemented in `src/algos/supervised/rr_gnt_conv.py::_transfer_bias` (lines 209-243). The function maps filter indices to flattened linear columns via `_resolve_output_columns` (lines 244-276).
- Direction sampling and scaling reuse `_sample_sigma_direction` (lines 155-185) and the same energy allocator logic.
- Weight assignment re-shapes vectors back into kernels in `_assign_weight` (lines 187-196).
- Outgoing zeroing resets either kernel slices or linear columns in `_zero_and_seed_outgoing` (lines 277-325).
- Utility and optimiser resets follow in `_post_replacement_housekeeping` (lines 346-373).

```python
if isinstance(next_layer, Conv2d):
    contribution = (next_layer.weight.data[:, unit_idx, ...] * bias_corrected_act).sum(dim=(1, 2))
    next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)
elif isinstance(next_layer, Linear):
    cols = self._resolve_output_columns(...)
    contribution = (next_layer.weight.data[:, cols] * bias_corrected_act).sum(dim=1)
    next_layer.bias.data += contribution.to(device=next_layer.bias.device, dtype=next_layer.bias.dtype)

if isinstance(layer, Conv2d):
    layer.weight.data[unit_idx] = w_in.view_as(layer.weight.data[unit_idx])
elif isinstance(layer, Linear):
    layer.weight.data[unit_idx, :] = w_in

if isinstance(next_layer, Conv2d):
    next_layer.weight.data[:, unit_idx, ...] = 0.0
    if use_seed:
        noise = torch.randn_like(next_layer.weight.data[:, unit_idx, ...])
        noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        norm = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)
        norm = torch.clamp(norm, min=self.rr_config.proj_eps)
        next_layer.weight.data[:, unit_idx, ...] = epsilon * (noise / norm.view(-1, 1, 1))

self.opt.state[next_layer.weight]["exp_avg_sq"][:, cols] = 0.0
self.opt.state[next_layer.weight]["step"][:, cols] = 0
```

## 7. Bias Centre Equation

**Formula**

$$
b^{(l)}_i = -\operatorname{mean}(a), \quad a = w^{\top} H^{(l-1)}
$$

**Code**

- `rr_gnt_fc.py::_center_bias` (lines 282-290) selects mean or median. Call site forms `acts = (w_in.unsqueeze(0) @ H_prev).squeeze(0)` preceding the assignment.
- `rr_gnt_conv.py::_center_bias` (lines 326-333) mirrors the same routine for filters.

```python
def _center_bias(self, activations: Tensor) -> Tensor:
    if activations.numel() == 0:
        return torch.zeros(1, device=activations.device, dtype=activations.dtype).squeeze(0)
    if self.rr_config.center_bias == "median":
        return -torch.median(activations)
    return -activations.mean()
```

## 8. Optimiser Reset and Safety

**Pseudocode**

$$
\begin{aligned}
\mathcal{S}[W]_{i,:} &\gets 0 \\[2pt]
\mathcal{S}[b]_{i} &\gets 0 \\[2pt]
\mathcal{S}[W_{\text{next}}]_{:, i} &\gets 0
\end{aligned}
$$

**Implementation**

- FC path: `_post_replacement_housekeeping` (lines 312-336) zeros all AdamGnT buffers for incoming weights, biases, and outgoing connectors.
- Conv path: `_post_replacement_housekeeping` (lines 346-373) handles Conv→Conv and Conv→Linear cases, clearing the appropriate tensor slices.

```python
self.opt.state[layer.weight]["exp_avg"][replace_idx, :] = 0.0
self.opt.state[layer.weight]["exp_avg_sq"][replace_idx, :] = 0.0
self.opt.state[layer.weight]["step"][replace_idx, :] = 0
self.opt.state[next_layer.weight]["exp_avg"][:, replace_idx] = 0.0
self.opt.state[next_layer.weight]["exp_avg_sq"][:, replace_idx] = 0.0
self.opt.state[next_layer.weight]["step"][:, replace_idx] = 0
```

## 9. Telemetry

**Formulas**

$$
\text{rank}(W \Sigma W^\top), \quad \lambda_{\min}(W \Sigma W^\top), \quad \lambda_{\min}(\widetilde{W} \widetilde{W}^\top)
$$**Implementation**

- FC: `_compute_rank_metrics` (lines 292-311) evaluates the Gram matrix, rank, eigenvalues, activation fraction, and allocation statistics; `_emit_rank_metrics` logs them.
- Conv: `_compute_rank_metrics` (lines 334-345) performs the identical analysis on flattened filters.

```python
if self.rr_config.diag_sigma_only:
    gram = (weight * sigma.unsqueeze(0)) @ weight.t()
else:
    gram = weight @ sigma @ weight.t()
eigvals = torch.linalg.eigvalsh(gram)
rank_val = float(torch.linalg.matrix_rank(gram, tol=self.rr_config.proj_eps).item())
active_fraction = float((activations > 0).float().mean().item()) if activations.numel() > 0 else 0.0
```

## 10. Remaining Gaps and Suggested Follow-Ups

1. **Conv batch orthonormalisation** – `_replace_units` in the conv helper does not orthonormalise multiple replacements within a layer. Porting `_orthonormalize_direction` from the FC path would enforce Σ-orthogonality for filters created in the same iteration.
2. **Nullspace seeding for conv layers** – `rr_gnt_conv.py` always returns the projected direction; optional nullspace perturbations controlled by `nullspace_seed_epsilon` are currently FC-only.
3. **Energy floors with tau = 0** – if a user sets `tau` and `lambda_star` to zero while the residual budget is exhausted, `EnergyAllocator` may hand back `q_alloc = 0`. Tightening the configuration validation or clamping to a minimum positive epsilon would harden this edge case.
4. **Unit tests** – add coverage that asserts Σ-orthogonality (`w_i^T Σ w_j ≈ 0`), verifies Kaiming-style scaling (`‖w‖_Σ^2 ≈ q_alloc`), and checks downstream bias continuity across both FC and conv replacements.

These items aside, each symbolic step from the RR-CBP instructions now has a direct implementation reference as outlined above.