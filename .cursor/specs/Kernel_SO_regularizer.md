# Kernel SO Regularizer Implementation Plan

## 1. Create the Regularizer Class
**File:** `src/losses/orthogonality.py`
- Add the `KernelSORegularizer` class extending `nn.Module` to this existing file.
- **Initialization:** 
  - Accept `main_loss_func`, `model`, `lambda_orth` (defaulting to `1e-4` or similar), and `normalization_mode` (default `"naive mse sum correction"`).
  - Iterate through `model.modules()` and cache references to `nn.Conv2d` and `nn.Linear` layers.
  - Print a warning if neither layer type is found.
- **Forward Pass:**
  - Compute `self.last_task_loss = self.main_loss_func(output, target)`.
  - Initialize `reg_loss = 0.0`.
  - **Helper Logic:** For a given Gram matrix `G` (which is either $W W^T$ or $W^T W$) and identity matrix `I` of dimension `D` (where `D` is $M$ or $N$):
    - If `normalization_mode == "naive mse sum correction"`: `P = F.mse_loss(G, I, reduction='mean')` (computes $\frac{1}{D^2} \|G - I\|_F^2$)
    - If `normalization_mode == "correct by input size"`: `P = F.mse_loss(G, I, reduction='sum') / D` (computes $\frac{1}{D} \|G - I\|_F^2$)
    - If `normalization_mode == "no correction"`: `P = F.mse_loss(G, I, reduction='sum')` (computes $\|G - I\|_F^2$)
  - **Conv2d Logic:** For each cached conv layer, get the weight tensor `W` (shape $M \times C \times k_H \times k_W$). Reshape to $M \times N$ where $N = C \cdot k_H \cdot k_W$.
    - If $M < N$ (fat): Compute `P` using `W @ W.T` and `I_M` with $D=M$.
    - If $M \ge N$ (tall): Compute `P` using `W.T @ W` and `I_N` with $D=N$.
    - `reg_loss += P`
  - **Linear Logic:** For each cached linear layer, get the weight tensor `W` (shape $M \times N$).
    - If $M < N$ (fat): Compute `P` using `W @ W.T` and `I_M` with $D=M$.
    - If $M \ge N$ (tall): Compute `P` using `W.T @ W` and `I_N` with $D=N$.
    - `reg_loss += P`
  - Compute `self.last_reg_loss = reg_loss * self.lambda_orth`.
  - Return `self.last_task_loss + self.last_reg_loss`.

## 2. Update Configurations
**File:** `configs/configurations.py`
- Update the `BaseLearnerConfig` class documentation/type hints to explicitly mention `"Kernel SO"` as a valid option for `additional_regularization`.
- Ensure `lambda_orth` is used to control the regularization strength (it already exists in `BaseLearnerConfig`).
- Add `normalization_mode: str = "naive mse sum correction"` to `BaseLearnerConfig`.

## 3. Integrate with Base Learner
**File:** `src/algos/supervised/base_learner.py`
- Import `KernelSORegularizer` from `src.losses.orthogonality`.
- In the `_init_loss` method, refactor the regularization check to handle different strings for `additional_regularization`:
  - If `self.config.additional_regularization == 'SVD_Orthogonal'`: Return `RegularizedLoss_SVD_conv(...)` (the existing logic).
  - If `self.config.additional_regularization == 'Kernel SO'`: Return `KernelSORegularizer(...)`, passing the `main_loss_func`, `model`, `lambda_orth`, and `normalization_mode`.
  - If `self.config.additional_regularization` is any other truthy string, raise a `ValueError` for unsupported regularization type.