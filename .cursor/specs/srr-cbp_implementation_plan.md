# Implementation Plan for SRR-CBP

## Overview
Implement Soft Rank-Restoring Continual Backpropagation (SRR-CBP) using forward hooks to compute the Asymmetric Soft Orthogonality (ASO) loss, reusing the base GnT classes from `gnt.py` (which contain the recent `use_map` improvements)  for the replacement phase.

## 1. Configuration Updates (`configs/configurations.py`)
- Add a new dataclass `SRRCBPConfig` that inherits from `ContinuousBackpropConfig`.
- Add the new hyperparameters required for the ASO loss:
  - `type: str = 'srr_cbp'`
  - `SO_reg_lambda: float = 0.1` (Initial ASO penalty)
  - `aso_age_decay_rate: float = 0.99` (Decay rate for ASO penalty)

## 2. Leaner Implementation (`src/algos/supervised/srr_cbp.py`)
Create a new file to house the SRR-CBP learners.

### `SRR_CBP_for_ConvNet` and `SRR_CBP_for_FC`
- Inherit from `Learner` (similar to `ContinuousBackprop_for_ConvNet`).
- **Initialization (`__init__`)**:
  - Initialize the standard `ConvGnT_for_ConvNet` (or `GnT_for_FC`) from `src/algos/gnt.py` for utility tracking and unit replacement.
  - Set up a dictionary `self.pre_activations = {}`.
  - Iterate through the network's layers (using `self.net.get_plasticity_map()` or legacy `self.net.layers`).
  - Register a PyTorch forward hook on each `weight_module` (e.g., `Conv2d` or `Linear`) to capture its output (the pre-activation) and store it in `self.pre_activations[layer_idx]`.

- **Learning Step (`learn` method)**:
  - Execute the forward pass: `output, features = self.net.predict(x)`.
  - Compute the standard task loss.
  - **Compute ASO Loss**:
    - Initialize `aso_loss = 0.0`.
    - Loop over each hidden layer `i`:
      - Retrieve unit ages from `self.gnt.ages[i]`.
      - Identify `mature_idx` (ages >= maturity threshold) and `young_idx` (ages < maturity threshold).
      - If both young and mature units exist:
        - Retrieve pre-activations `A = self.pre_activations[i]`.
        - Reshape `A` to `(num_units, m)` where `m` is `batch_size * H * W` for Conv2d or `batch_size` for Linear.
        - Extract `A_mature = A[mature_idx, :].detach()` (crucial: stop gradient).
        - Extract `A_young = A[young_idx, :]`.
        - Compute the covariance matrix: `cov = torch.matmul(A_young, A_mature.t())`.
        - Calculate the penalty weights for young units: `weights = (aso_age_decay_rate ** ages[young_idx]) / (2 * m**2)`.
        - Compute the squared L2 norm of each row in `cov` and multiply by `weights`.
        - Add the sum to `aso_loss`.
    - Multiply the accumulated `aso_loss` by `SO_reg_lambda`: `aso_loss = SO_reg_lambda * aso_loss`.
  - Add `aso_loss` to the task loss: `total_loss = task_loss + aso_loss`.
  - Perform backward pass on `total_loss`: `total_loss.backward()`.
  - Step the optimizer and zero gradients.
  - Execute the GnT replacement phase: `self.gnt.gen_and_test(features=features)`.
  - Return `task_loss.detach()` and `output.detach()`.

## 3. Factory Registration (`src/algos/supervised/supervised_factory.py`)
- Import `SRR_CBP_for_ConvNet`, `SRR_CBP_for_FC`, and `SRRCBPConfig`.
- Add an `elif normalized_type == 'srr_cbp':` branch in `create_learner`.
- Ensure the config is cast/validated as `SRRCBPConfig`.
- Return the appropriate learner based on `net_cls` ('conv' or 'fc').

## To-Do List
- [ ] Add `SRRCBPConfig` to `configs/configurations.py`
- [ ] Create `src/algos/supervised/srr_cbp.py` with `SRR_CBP_for_ConvNet` and `SRR_CBP_for_FC`
- [ ] Implement forward hooks for pre-activations in learner `__init__`
- [ ] Implement ASO loss computation in learner `learn` method (with global `SO_reg_lambda` multiplication)
- [ ] Register learners in `supervised_factory.py`