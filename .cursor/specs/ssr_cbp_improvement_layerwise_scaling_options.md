# Plan: Implement Layer-Size Normalization for SRR-CBP Asymmetric Orthogonality Penalty

## Objective
Update the SRR-CBP implementation (`srr_cbp.py`) to support different normalization modes for the Asymmetric Orthogonality (ASO) penalty, aligning its theoretical scaling with the `SO` regularizer in `src/losses/orthogonality.py`. Currently, SRR-CBP uses a "no correction" approach, which disproportionately penalizes larger layers based on the raw count of young/mature units.

## 1. Update Configurations (`configs/configurations.py`)
- Locate the `SRRCBPConfig` dataclass (and any related configs if they inherit or share this property).
- Add a new field: `aso_normalization_mode: str = "no correction"`.
- Ensure this field is properly typed and has a default value of `"no correction"` to maintain exact backward compatibility with existing experiments.

## 2. Update SRR-CBP Implementation (`src/algos/supervised/srr_cbp.py`)
- Locate the ASO loss calculation in both `SRR_CBP_for_FC` and `SRR_CBP_for_ConvNet` (specifically inside the `if len(mature_idx) > 0 and len(young_idx) > 0:` block).
- The current calculation is: `layer_aso_loss = torch.sum(weights * row_norms_sq)`
- Let $N_y$ = `len(young_idx)` and $N_m$ = `len(mature_idx)`.
- Introduce a scaling factor based on `self.config.aso_normalization_mode`:
  - **`"naive mse sum correction"`**: Divide `layer_aso_loss` by $(N_y \times N_m)$. This computes the true mean penalty over all penalized young-mature covariance pairs.
  - **`"correct by input size"`**: Divide `layer_aso_loss` by $N_y$. This computes the average penalty per young unit.
  - **`"no correction"`**: Leave as is (no division).
- Raise a `ValueError` if an unknown `aso_normalization_mode` is provided.
- *Safety check*: The existing `if len(mature_idx) > 0 and len(young_idx) > 0:` condition already guarantees that $N_y > 0$ and $N_m > 0$, so division by zero is naturally prevented.

## 3. Verify Factory Integration (`src/algos/supervised/supervised_factory.py`)
- Verify that `supervised_factory.py` correctly passes the updated `SRRCBPConfig` to the `SRR_CBP` classes. Since the factory typically passes the entire config object, this may not require any code changes, but the agent should double-check.

## 4. Update Tests (`tests/test_srr_cbp.py`)
- Add a specific test case to verify that the `aso_normalization_mode` correctly scales the ASO penalty.
- The test should:
  1. Mock or manually trigger the ASO loss computation with fixed activations/weights.
  2. Run the computation under all three modes: `"no correction"`, `"correct by input size"`, and `"naive mse sum correction"`.
  3. Assert that the resulting losses scale exactly by $1$, $1/N_y$, and $1/(N_y \times N_m)$ respectively.