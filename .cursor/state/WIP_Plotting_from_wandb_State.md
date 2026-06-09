# WIP State: Plotting Experimental Results from W&B

**Date:** Friday Feb 27, 2026
**Topic:** `Plotting_experimental_results_from_wandb`

## 1. What is currently working

*   **Modular Architecture:** The large `plasticity_plotting_pipeline.py` script has been successfully refactored into a modular orchestrator that imports specialized logic from the `analysis/helpers/` directory (`config_utils.py`, `data_processing.py`, `plotting_functions.py`, `plotting_utils.py`, `wandb_utils.py`).
*   **Robust Data Fetching:** W&B history extraction now uses `scan_history` to reliably pull metrics without throwing "No history data" warnings.
*   **Advanced Metric Computation:**
    *   `forward_loss` calculation supports multiple methods (`mean`, `median`, `ema`, `linear_decay`).
    *   The `ema` method correctly handles finite-window bias via normalization (`np.average(weights=...)`) and uses actual temporal indices to properly weight data points even if intermediate tasks are missing (NaNs).
*   **Outlier Filtering:** A robust `filter_outliers` function (supporting `quantile`, `iqr`, and `z_score` methods) is applied *per-target* to remove extreme spikes (e.g., loss explosions) before plotting.
*   **Dynamic Plot Titles & Labels:**
    *   All plot titles automatically append the outlier filtering status (e.g., `(top/bottom 1% forward_loss, epoch_loss outliers filtered)`).
    *   The `forward_loss` axis label globally includes the window size (e.g., `Forward Loss (w=10)`).
*   **Enhanced Scatter Plot Matrix:**
    *   Supports both "Square" (pairwise combinations) and "Rectangular" (Independent vs. Dependent variables) modes.
    *   Supports custom log-scaling for specific axes (e.g., `log2`, `log10`).
    *   Features a `custom_regplot` function that correctly calculates and draws linear regression lines in the log-transformed space, ensuring they appear straight on log-scaled plots.
*   **Reproducibility:** The pipeline automatically copies the active YAML configuration file to the output directory, prefixed with the `comparison_name` (e.g., `my_comparison_plasticity_metric_candidate.yaml`).

## 2. What is currently broken or throwing errors

*   **No known hard errors:** The pipeline logic is currently stable based on the latest refactoring and feature additions.
*   **Pending Verification:** The most recent change—updating the EMA calculation in `compute_forward_loss` to use explicit temporal indices for weights—has not yet been run against live W&B data by the user. While mathematically sound, it should be verified to ensure it behaves exactly as expected on real, potentially gappy data.

## 3. Exact next steps when we resume

1.  **Run the Pipeline:** Execute the pipeline using the updated `analysis/cfg/plasticity_metric_candidate.yaml` (or the recently viewed `analysis/cfg/reality_check.yaml`) to verify the end-to-end flow with real W&B data.
    ```bash
    python analysis/plasticity_plotting_pipeline.py --config analysis/cfg/plasticity_metric_candidate.yaml
    ```
2.  **Verify EMA Logic:** Inspect the generated `forward_loss` values to ensure the new index-based EMA weighting and bias correction are producing the desired smoothing effect without temporal distortion.
3.  **Review Visuals:** Check the generated plots in the output directory:
    *   Confirm the scatter matrix regression lines look correct on log-scaled axes.
    *   Verify that plot titles accurately reflect the outlier filtering parameters.
    *   Ensure the copied `.yaml` config file appears in the output folder.
4.  **Address `reality_check.yaml`:** If `analysis/cfg/reality_check.yaml` represents a new experiment or configuration, review its settings to ensure it aligns with the newly added features (like rectangular scatter matrices or outlier filtering).