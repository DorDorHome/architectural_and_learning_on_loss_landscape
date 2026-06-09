# Statistical Analysis Tool for W&B Runs — Design Plan

## 1. Overview

A YAML-driven statistical analysis framework that loads data from existing W&B runs, applies flexible filtering, and performs regression analysis with comprehensive diagnostics and visualizations.

**Goals:**
- Load metrics and config values from W&B runs with flexible filtering
- Support **panel data** (multiple rows per run, indexed by `task_idx` / `global_epoch`)
- Perform OLS (and other) regression on selected variables, optionally stratified by grouping variables
- Produce publication-quality regression diagnostics and plots
- Support **trajectory comparison** across groups (e.g., comparing learner types over task progression)
- Be driven entirely by a YAML config file for reproducibility and convenience

---

## 2. Directory Structure

```
analysis/
├── run_statistical_analysis.py          # Main launcher (reads YAML, orchestrates pipeline)
├── cfg/
│   ├── analysis_config.yaml             # Regression analysis config
│   └── trajectory_config.yaml          # Trajectory comparison config (loss/accuracy over task_idx by learner)
├── helpers/
│   ├── __init__.py
│   ├── wandb_data_loader.py             # W&B API wrapper: fetch runs, apply filters, extract metrics
│   ├── data_processing.py               # DataFrame assembly, cleaning, transforms, panel reduction
│   ├── regression_analysis.py           # Regression fitting (OLS, Ridge, Lasso, WLS, Robust)
│   ├── regression_diagnostics.py        # Diagnostic tests & statistics
│   └── plotting.py                      # All plots: partial regression, partial residual, QQ, trajectory, etc.
├── statistical_analysis_for_wandb_runs_plan.md   # This file
└── outputs/                             # (auto-created) saved figures and result tables
```

---

## 3. YAML Configuration Schema

The YAML file is the single source of truth for an analysis run. It has four top-level sections.

```yaml
# ============================================================
#  analysis_config.yaml — Example / Template
# ============================================================

# --- 1. W&B source -----------------------------------------
wandb:
  entity: "your-entity"            # W&B entity (user or team)
  project: "your-project"          # W&B project name

# --- 2. Run filtering --------------------------------------
filters:
  # Run state (select one or more)
  state: ["finished"]              # "finished", "running", "crashed", "failed", or list

  # Filter by W&B tags (runs must have ALL listed tags)
  tags: []                         # e.g. ["baseline", "v2"]

  # Filter by run name regex
  name_regex: null                 # e.g. "sgd.*layernorm"

  # Filter by creation date (ISO format)
  created_after: null              # e.g. "2025-01-01"
  created_before: null

  # Filter by W&B run config values (dot-notation paths into run.config)
  # Each entry is a key-value pair; all must match (AND logic).
  # Values can be a single value (exact match) or a list (any-of match).
  config_filters:
    # Examples:
    # net.type: "ConvNet_FC_layer_norm"
    # learner.type: ["backprop", "rr_cbp2"]
    # learner.step_size: 0.01

  # Filter by summary metric ranges
  # Each entry: metric_name: {min: ..., max: ...} (either bound is optional)
  summary_filters: {}
    # Examples:
    # final_accuracy: {min: 0.8}
    # final_loss: {max: 0.5}

  # Maximum number of runs to load (null = all)
  max_runs: null

# --- 3. Variable selection ---------------------------------
variables:
  # Config keys to extract as independent/grouping variables.
  # Use dot-notation for nested config values.
  # Each entry becomes a column in the analysis DataFrame.
  config_keys:
    - "learner.step_size"
    - "learner.weight_decay"
    - "learner.type"
    - "net.type"
    - "batch_size"

  # Summary metrics to extract (final values logged by the run).
  summary_metrics:
    - "final_accuracy"
    - "final_loss"
    - "effective_rank_normalized_aurc"

  # ---- History (time-series) data ----
  # History metrics — full time-series data (optional, expensive).
  # Each key is extracted at every logged step.
  history_metrics: []
    # e.g. ["global_epoch", "task_idx", "epoch_loss", "epoch_accuracy",
    #        "layer_0_num_dead_units", "effective_rank_normalized_aurc"]

  # How to sample/reduce history rows.
  history_sampling:
    method: "last"                 # "all", "last", "first", "every_n", "at_steps",
                                   # "last_per_group"   <-- NEW for panel data
    n: 1                           # for "every_n": step interval; for "last"/"first": how many
    steps: []                      # for "at_steps": explicit step list

    # --- For method: "last_per_group" (panel data reduction) ---
    # Group history rows by this column and keep the row with the max of sort_by.
    # Typical use: group_by task_idx, sort_by global_epoch → one row per task.
    group_by: null                 # e.g. "task_idx"
    sort_by: null                  # e.g. "global_epoch"

  # ---- Post-reduction filtering on history columns ----
  # Applied AFTER history_sampling reduction.
  # Supports: {min: ..., max: ...} for ranges, or a list for exact values.
  history_post_filters: {}
    # Examples:
    # task_idx: {min: 400}                # only tasks 400+
    # task_idx: [0, 100, 200, 300, 499]   # specific task indices

  # ---- Optional per-run derived metrics ----
  # Collapse each run's trajectory to a single value.
  # These become additional columns in a one-row-per-run DataFrame.
  # Use only when you need to compare run-level summaries.
  derived_metrics: []
    # - name: "loss_auc"
    #   method: "trapz"              # area under curve (numpy.trapz)
    #   y: "epoch_loss"
    #   x: "task_idx"
    # - name: "loss_final_vs_initial"
    #   method: "ratio_last_first"
    #   variable: "epoch_loss"
    # - name: "plasticity_slope"
    #   method: "linregress_slope"
    #   y: "epoch_loss"
    #   x: "task_idx"

# --- 4. Analysis specification -----------------------------
analysis:
  # --- Analysis type ---
  # "regression"              — fit a regression model with diagnostics & plots
  # "trajectory_comparison"   — compare metric trajectories across groups
  type: "regression"

  # =========================================================
  # Settings for type: "regression"
  # =========================================================
  regression:
    # The dependent (response) variable for regression.
    dependent_variable: "epoch_loss"

    # Independent (predictor) variables.
    # Must be numeric or will be one-hot encoded if categorical.
    independent_variables:
      - "layer_0_num_dead_units"
      - "layer_1_num_dead_units"
      - "effective_rank_normalized_aurc"

    # Regression type
    regression_type: "ols"           # "ols", "wls", "ridge", "lasso", "robust"

    # For ridge/lasso, regularization strength
    alpha: 1.0

    # Add polynomial or interaction terms
    transforms: []
      # - type: "log"
      #   variable: "layer_0_num_dead_units"
      # - type: "polynomial"
      #   variable: "layer_0_num_dead_units"
      #   degree: 2
      # - type: "interaction"
      #   variables: ["layer_0_num_dead_units", "effective_rank_normalized_aurc"]

    # --- Stratified / grouped regression ---
    # Run the SAME regression separately for each level of the grouping variable(s).
    # This answers: "does the dead_units → loss relationship hold across learner types?"
    group_by: null                   # e.g. "learner__type" or ["learner__type", "net__type"]
    compare_groups: true             # Print side-by-side comparison table across groups

    # Which diagnostics to compute
    diagnostics:
      # Goodness-of-fit
      r_squared: true
      adjusted_r_squared: true
      ssr: true                      # Sum of Squares due to Regression (model)
      sse: true                      # Sum of Squared Errors (residual)
      sst: true                      # Total Sum of Squares
      aic: true
      bic: true
      f_statistic: true
      log_likelihood: true

      # Coefficient diagnostics
      confidence_intervals: true     # 95% CI for coefficients
      p_values: true
      standardized_coefficients: true

      # Residual diagnostics
      durbin_watson: true            # Autocorrelation
      jarque_bera: true              # Normality of residuals
      breusch_pagan: true            # Heteroscedasticity
      goldfeld_quandt: true          # Heteroscedasticity (alternative)
      condition_number: true         # Multicollinearity
      vif: true                      # Variance Inflation Factor

      # Influence diagnostics
      cooks_distance: true
      leverage: true                 # Hat-values
      dffits: true

    # Which plots to produce
    plots:
      partial_regression: true       # a.k.a. added-variable / leverage plot
      partial_residual: true         # a.k.a. component-plus-residual (CCPR) plot
      residuals_vs_fitted: true
      qq_plot: true                  # Normal Q-Q of residuals
      scale_location: true           # sqrt(|standardized residuals|) vs fitted
      cooks_distance: true           # Cook's D bar chart
      leverage_vs_residuals: true    # Residuals vs leverage with Cook's D contours
      correlation_heatmap: true      # Heatmap of predictor correlations
      pairplot: true                 # Pairwise scatter matrix of all variables

  # =========================================================
  # Settings for type: "trajectory_comparison"
  # =========================================================
  trajectory:
    x_axis: "task_idx"               # the "time" / progression variable
    y_variables:                     # metrics to plot over x_axis
      - "epoch_loss"
      - "epoch_accuracy"

    # Group runs by config variable(s) — each group gets its own curve
    group_by: "learner__type"        # e.g. "learner__type" or ["learner__type", "net__type"]

    # How to summarize across runs within each group
    summary_statistics:
      - "mean"                       # mean curve across runs in the group
      - "std"                        # ± 1 std shaded band
      # - "median"
      # - "q25_q75"                  # interquartile range band

    # Optional quantitative per-group summaries (printed as table)
    comparisons:
      - "auc"                        # area under curve (lower loss = better plasticity)
      - "final_vs_initial"           # ratio of last vs first value
      - "max_degradation"            # worst value relative to best achieved
      # - "recovery_events"          # count of times metric improves after degrading

  # =========================================================
  # Output options (shared by all analysis types)
  # =========================================================
  output:
    save_figures: true
    figure_format: "png"           # "png", "pdf", "svg"
    figure_dpi: 150
    output_dir: "outputs"          # Relative to analysis/ directory
    print_summary: true            # Print summary tables to console
    save_summary_csv: true         # Save result tables as CSV
```

---

## 4. Module Responsibilities

### 4.1 `run_statistical_analysis.py` (Launcher)

**Responsibility:** Parse CLI args, load YAML config, orchestrate the full pipeline.

```
Flow:
  1. Load YAML config (with CLI override for config path)
  2. Call wandb_data_loader to fetch & filter runs → raw run list
  3. Call data_processing to assemble a clean pandas DataFrame
     - If history_metrics specified: build panel DataFrame (multi-row per run)
     - Apply history_sampling reduction (e.g., last_per_group)
     - Apply history_post_filters
     - Compute derived_metrics if any
  4. Dispatch based on analysis.type:
     a. "regression" → regression pipeline (fit, diagnose, plot)
        - If group_by set: run pipeline per group, then compare
     b. "trajectory_comparison" → trajectory pipeline (aggregate, plot, compare)
  5. Print summary, save outputs
```

**CLI interface:**
```bash
python run_statistical_analysis.py --config cfg/analysis_config.yaml
# Optional overrides:
#   --entity ENTITY   --project PROJECT   --dry-run
```

### 4.2 `helpers/wandb_data_loader.py`

**Responsibility:** All W&B API interactions.

**Key functions:**

| Function | Description |
|---|---|
| `load_runs(entity, project, filters) -> List[wandb.Run]` | Fetch runs from W&B API, applying state/tag/date filters via the API where possible, and post-filtering for regex/config/summary filters |
| `extract_run_data(run, config_keys, summary_metrics) -> dict` | Extract specified config values (dot-notation) and summary metrics from a single run |
| `extract_history(run, history_metrics, sampling) -> pd.DataFrame` | Extract time-series history for specified metrics from a single run |
| `build_dataframe(runs, variables_config) -> pd.DataFrame` | Orchestrate extraction across all runs into a single DataFrame. If history_metrics present, returns a panel DataFrame with `run_id` + `run_name` columns and one row per (run, logged_step). |

**Filtering strategy:**
- W&B API natively supports filtering by `state`, `tags`, `config.*` (via MongoDB-style query syntax in `wandb.Api().runs(filters=...)`).
- Post-fetch filtering for: `name_regex`, `created_after/before`, `summary_filters` (range checks), and complex config value list-matching.
- This two-stage approach minimizes API calls while allowing arbitrary filtering.

### 4.3 `helpers/data_processing.py`

**Responsibility:** Clean and transform the raw DataFrame for analysis.

**Key functions:**

| Function | Description |
|---|---|
| `reduce_history(df, sampling_config) -> pd.DataFrame` | Apply history sampling: `last_per_group` groups by `group_by` column within each run and keeps the row with the max `sort_by` value. Other methods: `all`, `last`, `first`, `every_n`, `at_steps`. |
| `apply_history_post_filters(df, filters) -> pd.DataFrame` | Filter panel rows by ranges or explicit values on history columns (e.g., `task_idx >= 400`) |
| `compute_derived_metrics(df, derived_config) -> pd.DataFrame` | Compute per-run aggregates: `trapz` (AUC), `linregress_slope`, `ratio_last_first`, `mean`, `std`, `max_minus_min`. Returns a one-row-per-run DataFrame with the new columns. |
| `clean_dataframe(df, required_columns) -> pd.DataFrame` | Drop rows with NaN in analysis columns, report missingness |
| `encode_categoricals(df, columns) -> pd.DataFrame` | One-hot encode categorical variables (e.g., `net.type`, `learner.type`) |
| `apply_transforms(df, transforms_config) -> pd.DataFrame` | Apply log, polynomial, interaction transforms as specified in YAML |
| `get_design_matrix(df, dep_var, indep_vars) -> (y, X)` | Return response vector y and design matrix X (with constant) for statsmodels |
| `sanitize_column_names(df) -> pd.DataFrame` | Replace dots in column names to valid Python identifiers (`learner.type` → `learner__type`) |

**Panel data flow (typical plasticity experiment):**
```
Raw history (25,000 rows per run × N runs)
  → reduce_history(method="last_per_group", group_by="task_idx", sort_by="global_epoch")
  → 500 rows per run (one per task_idx)
  → apply_history_post_filters(task_idx: {min: 400})
  → 100 rows per run
  → ready for regression or trajectory analysis
```

### 4.4 `helpers/regression_analysis.py`

**Responsibility:** Fit regression models, including stratified/grouped regression.

**Key functions:**

| Function | Description |
|---|---|
| `fit_regression(y, X, regression_type, **kwargs) -> RegressionResult` | Dispatch to OLS / WLS / Ridge / Lasso / RobustLM and return fitted results |
| `fit_ols(y, X) -> RegressionResult` | OLS via `statsmodels.api.OLS` |
| `fit_wls(y, X, weights) -> RegressionResult` | Weighted Least Squares |
| `fit_ridge(y, X, alpha) -> RegressionResult` | Ridge via sklearn, wrapped for diagnostics compatibility |
| `fit_lasso(y, X, alpha) -> RegressionResult` | Lasso via sklearn, wrapped |
| `fit_robust(y, X) -> RegressionResult` | Robust regression via `statsmodels.api.RLM` |
| `run_grouped_regression(df, config) -> Dict[str, RegressionResult]` | Split df by `group_by`, fit the same model in each group, return dict of results |
| `compare_group_results(results_dict) -> pd.DataFrame` | Side-by-side table of R², coefficients, p-values across groups |

**Return type:** A dataclass `RegressionResult` that holds the fitted model, residuals, predictions, coefficient info, design matrix, and response vector — providing a uniform interface regardless of backend.

### 4.5 `helpers/regression_diagnostics.py`

**Responsibility:** Compute all diagnostic statistics.

**Key functions:**

| Function | Description |
|---|---|
| `compute_diagnostics(result, config) -> dict` | Master function; computes all diagnostics enabled in YAML |
| `goodness_of_fit(result) -> dict` | R², adj-R², SSR, SSE, SST, AIC, BIC, F-stat, log-likelihood |
| `coefficient_diagnostics(result) -> pd.DataFrame` | Coefficients, std errors, t-stats, p-values, 95% CI, standardized betas |
| `residual_diagnostics(result) -> dict` | Durbin-Watson, Jarque-Bera, Breusch-Pagan, Goldfeld-Quandt |
| `multicollinearity_diagnostics(X) -> dict` | Condition number, VIF for each predictor |
| `influence_diagnostics(result) -> pd.DataFrame` | Cook's distance, leverage (hat values), DFFITS for each observation |
| `format_diagnostics_report(diagnostics, group_name) -> str` | Pretty-print all diagnostics to console, with optional group label |

**Libraries used:**
- `statsmodels.stats.stattools.durbin_watson`
- `statsmodels.stats.diagnostic.het_breuschpagan`
- `statsmodels.stats.diagnostic.het_goldfeldquandt`
- `statsmodels.stats.stattools.jarque_bera`
- `statsmodels.stats.outliers_influence.variance_inflation_factor`
- `statsmodels.stats.outliers_influence.OLSInfluence`

### 4.6 `helpers/plotting.py`

**Responsibility:** Generate all plots — both regression diagnostics and trajectory comparisons.

**Key functions — regression plots:**

| Function | Description |
|---|---|
| `plot_all_regression(result, diagnostics, config, group_name) -> List[Figure]` | Master function; produces all regression plots enabled in YAML |
| `plot_partial_regression(result, output_dir)` | Partial regression (added-variable) plots via `sm.graphics.plot_partregress_grid` |
| `plot_partial_residual(result, output_dir)` | Partial residual (CCPR) plots via `sm.graphics.plot_ccpr_grid` |
| `plot_residuals_vs_fitted(result, output_dir)` | Residuals vs. fitted values scatter |
| `plot_qq(result, output_dir)` | Normal Q-Q plot of residuals via `sm.graphics.qqplot` |
| `plot_scale_location(result, output_dir)` | Scale-location plot |
| `plot_cooks_distance(influence, output_dir)` | Cook's distance bar/stem plot with threshold line |
| `plot_leverage_vs_residuals(influence, output_dir)` | Leverage vs. standardized residuals with Cook's D contours |
| `plot_correlation_heatmap(X, output_dir)` | Heatmap of predictor correlations (seaborn) |
| `plot_pairplot(df, dep_var, indep_vars, output_dir)` | Pairwise scatter matrix (seaborn) |

**Key functions — trajectory plots:**

| Function | Description |
|---|---|
| `plot_trajectories(df, config, output_dir) -> List[Figure]` | Master function for trajectory comparison |
| `plot_grouped_trajectory(df, x, y, group_col, stats, output_dir)` | Mean curve ± std/IQR band per group, one figure per y_variable |
| `plot_trajectory_comparison_table(comparison_df, output_dir)` | Bar chart or table visualizing AUC, final_vs_initial, etc. per group |

---

## 5. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                       analysis_config.yaml                            │
│  (wandb source, filters, variable selection, analysis spec)          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │  run_statistical_        │
            │  analysis.py (launcher)  │
            └──────────┬───────────────┘
                       │
          ┌────────────┼────────────────────────┐
          ▼            ▼                        ▼
   ┌────────────┐ ┌──────────────────┐   ┌─────────────────────┐
   │ wandb_     │ │ data_            │   │ analysis dispatch:   │
   │ data_      │→│ processing.py    │──→│ regression OR        │
   │ loader.py  │ │ • reduce_history │   │ trajectory_comparison│
   │ (fetch &   │ │ • post_filters   │   └──────┬──────────────┘
   │  filter)   │ │ • derived_metrics│          │
   └────────────┘ │ • clean & encode │    ┌─────┴──────┐
                  └──────────────────┘    ▼            ▼
                                   ┌──────────┐  ┌──────────────────┐
                                   │REGRESSION│  │  TRAJECTORY      │
                                   │ PIPELINE │  │  PIPELINE        │
                                   ├──────────┤  ├──────────────────┤
                                   │fit_regr. │  │aggregate by group│
                                   │  ↓       │  │plot mean ± std   │
                                   │diagnose  │  │compute AUC,      │
                                   │  ↓       │  │final_vs_initial  │
                                   │plot      │  │comparison table  │
                                   │  ↓       │  └──────────────────┘
                                   │(per group│
                                   │ if strat)│
                                   │  ↓       │
                                   │compare   │
                                   └──────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  Console output  │
                              │  + outputs/ dir  │
                              │  (figures, CSV)  │
                              └──────────────────┘
```

---

## 6. Plasticity Experiment Data Model

### The problem: nested loops with task shifting

Each training run has an outer loop (task_idx) and inner loop (epochs within each task):
```
for task_idx in range(num_tasks):       # outer: shift dataset
    for epoch in range(epochs_per_task): # inner: train on current dataset
        # log: global_epoch = task_idx * epochs_per_task + epoch
        # log: task_idx, epoch_loss, epoch_accuracy, rank metrics, dead units, ...
```

`global_epoch` is the monotonically increasing master time index.

### Typical analysis workflow

1. **Load history** with `history_metrics: [global_epoch, task_idx, epoch_loss, epoch_accuracy, layer_0_num_dead_units, ...]`
2. **Reduce** with `last_per_group`: group by `task_idx`, keep row with max `global_epoch` → one row per (run, task_idx)
3. **Post-filter** to a task range of interest: `task_idx: {min: 100}`
4. **Analyze**:
   - **Regression**: "Do dead units predict loss at a given task, and is this stable across learner types?"
     - `dependent_variable: epoch_loss`, `independent_variables: [dead_units, rank_aurc]`, `group_by: learner__type`
   - **Trajectory**: "Which learner maintains low loss best across tasks?"
     - `x_axis: task_idx`, `y_variables: [epoch_loss]`, `group_by: learner__type`

---

## 7. Stratified Regression — Detail

When `group_by` is set, the regression pipeline runs as follows:

1. Split the DataFrame by `group_by` column(s)
2. For each group: fit the same model, compute diagnostics, generate plots (in a group-specific subfolder)
3. After all groups: produce a **comparison table**:

```
┌─────────────────┬────────┬────────────┬──────────────────────────────┐
│ Group           │   R²   │  F-stat    │  dead_units coeff (p-value)  │
├─────────────────┼────────┼────────────┼──────────────────────────────┤
│ backprop        │ 0.72   │ 34.2***    │  0.0043 (0.001)              │
│ rr_cbp2         │ 0.68   │ 28.7***    │  0.0038 (0.003)              │
│ rr_cbp_e_2      │ 0.71   │ 31.1***    │  0.0041 (0.002)              │
└─────────────────┴────────┴────────────┴──────────────────────────────┘
→ Interpretation: dead_units → loss relationship is consistent across all learner types
  (coefficient sign, magnitude, and significance are stable)
```

Without `group_by`, a single pooled regression is run.

---

## 8. Trajectory Comparison — Detail

Compares how a metric evolves over the progression variable (`task_idx`) across different groups.

**Plots produced:**
- One figure per `y_variable` (e.g., `epoch_loss`, `epoch_accuracy`)
- Each group gets a colored mean curve with shaded ± 1 std (or IQR) band
- X-axis: `task_idx`, Y-axis: the metric
- Legend: group labels

**Quantitative comparisons (printed as table):**

| Comparison | Description |
|---|---|
| `auc` | Area under the curve (trapezoidal rule). Lower loss AUC = better plasticity. |
| `final_vs_initial` | Ratio (or difference) of the metric at the last task vs. the first task. |
| `max_degradation` | Maximum metric value relative to best achieved. Captures worst-case plasticity loss. |
| `recovery_events` | Number of times the metric improves after a period of degradation. |

**Key distinction:** "maintaining low loss" (low AUC, low final_vs_initial ratio) is different from "having the lowest loss at any point" (min of the curve). The comparison table captures both.

---

## 9. Regression Diagnostics — Full Inventory

### 9.1 Goodness-of-Fit Measures

| Metric | Formula / Source | Interpretation |
|---|---|---|
| R² | `1 - SSE/SST` | Proportion of variance explained |
| Adjusted R² | `1 - (1-R²)(n-1)/(n-k-1)` | Penalizes extra predictors |
| SSR (regression) | `Σ(ŷᵢ - ȳ)²` | Variation explained by model |
| SSE (residual) | `Σ(yᵢ - ŷᵢ)²` | Unexplained variation |
| SST (total) | `Σ(yᵢ - ȳ)²` | Total variation |
| AIC | Akaike Information Criterion | Model selection (lower = better) |
| BIC | Bayesian Information Criterion | Model selection (penalizes complexity more) |
| F-statistic | `(SSR/k) / (SSE/(n-k-1))` | Overall model significance |
| Log-likelihood | from fitted model | Basis for AIC/BIC |

### 9.2 Coefficient Diagnostics

| Metric | Description |
|---|---|
| Coefficients (β̂) | Estimated regression coefficients |
| Standard Errors | Uncertainty in coefficient estimates |
| t-statistics | `β̂ / SE(β̂)` |
| p-values | Significance of each coefficient |
| 95% Confidence Intervals | Range for true coefficient |
| Standardized Coefficients | βᵢ × (σₓᵢ / σᵧ) — relative importance |

### 9.3 Residual Diagnostics

| Test | What it checks | Null hypothesis |
|---|---|---|
| Durbin-Watson | Autocorrelation in residuals | No autocorrelation (stat ≈ 2) |
| Jarque-Bera | Normality of residuals | Residuals are normally distributed |
| Breusch-Pagan | Heteroscedasticity | Constant variance |
| Goldfeld-Quandt | Heteroscedasticity (alternative) | Constant variance |

### 9.4 Multicollinearity Diagnostics

| Metric | Threshold | Action |
|---|---|---|
| Condition Number | > 30 suggests issues | Consider removing correlated predictors |
| VIF per predictor | > 5 moderate, > 10 severe | Remove or combine correlated predictors |

### 9.5 Influence Diagnostics

| Metric | Description | Threshold |
|---|---|---|
| Cook's Distance | Overall influence of each observation | > 4/n or > 1 |
| Leverage (hat values) | How extreme an observation's predictors are | > 2(k+1)/n |
| DFFITS | Change in fitted value when observation removed | > 2√(k/n) |

---

## 10. Plots — Full Inventory

### Regression Plots

| # | Plot | Purpose | Implementation |
|---|---|---|---|
| 1 | Partial Regression (Added-Variable) | Effect of each predictor after controlling for others | `sm.graphics.plot_partregress_grid` |
| 2 | Partial Residual (CCPR) | Predictor–response relationship adjusted for other predictors | `sm.graphics.plot_ccpr_grid` |
| 3 | Residuals vs. Fitted | Detect non-linearity, heteroscedasticity | Custom matplotlib scatter |
| 4 | Normal Q-Q | Check normality of residuals | `sm.graphics.qqplot` |
| 5 | Scale-Location | Check homoscedasticity | Custom matplotlib |
| 6 | Cook's Distance | Identify influential observations | Stem/bar plot |
| 7 | Leverage vs. Residuals | High-leverage + high-residual outliers | Custom with Cook's D contours |
| 8 | Correlation Heatmap | Visualize multicollinearity | `seaborn.heatmap` |
| 9 | Pairplot | Pairwise relationships | `seaborn.pairplot` |

### Trajectory Plots

| # | Plot | Purpose | Implementation |
|---|---|---|---|
| 10 | Grouped Trajectory | Mean ± std/IQR curves per group over task_idx | Custom matplotlib with fill_between |
| 11 | Comparison Bar Chart | AUC / final_vs_initial / max_degradation per group | matplotlib bar chart |

---

## 11. Dependencies

New packages needed beyond current `requirements.txt`:

| Package | Purpose |
|---|---|
| `statsmodels` | Core regression, diagnostics, partial regression/residual plots |
| `matplotlib` | Base plotting (likely already installed with torch) |
| `seaborn` | Heatmaps, pairplots, trajectory styling |
| `pandas` | DataFrame operations (likely already installed) |
| `scikit-learn` | Ridge/Lasso regression (likely already installed via stable-baselines3) |
| `wandb` | W&B API (already used in uploader) |
| `pyyaml` | YAML parsing (already in requirements.txt) |
| `tabulate` | Pretty-printing tables to console (optional) |

---

## 12. W&B Filtering — Implementation Detail

The `wandb.Api().runs()` method accepts a MongoDB-style `filters` dict:

```python
# Server-side filters (fast, done by W&B API)
api_filters = {}

# State filter
if filters.get("state"):
    states = filters["state"] if isinstance(filters["state"], list) else [filters["state"]]
    api_filters["state"] = {"$in": states}

# Tag filter (runs must have ALL tags)
if filters.get("tags"):
    api_filters["tags"] = {"$all": filters["tags"]}

# Config value filters (dot-notation)
if filters.get("config_filters"):
    for key, value in filters["config_filters"].items():
        wandb_key = f"config.{key}"
        if isinstance(value, list):
            api_filters[wandb_key] = {"$in": value}
        else:
            api_filters[wandb_key] = value

# Date filters
if filters.get("created_after"):
    api_filters["created_at"] = {"$gte": filters["created_after"]}
```

Post-fetch filtering (Python-side) handles:
- `name_regex` — `re.search(pattern, run.name)`
- `summary_filters` — min/max range checks on `run.summary` values
- `max_runs` — truncate the list

---

## 13. Error Handling & Edge Cases

| Scenario | Handling |
|---|---|
| No runs match filters | Print clear message with filter summary, exit gracefully |
| Missing metric in some runs | Drop those runs, print warning with count |
| All-NaN column | Drop column, warn user |
| Categorical variable with single level | Drop it, warn user |
| Singular design matrix | Report condition number, suggest removing correlated variables |
| < k+1 observations for k predictors | Error with clear message about insufficient data |
| W&B API timeout/rate limit | Retry with exponential backoff (up to 3 retries) |
| Non-numeric config value used as predictor | Attempt one-hot encoding; if not possible, error with clear message |
| Group in group_by has too few observations | Skip group with warning, continue with others |
| history_metrics requested but run has no history | Skip run with warning |
| last_per_group but group_by column missing | Error with clear message |

---

## 14. Example Usage

### Regression: dead units vs loss, stratified by learner type
```bash
cd analysis
python run_statistical_analysis.py --config cfg/analysis_config.yaml
```

### Trajectory comparison: loss over tasks by learner type
```bash
python run_statistical_analysis.py --config cfg/trajectory_config.yaml
```

### Override entity/project on CLI:
```bash
python run_statistical_analysis.py --config cfg/analysis_config.yaml \
    --entity my-team --project my-experiment
```

### Dry run (fetch data, print DataFrame shape & head, skip analysis):
```bash
python run_statistical_analysis.py --config cfg/analysis_config.yaml --dry-run
```

---

## 15. Implementation Order

| Phase | Files | Description |
|---|---|---|
| **Phase 1** | `helpers/__init__.py`, `helpers/wandb_data_loader.py` | W&B data loading: fetch, filter, extract |
| **Phase 2** | `helpers/data_processing.py` | DataFrame assembly, panel reduction, transforms, encoding |
| **Phase 3** | `helpers/regression_analysis.py` | Regression fitting + grouped regression + RegressionResult wrapper |
| **Phase 4** | `helpers/regression_diagnostics.py` | All diagnostic computations |
| **Phase 5** | `helpers/plotting.py` | All plots: regression diagnostics + trajectory comparison |
| **Phase 6** | `run_statistical_analysis.py`, `cfg/analysis_config.yaml` | Launcher and example config |
| **Phase 7** | Testing & documentation | End-to-end test, README |

---

## 16. Design Decisions & Rationale

1. **YAML-driven, not CLI-driven**: Complex analysis specs are far more readable and reproducible in YAML than CLI flags.

2. **Two analysis types as first-class citizens**: `regression` (with optional stratification) and `trajectory_comparison` serve the two primary research patterns — mechanism correlation and plasticity comparison.

3. **Panel data as the default**: The `last_per_group` sampling + `history_post_filters` pipeline naturally handles the (run × task_idx) panel structure of plasticity experiments.

4. **Stratified regression over single-value-per-run**: Rather than collapsing runs to single numbers, run the same regression within each hyperparameter group. This directly answers "is the correlation robust?" without losing information.

5. **statsmodels as the core engine**: Provides native OLS fitting with full diagnostics in a single coherent API. sklearn is used only for Ridge/Lasso.

6. **Two-stage W&B filtering**: Server-side API filters reduce data transfer; Python-side post-filtering handles complex logic.

7. **Helpers in a subfolder**: Keeps the analysis directory clean while making individual components testable and reusable.

8. **Uniform `RegressionResult` wrapper**: Allows diagnostics and plotting code to work identically regardless of backend.

9. **Column name sanitization**: W&B config keys use dots (`learner.step_size`), but statsmodels and pandas need clean identifiers. We convert to double underscores early (`learner__step_size`).

10. **Derived metrics are optional**: Per-run trajectory summaries (AUC, slope, etc.) are available when needed but the primary workflow operates on the full panel data.
