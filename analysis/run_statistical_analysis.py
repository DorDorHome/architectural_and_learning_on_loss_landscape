#!/usr/bin/env python3
"""
Statistical Analysis Launcher for W&B Runs

Reads a YAML config file and orchestrates the full analysis pipeline:
  1. Fetch & filter W&B runs
  2. Build a pandas DataFrame (cross-sectional or panel)
  3. Apply history reduction, post-filters, transforms
  4. Dispatch to regression or trajectory analysis
  5. Generate diagnostics and plots
  6. Save outputs

Output files (saved under analysis/outputs/ by default):
  - Regression: PNG plots (partial_regression, partial_residual, qq_plot, etc.),
    group_comparison.csv, coefficients_<group>.csv per learner type
  - Trajectory: trajectory_<metric>.png, trajectory_comparison.png

Usage:
    python run_statistical_analysis.py --config cfg/analysis_config.yaml
    python run_statistical_analysis.py --config cfg/analysis_config.yaml --dry-run
    python run_statistical_analysis.py --config cfg/analysis_config.yaml --entity my-team --project my-project
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd

# Ensure the analysis directory is importable
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from helpers.wandb_data_loader import load_runs, build_dataframe
from helpers.data_processing import (
    sanitize_column_names,
    reduce_history,
    apply_history_post_filters,
    compute_derived_metrics,
    clean_dataframe,
    encode_categoricals,
    sanitize_key,
)
from helpers.regression_analysis import (
    run_grouped_regression,
    compare_group_results,
)
from helpers.regression_diagnostics import (
    compute_diagnostics,
    format_diagnostics_report,
)
from helpers.plotting import (
    plot_all_regression,
    plot_all_trajectory,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML config as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with CLI arguments."""
    if args.entity:
        cfg.setdefault("wandb", {})["entity"] = args.entity
    if args.project:
        cfg.setdefault("wandb", {})["project"] = args.project
    return cfg


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_load_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Stage 1: Fetch W&B runs and build DataFrame."""
    wandb_cfg = cfg.get("wandb", {})
    entity = wandb_cfg.get("entity", "")
    project = wandb_cfg.get("project", "")

    if not entity or not project:
        raise ValueError(
            "Both 'wandb.entity' and 'wandb.project' must be set in the config "
            "or via --entity / --project CLI flags."
        )

    filters = cfg.get("filters", {})
    variables = cfg.get("variables", {})

    print(f"\n{'=' * 60}")
    print(f"Stage 1: Loading data from W&B")
    print(f"  Entity:  {entity}")
    print(f"  Project: {project}")
    print(f"{'=' * 60}")

    # Fetch and filter runs
    runs = load_runs(entity, project, filters)

    if not runs:
        print("\n  No runs matched the filters. Exiting.")
        sys.exit(0)

    # Build DataFrame
    df = build_dataframe(runs, variables)

    if df.empty:
        print("\n  DataFrame is empty after extraction. Exiting.")
        sys.exit(0)

    return df


def stage_process_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Stage 2: Sanitize, reduce, filter, clean, encode."""
    variables = cfg.get("variables", {})

    print(f"\n{'=' * 60}")
    print(f"Stage 2: Processing data")
    print(f"{'=' * 60}")

    # Sanitize column names (dots → __)
    df = sanitize_column_names(df)
    print(f"  Columns after sanitization: {list(df.columns)}")

    # History reduction
    history_metrics = variables.get("history_metrics", [])
    if history_metrics:
        sampling = variables.get("history_sampling", {})
        if sampling.get("method", "all") != "all":
            print(f"\n  Applying history sampling: method={sampling.get('method')}")
            df = reduce_history(df, sampling)

        # Post-filters
        post_filters = variables.get("history_post_filters", {})
        if post_filters:
            print(f"\n  Applying post-filters: {post_filters}")
            df = apply_history_post_filters(df, post_filters)

    # Derived metrics (optional)
    derived = variables.get("derived_metrics", [])
    if derived:
        print(f"\n  Computing {len(derived)} derived metrics...")
        derived_df = compute_derived_metrics(df, derived)
        if not derived_df.empty:
            print(f"  Derived metrics DataFrame: {derived_df.shape}")
            # We keep both the panel df and derived df; the launcher will
            # decide which to use based on analysis type.
            df.attrs["derived_df"] = derived_df

    print(f"\n  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string())

    return df


def stage_regression(df: pd.DataFrame, cfg: Dict[str, Any]):
    """Stage 3a: Run regression analysis pipeline."""
    analysis_cfg = cfg.get("analysis", {})
    reg_cfg = analysis_cfg.get("regression", {})
    output_cfg = analysis_cfg.get("output", {})

    # Resolve output_dir relative to analysis directory
    output_dir = output_cfg.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = str(ANALYSIS_DIR / output_dir)
    output_cfg["output_dir"] = output_dir

    print(f"\n{'=' * 60}")
    print(f"Stage 3: Regression Analysis")
    print(f"  Dependent:    {reg_cfg.get('dependent_variable')}")
    print(f"  Independent:  {reg_cfg.get('independent_variables')}")
    print(f"  Type:         {reg_cfg.get('regression_type', 'ols')}")
    print(f"  Group by:     {reg_cfg.get('group_by', 'None (pooled)')}")
    print(f"{'=' * 60}")

    # Determine required columns for cleaning
    dep_var = sanitize_key(reg_cfg["dependent_variable"])
    indep_vars = [sanitize_key(v) for v in reg_cfg["independent_variables"]]
    required = [dep_var] + indep_vars

    # Identify categorical columns among independent variables
    cat_cols = []
    for col in indep_vars:
        if col in df.columns and df[col].dtype in ("object", "category", "bool"):
            cat_cols.append(col)

    # Encode categoricals BEFORE cleaning (so we don't lose rows unnecessarily)
    if cat_cols:
        print(f"\n  Encoding categorical variables: {cat_cols}")
        df, new_cols = encode_categoricals(df, cat_cols)
        # Replace the categorical variable names in indep_vars with the encoded columns
        for cat_col in cat_cols:
            indep_vars.remove(cat_col)
            encoded = [c for c in new_cols if c.startswith(cat_col)]
            indep_vars.extend(encoded)
        # Update the config so grouped regression uses the encoded columns
        reg_cfg["independent_variables"] = [
            v for v in reg_cfg["independent_variables"]
            if sanitize_key(v) not in cat_cols
        ] + [c for c in new_cols]
        print(f"  Updated independent variables: {reg_cfg['independent_variables']}")

    # Clean DataFrame
    # Don't require group_by column to be non-NaN
    group_by_col = reg_cfg.get("group_by")
    df = clean_dataframe(df, required)

    if df.empty:
        print("\n  DataFrame is empty after cleaning. Cannot run regression.")
        return

    # Run grouped regression
    results = run_grouped_regression(df, reg_cfg)

    # Diagnostics and plots per group
    diag_cfg = reg_cfg.get("diagnostics", {})
    plots_cfg = reg_cfg.get("plots", {})
    all_diagnostics = {}

    for label, result in results.items():
        print(f"\n{'~' * 60}")
        print(f"  Results for: {label}")
        print(f"{'~' * 60}")

        # Print statsmodels summary
        if output_cfg.get("print_summary", True) and result.backend == "statsmodels":
            print(result.model.summary())

        # Compute diagnostics
        diagnostics = compute_diagnostics(result, diag_cfg)
        all_diagnostics[label] = diagnostics

        # Print diagnostics report
        report = format_diagnostics_report(diagnostics, group_name=label)
        print(report)

        # Generate plots
        if output_cfg.get("save_figures", True):
            plot_all_regression(result, diagnostics, plots_cfg, output_cfg, group_name=label)

    # Group comparison
    compare = reg_cfg.get("compare_groups", True)
    if compare and len(results) > 1:
        print(f"\n{'=' * 60}")
        print("Group Comparison")
        print(f"{'=' * 60}")
        comp_df = compare_group_results(results)
        print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        if output_cfg.get("save_summary_csv", True):
            csv_path = Path(output_dir) / "group_comparison.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            comp_df.to_csv(str(csv_path), index=False)
            print(f"\n  Saved: {csv_path}")

    # Save coefficient tables
    if output_cfg.get("save_summary_csv", True):
        for label, diag in all_diagnostics.items():
            coeff_df = diag.get("coefficients")
            if coeff_df is not None and isinstance(coeff_df, pd.DataFrame) and not coeff_df.empty:
                csv_path = Path(output_dir) / f"coefficients_{label}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                coeff_df.to_csv(str(csv_path), index=False)
                print(f"  Saved: {csv_path}")


def stage_trajectory(df: pd.DataFrame, cfg: Dict[str, Any]):
    """Stage 3b: Run trajectory comparison pipeline."""
    analysis_cfg = cfg.get("analysis", {})
    traj_cfg = analysis_cfg.get("trajectory", {})
    output_cfg = analysis_cfg.get("output", {})

    # Resolve output_dir
    output_dir = output_cfg.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = str(ANALYSIS_DIR / output_dir)
    output_cfg["output_dir"] = output_dir

    print(f"\n{'=' * 60}")
    print(f"Stage 3: Trajectory Comparison")
    print(f"  X-axis:    {traj_cfg.get('x_axis')}")
    print(f"  Y-vars:    {traj_cfg.get('y_variables')}")
    print(f"  Group by:  {traj_cfg.get('group_by')}")
    print(f"{'=' * 60}")

    figures = plot_all_trajectory(df, traj_cfg, output_cfg)

    # Save trajectory comparison table as CSV if requested
    if output_cfg.get("save_summary_csv", True):
        from helpers.plotting import compute_trajectory_comparisons
        from helpers.data_processing import sanitize_key

        x_axis = traj_cfg.get("x_axis", "task_idx")
        y_variables = traj_cfg.get("y_variables", [])
        group_by = traj_cfg.get("group_by")
        comparisons = traj_cfg.get("comparisons", [])

        x_col = sanitize_key(x_axis) if sanitize_key(x_axis) in df.columns else x_axis
        if isinstance(group_by, str):
            group_col = sanitize_key(group_by) if sanitize_key(group_by) in df.columns else group_by
        else:
            group_col = group_by[0] if group_by else None

        if comparisons and group_col and group_col in df.columns:
            comp_df = compute_trajectory_comparisons(df, x_col, y_variables, group_col, comparisons)
            if comp_df is not None and not comp_df.empty:
                csv_path = Path(output_dir) / "trajectory_comparison.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                comp_df.to_csv(str(csv_path), index=False)
                print(f"\n  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis of W&B Runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regression analysis
  python run_statistical_analysis.py --config cfg/analysis_config.yaml

  # Trajectory comparison (loss/accuracy over task_idx by learner type)
  python run_statistical_analysis.py --config cfg/trajectory_config.yaml

  # Override entity/project
  python run_statistical_analysis.py --config cfg/analysis_config.yaml \\
      --entity my-team --project my-project

  # Dry run (load data only, print summary)
  python run_statistical_analysis.py --config cfg/analysis_config.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML analysis config file",
    )
    parser.add_argument("--entity", help="Override wandb.entity")
    parser.add_argument("--project", help="Override wandb.project")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print summary only; skip analysis",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Stage 1: Load data
    df = stage_load_data(cfg)

    # Stage 2: Process data
    df = stage_process_data(df, cfg)

    # Dry run — stop here
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Data loaded and processed. Skipping analysis.")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Descriptive statistics:")
        print(df.describe().to_string())
        print(f"{'=' * 60}")
        return

    # Stage 3: Dispatch analysis
    analysis_type = cfg.get("analysis", {}).get("type", "regression")

    if analysis_type == "regression":
        stage_regression(df, cfg)
    elif analysis_type == "trajectory_comparison":
        stage_trajectory(df, cfg)
    else:
        raise ValueError(
            f"Unknown analysis type: {analysis_type!r}. "
            f"Choose 'regression' or 'trajectory_comparison'."
        )

    print(f"\n{'=' * 60}")
    print("Analysis complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
