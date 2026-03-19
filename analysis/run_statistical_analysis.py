#!/usr/bin/env python3
"""
Statistical Analysis Pipeline

Reads a YAML config (shared with plotting pipeline) and performs statistical analysis:
  1. Fetch & filter W&B runs (using shared pipeline logic)
  2. Build a pandas DataFrame with computed metrics (forward_loss, weight_norm)
  3. Run regression or trajectory analysis
  4. Save outputs (plots, CSVs)

Usage:
    python run_statistical_analysis.py --config cfg/plotting_config.yaml
    python run_statistical_analysis.py --config cfg/plotting_config.yaml --dry-run
"""

import argparse
import os
import sys
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import helpers from the shared pipeline
from analysis.helpers.config_utils import (
    load_config,
    resolve_targets,
    target_diff_label,
    flatten_target_config,
    get_nested_config,
    get_target_label,
    determine_output_dir
)
from analysis.helpers.wandb_utils import (
    fetch_runs_for_target,
    discover_weight_norm_keys,
    extract_run_history
)
from analysis.helpers.data_processing import (
    apply_history_sampling,
    compute_weight_norm_column,
    compute_forward_loss,
    filter_outliers,
    audit_nan_per_run,
    build_pipeline_summary,
    determine_required_base_metrics,
    sanitize_column_names,
    sanitize_key,
    apply_variable_transformations
)
from analysis.helpers.regression_analysis import (
    run_grouped_regression,
    compare_group_results,
)
from analysis.helpers.regression_diagnostics import (
    compute_diagnostics,
    format_diagnostics_report,
)
from analysis.helpers.plotting import (
    plot_all_regression,
    plot_all_trajectory,
    compute_trajectory_comparisons
)


# ---------------------------------------------------------------------------
# Analysis Stages
# ---------------------------------------------------------------------------

def stage_regression(df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str):
    """Run regression analysis pipeline."""
    analysis_cfg = cfg.get("analysis", {})
    reg_cfg = analysis_cfg.get("regression", {})
    output_cfg = analysis_cfg.get("output", {})
    
    # Ensure output_dir in config matches the actual output directory
    output_cfg["output_dir"] = output_dir

    print(f"\n{'=' * 60}")
    print(f"Stage: Regression Analysis")
    print(f"  Dependent:    {reg_cfg.get('dependent_variable')}")
    print(f"  Independent:  {reg_cfg.get('independent_variables')}")
    print(f"  Type:         {reg_cfg.get('regression_type', 'ols')}")
    print(f"  Group by:     {reg_cfg.get('group_by', 'None (pooled)')}")
    print(f"{'=' * 60}")

    # Sanitize column names in the dataframe to ensure compatibility
    # (The pipeline already produces sanitized names like learner__type, but let's be safe)
    # Note: sanitize_column_names replaces '.' with '__'
    # df = sanitize_column_names(df) 
    # Actually, the pipeline produces flat keys like "learner.type" which are then
    # converted to "learner__type" in extract_run_history.
    # But let's ensure the config variables match the dataframe columns.
    
    dep_var = reg_cfg["dependent_variable"]
    indep_vars = reg_cfg["independent_variables"]
    
    # Check if variables exist
    missing = []
    if dep_var not in df.columns:
        # Try sanitized version
        sanitized = sanitize_key(dep_var)
        if sanitized in df.columns:
            reg_cfg["dependent_variable"] = sanitized
            dep_var = sanitized
        else:
            missing.append(dep_var)
            
    new_indep = []
    for var in indep_vars:
        if var in df.columns:
            new_indep.append(var)
        else:
            sanitized = sanitize_key(var)
            if sanitized in df.columns:
                new_indep.append(sanitized)
            else:
                missing.append(var)
    reg_cfg["independent_variables"] = new_indep
    
    if missing:
        print(f"Error: Missing variables in DataFrame: {missing}")
        print(f"Available columns: {sorted(list(df.columns))}")
        return

    # Run grouped regression
    
    # Check for univariate toggle
    run_univariate = reg_cfg.get("univariate", False)
    
    # If univariate, we loop over each variable. If not, we run once with all variables.
    variable_sets = [[v] for v in indep_vars] if run_univariate else [indep_vars]

    original_output_dir = output_dir
    original_indep_vars = reg_cfg["independent_variables"]

    for current_vars in variable_sets:
        if run_univariate:
            var_name = current_vars[0]
            print(f"\n{'='*20} Univariate Analysis: {var_name} {'='*20}")
            # Temporarily update config to only use this variable
            reg_cfg["independent_variables"] = current_vars
            
            # Update output directory to keep plots separate
            # Sanitize variable name for directory
            safe_var_name = sanitize_key(var_name)
            current_output_dir = os.path.join(original_output_dir, safe_var_name)
            os.makedirs(current_output_dir, exist_ok=True)
            output_cfg["output_dir"] = current_output_dir
        else:
            current_output_dir = original_output_dir

        try:
            results = run_grouped_regression(df, reg_cfg)
        except Exception as e:
            print(f"Regression failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Diagnostics and plots per group
        diag_cfg = reg_cfg.get("diagnostics", {})
        plots_cfg = reg_cfg.get("plots", {})
        all_diagnostics = {}

        for label, result in results.items():
            print(f"\n{'~' * 60}")
            print(f"  Results for Group: {label}")
            if run_univariate:
                print(f"  Variable: {current_vars[0]}")
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
            if run_univariate:
                print(f"  Variable: {current_vars[0]}")
            print(f"{'=' * 60}")
            comp_df = compare_group_results(results)
            print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

            if output_cfg.get("save_summary_csv", True):
                csv_path = Path(current_output_dir) / "group_comparison.csv"
                comp_df.to_csv(str(csv_path), index=False)
                print(f"\n  Saved: {csv_path}")

        # Save coefficient tables
        if output_cfg.get("save_summary_csv", True):
            for label, diag in all_diagnostics.items():
                coeff_df = diag.get("coefficients")
                if coeff_df is not None and isinstance(coeff_df, pd.DataFrame) and not coeff_df.empty:
                    csv_path = Path(current_output_dir) / f"coefficients_{label}.csv"
                    coeff_df.to_csv(str(csv_path), index=False)
                    print(f"  Saved: {csv_path}")

    # Restore original config
    reg_cfg["independent_variables"] = original_indep_vars
    output_cfg["output_dir"] = original_output_dir


def stage_trajectory(df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str):
    """Run trajectory comparison pipeline."""
    analysis_cfg = cfg.get("analysis", {})
    traj_cfg = analysis_cfg.get("trajectory", {})
    output_cfg = analysis_cfg.get("output", {})
    
    output_cfg["output_dir"] = output_dir

    print(f"\n{'=' * 60}")
    print(f"Stage: Trajectory Comparison")
    print(f"  X-axis:    {traj_cfg.get('x_axis')}")
    print(f"  Y-vars:    {traj_cfg.get('y_variables')}")
    print(f"  Group by:  {traj_cfg.get('group_by')}")
    print(f"{'=' * 60}")

    # Check variables
    x_axis = traj_cfg.get("x_axis", "task_idx")
    y_variables = traj_cfg.get("y_variables", [])
    
    # Sanitize config keys if needed
    if x_axis not in df.columns and sanitize_key(x_axis) in df.columns:
        traj_cfg["x_axis"] = sanitize_key(x_axis)
        
    new_y = []
    for y in y_variables:
        if y in df.columns:
            new_y.append(y)
        elif sanitize_key(y) in df.columns:
            new_y.append(sanitize_key(y))
        else:
            print(f"Warning: Y-variable '{y}' not found in DataFrame.")
    traj_cfg["y_variables"] = new_y

    plot_all_trajectory(df, traj_cfg, output_cfg)

    # Save comparison table
    if output_cfg.get("save_summary_csv", True):
        group_by = traj_cfg.get("group_by")
        comparisons = traj_cfg.get("comparisons", [])
        
        x_col = traj_cfg["x_axis"]
        
        if isinstance(group_by, str):
            group_col = group_by if group_by in df.columns else sanitize_key(group_by)
        else:
            group_col = group_by[0] if group_by else None

        if comparisons and group_col and group_col in df.columns:
            comp_df = compute_trajectory_comparisons(df, x_col, traj_cfg["y_variables"], group_col, comparisons)
            if comp_df is not None and not comp_df.empty:
                csv_path = Path(output_dir) / "trajectory_comparison.csv"
                comp_df.to_csv(str(csv_path), index=False)
                print(f"\n  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_analysis_pipeline(cfg: Dict[str, Any], dry_run: bool = False, config_path: Optional[str] = None) -> None:
    """Execute the full statistical analysis pipeline."""
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]
    
    base_filters = cfg["filters"]
    targets = resolve_targets(cfg)
    metric_cfg = cfg["metric_computation"]
    
    comparison_name = cfg.get("comparison_name")
    defaults = cfg.get("target_defaults", {})
    
    # Determine base metrics to fetch (including those for analysis)
    base_metrics = determine_required_base_metrics(cfg)
    print(f"Base metrics to fetch: {base_metrics}")

    weight_norm_cfg = metric_cfg.get("weight_norm", {})
    source_pattern = weight_norm_cfg.get("source_pattern", "_mean_abs_weight")
    aggregation = weight_norm_cfg.get("aggregation", "weight_norm_mean")

    # Global variable scales
    global_scales = cfg.get("global_settings", {}).get("variable_scales", {})

    all_target_dfs = []
    target_inventory: List[Dict[str, Any]] = []

    # --- Data Loading Loop (Same as plotting pipeline) ---
    for target_idx, target in enumerate(targets):
        flat_label = flatten_target_config(target)
        diff_label = target_diff_label(target, defaults)
        learner_type = get_nested_config(target, "learner.type", "unknown")
        net_type = get_nested_config(target, "net.type", "unknown")

        inv: Dict[str, Any] = {
            "index": target_idx,
            "diff_label": diff_label,
            "full_flat": flat_label,
            "runs_found": 0,
            "run_names": [],
            "run_ids": [],
            "n_rows_raw": 0,
            "n_rows_sampled": 0,
            "forward_loss_valid": 0,
            "target_label": None,
        }

        print(f"\n{'='*60}")
        print(f"Target {target_idx + 1}/{len(targets)}: {diff_label}")
        print(f"  Full: {net_type} / {learner_type} — {flat_label}")
        print(f"{'='*60}")

        # Fetch runs
        runs = fetch_runs_for_target(entity, project, base_filters, target)
        inv["runs_found"] = len(runs)

        if not runs:
            warnings.warn(f"No runs found for target {flat_label}; skipping.")
            target_inventory.append(inv)
            continue

        inv["run_names"] = [r.name for r in runs]
        inv["run_ids"] = [r.id for r in runs]

        if dry_run:
            print(f"  [DRY RUN] Would process {len(runs)} runs")
            target_inventory.append(inv)
            continue

        # Discover weight_norm keys
        wn_keys = discover_weight_norm_keys(runs[0], source_pattern)
        
        # Extract history
        run_dfs = []
        for i, run in enumerate(runs):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Extracting run {i + 1}/{len(runs)}: {run.name}")
            hist = extract_run_history(run, base_metrics + wn_keys)
            if hist.empty:
                continue
            hist["run_id"] = run.id
            hist["run_name"] = run.name

            # Add config columns (flattened)
            for key, val in flat_label.items():
                col_name = key.replace(".", "__")
                hist[col_name] = val

            hist["_target_label"] = get_target_label(target, defaults)
            run_dfs.append(hist)

        if not run_dfs:
            warnings.warn(f"No history data for target {flat_label}; skipping.")
            target_inventory.append(inv)
            continue

        target_df = pd.concat(run_dfs, ignore_index=True)
        n_runs = target_df["run_id"].nunique()
        inv["n_rows_raw"] = len(target_df)
        print(f"  Raw history: {len(target_df)} rows from {n_runs} unique run(s)")

        target_df["_target_label"] = target_df["_target_label"] + f"\n[{n_runs} run{'s' if n_runs != 1 else ''}]"
        inv["target_label"] = target_df["_target_label"].iloc[0]

        # History sampling
        target_df = apply_history_sampling(target_df, metric_cfg["history_sampling"])
        inv["n_rows_sampled"] = len(target_df)

        # Compute weight norm
        target_df = compute_weight_norm_column(target_df, wn_keys, aggregation)

        # Compute forward loss
        target_df = compute_forward_loss(target_df, metric_cfg["forward_loss"])
        inv["forward_loss_valid"] = int(target_df["forward_loss"].notna().sum())

        # Outlier filtering
        if "per_target_group_filtering" in metric_cfg:
            target_df = filter_outliers(target_df, metric_cfg["per_target_group_filtering"])
            inv["n_rows_sampled"] = len(target_df)
        # Legacy support
        elif "outlier_filtering" in metric_cfg:
            target_df = filter_outliers(target_df, metric_cfg["outlier_filtering"])
            inv["n_rows_sampled"] = len(target_df)

        all_target_dfs.append(target_df)
        target_inventory.append(inv)

    if dry_run:
        print("\n[DRY RUN] Pipeline complete — no data was processed.")
        return

    if not all_target_dfs:
        print("No data collected from any target. Exiting.")
        return

    # Combine all targets
    combined = pd.concat(all_target_dfs, ignore_index=True)
    print(f"\nCombined DataFrame: {combined.shape[0]} rows x {combined.shape[1]} columns")

    # Determine output directory
    output_dir = determine_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # NaN audit & optional filtering
    combined, dropped_ids, nan_details = audit_nan_per_run(
        combined, base_metrics, base_filters.get("drop_nan_threshold"),
    )

    # Save summary
    summary = build_pipeline_summary(
        target_inventory, base_metrics,
        base_filters.get("drop_nan_threshold"),
        None, combined, nan_details, dropped_ids,
    )
    summary_fname = "analysis_pipeline_summary.txt"
    if comparison_name:
        summary_fname = f"{comparison_name}_{summary_fname}"
    summary_path = os.path.join(output_dir, summary_fname)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Pipeline summary saved: {summary_path}")

    # Copy config
    if config_path and os.path.exists(config_path):
        cfg_fname = os.path.basename(config_path)
        if comparison_name:
            cfg_fname = f"{comparison_name}_{cfg_fname}"
        cfg_dest = os.path.join(output_dir, cfg_fname)
        shutil.copy2(config_path, cfg_dest)

    if combined.empty:
        print("No data remaining after NaN filtering. Exiting.")
        return

    # --- Apply Global Transforms for Analysis ---
    # This renames columns (e.g. "epoch_loss" -> "epoch_loss (log)") and applies transform
    if global_scales:
        print(f"\nApplying global variable transformations: {global_scales}")
        combined = apply_variable_transformations(combined, global_scales)
        
        # Update config to match new column names
        analysis_cfg = cfg.get("analysis", {})
        
        # Update Regression Config
        reg_cfg = analysis_cfg.get("regression", {})
        if reg_cfg:
            # Dependent variable
            dep = reg_cfg.get("dependent_variable")
            if dep in global_scales:
                new_name = f"{dep} ({global_scales[dep]})"
                print(f"  Updating regression dependent variable: {dep} -> {new_name}")
                reg_cfg["dependent_variable"] = new_name
            
            # Independent variables
            indep = reg_cfg.get("independent_variables", [])
            new_indep = []
            for var in indep:
                if var in global_scales:
                    new_name = f"{var} ({global_scales[var]})"
                    print(f"  Updating regression independent variable: {var} -> {new_name}")
                    new_indep.append(new_name)
                else:
                    new_indep.append(var)
            reg_cfg["independent_variables"] = new_indep

        # Update Trajectory Config
        traj_cfg = analysis_cfg.get("trajectory", {})
        if traj_cfg:
            y_vars = traj_cfg.get("y_variables", [])
            new_y = []
            for var in y_vars:
                if var in global_scales:
                    new_name = f"{var} ({global_scales[var]})"
                    print(f"  Updating trajectory y-variable: {var} -> {new_name}")
                    new_y.append(new_name)
                else:
                    new_y.append(var)
            traj_cfg["y_variables"] = new_y

    # --- Analysis Dispatch ---
    analysis_type = cfg.get("analysis", {}).get("type", "regression")

    if analysis_type == "regression":
        stage_regression(combined, cfg, output_dir)
    elif analysis_type == "trajectory_comparison":
        stage_trajectory(combined, cfg, output_dir)
    else:
        print(f"Unknown analysis type: {analysis_type}. Skipping analysis.")

    print(f"\nPipeline complete. Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis Pipeline — W&B data extraction and analysis"
    )
    parser.add_argument(
        "--config", type=str, default="cfg/plotting_config.yaml",
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only fetch and report run counts without processing data",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        p1 = os.path.join(ANALYSIS_DIR, config_path)
        p2 = os.path.join(PROJECT_ROOT, config_path)
        if os.path.exists(p1):
            config_path = p1
        elif os.path.exists(p2):
            config_path = p2

    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    run_analysis_pipeline(cfg, dry_run=args.dry_run, config_path=config_path)


if __name__ == "__main__":
    main()
