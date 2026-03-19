#!/usr/bin/env python3
"""
Plasticity Plotting Pipeline

Connects to W&B, fetches neural network training logs for specified target
configurations, computes temporal metrics (forward loss, aggregated weight norm),
and generates three analytical plot types:

  1. Future Loss Distribution  — violin / box plot of forward_loss by learner type
  2. Parallel Coordinates      — multi-axis view with temporal subsampling
  3. Phase Space Portrait      — connected 2-D scatterplot (rank metric vs forward loss)
  4. Scatter Plot Matrix       — pairwise scatterplots for selected variables

Usage:
    python plasticity_plotting_pipeline.py --config cfg/plotting_config.yaml
    python plasticity_plotting_pipeline.py --config cfg/plotting_config.yaml --dry-run
"""

import argparse
import os
import sys
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import seaborn as sns

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import helpers
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
    determine_required_base_metrics
)
from analysis.helpers.plotting_utils import (
    update_display_names,
    build_filename
)
from analysis.helpers.plotting_functions import (
    plot_future_loss_distribution,
    plot_parallel_coordinates,
    plot_phase_space,
    plot_scatter_matrix
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: Dict[str, Any], dry_run: bool = False, config_path: Optional[str] = None) -> None:
    """Execute the full plasticity plotting pipeline."""
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]
    
    # Update display names from config if provided
    if "display_names" in cfg:
        update_display_names(cfg["display_names"])
        print(f"Updated display names with {len(cfg['display_names'])} custom entries.")

    base_filters = cfg["filters"]
    targets = resolve_targets(cfg)
    metric_cfg = cfg["metric_computation"]
    
    # Update forward_loss display name with window size
    fwd_window = metric_cfg.get("forward_loss", {}).get("window_size", 10)
    update_display_names({"forward_loss": f"Forward Loss (w={fwd_window})"})

    is_comparison = len(targets) > 1
    comparison_name = cfg.get("comparison_name")

    defaults = cfg.get("target_defaults", {})
    base_metrics = determine_required_base_metrics(cfg)
    print(f"Base metrics to fetch: {base_metrics}")

    weight_norm_cfg = metric_cfg.get("weight_norm", {})
    source_pattern = weight_norm_cfg.get("source_pattern", "_mean_abs_weight")
    aggregation = weight_norm_cfg.get("aggregation", "weight_norm_mean")

    all_target_dfs = []
    target_inventory: List[Dict[str, Any]] = []

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

        # Discover weight_norm keys from the first run
        wn_keys = discover_weight_norm_keys(runs[0], source_pattern)
        if wn_keys:
            print(f"  Discovered {len(wn_keys)} weight norm keys: {wn_keys[:3]}{'...' if len(wn_keys) > 3 else ''}")
        else:
            print(f"  No weight norm keys matching '{source_pattern}' found")

        # Extract history for each run
        run_dfs = []
        for i, run in enumerate(runs):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Extracting run {i + 1}/{len(runs)}: {run.name}")
            hist = extract_run_history(run, base_metrics + wn_keys)
            if hist.empty:
                continue
            hist["run_id"] = run.id
            hist["run_name"] = run.name

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
            inv["n_rows_sampled"] = len(target_df) # Update count after filtering
        # Legacy support
        elif "outlier_filtering" in metric_cfg:
            target_df = filter_outliers(target_df, metric_cfg["outlier_filtering"])
            inv["n_rows_sampled"] = len(target_df) # Update count after filtering

        all_target_dfs.append(target_df)
        target_inventory.append(inv)

    if dry_run:
        print("\n[DRY RUN] Pipeline complete — no data was processed or plotted.")
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

    # Build & save comprehensive pipeline summary
    summary = build_pipeline_summary(
        target_inventory, base_metrics,
        base_filters.get("drop_nan_threshold"),
        None, combined, nan_details, dropped_ids,
    )
    summary_fname = build_filename("pipeline_summary.txt", comparison_name)
    summary_path = os.path.join(output_dir, summary_fname)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Pipeline summary saved: {summary_path}")

    # Copy config file to output directory for reproducibility
    if config_path and os.path.exists(config_path):
        cfg_fname = os.path.basename(config_path)
        # If comparison_name is set, prepend it to avoid collisions if multiple configs used
        if comparison_name:
            cfg_fname = f"{comparison_name}_{cfg_fname}"
        cfg_dest = os.path.join(output_dir, cfg_fname)
        shutil.copy2(config_path, cfg_dest)
        print(f"  Config file copied to: {cfg_dest}")

    # Print a condensed version to console
    print("\n  Per-target inventory:")
    for inv in target_inventory:
        diff = inv["diff_label"]
        found = inv["runs_found"]
        kept_ids = [rid for rid in inv["run_ids"] if rid not in dropped_ids]
        dropped_count = found - len(kept_ids)
        if found == 0:
            print(f"    {diff}: NO RUNS in W&B")
        else:
            print(f"    {diff}: {found} found, {len(kept_ids)} kept, {dropped_count} dropped")

    if combined.empty:
        print("No data remaining after NaN filtering. Exiting.")
        return

    # Build a shared color map so all plots use identical target→color assignment
    hue_col = "_target_label"
    all_categories = sorted(combined[hue_col].unique())
    shared_palette = sns.color_palette("Set2", n_colors=max(len(all_categories), 1))
    shared_color_map = {cat: shared_palette[i] for i, cat in enumerate(all_categories)}

    # Outlier filtering info for plot titles
    outlier_info = ""
    outlier_cfg = metric_cfg.get("per_target_group_filtering") or metric_cfg.get("outlier_filtering")
    if outlier_cfg and outlier_cfg.get("enabled", False):
        o_cfg = outlier_cfg
        method = o_cfg.get("method", "quantile")
        cols = o_cfg.get("columns", [])
        col_str = ", ".join(cols) if cols else "data"
        
        if method == "quantile":
            low = o_cfg.get("lower_quantile", 0.01) * 100
            high = (1.0 - o_cfg.get("upper_quantile", 0.99)) * 100
            # If symmetric (e.g. 1% bottom and 1% top), say "top/bottom 1%"
            if abs(low - high) < 0.001:
                outlier_info = f" (top/bottom {low:g}% {col_str} outliers filtered)"
            else:
                outlier_info = f" ({low:g}%-{100-high:g}% {col_str} outliers filtered)"
        elif method == "iqr":
            k = o_cfg.get("iqr_multiplier", 1.5)
            outlier_info = f" (IQR x{k} {col_str} outliers filtered)"
        elif method == "z_score":
            t = o_cfg.get("threshold", 3.0)
            outlier_info = f" (Z-score > {t} {col_str} outliers filtered)"

    # Generate plots
    plots_cfg = cfg["plots"]

    if plots_cfg.get("future_loss_distribution", {}).get("enabled"):
        plot_future_loss_distribution(combined, plots_cfg["future_loss_distribution"], output_dir, is_comparison,
                                      comparison_name=comparison_name, color_map=shared_color_map,
                                      title_suffix=outlier_info,
                                      global_scales=cfg.get("global_settings", {}).get("variable_scales"))

    if plots_cfg.get("parallel_coordinates", {}).get("enabled"):
        fwd_window = metric_cfg["forward_loss"]["window_size"]
        plot_parallel_coordinates(combined, plots_cfg["parallel_coordinates"], output_dir, is_comparison,
                                  forward_loss_window=fwd_window, comparison_name=comparison_name,
                                  color_map=shared_color_map,
                                  title_suffix=outlier_info,
                                  global_scales=cfg.get("global_settings", {}).get("variable_scales"))

    if plots_cfg.get("phase_space", {}).get("enabled"):
        plot_phase_space(combined, plots_cfg["phase_space"], output_dir, is_comparison,
                         comparison_name=comparison_name, color_map=shared_color_map,
                         title_suffix=outlier_info,
                         global_scales=cfg.get("global_settings", {}).get("variable_scales"))

    if plots_cfg.get("scatter_matrix", {}).get("enabled"):
        plot_scatter_matrix(combined, plots_cfg["scatter_matrix"], output_dir, is_comparison,
                            comparison_name=comparison_name, color_map=shared_color_map,
                            title_suffix=outlier_info,
                            global_scales=cfg.get("global_settings", {}).get("variable_scales"))

    print(f"\nPipeline complete. Plots saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plasticity Plotting Pipeline — W&B data extraction and visualization"
    )
    parser.add_argument(
        "--config", type=str, default="cfg/plotting_config.yaml",
        help="Path to the plotting config YAML file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only fetch and report run counts without processing data",
    )
    args = parser.parse_args()

    config_path = args.config
    # Resolve config path: check CWD, then script dir, then project root
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        # 1. Check relative to script directory (analysis/)
        p1 = os.path.join(ANALYSIS_DIR, config_path)
        # 2. Check relative to project root
        p2 = os.path.join(PROJECT_ROOT, config_path)
        
        if os.path.exists(p1):
            config_path = p1
        elif os.path.exists(p2):
            config_path = p2

    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    run_pipeline(cfg, dry_run=args.dry_run, config_path=config_path)


if __name__ == "__main__":
    main()
