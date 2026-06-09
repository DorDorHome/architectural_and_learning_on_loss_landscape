import os
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from typing import Any, Dict, List, Optional, Tuple

from analysis.helpers.plotting_utils import (
    display_name,
    build_filename,
    resolve_color_column,
    slugify_target_label
)

def _gather_future_distributions(
    df: pd.DataFrame,
    metric: str,
    window_size: int,
    interval: int,
    hue_col: str,
) -> pd.DataFrame:
    """For each sampled checkpoint task t, gather raw metric values from [t+1 .. t+w]."""
    records = []
    for (run_id, label), grp in df.groupby(["run_id", hue_col]):
        grp = grp.sort_values("task_idx").reset_index(drop=True)
        task_vals = grp["task_idx"].values
        metric_vals = grp[metric].values
        n = len(grp)

        # Build a task_idx -> row-index lookup
        idx_map = {int(task_vals[i]): i for i in range(n)}

        # Iterate over checkpoint tasks (multiples of interval)
        for t in task_vals:
            if int(t) % interval != 0:
                continue
            row_i = idx_map[int(t)]
            # Gather raw metric values from the forward window
            end_i = min(row_i + 1 + window_size, n)
            for j in range(row_i + 1, end_i):
                records.append({
                    "checkpoint_task": int(t),
                    metric: metric_vals[j],
                    hue_col: label,
                    "run_id": run_id,
                })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

def plot_future_loss_distribution(
    df: pd.DataFrame,
    plot_cfg: Dict[str, Any],
    output_dir: str,
    is_comparison: bool,
    comparison_name: Optional[str] = None,
    color_map: Optional[Dict[str, Any]] = None,
    title_suffix: str = "",
    global_scales: Optional[Dict[str, str]] = None,
) -> None:
    """Plot 1: Distribution of raw future metric values at task checkpoints."""
    print("\n--- Plot 1: Future Loss Distribution ---")
    hue_col = "_target_label"
    plot_type = plot_cfg.get("plot_type", "violin")
    interval = plot_cfg.get("task_idx_interval", 10)
    window_size = plot_cfg.get("window_size", 10)
    metric = plot_cfg.get("metric", "epoch_loss")

    if metric not in df.columns:
        warnings.warn(f"Metric '{metric}' not found in DataFrame; skipping future loss distribution.")
        return

    # Gather forward-window distributions
    dist_df = _gather_future_distributions(df, metric, window_size, interval, hue_col)
    if dist_df.empty:
        warnings.warn("No future distribution data could be gathered; skipping.")
        return

    # Drop NaN metric values that were collected from the forward window
    n_before_nan = len(dist_df)
    dist_df = dist_df.dropna(subset=[metric])
    n_after_nan = len(dist_df)
    if n_before_nan != n_after_nan:
        print(f"  Dropped {n_before_nan - n_after_nan} NaN {metric} values from forward windows")

    checkpoints = sorted(dist_df["checkpoint_task"].unique())
    # Drop checkpoints with very few data points (near end of training)
    min_points = max(window_size // 2, 2)
    valid_checkpoints = []
    for cp in checkpoints:
        n_pts = len(dist_df[dist_df["checkpoint_task"] == cp])
        if n_pts >= min_points:
            valid_checkpoints.append(cp)
        else:
            print(f"  Dropping checkpoint task_idx={cp}: only {n_pts} future values (< {min_points})")

    dist_df = dist_df[dist_df["checkpoint_task"].isin(valid_checkpoints)]
    if dist_df.empty:
        warnings.warn("No checkpoints with enough future data; skipping.")
        return

    n_ticks = len(valid_checkpoints)
    n_targets = dist_df[hue_col].nunique()

    # Per-target diagnostics with value ranges
    print(f"\n  Future distribution diagnostics per target:")
    for lab, grp in dist_df.groupby(hue_col):
        n_runs = grp["run_id"].nunique()
        n_pts = len(grp)
        n_cps = grp["checkpoint_task"].nunique()
        vmin, vmax, vmean = grp[metric].min(), grp[metric].max(), grp[metric].mean()
        # Show full label on one line (replace newlines with " | ")
        lab_oneline = lab.replace("\n", " | ") if isinstance(lab, str) else str(lab)
        print(f"    {lab_oneline}:")
        print(f"      {n_runs} run(s), {n_pts} total points across {n_cps} checkpoints")
        print(f"      {metric} range: [{vmin:.4f}, {vmax:.4f}], mean={vmean:.4f}")

    print(f"  {n_ticks} checkpoints, window={window_size}, metric={metric}")

    fig_width = max(12, n_ticks * 0.55 * max(n_targets, 1))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    dist_df["checkpoint_task"] = dist_df["checkpoint_task"].astype(int)

    pal = color_map if color_map else "Set2"

    if plot_type == "violin":
        sns.violinplot(
            data=dist_df, x="checkpoint_task", y=metric,
            hue=hue_col, inner="quartile", cut=0, ax=ax,
            palette=pal, density_norm="width", linewidth=0.8,
        )
    else:
        sns.boxplot(
            data=dist_df, x="checkpoint_task", y=metric,
            hue=hue_col, ax=ax, palette=pal, linewidth=0.8,
        )

    ax.set_xlabel("Checkpoint Task Index (looking forward)")
    
    # Apply log scale if configured for the metric
    if global_scales:
        base = _get_log_base(global_scales.get(metric))
        if base:
            ax.set_yscale("log", base=base)
            suffix = " (log scale)" if base == 10 else f" (log{base:g} scale)"
            ax.set_ylabel(display_name(metric) + suffix)
        else:
            ax.set_ylabel(display_name(metric))
    else:
        ax.set_ylabel(display_name(metric))

    ax.set_title(
        f"Future {display_name(metric)} Distribution (window={window_size}){title_suffix}",
        fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if n_ticks > 15:
        ax.tick_params(axis="x", rotation=45)

    ax.legend(title="Target Configuration", loc="best", fontsize=8)

    plt.tight_layout()
    fname = build_filename("future_loss_distribution.png", comparison_name)
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _parse_pc_variables(raw_order: List) -> Tuple[List[str], Dict[str, bool]]:
    """Parse the variables_order config into (names, higher_is_better_map)."""
    names: List[str] = []
    hib_map: Dict[str, bool] = {}
    for entry in raw_order:
        if isinstance(entry, dict):
            name = entry["name"]
            hib = entry.get("higher_is_better", True)
        else:
            name = entry
            hib = True
        names.append(name)
        hib_map[name] = hib
    return names, hib_map

def plot_parallel_coordinates(
    df: pd.DataFrame,
    plot_cfg: Dict[str, Any],
    output_dir: str,
    is_comparison: bool,
    forward_loss_window: int = 10,
    comparison_name: Optional[str] = None,
    color_map: Optional[Dict[str, Any]] = None,
    title_suffix: str = "",
    global_scales: Optional[Dict[str, str]] = None,
) -> None:
    """Plot 2: Single-axes parallel coordinates with per-variable inversion."""
    print("\n--- Plot 2: Parallel Coordinates ---")
    variables, hib_map = _parse_pc_variables(plot_cfg["variables_order"])
    interval = plot_cfg.get("task_idx_interval")
    # Legacy support: accept an explicit list via 'temporal_subsampling'
    explicit_list = plot_cfg.get("temporal_subsampling")
    line_width = plot_cfg.get("line_width", 0.9)
    line_alpha = plot_cfg.get("line_alpha", 0.40)

    plot_data = df.copy()

    if interval:
        plot_data = plot_data[plot_data["task_idx"] % interval == 0]
        print(f"  Subsampled every {interval} task_idx: {len(plot_data)} rows")
    elif explicit_list:
        plot_data = plot_data[plot_data["task_idx"].isin(explicit_list)]
        print(f"  Subsampled to task_idx in {explicit_list}: {len(plot_data)} rows")

    var_cols = [v for v in variables if v in plot_data.columns]
    plot_data = plot_data.dropna(subset=var_cols)
    if plot_data.empty:
        warnings.warn("No data remaining after filtering for parallel coordinates; skipping.")
        return
    for v in variables:
        if v not in plot_data.columns:
            warnings.warn(f"Variable '{v}' not in DataFrame; skipping parallel coordinates.")
            return

    # Hue via _target_label
    hue_col = "_target_label"
    is_categorical = hue_col in plot_data.columns
    if not is_categorical:
        hue_col = resolve_color_column(plot_data, plot_cfg["color_mapping"]["color_by"])

    n_vars = len(variables)

    # --- Normalise each variable to [0, 1] (invert if higher_is_better=False) ---
    # Note: Parallel coordinates usually normalizes linearly. 
    # If we want log scale, we should log-transform the data BEFORE normalization.
    var_ranges: Dict[str, Tuple[float, float]] = {}
    var_inverted: Dict[str, bool] = {}
    norm_data = plot_data.copy()

    for var in variables:
        col = norm_data[var].astype(float)
        
        # Apply log transform if configured
        base = _get_log_base(global_scales.get(var)) if global_scales else None
        if base:
            # Filter > 0
            valid = col > 0
            if not valid.all():
                # For visualization, just clip or drop? 
                # Parallel coords needs aligned rows. Let's clip to small epsilon.
                col = col.clip(lower=1e-9)
            col = np.log(col) / np.log(base)
            # Update column in norm_data so normalization uses log values
            norm_data[var] = col
            
        vmin, vmax = col.min(), col.max()
        var_ranges[var] = (vmin, vmax)
        invert = not hib_map.get(var, True)
        var_inverted[var] = invert
        if vmax > vmin:
            normed = (col - vmin) / (vmax - vmin)
            if invert:
                normed = 1.0 - normed
            norm_data[f"_n_{var}"] = normed
        else:
            norm_data[f"_n_{var}"] = 0.5

    # --- Color map (use shared map if provided) ---
    if is_categorical:
        categories = sorted(plot_data[hue_col].unique())
        if color_map is None:
            palette = sns.color_palette("Set2", n_colors=len(categories))
            color_map = {cat: palette[i] for i, cat in enumerate(categories)}
        assert color_map is not None
    else:
        cmap_obj = cm.get_cmap("viridis")
        col_vals = plot_data[hue_col].astype(float)
        norm_color = (col_vals - col_vals.min()) / max(col_vals.max() - col_vals.min(), 1e-12)

    # --- Single-axes figure ---
    fig, host = plt.subplots(figsize=(3.0 * n_vars, 7))
    host.set_xlim(-0.2, n_vars - 1 + 0.2)
    host.set_ylim(-0.04, 1.04)
    host.set_xticks(range(n_vars))
    host.set_xticklabels([display_name(v) for v in variables], fontsize=9)
    host.tick_params(axis="y", left=False, labelleft=False)
    host.spines["top"].set_visible(False)
    host.spines["bottom"].set_visible(False)
    host.spines["left"].set_visible(False)
    host.spines["right"].set_visible(False)

    # Draw vertical axis spines
    for i in range(n_vars):
        host.axvline(x=i, color="black", linewidth=1.0, zorder=0)

    # --- Draw polylines ---
    for idx, row in norm_data.iterrows():
        if is_categorical:
            assert color_map is not None
            color = color_map[row[hue_col]]
        else:
            color = cmap_obj(norm_color.loc[idx])

        xs = list(range(n_vars))
        ys = [row[f"_n_{v}"] for v in variables]
        host.plot(xs, ys, color=color, alpha=line_alpha, linewidth=line_width, zorder=1)

    # --- Summary lines (beginning / ending snapshots) ---
    summary_cfg = plot_cfg.get("summary_lines", {})
    if summary_cfg.get("enabled") and is_categorical:
        summary_metric = summary_cfg.get("metric", "forward_loss")
        # For each target, pick the first and last valid task_idx
        for cat in categories:
            cat_rows = norm_data[norm_data[hue_col] == cat]
            if cat_rows.empty:
                continue
            assert color_map is not None
            color = color_map[cat]

            # Beginning: smallest task_idx (mean across runs at that task)
            min_task = cat_rows["task_idx"].min()
            begin_rows = cat_rows[cat_rows["task_idx"] == min_task]

            # Ending: if metric uses forward_loss, the last valid task_idx is
            # (max_task - forward_loss_window) so forward_loss has a full window.
            max_task = cat_rows["task_idx"].max()
            if summary_metric == "forward_loss":
                end_task = max_task - forward_loss_window
                end_rows = cat_rows[cat_rows["task_idx"] == end_task]
                if end_rows.empty:
                    # Fall back to closest available task_idx
                    valid = cat_rows[cat_rows["task_idx"] <= end_task]
                    if valid.empty:
                        valid = cat_rows
                    end_task = valid["task_idx"].max()
                    end_rows = cat_rows[cat_rows["task_idx"] == end_task]
            else:
                end_task = max_task
                end_rows = cat_rows[cat_rows["task_idx"] == end_task]

            for label_tag, rows, lstyle, lw_mult in [
                (f"t={int(min_task)}", begin_rows, (0, (5, 3)), 2.5),      # thick dashed
                (f"t={int(end_task)}", end_rows, (0, (3, 1, 1, 1)), 2.5),  # thick dash-dot
            ]:
                if rows.empty:
                    continue
                means = rows[[f"_n_{v}" for v in variables]].mean()
                xs = list(range(n_vars))
                ys = [means[f"_n_{v}"] for v in variables]
                host.plot(
                    xs, ys, color=color, alpha=0.9,
                    linewidth=line_width * lw_mult, linestyle=lstyle, zorder=5,
                )

    # --- Per-axis tick labels (real values) ---
    n_ticks = 5
    tick_fracs = np.linspace(0, 1, n_ticks)
    for i, var in enumerate(variables):
        vmin, vmax = var_ranges[var]
        inverted = var_inverted[var]
        
        # Check if log scaled
        base = _get_log_base(global_scales.get(var)) if global_scales else None
        
        # Alternate labels left/right so adjacent axes don't overlap
        if i % 2 == 0:
            x_offset = -0.06
            ha = "right"
        else:
            x_offset = 0.06
            ha = "left"
        for frac in tick_fracs:
            if inverted:
                val_norm = vmax - frac * (vmax - vmin)
            else:
                val_norm = vmin + frac * (vmax - vmin)
            
            # Inverse transform if log
            if base:
                real_val = base ** val_norm
            else:
                real_val = val_norm
                
            host.annotate(
                f"{real_val:.3g}", xy=(i, frac),
                xytext=(i + x_offset, frac),
                fontsize=7, ha=ha, va="center", color="0.3",
            )

    # Indicate inverted axes and log scales
    for i, var in enumerate(variables):
        labels = []
        if var_inverted[var]:
            labels.append("inverted")
        
        base = _get_log_base(global_scales.get(var)) if global_scales else None
        if base:
            labels.append(f"log{base:g}")
            
        if labels:
            host.annotate(
                f"({', '.join(labels)})", xy=(i, -0.02), fontsize=7, ha="center",
                va="top", color="0.45", style="italic",
            )

    # --- Legend ---
    if is_categorical:
        assert color_map is not None
        legend_handles = [
            Line2D([0], [0], color=color_map[cat], linewidth=2, label=cat)
            for cat in categories
        ]
        if summary_cfg.get("enabled"):
            legend_handles.append(
                Line2D([0], [0], color="grey", linewidth=2.0,
                       linestyle=(0, (5, 3)), label="Beginning (first task)")
            )
            legend_handles.append(
                Line2D([0], [0], color="grey", linewidth=2.0,
                       linestyle=(0, (3, 1, 1, 1)), label="Ending (last valid task)")
            )
        host.legend(
            handles=legend_handles, title="Target Configuration",
            loc="upper center", bbox_to_anchor=(0.5, 1.18),
            ncol=min(len(categories), 2), frameon=True, fontsize=8,
        )

    host.set_title(f"Parallel Coordinates — Plasticity Metrics{title_suffix}", fontsize=13, pad=40)

    plt.tight_layout()
    fname = build_filename("parallel_coordinates.png", comparison_name)
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _phase_space_axis_label(metric: str, higher_is_better: Optional[bool]) -> str:
    """Build axis label, appending '(inverted)' when higher_is_better is False."""
    base = display_name(metric)
    if higher_is_better is False:
        return f"{base} (inverted)"
    return base

def _get_log_base(scale_str: Optional[str]) -> Optional[float]:
    """Parse log scale string (e.g. 'log', 'log2', 'log10') to base number."""
    if not scale_str:
        return None
    if scale_str == "log":
        return 10.0
    if scale_str.startswith("log"):
        try:
            return float(scale_str[3:])
        except ValueError:
            return None
    return None

def _apply_phase_space_axes(
    ax: Any,
    x_cfg: Dict[str, Any],
    y_cfg: Dict[str, Any],
    x_metric: str,
    y_metric: str,
) -> None:
    """Apply axis scaling, inversion, and labels to a phase space axes."""
    x_base = _get_log_base(x_cfg.get("scale"))
    if x_base:
        ax.set_xscale("log", base=x_base)
    
    y_base = _get_log_base(y_cfg.get("scale"))
    if y_base:
        ax.set_yscale("log", base=y_base)

    if x_cfg.get("higher_is_better") is False:
        ax.invert_xaxis()
    if y_cfg.get("higher_is_better") is False:
        ax.invert_yaxis()
    ax.set_xlabel(_phase_space_axis_label(x_metric, x_cfg.get("higher_is_better")))
    ax.set_ylabel(_phase_space_axis_label(y_metric, y_cfg.get("higher_is_better")))
    ax.grid(True, linestyle="--", alpha=0.3)

def _add_phase_space_semantic_background(
    ax: Any,
    x_cfg: Dict[str, Any],
    y_cfg: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    show_labels: bool = True,
) -> None:
    """Add gradient background (good corner=grey, bad=white) and corner labels."""
    # Build gradient in normalized [0,1] space, then map to data extent
    n = 100
    xx = np.linspace(0, 1, n)
    yy = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xx, yy)
    Z = (X + Y) / 2  # 0 at lower-left, 1 at upper-right
    cmap = LinearSegmentedColormap.from_list("goodness", ["white", (0.72, 0.72, 0.72)])
    ax.imshow(
        Z,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        zorder=0,
        interpolation="bilinear",
    )
    # Corner labels outside plot, with arrows pointing to corners (toggleable)
    if show_labels:
        ax.annotate(
            "desirable basin",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            xytext=(1.08, 1.08),
            textcoords="axes fraction",
            fontsize=11,
            va="center",
            ha="left",
            color="0.35",
            arrowprops=dict(arrowstyle="fancy,head_length=2.0,head_width=2.5,tail_width=0.25", color="0.4", shrinkA=4, shrinkB=4),
            clip_on=False,
        )
        ax.annotate(
            "loss of plasticity attractor",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            xytext=(-0.08, -0.08),
            textcoords="axes fraction",
            fontsize=11,
            va="center",
            ha="right",
            color="0.5",
            arrowprops=dict(arrowstyle="fancy,head_length=2.0,head_width=2.5,tail_width=0.25", color="0.5", shrinkA=4, shrinkB=4),
            clip_on=False,
        )

def plot_phase_space(
    df: pd.DataFrame,
    plot_cfg: Dict[str, Any],
    output_dir: str,
    is_comparison: bool,
    comparison_name: Optional[str] = None,
    color_map: Optional[Dict[str, Any]] = None,
    title_suffix: str = "",
    global_scales: Optional[Dict[str, str]] = None,
) -> None:
    """Plot 3: Phase space portrait — connected 2-D scatterplot by task_idx."""
    print("\n--- Plot 3: Phase Space Portrait ---")
    x_cfg = plot_cfg["x_axis"]
    y_cfg = plot_cfg["y_axis"]
    x_metric = x_cfg["metric"]
    y_metric = y_cfg["metric"]
    line_width = plot_cfg.get("line_width", 0.8)
    line_alpha = plot_cfg.get("line_alpha", 0.4)
    plot_aggregate = plot_cfg.get("plot_aggregate_runs", True)
    plot_per_target = plot_cfg.get("plot_separate_subplots_per_target", False)

    required = [x_metric, y_metric, "task_idx"]
    plot_data = df.dropna(subset=[c for c in required if c in df.columns])
    
    # Merge plot-specific scales with global scales (plot-specific takes precedence if conflicting, though usually we want global)
    # Actually, let's prioritize global scales if passed
    if global_scales:
        # Update config with global scales
        if "scale" not in x_cfg or not x_cfg["scale"]:
            if x_metric in global_scales:
                x_cfg["scale"] = global_scales[x_metric]
        if "scale" not in y_cfg or not y_cfg["scale"]:
            if y_metric in global_scales:
                y_cfg["scale"] = global_scales[y_metric]

    interval = plot_cfg.get("task_idx_interval")
    if interval:
        plot_data = plot_data[plot_data["task_idx"] % interval == 0]
        print(f"  Subsampled every {interval} task_idx: {len(plot_data)} rows")

    if plot_data.empty:
        warnings.warn("No valid data for phase space plot; skipping.")
        return

    hue_col = "_target_label"
    categories = sorted(plot_data[hue_col].unique())
    if color_map is None:
        palette = sns.color_palette("Set2", n_colors=max(len(categories), 1))
        color_map = {cat: palette[i] for i, cat in enumerate(categories)}
    assert color_map is not None

    # --- 1. Aggregate plot: single .png, mean trajectory per target ---
    if plot_aggregate:
        agg_data = (
            plot_data.groupby([hue_col, "task_idx"], as_index=False)[[x_metric, y_metric]]
            .agg("mean")
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        # Compute limits and add background BEFORE data (so it appears behind)
        xmin, xmax = agg_data[x_metric].min(), agg_data[x_metric].max()
        ymin, ymax = agg_data[y_metric].min(), agg_data[y_metric].max()
        pad_x = max((xmax - xmin) * 0.02, 1e-6)
        pad_y = max((ymax - ymin) * 0.02, 1e-6)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        _apply_phase_space_axes(ax, x_cfg, y_cfg, x_metric, y_metric)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        _add_phase_space_semantic_background(
            ax, x_cfg, y_cfg, xmin, xmax, ymin, ymax,
            show_labels=plot_cfg.get("label_for_good_bad_basin_aggregate", True),
        )
        scatter_ax = None

        for cat in categories:
            cat_agg = agg_data[agg_data[hue_col] == cat].sort_values("task_idx")
            if cat_agg.empty:
                continue
            x_vals = cat_agg[x_metric].values
            y_vals = cat_agg[y_metric].values
            task_vals = cat_agg["task_idx"].values

            ax.plot(x_vals, y_vals, color=color_map[cat], alpha=line_alpha, linewidth=line_width, zorder=1)
            scatter = ax.scatter(
                x_vals, y_vals,
                c=task_vals, cmap="viridis", s=20, alpha=0.7,
                edgecolors=color_map[cat], linewidths=0.5, zorder=2,
                label=cat,
            )
            scatter_ax = scatter
            for x, y, t in zip(x_vals, y_vals, task_vals):
                ax.annotate(str(int(t)), (x, y), textcoords="offset points", xytext=(4, 4),
                            fontsize=5, color="0.3", zorder=4)
            if len(x_vals) > 0:
                ax.scatter(x_vals[0], y_vals[0], marker="o", s=100, color=color_map[cat],
                           edgecolors="black", linewidths=1.5, zorder=3)
                ax.scatter(x_vals[-1], y_vals[-1], marker="X", s=100, color=color_map[cat],
                           edgecolors="black", linewidths=1.5, zorder=3)

        if scatter_ax is not None:
            cbar = fig.colorbar(scatter_ax, ax=ax, pad=0.02)
            cbar.set_label("Task Index")
        ax.set_title(f"Phase Space (aggregate): {display_name(x_metric)} vs {display_name(y_metric)}{title_suffix}", fontsize=13)
        start_marker = Line2D([0], [0], marker="o", color="grey", markersize=8,
                              markeredgecolor="black", linestyle="None", label="Start (t=0)")
        end_marker = Line2D([0], [0], marker="X", color="grey", markersize=8,
                            markeredgecolor="black", linestyle="None", label="End (t=T)")
        cat_handles = [Line2D([0], [0], color=color_map[cat], linewidth=2, label=cat) for cat in categories]
        ax.legend(handles=cat_handles + [start_marker, end_marker], title="Target Configuration", loc="best", fontsize=8)
        plt.tight_layout()
        fname = build_filename("phase_space_portrait.png", comparison_name)
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- 2. Per-target subplots: one .png per target, subplots = one per run ---
    if plot_per_target:
        if "run_id" not in plot_data.columns:
            print("  [SKIP] plot_separate_subplots_per_target: run_id not in data columns. "
                  f"Available: {list(plot_data.columns)[:15]}...")
        else:
            print(f"  Generating per-target phase space files for {len(categories)} target(s)...")
            for cat in categories:
                cat_data = plot_data[plot_data[hue_col] == cat]
                run_ids = cat_data["run_id"].unique()
                n_runs = len(run_ids)
                if n_runs == 0:
                    continue
                n_cols = math.ceil(math.sqrt(n_runs))
                n_rows = math.ceil(n_runs / n_cols)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                axes_flat = np.atleast_1d(axes).flatten()

                for i, rid in enumerate(run_ids):
                    ax = axes_flat[i]
                    run_df = cat_data[cat_data["run_id"] == rid].sort_values("task_idx")
                    run_name = run_df["run_name"].iloc[0] if "run_name" in run_df.columns else str(rid)[:12]
                    x_vals = run_df[x_metric].values
                    y_vals = run_df[y_metric].values
                    task_vals = run_df["task_idx"].values

                    # Add background BEFORE data (compute limits, set, apply axes, add bg)
                    xmin, xmax = float(np.min(x_vals)), float(np.max(x_vals))
                    ymin, ymax = float(np.min(y_vals)), float(np.max(y_vals))
                    pad_x = max((xmax - xmin) * 0.02, 1e-6)
                    pad_y = max((ymax - ymin) * 0.02, 1e-6)
                    ax.set_xlim(xmin - pad_x, xmax + pad_x)
                    ax.set_ylim(ymin - pad_y, ymax + pad_y)
                    _apply_phase_space_axes(ax, x_cfg, y_cfg, x_metric, y_metric)
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    _add_phase_space_semantic_background(
                        ax, x_cfg, y_cfg, xmin, xmax, ymin, ymax,
                        show_labels=plot_cfg.get("label_for_good_bad_basin_per_target", True),
                    )

                    ax.plot(x_vals, y_vals, color=color_map[cat], alpha=line_alpha, linewidth=line_width, zorder=1)
                    ax.scatter(x_vals, y_vals, c=task_vals, cmap="viridis", s=15, alpha=0.7,
                               edgecolors=color_map[cat], linewidths=0.5, zorder=2)
                    for x, y, t in zip(x_vals, y_vals, task_vals):
                        ax.annotate(str(int(t)), (x, y), textcoords="offset points", xytext=(3, 3),
                                    fontsize=4, color="0.3", zorder=4)
                    if len(x_vals) > 0:
                        ax.scatter(x_vals[0], y_vals[0], marker="o", s=60, color=color_map[cat],
                                   edgecolors="black", linewidths=1, zorder=3)
                        ax.scatter(x_vals[-1], y_vals[-1], marker="X", s=60, color=color_map[cat],
                                   edgecolors="black", linewidths=1, zorder=3)

                    ax.set_title(run_name, fontsize=9)

                for j in range(n_runs, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                # Add shared legend for start/end symbols
                start_marker = Line2D([0], [0], marker="o", color="grey", markersize=8,
                                      markeredgecolor="black", linestyle="None", label="Start (t=0)")
                end_marker = Line2D([0], [0], marker="X", color="grey", markersize=8,
                                    markeredgecolor="black", linestyle="None", label="End (t=T)")
                fig.legend(handles=[start_marker, end_marker], loc="lower center", ncol=2,
                           bbox_to_anchor=(0.5, -0.02), fontsize=9)

                fig.suptitle(f"Phase Space: {cat}{title_suffix}", fontsize=12, y=1.02)
                slug = slugify_target_label(cat)
                fname = build_filename(f"phase_space_target_{slug}.png", comparison_name)
                path = os.path.join(output_dir, fname)
                fig.savefig(path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")

def plot_scatter_matrix(
    df: pd.DataFrame,
    plot_cfg: Dict[str, Any],
    output_dir: str,
    is_comparison: bool,
    comparison_name: Optional[str] = None,
    color_map: Optional[Dict[str, Any]] = None,
    title_suffix: str = "",
    global_scales: Optional[Dict[str, str]] = None,
) -> None:
    """Plot 4: Scatter plot matrix (pairwise scatterplots) for selected variables.
    
    Supports two modes:
    1. Square Matrix: If 'variables' list is provided. Plots all pairwise combinations.
    2. Rectangular Matrix: If 'target_independent_variables' and 'target_dependent_variables' 
       are provided. Plots Independent (X) vs Dependent (Y).
    """
    print("\n--- Plot 4: Scatter Plot Matrix ---")
    
    # Check for rectangular mode config
    indep_vars = plot_cfg.get("target_independent_variables", [])
    dep_vars = plot_cfg.get("target_dependent_variables", [])
    is_rectangular = bool(indep_vars and dep_vars)
    
    if is_rectangular:
        x_vars = indep_vars
        y_vars = dep_vars
        required_cols = list(set(x_vars + y_vars))
        print(f"  Mode: Rectangular (Independent vs Dependent)")
        print(f"  Independent (X): {x_vars}")
        print(f"  Dependent (Y): {y_vars}")
    else:
        # Fallback to square mode
        variables = plot_cfg.get("variables", [])
        if not variables:
            warnings.warn("No variables specified for scatter matrix; skipping.")
            return
        x_vars = variables
        y_vars = variables
        required_cols = variables
        print(f"  Mode: Square (Pairwise)")
        print(f"  Variables: {variables}")

    interval = plot_cfg.get("task_idx_interval")
    marker_size = plot_cfg.get("marker_size", 15)
    marker_alpha = plot_cfg.get("marker_alpha", 0.6)
    diagonal = plot_cfg.get("diagonal", "kde")
    plot_combined = plot_cfg.get("plot_combined", True)
    plot_per_target = plot_cfg.get("plot_per_target", False)
    add_best_fit_line = plot_cfg.get("add_best_fit_line", False)
    variable_scales = plot_cfg.get("variable_scales", {})
    # Merge with global scales if provided
    if global_scales:
        for k, v in global_scales.items():
            if k not in variable_scales:
                variable_scales[k] = v

    # Filter to required columns
    missing_cols = [v for v in required_cols if v not in df.columns]
    if missing_cols:
        warnings.warn(f"Missing columns for scatter matrix: {missing_cols}")
        # Filter down to available columns to try and plot partial data
        if is_rectangular:
            x_vars = [v for v in x_vars if v in df.columns]
            y_vars = [v for v in y_vars if v in df.columns]
            if not x_vars or not y_vars:
                warnings.warn("Not enough valid columns for rectangular plot. Skipping.")
                return
            required_cols = list(set(x_vars + y_vars))
        else:
            required_cols = [v for v in required_cols if v in df.columns]
            if len(required_cols) < 2:
                warnings.warn(f"Need at least 2 valid variables for square scatter matrix. Found: {required_cols}")
                return
            x_vars = required_cols
            y_vars = required_cols

    hue_col = "_target_label"
    plot_data = df.dropna(subset=required_cols)

    if interval:
        plot_data = plot_data[plot_data["task_idx"] % interval == 0]
        print(f"  Subsampled every {interval} task_idx: {len(plot_data)} rows")

    if plot_data.empty:
        warnings.warn("No valid data for scatter matrix; skipping.")
        return

    # Build color palette
    categories = sorted(plot_data[hue_col].unique())
    if color_map is None:
        palette = sns.color_palette("Set2", n_colors=max(len(categories), 1))
        color_map = {cat: palette[i] for i, cat in enumerate(categories)}

    # Map hue to colors for seaborn
    palette_dict = {cat: color_map[cat] for cat in categories}

    print(f"  Targets: {len(categories)}, rows: {len(plot_data)}")

    # Helper to apply log scales and update labels
    def _apply_scales_and_labels(g: sns.PairGrid, x_vars_list: List[str], y_vars_list: List[str]):
        # Iterate over the grid axes
        for i, row_var in enumerate(y_vars_list):
            for j, col_var in enumerate(x_vars_list):
                ax = g.axes[i, j]
                if ax is None: continue
                
                # Apply X-axis scale
                col_base = _get_log_base(variable_scales.get(col_var))
                if col_base:
                    ax.set_xscale("log", base=col_base)
                
                # Apply Y-axis scale
                row_base = _get_log_base(variable_scales.get(row_var))
                if row_base:
                    ax.set_yscale("log", base=row_base)

        # Update labels
        # X labels (columns)
        for j, col_var in enumerate(x_vars_list):
            label = display_name(col_var)
            base = _get_log_base(variable_scales.get(col_var))
            if base:
                suffix = " (log scale)" if base == 10 else f" (log{base:g} scale)"
                label += suffix
            
            # Set X labels on the bottom row
            if g.axes.shape[0] > 0:
                ax_bottom = g.axes[-1, j]
                if ax_bottom:
                    ax_bottom.set_xlabel(label)
        
        # Y labels (rows)
        for i, row_var in enumerate(y_vars_list):
            label = display_name(row_var)
            base = _get_log_base(variable_scales.get(row_var))
            if base:
                suffix = " (log scale)" if base == 10 else f" (log{base:g} scale)"
                label += suffix
            
            # Set Y labels on the left column
            if g.axes.shape[1] > 0:
                ax_left = g.axes[i, 0]
                if ax_left:
                    ax_left.set_ylabel(label)

    # Define wrapper for diagonal KDE/Hist (only used in square mode)
    def _diag_wrapper(x, **kwargs):
        var_name = x.name
        base = _get_log_base(variable_scales.get(var_name))
        # PairGrid passes 'color', 'label', etc. in kwargs
        if diagonal == "hist":
            sns.histplot(x, log_scale=base if base else False, **kwargs)
        elif diagonal == "kde":
            # log_scale=True (or base) in kdeplot computes density on log-transformed data
            sns.kdeplot(x, log_scale=base if base else False, **kwargs)

    # Custom regplot that handles log-log fitting correctly (linear in log space)
    def custom_regplot(x, y, **kwargs):
        # Extract style kwargs injected by PairGrid (color, label, etc.)
        color = kwargs.get('color')
        
        # Plot Scatter
        plt.scatter(x, y, s=marker_size, alpha=marker_alpha, color=color, linewidth=0)
        
        # Regression Logic
        # 1. Determine scales
        x_name = x.name
        y_name = y.name
        x_base = _get_log_base(variable_scales.get(x_name))
        y_base = _get_log_base(variable_scales.get(y_name))
        
        # 2. Prepare data for fitting
        # Create temp DF to handle NaNs and alignment
        df_fit = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        # Filter for log validity (must be > 0)
        if x_base:
            df_fit = df_fit[df_fit['x'] > 0]
        if y_base:
            df_fit = df_fit[df_fit['y'] > 0]
            
        if len(df_fit) < 2:
            return

        x_fit = df_fit['x'].values
        y_fit = df_fit['y'].values
        
        # 3. Transform to log space if needed
        # np.log is natural log. log_b(x) = ln(x) / ln(b)
        x_trans = np.log(x_fit) / np.log(x_base) if x_base else x_fit
        y_trans = np.log(y_fit) / np.log(y_base) if y_base else y_fit
        
        # 4. Fit linear model in transformed space: Y = mX + c
        try:
            slope, intercept = np.polyfit(x_trans, y_trans, 1)
        except Exception:
            return

        # 5. Generate line points in transformed space
        x_min, x_max = x_trans.min(), x_trans.max()
        # Create 100 points for smooth line
        x_line_trans = np.linspace(x_min, x_max, 100)
        y_line_trans = slope * x_line_trans + intercept
        
        # 6. Inverse transform back to original space for plotting
        x_line = x_base ** x_line_trans if x_base else x_line_trans
        y_line = y_base ** y_line_trans if y_base else y_line_trans
        
        # 7. Plot the fitted line
        # Since the axis itself will be set to log scale by _apply_scales_and_labels,
        # plotting these points will appear as a straight line.
        plt.plot(x_line, y_line, color=color, linewidth=1.5)

    # --- 1. Combined plot: all targets in one grid, color-coded ---
    if plot_combined:
        g = sns.PairGrid(
            plot_data,
            x_vars=x_vars,
            y_vars=y_vars,
            hue=hue_col,
            palette=palette_dict,
            diag_sharey=False,
            corner=False,
        )

        # Map diagonal ONLY if square mode
        if not is_rectangular:
            g.map_diag(_diag_wrapper, linewidth=2)

        # Map off-diagonal (or all cells in rectangular mode)
        map_func = g.map if is_rectangular else g.map_offdiag
        
        if add_best_fit_line:
            # Use custom regplot that handles log-space fitting
            map_func(custom_regplot)
        else:
            map_func(sns.scatterplot, s=marker_size, alpha=marker_alpha, linewidth=0)

        # Apply scales and labels
        _apply_scales_and_labels(g, x_vars, y_vars)

        # Move legend outside
        g.add_legend(title="Target Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")

        title_suffix_mode = "(Rectangular)" if is_rectangular else "(Square)"
        g.fig.suptitle(f"Scatter Plot Matrix {title_suffix_mode} (all targets){title_suffix}", y=1.02, fontsize=14)
        g.fig.tight_layout()

        fname = build_filename("scatter_matrix.png", comparison_name)
        path = os.path.join(output_dir, fname)
        g.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  Saved: {path}")

    # --- 2. Per-target plots: one .png per target ---
    if plot_per_target:
        print(f"  Generating per-target scatter matrix files for {len(categories)} target(s)...")
        for cat in categories:
            cat_data = plot_data[plot_data[hue_col] == cat]
            if cat_data.empty:
                continue

            # Single color for this target
            cat_color = color_map[cat]
            single_palette = {cat: cat_color}

            g = sns.PairGrid(
                cat_data,
                x_vars=x_vars,
                y_vars=y_vars,
                hue=hue_col,
                palette=single_palette,
                diag_sharey=False,
                corner=False,
            )

            if not is_rectangular:
                g.map_diag(_diag_wrapper, linewidth=2)

            map_func = g.map if is_rectangular else g.map_offdiag

            if add_best_fit_line:
                map_func(custom_regplot)
            else:
                map_func(sns.scatterplot, s=marker_size, alpha=marker_alpha, linewidth=0)

            _apply_scales_and_labels(g, x_vars, y_vars)

            g.add_legend(title="Target Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")

            g.fig.suptitle(f"Scatter Plot Matrix: {cat}{title_suffix}", y=1.02, fontsize=12)
            g.fig.tight_layout()

            slug = slugify_target_label(cat)
            fname = build_filename(f"scatter_matrix_target_{slug}.png", comparison_name)
            path = os.path.join(output_dir, fname)
            g.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(g.fig)
            print(f"  Saved: {path}")
