"""
Plotting — Regression diagnostic plots and trajectory comparison plots.

Regression plots (9):
  1. Partial regression (added-variable)
  2. Partial residual (CCPR)
  3. Residuals vs. fitted
  4. Normal Q-Q
  5. Scale-location
  6. Cook's distance
  7. Leverage vs. residuals
  8. Correlation heatmap
  9. Pairplot

Trajectory plots (2):
  10. Grouped trajectory (mean ± std/IQR per group)
  11. Trajectory comparison bar chart (AUC, final_vs_initial, etc.)
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server/headless use
import matplotlib.pyplot as plt

from .regression_analysis import RegressionResult
from .data_processing import unsanitize_name, sanitize_key


# ---------------------------------------------------------------------------
# Configuration / style
# ---------------------------------------------------------------------------

_DEFAULT_STYLE = {
    "figure.figsize": (10, 7),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}


def _setup_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(_DEFAULT_STYLE)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            pass  # fall back to default


def _ensure_dir(path: str) -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, output_dir: str, filename: str, fmt: str = "png", dpi: int = 150):
    """Save a figure and close it."""
    dirpath = _ensure_dir(output_dir)
    filepath = dirpath / f"{filename}.{fmt}"
    fig.savefig(str(filepath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Master functions
# ---------------------------------------------------------------------------

def plot_all_regression(
    result: RegressionResult,
    diagnostics: Dict[str, Any],
    plots_config: Dict[str, bool],
    output_config: Dict[str, Any],
    group_name: Optional[str] = None,
) -> List:
    """
    Generate all enabled regression diagnostic plots.

    Parameters
    ----------
    result : RegressionResult
    diagnostics : dict
        Output from compute_diagnostics.
    plots_config : dict
        Boolean flags for each plot type.
    output_config : dict
        Output settings (save_figures, figure_format, figure_dpi, output_dir).
    group_name : str, optional
        Group label for file naming.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    _setup_style()

    output_dir = output_config.get("output_dir", "outputs")
    fmt = output_config.get("figure_format", "png")
    dpi = output_config.get("figure_dpi", 150)
    save = output_config.get("save_figures", True)

    # Create group-specific subdirectory
    if group_name and group_name != "all":
        output_dir = os.path.join(output_dir, f"group_{group_name}")

    prefix = f"{group_name}_" if group_name and group_name != "all" else ""
    figures = []

    if plots_config.get("partial_regression", False):
        fig = plot_partial_regression(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}partial_regression", fmt, dpi)
        if fig:
            figures.append(fig)

    if plots_config.get("partial_residual", False):
        fig = plot_partial_residual(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}partial_residual", fmt, dpi)
        if fig:
            figures.append(fig)

    if plots_config.get("residuals_vs_fitted", False):
        fig = plot_residuals_vs_fitted(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}residuals_vs_fitted", fmt, dpi)
        figures.append(fig)

    if plots_config.get("qq_plot", False):
        fig = plot_qq(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}qq_plot", fmt, dpi)
        figures.append(fig)

    if plots_config.get("scale_location", False):
        fig = plot_scale_location(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}scale_location", fmt, dpi)
        figures.append(fig)

    if plots_config.get("cooks_distance", False):
        inf = diagnostics.get("influence", {})
        cooks_d = inf.get("cooks_distance")
        if cooks_d is not None:
            fig = plot_cooks_distance(cooks_d, inf.get("cooks_distance_threshold", 0))
            if fig and save:
                _save_fig(fig, output_dir, f"{prefix}cooks_distance", fmt, dpi)
            figures.append(fig)

    if plots_config.get("leverage_vs_residuals", False):
        inf = diagnostics.get("influence", {})
        leverage = inf.get("leverage")
        cooks_d = inf.get("cooks_distance")
        if leverage is not None:
            fig = plot_leverage_vs_residuals(result, leverage, cooks_d)
            if fig and save:
                _save_fig(fig, output_dir, f"{prefix}leverage_vs_residuals", fmt, dpi)
            figures.append(fig)

    if plots_config.get("correlation_heatmap", False):
        fig = plot_correlation_heatmap(result.X)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}correlation_heatmap", fmt, dpi)
        figures.append(fig)

    if plots_config.get("pairplot", False):
        fig = plot_pairplot(result)
        if fig and save:
            _save_fig(fig, output_dir, f"{prefix}pairplot", fmt, dpi)
        if fig:
            figures.append(fig)

    return figures


def plot_all_trajectory(
    df: pd.DataFrame,
    trajectory_config: Dict[str, Any],
    output_config: Dict[str, Any],
) -> List:
    """
    Generate all trajectory comparison plots.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame.
    trajectory_config : dict
        The ``analysis.trajectory`` section of the YAML config.
    output_config : dict
        Output settings.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    _setup_style()

    output_dir = output_config.get("output_dir", "outputs")
    fmt = output_config.get("figure_format", "png")
    dpi = output_config.get("figure_dpi", 150)
    save = output_config.get("save_figures", True)

    x_axis = trajectory_config.get("x_axis", "task_idx")
    y_variables = trajectory_config.get("y_variables", [])
    group_by = trajectory_config.get("group_by")
    summary_stats = trajectory_config.get("summary_statistics", ["mean", "std"])
    comparisons = trajectory_config.get("comparisons", [])

    # Resolve column names
    x_col = sanitize_key(x_axis) if sanitize_key(x_axis) in df.columns else x_axis

    if group_by:
        if isinstance(group_by, str):
            group_col = sanitize_key(group_by) if sanitize_key(group_by) in df.columns else group_by
        else:
            group_col = group_by[0]  # Use first for now
            group_col = sanitize_key(group_col) if sanitize_key(group_col) in df.columns else group_col
    else:
        group_col = None

    figures = []

    # --- Trajectory plots ---
    for y_var in y_variables:
        y_col = sanitize_key(y_var) if sanitize_key(y_var) in df.columns else y_var
        if y_col not in df.columns:
            warnings.warn(f"Trajectory variable '{y_var}' not found; skipping.")
            continue

        fig = plot_grouped_trajectory(
            df, x_col, y_col, group_col, summary_stats
        )
        if fig and save:
            _save_fig(fig, output_dir, f"trajectory_{y_var}", fmt, dpi)
        if fig:
            figures.append(fig)

    # --- Comparison table ---
    if comparisons and group_col:
        comp_df = compute_trajectory_comparisons(df, x_col, y_variables, group_col, comparisons)
        if comp_df is not None and not comp_df.empty:
            fig = plot_trajectory_comparison_table(comp_df, comparisons)
            if fig and save:
                _save_fig(fig, output_dir, "trajectory_comparison", fmt, dpi)
            figures.append(fig)

            # Also print the table
            print("\n--- Trajectory Comparison ---")
            print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return figures


# ---------------------------------------------------------------------------
# Regression diagnostic plots
# ---------------------------------------------------------------------------

def plot_partial_regression(result: RegressionResult):
    """
    Partial regression (added-variable) plots.

    Shows the effect of adding each predictor after accounting for all others.
    """
    if result.backend != "statsmodels":
        warnings.warn("Partial regression plots require statsmodels backend.")
        return None

    try:
        import statsmodels.graphics.regressionplots as smplots
        fig = plt.figure(figsize=(14, 10))
        smplots.plot_partregress_grid(result.model, fig=fig)
        fig.suptitle("Partial Regression Plots", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig
    except Exception as e:
        warnings.warn(f"Partial regression plot failed: {e}")
        return None


def plot_partial_residual(result: RegressionResult):
    """
    Partial residual (CCPR) plots.

    Shows the relationship between each predictor and the response,
    adjusted for other predictors.
    """
    if result.backend != "statsmodels":
        warnings.warn("Partial residual plots require statsmodels backend.")
        return None

    try:
        import statsmodels.graphics.regressionplots as smplots
        fig = plt.figure(figsize=(14, 10))
        smplots.plot_ccpr_grid(result.model, fig=fig)
        fig.suptitle("Partial Residual (CCPR) Plots", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig
    except Exception as e:
        warnings.warn(f"Partial residual plot failed: {e}")
        return None


def plot_residuals_vs_fitted(result: RegressionResult):
    """Residuals vs. fitted values — checks for non-linearity and heteroscedasticity."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(result.fitted_values, result.residuals, alpha=0.5, edgecolors="k", linewidths=0.5, s=30)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)

    # LOWESS smoothing line
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(result.residuals, result.fitted_values, frac=0.3)
        ax.plot(lowess[:, 0], lowess[:, 1], color="orange", linewidth=2, label="LOWESS")
        ax.legend()
    except Exception:
        pass

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    title = "Residuals vs. Fitted Values"
    if result.group_name and result.group_name != "all":
        title += f" [{result.group_name}]"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_qq(result: RegressionResult):
    """Normal Q-Q plot of residuals."""
    try:
        import statsmodels.api as sm
        fig = sm.qqplot(result.residuals, line="45", fit=True)
        fig.set_size_inches(8, 7)
        title = "Normal Q-Q Plot of Residuals"
        if result.group_name and result.group_name != "all":
            title += f" [{result.group_name}]"
        fig.axes[0].set_title(title)
        fig.tight_layout()
        return fig
    except Exception as e:
        warnings.warn(f"Q-Q plot failed: {e}")
        fig, ax = plt.subplots(figsize=(8, 7))
        from scipy import stats as sp_stats
        sp_stats.probplot(result.residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot of Residuals")
        fig.tight_layout()
        return fig


def plot_scale_location(result: RegressionResult):
    """Scale-location plot — sqrt(|standardized residuals|) vs fitted values."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Standardized residuals
    resid_std = result.residuals / np.std(result.residuals) if np.std(result.residuals) > 0 else result.residuals
    sqrt_abs_resid = np.sqrt(np.abs(resid_std))

    ax.scatter(result.fitted_values, sqrt_abs_resid, alpha=0.5, edgecolors="k", linewidths=0.5, s=30)

    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(sqrt_abs_resid, result.fitted_values, frac=0.3)
        ax.plot(lowess[:, 0], lowess[:, 1], color="red", linewidth=2, label="LOWESS")
        ax.legend()
    except Exception:
        pass

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
    title = "Scale-Location Plot"
    if result.group_name and result.group_name != "all":
        title += f" [{result.group_name}]"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cooks_distance(cooks_d, threshold: float = 0.0):
    """Cook's distance bar/stem plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(cooks_d)
    indices = np.arange(n)

    ax.stem(indices, cooks_d, linefmt="C0-", markerfmt="C0o", basefmt="k-")

    if threshold > 0:
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1,
                    label=f"Threshold (4/n = {threshold:.4f})")
        ax.legend()

    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Cook's Distance")
    fig.tight_layout()
    return fig


def plot_leverage_vs_residuals(result: RegressionResult, leverage, cooks_d=None):
    """Residuals vs. leverage plot with optional Cook's distance contours."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Standardized residuals
    resid_std = result.residuals / np.std(result.residuals) if np.std(result.residuals) > 0 else result.residuals

    scatter = ax.scatter(leverage, resid_std, alpha=0.5, edgecolors="k", linewidths=0.5, s=30)

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)

    # Cook's distance contours
    if cooks_d is not None:
        k = result.n_predictors
        n = result.n_obs
        # Draw Cook's D = 0.5 and 1.0 contour lines
        x_range = np.linspace(0.001, max(leverage) * 1.1, 100)
        for d_level, ls in [(0.5, "--"), (1.0, "-")]:
            y_pos = np.sqrt(d_level * k * (1 - x_range) / x_range)
            ax.plot(x_range, y_pos, color="red", linestyle=ls, linewidth=0.8, alpha=0.6)
            ax.plot(x_range, -y_pos, color="red", linestyle=ls, linewidth=0.8, alpha=0.6,
                    label=f"Cook's D = {d_level}" if d_level == 0.5 else None)

    # Leverage threshold
    k = result.n_predictors
    n = result.n_obs
    lev_thresh = 2 * (k + 1) / n
    ax.axvline(x=lev_thresh, color="orange", linestyle=":", linewidth=1,
               label=f"Leverage threshold ({lev_thresh:.3f})")

    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized Residuals")
    title = "Residuals vs. Leverage"
    if result.group_name and result.group_name != "all":
        title += f" [{result.group_name}]"
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(X: pd.DataFrame):
    """Heatmap of predictor correlations."""
    try:
        import seaborn as sns
    except ImportError:
        warnings.warn("seaborn required for correlation heatmap; install with: pip install seaborn")
        return None

    # Remove constant column
    X_no_const = X.drop(columns=["const"], errors="ignore")
    if X_no_const.shape[1] < 2:
        warnings.warn("Need >= 2 predictors for correlation heatmap.")
        return None

    # Use original names for display
    display_cols = [unsanitize_name(c) for c in X_no_const.columns]
    corr_matrix = X_no_const.corr()
    corr_matrix.index = display_cols
    corr_matrix.columns = display_cols

    fig, ax = plt.subplots(figsize=(max(8, len(display_cols) * 1.2), max(6, len(display_cols))))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Predictor Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_pairplot(result: RegressionResult):
    """Pairwise scatter matrix of predictors and response."""
    try:
        import seaborn as sns
    except ImportError:
        warnings.warn("seaborn required for pairplot; install with: pip install seaborn")
        return None

    # Build a combined DataFrame
    X_no_const = result.X.drop(columns=["const"], errors="ignore")
    plot_df = X_no_const.copy()
    dep_name = result.y.name if hasattr(result.y, "name") and result.y.name else "y"
    plot_df[dep_name] = result.y.values

    # Use original names
    rename_map = {c: unsanitize_name(c) for c in plot_df.columns}
    plot_df = plot_df.rename(columns=rename_map)

    # Limit to avoid extremely large pairplots
    if plot_df.shape[1] > 8:
        warnings.warn(f"Too many variables ({plot_df.shape[1]}) for pairplot; showing first 7 + response.")
        keep_cols = list(plot_df.columns[:7]) + [rename_map.get(dep_name, dep_name)]
        plot_df = plot_df[keep_cols]

    try:
        g = sns.pairplot(plot_df, diag_kind="hist", plot_kws={"alpha": 0.5, "s": 20})
        g.fig.suptitle("Pairwise Scatter Matrix", y=1.02)
        return g.fig
    except Exception as e:
        warnings.warn(f"Pairplot failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

def plot_grouped_trajectory(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str],
    summary_stats: List[str],
):
    """
    Plot mean curve ± std/IQR band per group over the x-axis variable.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str
        X-axis column (e.g., task_idx).
    y_col : str
        Y-axis column (e.g., epoch_loss).
    group_col : str or None
        Grouping column. If None, all data is one group.
    summary_stats : list of str
        Which summary statistics to show ("mean", "std", "median", "q25_q75").
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if group_col and group_col in df.columns:
        groups = df.groupby(group_col)
    else:
        groups = [("all", df)]

    colors = plt.cm.tab10.colors

    for idx, (group_label, group_df) in enumerate(groups):
        color = colors[idx % len(colors)]

        # Group by x_col and compute summary stats across runs
        agg_funcs = {}
        if "mean" in summary_stats:
            agg_funcs["mean"] = "mean"
        if "median" in summary_stats:
            agg_funcs["median"] = "median"
        if "std" in summary_stats:
            agg_funcs["std"] = "std"
        if "q25_q75" in summary_stats:
            agg_funcs["q25"] = lambda x: x.quantile(0.25)
            agg_funcs["q75"] = lambda x: x.quantile(0.75)

        if not agg_funcs:
            agg_funcs["mean"] = "mean"

        summary = group_df.groupby(x_col)[y_col].agg(**agg_funcs).reset_index()
        summary = summary.sort_values(x_col)

        x_vals = summary[x_col].values

        # Plot mean or median line
        if "mean" in summary.columns:
            ax.plot(x_vals, summary["mean"].values, color=color, linewidth=1.5,
                    label=f"{group_label}")
            center_col = "mean"
        elif "median" in summary.columns:
            ax.plot(x_vals, summary["median"].values, color=color, linewidth=1.5,
                    label=f"{group_label}")
            center_col = "median"
        else:
            center_col = None

        # Plot shaded band
        if "std" in summary.columns and center_col:
            center = summary[center_col].values
            std_vals = summary["std"].values
            ax.fill_between(x_vals, center - std_vals, center + std_vals,
                            alpha=0.15, color=color)

        if "q25" in summary.columns and "q75" in summary.columns:
            ax.fill_between(x_vals, summary["q25"].values, summary["q75"].values,
                            alpha=0.15, color=color, linestyle="--")

    ax.set_xlabel(unsanitize_name(x_col))
    ax.set_ylabel(unsanitize_name(y_col))
    ax.set_title(f"{unsanitize_name(y_col)} over {unsanitize_name(x_col)}")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Trajectory comparison computations
# ---------------------------------------------------------------------------

def compute_trajectory_comparisons(
    df: pd.DataFrame,
    x_col: str,
    y_variables: List[str],
    group_col: str,
    comparisons: List[str],
) -> Optional[pd.DataFrame]:
    """
    Compute quantitative trajectory comparisons per group.

    Returns a DataFrame with one row per (group, y_variable) and columns
    for each comparison metric.
    """
    rows = []

    for y_var in y_variables:
        y_col = sanitize_key(y_var) if sanitize_key(y_var) in df.columns else y_var
        if y_col not in df.columns:
            continue

        for group_label, group_df in df.groupby(group_col):
            # Aggregate: mean across runs at each x value
            agg = group_df.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)
            x_vals = agg[x_col].values
            y_vals = agg[y_col].values

            if len(y_vals) < 2:
                continue

            row = {
                "group": group_label,
                "metric": unsanitize_name(y_var),
            }

            if "auc" in comparisons:
                row["AUC"] = float(np.trapz(y_vals, x_vals))

            if "final_vs_initial" in comparisons:
                first_val = y_vals[0]
                last_val = y_vals[-1]
                row["final_value"] = float(last_val)
                row["initial_value"] = float(first_val)
                row["final_vs_initial_ratio"] = float(last_val / first_val) if first_val != 0 else np.nan
                row["final_minus_initial"] = float(last_val - first_val)

            if "max_degradation" in comparisons:
                best = np.min(y_vals)  # for loss, best = lowest
                worst = np.max(y_vals)
                row["best_value"] = float(best)
                row["worst_value"] = float(worst)
                row["max_degradation"] = float(worst - best)

            if "recovery_events" in comparisons:
                # Count how many times the metric decreases after increasing
                diffs = np.diff(y_vals)
                # A "recovery" is a negative diff following a positive diff
                n_recovery = 0
                for i in range(1, len(diffs)):
                    if diffs[i] < 0 and diffs[i - 1] > 0:
                        n_recovery += 1
                row["recovery_events"] = n_recovery

            rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_trajectory_comparison_table(
    comp_df: pd.DataFrame,
    comparisons: List[str],
):
    """
    Create a bar chart comparing groups on key trajectory metrics.
    """
    # Select numeric comparison columns
    metric_cols = []
    if "auc" in comparisons and "AUC" in comp_df.columns:
        metric_cols.append("AUC")
    if "final_vs_initial" in comparisons and "final_minus_initial" in comp_df.columns:
        metric_cols.append("final_minus_initial")
    if "max_degradation" in comparisons and "max_degradation" in comp_df.columns:
        metric_cols.append("max_degradation")

    if not metric_cols:
        return None

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_col in zip(axes, metric_cols):
        # If multiple y-variables, pick the first for simplicity
        if "metric" in comp_df.columns:
            first_metric = comp_df["metric"].unique()[0]
            plot_data = comp_df[comp_df["metric"] == first_metric]
        else:
            plot_data = comp_df

        groups = plot_data["group"].values
        values = plot_data[metric_col].values

        bars = ax.bar(range(len(groups)), values, color=plt.cm.tab10.colors[:len(groups)],
                       edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel(metric_col)
        ax.set_title(metric_col)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Trajectory Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
