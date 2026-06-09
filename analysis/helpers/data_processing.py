import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Tuple

def sanitize_key(key: str) -> str:
    """Sanitize a key (e.g. config path) for use as a DataFrame column."""
    return key.replace(".", "__")

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize all column names in the DataFrame."""
    df.columns = [sanitize_key(str(c)) for c in df.columns]
    return df

def unsanitize_name(name: str) -> str:
    """Reverse sanitization for display (replace __ with .)."""
    return name.replace("__", ".")

def encode_categoricals(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical columns.
    Returns (transformed_df, list_of_new_column_names).
    """
    if not columns:
        return df, []
    
    # Ensure columns exist
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return df, []
        
    df_encoded = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols, prefix_sep="__")
    
    # Identify new columns
    new_cols = [c for c in df_encoded.columns if any(c.startswith(vc + "__") for v in valid_cols)]
    
    return df_encoded, new_cols

def apply_transforms(df: pd.DataFrame, transforms: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a list of transformations to the DataFrame.
    Example transform: {'column': 'epoch_loss', 'method': 'log', 'base': 10}
    """
    df_t = df.copy()
    for t in transforms:
        col = t.get("column")
        method = t.get("method")
        if col not in df_t.columns:
            continue
            
        if method == "log":
            base = t.get("base", 10)
            df_t[col] = np.log(df_t[col]) / np.log(base)
        elif method == "sqrt":
            df_t[col] = np.sqrt(df_t[col])
        elif method == "square":
            df_t[col] = df_t[col] ** 2
            
    return df_t

def get_design_matrix(
    df: pd.DataFrame, 
    dep_var: str, 
    indep_vars: List[str], 
    add_constant: bool = True
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Extract response vector y and design matrix X from DataFrame.
    """
    y = df[dep_var]
    X = df[indep_vars].copy()
    
    if add_constant:
        import statsmodels.api as sm
        X = sm.add_constant(X, has_constant='add')
        
    return y, X

def clean_dataframe(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Drop rows with NaNs in required columns."""
    return df.dropna(subset=required_columns)

def filter_outliers(df: pd.DataFrame, outlier_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filter rows based on outlier detection in specified columns."""
    if not outlier_cfg.get("enabled", False):
        return df

    method = outlier_cfg.get("method", "quantile")
    # If columns not specified, default to forward_loss and epoch_loss if present
    target_cols = outlier_cfg.get("columns", [])
    
    # If empty, we might want to be careful. Let's just return if empty to avoid over-filtering.
    if not target_cols:
        return df

    # Check for missing columns
    existing_cols = []
    missing_cols = []
    for col in target_cols:
        if col in df.columns:
            existing_cols.append(col)
        else:
            missing_cols.append(col)
            
    if missing_cols:
        raise ValueError(
            f"Outlier filtering configuration error: The following columns are missing from the data: {missing_cols}. "
            f"Available columns: {sorted(list(df.columns))}"
        )

    # Filter
    mask = pd.Series(True, index=df.index)
    
    # Check if we should filter per-group (target label)
    # If the config implies per-group filtering, we should calculate quantiles per group.
    # The user request implies "per group of target runs".
    # We'll assume per-group filtering is the desired behavior if "_target_label" exists.
    
    group_col = "_target_label"
    
    for col in existing_cols:
        series = df[col]
        
        if method == "quantile":
            low_q = outlier_cfg.get("lower_quantile", 0.01)
            high_q = outlier_cfg.get("upper_quantile", 0.99)
            
            if group_col in df.columns:
                # Per-group quantile calculation
                # Calculate bounds per group
                bounds = df.groupby(group_col)[col].agg(
                    low=lambda x: x.quantile(low_q),
                    high=lambda x: x.quantile(high_q)
                )
                
                # Map bounds back to the original dataframe index
                # This is vectorized and faster than iterating
                merged = df[[group_col, col]].merge(bounds, left_on=group_col, right_index=True, how="left")
                
                # Create mask: keep if within bounds OR is NaN (NaNs handled by audit)
                col_mask = (merged[col] >= merged["low"]) & (merged[col] <= merged["high"])
                col_mask = col_mask | merged[col].isna()
                
                # Align mask index with original df
                col_mask.index = df.index
                mask &= col_mask
                
            else:
                # Global quantile calculation (fallback)
                lower_bound = series.quantile(low_q)
                upper_bound = series.quantile(high_q)
                col_mask = (series >= lower_bound) & (series <= upper_bound)
                col_mask = col_mask | series.isna()
                mask &= col_mask
        
        elif method == "iqr":
            k = outlier_cfg.get("iqr_multiplier", 1.5)
            
            if group_col in df.columns:
                def get_iqr_bounds(x):
                    q1 = x.quantile(0.25)
                    q3 = x.quantile(0.75)
                    iqr = q3 - q1
                    return pd.Series([q1 - k * iqr, q3 + k * iqr], index=['low', 'high'])

                bounds = df.groupby(group_col)[col].apply(get_iqr_bounds).unstack()
                merged = df[[group_col, col]].merge(bounds, left_on=group_col, right_index=True, how="left")
                
                col_mask = (merged[col] >= merged["low"]) & (merged[col] <= merged["high"])
                col_mask = col_mask | merged[col].isna()
                col_mask.index = df.index
                mask &= col_mask
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - k * iqr
                upper_bound = q3 + k * iqr
                col_mask = (series >= lower_bound) & (series <= upper_bound)
                col_mask = col_mask | series.isna()
                mask &= col_mask
            
        elif method == "z_score":
            threshold = outlier_cfg.get("threshold", 3.0)
            
            if group_col in df.columns:
                # Calculate z-score per group
                # (x - mean) / std
                groups = df.groupby(group_col)[col]
                z_scores = groups.transform(lambda x: (x - x.mean()) / x.std())
                
                col_mask = z_scores.abs() <= threshold
                col_mask = col_mask | df[col].isna() # Keep NaNs
                mask &= col_mask
            else:
                mean = series.mean()
                std = series.std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                col_mask = (series >= lower_bound) & (series <= upper_bound)
                col_mask = col_mask | series.isna()
                mask &= col_mask
            
        else:
            warnings.warn(f"Unknown outlier method '{method}'. Skipping.")
            continue

    before = len(df)
    df_filtered = df[mask].copy()
    after = len(df_filtered)
    if before != after:
        print(f"  Outlier filtering ({method}): dropped {before - after} rows ({before} -> {after})")
        
    return df_filtered

def apply_history_sampling(df: pd.DataFrame, sampling_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Keep only the last row per (run_id, task_idx) group, sorted by global_epoch."""
    method = sampling_cfg.get("method", "all")
    if method != "last_per_group":
        return df

    group_by = sampling_cfg["group_by"]
    sort_by = sampling_cfg["sort_by"]

    before = len(df)
    df = (
        df.sort_values(sort_by)
        .groupby(["run_id", group_by])
        .last()
        .reset_index()
    )
    print(f"  History sampling (last_per_group): {before} -> {len(df)} rows")
    return df

def compute_weight_norm_column(df: pd.DataFrame,
                               weight_norm_keys: List[str],
                               aggregation: str) -> pd.DataFrame:
    """Aggregate per-layer weight norms into a single scalar metric."""
    if not weight_norm_keys:
        warnings.warn("No weight norm keys found; weight_norm_mean will be NaN")
        df["weight_norm_mean"] = np.nan
        return df

    existing = [k for k in weight_norm_keys if k in df.columns]
    if not existing:
        warnings.warn("Weight norm keys not present in DataFrame; weight_norm_mean will be NaN")
        df["weight_norm_mean"] = np.nan
        return df

    # W&B may return missing values as string 'NaN'; coerce to proper numeric.
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if aggregation == "weight_norm_mean":
        df["weight_norm_mean"] = df[existing].mean(axis=1)
        print(f"  Computed weight_norm_mean from {len(existing)} layer columns")
    elif aggregation == "weight_norm_all":
        df["weight_norm_mean"] = df[existing].mean(axis=1)
        print(f"  Kept all {len(existing)} weight norm columns + weight_norm_mean")
    else:
        raise ValueError(f"Unknown weight_norm aggregation: {aggregation!r}")

    return df

def compute_forward_loss(df: pd.DataFrame,
                         forward_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute forward_loss for each row by looking ahead into future tasks."""
    window_size = forward_cfg["window_size"]
    metric = forward_cfg["metric_to_average"]
    method = forward_cfg["calculation_method"]
    ema_alpha = forward_cfg.get("ema_alpha", 0.3)

    if metric not in df.columns:
        raise KeyError(
            f"metric_to_average '{metric}' not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    results = []
    for run_id, run_df in df.groupby("run_id"):
        if run_df["task_idx"].duplicated().any():
            raise ValueError(
                f"Duplicate task_idx found for run_id={run_id}. "
                "Forward loss calculation requires unique task_idx per run. "
                "Please check 'history_sampling' in config (ensure method='last_per_group')."
            )

        run_df = run_df.sort_values("task_idx").reset_index(drop=True)
        n = len(run_df)
        fwd_loss = np.full(n, np.nan)

        for i in range(n):
            start = i + 1
            end = min(i + 1 + window_size, n)
            if start >= n:
                continue

            window_vals = run_df[metric].iloc[start:end].values
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) == 0:
                continue

            if method == "mean":
                fwd_loss[i] = np.mean(valid)

            elif method == "median":
                fwd_loss[i] = np.median(valid)

            elif method == "ema":
                # Calculate weights based on actual temporal distance (indices) to handle NaNs
                # and correct for finite window bias via normalization
                valid_indices = np.where(~np.isnan(window_vals))[0]
                weights = (1.0 - ema_alpha) ** valid_indices
                fwd_loss[i] = np.average(valid, weights=weights)

            elif method == "linear_decay":
                weights = np.linspace(1.0, 0.0, len(valid), endpoint=False)
                if weights.sum() > 0:
                    fwd_loss[i] = np.average(valid, weights=weights)

            else:
                raise ValueError(f"Unknown forward_loss calculation_method: {method!r}")

        run_df = run_df.copy()
        run_df["forward_loss"] = fwd_loss
        results.append(run_df)

    df = pd.concat(results, ignore_index=True)
    valid_count = df["forward_loss"].notna().sum()
    print(f"  Computed forward_loss ({method}, w={window_size}): {valid_count} valid values")
    return df

def apply_variable_transformations(
    df: pd.DataFrame,
    scales: Dict[str, str]
) -> pd.DataFrame:
    """
    Apply transformations to DataFrame columns based on scaling config.
    Returns a new DataFrame with transformed columns (renamed with suffix).
    
    Used for statistical analysis where regression needs linear inputs.
    """
    df_transformed = df.copy()
    
    for var, scale in scales.items():
        if var not in df.columns:
            continue
            
        if not scale or scale == "linear":
            continue
            
        # Determine base
        base = None
        if scale == "log":
            base = 10.0
        elif scale.startswith("log"):
            try:
                base = float(scale[3:])
            except ValueError:
                pass
        
        if base:
            # Filter <= 0
            valid_mask = df_transformed[var] > 0
            if not valid_mask.all():
                n_invalid = (~valid_mask).sum()
                warnings.warn(f"Variable '{var}' has {n_invalid} values <= 0; cannot apply {scale} transform. Dropping these rows.")
                df_transformed = df_transformed[valid_mask]
            
            # Apply transform
            # log_b(x) = ln(x) / ln(b)
            new_col_name = f"{var} ({scale})"
            df_transformed[new_col_name] = np.log(df_transformed[var]) / np.log(base)
            
            # Drop original, or keep? Usually we want to replace for analysis
            # But let's keep original and just return the new one, 
            # BUT the caller needs to know the new name.
            # Actually, for regression config, we specified "epoch_loss".
            # If we change the column name, we must update the config passed to regression.
            # It's better to RENAME the column so downstream config works if we update config too.
            # OR we return a map of old_name -> new_name
            
            # User requirement: "add a bracket to the transformed variable ... in all the tables"
            # So we should rename the column in the DF.
            # df_transformed = df_transformed.rename(columns={var: new_col_name})
            
    return df_transformed

def audit_nan_per_run(
    df: pd.DataFrame,
    base_metrics: List[str],
    drop_threshold: Optional[float],
) -> Any: # Tuple[pd.DataFrame, List[str], Dict[str, List[Dict[str, Any]]]]
    """Audit NaN values per run, optionally drop, and return structured info."""
    
    # Guardrail: Check for missing base metrics if dropping is enabled
    if drop_threshold is not None:
        missing_metrics = [m for m in base_metrics if m not in df.columns]
        # Ignore metrics that are not in DF but also not in base_metrics (logic below handles intersection)
        # But here we want to warn if a REQUIRED metric is missing.
        # base_metrics comes from determine_required_base_metrics.
        if missing_metrics:
             raise ValueError(
                f"NaN drop filter enabled (threshold={drop_threshold}), but the following "
                f"metrics are missing from the DataFrame: {missing_metrics}. "
                "Check if these metrics are correctly logged to W&B or configured in plotting_config.yaml."
            )

    audit_cols = [c for c in base_metrics if c in df.columns]
    for extra in ("weight_norm_mean", "forward_loss"):
        if extra in df.columns and extra not in audit_cols:
            audit_cols.append(extra)

    dropped_ids: List[str] = []
    nan_details: Dict[str, List[Dict[str, Any]]] = {}

    for (run_id, run_name, label), grp in df.groupby(["run_id", "run_name", "_target_label"]):
        n_rows = len(grp)
        max_frac = 0.0
        run_detail: List[Dict[str, Any]] = []

        for col in audit_cols:
            if col not in grp.columns:
                continue
            n_nan = int(grp[col].isna().sum())
            if n_nan > 0:
                frac = n_nan / n_rows
                max_frac = max(max_frac, frac)
                nan_tasks = grp.loc[grp[col].isna(), "task_idx"]
                task_lo = int(nan_tasks.min()) if not nan_tasks.empty else None
                task_hi = int(nan_tasks.max()) if not nan_tasks.empty else None
                run_detail.append({
                    "col": col, "n_nan": n_nan, "n_rows": n_rows,
                    "frac": frac, "task_lo": task_lo, "task_hi": task_hi,
                })

        dropped = drop_threshold is not None and max_frac > drop_threshold
        if dropped:
            dropped_ids.append(run_id)
        if run_detail:
            for d in run_detail:
                d["dropped"] = dropped
            nan_details[run_id] = run_detail

    if dropped_ids:
        before = df["run_id"].nunique()
        df = df[~df["run_id"].isin(dropped_ids)].reset_index(drop=True)
        after = df["run_id"].nunique()
        print(f"  NaN filter: dropped {before - after} run(s), {after} remaining")

    return df, dropped_ids, nan_details

def build_pipeline_summary(
    target_inventory: List[Dict[str, Any]],
    base_metrics: List[str],
    drop_threshold: Optional[float],
    combined_pre_filter: Optional[pd.DataFrame],
    combined_post_filter: Optional[pd.DataFrame],
    nan_details: Dict[str, List[Dict[str, Any]]],
    dropped_ids: List[str],
) -> str:
    """Build a comprehensive plain-text pipeline summary report."""
    from datetime import datetime

    L: List[str] = []
    L.append("=" * 76)
    L.append(f"Pipeline Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 76)
    L.append("")

    # --- Section 1: Per-target inventory ---
    L.append("TARGET INVENTORY")
    L.append("-" * 76)

    for inv in target_inventory:
        idx = inv["index"]
        diff_label = inv["diff_label"]
        full_flat = inv["full_flat"]
        runs_found = inv["runs_found"]
        run_names = inv.get("run_names", [])
        run_ids = inv.get("run_ids", [])
        n_rows_raw = inv.get("n_rows_raw", 0)
        n_rows_sampled = inv.get("n_rows_sampled", 0)
        fwd_valid = inv.get("forward_loss_valid", 0)

        L.append(f"\nTarget {idx + 1}: {diff_label}")
        L.append(f"  Full config: {full_flat}")
        L.append(f"  W&B runs found:  {runs_found}")

        if runs_found == 0:
            L.append(f"  >>> SKIPPED (no matching runs)")
            continue

        # Per-run details
        runs_kept = 0
        runs_dropped = 0
        for name, rid in zip(run_names, run_ids):
            was_dropped = rid in dropped_ids
            status = "DROPPED" if was_dropped else "KEPT"
            if was_dropped:
                runs_dropped += 1
            else:
                runs_kept += 1

            nan_info = nan_details.get(rid, [])
            if nan_info:
                nan_parts = []
                for d in nan_info:
                    col = d["col"]
                    task_range = f"task_idx=[{d['task_lo']}..{d['task_hi']}]" if d["task_lo"] is not None else ""
                    nan_parts.append(f"{col}: {d['n_nan']}/{d['n_rows']} ({d['frac']:.1%}) {task_range}")
                nan_str = "; ".join(nan_parts)
                L.append(f"    {name} (id={rid}): {status} — NaN: {nan_str}")
            else:
                L.append(f"    {name} (id={rid}): {status} — no NaN")

        L.append(f"  Runs kept / dropped:  {runs_kept} / {runs_dropped}")
        L.append(f"  Rows (raw / sampled): {n_rows_raw} / {n_rows_sampled}")
        L.append(f"  Valid forward_loss:   {fwd_valid}")

        # Post-filter metric summary (if target has surviving data)
        if combined_post_filter is not None and runs_kept > 0:
            target_label = inv.get("target_label")
            if target_label:
                tgt_data = combined_post_filter[combined_post_filter["_target_label"] == target_label]
                if not tgt_data.empty:
                    for metric in ["epoch_loss", "forward_loss", "weight_norm_mean",
                                   "effective_rank_rank_drop_gini"]:
                        if metric in tgt_data.columns:
                            col = tgt_data[metric].dropna()
                            if not col.empty:
                                L.append(
                                    f"  {metric}: min={col.min():.4f}, "
                                    f"max={col.max():.4f}, mean={col.mean():.4f}"
                                )

    # --- Section 2: Overall summary ---
    L.append("")
    L.append("=" * 76)
    L.append("OVERALL SUMMARY")
    L.append("-" * 76)
    total_targets = len(target_inventory)
    targets_with_data = sum(1 for inv in target_inventory if inv["runs_found"] > 0)
    targets_after_filter = 0
    if combined_post_filter is not None:
        targets_after_filter = combined_post_filter["_target_label"].nunique()
    total_runs = sum(inv["runs_found"] for inv in target_inventory)
    L.append(f"  Targets configured:     {total_targets}")
    L.append(f"  Targets with W&B data:  {targets_with_data}")
    L.append(f"  Targets after filtering: {targets_after_filter}")
    L.append(f"  Total runs fetched:     {total_runs}")
    L.append(f"  Runs dropped (NaN):     {len(dropped_ids)}")
    L.append(f"  NaN drop threshold:     {drop_threshold if drop_threshold is not None else 'disabled'}")
    L.append(f"  Metrics audited:        {base_metrics}")
    L.append("=" * 76)

    return "\n".join(L)

def determine_required_base_metrics(cfg: Dict[str, Any]) -> List[str]:
    """Determine which W&B history keys to fetch (excluding weight_norm keys)."""
    metrics = {"global_epoch", "task_idx"}

    # Forward loss source metric
    metrics.add(cfg["metric_computation"]["forward_loss"]["metric_to_average"])

    # Future loss distribution may use a different raw metric
    fld_cfg = cfg["plots"].get("future_loss_distribution", {})
    if fld_cfg.get("enabled"):
        fld_metric = fld_cfg.get("metric", "epoch_loss")
        metrics.add(fld_metric)

    # Parallel coordinates variables (may be list of strings or list of dicts)
    pc_cfg = cfg["plots"].get("parallel_coordinates", {})
    if pc_cfg.get("enabled"):
        for entry in pc_cfg.get("variables_order", []):
            var = entry["name"] if isinstance(entry, dict) else entry
            if var not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                metrics.add(var)

    # Phase space metrics
    ps_cfg = cfg["plots"].get("phase_space", {})
    if ps_cfg.get("enabled"):
        x_metric = ps_cfg["x_axis"]["metric"]
        y_metric = ps_cfg["y_axis"]["metric"]
        if x_metric != "forward_loss":
            metrics.add(x_metric)
        if y_metric != "forward_loss":
            metrics.add(y_metric)

    # Scatter matrix metrics
    sm_cfg = cfg["plots"].get("scatter_matrix", {})
    if sm_cfg.get("enabled"):
        # Standard square matrix variables
        for var in sm_cfg.get("variables", []):
            if var not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                metrics.add(var)
        
        # Rectangular matrix variables (independent vs dependent)
        indep_vars = sm_cfg.get("target_independent_variables", [])
        dep_vars = sm_cfg.get("target_dependent_variables", [])
        for var in indep_vars + dep_vars:
            if var not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                metrics.add(var)

    # --- Analysis Section (for run_statistical_analysis.py) ---
    analysis_cfg = cfg.get("analysis", {})
    if analysis_cfg:
        # Regression
        reg_cfg = analysis_cfg.get("regression", {})
        if reg_cfg:
            dep = reg_cfg.get("dependent_variable")
            if dep and dep not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                metrics.add(dep)
            for var in reg_cfg.get("independent_variables", []):
                if var not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                    metrics.add(var)
        
        # Trajectory
        traj_cfg = analysis_cfg.get("trajectory", {})
        if traj_cfg:
            for var in traj_cfg.get("y_variables", []):
                if var not in ("forward_loss", "weight_norm_mean", "weight_norm_all"):
                    metrics.add(var)

    return list(metrics)
