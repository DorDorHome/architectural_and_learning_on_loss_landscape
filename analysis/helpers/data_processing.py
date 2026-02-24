"""
Data Processing — DataFrame assembly, panel reduction, transforms, and encoding.

Handles the full pipeline from raw W&B data to analysis-ready design matrices:
  1. Column name sanitization (dots → double underscores)
  2. History sampling / panel reduction (last_per_group, etc.)
  3. Post-reduction filtering on history columns
  4. Optional per-run derived metrics
  5. Cleaning (drop NaN, report missingness)
  6. Categorical encoding (one-hot)
  7. Variable transforms (log, polynomial, interaction)
  8. Design matrix construction for statsmodels
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name sanitization
# ---------------------------------------------------------------------------

# Bidirectional mapping so we can display original names in reports
_SANITIZE_MAP: Dict[str, str] = {}


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace dots in column names with double underscores so that they are
    valid Python identifiers (required by statsmodels formulas and pandas
    attribute access).

    Also stores the mapping so ``unsanitize_name`` can recover the original.
    """
    rename_map = {}
    for col in df.columns:
        if "." in str(col):
            new_col = str(col).replace(".", "__")
            rename_map[col] = new_col
            _SANITIZE_MAP[new_col] = col

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def unsanitize_name(name: str) -> str:
    """Return the original dot-notation name if it was sanitized, else return as-is."""
    return _SANITIZE_MAP.get(name, name)


def sanitize_key(key: str) -> str:
    """Sanitize a single key (dots → double underscores)."""
    return key.replace(".", "__")


# ---------------------------------------------------------------------------
# History sampling / panel reduction
# ---------------------------------------------------------------------------

def reduce_history(df: pd.DataFrame, sampling_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Reduce history rows according to the sampling specification.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with a ``run_id`` column and one row per logged step.
    sampling_config : dict
        Keys: ``method``, ``n``, ``steps``, ``group_by``, ``sort_by``.

    Returns
    -------
    pd.DataFrame
        Reduced DataFrame.
    """
    method = sampling_config.get("method", "all")

    if method == "all":
        return df

    elif method == "last":
        n = sampling_config.get("n", 1)
        return df.groupby("run_id").tail(n).reset_index(drop=True)

    elif method == "first":
        n = sampling_config.get("n", 1)
        return df.groupby("run_id").head(n).reset_index(drop=True)

    elif method == "every_n":
        n = sampling_config.get("n", 1)
        if n < 1:
            n = 1
        return df.groupby("run_id").apply(
            lambda g: g.iloc[::n]
        ).reset_index(drop=True)

    elif method == "at_steps":
        steps = sampling_config.get("steps", [])
        if not steps:
            warnings.warn("at_steps method specified but 'steps' list is empty; returning all rows.")
            return df
        step_col = sampling_config.get("sort_by", "_step")
        return df[df[step_col].isin(steps)].reset_index(drop=True)

    elif method == "last_per_group":
        group_by = sampling_config.get("group_by")
        sort_by = sampling_config.get("sort_by")
        if not group_by or not sort_by:
            raise ValueError(
                "last_per_group requires both 'group_by' and 'sort_by' in history_sampling. "
                f"Got group_by={group_by!r}, sort_by={sort_by!r}"
            )
        # Sanitize column names if needed
        group_by_col = sanitize_key(group_by) if sanitize_key(group_by) in df.columns else group_by
        sort_by_col = sanitize_key(sort_by) if sanitize_key(sort_by) in df.columns else sort_by

        if group_by_col not in df.columns:
            raise ValueError(
                f"group_by column '{group_by}' (sanitized: '{group_by_col}') not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        if sort_by_col not in df.columns:
            raise ValueError(
                f"sort_by column '{sort_by}' (sanitized: '{sort_by_col}') not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # For each (run_id, group_by), keep the row with the max sort_by value
        before = len(df)
        df = (
            df.sort_values(sort_by_col)
            .groupby(["run_id", group_by_col])
            .last()
            .reset_index()
        )
        print(f"  last_per_group: {before} -> {len(df)} rows "
              f"(grouped by '{group_by}', sorted by '{sort_by}')")
        return df

    else:
        raise ValueError(f"Unknown history_sampling method: {method!r}")


# ---------------------------------------------------------------------------
# Post-reduction filtering
# ---------------------------------------------------------------------------

def apply_history_post_filters(
    df: pd.DataFrame,
    post_filters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Filter panel rows by ranges or explicit values on history columns.

    Parameters
    ----------
    df : pd.DataFrame
    post_filters : dict
        Keys are column names; values are either:
        - ``{min: ..., max: ...}`` for range filtering (either bound optional)
        - a list of exact values to keep

    Returns
    -------
    pd.DataFrame
    """
    if not post_filters:
        return df

    before = len(df)
    for col_name, spec in post_filters.items():
        # Try sanitized name
        col = sanitize_key(col_name) if sanitize_key(col_name) in df.columns else col_name
        if col not in df.columns:
            warnings.warn(f"Post-filter column '{col_name}' not found in DataFrame; skipping.")
            continue

        if isinstance(spec, dict):
            if "min" in spec:
                df = df[df[col] >= spec["min"]]
            if "max" in spec:
                df = df[df[col] <= spec["max"]]
        elif isinstance(spec, list):
            df = df[df[col].isin(spec)]
        else:
            # Exact value match
            df = df[df[col] == spec]

    after = len(df)
    if before != after:
        print(f"  Post-filters: {before} -> {after} rows")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-run derived metrics
# ---------------------------------------------------------------------------

def compute_derived_metrics(
    df: pd.DataFrame,
    derived_config: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute per-run aggregate metrics from panel data.

    Each derived metric collapses a run's trajectory into a single value.
    Returns a one-row-per-run DataFrame with the original config columns
    plus the new derived columns.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with ``run_id`` column.
    derived_config : list of dict
        Each dict has keys: ``name``, ``method``, and method-specific params.

    Returns
    -------
    pd.DataFrame
        One row per run.
    """
    from scipy import stats as sp_stats

    if not derived_config:
        return pd.DataFrame()

    results = []
    for run_id, run_df in df.groupby("run_id"):
        row = {"run_id": run_id}
        # Carry over config columns (constant within a run)
        for col in df.columns:
            if col not in ("run_id",) and run_df[col].nunique() <= 1:
                row[col] = run_df[col].iloc[0]

        for spec in derived_config:
            name = spec["name"]
            method = spec["method"]

            try:
                if method == "trapz":
                    y_col = sanitize_key(spec["y"]) if sanitize_key(spec["y"]) in run_df.columns else spec["y"]
                    x_col = sanitize_key(spec["x"]) if sanitize_key(spec["x"]) in run_df.columns else spec["x"]
                    sorted_df = run_df.sort_values(x_col)
                    row[name] = np.trapz(sorted_df[y_col].values, sorted_df[x_col].values)

                elif method == "linregress_slope":
                    y_col = sanitize_key(spec["y"]) if sanitize_key(spec["y"]) in run_df.columns else spec["y"]
                    x_col = sanitize_key(spec["x"]) if sanitize_key(spec["x"]) in run_df.columns else spec["x"]
                    valid = run_df[[x_col, y_col]].dropna()
                    if len(valid) >= 2:
                        slope, _, _, _, _ = sp_stats.linregress(valid[x_col], valid[y_col])
                        row[name] = slope
                    else:
                        row[name] = np.nan

                elif method == "ratio_last_first":
                    var_col = sanitize_key(spec["variable"]) if sanitize_key(spec["variable"]) in run_df.columns else spec["variable"]
                    vals = run_df[var_col].dropna()
                    if len(vals) >= 2:
                        first_val = vals.iloc[0]
                        last_val = vals.iloc[-1]
                        row[name] = last_val / first_val if first_val != 0 else np.nan
                    else:
                        row[name] = np.nan

                elif method == "mean":
                    var_col = sanitize_key(spec["variable"]) if sanitize_key(spec["variable"]) in run_df.columns else spec["variable"]
                    row[name] = run_df[var_col].mean()

                elif method == "std":
                    var_col = sanitize_key(spec["variable"]) if sanitize_key(spec["variable"]) in run_df.columns else spec["variable"]
                    row[name] = run_df[var_col].std()

                elif method == "max_minus_min":
                    var_col = sanitize_key(spec["variable"]) if sanitize_key(spec["variable"]) in run_df.columns else spec["variable"]
                    row[name] = run_df[var_col].max() - run_df[var_col].min()

                else:
                    warnings.warn(f"Unknown derived metric method: {method!r}")
                    row[name] = np.nan

            except Exception as e:
                warnings.warn(f"Error computing derived metric '{name}' for run {run_id}: {e}")
                row[name] = np.nan

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop rows with NaN in required columns and report missingness.

    Parameters
    ----------
    df : pd.DataFrame
    required_columns : list of str, optional
        Columns that must be non-NaN. If None, all columns are required.

    Returns
    -------
    pd.DataFrame
    """
    if required_columns is None:
        required_columns = list(df.columns)

    # Sanitize column names in the required list
    required_sanitized = []
    for col in required_columns:
        sanitized = sanitize_key(col)
        if sanitized in df.columns:
            required_sanitized.append(sanitized)
        elif col in df.columns:
            required_sanitized.append(col)
        else:
            warnings.warn(f"Required column '{col}' not found in DataFrame; skipping.")

    if not required_sanitized:
        return df

    before = len(df)
    missing_report = df[required_sanitized].isnull().sum()
    missing_cols = missing_report[missing_report > 0]

    if not missing_cols.empty:
        print("  Missing values per column:")
        for col, count in missing_cols.items():
            pct = 100 * count / before
            print(f"    {col}: {count} ({pct:.1f}%)")

    df_clean = df.dropna(subset=required_sanitized).reset_index(drop=True)
    after = len(df_clean)

    if before != after:
        print(f"  Dropped {before - after} rows with missing values ({before} -> {after})")

    return df_clean


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns to encode.
    drop_first : bool
        Drop the first level to avoid the dummy-variable trap.

    Returns
    -------
    (pd.DataFrame, list of str)
        Updated DataFrame and list of new column names created.
    """
    new_columns = []
    for col in columns:
        scol = sanitize_key(col) if sanitize_key(col) in df.columns else col
        if scol not in df.columns:
            continue
        if df[scol].dtype in ("object", "category", "bool"):
            n_levels = df[scol].nunique()
            if n_levels <= 1:
                warnings.warn(f"Categorical column '{col}' has only {n_levels} level(s); dropping.")
                df = df.drop(columns=[scol])
                continue

            dummies = pd.get_dummies(df[scol], prefix=scol, drop_first=drop_first, dtype=float)
            new_columns.extend(dummies.columns.tolist())
            df = pd.concat([df.drop(columns=[scol]), dummies], axis=1)

    return df, new_columns


# ---------------------------------------------------------------------------
# Variable transforms
# ---------------------------------------------------------------------------

def apply_transforms(
    df: pd.DataFrame,
    transforms_config: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Apply variable transforms (log, polynomial, interaction) as specified.

    Parameters
    ----------
    df : pd.DataFrame
    transforms_config : list of dict
        Each dict has a ``type`` key and type-specific parameters.

    Returns
    -------
    pd.DataFrame
    """
    for spec in transforms_config:
        t = spec.get("type")

        if t == "log":
            var = _resolve_col(df, spec["variable"])
            new_col = f"log_{var}"
            vals = df[var].astype(float)
            if (vals <= 0).any():
                warnings.warn(f"log transform on '{var}': {(vals <= 0).sum()} non-positive values replaced with NaN")
                vals = vals.where(vals > 0)
            df[new_col] = np.log(vals)
            print(f"  Transform: created '{new_col}'")

        elif t == "polynomial":
            var = _resolve_col(df, spec["variable"])
            degree = spec.get("degree", 2)
            for d in range(2, degree + 1):
                new_col = f"{var}_pow{d}"
                df[new_col] = df[var].astype(float) ** d
                print(f"  Transform: created '{new_col}'")

        elif t == "interaction":
            variables = spec.get("variables", [])
            if len(variables) < 2:
                warnings.warn("Interaction requires at least 2 variables; skipping.")
                continue
            resolved = [_resolve_col(df, v) for v in variables]
            new_col = "_x_".join(resolved)
            result = df[resolved[0]].astype(float)
            for v in resolved[1:]:
                result = result * df[v].astype(float)
            df[new_col] = result
            print(f"  Transform: created '{new_col}'")

        else:
            warnings.warn(f"Unknown transform type: {t!r}")

    return df


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def get_design_matrix(
    df: pd.DataFrame,
    dep_var: str,
    indep_vars: List[str],
    add_constant: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Extract the response vector y and design matrix X from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    dep_var : str
        Dependent variable name (will be sanitized).
    indep_vars : list of str
        Independent variable names (will be sanitized).
    add_constant : bool
        Whether to add a constant (intercept) column.

    Returns
    -------
    (y, X) : (pd.Series, pd.DataFrame)
    """
    import statsmodels.api as sm

    dep_col = _resolve_col(df, dep_var)
    indep_cols = [_resolve_col(df, v) for v in indep_vars]

    y = df[dep_col].astype(float)
    X = df[indep_cols].astype(float)

    if add_constant:
        X = sm.add_constant(X, has_constant="skip")

    return y, X


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_col(df: pd.DataFrame, name: str) -> str:
    """Return the column name as it appears in df (try sanitized first)."""
    sanitized = sanitize_key(name)
    if sanitized in df.columns:
        return sanitized
    if name in df.columns:
        return name
    raise KeyError(f"Column '{name}' (sanitized: '{sanitized}') not found. Available: {list(df.columns)}")
