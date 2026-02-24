"""
W&B Data Loader — Fetch, filter, and extract data from existing W&B runs.

Supports:
- Server-side filtering via W&B MongoDB-style queries (state, tags, config values)
- Client-side post-filtering (name regex, date ranges, summary metric ranges)
- Summary metric extraction
- Full history (time-series) extraction with configurable sampling
- Panel DataFrame construction (run_id × logged_step)
"""

import re
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_runs(
    entity: str,
    project: str,
    filters: Dict[str, Any],
) -> List:
    """
    Fetch runs from a W&B project, applying filters.

    Two-stage filtering:
      1. Server-side (fast): state, tags, config values via W&B API
      2. Client-side (flexible): name_regex, date range, summary ranges

    Parameters
    ----------
    entity : str
        W&B entity (user or team).
    project : str
        W&B project name.
    filters : dict
        Filter specification from the YAML config ``filters`` section.

    Returns
    -------
    list of wandb.apis.public.Run
        Filtered list of runs.
    """
    api = wandb.Api(timeout=60)

    # --- Stage 1: Server-side filters ------------------------------------
    api_filters = _build_api_filters(filters)
    path = f"{entity}/{project}"

    print(f"  Fetching runs from {path} ...")
    if api_filters:
        print(f"  Server-side filters: {api_filters}")

    runs_iter = _fetch_with_retry(api, path, api_filters, max_retries=3)
    runs = list(runs_iter)
    print(f"  Retrieved {len(runs)} runs from W&B API")

    # --- Stage 2: Client-side post-filtering -----------------------------
    runs = _apply_post_filters(runs, filters)

    # --- max_runs --------------------------------------------------------
    max_runs = filters.get("max_runs")
    if max_runs is not None and len(runs) > max_runs:
        print(f"  Limiting to max_runs={max_runs} (from {len(runs)})")
        runs = runs[:max_runs]

    print(f"  Final run count after filtering: {len(runs)}")
    return runs


def extract_run_data(
    run,
    config_keys: List[str],
    summary_metrics: List[str],
) -> Dict[str, Any]:
    """
    Extract specified config values and summary metrics from a single run.

    Parameters
    ----------
    run : wandb.apis.public.Run
    config_keys : list of str
        Dot-notation paths into ``run.config`` (e.g. ``"learner.step_size"``).
    summary_metrics : list of str
        Keys in ``run.summary`` to extract.

    Returns
    -------
    dict
        Flat dictionary with ``run_id``, ``run_name``, config values, and
        summary metrics.
    """
    row: Dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
    }

    # Config values (dot-notation traversal)
    for key in config_keys:
        row[key] = _get_nested(run.config, key)

    # Summary metrics
    for metric in summary_metrics:
        row[metric] = run.summary.get(metric)

    return row


def extract_history(
    run,
    history_metrics: List[str],
    sampling_config: Optional[Dict[str, Any]] = None,
    page_size: int = 10_000,
) -> pd.DataFrame:
    """
    Extract time-series history for specified metrics from a single run.

    Parameters
    ----------
    run : wandb.apis.public.Run
    history_metrics : list of str
        Metric keys to fetch from run history.
    sampling_config : dict, optional
        Sampling specification (method, n, steps, group_by, sort_by).
        If None, all rows are returned.
    page_size : int
        Number of rows per API page (larger = fewer round-trips).

    Returns
    -------
    pd.DataFrame
        History rows with the requested columns plus ``_step``.
    """
    # Always request _step as a reference column
    keys_to_fetch = list(set(history_metrics) | {"_step"})

    rows = []
    try:
        for row in run.scan_history(keys=keys_to_fetch, page_size=page_size):
            rows.append({k: row.get(k) for k in keys_to_fetch})
    except Exception as e:
        warnings.warn(f"Error scanning history for run {run.name}: {e}")
        return pd.DataFrame(columns=keys_to_fetch)

    if not rows:
        return pd.DataFrame(columns=keys_to_fetch)

    df = pd.DataFrame(rows)
    return df


def build_dataframe(
    runs: List,
    variables_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a single DataFrame from multiple runs.

    If ``history_metrics`` is non-empty, returns a **panel** DataFrame with
    one row per (run, logged_step).  Config keys are broadcast to every row
    of a run.

    If ``history_metrics`` is empty, returns a **cross-sectional** DataFrame
    with one row per run using summary metrics.

    Parameters
    ----------
    runs : list of wandb.apis.public.Run
    variables_config : dict
        The ``variables`` section of the YAML config.

    Returns
    -------
    pd.DataFrame
    """
    config_keys = variables_config.get("config_keys", [])
    summary_metrics = variables_config.get("summary_metrics", [])
    history_metrics = variables_config.get("history_metrics", [])
    sampling_config = variables_config.get("history_sampling", {})

    use_history = bool(history_metrics)

    all_rows = []
    skipped = 0

    for i, run in enumerate(runs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Extracting data from run {i + 1}/{len(runs)}: {run.name}")

        # Always extract config
        base_row = extract_run_data(run, config_keys, summary_metrics if not use_history else [])

        if use_history:
            hist_df = extract_history(run, history_metrics, sampling_config)
            if hist_df.empty:
                skipped += 1
                continue

            # Broadcast config columns onto every history row
            for col, val in base_row.items():
                hist_df[col] = val

            # Also include summary metrics if requested alongside history
            if summary_metrics:
                for metric in summary_metrics:
                    hist_df[metric] = run.summary.get(metric)

            all_rows.append(hist_df)
        else:
            all_rows.append(base_row)

    if skipped > 0:
        print(f"  Warning: {skipped} runs skipped (no history data)")

    if not all_rows:
        warnings.warn("No data extracted from any run.")
        return pd.DataFrame()

    if use_history:
        df = pd.concat(all_rows, ignore_index=True)
    else:
        df = pd.DataFrame(all_rows)

    print(f"  Built DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_api_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML filter spec to W&B MongoDB-style query dict."""
    api_filters: Dict[str, Any] = {}

    # State
    state = filters.get("state")
    if state:
        states = state if isinstance(state, list) else [state]
        api_filters["state"] = {"$in": states}

    # Tags (ALL must match)
    tags = filters.get("tags")
    if tags:
        api_filters["tags"] = {"$all": tags}

    # Config value filters
    config_filters = filters.get("config_filters")
    if config_filters:
        for key, value in config_filters.items():
            wandb_key = f"config.{key}"
            if isinstance(value, list):
                api_filters[wandb_key] = {"$in": value}
            else:
                api_filters[wandb_key] = value

    # Date filters
    created_after = filters.get("created_after")
    created_before = filters.get("created_before")
    if created_after or created_before:
        date_filter: Dict[str, str] = {}
        if created_after:
            date_filter["$gte"] = str(created_after)
        if created_before:
            date_filter["$lte"] = str(created_before)
        api_filters["created_at"] = date_filter

    return api_filters


def _fetch_with_retry(api, path: str, api_filters: dict, max_retries: int = 3):
    """Fetch runs with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return api.runs(path, filters=api_filters if api_filters else {})
        except Exception as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"  W&B API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to fetch runs from {path} after {max_retries} attempts: {e}"
                ) from e


def _apply_post_filters(runs: list, filters: Dict[str, Any]) -> list:
    """Apply client-side filters that can't be done via the W&B API."""
    original_count = len(runs)

    # Name regex
    name_regex = filters.get("name_regex")
    if name_regex:
        pattern = re.compile(name_regex)
        runs = [r for r in runs if pattern.search(r.name)]
        print(f"  name_regex '{name_regex}': {original_count} -> {len(runs)} runs")
        original_count = len(runs)

    # Date filters (client-side fallback for more precise control)
    created_after = filters.get("created_after")
    created_before = filters.get("created_before")
    if created_after or created_before:
        def _parse_date(d):
            if isinstance(d, str):
                return datetime.fromisoformat(d)
            return d

        filtered = []
        for r in runs:
            try:
                created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
                created_naive = created.replace(tzinfo=None)
                if created_after and created_naive < _parse_date(created_after):
                    continue
                if created_before and created_naive > _parse_date(created_before):
                    continue
                filtered.append(r)
            except Exception:
                filtered.append(r)  # keep if we can't parse date
        runs = filtered
        if len(runs) != original_count:
            print(f"  Date filter: {original_count} -> {len(runs)} runs")
            original_count = len(runs)

    # Summary metric range filters
    summary_filters = filters.get("summary_filters", {})
    if summary_filters:
        filtered = []
        for r in runs:
            keep = True
            for metric, bounds in summary_filters.items():
                val = r.summary.get(metric)
                if val is None:
                    keep = False
                    break
                if isinstance(bounds, dict):
                    if "min" in bounds and val < bounds["min"]:
                        keep = False
                        break
                    if "max" in bounds and val > bounds["max"]:
                        keep = False
                        break
            if keep:
                filtered.append(r)
        runs = filtered
        if len(runs) != original_count:
            print(f"  Summary filters: {original_count} -> {len(runs)} runs")

    return runs


def _get_nested(d: dict, dot_path: str, default=None):
    """
    Traverse a nested dict using a dot-separated path.

    Example: _get_nested({"a": {"b": 1}}, "a.b") -> 1
    """
    keys = dot_path.split(".")
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
        if current is None:
            return default
    return current
