import time
import wandb
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional
from analysis.helpers.config_utils import flatten_target_config

def build_target_api_filters(
    base_filters: Dict[str, Any],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge base YAML filters with a single target's config into a W&B query."""
    api_filters: Dict[str, Any] = {}

    state = base_filters.get("state")
    if state:
        states = state if isinstance(state, list) else [state]
        api_filters["state"] = {"$in": states}

    tags = base_filters.get("tags")
    if tags:
        api_filters["tags"] = {"$all": tags}

    # Merge base config_filters + target-specific config (AND semantics)
    config_filters = dict(base_filters.get("config_filters") or {})
    config_filters.update(flatten_target_config(target))

    for key, value in config_filters.items():
        wandb_key = f"config.{key}"
        if isinstance(value, list):
            api_filters[wandb_key] = {"$in": value}
        else:
            api_filters[wandb_key] = value

    return api_filters

def fetch_runs_for_target(
    entity: str,
    project: str,
    base_filters: Dict[str, Any],
    target: Dict[str, Any],
) -> List:
    """Fetch W&B runs matching a single target configuration."""
    api = wandb.Api(timeout=60)
    api_filters = build_target_api_filters(base_filters, target)

    path = f"{entity}/{project}"
    flat_target = flatten_target_config(target)
    print(f"  Fetching runs for target: {flat_target}")
    print(f"  API filters: {api_filters}")

    runs = []
    for attempt in range(3):
        try:
            runs_iter = api.runs(path, filters=api_filters)
            runs = list(runs_iter)
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  W&B API error (attempt {attempt + 1}/3): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to fetch runs after 3 attempts: {e}") from e

    # Client-side post-filtering
    name_regex = base_filters.get("name_regex")
    if name_regex:
        import re
        pattern = re.compile(name_regex)
        runs = [r for r in runs if pattern.search(r.name)]

    max_runs = base_filters.get("max_runs_per_target") or base_filters.get("max_runs")
    if max_runs and len(runs) > max_runs:
        runs = runs[:max_runs]

    print(f"  Found {len(runs)} matching runs")
    return runs

def discover_weight_norm_keys(run, source_pattern: str) -> List[str]:
    """Discover all per-layer weight norm keys from a run's summary."""
    # Check summary first
    all_keys = set(run.summary.keys())
    keys = [k for k in all_keys if k.endswith(source_pattern)]
    
    if not keys:
        # Fallback to history scan
        try:
            hist = run.history(keys=None, pandas=False, samples=10)
            if hist:
                keys = [k for k in hist[0].keys() if k.endswith(source_pattern)]
        except Exception:
            pass
            
    return sorted(list(set(keys)))

def extract_run_history(
    run,
    keys: List[str],
    page_size: int = 5000,
) -> pd.DataFrame:
    """Fetch history for specific keys from a W&B run."""
    # Always include step/epoch/task_idx if available
    # Removed 'epoch' to avoid potential issues if it doesn't exist and scan_history is strict
    default_keys = ["_step", "task_idx"]
    # Add keys requested by the caller
    query_keys = list(set(default_keys + keys))
    
    rows = []
    try:
        # Debug print to diagnose empty history
        # print(f"    DEBUG: Scanning history for {run.name} with keys: {query_keys}")
        for row in run.scan_history(keys=query_keys, page_size=page_size):
            rows.append({k: row.get(k) for k in query_keys})
    except Exception as e:
        warnings.warn(f"Error scanning history for run {run.name}: {e}")
        return pd.DataFrame()

    if not rows:
        # print(f"    DEBUG: No rows found for {run.name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Coerce numeric
    for col in df.columns:
        if col not in ("_step", "run_id", "run_name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
