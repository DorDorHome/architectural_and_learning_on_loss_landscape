import re
import pandas as pd
from typing import Any, Dict, List, Optional

_DISPLAY_NAMES = {
    "task_idx": "Task Index",
    "epoch_loss": "Loss",
    "epoch_accuracy": "Accuracy",
    "forward_loss": "Forward Loss",
    "weight_norm_mean": "Mean Weight Norm",
    "effective_rank_rank_drop_gini": "Rank Drop Gini (Effective)",
    "approximate_rank_rank_drop_gini": "Rank Drop Gini (Approximate)",
    "l1_distribution_rank_rank_drop_gini": "Rank Drop Gini (L1)",
    "numerical_rank_rank_drop_gini": "Rank Drop Gini (Numerical)",
}

def update_display_names(custom_names: Dict[str, str]) -> None:
    """Update the global display names dictionary."""
    _DISPLAY_NAMES.update(custom_names)

def display_name(metric: str) -> str:
    """Return a human-readable display name for a metric key."""
    if metric in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[metric]
    # Auto-format: replace underscores with spaces, title case
    return metric.replace("_", " ").title()

def build_filename(base_name: str, comparison_name: Optional[str]) -> str:
    """Build output filename, optionally prefixed with comparison_name."""
    if comparison_name:
        return f"{comparison_name}_{base_name}"
    return base_name

def resolve_color_column(df: pd.DataFrame, color_by: str) -> str:
    """Resolve a dot-notation color_by key to an actual DataFrame column."""
    sanitized = color_by.replace(".", "__")
    if sanitized in df.columns:
        return sanitized
    if color_by in df.columns:
        return color_by
    raise KeyError(f"Color column '{color_by}' not found. Available: {list(df.columns)}")

def slugify_target_label(label: str, max_len: int = 100) -> str:
    """Create a filesystem-safe slug from target label."""
    # Extract parenthetical part (distinguishing hyperparams) and put it first
    match = re.search(r"\(([^)]+)\)", label)
    if match:
        params = match.group(1).replace(" ", "_").replace(".", "_")
        params = re.sub(r"[^\w\-.]", "_", params)
        rest = label[: match.start()] + label[match.end() :]
    else:
        params = ""
        rest = label

    rest = rest.replace("\n", "_").replace(" ", "_").replace("[", "").replace("]", "")
    rest = re.sub(r"[^\w\-.]", "_", rest).replace(".", "_")
    s = f"{params}_{rest}" if params else rest
    s = re.sub(r"_+", "_", s).strip("_")  # collapse multiple underscores
    return (s[:max_len] + "_") if len(s) > max_len else (s or "target")
