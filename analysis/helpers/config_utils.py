import os
import yaml
import copy
from typing import Any, Dict, List

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged

def flatten_target_config(target: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested target dict into dot-notation keys for W&B config filtering.

    Example: {"net": {"type": "X"}, "learner": {"type": "Y", "opt": "sgd"}}
    becomes  {"net.type": "X", "learner.type": "Y", "learner.opt": "sgd"}
    """
    flat: Dict[str, Any] = {}
    for k, v in target.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_target_config(v, key))
        else:
            flat[key] = v
    return flat

def resolve_targets(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge target_defaults into each target_runs entry."""
    defaults = cfg.get("target_defaults", {})
    raw_targets = cfg.get("target_runs", [])
    return [deep_merge(defaults, t) for t in raw_targets]

def target_diff_label(target: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """Return a short label showing only what *target* overrides from *defaults*.

    Example: ``sigma_ridge=0.05`` or ``net.type=ConvNet, sigma_ridge=0.1``.
    Falls back to the full flat config if no defaults are provided.
    """
    flat_target = flatten_target_config(target)
    flat_defaults = flatten_target_config(defaults) if defaults else {}
    diffs = {k: v for k, v in flat_target.items() if flat_defaults.get(k) != v}
    if not diffs:
        return "(defaults — no overrides)"
    return ", ".join(f"{k}={v}" for k, v in sorted(diffs.items()))

def get_nested_config(config: dict, dot_path: str, default=None):
    """Traverse nested dict with dot-separated key."""
    keys = dot_path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
        if current is None:
            return default
    return current

def get_target_label(target: Dict[str, Any], defaults: Dict[str, Any] = None) -> str:
    """Build a descriptive legend label from the full target config.

    Format: ``net_type / learner_type (key=val, ...)``
    where the parenthetical includes:
    1. Any learner config keys beyond ``type`` and ``opt``.
    2. Any other keys that differ from defaults (if defaults provided).
    """
    net_type = get_nested_config(target, "net.type", "unknown_net")
    learner_cfg = target.get("learner", {})
    learner_type = learner_cfg.get("type", "unknown_learner")

    # 1. Collect extra learner hyperparameters
    _STANDARD_KEYS = {"type", "opt"}
    extras = {k: v for k, v in learner_cfg.items()
              if k not in _STANDARD_KEYS and v is not None}

    # 2. If defaults provided, find other diffs (excluding net/learner which are handled)
    if defaults:
        flat_target = flatten_target_config(target)
        flat_defaults = flatten_target_config(defaults)
        
        for k, v in flat_target.items():
            # Skip keys we already handled or standard ones
            if k.startswith("net.") or k.startswith("learner."):
                continue
            
            # If value differs from default (or is new)
            if flat_defaults.get(k) != v:
                # Use the last part of the key for brevity if unambiguous, else full key
                short_key = k.split(".")[-1]
                extras[short_key] = v

    label = f"{net_type} / {learner_type}"
    if extras:
        # Sort for consistency
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        label += f"\n({params_str})"
    return label

def get_target_short_label(target: Dict[str, Any]) -> str:
    """Short label for directory names."""
    flat = flatten_target_config(target)
    parts = []
    for key in ["net.type", "learner.type", "learner.opt", "learner.step_size"]:
        if key in flat:
            parts.append(str(flat[key]))
    extra = {k: v for k, v in flat.items()
             if k not in ("net.type", "learner.type", "learner.opt", "learner.step_size")}
    for k, v in sorted(extra.items()):
        parts.append(f"{k.split('.')[-1]}={v}")
    return "_".join(parts)

def determine_output_dir(cfg: Dict[str, Any]) -> str:
    base = cfg["global_settings"]["base_output_dir"]
    subfolder = cfg["global_settings"].get("output_subfolder")
    if subfolder:
        return os.path.join(base, subfolder)
    return base
