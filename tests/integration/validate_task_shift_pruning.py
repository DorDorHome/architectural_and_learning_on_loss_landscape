"""Quick validation of task shift logging pruning.
Run: python validate_task_shift_pruning.py
It fabricates a minimal OmegaConf-like object if Hydra not invoked.
"""
import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from types import SimpleNamespace
from omegaconf import OmegaConf
from src.utils.task_shift_logging import build_logging_config_dict

# Mock structure resembling cfg.task_shift_param
cfg_dict = {
    'task_shift_mode': 'continuous_input_deformation',
    'prune_inactive_task_shift_params': True,
    'task_shift_param': {
        'continuous_input_deformation': {
            'drift_mode': 'sinusoidal',
            'sinusoidal': {'amplitude': 0.05, 'frequency': 0.2},
            'linear': {'max_drift': 0.1},
            'random_walk': {'drift_std_dev': 0.01}
        },
        'drifting_values': {
            'drift_std_dev': 0.02,
            'repulsion_strength': 0.5,
            'min_gap': 0.1,
            'value_bounds': {
                'lower_bound': -5.0,
                'upper_bound': 5.0
            }
        }
    }
}

cfg = OmegaConf.create(cfg_dict)

sanitized = build_logging_config_dict(cfg)
print("Keys in sanitized root:", list(sanitized.keys()))
print("Contains raw snapshot:", 'task_shift_param_raw_str' in sanitized)
print("Active task shift entry:", sanitized.get('active_task_shift'))
print("Original task_shift_param present:", 'task_shift_param' in sanitized)

# Expect: active_task_shift exists, original task_shift_param removed, raw snapshot string present.
