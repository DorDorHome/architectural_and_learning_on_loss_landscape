from typing import Any, Dict, Optional
from omegaconf import OmegaConf


ACTIVE_KEY = 'active_task_shift'


def extract_active_task_shift(cfg: Any) -> Optional[Dict[str, Any]]:
    """Return a plain dict describing only the active task shift sub-config.

    Expected structure (Hydra/OmegaConf):
        cfg.task_shift_mode: str or None
        cfg.task_shift_param: node with possible children like drifting_values, continuous_input_deformation, etc.
    For continuous_input_deformation we also dive into the selected drift_mode to only surface its params.
    Returns None if no shift is configured.
    """
    task_shift_mode = getattr(cfg, 'task_shift_mode', None)
    task_shift_param = getattr(cfg, 'task_shift_param', None)
    if not task_shift_mode or task_shift_param is None:
        return None

    # OmegaConf nodes access via attribute; convert only what we need
    if not hasattr(task_shift_param, task_shift_mode) and task_shift_mode not in ['permuted_input', 'permuted_output']:
        # Stateless modes with on-the-fly randomization don't need stored params
        return { 'mode': task_shift_mode, 'params': {} }

    # Drifting values: simple leaf dict
    if task_shift_mode == 'drifting_values':
        params_node = getattr(task_shift_param, 'drifting_values', None)
        params_dict = OmegaConf.to_container(params_node, resolve=True) if params_node is not None else {}
        return { 'mode': task_shift_mode, 'params': params_dict }

    if task_shift_mode == 'continuous_input_deformation':
        cid_node = getattr(task_shift_param, 'continuous_input_deformation', None)
        if cid_node is None:
            return { 'mode': task_shift_mode, 'params': {} }
        drift_mode = getattr(cid_node, 'drift_mode', None)
        if drift_mode is None:
            return { 'mode': task_shift_mode, 'params': {} }
        drift_mode_node = getattr(cid_node, drift_mode, None)
        drift_mode_params = OmegaConf.to_container(drift_mode_node, resolve=True) if drift_mode_node is not None else {}
        # include high-level fields (e.g., drift_mode) explicitly
        top_level = {k: v for k, v in OmegaConf.to_container(cid_node, resolve=True).items() if k in ['drift_mode']}
        return { 'mode': task_shift_mode, 'params': { **top_level, **{drift_mode: drift_mode_params} } }

    # Stateless permutations: nothing stored; we still record the mode
    if task_shift_mode in ['permuted_input', 'permuted_output']:
        return { 'mode': task_shift_mode, 'params': {} }

    # Fallback
    return { 'mode': task_shift_mode, 'params': {} }


def build_logging_config_dict(cfg: Any) -> Dict[str, Any]:
    """Return a dict for external logging (e.g., W&B) honoring pruning flag.

    If cfg.prune_inactive_task_shift_params is True, we replace the original task_shift_param
    structure with only the active one under ACTIVE_KEY and drop inactive siblings.
    We still keep the original raw shift params hash (string) for reproducibility.
    """
    base = OmegaConf.to_container(cfg, resolve=True)
    # Store hash / string snapshot for reproducibility if task_shift_param exists
    task_shift_param = getattr(cfg, 'task_shift_param', None)
    if task_shift_param is not None:
        base['task_shift_param_raw_str'] = OmegaConf.to_yaml(task_shift_param)
    if getattr(cfg, 'prune_inactive_task_shift_params', False):
        active = extract_active_task_shift(cfg)
        base.pop('task_shift_param', None)
        if active is not None:
            base[ACTIVE_KEY] = active
    return base
