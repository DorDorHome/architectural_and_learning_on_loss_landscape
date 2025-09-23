"""
Sweep orchestrator for rank_drop_metrics experiment.

This script reads a sweep config YAML (cfg/sweep_rank_drop_metrics.yaml) that declares
which hyperparameters to sweep and how to generate seeds. It then launches multiple
instances of train_data_shift_mode.py with Hydra overrides, relying on the base
cfg/rank_drop_metrics.yaml for defaults not overridden.

Usage:
  - Default sweep config path: cfg/sweep_rank_drop_metrics.yaml
  - Optional CLI args:
      --sweep <path>    Path to sweep YAML (defaults above)
      --dry-run         Only print planned runs (do not execute)

Notes:
  - Concurrency is controlled by max_concurrent in the sweep YAML.
  - Devices can be assigned round-robin by listing them in sweep YAML under run.devices.
  - Each run adds `seed=<...>` to overrides to ensure reproducibility.
  - For values unspecified in the sweep, the base config (rank_drop_metrics.yaml) applies.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import csv
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, IO, cast

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Will fail with a clear message if used without install


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SWEEP_PATH = THIS_DIR / "cfg" / "sweep_rank_drop_metrics.yaml"
TRAIN_SCRIPT = THIS_DIR / "train_data_shift_mode.py"


def _ensure_yaml_available():
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Please add 'pyyaml' to requirements and install it."
        )


def load_sweep_yaml(path: Path) -> Dict[str, Any]:
    _ensure_yaml_available()
    # Import locally to satisfy static analyzers when global may be None
    import yaml as _yaml  # type: ignore
    with open(path, "r") as f:
        data: Dict[str, Any] = _yaml.safe_load(f)
    return data


def to_hydra_token(value: Any) -> str:
    """Format a Python value as a Hydra CLI token.

    - bool -> true/false
    - str  -> as-is (caller should avoid spaces/special chars)
    - numbers -> str(value)
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def enumerate_param_grid(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not parameters:
        return [{}]
    keys = list(parameters.keys())
    values_product = list(itertools.product(*(parameters[k] for k in keys)))
    combos: List[Dict[str, Any]] = []
    for vals in values_product:
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def build_overrides(param_values: Dict[str, Any]) -> List[str]:
    overrides: List[str] = []
    for key, val in param_values.items():
        overrides.append(f"{key}={to_hydra_token(val)}")
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=str(DEFAULT_SWEEP_PATH), help="Path to sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--python", type=str, default=None, help="Path to Python executable to run training (overrides run.python_executable)")
    args = parser.parse_args()

    sweep_path = Path(args.sweep).resolve()
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    sweep_cfg = load_sweep_yaml(sweep_path)

    # Parse sweep settings
    parameters: Dict[str, List[Any]] = sweep_cfg.get("parameters", {}) or {}
    sampling: Dict[str, Any] = sweep_cfg.get("sampling", {}) or {}
    run_cfg: Dict[str, Any] = sweep_cfg.get("run", {}) or {}
    seeds_cfg: Dict[str, Any] = sweep_cfg.get("seeds", {}) or {}

    method = sampling.get("method", "grid")
    if method != "grid":
        raise NotImplementedError(f"Only grid sampling is implemented, got: {method}")

    # Build parameter combinations (cartesian product)
    combos = enumerate_param_grid(parameters)

    # Seeds list
    seeds: List[int]
    if "values" in seeds_cfg and seeds_cfg["values"] is not None:
        seeds = list(seeds_cfg["values"])  # explicit seeds
    else:
        count = int(seeds_cfg.get("count", 1))
        base = int(seeds_cfg.get("base", 12345))
        stride = int(seeds_cfg.get("stride", 1))
        # Deterministic sequence based on base and stride
        seeds = [base + i * stride for i in range(count)]

    # Concurrency and devices
    max_concurrent = int(run_cfg.get("max_concurrent", 1))
    devices = list(run_cfg.get("devices", []))
    set_cuda_visible = bool(run_cfg.get("set_cuda_visible_devices", False))
    # Python interpreter to use for launching training
    python_exe = args.python or run_cfg.get("python_executable") or sys.executable

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_name = sweep_cfg.get("name", f"rank_drop_sweep_{timestamp}")
    out_dir = THIS_DIR / "outputs" / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preflight: verify required packages are available in the chosen interpreter
    if not args.dry_run:
        try:
            preflight_cmd: List[str] = [python_exe, "-c", "import hydra, omegaconf; import torch; print('OK')"]
            subprocess.run(preflight_cmd, cwd=str(THIS_DIR), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Preflight failed: The selected Python interpreter is missing required packages.")
            print(f"Interpreter: {python_exe}")
            print("Required imports: hydra, omegaconf, torch")
            print("stderr:\n" + (e.stderr.decode(errors='ignore') if e.stderr else ""))
            print("Fix by either:\n- Installing requirements in that interpreter: pip install -r requirements.txt\n- Or pass a different interpreter via --python or run.python_executable in the sweep YAML")
            raise SystemExit(2)

    # Prepare all runs
    planned_runs: List[Dict[str, Any]] = []
    run_idx = 0
    repeats_per_combo = int(run_cfg.get("repeats_per_combo", 1))

    for combo in combos:
        for seed in seeds:
            for rep in range(repeats_per_combo):
                # Create a unique seed for this repeat to avoid duplicates
                unique_seed = int(seed) + rep
                params = dict(combo)
                params["seed"] = unique_seed

                # Assign device round-robin if provided
                assigned_device = None
                if devices:
                    assigned_device = devices[run_idx % len(devices)]
                    # If we're going to set CUDA_VISIBLE_DEVICES to a single physical index,
                    # the visible device inside the process should be cuda:0
                    if set_cuda_visible and isinstance(assigned_device, str) and assigned_device.startswith("cuda:"):
                        override_device = "cuda:0"
                    else:
                        override_device = assigned_device
                    if override_device is not None:
                        params["device"] = override_device
                    # net.device tracks ${device} in base config

                overrides = build_overrides(params)

                planned_runs.append({
                    "idx": run_idx,
                    "overrides": overrides,
                    "device": assigned_device,
                    "override_device": params.get("device"),
                    "seed": unique_seed,
                })
                run_idx += 1

    # Logs and CSV summary
    csv_path = out_dir / "runs.csv"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Print plan if dry-run
    if args.dry_run:
        print(f"[DRY RUN] Planned {len(planned_runs)} runs. Commands:")
        for pr in planned_runs:
            cmd = [python_exe, str(TRAIN_SCRIPT)] + pr["overrides"]
            print(" ", " ".join(cmd))
        # Write planned CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["idx", "seed", "device", "status", "returncode", "log_path"] + list(parameters.keys())
            writer.writerow(header)
            for pr in planned_runs:
                row_params = {}
                for k in parameters.keys():
                    # parse value from overrides
                    match = [ov for ov in pr["overrides"] if ov.startswith(f"{k}=")]
                    row_params[k] = match[0].split("=", 1)[1] if match else ""
                writer.writerow([
                    pr["idx"], pr["seed"], pr["device"] or "", "PLANNED", "", "",
                    *[row_params[k] for k in parameters.keys()],
                ])
        print(f"[DRY RUN] Plan written to: {csv_path}")
        return

    # Execute with bounded concurrency
    # Active and finished runs
    running: List[Tuple[subprocess.Popen[bytes], Dict[str, Any]]] = []  # list of tuples: (subprocess.Popen, run_dict)
    completed: List[Dict[str, Any]] = []

    def launch(pr: Dict[str, Any]) -> Tuple[subprocess.Popen[bytes], Dict[str, Any]]:
        cmd = [python_exe, str(TRAIN_SCRIPT)] + pr["overrides"]
        env = os.environ.copy()
        if set_cuda_visible and pr["device"] and isinstance(pr["device"], str) and pr["device"].startswith("cuda:"):
            # Restrict the visible device index if format is cuda:<idx>
            try:
                idx_str = pr["device"].split(":", 1)[1]
                int(idx_str)  # validate
                env["CUDA_VISIBLE_DEVICES"] = idx_str
            except Exception:
                pass
        log_file = log_dir / f"run_{pr['idx']:04d}.log"
        # Open log file for both stdout and stderr
        lf: IO[str] = open(log_file, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=str(THIS_DIR),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        pr["log_path"] = str(log_file)
        pr["_log_handle"] = lf
        return proc, pr

    # Main scheduling loop
    todo = list(planned_runs)
    while todo or running:
        # Fill available slots
        while todo and len(running) < max_concurrent:
            pr = todo.pop(0)
            proc, pr = launch(pr)
            running.append((proc, pr))
            print(f"[LAUNCHED] idx={pr['idx']} seed={pr['seed']} device={pr['device']} -> {pr['log_path']}")

        # Poll running processes
        new_running: List[Tuple[subprocess.Popen[bytes], Dict[str, Any]]] = []
        for proc, pr in running:
            code = proc.poll()
            if code is None:
                new_running.append((proc, pr))
                continue
            # Completed
            try:
                handle = pr.get("_log_handle")
                if handle is not None:
                    cast(IO[str], handle).close()
            except Exception:
                pass
            status = "COMPLETED" if code == 0 else "FAILED"
            pr["returncode"] = code
            pr["status"] = status
            completed.append(pr)
            print(f"[DONE] idx={pr['idx']} status={status} rc={code} log={pr['log_path']}")
        running = new_running
        if running:
            time.sleep(1.0)

    # Write CSV summary
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx", "seed", "device", "status", "returncode", "log_path"] + list(parameters.keys())
        writer.writerow(header)
        for pr in completed:
            row_params = {}
            for k in parameters.keys():
                match = [ov for ov in pr["overrides"] if ov.startswith(f"{k}=")]
                row_params[k] = match[0].split("=", 1)[1] if match else ""
            writer.writerow([
                pr.get("idx"), pr.get("seed"), pr.get("device") or "",
                pr.get("status", ""), pr.get("returncode", ""), pr.get("log_path", ""),
                *[row_params[k] for k in parameters.keys()],
            ])

    print(f"All runs completed. Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
