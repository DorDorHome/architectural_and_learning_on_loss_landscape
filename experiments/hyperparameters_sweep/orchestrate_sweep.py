"""
Orchestrator for hyperparameter sweeps.

- Reads cfg/sweep.yaml to expand a set of runs (grid or random sampling)
- Applies constraints to adjust or skip invalid combinations
- Assigns devices (CUDA) round-robin and controls concurrency
- Launches train_single.py with dotted overrides for each run
- Optionally groups and names runs in W&B by injecting overrides
- Writes a CSV summary (status, overrides, key identifiers)

This script avoids importing the base training logic; it only manages processes
and config overrides.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import itertools
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import OmegaConf

SWEEP_DIR = pathlib.Path(__file__).resolve().parent
CFG_DIR = SWEEP_DIR / "cfg"
TRAIN_SINGLE = SWEEP_DIR / "train_single.py"


@dataclass
class SweepConfig:
    method: str
    samples: Optional[int]
    parameters: Mapping[str, Sequence[Any]]
    constraints: Sequence[Mapping[str, Any]]
    repeats: int
    seeds: Optional[Sequence[int]]
    max_concurrent: int
    devices: Sequence[int]
    retry: int
    continue_on_error: bool
    dry_run: bool
    logging: Mapping[str, Any]


def load_sweep_cfg() -> SweepConfig:
    path = CFG_DIR / "sweep.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=False)  # keep interpolations as text
    return SweepConfig(
        method=raw.get("method", "grid"),
        samples=raw.get("samples"),
        parameters=raw.get("parameters", {}),
        constraints=raw.get("constraints", []) or [],
        repeats=int(raw.get("repeats", 1)),
        seeds=raw.get("seeds"),
        max_concurrent=int(raw.get("max_concurrent", 1)),
        devices=list(raw.get("devices", [0])),
        retry=int(raw.get("retry", 0)),
        continue_on_error=bool(raw.get("continue_on_error", False)),
        dry_run=bool(raw.get("dry_run", False)),
        logging=raw.get("logging", {}),
    )


def product_space(parameters: Mapping[str, Sequence[Any]]) -> List[Mapping[str, Any]]:
    keys = list(parameters.keys())
    values_lists = [list(parameters[k]) for k in keys]
    combos = []
    for values in itertools.product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def random_space(parameters: Mapping[str, Sequence[Any]], samples: int) -> List[Mapping[str, Any]]:
    keys = list(parameters.keys())
    drawn = []
    for _ in range(samples):
        combo = {k: random.choice(list(parameters[k])) for k in keys}
        drawn.append(combo)
    return drawn


def apply_constraints(combo: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]) -> Tuple[Optional[Mapping[str, Any]], bool]:
    """Return (possibly-modified combo, skipped?)."""
    c = dict(combo)
    for rule in constraints:
        when = rule.get("when", {})
        skip = rule.get("skip", False)
        setv = rule.get("set", {})
        # Check match
        matched = all(c.get(k) == v for k, v in when.items())
        if not matched:
            continue
        if skip:
            return None, True
        # apply set
        for k, v in setv.items():
            c[k] = v
    return c, False


def format_name(template: str, values: Mapping[str, Any]) -> str:
    # simple ${key} replacement
    name = template
    for k, v in values.items():
        name = name.replace(f"${{{k}}}", str(v))
    return name


def build_overrides(combo: Mapping[str, Any], seed: Optional[int], log_cfg: Mapping[str, Any]) -> List[str]:
    """Build CLI overrides that are known to exist in the base config.

    We avoid passing unsupported keys (like wandb.group/name) to keep
    compatibility with structured configs. W&B metadata is injected via env.
    """
    overrides: List[str] = []
    if seed is not None:
        overrides.append(f"seed={seed}")
    # wandb.project is present in base config; allow override if desired
    wandb_cfg_obj: Mapping[str, Any] = {}
    if isinstance(log_cfg, dict):
        maybe_wandb = log_cfg.get("wandb")
        if isinstance(maybe_wandb, dict):
            wandb_cfg_obj = maybe_wandb
    project = wandb_cfg_obj.get("project") if wandb_cfg_obj else None
    if project:
        proj_val = project
        if isinstance(proj_val, str) and any(ch.isspace() for ch in proj_val):
            proj_val = f'"{proj_val}"'
        overrides.append(f"wandb.project={proj_val}")

    # add the hyperparameters themselves
    for k, v in combo.items():
        if isinstance(v, str) and any(ch.isspace() for ch in v):
            overrides.append(f"{k}='{v}'")
        else:
            overrides.append(f"{k}={v}")

    return overrides


def assign_device(devices: Sequence[int], idx: int) -> str:
    if not devices:
        return "cuda:0"
    dev = devices[idx % len(devices)]
    return f"cuda:{dev}"


@dataclass
class RunSpec:
    idx: int
    combo: Mapping[str, Any]
    seed: Optional[int]
    device: str
    retries_left: int


def run_once(spec: RunSpec, log_cfg: Mapping[str, Any], metrics: Sequence[str]) -> Tuple[int, str, Dict[str, Any]]:
    overrides = build_overrides(spec.combo, spec.seed, log_cfg)
    # ensure device override wins
    overrides.append(f"device={spec.device}")

    cmd = [sys.executable, str(TRAIN_SINGLE), "--"] + overrides

    env = os.environ.copy()
    # Set CUDA visible to the single device to be safe; trainer uses device= override anyway
    if spec.device.startswith("cuda:"):
        _, dev_id = spec.device.split(":")
        env["CUDA_VISIBLE_DEVICES"] = dev_id

    # Inject W&B metadata via env to avoid changing structured config
    wandb_cfg_obj2: Mapping[str, Any] = {}
    if isinstance(log_cfg, dict):
        maybe_wandb2 = log_cfg.get("wandb")
        if isinstance(maybe_wandb2, dict):
            wandb_cfg_obj2 = maybe_wandb2
    group_by_raw = wandb_cfg_obj2.get("group_by", []) if wandb_cfg_obj2 else []
    group_by: List[str] = [str(x) for x in group_by_raw] if isinstance(group_by_raw, (list, tuple)) else []
    if group_by:
        env["WANDB_RUN_GROUP"] = "+".join(str(spec.combo.get(k, "")) for k in group_by)
    name_template_raw = wandb_cfg_obj2.get("name_template") if wandb_cfg_obj2 else None
    name_template: Optional[str] = str(name_template_raw) if isinstance(name_template_raw, str) else None
    if name_template:
        env["WANDB_NAME"] = format_name(name_template, {**spec.combo, "seed": spec.seed})
    tags_raw = wandb_cfg_obj2.get("tags", []) if wandb_cfg_obj2 else []
    tags: List[str] = [str(t) for t in tags_raw] if isinstance(tags_raw, (list, tuple)) else []
    if tags:
        env["WANDB_TAGS"] = ",".join(tags)

    import subprocess

    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    stdout = completed.stdout or ""
    code = completed.returncode
    status = "ok" if code == 0 else "fail"
    parsed: Dict[str, Any] = {}
    if stdout:
        # Parse METRIC lines (take last occurrence per key)
        for line in stdout.strip().splitlines():
            if line.startswith("METRIC "):
                try:
                    kv = line[len("METRIC "):].strip()
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k in metrics:
                            try:
                                if v.lower() in {"nan", "inf", "-inf"}:
                                    parsed[k] = v
                                else:
                                    parsed[k] = float(v)
                            except Exception:
                                parsed[k] = v
                except Exception:
                    pass
    # Echo stdout/stderr to parent for transparency
    if stdout:
        sys.stdout.write(stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return code, status, parsed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate hyperparameter sweep")
    parser.add_argument("--dry_run", action="store_true", help="Override sweep.yaml dry_run")
    args = parser.parse_args(argv)

    cfg = load_sweep_cfg()
    if args.dry_run:
        cfg.dry_run = True

    # Build combinations
    if cfg.method == "grid":
        combos = product_space(cfg.parameters)
    elif cfg.method == "random":
        if not cfg.samples:
            print("samples is required for random method", file=sys.stderr)
            return 2
        combos = random_space(cfg.parameters, cfg.samples)
    else:
        print(f"Unknown method: {cfg.method}", file=sys.stderr)
        return 2

    # Apply constraints and replicate
    expanded: List[RunSpec] = []
    seeds = list(cfg.seeds) if cfg.seeds else []
    seed_iter = iter(seeds)
    idx = 0
    seen_keys: set[Tuple[Tuple[str, Any], ...]] = set()
    for combo in combos:
        adjusted, skipped = apply_constraints(combo, cfg.constraints)
        if skipped or adjusted is None:
            continue
        # Deduplicate by the adjusted hyperparameter tuple to avoid redundant runs
        key_tuple: Tuple[Tuple[str, Any], ...] = tuple(sorted(adjusted.items()))
        if key_tuple in seen_keys:
            continue
        seen_keys.add(key_tuple)
        for _ in range(cfg.repeats):
            seed = next(seed_iter, None)
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            device = assign_device(cfg.devices, idx)
            expanded.append(RunSpec(idx=idx, combo=adjusted, seed=seed, device=device, retries_left=cfg.retry))
            idx += 1

    # Prepare CSV
    summary_csv = (CFG_DIR / (cfg.logging.get("local", {}).get("summary_csv", "runs.csv"))).resolve()
    metrics: List[str] = []
    if isinstance(cfg.logging, dict):
        local_log_raw = cfg.logging.get("local", {})
        local_log: Dict[str, Any] = dict(local_log_raw) if isinstance(local_log_raw, dict) else {}
        maybe_metrics = local_log.get("metrics", [])
        if isinstance(maybe_metrics, (list, tuple)):
            metrics = [str(m) for m in maybe_metrics]
    if cfg.dry_run:
        print("Dry run. Planned commands:")
        for spec in expanded:
            overrides = build_overrides(spec.combo, spec.seed, cfg.logging)
            overrides.append(f"device={spec.device}")
            cmd = [sys.executable, str(TRAIN_SINGLE), "--"] + overrides
            print(" ", " ".join(cmd))
        if not any("wandb.project" in o for spec in expanded for o in build_overrides(spec.combo, spec.seed, cfg.logging)):
            print("[warn] wandb.project override not present in commands; ensure cfg/config.yaml has wandb.project set.")
        return 0

    # Concurrent execution with a simple thread pool (I/O bound)
    os.makedirs(summary_csv.parent, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        key_set = sorted({k for spec in expanded for k in spec.combo.keys()})
        header: List[str] = [
            "run_idx",
            "status",
            "exit_code",
            "seed",
            "device",
        ] + key_set + metrics
        writer.writerow(header)
        # Flush header so monitoring tools can see it before runs finish
        f.flush()

        def submit_run(spec: RunSpec) -> Tuple[RunSpec, int, str, Dict[str, Any]]:
            while True:
                code, status, parsed_metrics = run_once(spec, cfg.logging, metrics)
                if code == 0 or spec.retries_left <= 0:
                    return spec, code, status, parsed_metrics
                spec.retries_left -= 1
                time.sleep(1.0)

        with futures.ThreadPoolExecutor(max_workers=cfg.max_concurrent) as pool:
            futs = [pool.submit(submit_run, spec) for spec in expanded]
            for fut in futures.as_completed(futs):
                spec, code, status, parsed_metrics = fut.result()
                row_vals_base: List[Any] = [
                    spec.idx,
                    status,
                    code,
                    spec.seed,
                    spec.device,
                ]
                param_vals: List[Any] = [spec.combo.get(k) for k in key_set]
                metric_vals: List[Any] = ["" if m not in parsed_metrics else parsed_metrics[m] for m in metrics]
                row_vals = row_vals_base + param_vals + metric_vals
                writer.writerow(["" if v is None else str(v) for v in row_vals])
                if code != 0 and not cfg.continue_on_error:
                    print(f"Run {spec.idx} failed with exit code {code}. Aborting remaining runs.")
                    return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
