## Hyperparameters Sweep — Plan (design only)

This document lays out how we will build a robust hyperparameter sweep workflow under `experiments/hyperparameters_sweep/`, reusing the latest base config in `experiments/improving_plasticity_via_advanced_optimizer/cfg/config.yaml` and keeping the single-run training logic similar to `train_with_improved_optimizer.py`.

### Checklist of requirements

- Create `experiments/hyperparameters_sweep/cfg` and a config that references the latest base config in `experiments/improving_plasticity_via_advanced_optimizer/cfg/config.yaml` (no duplication of defaults).
- Provide a training entrypoint that manages a single run (same behavior contract as `train_with_improved_optimizer.py`).
- Provide an orchestrator that enumerates/schedules multiple runs (grid/random) with overrides (e.g., learning rate) and groups/logs them cleanly.
- Allow sweeping multiple hyperparameters at once, even though the current config expects single scalar values per key.
- Do not implement code yet. Only define structure, approach, and interfaces.

---

## Design summary

- We’ll separate concerns:
	- A Single-Run Trainer that executes exactly one configuration (mirroring the current training code).
	- A Sweep Orchestrator that expands a search space and launches many single-run processes with overrides.
- The orchestrator approach avoids invasive changes to the existing config and keeps the single-run code simple and reproducible.
- Referencing the “latest” base config is handled at runtime by loading/merging the base config into the single-run trainer, so we always inherit updates made under `improving_plasticity_via_advanced_optimizer`.

Why not Hydra-native multirun only? We can support it later, but an explicit orchestrator gives us:
- Cross-parameter constraints and filtering (e.g., skip invalid combos for a given optimizer or task mode).
- Custom grouping/naming in W&B and local summaries.
- Control over concurrency and GPU assignment.

---

## Proposed folder/file structure

```
experiments/hyperparameters_sweep/
	cfg/
		config.yaml          # Thin wrapper config; owns sweep-specific defaults (e.g., tags, grouping), not the search space itself
		sweep.yaml           # Declarative sweep space and policy (grid/random), optional constraints, repeats, seeds

	train_single.py       # Single-run trainer (reuses base config by merging; mirrors train_with_improved_optimizer behavior)
	orchestrate_sweep.py  # Orchestrator: expands sweep.yaml, spawns train_single with overrides, manages logging/grouping

	README.md             # This plan
```

Notes:
- `cfg/config.yaml` is intentionally minimal. It ensures the script has a local config home and can introduce sweep-scoped defaults (e.g., W&B group/tag conventions) without duplicating the base.
- `cfg/sweep.yaml` contains the sweep specification (parameters, values, sampling method, constraints) and orchestrator runtime knobs (repeats, seeds, max_concurrent, etc.).

---

## Config referencing strategy (staying synced with the latest base)

Goal: Never fork the base config. Always source from `experiments/improving_plasticity_via_advanced_optimizer/cfg/config.yaml` at runtime.

Approach in `train_single.py`:
1. Compute `BASE_CFG_PATH = PROJECT_ROOT/experiments/improving_plasticity_via_advanced_optimizer/cfg/config.yaml`.
## Hyperparameters Sweep — Minimal, single-run search

This experiment is purpose-built to find good optimizer hyperparameters quickly on a single run — no task-shifting, no rank tracking, no experiment-specific extras. It trains a small CNN on MNIST for a few epochs and reports final loss/accuracy.

Key goals:
- Keep runs fast and comparable
- Sweep only core hyperparameters (lr, weight decay, momentum, optimizer, batch size)
- Collect final metrics into a simple CSV and group/name runs in W&B

### Files

- `cfg/config.yaml`: Minimal single-run config (device, epochs, batch_size, learner.{opt, step_size, weight_decay, momentum}, data path, and W&B project).
- `cfg/sweep.yaml`: Sweep space + runtime policy (grid/random, repeats, seeds, concurrency, devices). Includes a rule to force `momentum=0.0` when `opt=adam`.
- `train_sweep_experiment.py`: Standalone PyTorch trainer (SimpleCNN on MNIST). Emits final lines: `METRIC epoch_loss=...` and `METRIC epoch_accuracy=...`.
- `train_single.py`: Thin wrapper that forwards CLI dotted overrides to the local trainer.
- `orchestrate_sweep.py`: Expands combinations, applies constraints, assigns GPUs round‑robin, captures METRIC lines, and appends them to a CSV.

### What’s intentionally NOT here

- No task_shift_mode, drifting datasets, or multi-task loops
- No rank/effective-rank/saturation or weight magnitude tracking
- No dependency on other experiment configs

### How it works

1) The orchestrator reads `cfg/sweep.yaml` and enumerates combinations (grid or random).
2) It launches one process per run (`train_single.py -- ...overrides`).
3) The trainer runs `epochs` on the MNIST training set and prints final `METRIC` lines.
4) The orchestrator parses those and writes `cfg/runs.csv` with one row per run.

### Configure

- Edit `cfg/config.yaml` for global defaults (epochs, batch_size, device). W&B project is `hyperparameters_sweep`.
- Edit `cfg/sweep.yaml` to change the search space and execution policy:
  - `parameters`: learner.step_size, learner.weight_decay, learner.momentum, learner.opt, batch_size
  - `constraints`: includes `opt=adam -> momentum=0.0`
  - `repeats`, `seeds`: replicate or control sampling
  - `max_concurrent`, `devices`: concurrency and GPU IDs
  - `logging.local.metrics`: metric columns to log (default: epoch_loss, epoch_accuracy)

### Run

1) Dry run (see planned commands only)
	- The orchestrator prints the command lines without executing.

2) Launch the sweep
	- One process per combo, up to `max_concurrent` in parallel.
	- Output log: `experiments/hyperparameters_sweep/outputs/sweep_YYYYMMDD_HHMMSS.log`
	- CSV: `experiments/hyperparameters_sweep/cfg/runs.csv`

3) Monitor
	- GPU: `nvidia-smi`
	- Processes: look for `orchestrate_sweep.py` and `train_sweep_experiment.py`
	- W&B: project `hyperparameters_sweep` (group/name from sweep config)

### Notes

- If you observe low GPU utilization, you can increase concurrency (`max_concurrent`) or increase `batch_size` in the sweep.
- The trainer uses only the training portion of MNIST for speed and consistency across runs.
- METRIC parsing is focused on final epoch loss/accuracy; you can add more metrics by extending the trainer’s printed lines and listing them in `logging.local.metrics`.

### Reproducibility

- Each run seeds Python/NumPy/Torch; if `seeds` is omitted, seeds are auto-generated and stored in the CSV.
- This experiment is self-contained and won’t drift with other experiments’ changes.
### Requirements coverage (updated)
