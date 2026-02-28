# AGENTS.md

## Cursor Cloud specific instructions

### Environment

- **Python 3.9** is required. The VM has it installed via deadsnakes PPA at `/usr/bin/python3.9`.
- A virtualenv lives at `/workspace/.venv` (created with `python3.9 -m venv .venv`). Activate it before running anything: `source /workspace/.venv/bin/activate`.
- PyTorch is installed as **CPU-only** (`torch+cpu`). Any experiment config that defaults to `device: cuda` must be overridden with `device=cpu` (and `net.device=cpu`, `learner.device=cpu` if the config has those fields).
- `wandb` is installed but not authenticated. Set `use_wandb=false` in Hydra overrides unless you have credentials configured.

### Running tests

```bash
source /workspace/.venv/bin/activate
python -m pytest tests/ -v
```

All 18 tests pass on CPU.

### Running experiments

Experiments live under `experiments/<name>/` and use Hydra YAML configs. Example:

```bash
source /workspace/.venv/bin/activate
python experiments/basic_training/single_run.py epochs=2 use_wandb=false use_json=false
```

### Type checking

```bash
source /workspace/.venv/bin/activate
python -m pyright
```

The codebase has ~1000 pre-existing pyright errors at `basic` type-checking level; these are expected for a research codebase.

### Gotchas

- The `requirements.txt` uses `--index-url https://download.pytorch.org/whl/cpu` which prevents pip from finding non-PyTorch packages on PyPI. When installing, use `--extra-index-url https://pypi.org/simple/` or install torch/torchvision separately from the rest.
- Some experiment configs (e.g., the LLA task-shift config) default `data.data_path` to `/hdda` (a lab NFS mount). Override with `data.data_path=./data` to use a local directory.
- Some experiment configs default `device` to `cuda`. Always override to `cpu` when running without a GPU.
- MNIST/CIFAR-10 datasets auto-download via torchvision on first run.
- For RR-CBP2 numerical stability, set `SIGMA_FORCE_CPU_EIGH=1` if encountering eigendecomposition errors.
