# Experiments: structure and templates

This document describes the standard structure for typical supervised learning experiments.
Each experiment is a self-contained folder under `experiments/<experiment_name>/`.

Specialized experiments (e.g., for Hessian analysis or Reinforcement Learning) may have a different structure and entry points. For those, please refer to the `README.md` file inside the specific experiment's folder.

A standard experiment contains:
- `train.py` (or similar entrypoint) annotated with `@hydra.main(config_path="cfg", config_name="config")`
- `cfg/config.yaml` providing an `ExperimentConfig`-shaped config (with nested `data`, `net`, `learner`, `evaluation`, etc.)

## Required wiring in train.py

Minimal contract:
- Build transform: `transform = transform_factory(cfg.data.dataset, cfg.net.type)`
- Build dataset(s): `train_set, test_set = dataset_factory(cfg.data, transform, with_testset=cfg.evaluation.use_testset)`
- Optionally wrap: `create_stateful_dataset_wrapper`/`create_stateless_dataset_wrapper`
- Build model: `net = model_factory(cfg.net)` and move to `cfg.net.device`
- Build learner: `learner = create_learner(cfg.learner, net, cfg.net)` (Note: some older scripts like `basic_training/single_run.py` may use local logic instead of the factory).
- Dataloaders should use `cfg.batch_size` and `cfg.num_workers` (support "auto")
- Respect `cfg.runs` and `cfg.run_id` or document single-run behavior

## Config conventions (Hydra)

Place a `cfg/config.yaml` that aligns with `configs/configurations.py` dataclasses, plus any experiment-specific fields. Examples exist in:
- `experiments/improving_plasticity_via_advanced_optimizer/cfg/config.yaml`
- `experiments/basic_training/cfg/basic_config.yaml`

Key fields (non-exhaustive):
- `data`: `dataset`, `use_torchvision`, `data_path`, `num_classes`
- `net`: `type`, `network_class` (e.g., `conv`|`fc`), `device`, `netparams`
- `learner`: `type`, optimizer/loss params, and `network_class`
- `evaluation`: `use_testset`, `eval_freq_epoch`, `eval_metrics`
- Optional task-shift: `task_shift_mode`, `task_shift_param`

## Template: new experiment

Create `experiments/<name>/` with two files:

1) `experiments/<name>/cfg/config.yaml`

```yaml
runs: 1
run_id: 0
seed: null
device: 'cuda:0'
epochs: 10
batch_size: 128
num_tasks: 1
num_workers: 2

use_wandb: false
use_json: false
wandb:
  project: '<project_name>'
  entity: ''

task_shift_mode: null  # or 'permuted_input' | 'permuted_output' | 'continuous_input_deformation' | 'drifting_values'
task_shift_param: {}

data:
  dataset: 'MNIST'
  use_torchvision: true
  data_path: '/hdda/datasets'
  num_classes: 10

net:
  type: 'ConvNet'
  network_class: 'conv'
  device: ${device}
  netparams:
    pretrained: false
    num_classes: ${data.num_classes}
    initialization: 'kaiming'
    input_height: null
    input_width: null
    activation: 'leaky_relu'

learner:
  type: 'backprop'
  network_class: ${net.network_class}
  device: ${device}
  opt: 'sgd'
  loss: 'cross_entropy'
  step_size: 0.01
  momentum: 0.9

evaluation:
  use_testset: true
  eval_freq_epoch: 1
  eval_metrics: ['accuracy', 'loss']
```

2) `experiments/<name>/train.py`

See `experiments/improving_plasticity_via_advanced_optimizer/train_with_improved_optimizer.py` for a fully wired example including task-shift wrappers, rank tracking, and W&B logging. For a minimal baseline, keep the same structure and drop optional features.

## Adding parameters safely

- Extend the YAML under `cfg/` and read via Hydra. If it maps to an existing dataclass field, it’s validated automatically.
- For new feature flags, prefer adding nested sections (e.g., `rank_tracking.*`) to keep the top level clean.
- Document any new fields in the experiment folder’s README or in this docs folder.
