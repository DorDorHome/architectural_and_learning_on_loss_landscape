# Codebase Overview

This project studies learning dynamics and loss landscapes with pluggable learners and models.

## Folder map
- `configs/` — dataclasses for Hydra configs (`ExperimentConfig`, `DataConfig`, `NetConfig`, etc.)
- `src/`
  - `algos/` — learning algorithms and factories
    - `supervised/`: `basic_backprop.py`, `continuous_backprop_with_GnT.py`, `supervised_factory.py`
  - `data_loading/` — dataset orchestration
    - `dataset_factory.py`, `transform_factory.py`, `shifting_dataset.py`
  - `models/` — implementations and `model_factory.py`
  - `utils/` — ranks, weight stats, misc
- `experiments/` — self-contained experiments with `cfg/config.yaml` and a `train.py`
- `analysis/` — W&B uploaders and plots
- `docs/` — how this all fits together
- `tests/` — unit and smoke tests

## Typical flow
1. Experiment entrypoint with `@hydra.main(config_path="cfg", config_name="config")`
2. Build transforms with `transform_factory(cfg.data.dataset, cfg.net.type)`
3. Build dataset(s) with `dataset_factory(cfg.data, transform, with_testset=cfg.evaluation.use_testset)`
4. Optionally wrap dataset for task shifts (`shifting_dataset.py`)
5. Build model with `model_factory(cfg.net)`
6. Build learner with `create_learner(cfg.learner, net, cfg.net)`
7. Train/evaluate, compute ranks (`src.utils.zeroth_order_features`) and log

## Start here
- Architecture: `docs/ARCHITECTURE.md`
- Data orchestration: `docs/DATA_LOADING.md`
- Experiments & templates: `docs/EXPERIMENTS.md`
- Extending (models/learners/datasets): `docs/EXTENDING.md`