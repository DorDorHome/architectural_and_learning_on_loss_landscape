# Codebase Overview

This project studies learning dynamics and loss landscapes with pluggable learners and models.

## Folder map
- `configs/` — dataclasses for Hydra configs (`ExperimentConfig`, `DataConfig`, `NetConfig`, etc.)
- `src/`
  - `algos/` — learning algorithms and factories.
    - `supervised/`: Contains core training algorithms (learners) like `basic_backprop.py`, `continuous_backprop_with_GnT.py`, and various implementations of Rank-Restoring Continuous Backprop (RR-CBP), such as `rr_cbp_conv.py` and `rr_cbp2_fc.py`. `supervised_factory.py` orchestrates their creation.
  - `data_loading/` — dataset orchestration.
    - `dataset_factory.py`, `transform_factory.py`, `shifting_dataset.py` for handling various datasets and transformations.
  - `models/` — neural network implementations and `model_factory.py`. This includes a wide range of architectures (ConvNets, VGG, ResNet, FFNNs), with `model_factory.py` serving as a central dispatcher. It also contains utilities like `_infer_conv_input_dims` for automatic input dimension inference.
  - `utils/` — various utilities including rank computation, weight statistics, and miscellaneous functions. Specifically, `zeroth_order_features.py` is used for calculating different rank metrics.
- `experiments/` — self-contained experiments, each typically with its own `cfg/config.yaml` and a `train.py` entry point.
- `analysis/` — scripts for W&B uploaders and plotting results.
- `docs/` — documentation files, including architectural overviews and guides for extending the codebase.
- `tests/` — unit and smoke tests to ensure code correctness.

## Typical flow
1. Experiment entrypoint with `@hydra.main(config_path="cfg", config_name="config")`
2. Build model with `model_factory(cfg.net)`. For rank analysis, models must implement a `.predict()` method that returns both the final output and a list of intermediate feature activations.
3. Build transforms with `transform_factory(cfg.data.dataset, cfg.net.type)`
4. Build dataset(s) with `dataset_factory(cfg.data, transform, with_testset=cfg.evaluation.use_testset)`
5. Optionally wrap dataset for task shifts (`shifting_dataset.py`)
6. Build learner with `create_learner(cfg.learner, net, cfg.net)`. The learner encapsulates the training logic (e.g., `Backprop`, `GnT`, or RR-CBP variants).
7. Train/evaluate, compute ranks (`src.utils.zeroth_order_features`) and log

## Start here
- Architecture: `docs/ARCHITECTURE.md`
- Data orchestration: `docs/DATA_LOADING.md`
- Experiments & templates: `docs/EXPERIMENTS.md`
- Extending (models/learners/datasets): `docs/EXTENDING.md`