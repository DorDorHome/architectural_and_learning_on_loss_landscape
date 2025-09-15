# Architecture

The project uses factories and Hydra configs to keep experiments small and reusable.

## Core components

- Configs (`configs/configurations.py`): dataclasses for `ExperimentConfig`, `DataConfig`, `NetConfig`, `BackpropConfig`, etc.
- Models (`src/models/` + `model_factory.py`): implemented PyTorch modules selected by `cfg.net.type`
- Learners (`src/algos/supervised/` + `supervised_factory.py`): optimization/training strategies with a simple `learn()` contract
- Data (`src/data_loading/`): dataset creation and transform selection; wrappers for dataset shifts
- Utils (`src/utils/`): rank and diagnostics utilities invoked by experiments

## Training lifecycle

1. Parse Hydra config from `experiments/<name>/cfg/config.yaml`
2. Build transforms (`transform_factory`)
3. Build datasets (`dataset_factory`); optionally wrap for task shifts
4. Instantiate model (`model_factory`) and move to device
5. Instantiate learner (`create_learner`)
6. Train loop: forward -> loss -> backward/step; optional structural steps (GnT)
7. Metrics: ranks (`src.utils.zeroth_order_features`), weight stats, dead units, etc.
8. Logging: W&B or JSON (future)

## Contracts

- Learner interface:
  - `learn(input, target) -> (loss, output)`
  - optional `learn_from_partial_values(input, drifting_values, labels)` for regression drift
  - optional `previous_features: List[Tensor]` for rank metrics
- Model expectations:
  - Accept tensors from `transform_factory` (correct shape/normalization)
  - Optional helper for features (e.g., `predict()` returning features)

## Extension points

- Add model types to `model_factory`
- Add learners to `supervised_factory`
- Add datasets/custom modes to `dataset_factory`
- Add transforms per dataset+model pairing in `transform_factory`
- Extend rank metrics in `src/utils/zeroth_order_features.py`
