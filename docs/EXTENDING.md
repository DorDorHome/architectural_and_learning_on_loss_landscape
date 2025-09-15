# Extending the Codebase

This guide explains how to add models, learners/optimizers, datasets, transforms, and metrics within the existing factory-based architecture.

## Add a new model
Where: `src/models/`

Steps:
1. Implement `YourModel` (standard `nn.Module`) in `src/models/YourModel.py`
2. Register in `src/models/model_factory.py` by adding a `model_type` branch returning an instance using `NetConfig.netparams`
3. Ensure output shape and `forward()` match downstream learners; optional: expose feature hooks or `predict()` that returns features if needed by rank metrics
4. If the model expects specific input size/normalization, add its branch in `transform_factory` under each supported dataset

Checklist:
- Unit test: construction + a single forward pass with a dummy batch
- Update docs if the model has special requirements

## Add a new learner/optimizer
Where: `src/algos/supervised/`

Steps:
1. Create `your_learner.py` implementing a class with `learn(input, target) -> (loss, output)` and, if needed, `learn_from_partial_values(...)`
2. If you maintain features for rank tracking, set `self.previous_features` to a list of feature tensors per layer (2D preferred: [batch, features])
3. Register in `src/algos/supervised/supervised_factory.py` so experiments can call `create_learner(cfg.learner, net, cfg.net)`
4. If you perform structural changes (e.g., GnT), respect invariants in `continuous_backprop_with_GnT.py` (optimizer state reset, maturity gating)

Checklist:
- Basic training loop test on a toy dataset (2-3 batches)
- Works with `cfg.learner.network_class` consistency checks

## Add a new dataset or a custom dataset mode
Where: `src/data_loading/`

Options:
- Torchvision dataset: ensure `DataConfig.dataset` matches a class in `torchvision.datasets`. Add special-casing if you need extra validation.
- Custom dataset: with `use_torchvision=False`, add a new branch in `dataset_factory()` that constructs and returns your dataset or returns a dataset-factory.

Checklist:
- Honors `DataConfig.data_path`
- If classification, keep `num_classes` consistent; if missing, derive or document
- Add transform branches if preprocessing differs by model

## Add or adjust transforms
Where: `src/data_loading/transform_factory.py`

Steps:
1. Add a `(dataset_name, model_name)` branch returning a `transforms.Compose`
2. Prefer standard stats for pretrained backbones; otherwise, consistent zero-centered scaling

Checklist:
- Minimal resize/channel handling to match model
- Clear error if unsupported pair is requested

## Add a new rank metric
Where: `src/utils/zeroth_order_features.py`

Steps:
1. Extend rank computation helpers and ensure `compute_all_rank_measures_list` returns your metric in each layer’s dict
2. Experiments that log metrics will record any extra keys you add

Checklist:
- Unit test on synthetic feature tensors
- O(N) memory with batching for large layers if needed

## Data/task-shift wrappers
Where: `src/data_loading/shifting_dataset.py`

Patterns available:
- Stateless: `permuted_input`, `permuted_output` via `create_stateless_dataset_wrapper`
- Stateful: `continuous_input_deformation`, `drifting_values` via `create_stateful_dataset_wrapper`

Guidelines:
- Keep wrappers as thin `Dataset` adapters around a base dataset
- Expose `update_task()`/`update_drift()` to evolve state between tasks
- For stateful wrappers, consider `num_workers=0` and explicit worker seeding
