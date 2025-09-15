# Data Loading and Dataset Orchestration

This codebase centralizes dataset access and preprocessing so experiments remain small and composable.

Key modules:
- `src/data_loading/dataset_factory.py` — entrypoint for building datasets (torchvision and custom)
- `src/data_loading/transform_factory.py` — builds transforms per dataset+model pair
- `src/data_loading/shifting_dataset.py` — wrappers for task shifts (permutations, continuous deformation, drifting targets)

## Contracts

- Input: `DataConfig` (see `configs/configurations.py`) and a torchvision transform
- Output: a PyTorch Dataset for training, and optionally one for testing

```python
train_set, test_set = dataset_factory(cfg.data, transform=transform, with_testset=True)
```

### DataConfig fields
- `dataset` (str): e.g., `MNIST`, `CIFAR10`
- `use_torchvision` (bool): if true, load from `torchvision.datasets`
- `data_path` (str): root folder for dataset cache
- `num_classes` (int): used to validate class count and for wrappers

## Transform selection

Use `transform_factory(dataset_name, model_name)` to select a deterministic preprocessing pipeline that matches the model’s expected input size and normalization.

Examples implemented:
- `CIFAR10` + `ResNet18|resnet_custom|full_rank_resnet_custom`
- `CIFAR10` + `VGG16|vgg_custom|vgg_custom_norm`
- `MNIST` + `ConvNet*` variants, `ResNet18|resnet_custom`, and `deep_ffnn*`

If a dataset-model pair is unsupported, `transform_factory` raises a clear error so you can add a branch.

## Torchvision vs. custom datasets

- Torchvision path: `dataset_factory` finds the dataset class and constructs it with the given transform. It sanity-checks `num_classes` when available.
- Custom path: when `use_torchvision=False`, `dataset_factory` returns custom datasets or dataset factories. One example is `imagenet_for_plasticity`, which returns a factory for class-sliced ImageNet data.

## Task-shift wrappers

Wrap base datasets to create non-stationary tasks.

- Stateless per-task wrappers (`create_stateless_dataset_wrapper`):
  - `permuted_input`: pixel-permutation of inputs; new permutation per task
  - `permuted_output`: label permutation; new permutation per task
- Stateful wrappers (`create_stateful_dataset_wrapper`):
  - `continuous_input_deformation`: applies evolving affine transforms. Modes: `linear`, `random_walk`, `sinusoidal`
  - `drifting_values`: turns classification into regression with class-wise values that drift while maintaining order and bounds

Usage pattern inside an experiment loop:

```python
is_stateful = cfg.task_shift_mode in ["drifting_values", "continuous_input_deformation"]
dataset_wrapper = create_stateful_dataset_wrapper(cfg, train_set) if is_stateful else None

for task_idx in range(cfg.num_tasks):
    current_train_set = (
        dataset_wrapper if is_stateful
        else create_stateless_dataset_wrapper(cfg, train_set, task_idx) or train_set
    )
    # For stateful shifts, evolve between tasks
    if is_stateful and hasattr(dataset_wrapper, 'update_drift'):
        dataset_wrapper.update_drift()
```

Notes:
- For stateful wrappers, prefer `num_workers=0` initially to avoid multiprocessing/pickling pitfalls, then tune.
- Always seed workers when using multiple workers to keep per-epoch stochasticity controlled.
