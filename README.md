## Loss Landscape Analysis (LLA) suite — quick guide for contributors

This repo includes a focused Loss Landscape Analysis suite under `experiments/Hessian_and_landscape_plot_with_plasticity_loss/` that you can run as a demo, as a pipeline module, or integrated into training with task shifts.

### Components and purpose

- `lla_pipeline.py`
	- Importable, single-file module that exposes functions to compute and save:
		- Hessian-aligned planes, random planes, and trajectory-PCA planes (2D heatmap + optional 3D surface)
		- Hessian spectrum (top-k, Hutchinson trace estimate, ESD via SLQ)
		- SAM-style “robust” surface on the same plane
		- Mode connectivity (Bézier midpoint) metrics and curve plot
	- All outputs saved locally under `results/`; includes timings in JSON summaries.

- `LLA_demo.ipynb`
	- Demonstrates all features above end-to-end on small datasets/models available in this repo (see `docs/DATA_LOADING.md`).
	- Includes short setup notes and a quick fine-tune to avoid degenerate minima.

- `loss_landscape_change_during_task_shift.py`
	- Trains across tasks in a chosen task-shift mode and computes LLA metrics only at the end of each task.
	- Numeric-only logging to Weights & Biases (no images or large artifacts), local plots saved to disk.
	- Uses a single, local YAML config: `experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/task_shift_config.yaml`.

### Configuration and toggles

- The task-shift experiment reads `cfg/task_shift_config.yaml` in its folder. Important fields:
	- `data`, `net`, `learner`, and `task_shift_mode` mirror existing training configs in this repo.
	- `lla` block is evaluation-only and controls what to run at the end of each task:
		- `enable_planes: true|false`, `enable_spectrum: true|false`, `enable_sam: true|false`
		- `evaluation_data.eval_batch_size`, plane grid controls, spectrum ESD controls, etc.
	- Top-level W&B isolation: `wandb.project: loss_landscape_taskshift_${task_shift_mode}` to avoid mixing with other experiments.

### Logging policy (important)

- Per-epoch logging (during training):
	- `global_epoch` (task-aware step), `task_idx`, `epoch_loss`, and `epoch_accuracy` (when applicable) are logged each epoch.
	- `global_epoch = task_idx * epochs_per_task + epoch` ensures there’s no overwrite across tasks.

- End-of-task logging (after last epoch of a task):
	- Only numeric summaries are logged: LLA timings, Hessian top-k (when spectrum enabled), layer-wise rank/gini/weight stats, and convenience fields `task_last_epoch_loss` and `task_last_epoch_accuracy`.
	- The log step aligns with the last epoch of the task: `global_epoch = (task_idx + 1) * epochs_per_task - 1`.
	- Images and large files are not sent to W&B; they are saved locally under `outputs/loss_landscape_taskshift/<arch>/task_XXXX_<arch>/`.

### Correctness and stability details

- Evaluations run with `model.eval()` to stabilize BN/Dropout; HVP/eigendecompositions are performed in eval mode and the original mode is restored afterward where relevant.
- To ensure the center of the plane is a minimum after training a task, the LLA evaluation uses a snapshot of the current task’s dataset (best-effort freeze) to avoid drift during eval. This helps match the training distribution and fixes “off-center minima” issues.
- For large models, reduce plane span/resolution, fewer ESD probes/Lanczos steps, or exclude BN/bias from directions for better runtime.

### Quick start (task-shift LLA)

```bash
# Minimal dry run (one task, one epoch, no W&B)
python experiments/Hessian_and_landscape_plot_with_plasticity_loss/loss_landscape_change_during_task_shift.py \
	num_tasks=1 epochs=1 use_wandb=false lla.enable_spectrum=true lla.enable_planes=true lla.enable_sam=false
```

### What the next contributor can pick up

- Add CLI flags that override YAML LLA toggles (currently YAML-only is supported).
- Extend dataset “freeze” support if your wrappers expose a more explicit API.
- Add smoke tests for the end-of-task LLA output structure and W&B numeric keys.
- Optimize runtime for larger models by shrinking plane grids and ESD probes.

# architectural_and_learning_on_loss_landscape

## Documentation
- Codebase overview: `docs/CODEBASE_OVERVIEW.md` (see also `docs/ARCHITECTURE.md`)
- Extending the codebase: `docs/EXTENDING.md`
- Datasets and transforms: `docs/DATA_LOADING.md`
- Experiments: structure and templates: `docs/EXPERIMENTS.md`

# to-do:
# layer norm models need to pair with use of effective learning rate 
# add support for partial jacabian rank.
# add timer for rank computation to assess computational bottleneck.



### create flexible way to manage data logging for raw data
### manage experiments with many runs, one after another, save to different folders under different names
### create flexible way to plot from raw data
### 



#### replicate "rank diminising in deep neural networks"
- subtasks:
- function for partial rank calculation
- function for numerical rank
- function for effective rank


#### Refer to "How does batch normalization help optimization", 
- subtask: mechanism/function for measure change in loss (Lipschitzness of loss function)
- subtask: mechanism/function for measuring gradient predictiveness ("beta"-smoothness of loss function, or Lipschitzness of gradient of loss function)
- Then, test it on both VGG and DLN.

#### new architecture:
##### Full-rank projection network:
Instead of mapping each layer to a lower dimension, map it to the same dimension, followed by a projection, then add the constant.
##### decomposed normalized CNN:
- for each layer, with normalized weights, compute normalized weights, then obtained an normlized weights. Used that for forward pass. 
(efficiency? Reference layer norm implementation for idea.)
--- update normalized_weights_FC with better track of variance of input
--- how should input variance be related to the best weight norm for best loss landscape? 
- finally, multiply a constant factor to each filter with normalized weights.
-- separate behavior of .train() vs .
- calculate the time of forward/backward pass compared to baseline.
#### parametric non-linear relu:
- use parametric relu
- use MaxOut as activation.

### SVD decomposition parametrisation layer:
Use SVD decomposition, with a non-linear map sandwiched between.



## handling loss landscape of layer dependency:
#### main idea: Invent algorithms for having different learning rates for different layers 
- use and exponentially decaying version, with faster decay at earlier layers, vs faster decay at later layers.
- updates later layers more frequently
- higher learning rates for later layers





write model loading (done)
- model factory, outline from chatgpt
- use VGG and convnet first
- test model loading.

- write backprop (done!)
- test_backprop.py (done)
- generate results_raw from basic_config
- plot results


For each subfolder in experiments (test on basic_training):
- design of experiments (many runs) vs single run
- build generate_cfg_for_different_hyperparam.py
- this should generate many cfg corresponding to different hyperparm for experiments
- design save logic for single_expr, so that different settings can be saved in results/results_raw 
- 


- use superivsed_factory.py to centralize the creation of learners in src.algos.supervised
- test the function

-- design the ideal data format. Does loading the line-by-line as in loss-of-plasticity offer any advantages? 




- build 


basic folder structure:

project/
├── src/
│   ├── algos/
│   ├── envs/
│   ├── nets/
│   └── utils/
├── experiments/
│   ├── imagenet/
│   ├── incremental_cifar/
│   ├── permuted_mnist/
│   ├── rl/
│   └── slowly_changing_regression/
├── configs/
│   ├── imagenet/
│   ├── incremental_cifar/
│   └── ... etc.
├── tests/
├── docs/
├── scripts/
├── data/
├── README.md
├── requirements.txt
└── setup.py