# LLA Suite: Action Plan (decisions and scope)

This file records the decisions and concrete plan for implementing the LLA-centric loss-landscape suite.

## Goals
- Provide a self-contained pipeline (`lla_pipeline.py`) and a demonstration notebook (`LLA_demo.ipynb`) to compute and visualize:
  - Hessian-aligned, random, and trajectory-PCA planes
  - Hessian spectrum: top-k eigenvalues, trace (Hutchinson), ESD via SLQ
  - SAM-style robust surfaces on the same plane
  - Mode connectivity curves (Bézier midpoint optimization)
- Couple rank statistics with these checkpoints to study correlations between rank dynamics and plasticity (loss changes).
- Keep all new code and artifacts within `experiments/Hessian_and_landscape_plot_with_plasticity_loss/`.

## Runtime budgets
- `lla_pipeline.py`: target ≤ 8 hours on a single GPU (no task shifts here). Slight overrun is acceptable.
- `LLA_demo.ipynb`: target ≤ 1 hour end-to-end on CPU/GPU with CIFAR-10 subset.
- `loss_landscape_change_during_task_shift.py`: may run for weeks; design later once we benchmark costs.

## Data and models
- Use the existing factories:
  - Transforms: `src/data_loading/transform_factory.py`
  - Datasets: `src/data_loading/dataset_factory.py`
  - Models: `src/models/model_factory.py`
- Default model: `ConvNet`. Also support `resnet_custom` and `vgg_custom` via config.
- Evaluation batch: use a single fixed batch by default for planes and spectrum for speed/repro. Expose option to average across batches.

## Coupling with rank statistics (critical)
- Save N checkpoints during a short fine-tune run (1–3 epochs). Use these for:
  - Trajectory PCA axes (top-2 PCs of parameter deltas).
  - Rank statistics at the same checkpoints using approximate rank only (no entropy/effective-rank proxies by default).
  - Compute per-layer rank and the distribution of rank drops between consecutive checkpoints; summarize with Gini coefficient and basic stats.
  - Plasticity proxy: changes in loss between adjacent checkpoints (or per-epoch loss where applicable). Compute rank–plasticity correlation (Spearman and Pearson) and include in summary.
- Keep feature extraction batches small and consistent (seeded) to control cost.

## Planes and spectrum (LLA)
- Use LLA’s public APIs for axes and spectrum (fast variants where available), not custom Hessian code.
- Hessian-aligned plane: top-2 eigenvectors at the final checkpoint using Lanczos/power methods provided by LLA.
- Random plane: same normalization.
- Trajectory-PCA plane: top-2 PCs from saved checkpoints.
- Normalization: default `filter` if supported by LLA; configurable.
- Grid: default 41×41 over [-1, 1]^2; allow `high_res: true` to use 51×51. Chunk evaluation to limit memory.
- BatchNorm treatment: model.eval() always; config toggle `exclude_bn_and_bias` for direction construction.
- Spectrum: top-k, Hutchinson trace (K probes), and ESD via SLQ (K probes, m steps). Configurable with fallbacks if OOM.

## SAM-style robust surface
- On the Hessian-aligned plane, compute a single-step SAM surrogate per grid node:
  - θ' = θ + ρ g / ||g||, evaluate L(θ').
- Default `rho=0.05` of parameter-norm scale; configurable.
- Use the same resolution as the standard plane by default; allow subsampling (stride) via config.

## Mode connectivity (Garipov)
- For `lla_pipeline.py`: use a second independent quick run (different seed) to obtain θ_B. Fit the quadratic Bézier midpoint θ_M using the official repo routine on a small eval subset.
- For `loss_landscape_change_during_task_shift.py` (later): compare modes across tasks within the same run (θ_A = end of task t, θ_B = end of task t+Δ).
- Runtime assessment (estimates):
  - Evaluating the curve without optimization (default midpoint) is cheap (~minutes with small subset).
  - Optimizing θ_M adds iterative steps (tens to low hundreds) with repeated curve evaluations; with small data subsets (single minibatch) we expect 10–40 minutes for ConvNet, and longer for ResNet/VGG. We will make optimization steps and subset size configurable. If runtime becomes excessive, we fall back to default midpoint.

## Trajectory PCA checkpoint policy
- Goal: robust axes and meaningful rank/plasticity dynamics without heavy IO.
- Policy:
  - Choose a target number of checkpoints `max_checkpoints` (default 10 for ConvNet; 6 for ResNet/VGG on demo to save time).
  - Uniformly sample across training steps: save every `floor(total_steps / max_checkpoints)` steps.
  - For short fine-tune (1–3 epochs), this yields evenly spaced snapshots capturing trajectory curvature.
  - Use the last `max_checkpoints` if more were saved (cap in memory).

## Config schema (new `lla` section)
- `lla.training`: epochs_short, optimizer, lr, weight_decay, save_checkpoints: true, max_checkpoints, second_run_seed.
- `lla.evaluation_data`: eval_batch_size, dataset_fraction_for_landscape, fixed_eval_seed.
- `lla.planes`: enable, grid_resolution, high_res, span, normalization, include_{random,hessian,trajectory}, bn: {eval_mode_only, exclude_bn_and_bias}, chunk_size, max_eval_points_cap.
- `lla.spectrum`: enable, top_k, hutchinson_probes, esd: {num_probes, lanczos_steps, bins}.
- `lla.sam`: enable, rho, subsample_stride.
- `lla.mode_connectivity`: enable, curve_points, opt_steps, lr, use_second_run, data_subset_batches.
- `lla.rank_tracking`: enable, batch_size, approximate_rank_prop (default 0.99), compute_gini: true.
- `lla.runtime`: device, fp16(false), workers, pbar(true), abort_if_estimated_hours_over(optional).
- `lla.logging`: out_root (default `experiments/.../results`), save_numpy, save_csv, wandb (inherit top-level `use_wandb`).

## Outputs layout
- `results/planes/{hessian|random|trajectory}/(contour.png, surface.png, grid.npy, grid.csv, meta.json)`
- `results/spectrum/{esd.png, metrics.json}`
- `results/sam/{sam_surface.png, comparison.png, meta.json}`
- `results/mode_connectivity/{curve.png, metrics.json}`
- `results/summary/summary_table.json` (includes rank–plasticity correlations)

## Performance controls and fallbacks
- Use single fixed eval batch by default; allow averaging across N batches.
- Chunk plane evaluation; release GPU memory between stages (`del` + `torch.cuda.empty_cache()`).
- On OOM during Hessian/SLQ: reduce `num_probes`, `lanczos_steps`, grid resolution, or move to CPU if configured.
- SAM plane can be computed on a subsampled grid if needed.

## Logging
- Save to local results folder and (optionally) log scalar summaries and images to W&B, controlled by config.
- Ensure large files (.npy, checkpoints, images) are ignored by VCS via .gitignore.

## Risks and mitigations
- Heavy curvature costs on large nets: default to ConvNet; ResNet/VGG are opt-in.
- Rank SVD cost: use approximate rank only and small batches. Consider limiting rank computation to selected layers (config).
- Mode connectivity runtime: expose step/batch caps; fall back to default midpoint if over budget.

## Acceptance criteria mapping
- Notebook runs under 1 hour, produces the requested artifacts.
- Pipeline script produces planes, spectrum, SAM, mode connectivity, and rank–plasticity summaries within 8 hours default for ConvNet.
- Artifacts saved under `results/` and summarized in a single JSON.

## Implementation steps (next)
1) Add config at `experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config.yaml` with `lla.*` sections.
2) Scaffold `lla_pipeline.py` with function stubs and argparse; import model/data factories from `src` and LLA/mode-connectivity via submodules.
3) Implement data/model prep and short fine-tune with checkpoint saving.
4) Implement planes (random, Hessian, Traj-PCA) with grid eval + plotting + saving.
5) Implement spectrum (top-k, trace, ESD) and saving.
6) Implement SAM surface and side-by-side plots.
7) Implement rank tracking at checkpoints and correlation computations.
8) Implement mode connectivity (default midpoint + optional optimized midpoint) + metrics.
9) Write summary and W&B logging hooks.
10) Create `LLA_demo.ipynb` using the pipeline functions.
