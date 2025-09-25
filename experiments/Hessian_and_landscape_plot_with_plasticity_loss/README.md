Hessian_and_landscape_plot_with_plasticity_loss

Quick start

- Run the pipeline with the default config:

	python lla_pipeline.py --config experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config.yaml --task all --outdir experiments/Hessian_and_landscape_plot_with_plasticity_loss/results

- Tasks: planes, spectrum, sam, mode, rank (or 'all').

Notebook demo

- Open the notebook at:

  experiments/Hessian_and_landscape_plot_with_plasticity_loss/LLA_demo.ipynb

- It performs a minimal end-to-end run using the same config and writes artifacts under `experiments/Hessian_and_landscape_plot_with_plasticity_loss/results/demo_notebook`.

Notes

- The current implementation uses lightweight PyTorch routines to approximate Hessian eigenvectors, spectrum, SAM surface, and a simple mode connectivity midpoint (average). It does not require external repos.
- Defaults are conservative: short or zero fine-tune, single fixed batch for evaluation, and small probe counts.

## ResNet18 quick profile (CIFAR-10)

To gauge compute on a heavier model, use the provided ResNet config:

- Run planes (Hessian directions + random):

	python lla_pipeline.py --config experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config_resnet.yaml --task planes --outdir experiments/Hessian_and_landscape_plot_with_plasticity_loss/results_resnet

- Run spectrum (top-k, Hutchinson trace, ESD via SLQ):

	python lla_pipeline.py --config experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config_resnet.yaml --task spectrum --outdir experiments/Hessian_and_landscape_plot_with_plasticity_loss/results_resnet

- Run SAM surface:

	python lla_pipeline.py --config experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config_resnet.yaml --task sam --outdir experiments/Hessian_and_landscape_plot_with_plasticity_loss/results_resnet

Observed timings on a single GPU (CIFAR-10, eval batch=64):

- Hessian top-2 (power iters=20): ~6.3 s; parameter dim ≈ 11,181,642
- Plane grids (41x41=1681 points): ~12–13 s per plane; ~7.1–7.6 ms/point
- Spectrum: total ~26 s (top-k k=5: ~7.6 s; Hutchinson: ~0.3 s; ESD SLQ m=50, probes=8: ~18.3 s)
- SAM surface (same grid): ~23.6 s grid eval; ~14 ms/point (grad-norm included)

Artifacts are written under `experiments/Hessian_and_landscape_plot_with_plasticity_loss/results_resnet/`.

Tips to trim compute if needed:

- Set `lla.planes.grid_resolution` to 31 or increase `sam.subsample_stride` (e.g., 2–4)
- Reduce `lla.spectrum.esd.lanczos_steps` and `num_probes`
- Toggle `lla.planes.bn.exclude_bn_and_bias: true` to shrink Hessian/vector dims