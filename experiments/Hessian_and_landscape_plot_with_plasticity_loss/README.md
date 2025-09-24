Hessian_and_landscape_plot_with_plasticity_loss

Quick start

- Run the pipeline with the default config:

	python lla_pipeline.py --config experiments/Hessian_and_landscape_plot_with_plasticity_loss/cfg/config.yaml --task all --outdir experiments/Hessian_and_landscape_plot_with_plasticity_loss/results

- Tasks: planes, spectrum, sam, mode, rank (or 'all').

Notes

- The current implementation uses lightweight PyTorch routines to approximate Hessian eigenvectors, spectrum, SAM surface, and a simple mode connectivity midpoint (average). It does not require external repos.
- Defaults are conservative: short or zero fine-tune, single fixed batch for evaluation, and small probe counts.