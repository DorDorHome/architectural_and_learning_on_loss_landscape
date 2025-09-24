Prompt: “LLA-centric loss-landscape suite (single-script + notebook)”
You are to generate two deliverables:

1. LLA_demo.ipynb — a self-contained Jupyter notebook that:
    * installs and imports the Loss Landscape Analysis (LLA) library from its GitHub repository,
    * demonstrates: Hessian-aligned plane plots, random & trajectory planes, BN-aware normalization, Hessian spectrum (top-k, trace, ESD), SAM-style robust surface, and Mode Connectivity plots,
    * saves all figures and JSON summaries to a results folder.
    * if you need to use dataset, DO NOT downlaod directly. Use the instructions in docs\DATALOADING.md
    * this notebook is for demonstration of different options available in the LLA library.

2. lla_pipeline.py — a single, importable Python module exposing clean functions you can call from other code:
    * get_hessian_axes(...), plot_plane(...), compute_spectrum(...), plot_sam_surface(...), fit_mode_connectivity(...), plot_mode_curve(...), plus a small main() with argparse so it can also be run as a script.
    * No extra local packages. Do not create hessian.py, bn.py, esd.py, etc.
3. finally, a loss_landscape_change_during_task_shift.py This file uses functions in lla_pipeline.py, or directly from the LLA library, whichever is more convenient.

A. Frameworks and repos to use (use these, don’t re-implement)
* LLA (Loss Landscape Analysis) GitHub:
    * Repo: https://github.com/GabdullinN/loss-landscape-analysis
    * Use its public entry points (e.g., src_lla modules such as viz_lla, viz_esd, metrics or their current equivalents documented in the repo). If pip install isn’t available, clone and pip install -e ., or add src to sys.path. GitHub
* Mode Connectivity (Garipov et al.):
    * Repo: https://github.com/timgaripov/dnn-mode-connectivity
    * Use it solely to fit a Bézier midpoint and evaluate loss along the path; plotting is done with matplotlib. GitHub+1
(Background references you may cite in notebook markdown cells, but do not clone unless needed: loss-landscapesclassic lib, PyHessian page.) GitHub+1

B. Environment & runtime constraints
* Target: use the default requirements.txt in root as starting point. Consider whether there would be conflicts. If there are, create a dedicated requirements.txt for this experiments.
* Keep everything runnable within ~10–25 minutes on CPU for MNIST/CIFAR-10 subsets.
* Do not create any extra .py modules beyond lla_pipeline.py.
* Notebook and script must be self-sufficient (install steps included at top).
Conda blueprint (show in notebook markdown + script docstring):
* Create an env with python=3.11; install torch, torchvision, matplotlib, numpy, scipy, tqdm, ipykernel.
* Clone LLA and pip install -e . (or sys.path.append("loss-landscape-analysis/src") as fallback).
* Clone dnn-mode-connectivity for the mode path routine.

C. Dataset, models, and seeds
* Dataset: CIFAR-10 (or MNIST) with a small eval split (e.g., 2k examples).
* Models: (i) small CNN and (ii) torchvision.models.resnet18(pretrained=False) to exercise BN.
* Training: do a very short fine-tune (e.g., 1–3 epochs) to get non-degenerate minima; also save an earlier checkpoint to enable trajectory-PCA axes.
* Set global seeds (Python/NumPy/PyTorch), deterministic flags where reasonable.

D. Exact functionality to implement
D1) Hessian-aligned and random planes (LLA)
* Using LLA, compute a Hessian-aligned plane at the final checkpoint:
    * axes = “hessian” (top-2 eigenvectors)
    * normalization = “filter” (BN-aware option if available)
    * evaluate a grid (α,β)∈[−1,1]2(α,β)∈[−1,1]2 with resolution 51×51.
* Also plot a random plane with the same settings.
* BN treatment: Evaluate with model.eval(). Provide a toggle to (A) ignore BN/bias params in directions and (B) include them; comment on differences in the notebook.
* Save: contour + 3D surface (.png) and the numeric grid (.npy/.csv) to ./results/planes/.
(LLA exposes multiple axes/normalization options — use what the repo documents; do not write your own Hessian/Lanczos here.) GitHub
D2) Trajectory-PCA plane (LLA)
* From saved checkpoints (e.g., last 5–10 checkpoints of the quick fine-tune), derive a trajectory subspace (PCA on parameter deltas), then plot a 2D plane using the top-2 trajectory directions, with the same grid/normalization.
* Save plots and grid files to ./results/planes_traj/.
D3) Hessian spectrum & quantitative metrics (LLA)
* With LLA’s Hessian analysis:
    * Print top-k eigenvalues (e.g., k=5),
    * Estimate trace (Hutchinson) with a small number of probes (e.g., K=10),
    * Compute ESD via stochastic Lanczos quadrature (choose e.g., K=10 probes, m=50 Lanczos steps), and plot the ESD histogram/density.
* Save values to ./results/spectrum/metrics.json and the plot to ./results/spectrum/esd.png.
(SLQ and ESD are standard in LLA; if the repo provides a helper like viz_esd, use it.) GitHub
D4) SAM-style robust surface (custom, minimal)
* On the same Hessian-aligned plane, compute an approximate robust loss at each grid node using the single-step SAM surrogate:
    * For parameters θ=θ0+αv1+βv2θ=θ0 +αv1 +βv2 , compute g=∇L(θ)g=∇L(θ), step to θ′=θ+ρ g/∥g∥θ′=θ+ρg/∥g∥ (use ρρ default 0.05 of the parameter norm scale), and evaluate L(θ′)L(θ′).
* Plot side-by-side: standard vs SAM-style surfaces. Save to ./results/sam/.
(This is the only bit you implement yourself; it’s ~30 lines. Keep it simple and robust.)
D5) Mode connectivity (Garipov)
* Load two solutions: θAθA  = your final checkpoint; θBθB  = a second run with different seed or an earlier solution.
* Fit a quadratic Bézier midpoint θMθM  minimizing the maximum loss along γ(t)=(1−t)2θA+2t(1−t)θM+t2θBγ(t)=(1−t)2θA +2t(1−t)θM +t2θB .
* Plot t↦L(γ(t))t↦L(γ(t)) before and after optimization; compute and print the barrier heights for linear interpolation and for the fitted curve.
* Save to ./results/mode_connectivity/curve.png and metrics.json.
(Use the official dnn-mode-connectivity routines to build the curve and evaluate the path.) GitHub

E. Code structure & quality requirements
Notebook (LLA_demo.ipynb)
1. Intro cell: What will be demonstrated, with a tiny math recap of the plane S(α,β)=L(θ0+αd1+βd2)S(α,β)=L(θ0 +αd1 +βd2 ).
2. Install/setup cell:
    * pip-install core libs; clone LLA; pip install -e ./loss-landscape-analysis (or add src to sys.path if needed); clone mode connectivity.
3. Imports & utility cell:
    * Torch, TorchVision, Matplotlib, NumPy/SciPy, TQDM; LLA modules (e.g., viz_lla, viz_esd, metrics).
    * Small helpers for saving grids/plots and timing.
4. Data & models:
    * CIFAR-10 subset loaders; define CNN and load resnet18; a quick train/fine-tune loop; checkpoint saving (e.g., every N steps).
5. Planes:
    * Random; Hessian-aligned; Trajectory-PCA — each with consistent grid and normalization; BN toggle experiments.
6. Spectrum:
    * Top-k eigenvalues, trace (Hutchinson), ESD (SLQ) + plot.
7. SAM surface:
    * Robust grid evaluation and side-by-side plots.
8. Mode connectivity:
    * Fit midpoint; plot loss vs tt; report barriers.
9. Summary cell:
    * Short table summarizing: top-k eigenvalues, trace, percentage of mass near zero (from ESD), barrier numbers, and a sentence comparing CNN vs ResNet.
Script (lla_pipeline.py)
* Header docstring describing dependencies and the five tasks above.
* Functions (all pure, importable):
    * prepare_data_and_models(cfg) -> (loaders, models, checkpoints).
    * get_hessian_axes(model, metric, cfg) -> (v1, v2, lambdas) using LLA’s Hessian routines.
    * plot_plane(model, metric, axes, cfg) -> dict (returns file paths and numeric summaries).
    * compute_spectrum(model, metric, cfg) -> dict (top-k, trace, ESD plot).
    * plot_sam_surface(model, metric, axes, cfg) -> dict.
    * fit_mode_connectivity(model_fn, thetaA, thetaB, cfg) -> (thetaM, logs).
    * plot_mode_curve(model_fn, thetaA, thetaM, thetaB, cfg) -> dict.
* Config: @dataclass with sensible defaults (grid size 51, plane range [-1,1], probes K=10, Lanczos m=50, SAM rho=0.05, seeds, outdir).
* CLI (argparse): flags like --task planes|spectrum|sam|mode, --model resnet18|cnn, --bn-mode ignore|include, --outdir.
* Logging & outputs: write a results/ tree with subfolders and a run_summary.json.

F. UX & plotting
* Use matplotlib; include contour and 3D surface for each plane; consistent colormaps, labels (α,β)(α,β), colorbar = loss.
* Persist grids (.npy) and small CSV summaries (e.g., center loss, min/max over grid).
* Print timing for major blocks.

G. Guardrails & gotchas
* Do not write new helper modules (hessian.py, bn.py, esd.py, axes.py, models.py). Use LLA and mode-connectivity APIs as-is.
* If LLA import fails, first try pip install -e ./loss-landscape-analysis; only then fall back to sys.pathinsertion pointing at /src.
* Always set model.eval() when evaluating grids, spectrum, and SAM surfaces; don’t update BN running stats accidentally.
* For filter normalization, follow LLA’s documented options; don’t home-brew a variant in this project.
* If CUDA is available, select the correct torch wheel and move tensors/models accordingly.

H. Acceptance criteria (must all pass)
* Notebook runs top-to-bottom on CPU on CIFAR-10 subset in ≤25 minutes.
* Produces three planes (random, Hessian, trajectory) for one model (and at least the random plane for the second model).
* Saves spectrum metrics (top-k, trace) and an ESD plot.
* Generates a SAM vs standard surface comparison for the Hessian plane.
* Fits a mode-connectivity curve and reports barrier values (linear vs curve).
* lla_pipeline.py functions can be imported and used from an external Python REPL; python lla_pipeline.py --task planes runs a demo and writes outputs.

I. Links (for the agent to clone/read)
* LLA (Loss Landscape Analysis) GitHub repo (features: axes including Hessian/Adam/trajectory, normalization, Hessian analysis incl. ESD/trace): GitHub
* Mode Connectivity (Garipov et al.) repo (Bézier curve finding and FGE): GitHub+1
* Classic loss-landscapes library (reference only; don’t install here): GitHub
* PyHessian project page (reference only; LLA already does spectrum, but this explains ESD/trace/top-k context): 
