# Plasticity Analysis & Plotting Pipeline

This directory contains the analysis pipeline for studying neural network plasticity, specifically focusing on "Forward Loss" as a leading indicator of learnability.

## 🚀 Quick Start

The main entry point is `plasticity_plotting_pipeline.py`. It fetches runs from W&B, processes them, and generates plots.

```bash
# Run with default configuration
python plasticity_plotting_pipeline.py --config cfg/plotting_config.yaml

# Dry run (check W&B connection and run counts without processing)
python plasticity_plotting_pipeline.py --config cfg/plotting_config.yaml --dry-run
```

## 📂 File Structure

The pipeline has been refactored into a modular structure:

- `plasticity_plotting_pipeline.py`: The main entry point script.
- `analysis/helpers/`: Directory containing helper modules:
    - `config_utils.py`: Configuration loading, merging, and parsing.
    - `wandb_utils.py`: W&B API interaction (fetching runs, history extraction).
    - `data_processing.py`: Data manipulation, history sampling, and metric computation (Forward Loss, Weight Norm).
    - `plotting_utils.py`: Shared plotting constants, filename generation, and display name management.
    - `plotting_functions.py`: Core plotting logic for all visualization types.
- `cfg/plotting_config.yaml`: The master configuration file controlling data fetching, processing, and plotting.
- `output_plots/`: Generated plots and summaries are saved here.

## 🧠 Key Concepts & Methodology

### 1. History Sampling
Experiments run over an "outer loop" of tasks (e.g., 1 to 500). The pipeline groups W&B history by `task_idx` and keeps only the **last row per group** (the state at the end of each task).

### 2. Forward Loss ($\hat{L}_t$)
We measure plasticity using a custom metric called **Forward Loss**. For any given task $t$, forward_loss is calculated by looking ahead into a future window of tasks $[t+1, \dots, t+w]$.
- **Window Size**: Configurable in `plotting_config.yaml` (default: 10).
- **Calculation**: Typically an Exponential Moving Average (EMA) where immediate future tasks are weighted more heavily.

## 📊 Plot Types

The pipeline generates four types of visualizations (configurable in YAML):

1.  **Future Loss Distribution**: Violin/Box plots showing the distribution of forward loss for different learners.
2.  **Parallel Coordinates**: High-dimensional view of metrics (Rank, Loss, Weight Norm) across time.
3.  **Phase Space Portrait**: A 2D connected scatterplot (Advance-Coordinate Embedding) mapping internal state (e.g., Rank Drop Gini) to future outcome (Forward Loss).
4.  **Scatter Matrix**: Pairwise scatterplots of selected metrics to identify correlations.
    - **Best Fit Lines**: Can be enabled via `add_best_fit_line: True`.
    - **Log Scaling**: Variables can be log-scaled via `variable_scales` (e.g., `forward_loss: "log"`). Diagonal KDEs automatically adjust to represent density in log-space.

## ⚙️ Configuration (`cfg/plotting_config.yaml`)

The YAML file controls the entire pipeline:
- **`wandb`**: Project entity/name.
- **`filters`**: Which runs to fetch (state, tags, config overrides).
- **`target_runs`**: Specific model/learner configurations to compare.
- **`plots`**: Toggles and settings for each plot type.
- **`display_names`**: Override or extend axis labels (e.g., `effective_rank: "Eff. Rank"`).

### Output Directory Logic
Unlike earlier designs that branched based on the number of targets, the current code uses a unified output structure defined in `global_settings`:
- **`base_output_dir`**: The root folder for plots (e.g., `./output_plots`).
- **`output_subfolder`**: An optional subfolder (e.g., `learner_comparison_on_ConvNet`).
- **`comparison_name`**: A prefix added to all generated filenames (e.g., `rr_cbp2_vs_cbp_...`).

All plots are saved to `base_output_dir / output_subfolder`.

## 🛠️ Dependencies

Requires standard data science stack + `wandb`:
`pip install pandas matplotlib seaborn wandb pyyaml`
