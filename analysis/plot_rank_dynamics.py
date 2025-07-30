import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from architectural_and_learning_on_loss_landscape.src.utils.zeroth_order_features import compute_rank_decay_dynamics

def analyze_run(run_path, rank_type, num_layers, analysis_mode, output_dir):
    """
    Fetches data for a single W&B run, computes rank decay dynamics,
    and saves a plot.
    """
    print(f"Analyzing run: {run_path}")
    api = wandb.Api()
    
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        print(f"Error fetching run {run_path}. Skipping. Details: {e}")
        return

    # Define the metric keys to fetch
    rank_metric_prefix = "ranks/layer_"
    rank_keys = [f"{rank_metric_prefix}{i}/{rank_type}" for i in range(num_layers)]

    # Fetch history
    history = run.history(keys=rank_keys, pandas=True)
    
    if history.empty:
        print(f"Warning: No data found for the specified rank keys in run {run.name}. Skipping.")
        return

    # Compute derived metrics
    dynamics_results = []
    for index, row in history.iterrows():
        ranks_at_step = [row[key] for key in rank_keys if key in row and pd.notna(row[key])]
        
        if len(ranks_at_step) == num_layers:
            dynamics = compute_rank_decay_dynamics(ranks_at_step, mode=analysis_mode)
            dynamics['_step'] = row['_step']
            dynamics_results.append(dynamics)

    if not dynamics_results:
        print(f"Warning: Could not compute dynamics for run {run.name}. Check if all rank keys were present.")
        return

    dynamics_df = pd.DataFrame(dynamics_results)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Rank Decay Dynamics ({analysis_mode.capitalize()} Mode)\\nRun: {run.name}', fontsize=16)

    axes[0].plot(dynamics_df['_step'], dynamics_df['rank_drop_gini'])
    axes[0].set_title('Rank Drop Gini Coefficient')
    axes[0].set_ylabel('Gini Coefficient')
    axes[0].grid(True, linestyle='--')

    axes[1].plot(dynamics_df['_step'], dynamics_df['rank_decay_centroid'])
    axes[1].set_title('Rank Decay Centroid')
    axes[1].set_ylabel('Centroid (Layer Index)')
    axes[1].grid(True, linestyle='--')

    axes[2].plot(dynamics_df['_step'], dynamics_df['normalized_aurc'])
    axes[2].set_title('Normalized AURC')
    axes[2].set_ylabel('Normalized Area')
    axes[2].set_xlabel('Training Step')
    axes[2].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    plot_filename = f"{run.id}_rank_dynamics_{analysis_mode}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze rank decay dynamics from W&B runs.")
    parser.add_argument('--run_id', type=str, required=True, help='W&B run path (e.g., entity/project/run_id).')
    parser.add_argument('--rank_type', type=str, default='effective_rank', help='Type of rank to analyze.')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers whose ranks were tracked.')
    parser.add_argument('--mode', type=str, default='difference', choices=['difference', 'ratio'], help='Analysis mode for rank drops.')
    parser.add_argument('--output_dir', type=str, default='results/rank_plots', help='Directory to save plots.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    analyze_run(args.run_id, args.rank_type, args.num_layers, args.mode, args.output_dir)

if __name__ == '__main__':
    main()
