import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import sys
from pathlib import Path

# Add the src directory to the path to allow imports
script_dir = Path(__file__).parent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# src_dir = script_dir.parent / "src"
# sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(PROJECT_ROOT))



from src.utils.zeroth_order_features import compute_rank_decay_dynamics

def plot_rank_dynamics(dynamics_df, history_cleaned, rank_keys, run_name, rank_type, analysis_mode, output_path=None):
    """
    Enhanced plotting function that shows both derived dynamics and individual layer ranks.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(f'Rank Dynamics Analysis for Run: {run_name}\n(Rank Type: {rank_type}, Mode: {analysis_mode})', fontsize=16)

    # Plot 1: Individual Layer Ranks Over Time
    colors = plt.cm.Set1(np.linspace(0, 1, len(rank_keys)))
    for i, (key, color) in enumerate(zip(rank_keys, colors)):
        layer_name = key.replace(f'_{rank_type}', '')
        axes[0].plot(history_cleaned['_step'], history_cleaned[key], 
                    label=f'Layer {i+1}: {layer_name}', color=color, linewidth=2)
    
    axes[0].set_title(f'Individual Layer {rank_type.replace("_", " ").title()} vs. Training Step')
    axes[0].set_ylabel(f'{rank_type.replace("_", " ").title()}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Rank Drop Gini Coefficient
    axes[1].plot(dynamics_df['_step'], dynamics_df['rank_drop_gini'], 
                label='Gini Coefficient', color='red', linewidth=2)
    axes[1].set_title('Rank Drop Gini Coefficient vs. Training Step')
    axes[1].set_ylabel('Gini Coefficient')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Rank Decay Centroid
    axes[2].plot(dynamics_df['_step'], dynamics_df['rank_decay_centroid'], 
                label='Rank Decay Centroid', color='blue', linewidth=2)
    axes[2].set_title('Rank Decay Centroid vs. Training Step')
    axes[2].set_ylabel('Centroid (Layer Index)')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Normalized AURC
    axes[3].plot(dynamics_df['_step'], dynamics_df['normalized_aurc'], 
                label='Normalized AURC', color='green', linewidth=2)
    axes[3].set_title('Normalized AURC vs. Training Step')
    axes[3].set_ylabel('Normalized Area Under Rank Curve')
    axes[3].set_xlabel('Training Step')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig

def get_correct_layer_order(run, rank_type):
    """
    Get the correct layer order using model reconstruction.
    If that fails, return None to trigger fallback methods.
    """
    try:
        # Method 1: Try to get from W&B config (if we added it)
        if hasattr(run, 'config') and 'layer_order' in run.config:
            layer_names = run.config['layer_order']
            rank_keys = [f"{name}_{rank_type}" for name in layer_names]
            print(f"✅ Using layer order from W&B config: {layer_names}")
            return rank_keys
        
        # Method 2: Reconstruct model from config
        if hasattr(run, 'config') and all(key in run.config for key in ['net', 'learner']):
            print("🔧 Reconstructing model to get correct layer order...")
            
            try:
                # Import necessary modules (adjust paths as needed)
                from src.models.model_factory import model_factory
                from src.algos.supervised.backprop_with_semantic_features import BackpropWithSemanticFeatures
                
                # Reconstruct model
                config = run.config
                print(f"📊 Config keys: {list(config.keys())}")
                print(f"📊 Learner config: {config.get('learner', 'Not found')}")
                print(f"📊 Net config: {config.get('net', 'Not found')}")
                
                # Convert dict configs to objects with dot notation (for model_factory compatibility)
                class DictToObj:
                    def __init__(self, d):
                        for key, value in d.items():
                            if isinstance(value, dict):
                                setattr(self, key, DictToObj(value))
                            else:
                                setattr(self, key, value)
                    
                    def get(self, key, default=None):
                        """Support dict-like .get() method"""
                        return getattr(self, key, default)
                    
                    def __getitem__(self, key):
                        """Support dict-like [key] access"""
                        return getattr(self, key)
                    
                    def __contains__(self, key):
                        """Support 'key in obj' syntax"""
                        return hasattr(self, key)
                
                # Convert net config from dict to object
                net_config_obj = DictToObj(config['net'])
                learner_config_obj = DictToObj(config['learner'])
                
                print(f"📊 Net config type: {net_config_obj.type}")  # Verify conversion worked
                
                net = model_factory(net_config_obj)
                
                # Create learner - use the converted object
                if learner_config_obj.type == 'backprop':
                    learner = BackpropWithSemanticFeatures(net, learner_config_obj, net_config_obj)
                else:
                    raise NotImplementedError(f"Learner type {learner_config_obj.type} not supported")
                
                # Get layer names using the same method as training
                if hasattr(learner, 'get_layer_names'):
                    layer_names = learner.get_layer_names()
                    rank_keys = [f"{name}_{rank_type}" for name in layer_names]
                    print(f"✅ Reconstructed layer order: {layer_names}")
                    return rank_keys
                    
            except ImportError as e:
                print(f"⚠️  Could not import required modules: {e}")
            except Exception as e:
                print(f"⚠️  Model reconstruction failed: {e}")
                import traceback
                print(f"📊 Full error: {traceback.format_exc()}")
        
        print("⚠️  Model reconstruction not possible, using fallback methods...")
        return None
        
    except Exception as e:
        print(f"⚠️  Error in get_correct_layer_order: {e}")
        return None

def simple_fallback_sort(discovered_keys, rank_type):
    """
    Handle only the two cases from the training code:
    1. Indexed keys (layer_0, layer_1, etc.) - sort by index
    2. Semantic keys - require manual specification (no heuristics!)
    """
    if not discovered_keys:
        print("⚠️  No keys to sort")
        return None
    
    # Check if ALL keys are indexed (layer_X_effective_rank pattern)
    indexed_pattern = "layer_"
    indexed_keys = [key for key in discovered_keys 
                   if key.startswith(indexed_pattern) and key.endswith(f"_{rank_type}")]
    
    if len(indexed_keys) == len(discovered_keys):
        # All keys are indexed - sort them numerically (matches training fallback)
        def extract_layer_index(key):
            try:
                # Extract X from "layer_X_effective_rank"
                parts = key.replace(f"_{rank_type}", "").split("_")
                return int(parts[1])  # layer_X -> X
            except (IndexError, ValueError):
                return float('inf')  # Put malformed keys at the end
        
        discovered_keys.sort(key=extract_layer_index)
        layer_names = [key.replace(f'_{rank_type}', '') for key in discovered_keys]
        print(f"✅ Sorted indexed keys (training fallback case): {layer_names}")
        return discovered_keys
    
    else:
        # These are semantic names - require manual specification
        layer_names = [key.replace(f"_{rank_type}", "") for key in discovered_keys]
        print(f"⚠️  Found semantic layer names: {layer_names}")
        print("❌ Manual specification required - no automatic sorting for semantic names.")
        print("   (This matches the training code: semantic names work OR fallback to indexed names)")
        return None

def discover_and_sort_rank_keys(run, rank_type):
    """
    Discover and sort rank metric keys from a W&B run using the original training code logic.
    
    Args:
        run: W&B run object
        rank_type: Type of rank metric to look for (e.g., 'effective_rank')
    
    Returns:
        list: Sorted list of rank metric keys, or None if manual specification required
    """
    print("=== Starting Layer Order Discovery ===")
    
    # First, try the robust model reconstruction approach
    rank_keys = get_correct_layer_order(run, rank_type)
    if rank_keys is not None:
        return rank_keys
    
    # Fall back to discovering keys from the run and simple sorting
    print("🔍 Discovering available rank keys from W&B run...")
    all_keys = run.summary.keys()
    rank_suffix = f"_{rank_type}"
    discovered_keys = [key for key in all_keys if key.endswith(rank_suffix)]
    
    print(f"📊 Found {len(discovered_keys)} keys ending with '{rank_suffix}'")
    if len(discovered_keys) <= 10:  # Only print if not too many
        print(f"   Keys: {discovered_keys}")
    
    if not discovered_keys:
        raise ValueError(f"No keys found ending with '{rank_suffix}'. Available keys: {list(all_keys)[:10]}...")
    
    # Try simple fallback sorting
    rank_keys = simple_fallback_sort(discovered_keys, rank_type)
    
    if rank_keys is None:
        print("\n❌ ERROR: Cannot automatically determine layer order!")
        print("🔧 SOLUTION: Please specify the layer order manually.")
        print("   Example for script usage:")
        print("   MANUAL_LAYER_ORDER = ['conv1', 'conv2', 'fc1']")
        print("   Then modify the script to use this order.")
        return None
    
    print("=== Layer Order Discovery Complete ===\n")
    return rank_keys

def analyze_run(run_path, rank_type, analysis_mode, output_dir):
    """
    Fetches data for a single W&B run, computes rank decay dynamics,
    and saves enhanced plots.
    """
    print(f"Analyzing run: {run_path}")
    api = wandb.Api()
    
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        print(f"Error fetching run {run_path}. Skipping. Details: {e}")
        return

    # Discover and sort rank keys using the robust approach
    try:
        rank_keys = discover_and_sort_rank_keys(run, rank_type)
        if rank_keys is None:
            print(f"❌ Cannot determine layer order for run {run.name}. Skipping.")
            print("   Please specify manual layer order and modify the script.")
            return
    except ValueError as e:
        print(f"Error discovering rank keys: {e}")
        return

    # Fetch history using scan_history for complete data
    print(f"🔄 Fetching complete history for {len(rank_keys)} metrics...")
    try:
        keys_to_fetch = rank_keys + ['_step']  # Include step column
        scan = run.scan_history(keys=keys_to_fetch)
        history_list = [row for row in scan if row.get('_step') is not None]
        history = pd.DataFrame(history_list)
        
        if history.empty:
            print(f"❌ No data found for run {run.name}")
            return
            
        print(f"✅ Fetched {len(history)} steps from run: {run.name}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Clean data
    print(f"📊 Cleaning data...")
    print(f"   Original: {len(history)} rows")
    history_cleaned = history.dropna(subset=rank_keys)
    print(f"   After cleaning: {len(history_cleaned)} rows with complete rank data")
    
    if len(history_cleaned) == 0:
        print(f"❌ No complete data rows found for run {run.name}")
        return

    # Compute rank decay dynamics
    print(f"🔄 Computing rank decay dynamics...")
    dynamics_results = []
    for index, row in history_cleaned.iterrows():
        ranks_at_step = [row[key] for key in rank_keys]
        dynamics = compute_rank_decay_dynamics(ranks_at_step, mode=analysis_mode)
        dynamics['_step'] = row['_step']
        dynamics_results.append(dynamics)

    dynamics_df = pd.DataFrame(dynamics_results)
    print(f"✅ Computed dynamics for {len(dynamics_df)} steps")

    # Create enhanced plots
    output_filename = f"rank_dynamics_{run.name}_{rank_type}_{analysis_mode}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"📈 Creating enhanced plots...")
    plot_rank_dynamics(dynamics_df, history_cleaned, rank_keys, run.name, rank_type, analysis_mode, output_path)
    
    print(f"✅ Analysis complete for run: {run.name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze rank decay dynamics from W&B runs.")
    parser.add_argument('--run_id', type=str, required=True, help='W&B run path (e.g., entity/project/run_id).')
    parser.add_argument('--rank_type', type=str, default='effective_rank', help='Type of rank to analyze.')
    parser.add_argument('--mode', type=str, default='difference', choices=['difference', 'ratio'], help='Analysis mode for rank drops.')
    parser.add_argument('--output_dir', type=str, default='results/rank_plots', help='Directory to save plots.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    analyze_run(args.run_id, args.rank_type, args.mode, args.output_dir)

if __name__ == "__main__":
    main()
