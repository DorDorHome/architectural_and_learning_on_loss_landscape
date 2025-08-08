#!/usr/bin/env python3
"""
WandB Rank Dynamics Uploader

Computes rank dynamics for existing W&B runs and logs them back to the same runs.
Handles all rank measures (effective_rank, approximate_rank, l1_distribution_rank, numerical_rank)
and all dynamics metrics (gini, centroid, AURC) for each rank measure.

Features:
- Duplicate detection and prevention
- Support for ongoing runs (only new steps)
- Dry run mode for safety
- Batch confirmation
- Progress tracking
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import wandb
import time
import torch
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.zeroth_order_features import compute_rank_decay_dynamics

class DictToObj:
    """Convert dictionary configs to objects with dot notation support."""
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
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)

def compute_first_feature_dim_once(run) -> Optional[int]:
    """
    Compute the first feature dimension once per run using model reconstruction.
    This is called only once during run setup, not in the training loop.
    """
    try:
        print(f"  🔧 Reconstructing model to get first feature dimensions (one-time setup)...")
        
        # Import required modules
        from src.models.model_factory import model_factory
        from src.data_loading.dataset_factory import dataset_factory
        from src.data_loading.transform_factory import transform_factory
        
        # Reconstruct model from config
        config = DictToObj(run.config)
        net = model_factory(config.net)
        net.eval()
        
        # Get dataset info for creating sample input
        transform = transform_factory(config.data.dataset, config.net.type)
        train_set, _ = dataset_factory(config.data, transform=transform, with_testset=False)
        
        # Create a tiny sample batch (just 2 samples for efficiency)
        sample_size = min(2, len(train_set))
        sample_inputs = []
        for i in range(sample_size):
            sample_inputs.append(train_set[i][0])
        
        sample_batch = torch.stack(sample_inputs)
        
        # Use .predict method to get features (same as train_data_shift_mode.py)
        with torch.no_grad():
            _, feature_list = net.predict(sample_batch)
        
        if not feature_list:
            print(f"  ❌ No features extracted from model")
            return None
            
        # Get first feature and flatten it (same as train_data_shift_mode.py line 339-340)
        first_feature = feature_list[0]
        first_feature_flattened = first_feature.view(first_feature.size(0), -1)
        
        # Get flattened feature dimension (this is fixed for the architecture)
        first_feature_dim = first_feature_flattened.shape[1]
        
        print(f"  📊 First feature shape: {first_feature.shape}")
        print(f"  📊 First feature flattened dim: {first_feature_dim}")
        
        # Clean up to save memory
        del net, sample_batch, feature_list, first_feature, first_feature_flattened
        
        return first_feature_dim
        
    except Exception as e:
        print(f"  ❌ Error computing first feature dimension: {e}")
        import traceback
        print(f"  📊 Full traceback: {traceback.format_exc()}")
        return None

def compute_theoretical_max_rank_fast(first_feature_dim: int, batch_size: int) -> int:
    """
    Fast computation of theoretical max rank given pre-computed feature dimension.
    This is called for every training step - must be very fast.
    """
    return min(first_feature_dim, batch_size)

def discover_and_sort_rank_keys(run, rank_types: List[str] = None) -> Optional[Dict[str, List[str]]]:
    """
    Discover and sort rank keys for all rank measures with enhanced progress tracking.
    
    Args:
        run: W&B run object
        rank_types: List of rank types to look for. If None, uses all available types.
    
    Returns:
        Dict mapping rank_type -> sorted list of rank keys, or None if discovery fails
    """
    if rank_types is None:
        rank_types = ['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank']
    
    try:
        print(f"  🔍 Discovering available rank keys...")
        
        # Use summary keys first (much faster than scanning history)
        all_keys = set(run.summary.keys())
        print(f"  📊 Found {len(all_keys)} keys in run summary")
        
        # If summary is empty, fallback to history scan (slower)
        if not all_keys:
            print(f"  ⚠️  Summary empty, scanning history (this may take time)...")
            keys_sample = run.scan_history(keys=[], page_size=10)  # Reduced page size
            for i, row in enumerate(keys_sample):
                all_keys.update(row.keys())
                if i >= 5:  # Only sample first few rows
                    break
            print(f"  📊 Found {len(all_keys)} keys from history sample")
        
        # Find rank keys for each rank type
        rank_keys_by_type = {}
        
        # loop through the proxy rank measures
        for rank_type in rank_types:
            print(f"  🔍 Looking for {rank_type} keys...")
            rank_keys = [key for key in all_keys if key.endswith(f'_{rank_type}')]
            
            if not rank_keys:
                print(f"  ⚠️  No {rank_type} keys found")
                continue
            
            print(f"  📋 Found {len(rank_keys)} {rank_type} keys: {rank_keys[:3]}{'...' if len(rank_keys) > 3 else ''}")
            
            # Try to get correct layer order using model reconstruction
            print(f"  🔧 Attempting model reconstruction for {rank_type}...")
            correct_order = get_correct_layer_order(run, rank_type)
            if correct_order:
                rank_keys_by_type[rank_type] = correct_order
                print(f"  ✅ {rank_type}: {len(correct_order)} layers (model reconstruction)")
            else:
                # Fallback to simple sorting
                print(f"  🔄 Using fallback sorting for {rank_type}...")
                sorted_keys = simple_fallback_sort(rank_keys, rank_type)
                if sorted_keys:
                    rank_keys_by_type[rank_type] = sorted_keys
                    print(f"  ✅ {rank_type}: {len(sorted_keys)} layers (fallback sort)")
                else:
                    print(f"  ❌ Could not sort {rank_type} keys")
        
        return rank_keys_by_type if rank_keys_by_type else None
        
    except Exception as e:
        print(f"  ❌ Error discovering rank keys: {e}")
        return None

def get_correct_layer_order(run, rank_type: str) -> Optional[List[str]]:
    """Get correct layer order using model reconstruction."""
    try:
        # Method 1: Try to get from W&B config
        if hasattr(run, 'config') and 'layer_order' in run.config:
            layer_names = run.config['layer_order']
            rank_keys = [f"{name}_{rank_type}" for name in layer_names]
            return rank_keys
        
        # Method 2: Reconstruct model from config
        if hasattr(run, 'config') and all(key in run.config for key in ['net', 'learner']):
            try:
                from src.models.model_factory import model_factory
                from src.algos.supervised.backprop_with_semantic_features import BackpropWithSemanticFeatures
                
                config = run.config
                net_config_obj = DictToObj(config['net'])
                learner_config_obj = DictToObj(config['learner'])
                
                net = model_factory(net_config_obj)
                
                if learner_config_obj.type == 'backprop':
                    learner = BackpropWithSemanticFeatures(net, learner_config_obj)
                else:
                    print(f"  ⚠️  Unsupported learner type: {learner_config_obj.type}")
                    return None
                
                layer_names = list(learner.feature_hook_manager.feature_names)
                rank_keys = [f"{name}_{rank_type}" for name in layer_names]
                return rank_keys
                
            except ImportError as e:
                print(f"  ⚠️  Import error during model reconstruction: {e}")
                return None
            except Exception as e:
                print(f"  ⚠️  Model reconstruction failed: {e}")
                return None
        
        return None
        
    except Exception as e:
        print(f"  ⚠️  Layer order discovery failed: {e}")
        return None

def simple_fallback_sort(discovered_keys: List[str], rank_type: str) -> Optional[List[str]]:
    """Simple fallback sorting for indexed or semantic keys."""
    try:
        # Check if keys follow indexed pattern (layer_0, layer_1, etc.)
        indexed_pattern = all(any(f"layer_{i}_{rank_type}" in key for key in discovered_keys) 
                             for i in range(len(discovered_keys)))
        
        if indexed_pattern:
            # Sort by index
            def extract_index(key):
                try:
                    parts = key.replace(f'_{rank_type}', '').split('_')
                    for part in parts:
                        if part.isdigit():
                            return int(part)
                    return float('inf')
                except:
                    return float('inf')
            
            return sorted(discovered_keys, key=extract_index)
        else:
            # Handle specific semantic patterns for your project
            print(f"  🔄 Attempting semantic pattern matching for {rank_type}...")
            
            # Define the expected layer order for your architecture
            expected_patterns = [
                'conv1_pooled',
                'conv2_pooled', 
                'conv3_flattened',
                'fc1_output(with ln)',
                'fc2_output(with ln)'
            ]
            
            # Try to match discovered keys to expected patterns
            matched_keys = []
            for pattern in expected_patterns:
                target_key = f"{pattern}_{rank_type}"
                if target_key in discovered_keys:
                    matched_keys.append(target_key)
            
            # Check if we found all expected keys
            if len(matched_keys) == len(discovered_keys) and len(matched_keys) == 5:
                print(f"  ✅ Successfully matched semantic layer pattern")
                print(f"      Layer order: {[key.replace(f'_{rank_type}', '') for key in matched_keys]}")
                return matched_keys
            else:
                print(f"  ⚠️  Partial semantic match. Found {len(matched_keys)}/{len(discovered_keys)} expected keys")
                print(f"      Expected: {expected_patterns}")
                print(f"      Found: {[key.replace(f'_{rank_type}', '') for key in discovered_keys]}")
                
                # If we have exactly 5 keys but they don't match perfectly, 
                # try a more flexible approach based on the layer type order
                if len(discovered_keys) == 5:
                    print(f"  🔄 Attempting flexible semantic sorting...")
                    
                    def get_layer_priority(key):
                        layer_name = key.replace(f'_{rank_type}', '')
                        if 'conv1' in layer_name:
                            return (0, layer_name)
                        elif 'conv2' in layer_name:
                            return (1, layer_name)
                        elif 'conv3' in layer_name:
                            return (2, layer_name)
                        elif 'fc1' in layer_name:
                            return (3, layer_name)
                        elif 'fc2' in layer_name:
                            return (4, layer_name)
                        else:
                            return (999, layer_name)  # Unknown layers go to end
                    
                    sorted_keys = sorted(discovered_keys, key=get_layer_priority)
                    layer_names = [key.replace(f'_{rank_type}', '') for key in sorted_keys]
                    print(f"  ✅ Flexible sorting successful: {layer_names}")
                    return sorted_keys
                
                return None
            
    except Exception as e:
        print(f"  ❌ Fallback sorting failed: {e}")
        return None

def get_existing_dynamics_steps(run, rank_type: str) -> Set[int]:
    """Get steps that already have rank dynamics logged with enhanced progress tracking."""
    try:
        print(f"    🔍 Checking existing {rank_type} dynamics...")
        
        # Check summary first for existence
        dynamics_keys = [
            f'{rank_type}_rank_drop_gini',
            f'{rank_type}_rank_decay_centroid', 
            f'{rank_type}_normalized_aurc'
        ]
        
        summary_keys = set(run.summary.keys())
        has_any_dynamics = any(key in summary_keys for key in dynamics_keys)
        
        if not has_any_dynamics:
            print(f"    📊 No existing {rank_type} dynamics found in summary")
            return set()
        
        print(f"    🔄 Scanning history for existing {rank_type} dynamics steps...")
        existing_steps = set()
        
        # Use minimal scan to get step information
        try:
            scan = run.scan_history(keys=[dynamics_keys[0], 'global_epoch'], page_size=100)
            step_count = 0
            for row in scan:
                if row.get(dynamics_keys[0]) is not None and row.get('global_epoch') is not None:
                    existing_steps.add(row['global_epoch'])
                    step_count += 1
                    if step_count % 1000 == 0:
                        print(f"    📊 Scanned {step_count} existing dynamics steps...")
            
            print(f"    ✅ Found {len(existing_steps)} existing {rank_type} dynamics steps")
            
        except Exception as e:
            print(f"    ⚠️  Error scanning {rank_type} dynamics history: {e}")
            return set()
                
        return existing_steps
    except Exception as e:
        print(f"    ❌ Error checking existing dynamics: {e}")
        return set()

def check_for_new_data(run, rank_types: List[str]) -> bool:
    """Check if an ongoing run has new data since last analysis."""
    try:
        # Get the latest step with any rank dynamics
        latest_dynamics_step = 0
        for rank_type in rank_types:
            dynamics_scan = run.scan_history(keys=[f'{rank_type}_rank_drop_gini', 'global_epoch'])
            for row in dynamics_scan:
                if row.get(f'{rank_type}_rank_drop_gini') is not None and row.get('global_epoch') is not None:
                    latest_dynamics_step = max(latest_dynamics_step, row['global_epoch'])
        
        # Get the latest step with rank data (check first rank type)
        if rank_types:
            rank_keys_by_type = discover_and_sort_rank_keys(run, [rank_types[0]])
            if rank_keys_by_type and rank_types[0] in rank_keys_by_type:
                rank_keys = rank_keys_by_type[rank_types[0]]
                if rank_keys:
                    rank_scan = run.scan_history(keys=[rank_keys[0], 'global_epoch'])
                    latest_rank_step = 0
                    for row in rank_scan:
                        if row.get(rank_keys[0]) is not None and row.get('global_epoch') is not None:
                            latest_rank_step = max(latest_rank_step, row['global_epoch'])
                    
                    return latest_rank_step > latest_dynamics_step
        
        return False
    except:
        return True  # If we can't check, assume we should update

def fetch_run_history_optimized(run, keys_to_fetch: List[str]) -> pd.DataFrame:
    """Optimized history fetching with progress tracking and timeout."""
    print(f"  🔄 Fetching history for {len(keys_to_fetch)} keys...")
    print(f"  📋 Keys: {keys_to_fetch[:5]}{'...' if len(keys_to_fetch) > 5 else ''}")
    
    try:
        import time
        start_time = time.time()
        
        # Use smaller page size for better progress tracking
        scan = run.scan_history(keys=keys_to_fetch, page_size=50)
        history_list = []
        
        batch_count = 0
        for row in scan:
            if row.get('global_epoch') is not None:
                history_list.append(row)
                batch_count += 1
                
                # Progress updates
                if batch_count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"    📊 Fetched {batch_count} rows in {elapsed:.1f}s...")
                
                # Timeout protection (adjust as needed)
                if time.time() - start_time > 300:  # 5 minutes timeout
                    print(f"    ⚠️  Timeout reached after {batch_count} rows, continuing with partial data...")
                    break
        
        elapsed = time.time() - start_time
        print(f"  ✅ Fetched {len(history_list)} rows in {elapsed:.1f}s")
        
        if not history_list:
            print(f"  ❌ No history data found")
            return pd.DataFrame()
        
        return pd.DataFrame(history_list)
        
    except Exception as e:
        print(f"  ❌ Error fetching history: {e}")
        return pd.DataFrame()

def compute_dynamics_for_step(ranks_by_type: Dict[str, List[float]], 
                             analysis_mode: str = 'difference',
                             first_feature_dim: Optional[int] = None,
                             batch_size: Optional[int] = None) -> Dict[str, float]:
    """
    Compute rank dynamics for all rank types at a single step with efficient input-aware approach.
    
    Args:
        ranks_by_type: Dict mapping rank_type -> list of ranks for that type
        analysis_mode: 'difference' or 'ratio'
        first_feature_dim: Pre-computed first feature dimension (from compute_first_feature_dim_once)
        batch_size: Batch size for computing theoretical max input rank
    
    Returns:
        Dict with keys like '{rank_type}_rank_drop_gini', etc.
    """
    dynamics_results = {}
    
    for rank_type, ranks in ranks_by_type.items():
        if len(ranks) < 2:
            continue
            
        try:
            if first_feature_dim is not None and batch_size is not None:
                # Use efficient input-aware dynamics with theoretical max first feature rank
                theoretical_max_input_rank = compute_theoretical_max_rank_fast(first_feature_dim, batch_size)
                dynamics = compute_rank_decay_dynamics(
                    ranks, 
                    mode=analysis_mode,
                    decay_from_input=True,
                    assume_full_rank_input=True,
                    input_rank=theoretical_max_input_rank
                )
            else:
                # Fall back to original behavior (no input consideration)
                dynamics = compute_rank_decay_dynamics(
                    ranks, 
                    mode=analysis_mode,
                    decay_from_input=False
                )
            
            # Add prefixed metrics
            dynamics_results[f'{rank_type}_rank_drop_gini'] = dynamics['rank_drop_gini']
            dynamics_results[f'{rank_type}_rank_decay_centroid'] = dynamics['rank_decay_centroid']
            dynamics_results[f'{rank_type}_normalized_aurc'] = dynamics['normalized_aurc']
            
        except Exception as e:
            print(f"    ⚠️  Error computing dynamics for {rank_type}: {e}")
            continue
    
    return dynamics_results

def add_rank_dynamics_to_existing_runs(entity: str, project: str, 
                                     rank_types: List[str] = None,
                                     analysis_mode: str = 'difference',
                                     dry_run: bool = True,
                                     max_runs: Optional[int] = None,
                                     force_overwrite: bool = False) -> Dict[str, int]:
    """
    Compute rank dynamics for existing runs and log back to the same runs.
    
    Args:
        entity: W&B entity name
        project: W&B project name
        rank_types: List of rank types to analyze. If None, uses all available.
        analysis_mode: 'difference' or 'ratio' for computing rank drops
        dry_run: If True, don't actually log to W&B
        max_runs: Maximum number of runs to process (for testing)
    
    Returns:
        Dict with processing statistics
    """
    if rank_types is None:
        rank_types = ['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank']
    
    if dry_run:
        print("🧪 DRY RUN MODE - No data will be logged to W&B")
        print("   Set dry_run=False to actually log data")
    
    api = wandb.Api()
    
    # extract all runs in the project
    runs = api.runs(f"{entity}/{project}")
    
    if max_runs:
        runs = runs[:max_runs]
        print(f"🔍 Limited to first {max_runs} runs for testing")
    
    print(f"Found {len(runs)} runs in {entity}/{project}")
    
    # Analyze what would be done with enhanced progress tracking
    stats = {
        'total_runs': len(runs),
        'processed_runs': 0,
        'skipped_runs': 0,
        'error_runs': 0,
        'timeout_runs': 0,
        'new_steps_added': 0
    }
    
    print(f"📊 Analyzing run configurations...")
    analysis_summary = {}
    for i, run in enumerate(runs):
        try:
            print(f"  📋 Analyzing run {i+1}/{len(runs)}: {run.name}")
            existing_metrics = set(run.summary.keys())
            dynamics_metrics = set()
            for rank_type in rank_types:
                dynamics_metrics.update([
                    f'{rank_type}_rank_drop_gini',
                    f'{rank_type}_rank_decay_centroid', 
                    f'{rank_type}_normalized_aurc'
                ])
            
            has_any_dynamics = bool(dynamics_metrics.intersection(existing_metrics))
            
            analysis_summary[run.name] = {
                'state': run.state,
                'has_dynamics': has_any_dynamics,
                'needs_processing': force_overwrite or not has_any_dynamics or run.state == "running"
            }
        except Exception as e:
            print(f"  ❌ Error analyzing {run.name}: {e}")
            analysis_summary[run.name] = {'needs_processing': False, 'error': str(e)}
    
    # Show summary
    to_process = [name for name, info in analysis_summary.items() if info.get('needs_processing', False)]
    print(f"\nSummary:")
    print(f"  Runs that will be processed: {len(to_process)}")
    print(f"  Runs that will be skipped: {len(runs) - len(to_process)}")
    
    if to_process:
        print(f"\nRuns to process:")
        for name in to_process[:10]:  # Show first 10
            info = analysis_summary[name]
            reason = "ongoing" if info.get('state') == "running" else "missing dynamics"
            print(f"  - {name} ({reason})")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
    
    if not dry_run and to_process:
        confirm = input(f"\nProceed with processing {len(to_process)} runs? (y/N): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return stats
    
    # Process runs with enhanced tracking and timeout protection
    import time
    timeout_per_run = 600  # 10 minutes per run
    
    for i, run in enumerate(runs):
        if run.name not in to_process:
            stats['skipped_runs'] += 1
            continue
            
        print(f"\n{'='*60}")
        print(f"📊 Processing run {i+1}/{len(runs)}: {run.name}")
        print(f"   State: {run.state}")
        print(f"   URL: https://wandb.ai/{entity}/{project}/runs/{run.id}")
        print(f"{'='*60}")
        
        run_start_time = time.time()
        
        try:
            # 1. Discover rank keys for all rank types
            print(f"🔍 Step 1: Discovering rank keys...")
            rank_keys_by_type = discover_and_sort_rank_keys(run, rank_types)
            if not rank_keys_by_type:
                print(f"  ❌ Skipping {run.name} - cannot determine layer order for any rank type")
                stats['error_runs'] += 1
                continue
            
            print(f"  ✅ Found rank keys for: {list(rank_keys_by_type.keys())}")
            
            # 1.5. Extract batch_size from config for input-aware dynamics
            batch_size = None
            if hasattr(run, 'config'):
                # Try different possible config keys for batch size
                batch_size_keys = ['specified_batch_size', 'batch_size', 'train_batch_size']
                for key in batch_size_keys:
                    if key in run.config:
                        batch_size = run.config[key]
                        print(f"  📊 Found batch_size: {batch_size} (from config.{key})")
                        break
                
                if batch_size is None:
                    print(f"  ⚠️  No batch_size found in config, will use original dynamics (no input consideration)")
                else:
                    print(f"  ✅ Will use input-aware dynamics with theoretical max input rank")
            
            # 1.6. Compute first feature dimension once per run (efficient approach)
            first_feature_dim = None
            if batch_size is not None:
                print(f"🔧 Step 1.6: Computing first feature dimension (one-time setup)...")
                first_feature_dim = compute_first_feature_dim_once(run)
                if first_feature_dim is not None:
                    theoretical_max_example = compute_theoretical_max_rank_fast(first_feature_dim, batch_size)
                    print(f"  ✅ First feature dim: {first_feature_dim}")
                    print(f"  📊 Theoretical max input rank: min({first_feature_dim}, {batch_size}) = {theoretical_max_example}")
                else:
                    print(f"  ❌ Failed to compute first feature dimension, falling back to original dynamics")
                    batch_size = None  # Disable input-aware dynamics
            
            # 2. Check existing dynamics with progress tracking
            print(f"🔍 Step 2: Checking existing dynamics...")
            existing_steps_by_type = {}
            if not force_overwrite:
                for rank_type in rank_keys_by_type.keys():
                    existing_steps_by_type[rank_type] = get_existing_dynamics_steps(run, rank_type)
            else:
                print(f"  🔄 Force overwrite mode - will recompute all dynamics")
                for rank_type in rank_keys_by_type.keys():
                    existing_steps_by_type[rank_type] = set()  # Empty set means all steps are new
            
            # 3. Fetch run history with optimized method
            print(f"🔍 Step 3: Fetching run history...")
            all_rank_keys = []
            for keys in rank_keys_by_type.values():
                all_rank_keys.extend(keys)
            
            keys_to_fetch = list(set(all_rank_keys + ['global_epoch']))
            
            # Check timeout
            if time.time() - run_start_time > timeout_per_run:
                print(f"  ⏰ Timeout reached for {run.name}")
                stats['timeout_runs'] += 1
                continue
            
            history = fetch_run_history_optimized(run, keys_to_fetch)
            
            if len(history) == 0:
                print(f"  ❌ No history data found for {run.name}")
                stats['error_runs'] += 1
                continue
            
            # 4. Process steps with detailed progress tracking using global_epoch
            print(f"🔍 Step 4: Computing dynamics...")
            new_dynamics_by_epoch = {}
            steps_processed = 0
            
            # Filter for rows with valid global_epoch values
            valid_epoch_rows = history[history['global_epoch'].notna()]
            print(f"  📊 Found {len(valid_epoch_rows)} rows with global_epoch out of {len(history)} total rows")
            
            if len(valid_epoch_rows) == 0:
                print(f"  ❌ No rows with global_epoch found for {run.name}")
                stats['error_runs'] += 1
                continue
            
            print(f"  📊 Processing {len(valid_epoch_rows)} steps with global_epoch...")
            for index, row in valid_epoch_rows.iterrows():
                epoch = int(row['global_epoch'])
                
                # Progress updates every 1000 steps
                if steps_processed % 1000 == 0 and steps_processed > 0:
                    elapsed = time.time() - run_start_time
                    progress = (steps_processed / len(valid_epoch_rows)) * 100
                    print(f"    📊 Progress: {steps_processed}/{len(valid_epoch_rows)} ({progress:.1f}%) in {elapsed:.1f}s")
                
                # Timeout check
                if time.time() - run_start_time > timeout_per_run:
                    print(f"    ⏰ Timeout reached at step {steps_processed}, using partial results...")
                    break
                
                # Check if any rank type needs processing for this epoch
                needs_processing = False
                for rank_type in rank_keys_by_type.keys():
                    if epoch not in existing_steps_by_type[rank_type]:
                        needs_processing = True
                        break
                
                if not needs_processing:
                    continue
                
                # Collect ranks for each type at this epoch
                ranks_by_type = {}
                for rank_type, rank_keys in rank_keys_by_type.items():
                    ranks_at_epoch = []
                    for key in rank_keys:
                        value = row.get(key)
                        if value is not None:
                            ranks_at_epoch.append(float(value))
                        else:
                            break
                    
                    # Only include if we have complete data
                    if len(ranks_at_epoch) == len(rank_keys):
                        ranks_by_type[rank_type] = ranks_at_epoch
                
                if not ranks_by_type:
                    continue
                
                # Compute dynamics for this epoch using efficient input-aware approach
                dynamics = compute_dynamics_for_step(ranks_by_type, analysis_mode, first_feature_dim, batch_size)
                if dynamics:
                    new_dynamics_by_epoch[epoch] = dynamics
                
                steps_processed += 1
            
            elapsed = time.time() - run_start_time
            print(f"  ✅ Computed dynamics for {steps_processed} new epochs in {elapsed:.1f}s")
            
            if not new_dynamics_by_epoch:
                print(f"  ✅ No new epochs to process for {run.name}")
                stats['skipped_runs'] += 1
                continue
            
            # 5. Log dynamics back to the run using global_epoch as x-axis
            if not dry_run:
                print(f"  📤 Logging {len(new_dynamics_by_epoch)} epochs to W&B using global_epoch...")
                try:
                    with wandb.init(entity=entity, project=project, id=run.id, resume="allow") as resumed_run:
                        # Define metrics to use global_epoch as x-axis (avoids step monotonicity issues)
                        print(f"    🔧 Defining metrics to use global_epoch as x-axis...")
                        for rank_type in rank_keys_by_type.keys():
                            resumed_run.define_metric(f'{rank_type}_rank_drop_gini', step_metric='global_epoch')
                            resumed_run.define_metric(f'{rank_type}_rank_decay_centroid', step_metric='global_epoch') 
                            resumed_run.define_metric(f'{rank_type}_normalized_aurc', step_metric='global_epoch')
                        
                        batch_size = 100
                        epochs_logged = 0
                        
                        for epoch, dynamics in new_dynamics_by_epoch.items():
                            # Add global_epoch to the data dict instead of using step parameter
                            dynamics_with_epoch = dict(dynamics)
                            dynamics_with_epoch['global_epoch'] = epoch
                            resumed_run.log(dynamics_with_epoch)
                            epochs_logged += 1
                            
                            if epochs_logged % batch_size == 0:
                                print(f"    📤 Logged {epochs_logged}/{len(new_dynamics_by_epoch)} epochs...")
                    
                    print(f"  ✅ Successfully logged {len(new_dynamics_by_epoch)} epochs to {run.name}")
                    stats['processed_runs'] += 1
                    stats['new_steps_added'] += len(new_dynamics_by_epoch)
                    
                except Exception as e:
                    print(f"  ❌ Error logging to W&B for {run.name}: {e}")
                    stats['error_runs'] += 1
            else:
                print(f"  [DRY RUN] Would log dynamics for {len(new_dynamics_by_epoch)} epochs")
                stats['processed_runs'] += 1
                stats['new_steps_added'] += len(new_dynamics_by_epoch)
            
            elapsed = time.time() - run_start_time
            print(f"⏱️  Total time for {run.name}: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error processing {run.name}: {e}")
            import traceback
            print(f"  📊 Full error: {traceback.format_exc()}")
            stats['error_runs'] += 1
            continue
    
    # Print final statistics with enhanced details
    print(f"\n{'='*60}")
    print(f"📊 Processing Complete!")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Processed: {stats['processed_runs']}")
    print(f"  Skipped: {stats['skipped_runs']}")
    print(f"  Errors: {stats['error_runs']}")
    print(f"  Timeouts: {stats['timeout_runs']}")
    print(f"  New steps added: {stats['new_steps_added']}")
    print(f"{'='*60}")
    
    return stats

def main():
    """Interactive main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add rank dynamics to W&B runs')
    parser.add_argument('--entity', required=True, help='W&B entity name')
    parser.add_argument('--project', required=True, help='W&B project name')
    parser.add_argument('--rank-types', nargs='+', 
                       choices=['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank'],
                       default=['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank'],
                       help='Rank types to analyze')
    parser.add_argument('--analysis-mode', choices=['difference', 'ratio'], default='difference',
                       help='Mode for computing rank drops')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t actually log to W&B')
    parser.add_argument('--max-runs', type=int, help='Maximum number of runs to process (for testing)')
    parser.add_argument('--force-overwrite', action='store_true', 
                       help='Recompute and overwrite existing dynamics (useful for testing new features)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting rank dynamics analysis...")
    print(f"   Entity: {args.entity}")
    print(f"   Project: {args.project}")
    print(f"   Rank types: {args.rank_types}")
    print(f"   Analysis mode: {args.analysis_mode}")
    print(f"   Dry run: {args.dry_run}")
    if args.max_runs:
        print(f"   Max runs: {args.max_runs}")
    if args.force_overwrite:
        print(f"   Force overwrite: {args.force_overwrite}")
    
    stats = add_rank_dynamics_to_existing_runs(
        entity=args.entity,
        project=args.project,
        rank_types=args.rank_types,
        analysis_mode=args.analysis_mode,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
        force_overwrite=args.force_overwrite
    )
    
    print(f"\n🎉 Analysis complete! Check your W&B project for the new rank dynamics metrics.")

if __name__ == "__main__":
    main()
