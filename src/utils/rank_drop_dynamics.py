"""
Rank Drop Dynamics Computation Functions

This module provides functions to compute rank drop dynamics metrics including:
- rank_drop_gini: Gini coefficient of rank drops
- rank_decay_centroid: Centroid of rank decay
- normalized_aurc: Normalized Area Under the Rank Curve

These functions are designed to be reusable and can be used directly in training loops
without needing to read from W&B runs.
"""

import torch
from typing import List, Dict, Union, Optional
import numpy as np


def compute_rank_drops(ranks: List[float], mode: str = 'ratio') -> List[float]:
    """
    Compute rank drops between consecutive layers.
    
    Args:
        ranks: List of rank values across layers
        mode: 'difference' (absolute drops) or 'ratio' (relative drops)
    
    Returns:
        List of rank drops
    """
    if len(ranks) < 2:
        return []
    
    rank_drops = []
    for i in range(len(ranks) - 1):
        if mode == 'difference':
            drop = ranks[i] - ranks[i + 1]  # Positive if rank decreases
        elif mode == 'ratio':
            if ranks[i] > 0:
                drop = (ranks[i] - ranks[i + 1]) / ranks[i]  # Relative drop
            else:
                drop = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rank_drops.append(drop)
    
    return rank_drops


def compute_rank_drop_gini(ranks: List[float], mode: str = 'ratio') -> float:
    """
    Compute Gini coefficient of rank drops to measure inequality in rank decay.
    
    Args:
        ranks: List of rank values across layers
        mode: 'difference' or 'ratio'
    
    Returns:
        Gini coefficient (0 = perfectly equal drops, 1 = maximally unequal)
    """
    rank_drops = compute_rank_drops(ranks, mode)
    
    if len(rank_drops) == 0:
        return 0.0
    
    # Ensure non-negative values for Gini computation
    rank_drops = [max(0, drop) for drop in rank_drops]
    
    if sum(rank_drops) == 0:
        return 0.0
    
    # Sort in ascending order
    sorted_drops = sorted(rank_drops)
    n = len(sorted_drops)
    
    # Compute Gini coefficient
    numerator = sum((2 * i - n - 1) * drop for i, drop in enumerate(sorted_drops, 1))
    denominator = n * sum(sorted_drops)
    
    return numerator / denominator if denominator > 0 else 0.0


def compute_rank_decay_centroid(ranks: List[float]) -> float:
    """
    Compute the centroid (center of mass) of rank decay.
    
    This measures where most of the rank reduction occurs:
    - Lower values: rank drops early (shallow layers)
    - Higher values: rank drops late (deeper layers)
    
    Args:
        ranks: List of rank values across layers
    
    Returns:
        Centroid position (layer index weighted by rank values)
    """
    if len(ranks) <= 1:
        return 0.0
    
    # Use ranks as weights, layer indices as positions
    total_weight = sum(ranks)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(i * rank for i, rank in enumerate(ranks))
    return weighted_sum / total_weight


def compute_normalized_aurc(ranks: List[float]) -> float:
    """
    Compute normalized Area Under the Rank Curve (AURC).
    
    This measures how much rank is preserved across layers:
    - Higher values: better rank preservation
    - Lower values: more aggressive rank reduction
    
    Args:
        ranks: List of rank values across layers
    
    Returns:
        Normalized AURC (0 to 1, where 1 = perfect rank preservation)
    """
    if len(ranks) <= 1:
        return 1.0 if len(ranks) == 1 else 0.0
    
    # Compute area under the curve using trapezoidal rule
    area = 0.0
    for i in range(len(ranks) - 1):
        area += (ranks[i] + ranks[i + 1]) / 2
    
    # Normalize by the maximum possible area (if rank stayed constant)
    max_area = ranks[0] * (len(ranks) - 1)
    
    return area / max_area if max_area > 0 else 0.0


def compute_rank_dynamics_for_single_measure(ranks: List[float], 
                                           mode: str = 'ratio') -> Dict[str, float]:
    """
    Compute all rank dynamics metrics for a single rank measure.
    
    Args:
        ranks: List of rank values across layers for one rank measure
        mode: 'difference' or 'ratio' for computing drops
    
    Returns:
        Dictionary with rank dynamics metrics
    """
    return {
        'rank_drop_gini': compute_rank_drop_gini(ranks, mode),
        'rank_decay_centroid': compute_rank_decay_centroid(ranks),
        'normalized_aurc': compute_normalized_aurc(ranks)
    }


def compute_theoretical_max_first_feature_rank(first_feature_shape: tuple, batch_size: int) -> int:
    """
    Compute theoretical maximum rank for the first feature layer.
    
    Args:
        first_feature_shape: Shape of the first feature tensor (batch_size, feature_dim)
        batch_size: Batch size
    
    Returns:
        Theoretical maximum rank = min(batch_size, flattened_feature_dim)
    """
    if len(first_feature_shape) < 2:
        raise ValueError(f"Expected at least 2D feature shape, got {first_feature_shape}")
    
    # Calculate flattened feature dimension (all dims except batch)
    flattened_feature_dim = 1
    for dim in first_feature_shape[1:]:
        flattened_feature_dim *= dim
    
    return min(batch_size, flattened_feature_dim)


def compute_all_rank_dynamics(rank_summary_list: List[Dict[str, float]], 
                             first_feature_shape: tuple,
                             batch_size: int,
                             mode: str = 'ratio',
                             use_theoretical_max_first: bool = True) -> Dict[str, float]:
    """
    Compute rank dynamics for all rank measures with optional theoretical max first feature rank.
    
    Args:
        rank_summary_list: List of dictionaries, each containing rank measures for one layer
        first_feature_shape: Shape of the first feature tensor
        batch_size: Batch size used for training
        mode: 'difference' or 'ratio' for computing rank drops
        use_theoretical_max_first: Whether to prepend theoretical max first feature rank
    
    Returns:
        Dictionary with all rank dynamics metrics, keyed by '{measure_name}_{metric_name}'
    """
    rank_measures = ['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank']
    results = {}
    
    # Compute theoretical max first feature rank if requested
    theoretical_max_first = None
    if use_theoretical_max_first:
        theoretical_max_first = compute_theoretical_max_first_feature_rank(first_feature_shape, batch_size)
    
    for measure in rank_measures:
        # Extract ranks for this measure across all layers
        ranks = []
        
        # Add theoretical max first feature rank if enabled
        if use_theoretical_max_first:
            ranks.append(float(theoretical_max_first))
        
        # Add ranks from all layers
        for layer_data in rank_summary_list:
            if measure in layer_data:
                ranks.append(float(layer_data[measure]))
        
        # Skip if we don't have enough data points
        if len(ranks) < 2:
            continue
        
        # Compute dynamics for this measure
        dynamics = compute_rank_dynamics_for_single_measure(ranks, mode)
        
        # Add to results with prefixed keys
        for metric_name, value in dynamics.items():
            key = f'{measure}_{metric_name}'
            results[key] = value
    
    return results


def compute_rank_dynamics_from_features(feature_list: List[torch.Tensor],
                                      rank_summary_list: List[Dict[str, float]],
                                      batch_size: int,
                                      mode: str = 'ratio',
                                      use_theoretical_max_first: bool = True) -> Dict[str, float]:
    """
    Convenience function to compute rank dynamics directly from feature tensors.
    
    Args:
        feature_list: List of feature tensors from network layers
        rank_summary_list: List of rank summaries for each layer  
        batch_size: Batch size
        mode: 'difference' or 'ratio'
        use_theoretical_max_first: Whether to use theoretical max first feature rank
    
    Returns:
        Dictionary with rank dynamics metrics
    """
    if not feature_list:
        return {}
    
    # Get first feature shape
    first_feature_shape = feature_list[0].shape
    
    return compute_all_rank_dynamics(
        rank_summary_list=rank_summary_list,
        first_feature_shape=first_feature_shape,
        batch_size=batch_size,
        mode=mode,
        use_theoretical_max_first=use_theoretical_max_first
    )
