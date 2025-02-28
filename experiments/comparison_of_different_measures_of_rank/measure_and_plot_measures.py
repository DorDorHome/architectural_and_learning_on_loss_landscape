import numpy as np
import time
import tracemalloc
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

from typing import Any
import sys
import pathlib
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from src.utils.zeroth_order_features import compute_effective_rank, compute_approximate_rank, compute_l1_distribution_rank, compute_numerical_rank


def assess_efficiency_and_memory_of_rank_measures(measure_func, input:torch.Tensor , compute_svd_externally = False, num_runs = 5):
    '''
    assess the execution time and memory usage of a rank measure function
    
    '''
    
    if compute_svd_externally:
        assert len(input.shape) == 2, "The input tensor should be 2d"
    
    elif not compute_svd_externally:
    # check the shape of input is 3d:
        assert len(input.shape) == 3, "The input input should be 3d, first dimension is the batch size"
    
    batch_size = input.shape[0]
    
    device = input.device
    times = []
    memories = []
    list_average_values_over_batch = []

    for _ in tqdm(range(num_runs), desc="Running..."):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        else:
            tracemalloc.start()

        start_time = time.perf_counter()
        
        # running the function:
        batch_values = measure_func(input = input, input_is_svd = compute_svd_externally)
        average_value_over_batch = batch_values.sum()/batch_size
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        end_time = time.perf_counter()
        times.append(end_time - start_time)
        memories.append(peak_memory)
        list_average_values_over_batch.append(average_value_over_batch)

    avg_time = sum(times) / num_runs
    avg_memory = sum(memories) / num_runs
    average_value_over_batch_and_run = sum(list_average_values_over_batch) / num_runs
    return average_value_over_batch_and_run , avg_time, avg_memory

    
if __name__ == '__main__':
    
    folder_for_raw_results = os.path.join(PROJECT_ROOT, 'experiments','comparison_of_different_measures_of_rank', 'results_raw')
    folder_for_plots = os.path.join(PROJECT_ROOT, 'experiments','comparison_of_different_measures_of_rank', 'results_plots')
    # statistical hyperparameters:
    num_runs = 30
    
    # channel size to test on:
    channel_sizes = [3, 9, 27, 81]
    
    # feature size to test on:
    feature_sizes = [10, 50, 100, 500, 1000, 5000]
    
    # placeholder for results:
    internal_svd_results = []
    external_svd_results = []
    
    # device to test on:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    batch_size = 1000
    # feature_dim = 1000, 512  # Typical layer output size
    # height, width = 28, 28  # Typical image size

#compute_effective_rank, compute_approximate_rank, compute_l1_distribution_rank, compute_numerical_rank
    # List of measures to evaluate
    measures = [
        # ("Stable Rank", stable_rank),
        ("Effective Rank", compute_effective_rank),
        ("Approximate Rank", compute_approximate_rank),
        ("Numerical Rank", compute_numerical_rank),
        ("L1 Distribution Rank", compute_l1_distribution_rank),
    ]
    
    
    for (channel_size, feature_size) in zip(channel_sizes, feature_sizes):
        # Generate a random tensor of the appropriate size
        input = torch.rand(batch_size, channel_size, feature_size, device=device)
        for name, func in measures:
            
            # for internal svd:
            average_value_over_batch_and_run, avg_time, avg_memory = assess_efficiency_and_memory_of_rank_measures(func, input, compute_svd_externally= False, num_runs=num_runs)
            
            memory_unit = "bytes" if device.type == 'cuda' else "bytes (CPU)"
            internal_svd_results.append({
                "batch_size": batch_size,
                "feature Size": feature_size,
                "channel_size": channel_size,
                "Input_size_without_batch": channel_size * feature_size,
                "Input_Size": batch_size * channel_size * feature_size,
                "Measure": name,
                "Value_averaged_over_batch_and_run": average_value_over_batch_and_run,
                "Time (s)": avg_time,
                "Memory (bytes)": round(avg_memory, -2)  # Round to the nearest hundred
            })
 
        
            print(f"{name}:")
            print('  batch_size:', batch_size)
            print('  feature Size:', feature_size)
            print('  channel_size:', channel_size)
            print(f"  Input_Size: {batch_size * channel_size * feature_size}")
            print(f"  Value_averaged_over_batch_and_run: {average_value_over_batch_and_run:.4f}")
            print(f"  Average Time: {avg_time:.6f} seconds")
            print(f"  Peak Memory: {avg_memory:.2f} {memory_unit}")
            print()
            

    # Convert results to a DataFrame
    df_internal_svd = pd.DataFrame(internal_svd_results)
    print(df_internal_svd)
    # save the df in the results folder:
    path_to_save_df_internal_svd = os.path.join(folder_for_raw_results, 'comparison_of_rank_measures_internal_svd_raw.csv')
    df_internal_svd.to_csv(path_to_save_df_internal_svd)

    # Step 5: Visualize the comparison
    plt.figure(figsize=(10, 6))

    # Plot time vs. input size for each measure
    for name in df_internal_svd["Measure"].unique():
        subset = df_internal_svd[df_internal_svd["Measure"] == name]
        plt.plot(subset["Input_Size"], subset["Time (s)"], label=f"{name} Time", marker="o")

    plt.xlabel("Input_Size")
    plt.ylabel("Time (s)")
    plt.title("Computational Efficiency of Proxy Measures using internal svd")
    plt.legend()
    plt.grid(True)
    #plt.show()
    path_to_save_time_plot = os.path.join(folder_for_plots,
                                          'comparison_of_internal_svd_rank_measures_time_plot.png')
    plt.savefig(path_to_save_time_plot)

    # Plot memory vs. feature size for each measure
    plt.figure(figsize=(10, 6))

    for name in df_internal_svd["Measure"].unique():
        subset = df_internal_svd[df_internal_svd["Measure"] == name]
        plt.plot(subset["Input_Size"], subset["Memory (bytes)"], label=f"{name} Memory", marker="o")

    plt.xlabel("Input_Size")
    plt.ylabel("Memory (bytes)")
    plt.title("Memory Requirements of Proxy Measures using internal svd")
    plt.legend()
    plt.grid(True)
    path_to_save_memory_plot = os.path.join(folder_for_plots, 'comparison_of_internal_svd_rank_measures_memory_plot.png')
    plt.savefig(path_to_save_memory_plot)
    #plt.show()
    
    # plot 
    for (channel_size, feature_size) in zip(channel_sizes, feature_sizes):
        # Generate a random tensor of the appropriate size
        input = torch.rand(batch_size, channel_size, feature_size, device=device)
        # for external svd:
        svd_input = torch.linalg.svdvals(input)
        
        for name, func in measures:        
            # for external svd:
            average_value_over_batch_and_run, avg_time, avg_memory = assess_efficiency_and_memory_of_rank_measures(func, svd_input,
                                                                                                                    compute_svd_externally= True,
                                                                                                                    num_runs=num_runs)
                
            external_svd_results.append({
            "batch_size": batch_size,
            "feature Size": feature_size,
            "channel_size": channel_size,
            "Input_size_without_batch": channel_size * feature_size,
            "Input_Size": batch_size * channel_size * feature_size,
            "Measure": name,
            "Value_averaged_over_batch_and_run": average_value_over_batch_and_run,
            "Time (s)": avg_time,
            "Memory (bytes)": round(avg_memory, -2)  # Round to the nearest hundred
            })
    
            
            print(f"{name}:")
            print('  batch_size:', batch_size)
            print('  feature Size:', feature_size)
            print('  channel_size:', channel_size)
            print(f"  Input_Size: {batch_size * channel_size * feature_size}")
            print(f"  Value_averaged_over_batch_and_run: {average_value_over_batch_and_run:.4f}")
            print(f"  Average Time: {avg_time:.6f} seconds")
            print(f"  Peak Memory: {avg_memory:.2f} {memory_unit}")
            print()
                

    # Convert results to a DataFrame
    df_external_svd = pd.DataFrame(external_svd_results)
    print(df_external_svd)
    # save the df in the results folder:
    path_to_save_df_external_svd = os.path.join(folder_for_raw_results, 'comparison_of_rank_measures_external_svd_raw.csv')
    df_external_svd.to_csv(path_to_save_df_external_svd)

    # Step 5: Visualize the comparison
    plt.figure(figsize=(10, 6))

    # Plot time vs. input size for each measure
    for name in df_external_svd["Measure"].unique():
        subset = df_external_svd[df_external_svd["Measure"] == name]
        plt.plot(subset["Input_Size"], subset["Time (s)"], label=f"{name} Time", marker="o")

    plt.xlabel("Input_Size")
    plt.ylabel("Time (s)")
    plt.title("Computational Efficiency of Proxy Measures with external svd")
    plt.legend()
    plt.grid(True)
    #plt.show()
    path_to_save_time_plot = os.path.join(folder_for_plots,
                                          'comparison_of_external_svd_rank_measures_time_plot.png')
    plt.savefig(path_to_save_time_plot)

    # Plot memory vs. feature size for each measure
    plt.figure(figsize=(10, 6))

    for name in df_external_svd["Measure"].unique():
        subset = df_external_svd[df_external_svd["Measure"] == name]
        plt.plot(subset["Input_Size"], subset["Memory (bytes)"], label=f"{name} Memory", marker="o")

    plt.xlabel("Input_Size")
    plt.ylabel("Memory (bytes)")
    plt.title("Memory Requirements of Proxy Measures with external svd")
    plt.legend()
    plt.grid(True)
    path_to_save_memory_plot = os.path.join(folder_for_plots, 'comparison_of_external_svd_rank_measures_memory_plot.png')
    plt.savefig(path_to_save_memory_plot)

