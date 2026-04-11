#!/bin/bash

# ==============================================================================
# Purpose:
# This script runs a hyperparameter sweep for train_with_improved_optimizer.py
# across an arbitrary number of GPUs concurrently. It dynamically detects free
# GPUs by checking their memory usage using nvidia-smi. If a GPU has less than
# 500MB of memory used, it is considered free and a new job is assigned to it.
# This maximizes resource usage without duplicating jobs.
#
# Usage Instructions:
# 1. Make the script executable (already done): chmod +x sweep.sh
# 2. Run the script: ./sweep.sh
# ==============================================================================

# Define hyperparameter arrays to sweep
step_sizes=(0.01 0.005 0.001)
replacement_rates=(0.001 0.01)

# Threshold for considering a GPU "free" (in MB)
MEM_THRESHOLD=500

# Function to get the ID of a free GPU
get_free_gpu() {
    local num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    
    for (( i=0; i<$num_gpus; i++ )); do
        local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$mem_used" -lt "$MEM_THRESHOLD" ]; then
            echo "$i"
            return 0
        fi
    done
    
    # Return -1 if no GPU is free
    echo "-1"
    return 1
}

# Iterate through all combinations of hyperparameters
for lr in "${step_sizes[@]}"; do
    for rr in "${replacement_rates[@]}"; do
        
        # Loop until a free GPU is found
        free_gpu=$(get_free_gpu)
        while [ "$free_gpu" -eq "-1" ]; do
            echo "No free GPUs available. Waiting 10 seconds..."
            sleep 10
            free_gpu=$(get_free_gpu)
        done
        
        echo "Found free GPU: cuda:$free_gpu. Launching LR=$lr, RR=$rr..."
        
        # Launch the job in the background
        python train_with_improved_optimizer.py \
            learner.step_size=$lr \
            learner.neurons_replacement_rate=$rr \
            device="cuda:$free_gpu" &
            
        # Sleep for a few seconds to allow the process to allocate GPU memory
        # This prevents the next iteration from mistakenly thinking the same GPU is still free
        sleep 5
        
    done
done

# Wait for all background jobs to finish
echo "All jobs launched. Waiting for them to complete..."
wait
echo "Sweep completed!"
