#!/bin/bash

# ==============================================================================
# Purpose:
# Run a hyperparameter sweep for train_with_improved_optimizer.py
# across an arbitrary number of GPUs concurrently.
# ==============================================================================

# Define hyperparameter arrays to sweep
replacement_rates=(0.002 0.005 0.001) # 0.01 is too high, 0.005 might be too high, 0.002 seems ideal
age_decay_rates=(0.99 0.999 0.9)
util_types=("contribution" "adaptable_contribution") # add or remove options as needed
normalization_modes=("correct by input size" "naive mse sum correction" "no correction" )
reg_lambdas=(0.01 0.001 0.1 0.0001)


# Get total number of GPUs
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)

# Array to track the PID of the job currently running on each GPU
gpu_pids=()
for ((i=0; i<num_gpus; i++)); do
    gpu_pids[$i]=""
done

# Function to get the ID of a free GPU by checking if its assigned process is still running
get_free_gpu() {
    for (( i=0; i<$num_gpus; i++ )); do
        local pid="${gpu_pids[$i]}"
        # If no PID is assigned, or the assigned PID is no longer running (kill -0 checks status)
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
            echo "$i"
            return 0
        fi
    done
    
    # Return -1 if all GPUs are currently running a process
    echo "-1"
    return 1
}

# Iterate through all combinations of hyperparameters
for rr in "${replacement_rates[@]}"; do
    for decay in "${age_decay_rates[@]}"; do
        for util in "${util_types[@]}"; do
            for norm_mode in "${normalization_modes[@]}"; do
                for lambda in "${reg_lambdas[@]}"; do
                    
                    # Loop until a free GPU is found
                    free_gpu=$(get_free_gpu)
                    while [ "$free_gpu" -eq "-1" ]; do
                        sleep 2
                        free_gpu=$(get_free_gpu)
                    done
                    
                    echo "Found free GPU: cuda:$free_gpu. Launching lambda=$lambda, decay=$decay, util=$util, norm_mode='$norm_mode', rr=$rr..."
                    
                    # Launch the job in the background
                    # Note: We wrap norm_mode in quotes because it contains spaces
                    python train_with_improved_optimizer.py \
                        learner=srr_cbp \
                        learner.SO_reg_lambda=$lambda \
                        learner.aso_age_decay_rate=$decay \
                        "learner.aso_normalization_mode=$norm_mode" \
                        learner.neurons_replacement_rate=$rr \
                        learner.util_type=$util \
                        device="cuda:$free_gpu" &
                        
                    # Record the PID of the background process we just launched to that GPU
                    gpu_pids[$free_gpu]=$!
                    
                done
            done
        done
    done
done

# Wait for all background jobs to finish
echo "All jobs launched. Waiting for them to complete..."
wait
echo "Sweep completed!"