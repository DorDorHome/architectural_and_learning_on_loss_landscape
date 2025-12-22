import subprocess
import random
import sys
import os

def main():
    """
    Orchestrates running the grokking experiment multiple times with different seeds.
    """
    num_runs = 10
    print(f"Starting orchestration of {num_runs} grokking experiments...")

    # Get the path to the training script relative to this orchestration script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(current_dir, "train_grokking.py")

    if not os.path.exists(train_script_path):
        print(f"Error: Training script not found at {train_script_path}")
        sys.exit(1)

    for i in range(num_runs):
        # Generate a random 32-bit integer for the seed
        seed = random.randint(0, 2**32 - 1)
        
        print(f"--- Starting Run {i+1}/{num_runs} with seed: {seed} ---")
        
        command = [
            "python",
            train_script_path,
            f"seed={seed}",
            # The config path and name are already in the decorator, so we don't need them,
            # but we can override other parameters here if needed.
        ]
        
        try:
            # We use subprocess.run and check the return code.
            # We also capture stdout/stderr and print them in case of an error.
            result = subprocess.run(
                command,
                check=True,        # Raises CalledProcessError if the command returns a non-zero exit code.
                capture_output=True, # Captures stdout and stderr.
                text=True          # Decodes stdout/stderr as text.
            )
            print(f"--- Run {i+1}/{num_runs} completed successfully. ---")
            # To avoid flooding the console, we only print stdout if needed for debugging,
            # or you can print a summary.
            # print(result.stdout) 

        except subprocess.CalledProcessError as e:
            print(f"!!! Run {i+1}/{num_runs} failed with seed: {seed} !!!")
            print(f"Return code: {e.returncode}")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            # Decide if you want to stop on failure or continue with the next run.
            # For this example, we'll stop.
            print("Orchestration stopped due to failure.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    print("Orchestration finished.")

if __name__ == "__main__":
    main()
