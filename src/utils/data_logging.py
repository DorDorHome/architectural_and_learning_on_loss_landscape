# this code contains tools for simple data logging and loading:

import json
import os




def save_data_json(data, run_dir, filename='performance_data.json'):
    """
    Saves training performance data to a JSON file within the specified run directory.

    Args:
        data (dict): Data to be saved (e.g., train and test accuracies).
        run_dir (str): Directory where the data file will be saved.
        filename (str): Name of the data file.
    """
    data_file = os.path.join(run_dir, filename)
    # with open(data_file, 'w') as f:
    #     # Convert numpy arrays to lists for JSON serialization
    #     json_data = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in data.items()}
    #     json.dump(json_data, f, indent=4)
        
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                existing_data = json.load(f)
            # Append the new log_entry to the 'logs' list
            existing_data.setdefault('logs', []).append(log_entry)
        except json.JSONDecodeError:
            # If the file is corrupted or empty, start fresh
            existing_data = {"logs": [log_entry]}
    else:
        # If the file doesn't exist, create a new structure
        existing_data = {"logs": [log_entry]}

    with open(data_file, 'w') as f:
        json.dump(existing_data, f, indent=4)