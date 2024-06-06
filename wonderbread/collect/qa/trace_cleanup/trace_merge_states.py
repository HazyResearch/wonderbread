import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from workflows.record_utils import (
    merge_consecutive_states,
)
from config import path_to_dir

for path_to_task_dir in tqdm(os.listdir(path_to_dir)):
    # check if path_to_task_dir is a directory
    if not os.path.isdir(os.path.join(path_to_dir, path_to_task_dir)):
        continue
    # Loop thru all files in directory
    for path_to_file in os.listdir(os.path.join(path_to_dir, path_to_task_dir)):
        # Find file that ends in '.json' and does not start with '[raw]'
        if path_to_file.endswith('.json') and not path_to_file.startswith('[raw]'):
            path_to_json_file: str = os.path.join(path_to_dir, path_to_task_dir, path_to_file)
            # Load JSON
            with open(path_to_json_file, 'r') as f:
                try:
                    json_data = json.load(f)
                except Exception as e:
                    print(f"Error loading {path_to_json_file}")
                    raise e
            trace = json_data['trace']
            new_trace = merge_consecutive_states(trace)
            json_data['trace'] = new_trace
            # Write JSON
            path_to_gt_file: str = os.path.join(os.path.dirname(path_to_json_file), f"{os.path.basename(path_to_json_file)}")
            with open(path_to_gt_file, 'w') as f:
                json.dump(json_data, f, indent=2)