"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

import os
import pandas as pd
from wonderbread.helpers import (
    add_standard_experiment_args,
    get_folders_for_task_id,
    get_path_to_screenshots_dir,
    get_path_to_trace_json,
    get_webarena_task_json,
)
from typing import Dict, List, Optional
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument( "--is_ablation_1", action="store_true", default=False, help="If TRUE, then do Ablation #1" )
    parser.add_argument( "--is_ablation_2", action="store_true", default=False, help="If TRUE, then do Ablation #2" )
    return parser.parse_args()

def run_single_demo(path_to_demo_folder: str, 
        path_to_output_dir: str,
        task_id: int,
        is_ablation_1: bool,
        is_ablation_2: bool):
    """Example experiment for considering a single demo at a time."""
    # Create output directory
    demo_name: str = os.path.basename(path_to_demo_folder.strip("/"))
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Get WebArena task description
    task_json: Optional[Dict[str, str]] = get_webarena_task_json(task_id)
    assert task_json is not None, f"Could not find WebArena task json for {task_id}"
    
    # Get .json trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    
    results = [
        { 'name': 'exp1', 'is_correct': True, 'note': '' },
        { 'name': 'exp2', 'is_correct': False, 'note': '' },
    ]
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path_to_output_dir, "results.csv"), index=False)

def run_mutiple_demos(path_to_demo_folders: List[str], 
                        path_to_output_dir: str,
                        task_id: int,
                        is_ablation_1: bool,
                        is_ablation_2: bool):
    """Example experiment for considering all demos for a given task."""
    # Create output directory
    demo_name: str = os.path.basename(path_to_demo_folder.strip("/"))
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Get WebArena task description
    task_json: Optional[Dict[str, str]] = get_webarena_task_json(task_id)
    assert task_json is not None, f"Could not find WebArena task json for {task_id}"
    
    results = [
        { 'name': 'demo1', 'is_correct': True, 'note': '' },
        { 'name': 'demo2', 'is_correct': False, 'note': '' },
    ]
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path_to_output_dir, "results.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"
    
    # Task-specific flags
    is_ablation_1: bool = args.is_ablation_1
    is_ablation_2: bool = args.is_ablation_2
    assert is_ablation_1 + is_ablation_2 == 1, "Must specify exactly one of --is_ablation_1, --is_ablation_2"

    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(path_to_input_dir, task_id)

    # Loop through each demos's folder...
    for path_to_demo_folder in path_to_demo_folders:
        run_single_demo(
            path_to_demo_folder, 
            path_to_output_dir, 
            task_id,
            is_ablation_1,
            is_ablation_2
        )
    
    # Or consider all demos simultaneously
    run_mutiple_demos(
        path_to_demo_folders, 
        path_to_output_dir, 
        task_id,
        is_ablation_1,
        is_ablation_2
    )