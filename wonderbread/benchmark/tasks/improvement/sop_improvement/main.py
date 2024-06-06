"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

from tqdm import tqdm
from typing import Any, Dict, List
import pandas as pd
import os
import json
import argparse
from wonderbread.helpers import (
    _fetch_completion,
    build_prompt_s_a_sequence,
    add_standard_experiment_args,
    filter_prompt_s_a_sequence,
    get_folders_for_task_id,
    get_path_to_screenshots_dir,
    find_files_by_prefix_suffix,
    get_path_to_trace_json
)
from wonderbread.benchmark.tasks.improvement.sop_improvement.prompts import (
    prompt__rewrite_sop__intro,
    prompt__rewrite_sop__close,
    prompt__rewrite_sop__intro_kf,
    prompt__rewrite_sop__close_kf,
    prompt__rewrite_sop__intro_act,
    prompt__rewrite_sop__close_act
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument("--max_depth", type=int, default=1, help='Number of self reflections performed (default=1)')
    parser.add_argument("--is_td", action="store_true", default=False, help="If TRUE, then include task descriptions in prompts")
    parser.add_argument("--is_kf", action="store_true", default=False, help="If TRUE, then include screenshots into prompts")
    parser.add_argument("--is_act", action="store_true", default=False, help="If TRUE, then include action trace into prompts")
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for self reflection, one of [GPT4, GeminiPro]",
                        choices=["GPT4", "GeminiPro"] )
    return parser.parse_args()

def helper_improve_sop(prompt_s_a_sequence: Dict[str, Any], task_descrip: str, old_sop: str, model: str, is_td: bool, is_kf: bool, is_act: bool) -> str:
    '''Helper function that runs a single iteration of the SOP self-improvement'''
    if is_act and is_kf:
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__intro(task_descrip) if is_td else prompt__rewrite_sop__intro(None)
            }]
        }
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__close(old_sop)
            }]
        }
    elif is_kf:
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__intro_kf(task_descrip) if is_td else prompt__rewrite_sop__intro_kf(None)
            }]
        }
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__close_kf(old_sop)
            }]
        }
    else:
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__intro_act(task_descrip) if is_td else prompt__rewrite_sop__intro_act(None)
            }]
        }
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__rewrite_sop__close_act(old_sop)
            }]
        }

    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    new_sop: str = _fetch_completion(messages, model)
    return new_sop

def improve_sop(gt_trace: Dict[str, Any], task_descrip: str, sop: str, path_to_screenshots: str, max_depth: int,
                model: str, is_td: bool, is_kf: bool, is_act: bool, is_verbose: bool = False) -> List[str]:
    '''Calls the improve_sop function recursively n times and returns a list of strings corresponding to the updated sops'''
    prompt_s_a_sequence, _ = build_prompt_s_a_sequence(gt_trace, path_to_screenshots)
    prompt_s_a_sequence, _ = filter_prompt_s_a_sequence(prompt_s_a_sequence, is_kf, is_act)
    
    sop_list = []
    old_sop = sop

    enumerator = tqdm(range(max_depth)) if is_verbose else range(max_depth)
    for i in enumerator:
        new_sop = helper_improve_sop(prompt_s_a_sequence, task_descrip, old_sop, model, is_td, is_kf, is_act)
        sop_list.append(new_sop)
        old_sop = new_sop

    return sop_list

def run(path_to_demo_folder: str, 
        path_to_output_dir: str,
        task_id: int,
        model: str,
        max_depth: int,
        is_td: bool,
        is_kf: bool,
        is_act: bool,
        is_verbose: bool = False):
    # Create output directory and files
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load files
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    path_to_sop_file: str = os.path.join(path_to_demo_folder, find_files_by_prefix_suffix(path_to_demo_folder, '[orig] SOP', 'txt')[0])
    
    # Read files
    trace_json: Dict[str, Any] = json.load(open(path_to_trace, 'r'))
    sop: str = open(path_to_sop_file, 'r').read()

    # Execute
    gt_trace: Dict[str, str] = trace_json["trace"]
    task_id: int = trace_json['webarena']['task_id']
    task_descrip: str = trace_json['webarena']['intent']
    sop_list: List[str] = improve_sop(gt_trace, task_descrip, sop, path_to_screenshots_dir, max_depth, model, is_td, is_kf, is_act, is_verbose=is_verbose)

    # Save to CSV
    short_name: str = ('_td' if is_td else '') + ('_act' if is_act else '') + ('_kf' if is_kf else '') + f'__max_depth-{max_depth}' + f'__model-{model}'
    path_to_output_csv: str = os.path.join(path_to_output_dir, f"self_improvement__{task_id}{short_name}.csv")
    df = pd.DataFrame([{
        'demo_name' : demo_name,
        'task_id' : task_id,
        'orig_sop' : sop,
        'new_sop' : x,
        'n_self_reflection' : i+1,
        'ablation--max_self_reflection_depth' : max_depth,
        'ablation--is_td' : is_td,
        'ablation--is_act' : is_act,
        'ablation--is_kf' : is_kf,
        'ablation--model' : model,
    } for i, x in enumerate(sop_list) ])
    df.to_csv(path_to_output_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"

    # Task-specific flags
    max_depth: bool = args.max_depth
    model: str = args.model
    is_td: bool = args.is_td
    is_kf: bool = args.is_kf
    is_act: bool = args.is_act
    assert sum([is_act, is_kf]) >= 1, "Must specify AT LEAST ONE of --is_act or --is_kf"

    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(path_to_input_dir, task_id)

    # Loop through each demos's folder...
    for path_to_demo_folder in path_to_demo_folders:
        run(
            path_to_demo_folder, 
            path_to_output_dir, 
            task_id,
            model,
            max_depth,
            is_td,
            is_kf,
            is_act,
        )
