"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

import random
import traceback
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import os
import argparse
import json
from wonderbread.helpers import (
    _fetch_completion,
    build_prompt_s_a_sequence,
    add_standard_experiment_args,
    filter_prompt_s_a_sequence,
    get_folders_for_task_id,
    get_path_to_screenshots_dir,
    get_path_to_sop_txt,
    get_path_to_trace_json,
)
from wonderbread.benchmark.tasks.helpers import string_to_random_int
from wonderbread.benchmark.tasks.knowledge_transfer.demo_validation.prompts import (
    prompt__validate_task_completion__intro,
    prompt__validate_task_completion__close,
    prompt__validate_task_trajectory__intro,
    prompt__validate_task_trajectory__close,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument("--version", type=str, required=True, help="Either 'task_completion' or 'task_trajectory'")
    parser.add_argument("--is_include_sop", action="store_true", help="If TRUE, include SOP.txt in model input")
    parser.add_argument("--n_negative_samples", type=int, default=2, help="# of negative examples")
    parser.add_argument( "--is_td", action="store_true", default=False, help="If TRUE, then include task description into prompts",)
    parser.add_argument( "--is_kf", action="store_true", default=False, help="If TRUE, then include screenshots as key frames into prompts", )
    parser.add_argument( "--is_act", action="store_true", default=False, help="If TRUE, then include action traces into prompts", )
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for self monitoring, one of [GPT4, GeminiPro]",
                        choices=["GPT4", "GeminiPro"] )
    parser.add_argument("--is_verbose", action="store_true", help="If TRUE, then print out stuff")
    return parser.parse_args()

def helper_task_completion(gt_trace: Dict[str, Any], task_descrip: str, model: str, sop: str, gt_is_met: bool, path_to_screenshots: str, is_td: bool, is_kf: bool, is_act: bool, task_type: str) -> Dict[str, str]:
    """Helper fx to eval a single POSITIVE or NEGATIVE example."""
    prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(gt_trace, path_to_screenshots)
    prompt_s_a_sequence, paths_to_screenshots = filter_prompt_s_a_sequence(prompt_s_a_sequence, is_kf, is_act, paths_to_screenshots)
    if not is_td:
        task_descrip = None

    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_completion__intro(task_descrip, sop)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_completion__close()
        }]
    }
    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    pred_raw_response: str = _fetch_completion(messages, model)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['thinking']
        pred_is_met: bool = pred_json['was_completed']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # gt
        "gt_is_met" : gt_is_met,
        "paths_to_screenshots" : paths_to_screenshots,
        # preds
        "pred_rationale": pred_rationale,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
        "task_type": task_type,
        "model": model,
    }

def validate_task_completion(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model: str,
    sop: Optional[str],
    config: Dict[str, Any],
    is_td: bool,
    is_kf: bool,
    is_act: bool,
    is_verbose: bool = False,
) -> pd.DataFrame:
    """
        Evaluate overall task completion success.
        
        If S_1, S_2, ..., S_n are the sequence of screenshots, then...
        
            For "task completed":
                A "TRUE" example is sequence of screenshots of the form [ S_1, ..., S_n ]
                A "FALSE" example is any sequence terminated before S_n, so [S_1, ..., S_j < S_n]
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']
    task_descrip: str = gt_task_data['webarena']['intent']
    results: List[Dict[str, Any]] = []

    # Eval "TRUE" example
    results.append(helper_task_completion(gt_trace, task_descrip, model, sop, True, path_to_screenshots, is_td, is_kf, is_act, task_type='true'))

    # Eval "FALSE" examples
    for end_state_id in config['truncate']['state_ids']:
        gt_trace_negative: List[Dict[str, Any]] = [ x for x in gt_trace if x['data']['id'] < end_state_id - 1 ] # -1 to chop off preceding action as well
        results.append(helper_task_completion(gt_trace_negative, task_descrip, model, sop, False, path_to_screenshots, is_td, is_kf, is_act, task_type='truncate'))

    df = pd.DataFrame(results)
    return df


def helper_task_trajectory(gt_trace: Dict[str, Any], task_descrip, model: str, sop: str, gt_is_met: bool, path_to_screenshots: str, is_td: bool, is_kf: bool, is_act: bool, task_type: str) -> Dict[str, str]:
    prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(gt_trace, path_to_screenshots)
    prompt_s_a_sequence, paths_to_screenshots = filter_prompt_s_a_sequence(prompt_s_a_sequence, is_kf, is_act, paths_to_screenshots)
    if not is_td:
        task_descrip = None
    
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_trajectory__intro(task_descrip)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_trajectory__close(sop)
        }]
    }
    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    pred_raw_response: str = _fetch_completion(messages, model)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['thinking']
        pred_inaccurate_steps: List[str] = pred_json.get('inaccurate_steps', [])
        pred_is_met: bool = pred_json['was_accurate'] if 'was_accurate' in pred_json else pred_json['was_acurate']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_inaccurate_steps = None
        pred_is_met = None
        is_correct = False

    return {
        # gt
        "gt_is_met" : gt_is_met,
        "paths_to_screenshots" : paths_to_screenshots,
        # preds
        "pred_rationale": pred_rationale,
        "pred_inaccurate_steps": pred_inaccurate_steps,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
        "task_type": task_type,
        "model": model,
    }


def validate_task_trajectory_valid(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model: str,
    sop: str,
    config: Dict[str, Any],
    is_td: bool,
    is_kf: bool,
    is_act: bool,
    is_verbose: bool = False,
) -> pd.DataFrame:
    """
        Evaluate overall task completion success.
        
        If S_1, S_2, ..., S_n are the sequence of screenshots, then...
            For "trajectory valid":
                A "TRUE" example is [ S_1, ..., S_n ]
                A "FALSE" example is anything else, e.g. shuffling the order of screenshots, skipping screenshots, etc.
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']
    task_descrip: str = gt_task_data['webarena']['intent']
    results: List[Dict[str, Any]] = []

    # Eval "TRUE" example
    results.append(helper_task_trajectory(gt_trace, task_descrip, model, sop, True, path_to_screenshots, is_td, is_kf, is_act, task_type='true'))

    # Eval "FALSE" examples
    ## Skip 2 screenshots at a random interval
    for (skip_state_ids, skip_action_ids) in zip(config['skip']['state_ids'], config['skip']['action_ids']):
        gt_trace_negative: List[Dict[str, Any]] = [ x for x in gt_trace if x['data']['id'] not in skip_state_ids  and x['data']['id'] not in skip_action_ids ]
        results.append(helper_task_trajectory(gt_trace_negative, task_descrip, model, sop, False, path_to_screenshots, is_td, is_kf, is_act, task_type='skip'))
    ## Shuffle 2 random screenshots
    for (shuffle_id_1, shuffle_id_2) in config['shuffle']['state_ids']:
        gt_trace_negative: List[Dict[str, Any]] = []
        for idx, x in enumerate(gt_trace):
            if x['data']['id'] in [ shuffle_id_1+1, shuffle_id_2+1 ]:
                # skip actions after shuffled states
                continue
            elif x['data']['id'] == shuffle_id_1:
                new_state_idx: Dict[str, Any] = [ idx2 for idx2, x2 in enumerate(gt_trace) if x2['data']['id'] == shuffle_id_2][0]
                gt_trace_negative.append(gt_trace[new_state_idx]) # add state
                gt_trace_negative.append(gt_trace[new_state_idx+1]) # add action
            elif x['data']['id'] == shuffle_id_2:
                new_state_idx: Dict[str, Any] = [ idx2 for idx2, x2 in enumerate(gt_trace) if x2['data']['id'] == shuffle_id_1][0]
                gt_trace_negative.append(gt_trace[new_state_idx]) # add state
                gt_trace_negative.append(gt_trace[new_state_idx+1]) # add action
            else:
                gt_trace_negative.append(x)
        assert len(gt_trace) == len(gt_trace_negative), f"Length of `gt_trace` ({len(gt_trace)}) != length of `gt_trace_negative` ({len(gt_trace_negative)})"
        results.append(helper_task_trajectory(gt_trace_negative, task_descrip, model, sop, False, path_to_screenshots, is_td, is_kf, is_act, task_type='shuffle'))

    df = pd.DataFrame(results)
    return df

def kwarg_setting_to_ablation(kwarg_setting: Dict[str, Any]) -> str:
    # Parse kwargs
    version: bool = kwarg_setting['version']
    is_include_sop: bool = kwarg_setting['is_include_sop']
    n_negative_samples: int = kwarg_setting['n_negative_samples']
    is_td: bool = kwarg_setting['is_td']
    is_kf: bool = kwarg_setting['is_kf']
    is_act: bool = kwarg_setting['is_act']
    model: str = kwarg_setting['model']
    # Generate ablation string
    short_name: str = model + '_' + version
    if is_include_sop:
        short_name += "_sop"
    if is_td:
        short_name += "_td"
    if is_kf:
        short_name += "_kf"
    if is_act:
        short_name += "_act"
    short_name += f"_samples={n_negative_samples}"
    return short_name

def run(path_to_demo_folder: str, 
        path_to_output_dir: str,
        task_id: int,
        model: str,
        version: str,
        is_include_sop: bool,
        is_td: bool,
        is_kf: bool,
        is_act: bool,
        n_negative_samples: int,
        is_verbose: bool = False):
    # Demo name
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])

    # Load ablation configs (for reproducibility across negative sampling)
    path_to_config_folder: str = os.path.abspath(os.path.join(path_to_output_dir, '../', 'demo_2_config'))
    demo_2_config: Dict[str, Dict] = get_demo_config(path_to_demo_folder, path_to_config_folder, n_negative_samples)
    config: Dict[str, Any] = demo_2_config[demo_name][str(n_negative_samples)]
    
    # Create output directory
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load files
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    path_to_sop_file: str = get_path_to_sop_txt(path_to_demo_folder)

    # Read files
    trace_json: Dict[str, Any] = json.load(open(path_to_trace, 'r'))
    sop: str = open(path_to_sop_file, 'r').read() if is_include_sop else None
    
    # Execute eval
    try:
        if version == 'task_completion':
            if config['truncate']['state_ids'] is None:
                # Ignore any demonstrations that have fewer states/actions than negative samples require
                print(f"Skipping task: {task_id} | Reason: config['truncate']['state_ids'] is None")
                return None
            if is_verbose: print(f"Running validate_task_completion() for: task: {task_id} | model: {model}")
            df = validate_task_completion(
                trace_json, path_to_screenshots_dir, model, sop, config,
                is_td=is_td, is_kf=is_kf, is_act=is_act,
                is_verbose=is_verbose,
            )
        elif version == 'task_trajectory':
            if config['skip']['state_ids'] is None:
                # Ignore any demonstrations that have fewer states/actions than negative samples require
                print(f"Skipping task: {task_id} | Reason: config['skip']['state_ids'] is None")
                return None
            if is_verbose: print(f"Running validate_task_trajectory_valid() for: task: {task_id} | model: {model}")
            df = validate_task_trajectory_valid(
                trace_json, path_to_screenshots_dir, model, sop, config,
                is_td=is_td, is_kf=is_kf, is_act=is_act,
                is_verbose=is_verbose,
            )
        else:
            raise ValueError(f"Unrecognized version: {version}")
    except Exception as e:
        print(f"Error with demo folder: {path_to_demo_folder} | task_id: {task_id}")
        print(traceback.format_exc())
        print(str(e))
        raise e
    
    # Save results
    ablation = kwarg_setting_to_ablation({
        "is_include_sop" : is_include_sop,
        "is_td" : is_td,
        "is_kf" : is_kf,
        "is_act" : is_act,
        "version" : version,
        "n_negative_samples" : n_negative_samples,
        "model" : model,
    })
    df['demo_name'] = demo_name
    df['task_id'] = task_id
    df['ablation--is_td'] = is_td
    df['ablation--is_kf'] = is_kf
    df['ablation--is_act'] = is_act
    df['ablation--is_include_sop'] = is_include_sop
    df['ablation--version'] = version
    df['ablation--n_negative_samples'] = n_negative_samples
    df['ablation--model'] = model

    # Print metrics
    accuracy: float = df['is_correct'].mean() if 'is_correct' in df.columns else 'N/A'
    all_correct: bool = df['is_correct'].all() if 'is_correct' in df.columns else 'N/A'
    if is_verbose: 
        print(f"Task: {task_id}")
        print(f"Accuracy: {accuracy}")
        print(f"All correct? {all_correct}")
    df.to_csv(os.path.join(path_to_output_dir, f"self_monitoring__{ablation}.csv"), index=False)

def get_demo_config(path_to_demo_folder: str, path_to_config_folder: str, n_negative_samples: int) -> Dict:
    """Returns `demo_2_config`, which contains negatives for each task. 
        First, checks if the config exists. If it does, then loads it.
        If not, then creates it and saves to JSON file."""
    # Get demo name from path
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])
    
    # Load .json config file
    path_to_demo_2_config: str = os.path.join(path_to_config_folder, f"{demo_name}.json")
    os.makedirs(os.path.dirname(path_to_demo_2_config), exist_ok=True)
    demo_2_config: Dict[str, Dict] = json.load(open(path_to_demo_2_config, 'r')) if os.path.exists(path_to_demo_2_config) else {}
    
    # Skip if already processed
    if demo_name in demo_2_config and str(n_negative_samples) in demo_2_config[demo_name]:
        return demo_2_config

    # Load trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    trace_json: Dict[str, Any] = json.load(open(path_to_trace, 'r'))
    
    # Get states in trace
    gt_trace = trace_json['trace']
    states = [ x for x in gt_trace if x['type'] == 'state' ]
    n_states: int = len(states)

    ## TASK: COMPLETION
    # State idxs for ending truncation
    random.seed(string_to_random_int(demo_name + '_truncation'))
    random_trunc_idxs: List[int] = list(range(1, n_states - 1))
    random.shuffle(random_trunc_idxs) # randomize this way to prevent duplicates
    if len(random_trunc_idxs) < n_negative_samples:
        # If there are fewer than `n_negative_samples` possible shuffles, then return None
        truncation_state_ids = None
    else:
        truncation_state_ids: List[int] = [ states[random_trunc_idxs[x]]['data']['id'] for x in range(n_negative_samples) ]

    ## TASK: TRAJECTORY
    ## Skip `skip_len` screenshots at a random interval
    random.seed(string_to_random_int(demo_name + '_skip'))
    skip_len: int = 2
    random_skip_idxs: List[int] = list(range(0, n_states - skip_len))
    random.shuffle(random_skip_idxs) # randomize this way to prevent duplicates
    if len(random_skip_idxs) < n_negative_samples:
        # If there are fewer than `n_negative_samples` possible skips, then return None
        skip_state_ids = None
        skip_action_ids = None
    else:
        skip_state_ids: List[List[int]] = [ 
            tuple([ states[random_skip_idxs[x] + i]['data']['id'] for i in range(skip_len) ])
            for x in range(n_negative_samples)
        ]
        skip_action_ids: List[List[int]] = [
            tuple([ x-1 for x in skip_state_id ]) # skip actions immediately preceding skipped states
            for skip_state_id in skip_state_ids
        ]

    ## Shuffle 2 random screenshots
    random.seed(string_to_random_int(demo_name + '_shuffle'))
    random_shuffle_idxs: List[int] = list(range(0, max(0, n_states - 2))) # -2 so we ignore the last state (don't shuffle b/c no subsequent action)
    random.shuffle(random_shuffle_idxs) # randomize this way to prevent duplicates
    if len(random_shuffle_idxs) < n_negative_samples:
        # If there are fewer than `n_negative_samples` possible shuffles, then return None
        shuffle_state_ids = None
    else:
        shuffle_state_ids: List[Tuple[int]] = []
        for x in range(n_negative_samples):
            shuffle_id_1: int = states[random_shuffle_idxs[x]]['data']['id']
            shuffle_id_2: int = random.choice([ y['data']['id'] for y in states[:-1] if y['data']['id'] != shuffle_id_1 ])
            shuffle_state_ids.append((shuffle_id_1, shuffle_id_2))

    # Sanity checks
    if truncation_state_ids is not None:
        assert len(truncation_state_ids) == n_negative_samples, f"Expected `len(truncation_state_ids)` == {n_negative_samples}, got {len(truncation_state_ids)}"
        assert len(truncation_state_ids) == len(set(truncation_state_ids)), f"Expected `truncation_state_ids` to be unique, got {truncation_state_ids}"
    if skip_state_ids is not None:
        assert len(skip_state_ids) == n_negative_samples, f"Expected `len(skip_state_ids)` == {n_negative_samples}, got {len(skip_state_ids)}"
        assert len(skip_state_ids) == len(set(skip_state_ids)), f"Expected `skip_state_ids` to be unique, got {skip_state_ids}"
    if skip_action_ids is not None:
        assert len(skip_action_ids) == n_negative_samples, f"Expected `len(skip_action_ids)` == {n_negative_samples}, got {len(skip_action_ids)}"
        assert len(skip_action_ids) == len(set(skip_action_ids)), f"Expected `skip_action_ids` to be unique, got {skip_action_ids}"
    if shuffle_state_ids is not None:
        assert len(shuffle_state_ids) == n_negative_samples, f"Expected `len(shuffle_state_ids)` == {n_negative_samples}, got {len(shuffle_state_ids)}"
        assert len(shuffle_state_ids) == len(set(shuffle_state_ids)), f"Expected `shuffle_state_ids` to be unique, got {shuffle_state_ids}"

    # Save results
    if demo_name not in demo_2_config:
        demo_2_config[demo_name] = {}
    if str(n_negative_samples) not in demo_2_config[demo_name]:
        demo_2_config[demo_name][str(n_negative_samples)] = {
            'skip' : {
                'state_ids' : skip_state_ids, # skip all states with these IDs
                'action_ids' : skip_action_ids, # skip all actions with these IDs
            },
            'shuffle' : {
                'state_ids' : shuffle_state_ids, # tuple of (state_id_1, state_id_2) to shuffle
            },
            'truncate' : {
                'state_ids' : truncation_state_ids, # skip all states after this state
            }
        }
    json.dump(demo_2_config, open(path_to_demo_2_config, 'w'), indent=2)
    
    return demo_2_config

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    is_verbose: bool = args.is_verbose
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"

    # Task-specific flags
    is_include_sop: bool = args.is_include_sop
    is_td: bool = args.is_td
    is_kf: bool = args.is_kf
    is_act: bool = args.is_act
    version: bool = args.version
    n_negative_samples: int = args.n_negative_samples
    model: str = args.model
    
    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(path_to_input_dir, task_id)   
            
    # Loop through each demos's folder...
    for path_to_demo_folder in path_to_demo_folders:
        run(
            path_to_demo_folder, 
            path_to_output_dir, 
            task_id,
            model,
            version,
            is_td,
            is_kf,
            is_act,
            is_include_sop,
            n_negative_samples,
            is_verbose
        )