"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

import collections
import random
import dirtyjson
import argparse
import json
from typing import Any, Dict, List
import pandas as pd
import os
import sklearn.metrics.cluster
from wonderbread.helpers import (
    _fetch_completion,
    build_prompt_s_a_sequence,
    add_standard_experiment_args,
    get_path_to_screenshots_dir,
    get_path_to_sop_txt,
    get_path_to_trace_json,
)
from wonderbread.benchmark.tasks.documentation.demo_segmentation.prompts import (
    prompt__intro,
    prompt__close_uuid,
    prompt__close_start_end,
)

ACTION_TRANSITION: str = f"Action: PRESS(Cmd+Tab)"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    # Dataset creation
    parser.add_argument("--is_interleave", action="store_true", help="If TRUE, then create examples by interleaving traces (easy)")
    parser.add_argument("--is_concatenate", action="store_true", help="If TRUE, then create examples by concatenating traces (hard)")
    parser.add_argument("--n_tasks", type=int, default=3, help="Number of tasks to merge")
    parser.add_argument("--is_same_site", action="store_true", help="If TRUE, then sample all tasks from same site")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials to run")
    # Prompting strats
    parser.add_argument("--is_act", action="store_true", default=False, help="If TRUE, then include action trace into prompts",)
    parser.add_argument("--is_kf", action="store_true", default=False, help="If TRUE, then include screenshots into prompts",)
    parser.add_argument("--is_td", action="store_true", default=False, help="If TRUE, then include task descriptions in prompts" )
    parser.add_argument("--is_include_sop", action="store_true", help="If TRUE, include SOP.txt in model input")
    parser.add_argument("--is_prompt_uuid", action="store_true", help="If TRUE, then use UUID frame-level classification prompt. Otherwise, use start/end prompt")
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for self monitoring, one of [GPT4, GeminiPro]",
                        choices=["GPT4", "GeminiPro"] )
    return parser.parse_args()

def create_merged_trace(path_to_demo_folders: List[str], 
                        is_interleave: bool, 
                        is_concatenate: bool, 
                        is_keep_act: bool = True, 
                        is_keep_kfs: bool = True,
                        random_seed: int = 1) -> List[Dict[str, Any]]:
    # Load demos
    logs: Dict[List[Dict[str, Any]]] = collections.defaultdict(list)
    for demo in path_to_demo_folders:
        # Load files
        path_to_trace: str = get_path_to_trace_json(demo)
        path_to_screenshots_dir: str = get_path_to_screenshots_dir(demo)
        # Read files
        trace_json: Dict[str, Any] = json.load(open(path_to_trace, 'r'))
        task_id: int = int(trace_json['webarena']['task_id'])
        prompt_s_a_sequence, __ =  build_prompt_s_a_sequence(trace_json['trace'], path_to_screenshots_dir)
        for item_idx, item in enumerate(prompt_s_a_sequence):
            logs[task_id].append({
                'task_id': task_id,
                'item_idx' : item_idx,
                'item' : { 'role' : 'user', 'content' : item['content'] },
                'item_type' : 'action' if item['content'][0]['type'] == 'text' else 'state'
            })
        # Add a final "transition" action between non-final demos
        if demo != path_to_demo_folders[-1]:
            logs[task_id].append({
                'task_id': task_id,
                'item_idx': item_idx + 1,
                'item' : {
                    'role' : 'user',
                    'content': [{
                        'type': 'text',
                        'text': ACTION_TRANSITION,
                    }]
                },
                'item_type' : 'action'
            })
    
    # Create merged log
    random.seed(random_seed)
    merged_log: List[Dict[str, Any]] = []
    if is_concatenate:
        # Concatenate consecutively
        keys: List[int] = sorted(list(logs.keys()))
        random.shuffle(keys)
        for key in keys:
            merged_log.extend(logs[key])
    elif is_interleave:
        # Interleave
        curr_idxs: Dict[int, int] = { task_id: 0 for task_id in logs.keys() }
        non_empty_tasks: List[int] = [ task_id for task_id, log in logs.items() if len(log) > 0 ]
        while len(non_empty_tasks) > 0:
            # Randomly choose a task to add
            next_task_id: int = random.choice(non_empty_tasks)
            # Add next state + action from that task into merged log
            merged_log.append(logs[next_task_id][curr_idxs[next_task_id]])
            merged_log.append(logs[next_task_id][curr_idxs[next_task_id + 1]])
            # Move ptr to next item in that task
            curr_idxs[next_task_id] += 2
            # Remove task from non_empty_tasks if it's empty
            if curr_idxs[next_task_id] >= len(logs[next_task_id]):
                non_empty_tasks.remove(next_task_id)
    
    # Filtering
    if not is_keep_act:
        # Remove actions
        merged_log = [ x for x in merged_log if x['item_type'] != 'action' ]
    if not is_keep_kfs:
        # Remove key frames
        merged_log = [ x for x in merged_log if x['item_type'] != 'state' ]
    
    # Add UUID to each item
    for idx, item in enumerate(merged_log):
        item['uuid'] = idx

    return merged_log

def kwarg_setting_to_ablation(kwarg_setting: Dict[str, Any]) -> str:
    # Parse kwargs
    model: str = kwarg_setting['model']
    is_td: bool = kwarg_setting['is_td']
    is_kf: bool = kwarg_setting['is_kf']
    is_act: bool = kwarg_setting['is_act']
    is_include_sop: bool = kwarg_setting['is_include_sop']
    is_interleave: bool = kwarg_setting['is_interleave']
    is_concatenate: bool = kwarg_setting['is_concatenate']
    n_tasks: int = kwarg_setting['n_tasks']
    is_same_site: bool = kwarg_setting['is_same_site']
    is_prompt_uuid: bool = kwarg_setting['is_prompt_uuid']
    n_trials: int = kwarg_setting['n_trials']
    output_file_name: str = []
    if is_td:
        output_file_name += ["td"]
    if is_kf:
        output_file_name += ["kf"]
    if is_act:
        output_file_name += ["act"]
    if is_include_sop:
        output_file_name += ["sop"]
    if is_interleave:
        output_file_name += ["interleave"]
    if is_concatenate:
        output_file_name += ["concatenate"]
    output_file_name = "_".join(output_file_name)
    output_file_name += f'_n_tasks-{n_tasks}'
    output_file_name += f'_same_site-{is_same_site}'
    output_file_name += f'_prompt_uuid-{is_prompt_uuid}'
    output_file_name += f'_n_trials-{n_trials}'
    output_file_name += f'_{model}'
    return output_file_name

def run(path_to_demo_folder: str, 
        path_to_output_dir: str, 
        task_id: int,
        model: str,
        n_trials: int,
        is_interleave: bool,
        is_same_site: bool,
        is_concatenate: bool,
        is_prompt_uuid: bool,
        n_tasks: int,
        is_td: bool,
        is_kf: bool,
        is_act: bool,
        is_include_sop: bool,
        is_verbose: bool = True):
    # Demo name
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])
    
    # Load ablation configs (for reproducibility across negative sampling)
    path_to_config_folder: str = os.path.abspath(os.path.join(path_to_output_dir, '../', 'demo_2_config'))
    demo_2_config: Dict[str, Dict] = get_demo_config(path_to_demo_folder, path_to_config_folder, n_tasks, is_same_site)
    config: Dict[str, Any] = demo_2_config[demo_name][str(n_tasks)][f'is_same_site={is_same_site}']
    path_to_demo_folders: List[str] = [ 
        os.path.join(os.path.abspath(os.path.join(path_to_demo_folder, '../')), x) 
        for x in config['concatenate']['demo_names'] 
    ]

    # Create output directory
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    intents: List[str] = []
    task_ids: List[int] = []
    for path_to_demo_folder in path_to_demo_folders:
        trace_json: str = json.loads(open(get_path_to_trace_json(path_to_demo_folder), 'r').read())
        intents.append(trace_json['webarena']['intent'])
        task_ids.append(int(trace_json['webarena']['task_id']))

    # Provide task descriptions if `--is_td` is TRUE
    if is_td:
        task_descriptions: List[str] = [ f"({chr(idx + 65)}) {intent}" for idx, intent in enumerate(intents) ]
    else:
        task_descriptions: List[str] = [ f"({chr(idx + 65)}) Some unspecified workflow {chr(idx + 65)}" for idx, intent in enumerate(intents) ]

    # Provide SOPs if `--is_include_sop` is TRUE
    if is_include_sop:
        sops: str = "Below, we've also provided the step-by-step guides that each workflow was supposed to follow.\n" + \
            "\n".join([ f"#### Workflow ({chr(idx + 65)}) {open(get_path_to_sop_txt(path_to_demo_folder), 'r').read()}" for idx, path_to_demo_folder in enumerate(path_to_demo_folders) ])
    else:
        sops: str = None

    # Create merged trace
    results: List[Dict[str, Any]] = []
    for i in range(n_trials):
        merged_trace: List[Dict[str, Any]] = create_merged_trace(path_to_demo_folders, 
                                                                is_interleave, 
                                                                is_concatenate, 
                                                                is_keep_kfs=is_kf,
                                                                is_keep_act=is_act,
                                                                random_seed=i)
        
        # Add an "UUID" to each item in the merged trace for each prompt
        # Ask model to predict task ID for each UUID
        for idx, item in enumerate(merged_trace):
            if merged_trace[idx]['item_type'] == 'action':
                merged_trace[idx]['item']['content'][0]['text'] = f"UUID of next action: {item['uuid']}\n" + item['item']['content'][0]['text']
            else:
                merged_trace[idx]['item']['content'] = [{
                    'type' : 'text',
                    'text' : f"UUID of next screenshot: {item['uuid']}\n"
                }] + merged_trace[idx]['item']['content']
        
        # Execute eval
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__intro(len(task_descriptions), "\n".join(task_descriptions))
            }]
        }
        merged_prompts: List[Dict[str, str]] = [
            x['item']
            for x in merged_trace
        ]
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__close_uuid(len(task_descriptions), "\n".join(task_descriptions), sops) if is_prompt_uuid else prompt__close_start_end(len(task_descriptions), "\n".join(task_descriptions), sops)
            }]
        }
        messages: List[str] = [intro_prompt] + merged_prompts + [close_prompt]
        pred_raw_response: str = _fetch_completion(messages, model)
        
        # Logging
        if is_verbose:
            print(f"===== Trial {i+1} =====")
            print(pred_raw_response)
            print("---- True ordering ----")
            for item_idx, item in enumerate(merged_trace):
                if item_idx == 0:
                    print('start', f"uuid={item['uuid']}", f"task_id={item['task_id']}")
                elif item['task_id'] != merged_trace[item_idx-1]['task_id']:
                    print('end', f"uuid={item['uuid']}", f"task_id={merged_trace[item_idx-1]['task_id']}")
                    print('start', f"uuid={item['uuid']}", f"task_id={item['task_id']}")
                elif item_idx == len(merged_trace) - 1:
                    print('end', f"uuid={item['uuid']}", f"task_id={item['task_id']}")

        # Evaluate
        try:
            # Parse response
            if '```json' in pred_raw_response:
                pred_raw_response = pred_raw_response.split('```json')[1]
            if '```' in pred_raw_response:
                pred_raw_response = pred_raw_response.split('```')[0]
            pred_json = dirtyjson.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
            if is_prompt_uuid:
                # Format: { "UUID_1": "A", "UUID_2": "B", ... }
                pred_uuid_2_task_id: Dict[str, int] = pred_json
            else:
                # Format: { "A" : { "start": UUID_1, "end": UUID_2 }, "B" : { "start": UUID_3, "end": UUID_4 }, ...
                pred_uuid_2_task_id: Dict[str, int] = {}
                for key, val in pred_json.items():
                    start, end = val.get('start'), val.get('end')
                    if None not in [start, end]:
                        for uuid in range(int(start), int(end) + 1):
                            pred_uuid_2_task_id[str(uuid)] = key
            # Record results
            for item in merged_trace:
                if item['item_type'] == 'action' and item['item']['content'][0]['text'] == ACTION_TRANSITION:
                    # Ignore transitions
                    continue
                uuid: str = str(item['uuid']) # need to convert to string
                results.append({
                    'trial': i,
                    'uuid': uuid,
                    'pred_task_id': pred_uuid_2_task_id.get(uuid),
                    'gt_task_id': str(item['task_id']),
                    'item_type': item['item_type'],
                })
        except Exception as e:
            print(f"Error in trial {i}: {e}")
            for item in merged_trace:
                if item['item_type'] == 'action' and item['item']['content'][0]['text'] == ACTION_TRANSITION:
                    # Ignore transitions
                    continue
                results.append({
                    'trial': i,
                    'uuid': item['uuid'],
                    'pred_task_id': None,
                    'gt_task_id': str(item['task_id']),
                    'item_type': item['item_type'],
                })
    df = pd.DataFrame(results)

    # Calculate metrics
    # If the explicit task is provided for each workflow class label, then enforce that workflow labels exactly match task description.
    # Otherwise, if we just have a set of workflow labels, then we can report clustering metrics
    if is_td:
        ## Convert "A" -> 104, "B" -> 743, etc.
        df['pred_task_id'] = df['pred_task_id'].apply(lambda x: str(task_ids[ord(x) - 65]) if x is not None and ord(x) - 65 < len(task_ids) else None)
        df['is_correct'] = df['pred_task_id'] == df['gt_task_id']
    
    # Calculate metrics
    accuracies, v_measures, adj_rand_scores = [], [], []
    for i in range(n_trials):
        df_ = df[df['trial'] == i].copy()
        # Accuracy
        accuracies.append(df_['is_correct'].mean() if 'is_correct' in df_.columns else None)
        # Clustering
        ## Note: We map Nones -> dummy class 'Z' for clustering metrics
        v_measures.append(sklearn.metrics.cluster.v_measure_score(df_['gt_task_id'], df_['pred_task_id'].fillna('Z')))
        adj_rand_scores.append(sklearn.metrics.cluster.adjusted_rand_score(df_['gt_task_id'], df_['pred_task_id'].fillna('Z')))
    parse_error_freq: int = df['pred_task_id'].isna().mean()
    
    # Ablation
    ablation = kwarg_setting_to_ablation({
        'model' : model,
        'n_tasks' : n_tasks,
        'is_same_site' : is_same_site,
        'is_interleave' : is_interleave,
        'is_concatenate' : is_concatenate,
        'is_td' : is_td,
        'is_kf' : is_kf,
        'is_act' : is_act,
        'is_include_sop' : is_include_sop,
        'is_prompt_uuid' : is_prompt_uuid,
        'n_trials' : n_trials,
    })
    df['demo_name'] = demo_name
    df['task_id'] = task_id
    df['demos'] = [ [ os.path.basename(x) for x in path_to_demo_folders ] ] * len(df)
    df['ablation--model'] = model
    df['ablation--n_tasks'] = n_tasks
    df['ablation--is_same_site'] = is_same_site
    df['ablation--is_interleave'] = is_interleave
    df['ablation--is_concatenate'] = is_concatenate
    df['ablation--is_td'] = is_td
    df['ablation--is_kf'] = is_kf
    df['ablation--is_act'] = is_act
    df['ablation--is_include_sop'] = is_include_sop
    df['ablation--is_prompt_uuid'] = is_prompt_uuid
    df['ablation--n_trials'] = n_trials
    
    # Print metrics
    if is_verbose:
        print(f"Accuracy: {sum(accuracies) / len(accuracies) if is_td else None}")
        print(f"v_measure: {sum(v_measures) / len(v_measures)}")
        print(f"adj_rand_score: {sum(adj_rand_scores) / len(adj_rand_scores)}")
        print(f"Parse error freq: {parse_error_freq}")
    df.to_csv(os.path.join(path_to_output_dir, f"task_segmentation__{ablation}.csv"), index=False)

def get_demo_config(path_to_demo_folder: str, path_to_config_folder: str, n_tasks: int, is_same_site: bool) -> Dict:
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
    if (
        demo_name in demo_2_config 
        and str(n_tasks) in demo_2_config[demo_name] 
        and f'is_same_site={is_same_site}' in demo_2_config[demo_name][str(n_tasks)]
    ):
        return demo_2_config

    # Load trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    trace_json: Dict[str, Any] = json.load(open(path_to_trace, 'r'))
    anchor_site: str = trace_json['webarena']['sites'][0]
    
    # Get a list of all other task IDs in the input directory
    path_to_input_dir: str = os.path.abspath(os.path.join(path_to_demo_folder, '../'))
    task_id_2_demo_name: Dict[int, str] = collections.defaultdict(list)
    for x in os.listdir(path_to_input_dir):
        if ' @ ' not in x: 
            continue # skip non-demo folders
        if demo_name != x:
            try:
                trace_json: str = json.loads(open(get_path_to_trace_json(os.path.join(path_to_input_dir, x)), 'r').read())
                if is_same_site:
                    if anchor_site == trace_json['webarena']['sites'][0]:
                        task_id_2_demo_name[int(trace_json['webarena']['task_id'])].append(x)
                else:
                    task_id_2_demo_name[int(trace_json['webarena']['task_id'])].append(x)
            except Exception as e:
                print(f"Error reading `{x}` for task={task_id}: {e}")
                raise e
    
    # Sanity checks
    assert len(task_id_2_demo_name.keys()) >= n_tasks-1, f"Expected at least {n_tasks-1} other tasks in {path_to_input_dir}, got {len(task_id_2_demo_name.keys())}. Try reducing `--n_tasks`."

    # Sample a demo folder for `task_id` and a set of `n_tasks-1` random other tasks IDs
    random.seed(string_to_random_int(demo_name + '_segmentation'))
    other_task_ids: List[int] = random.sample(sorted(list(task_id_2_demo_name.keys())), n_tasks-1) # sample n_tasks-1 other tasks
    other_demo_names: List[str] = [ random.choice(task_id_2_demo_name[other_task_ids[x]]) for x in range(n_tasks-1) ] # sample one demo for each other task
    demo_names: List[str] = [ demo_name ] + other_demo_names

    # Sanity checks
    assert len(demo_names) == n_tasks, f"Expected {n_tasks} tasks, got {len(demo_names)}"
    assert len(set(demo_names)) == len(demo_names), f"Expected unique demos, got {demo_names}"
    assert len(set(other_task_ids)) == len(other_task_ids), f"Expected unique task IDs, got {other_task_ids}"

    # Save results
    if demo_name not in demo_2_config:
        demo_2_config[demo_name] = {}
    if str(n_tasks) not in demo_2_config[demo_name]:
        demo_2_config[demo_name][str(n_tasks)] = {}
    if f'is_same_site={is_same_site}' not in demo_2_config[demo_name][str(n_tasks)]:
        demo_2_config[demo_name][str(n_tasks)][f'is_same_site={is_same_site}'] = {
            'concatenate' : {
                'demo_names' : demo_names,
            }
        }
    json.dump(demo_2_config, open(path_to_demo_2_config, 'w'), indent=2)
    
    return demo_2_config

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"

    # Task-specific flags
    ## Dataset construction
    is_interleave: bool = args.is_interleave
    is_concatenate: bool = args.is_concatenate
    n_tasks: int = args.n_tasks
    is_same_site: bool = args.is_same_site
    ## Prompting
    is_td: bool = args.is_td
    is_kf: bool = args.is_kf
    is_act: bool = args.is_act
    is_include_sop: bool = args.is_include_sop
    is_prompt_uuid: bool = args.is_prompt_uuid
    n_trials: int = args.n_trials
    model: str = args.model
    assert sum([is_interleave, is_concatenate]) == 1, "Must specify EXACTLY ONE of --is_interleave or --is_concatenate"
    assert sum([is_act, is_kf]) >= 1, "Must specify AT LEAST ONE of --is_act or --is_kf"
    
    run(
        path_to_input_dir, 
        path_to_output_dir, 
        task_id,
        model,
        n_trials,
        is_interleave,
        is_same_site,
        is_concatenate,
        is_prompt_uuid,
        n_tasks,
        is_td,
        is_kf,
        is_act,
    )