"""
Usage:

python main.py '/Users/mwornow/Downloads/tasks/' --is_td
"""
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Dict, List, Optional, Any
import json
import argparse
from wonderbread.helpers import (
    _fetch_completion,
    add_standard_experiment_args,
    build_prompt_s_a_sequence,
    filter_prompt_s_a_sequence,
    get_folders_for_task_id,
    get_path_to_screenshots_dir,
    get_path_to_sop_txt,
    get_path_to_trace_json,
    get_webarena_task_json,
)
from wonderbread.benchmark.tasks.documentation.sop_generation.prompts import (
    prompt__td_intro,
    prompt__td_close,
    prompt__td_kf_intro,
    prompt__td_kf_close,
    prompt__td_act_intro,
    prompt__td_act_close,
    prompt__td_kf_act_intro,
    prompt__td_kf_act_close,
    prompt__td_kf_intro__pairwise,
    prompt__td_kf_close__pairwise,
    prompt__td_kf_act_intro__pairwise,
    prompt__td_kf_act_close__pairwise,
    prompt__join_pairwise,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument( "--is_td", action="store_true", default=False, help="If TRUE, then include task description into prompts",)
    parser.add_argument( "--is_kf", action="store_true", default=False, help="If TRUE, then include screenshots as key frames into prompts", )
    parser.add_argument( "--is_act", action="store_true", default=False, help="If TRUE, then include action traces into prompts", )
    parser.add_argument( "--is_pairwise", action="store_true", default=False, help="If TRUE, then instead of prompting all screenshots / action traces at once, prompt two at a time (before/after), and piece together SOP afterwards" )
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for self monitoring, one of [GPT4, GeminiPro]",
                        choices=["GPT4", "GeminiPro", "Claude3"] )
    parser.add_argument("--is_verbose", action="store_true", help="If TRUE, then print out stuff")
    return parser.parse_args()

def kwarg_setting_to_ablation(kwargs: Dict[str, str]) -> str:
    # Parse kwargs
    is_td = kwargs.get('is_td', False)
    is_kf = kwargs.get('is_kf', False)
    is_act = kwargs.get('is_act', False)
    is_pairwise = kwargs.get('is_pairwise', False)
    model = kwargs.get('model', None)
    # Generate ablation string
    short_name: str = []
    if is_td:
        short_name.append('td')
    if is_kf:
        short_name.append('kf')
    if is_act:
        short_name.append('act')
    short_name = '_'.join(short_name)
    short_name += '__pairwise' if is_pairwise else ''
    short_name += f"__{model}"
    return short_name

def run(path_to_demo_folder: str, 
        path_to_output_dir: str, 
        task_id: int,
        model: str, 
        is_td: bool, 
        is_kf: bool, 
        is_act: bool,
        is_pairwise: bool,
        is_verbose: bool = False) -> None:

    # Create output directory
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Get WebArena task description
    task_json: Optional[Dict[str, str]] = get_webarena_task_json(task_id)
    assert task_json is not None, f"Could not find WebArena task json for {task_id}"
    task_descrip: str = task_json["intent"]
    start_url: str = task_json["start_url"]
    ui_name: str = {
        'gitlab' : 'Gitlab',
        'shopping' : 'Generic e-commerce site',
        'shopping_admin' : 'Generic e-commerce admin based on Adobe Magneto',
        'reddit' : 'Generic open source Reddit clone',
    }[task_json["sites"][0]]
    
    # Get .json trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    trace_json: Dict[str, str] = json.loads(open(path_to_trace, "r").read())['trace']
    
    # Copy over ground truth SOP.txt
    if is_td:
        path_to_sop_file: str = get_path_to_sop_txt(path_to_demo_folder)
        gt_sop = open(path_to_sop_file, "r").read()
        with open(os.path.join(path_to_output_dir, path_to_sop_file.split('/')[-1]), 'w') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Task: {task_descrip}\n")
            f.write(f"UI: {ui_name}\n")
            f.write(f"Start URL: {start_url}\n")
            f.write("----------------------------------------\n")
            f.write(gt_sop)

    # Loop through trace, interleaving screenshots (states) and actions
    prompt_s_a_sequence, __ = build_prompt_s_a_sequence(trace_json, path_to_screenshots_dir)
    prompt_s_a_sequence, __ = filter_prompt_s_a_sequence(prompt_s_a_sequence, is_kf, is_act)
    
    if is_td and is_kf and is_act:
        # TD + KF + ACT
        intro_prompt_fn = prompt__td_kf_act_intro__pairwise if is_pairwise else prompt__td_kf_act_intro
        close_prompt_fn = prompt__td_kf_act_close__pairwise if is_pairwise else prompt__td_kf_act_close
    elif is_td and is_kf and not is_act:
        # TD + KF
        intro_prompt_fn = prompt__td_kf_intro__pairwise if is_pairwise else prompt__td_kf_intro
        close_prompt_fn = prompt__td_kf_close__pairwise if is_pairwise else prompt__td_kf_close
    elif is_td and not is_kf and is_act:
        # TD + ACT
        assert is_pairwise is False, "Pairwise not supported for -is_act without --is_kf"
        intro_prompt_fn = prompt__td_act_intro
        close_prompt_fn = prompt__td_act_close
    elif is_td and not is_kf and not is_act:
        assert is_pairwise is False, "Pairwise not supported for -is_td only"
        assert len(prompt_s_a_sequence) == 0, "Expected no screenshots or actions in prompt_s_a_sequence"
        # TD
        intro_prompt_fn = prompt__td_intro
        close_prompt_fn = prompt__td_close
    else:
        raise ValueError("Invalid combination of flags")
    
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : intro_prompt_fn(task_descrip if is_td else None, ui_name)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : close_prompt_fn()
        }]
    }

    if is_pairwise:
        responses: List[str] = []
        if is_kf and is_act:
            # Feed (S, A, S') -- i.e. a pair of before/after screenshots
            for i in range(0, len(prompt_s_a_sequence)-2, 2):
                messages: List[str] = [intro_prompt] + prompt_s_a_sequence[i:i+3] + [close_prompt]
                assert len(messages) == 1 + 3 + 1, f"Expected 5 prompts, got {len(messages)}"
                try:
                    response: str = _fetch_completion(messages, model)
                except Exception as e:
                    print(f"Error for task_id={task_id} | demo_name={demo_name} | i={i}: {e}")
                    raise e
                responses.append(response)
        else:
            # Feed (S, S') or (A, A') -- i.e. a pair of before/after items
            for i in range(0, len(prompt_s_a_sequence)-1, 1):
                messages: List[str] = [intro_prompt] + prompt_s_a_sequence[i:i+2] + [close_prompt]
                assert len(messages) == 1 + 2 + 1, f"Expected 4 prompts, got {len(messages)}"
                try:
                    response: str = _fetch_completion(messages, model)
                except Exception as e:
                    print(f"Error for task_id={task_id} | demo_name={demo_name} | i={i}: {e}")
                    raise e
                responses.append(response)
        response: str = "\n>>>>>>>>>>>\n".join(responses)
    else:
        # Feed (S, A, S', A', S'', A'', ...) -- i.e. everything at once. Note: This might be (S, S', S'') or (A, A', A'') depending on the ablation
        messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
        try:
            if model == 'Claude3':
                # Claude 3 limits to <= 20 images, so split query into multiple parts (if necessary) and combine later
                # We'll be conservative and limit to 10 images
                if len([ x for x in prompt_s_a_sequence if x['content'][0]['type'] == 'image_url' ]) > 10:
                    step: int = 20 if is_kf and is_act else 10
                    responses: List[str] = []
                    for i in range(0, len(prompt_s_a_sequence), step):
                        prompt_s_a_sequence_part = prompt_s_a_sequence[i:i+step]
                        messages = [intro_prompt] + prompt_s_a_sequence_part + [close_prompt]
                        responses.append(_fetch_completion(messages, model))
                    response: str = "\n>>>>>>>>>>>\n".join(responses)
                    # Merge responses
                    messages: List[str] = [{
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": prompt__join_pairwise(response, '>>>>>>>>>>>'),
                        }],
                    }]
                    response: str = _fetch_completion(messages, model)
                else:
                    response: str = _fetch_completion(messages, model)
            else:
                response: str = _fetch_completion(messages, model)
        except Exception as e:
            print(f"Error for task_id={task_id} | demo_name={demo_name}: {e}")
            raise e

    if is_verbose: print(response)
    
    # Remove extraneous chars
    response = response.replace("```\n", "").replace("```", "").strip()

    # Save SOP
    short_name: str = kwarg_setting_to_ablation({
        'model' : model,
        'is_td' : is_td,
        'is_kf' : is_kf,
        'is_act' : is_act,
        'is_pairwise' : is_pairwise,
    })
    with open( os.path.join(path_to_output_dir, f"Generated-SOP - {short_name} - {demo_name}.txt"), "w" ) as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Task: {task_descrip}\n")
        f.write(f"UI: {ui_name}\n")
        f.write(f"Start URL: {start_url}\n")
        f.write(f"Ablation: {short_name}\n")
        f.write("----------------------------------------\n")
        f.write(response)

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    is_verbose: bool = args.is_verbose
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"

    # Task-specific flags
    model: str = args.model
    is_td: bool = args.is_td
    is_kf: bool = args.is_kf
    is_act: bool = args.is_act
    is_pairwise: bool = args.is_pairwise
    assert is_td + is_kf + is_act > 0, "Must specify at least one of --is_td, --is_kf, --is_act, --is_kf_act"

    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(path_to_input_dir, task_id)

    # Loop through each demos's folder...
    for path_to_demo_folder in path_to_demo_folders:
        run(
            path_to_demo_folder, 
            path_to_output_dir, 
            task_id,
            model,
            is_td, 
            is_kf, 
            is_act, 
            is_pairwise,
            is_verbose,
        )