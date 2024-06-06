"""
Usage:

python generate_responses.py /Users/michaelwornow/Desktop/demonstration-collection/data/gold_demos \
    --path_to_input_csv /Users/michaelwornow/Desktop/demonstration-collection/data/qa_dataset.csv \
    --model Claude3
"""

import os
import traceback
from typing import List
from tqdm import tqdm
import pandas as pd
import argparse
from wonderbread.helpers import (
    _fetch_completion,
    create_merged_trace,
    get_path_to_sop_txt,
)
from wonderbread.benchmark.tasks.knowledge_transfer.question_answering.prompts import (
    prompt__merged_workflows_trace_act_only,
    prompt__two_workflows_trace_act_only,
    second_workflow,
    prompt__qa_trace_action_only,
    prompt__qa_two_sops_only,
    prompt__qa_sop_only,
    prompt__qa_question
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "path_to_input_dir", type=str, help="Path to directory containing demos" )
    parser.add_argument( "--path_to_input_csv", default=None, type=str, required=True, help="Path to input CSV file with QnA data", )
    parser.add_argument( "--path_to_output_dir", default="./outputs", type=str, required=False, help="Path to output dir to save output CSV", )
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for abstraction, one of [GPT4, GeminiPro, Human], \
                        `Human` just copies the human labels provided in the input CSV file" ,
                        choices=["GPT4", "GeminiPro", "Claude3", "Human"] )

    return parser.parse_args()

def single_sop_response(path_to_input_dir: str, task_id: str, question: str, model: str) -> str:
    # Load SOP
    demo_folder: str = os.path.join(path_to_input_dir, task_id)
    path_to_sop_file: str = get_path_to_sop_txt(demo_folder)
    sop: str = open(path_to_sop_file, 'r').read()
    sop = sop[sop.index('\n'):] # Remove first line
    # Make prompt
    prompt: str = prompt__qa_sop_only(sop) + prompt__qa_question(question)
    # Fetch response
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    response: str = _fetch_completion(messages, model)
    return response

def single_trace_response(path_to_input_dir: str, task_id: str, question: str, model: str) -> str:
    # Load trace
    path_to_demo_folder: str = os.path.join(path_to_input_dir, task_id)
    trace_logs = create_merged_trace([path_to_demo_folder], is_interleave=False, is_concatenate=True, is_keep_act=True, is_keep_kfs=True, random_seed=1)
    # Make prompt
    intro_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__qa_trace_action_only
        }]
    }
    merged_prompts: list[dict[str, str]] = [
        x['item']
        for x in trace_logs
    ]
    close_prompt: dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__qa_question(question)
        }]
    }
    messages: list[dict] = [intro_prompt] + merged_prompts + [close_prompt]
    # Fetch response
    try:
        response: str = _fetch_completion(messages, model)
    except Exception as e:
        print(f"Error for task_id={task_id} | demo_name={task_id}: {e}")
        raise e
    return response

def double_sop_response(path_to_input_dir: str, task_id1: str, task_id2: str, question: str, model: str) -> str:
    # Load SOP
    demo_folder1: str = os.path.join(path_to_input_dir, task_id1)
    path_to_sop_file1: str = get_path_to_sop_txt(demo_folder1)
    sop1: str = open(path_to_sop_file1, 'r').read()
    sop1 = sop1[sop1.index('\n'):] # Remove first line
    demo_folder2: str = os.path.join(path_to_input_dir, task_id2)
    path_to_sop_file2: str = get_path_to_sop_txt(demo_folder2)
    sop2: str = open(path_to_sop_file2, 'r').read()
    sop2 = sop2[sop2.index('\n'):] # Remove first line
    # Make prompt
    prompt: str = prompt__qa_two_sops_only(sop1, sop2) + prompt__qa_question(question)
    # Fetch response
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    response: str = _fetch_completion(messages, model)
    return response

def double_trace_response(path_to_input_dir: str, task_id1: str, task_id2: str, question: str, model: str) -> str:
    # Load trace
    path_to_demo_folder1: str = os.path.join(path_to_input_dir, task_id1)
    path_to_demo_folder2: str = os.path.join(path_to_input_dir, task_id2)
    trace1_logs = create_merged_trace([path_to_demo_folder1], is_interleave=False, is_concatenate=True, is_keep_act=True, is_keep_kfs=True, random_seed=1)
    trace2_logs = create_merged_trace([path_to_demo_folder2], is_interleave=False, is_concatenate=True, is_keep_act=True, is_keep_kfs=True, random_seed=1)
    # Make prompt
    intro_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__two_workflows_trace_act_only
        }]
    }
    trace1_prompts: list[dict[str, str]] = [
        x['item']
        for x in trace1_logs
    ]
    task_separator_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : second_workflow
        }]
    }
    trace2_prompts: list[dict[str, str]] = [
        x['item']
        for x in trace2_logs
    ]
    close_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__qa_question(question)
        }]
    }
    messages: list[dict] = [intro_prompt] + trace1_prompts + [task_separator_prompt] + trace2_prompts + [close_prompt]
    # Fetch response
    try:
        response: str = _fetch_completion(messages, model)
    except Exception as e:
        print(f"Error for task_id={task_id1}, task_id2={task_id2} | demo_name={task_id1}, {task_id2}: {e}")
        raise e
    return response

def triple_trace_response(path_to_input_dir: str, task_id1: str, task_id2: str, task_id3: str, question: str, model: str) -> str:
    path_to_demo_folders = [os.path.join(path_to_input_dir, task_id) for task_id in [task_id1, task_id2, task_id3]]
    trace_logs = create_merged_trace(path_to_demo_folders, is_interleave=False, is_concatenate=True, is_keep_act=True, is_keep_kfs=True, random_seed=1)
    # Make prompt
    intro_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__merged_workflows_trace_act_only
        }]
    }
    merged_trace_prompts: list[dict[str, str]] = [
        x['item']
        for x in trace_logs
    ]
    close_prompt: dict = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__qa_question(question)
        }]
    }
    messages: list[dict] = [intro_prompt] + merged_trace_prompts + [close_prompt]
    # Fetch response
    try:
        response: str = _fetch_completion(messages, model)
    except Exception as e:
        print(f"Error for task_id={task_id1}, task_id2={task_id2}, task_id3={task_id3} | demo_name={task_id1}, {task_id2}, {task_id3}: {e}")
        raise e
    return response

def generate_response(path_to_input_dir: str, task_ids: str, question: str, evidence: str, model: str) -> str:
    if evidence == 'SOP':
        task_id = task_ids.strip()
        response = single_sop_response(path_to_input_dir, task_id, question, model)
    elif evidence == 'Trace+Screenshots':
        task_id = task_ids.strip()
        response = single_trace_response(path_to_input_dir, task_id, question, model)
    elif evidence == 'SOP, SOP':
        task_id_list = task_ids.split(',')
        assert len(task_id_list) == 2, f"Expected 2 task IDs, got {len(task_id_list)}"
        task_id1, task_id2 = task_id_list[0].strip(), task_id_list[1].strip()
        response = double_sop_response(path_to_input_dir, task_id1, task_id2, question, model)
    elif evidence == 'Trace+Screenshots, Trace+Screenshots':
        task_id_list = task_ids.split(',')
        assert len(task_id_list) == 2, f"Expected 2 task IDs, got {len(task_id_list)}"
        task_id1, task_id2 = task_id_list[0].strip(), task_id_list[1].strip()
        response = double_trace_response(path_to_input_dir, task_id1, task_id2, question, model)
    elif evidence == 'Trace+Screenshots, Trace+Screenshots, Trace+Screenshots':
        task_id_list = task_ids.split(',')
        assert len(task_id_list) == 3, f"Expected 3 task IDs, got {len(task_id_list)}"
        task_id1, task_id2, task_id3 = task_id_list[0].strip(), task_id_list[1].strip(), task_id_list[2].strip()
        response = triple_trace_response(path_to_input_dir, task_id1, task_id2, task_id3, question, model)
    else:
        raise ValueError(f"Unknown evidence type: {evidence}")
    return response

def run(path_to_input_dir: str,
        path_to_input_csv: str,
        path_to_output_dir: str,
        model: str) -> None:
    """Generate QnA responses for each task in the input CSV file and save to output CSV file"""

    df_input = pd.read_csv(path_to_input_csv)

    # Create output directory
    os.makedirs(path_to_output_dir, exist_ok=True)
    path_to_output_csv: str = os.path.join(path_to_output_dir, f"qna_{model}.csv")
    
    # Skip rows we've already done
    df_output = pd.read_csv(path_to_output_csv) if os.path.exists(path_to_output_csv) else df_input.copy()

    responses: List[str] = []
    if model != "Human":
        # Generate responses
        for i, row in tqdm(df_input.iterrows(), desc='Generating responses', total=df_input.shape[0]):
            task_ids = row['Task ID(s)']
            question = row['Question Instantiation']
            evidence = row['Evidence']
            # Skip if we've already done this row
            if not df_output.empty:
                existing_row = df_output[(df_output['Task ID(s)'] == task_ids) & (df_output['Question Instantiation'] == question)]
                if (
                    existing_row.shape[0] > 0 
                    and 'Response' in existing_row.columns
                    and len(str(existing_row['Response'].values[0])) > 0
                    and str(existing_row['Response'].values[0]) != "#NAME?"
                    and str(existing_row['Response'].values[0]) != "nan"
                ):
                    responses.append(existing_row['Response'].values[0])
                    continue
            try:
                responses.append(generate_response(path_to_input_dir, task_ids, question, evidence, model))
            except Exception as e:
                print(f"Error for row {i}: {e}")
                print(traceback.format_exc())
                responses.append(None)
        df_input['Response'] = responses
    else:
        # Copy human labels
        df_input['Response'] = df_input['Human Label']
    df_input['ablation--model'] = model

    # Save output
    if 'Unnamed: 0' in df_input.columns: df_input = df_input.drop(columns=['Unnamed: 0'])
    df_input.to_csv(path_to_output_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_input_csv: str = args.path_to_input_csv
    path_to_output_dir: str = args.path_to_output_dir
    
    # Task-specific flags
    model: str = args.model

    run(
        path_to_input_dir, 
        path_to_input_csv,
        path_to_output_dir, 
        model,
    )

