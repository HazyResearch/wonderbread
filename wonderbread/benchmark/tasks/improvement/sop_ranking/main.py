"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

from typing import Any, Dict, List
import pandas as pd
import os
import argparse
import json
import dirtyjson
from scipy.stats import spearmanr, kendalltau
from wonderbread.helpers import (
    _fetch_completion,
    add_standard_experiment_args,
    get_folders_for_task_id,
    get_path_to_sop_txt,
    get_path_to_trace_json,
    get_rel_path,
)
from wonderbread.benchmark.tasks.improvement.sop_ranking.prompts import prompt__rank_sop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument( "--model", type=str, default="GPT4", help="Model to use for self monitoring, one of [GPT4, GeminiPro]",
                        choices=["GPT4", "GeminiPro"] )
    return parser.parse_args()


def helper_task_completion(task_descrip: str, sops: List[str], gt_ranking: List[int], model: str) -> Dict[str, str]:
    sop_str: str = ""
    for i in range(len(sops)):
        sop_str += f"#{i+1}\n ```\n{sops[i]}\n```\n\n"

    messages: List[Dict[str, str]] = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt__rank_sop(task_descrip, sop_str),
            }
        ],
    }]

    pred_raw_response: str = _fetch_completion(messages, model)

    try:
        json_dict = dirtyjson.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"JSON Parsing error: {e}. Retrying...")
        # Retry
        return helper_task_completion(task_descrip, sops, gt_ranking, model)

    pred_raw_response = json_dict["pred_ranking"]
    return {"pred_ranking": str(pred_raw_response), "gt_ranking": gt_ranking}


def evaluate_ranking(pred_ranking: List[int], gt_ranking: List[int]) -> Dict[str, Any]:
    # generate a metric comparing the two rankings

    # Calculate Spearman's rho
    spearman_corr, spearman_p_value = spearmanr(pred_ranking, gt_ranking)

    # Calculate Kendall's tau
    kendall_corr, kendall_p_value = kendalltau(pred_ranking, gt_ranking)

    return {
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p_value,
        "kendall_corr": kendall_corr,
        "kendall_p_value": kendall_p_value,
    }

def run(path_to_input_dir: str, 
        path_to_output_dir: str,
        task_id: int,
        model: str,
        is_verbose: bool = False):
    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(
        path_to_input_dir, task_id
    )
    
    # gt ranking
    df_rankings = pd.read_csv(os.path.join(get_rel_path(__file__, '../../../data/df_rankings.csv')))
    df_rankings = df_rankings[df_rankings['task_id'] == task_id]
    
    # Get task description
    task_descrip: str = json.load(open(get_path_to_trace_json(path_to_demo_folders[0]), 'r'))['webarena']['intent']

    # Loop through each demos's folder...
    list_of_sops: List[str] = []
    gt_rankings: List[int] = []
    folder_names: List[str] = []
    for path_to_demo_folder in path_to_demo_folders:
        folder_name: str = os.path.basename(path_to_demo_folder)
        if folder_name not in df_rankings['folder_name'].values:
            # Skip any unranked / invalid demos for this task
            continue
        # get all sops
        path_to_sop_file: str = get_path_to_sop_txt(path_to_demo_folder)
        sop: str = open(path_to_sop_file, "r").read()
        list_of_sops.append(sop)
        gt_rankings.append(df_rankings[df_rankings['folder_name'] == folder_name]['rank'].values[0])
        folder_names.append(folder_name)

    demo_name: str = os.path.basename(path_to_demo_folder.strip("/").split("/")[-1])
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Run task completion
    result: Dict[str, str] = helper_task_completion(task_descrip, list_of_sops, gt_rankings, model)

    # convert result into a list of rankings
    pred_rankings: List[int] = json.loads(result["pred_ranking"])

    # Evaluate ranking
    eval_results: Dict[str, Any] = evaluate_ranking(pred_rankings, gt_rankings)

    # save results
    df = pd.DataFrame(
        {
            "folder_name": folder_names, 
            "sop": list_of_sops,
            "gt_ranking": gt_rankings,
            "pred_ranking": pred_rankings,
            "spearman_corr": eval_results["spearman_corr"],
            "spearman_p_value": eval_results["spearman_p_value"],
            "kendall_corr": eval_results["kendall_corr"],
            "kendall_p_value": eval_results["kendall_p_value"],
            # Ablations
            "ablation--model": [model]*len(list_of_sops),
        }
    )

    df.to_csv(
        os.path.join(path_to_output_dir, f"sop_ranking__{model}.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = int(args.task_id)
    model: str = args.model
    assert 0 <= task_id <= 811, f"Expected task_id in [0, 811], got {task_id}"

    # Run
    run(
        path_to_input_dir,
        path_to_output_dir,
        task_id,
        model,
    )
