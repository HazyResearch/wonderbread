"""
Usage:

python eval.py path_to_input_csv
"""

import os
from tqdm import tqdm
import pandas as pd
import argparse
from wonderbread.helpers import (
    _fetch_completion,
)
from wonderbread.benchmark.tasks.knowledge_transfer.question_answering.prompts import (
    prompt__completeness_score,
    prompt__soundness_score,
    prompt__clarity_score,
    prompt__compactness_score
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "path_to_input_csv", default=None, type=str, help="Path to input CSV file with QnA data", )
    parser.add_argument( "--path_to_output_dir", default="./outputs", type=str, help="Path to output dir to save output CSV", )
    return parser.parse_args()

def get_completeness_score(question: str, human_label: str, response: str) -> int:
    """Given a question and response, return the completeness score, on a scale of 1-3. 1 is high, 3 is low. Please score this metric harshly."""
    if pd.isna(response) or len(response) < 5:
        return "NA" # Skip if response is too short
    prompt = prompt__completeness_score(question, human_label, response)
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    score = _fetch_completion(messages, 'GPT4')
    return score

def get_soundness_score(question: str, human_label: str, response: str) -> int:
    """Given a question and response, return the soundness score, on a scale of 1-3. 1 is high, 3 is low. Please score this metric harshly."""
    if pd.isna(response) or len(response) < 5:
        return "NA" # Skip if response is too short
    prompt = prompt__soundness_score(question, human_label, response)
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    score = _fetch_completion(messages, 'GPT4')
    return score

def get_clarity_score(question: str, response: str) -> int:
    """Given a question and response, return the clarity score, on a scale of 1-3. 1 is high, 3 is low. Please score this metric harshly."""
    if pd.isna(response) or len(response) < 5:
        return "NA" # Skip if response is too short
    prompt = prompt__clarity_score(question, response)
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    score = _fetch_completion(messages, 'GPT4')
    return score

def get_compactness_score(question: str, response: str) -> int:
    """Given a question and response, return the compactness score, on a scale of 1-3. 1 is high, 3 is low. Please score this metric harshly."""
    if pd.isna(response) or len(response) < 5:
        return "NA" # Skip if response is too short
    prompt = prompt__compactness_score(question, response)
    messages = [ { 'role' : 'user', 'content' : [{ 'type' : 'text', 'text' : prompt }] } ]
    score = _fetch_completion(messages, 'GPT4')
    return score

def get_eval_for_qa(question: str, human_label: str, response: str) -> dict:
    """Given a question, human label and response, return the evaluation metrics."""
    completeness_score = get_completeness_score(question, human_label, response)
    soundness_score = get_soundness_score(question, human_label, response)
    clarity_score = get_clarity_score(question, response)
    compactness_score = get_compactness_score(question, response)
    evals = {
        "completeness_score": completeness_score,
        "soundness_score": soundness_score,
        "clarity_score": clarity_score,
        "compactness_score": compactness_score,
    }
    return evals

def run(path_to_input_csv: str, path_to_output_dir: str) -> None:
    df_input = pd.read_csv(path_to_input_csv)
    os.makedirs(path_to_output_dir, exist_ok=True)
    input_filename = os.path.basename(path_to_input_csv)
    path_to_output_csv: str = os.path.join(path_to_output_dir, input_filename)

    df_output = df_input.copy()

    evals = []
    for _, row in tqdm(df_input.iterrows(), total=df_input.shape[0]):
        question = row['Question Instantiation']
        human_label = row['Human Label']
        response = row['Response']
        evals_row = get_eval_for_qa(question, human_label, response)
        evals.append(evals_row)
    
    evals_df = pd.DataFrame(evals)
    df_output = pd.concat([df_output, evals_df], axis=1)
    df_output.to_csv(path_to_output_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    path_to_input_csv: str = args.path_to_input_csv
    path_to_output_dir: str = args.path_to_output_dir
    
    run(
        path_to_input_csv,
        path_to_output_dir, 
    )


