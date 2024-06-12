import pandas as pd
import argparse
import os
from typing import List, Dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "path_to_input_dir", help="Path to folder containing all task folders.", )
    parser.add_argument( "--path_to_output_dir", default="./outputs/", type=str, required=False, help="Path to output directory", )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    results: List[pd.DataFrame] = []
    for file in os.listdir(path_to_output_dir):
        if file.startswith('qna_') and file.endswith(".csv"):
            results.append(pd.read_csv(os.path.join(path_to_output_dir, file)))

    # Save results to CSV
    if len(results) > 0:
        df = pd.concat(results)
        df['ablation'] = df[[col for col in df.columns if col.startswith('ablation--')]].apply(lambda x: '--'.join(map(str, x)), axis=1)
        df.to_csv(os.path.join(path_to_output_dir, 'knowledge_transfer_all_results.csv'), index=False)
        print(f"Saved output CSV to {os.path.join(path_to_output_dir, 'knowledge_transfer_all_results.csv')}")