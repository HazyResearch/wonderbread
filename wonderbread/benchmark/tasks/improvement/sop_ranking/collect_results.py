import pandas as pd
import argparse
import os
from typing import List, Dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_input_dir",
        help="Path to folder containing all task folders.",
    )
    parser.add_argument(
        "--path_to_output_dir",
        default="./outputs/",
        type=str,
        required=False,
        help="Path to output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir

    results: List[pd.DataFrame] = []
    for demo_folder in os.listdir(path_to_input_dir):
        # Only consider folders
        if not os.path.isdir(os.path.join(path_to_input_dir, demo_folder)):
            continue
        for file in os.listdir(os.path.join(path_to_input_dir, demo_folder)):
            if file.startswith("sop_ranking") and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(path_to_input_dir, demo_folder, file))
                df['demo_name'] = demo_folder
                df['task_id'] = int(demo_folder.split(" @ ")[0])
                results.append(df)

    # Save results to CSV
    df = pd.concat(results)
    df['ablation'] = df[[col for col in df.columns if col.startswith('ablation--')]].apply(lambda x: '--'.join(map(str, x)), axis=1)
    df.to_csv(os.path.join(path_to_output_dir, "sop_ranking_all_results.csv"), index=False)
    print(f"Saved output CSV to {os.path.join(path_to_output_dir, 'sop_ranking_all_results.csv')}")
