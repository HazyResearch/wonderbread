import re
import pandas as pd
import argparse
import os
from typing import List, Dict

HEADER_DEMARCATOR: str = '----------------------------------------'

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
    
    results: List[Dict[str, str]] = []
    for demo_folder in os.listdir(path_to_input_dir):
        # Only consider folders
        if not os.path.isdir(os.path.join(path_to_input_dir, demo_folder)):
            continue

        for file in os.listdir(os.path.join(path_to_input_dir, demo_folder)):
            if file.startswith("Generated-SOP") and file.endswith('.txt'):
                contents: str = open(os.path.join(path_to_input_dir, demo_folder, file)).read()
                header: str = contents[:contents.index(HEADER_DEMARCATOR)].strip()
                task_id = re.search(r'Task ID: (\d+)', contents).group(1)
                sop: str = contents[contents.index(HEADER_DEMARCATOR) + len(HEADER_DEMARCATOR):].strip()
                # parse ablation
                ablation: str = re.search(r'Ablation: (.+)', contents).group(1)
                ablation__is_pairwise: bool = '__pairwise' in ablation
                ablation__is_td: bool = 'td' in ablation
                ablation__is_kf: bool = 'kf' in ablation
                ablation__is_act: bool = 'act' in ablation
                ablation__model: str = ablation.split('__')[-1]
                results.append({
                    'sop' : sop,
                    'demo_name' : demo_folder,
                    'task_id' : task_id,
                    'ablation--model' : ablation__model,
                    'ablation--is_pairwise' : ablation__is_pairwise,
                    'ablation--is_td' : ablation__is_td,
                    'ablation--is_kf' : ablation__is_kf,
                    'ablation--is_act' : ablation__is_act,
                })

    # Save results to CSV
    if len(results) > 0:
        df = pd.DataFrame(results)
        df['ablation'] = df[[col for col in df.columns if col.startswith('ablation--')]].apply(lambda x: '--'.join(map(str, x)), axis=1)
        df.to_csv(os.path.join(path_to_output_dir, 'sop_generation_all_results.csv'), index=False)
        print(f"Saved output CSV to {os.path.join(path_to_output_dir, 'sop_generation_all_results.csv')}")