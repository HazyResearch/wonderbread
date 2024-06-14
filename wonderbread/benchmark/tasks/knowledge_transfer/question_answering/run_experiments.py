from wonderbread.benchmark.tasks.helpers import get_rel_path
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3", "Human"] )
parser.add_argument("--is_debug", action='store_true', default=False, help="If set, run in debug mode (only 3 examples)")
args = parser.parse_args()

path_to_demos: str = get_rel_path(__file__, "../../../../../data/demos")
path_to_qa: str = get_rel_path(__file__, "../../../../../data/qa_dataset.csv")

os.system(f"python generate_responses.py {path_to_demos} --path_to_input_csv {path_to_qa} --model {args.model} {'--is_debug' if args.is_debug else ''}")

if not os.path.exists(f"./outputs/qna_{args.model}.csv"):
    print(f"ERROR: No output file generated from `generate_responses.py`. Expected file at: `./outputs/qna_{args.model}.csv`")
    exit(1)
    
os.system(f'python eval.py "./outputs/qna_{args.model}.csv"')
os.system("python collect_results.py ./outputs/")