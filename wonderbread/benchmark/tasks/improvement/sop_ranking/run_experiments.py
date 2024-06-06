from wonderbread.benchmark.tasks.helpers import run_experiment
from wonderbread.benchmark.tasks.improvement.sop_ranking.main import run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3"] )
args = parser.parse_args()

kwarg_settings = [
    {},
]

run_experiment(run, __file__, kwarg_settings, n_threads=1, model=args.model, is_path_to_demo_folder=False, is_use_rank_1_df=True, is_skip_completed_ablations=True)