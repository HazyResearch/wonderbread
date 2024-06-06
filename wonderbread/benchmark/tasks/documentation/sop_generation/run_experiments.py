from wonderbread.benchmark.tasks.helpers import run_experiment
from wonderbread.benchmark.tasks.documentation.sop_generation.main import run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3"] )
args = parser.parse_args()

kwarg_settings = [
    {
        'is_td' : True, 
        'is_kf' : False,
        'is_act' : False,
        'is_pairwise' : False,
    },
    {
        'is_td' : True, 
        'is_kf' : True,
        'is_act' : False,
        'is_pairwise' : False,
    },
    {
        'is_td' : True, 
        'is_kf' : True,
        'is_act' : True,
        'is_pairwise' : False,
    },
]

run_experiment(run, __file__, kwarg_settings, n_threads=1, model=args.model, is_path_to_demo_folder=True, is_use_rank_1_df=True, is_skip_completed_ablations=True)