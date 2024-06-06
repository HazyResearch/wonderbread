from wonderbread.benchmark.tasks.helpers import run_experiment
from wonderbread.benchmark.tasks.improvement.sop_improvement.main import run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3"] )
args = parser.parse_args()

kwarg_settings = [
    {
        'max_self_reflection_depth' : 1,
        'is_td' : True,
        'is_kf' : True,
        'is_act' : False,
    },
    {
        'max_self_reflection_depth' : 1,
        'is_td' : True,
        'is_kf' : False,
        'is_act' : True,
    },
    {
        'max_self_reflection_depth' : 1,
        'is_td' : True,
        'is_kf' : True,
        'is_act' : True,
    },
]

run_experiment(run, __file__, kwarg_settings, model=args.model, n_threads=1, is_path_to_demo_folder=True, is_use_rank_1_df=True, is_skip_completed_ablations=True)