from wonderbread.benchmark.tasks.helpers import run_experiment
from wonderbread.benchmark.tasks.documentation.demo_segmentation.main import run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3"] )
parser.add_argument("--is_debug", action='store_true', default=False, help="If set, run in debug mode (only 3 examples)")
args = parser.parse_args()

kwarg_settings = [
    {
        'n_tasks' : 3, 
        'is_td' : False,
        'is_kf' : True,
        'is_act' : False,
        'is_include_sop' : False,
    },
    {
        'n_tasks' : 3,
        'is_td' : False,
        'is_kf' : True,
        'is_act' : False,
        'is_include_sop' : True,
    },
    {
        'n_tasks' : 3,
        'is_td' : True,
        'is_kf' : True,
        'is_act' : False,
        'is_include_sop' : True,
    },
]

# Inject default shared settings
kwarg_settings = [
    {
        'n_trials' : 1,
        'is_interleave' : False,
        'is_same_site' : True,
        'is_concatenate' : True,
        'is_prompt_uuid' : False,
    } | x
    for x in kwarg_settings
]

run_experiment(run, __file__, kwarg_settings, model=args.model, is_debug=args.is_debug, n_threads=1, is_path_to_demo_folder=True, is_use_rank_1_df=True, is_skip_completed_ablations=True)
