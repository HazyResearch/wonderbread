from wonderbread.benchmark.tasks.helpers import run_experiment
from wonderbread.benchmark.tasks.knowledge_transfer.demo_validation.main import run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, default="GPT4", choices=["GPT4", "GeminiPro", "Claude3"] )
args = parser.parse_args()

kwarg_settings = [
    # Task completion
    {
        'version' : 'task_completion',
        'is_include_sop' : True,
        'is_td' : True, 
        'is_kf' : True, 
        'is_act' : False, 
        'n_negative_samples' : 3,
    },
    {
        'version' : 'task_completion',
        'is_include_sop' : False, 
        'is_td' : True, 
        'is_kf' : True, 
        'is_act' : False, 
        'n_negative_samples' : 3,
    },
    {
        'version' : 'task_completion',
        'is_include_sop' : False, 
        'is_td' : True, 
        'is_kf' : True, 
        'is_act' : False, 
        'n_negative_samples' : 3,
    },
    # Task trajectory
    {
        'version' : 'task_trajectory',
        'is_include_sop' : True, 
        'is_td' : True, 
        'is_kf' : True, 
        'is_act' : False, 
        'n_negative_samples' : 3,
    },
]

run_experiment(run, __file__, kwarg_settings, n_threads=1, model=args.model, is_path_to_demo_folder=True, is_use_rank_1_df=True, is_verbose=False, is_skip_completed_ablations=True)