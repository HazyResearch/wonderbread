import pandas as pd
import os
import concurrent.futures
from typing import Callable, Dict, List, Optional
from tqdm import tqdm
from wonderbread.helpers import find_files_by_prefix_suffix, get_rel_path
import hashlib

def run_experiment_process_demo(task):
    vals = [ val for k, val in task.items() if k != 'func' ]
    try:
        task['func'](*vals)
    except Exception as e:
        print(f"====> Error in task_id={task['task_id']} | demo_folder={task['path_to_demo_folder']}")
        print(str(e))

def run_experiment_execute_thread(df: pd.DataFrame, 
                                    run: Callable, 
                                    path_to_output_dir: str,
                                    model: str, 
                                    kwargs: Dict[str, str], 
                                    n_threads: int = 10, 
                                    is_path_to_demo_folder: bool = True,
                                    is_verbose: bool = False):
    """Run experiment for each row in `df`.
    
        If `is_path_to_demo_folder` is True, then the experiment evaluates a specific demo folder at a time.
            `df` should have a column `path_to_demo_folder` which contains the path to the specific demo folder we want to run an experiment on.
        If `is_path_to_demo_folder` is False, then the experiment can evaluate multiple demo folders at once, so we instead just pass it the parent demos/ folder.
    """
    tasks = []
    for idx, row in df.iterrows():
        path_to_demo_folder: str = get_rel_path(__file__, os.path.join('../../../', row['path_to_demo_folder']))
        tasks.append({
            'func' : run,
            'path_to_demo_folder' : path_to_demo_folder if is_path_to_demo_folder else get_rel_path(__file__, '../../../data/demos/'), # either provide a path to a specific demo folder or use the parent demos/ folder
            'path_to_output_dir' : path_to_output_dir, 
            'task_id' : int(row['task_id']), 
            'model' : model, 
            **kwargs,
            'is_verbose' : is_verbose,
        })
    if n_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(tqdm(executor.map(run_experiment_process_demo, tasks), total=len(tasks)))
    else:
        results = list(tqdm(map(run_experiment_process_demo, tasks), total=len(tasks)))
    return results

def get_completed_ablations(path_to_csv: str = './outputs/*_all_results.csv') -> Dict[int, List[Dict[str, str]]]:
    """For each task and ablation, check if we have already run an experiment for it. If so, add it to a list of completed ablations."""
    if path_to_csv is None or not os.path.exists(path_to_csv):
        return []

    df: pd.DataFrame = pd.read_csv(path_to_csv)

    # Casting
    for col in df.columns:
        if col.startswith('ablation--is_'):
            df[col] = df[col].astype(bool)

    task_id_2_done_ablations: Dict[int, List[Dict[str, str]]] = {}
    for idx, row in df.iterrows():
        task_id = row['task_id']
        if task_id not in task_id_2_done_ablations: task_id_2_done_ablations[task_id] = []
        task_id_2_done_ablations[task_id].append({
            key.replace('ablation--', ''): val for key, val in row.items() if key.startswith('ablation--')
        })

    print('# of completed ablations:', len(task_id_2_done_ablations))
    return task_id_2_done_ablations

def string_to_random_int(string: str) -> int:
    """Useful for seeding randomness based on a string."""
    # Hash the string using SHA256
    string = str(string)
    hashed_string = hashlib.sha256(string.encode()).hexdigest()
    # Take the first 8 characters of the hash
    hash_int = int(hashed_string[:8], 16)
    # Map the hash value to a range between 0 and 9999999
    random_int = hash_int % 9999999
    return random_int

def run_experiment(run: Callable, 
                    current__file__: str,
                    kwarg_settings: List[Dict[str, str]], 
                    is_path_to_demo_folder: bool,
                    n_threads: int = 20, 
                    model: str = 'GPT4', 
                    is_use_rank_1_df: bool = False,
                    is_skip_completed_ablations: bool = False,
                    is_debug: bool = False,
                    is_verbose: bool = False):

    # Create output directory
    path_to_output_dir = get_rel_path(current__file__, './outputs')
    print(f"Saving outputs to: ", path_to_output_dir)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load dataset
    if is_use_rank_1_df:
        df_rankings = pd.read_csv(get_rel_path(__file__, '../../../data/df_rankings.csv'))
        df = df_rankings[df_rankings['rank'] == 1]
    else:
        df = pd.read_csv(get_rel_path(__file__, '../../../data/df_valid.csv'))
    
    if is_debug:
        df = df.iloc[:3]
        n_threads = 1

    # Collect results before run (in case we missed any previously)
    os.system(f'python {get_rel_path(current__file__, "collect_results.py")} {path_to_output_dir} --path_to_output_dir {path_to_output_dir}')

    # Run experiment
    for kwargs in kwarg_settings:
        # Skip ablations we've already done (if desired)
        df_filtered = df.copy()
        if is_skip_completed_ablations:
            ablation_settings: Dict[str, str] = kwargs | { 'model' : model }
            path_to_csv: List[str] = find_files_by_prefix_suffix(path_to_output_dir, suffix='_all_results.csv')
            task_id_2_done_ablations: Dict[int, List[Dict[str, str]]]= get_completed_ablations(path_to_csv=os.path.join(path_to_output_dir, path_to_csv[0]) if len(path_to_csv) > 0 else None)
            df_filtered = []
            for idx, row in df.iterrows():
                task_id: str = int(row['task_id'])
                if not (
                    task_id in task_id_2_done_ablations
                    and ablation_settings in task_id_2_done_ablations[task_id]
                ):
                    df_filtered.append(row)
            print(f"Doing {len(df_filtered)} / {len(df)} runs that we haven't done yet.")
            df_filtered = pd.DataFrame(df_filtered)
        
        run_experiment_execute_thread(df_filtered, 
                                        run, 
                                        path_to_output_dir, 
                                        model, 
                                        kwargs, 
                                        n_threads=n_threads, 
                                        is_path_to_demo_folder=is_path_to_demo_folder, 
                                        is_verbose=is_verbose)

        # Collect results after each kwarg
        os.system(f'python {get_rel_path(current__file__, "collect_results.py")} {path_to_output_dir} --path_to_output_dir {path_to_output_dir}')
