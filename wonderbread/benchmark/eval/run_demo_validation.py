import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import seaborn as sns
from wonderbread.helpers import get_rel_path
sns.set_style("whitegrid", {'axes.grid' : False})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_experimental_results_dir", type=str, default=get_rel_path(__file__, '../../../data/experimental_results'), help="Path to directory containing `all_results.csv` file for experiment", )
    parser.add_argument( "--is_bucket_rankings_to_3", action="store_true", help="Whether to bucket rankings into 3 categories instead of 5", )
    return parser.parse_args()

def make_plots(df: pd.DataFrame, path_to_output_dir: str, is_task_completion: bool):
    """Create plots of precision and recall + print out some statistics."""
    # Scatter plot of # of screenshots v. accuracy
    df.groupby(['n_screenshots']).agg({
        'is_correct' : 'mean',
    }).reset_index(drop=False)
    combos = [
        ("n_screenshots", "is_correct"),
    ]
    for (x_col, y_col) in combos:
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(os.path.join(path_to_output_dir, f"{x_col}_vs_{y_col}.pdf"))

def make_tables(df: pd.DataFrame, path_to_output_dir: str, is_task_completion: bool):
    """Create LaTeX tables."""
    # Make table with:
    #   - columns: Task Type, Accuracy
    #   - rows: ablation
    df_ = df.groupby(['ablation', 'task_type']).agg({
        'is_correct' : 'mean',
        'ablation_human' : 'first',
    }).reset_index(drop=False)
    # Make floats pretty
    df_['is_correct'] = df_['is_correct'].apply(lambda x: f"{x:.2f}")
    # Plot
    df_ = df_[['ablation_human', 'task_type', 'is_correct']]
    df_ = df_.sort_values(by='is_correct', ascending=False)
    df_ = df_.rename(columns={ 'ablation_human': 'Ablation', 'task_type' : 'Setting', 'is_correct': 'Accuracy', })
    df_.to_latex(os.path.join(path_to_output_dir, f'df_task_type_{"task_completion" if is_task_completion else "task_trajectory"}.tex'), index=False)
    
    # Make table with:
    #   - columns: Task Type, Precision, Recall, F1
    #   - rows: ablation
    results = []
    for ablation in df['ablation'].unique():
        df_= df[df['ablation'] == ablation]
        tp: int = df_[df_['is_positive']]['is_correct'].sum()
        fp: int = df_[df_['is_positive']].shape[0] - tp
        tn: int = df_[df_['is_negative']]['is_correct'].sum()
        fn: int = df_[df_['is_negative']].shape[0] - tn
        results.append({
            'ablation': ablation,
            'ablation_human': df_['ablation_human'].iloc[0],
            'ablation--model': df_['ablation--model'].iloc[0],
            'ablation--is_td': df_['ablation--is_td'].iloc[0],
            'ablation--is_kf': df_['ablation--is_kf'].iloc[0],
            'ablation--is_act': df_['ablation--is_act'].iloc[0],
            'ablation--is_include_sop': df_['ablation--is_include_sop'].iloc[0],
            'tp' : tp,
            'fp' : fp,
            'tn' : tn,
            'fn' : fn,
            'count' : tp + fp + tn + fn,
        })
    df_ = pd.DataFrame(results)
    df_['precision'] = df_['tp'] / (df_['tp'] + df_['fp'])
    df_['recall'] = df_['tp'] / (df_['tp'] + df_['fn'])
    df_['f1'] = 2 * (df_['precision'] * df_['recall']) / (df_['precision'] + df_['recall'])
    df_['f1'] = df_['f1'].fillna(0) # Fix divide by zero => F1 = 0
    # Make floats pretty
    df_['precision'] = df_['precision'].apply(lambda x: f"{x:.2f}")
    df_['recall'] = df_['recall'].apply(lambda x: f"{x:.2f}")
    df_['f1'] = df_['f1'].apply(lambda x: f"{x:.2f}")
    # Set columns
    df_['model'] = df_['ablation--model']
    df_['is_td'] = df_['ablation--is_td'].apply(lambda x: '\checkmark' if x else '')
    df_['is_kf'] = df_['ablation--is_kf'].apply(lambda x: '\checkmark' if x else '')
    df_['is_act'] = df_['ablation--is_act'].apply(lambda x: '\checkmark' if x else '')
    df_['is_include_sop'] = df_['ablation--is_include_sop'].apply(lambda x: '\checkmark' if x else '')
    df_ = df_[['ablation_human', 'model', 'is_td', 'is_kf', 'is_act', 'is_include_sop', 'precision', 'recall', 'f1', 'count']]
    # Plot
    df_ = df_.sort_values(by='ablation_human', ascending=False)
    df_ = df_.rename(columns={ 'ablation_human': 'Ablation', 'precision' : 'Precision', 'recall': 'Recall',  'f1' : 'F1', 'count' : 'Count', })
    df_ = df_.drop(columns=['Ablation'])
    df_.to_latex(os.path.join(path_to_output_dir, f'df_{"task_completion" if is_task_completion else "task_trajectory"}.tex'), index=False)

def gen_ablation_name(row: pd.Series) -> str:
    human_readable_ablation = row['ablation--model'] + ' - ' + '+'.join(
        ([ 'TD' ] if row['ablation--is_td'] else []) + 
        ([ 'KF' ] if row['ablation--is_kf'] else []) + 
        ([ 'ACT' ] if row['ablation--is_act'] else []) + 
        ([ 'SOP' ] if row['ablation--is_include_sop'] else [])
    )
    return human_readable_ablation

if __name__ == "__main__":
    args = parse_args()
    path_to_experimental_results_dir: str = args.path_to_experimental_results_dir
    experiment_name: str = 'demo_validation'
    path_to_output_dir: str = os.path.join(path_to_experimental_results_dir, experiment_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Get paths
    path_to_results_csv = os.path.join(path_to_experimental_results_dir, f"{experiment_name}_all_results.csv")

    # Read in the all_results.csv file
    df_all_results = pd.read_csv(path_to_results_csv)
    df_all_results['n_screenshots'] = df_all_results['paths_to_screenshots'].apply(lambda x: len(x.split(',')))
    
    # ! HACK: Need to do hacky fillna() here because of a bug in the data collection
    df_all_results['ablation--is_td'] = df_all_results['ablation--is_td'].fillna(True)
    df_all_results['ablation--is_kf'] = df_all_results['ablation--is_kf'].fillna(True)
    df_all_results['ablation--is_act'] = df_all_results['ablation--is_act'].fillna(True)

    # Rename ablations to human friendly names
    df_all_results['ablation_human'] = df_all_results.apply(gen_ablation_name, axis=1)
    
    # Group positives / negatives -> measure precision, recall, f1
    df_all_results['is_positive'] = df_all_results['task_type'].str.lower() == 'true'
    df_all_results['is_negative'] = df_all_results['task_type'].str.lower() != 'true'

    # Make tables
    for x in [True, False]:
        df = df_all_results[df_all_results['ablation--is_task_completion'] == x]
        make_tables(df, path_to_output_dir, is_task_completion=x)
        make_plots(df, path_to_output_dir, is_task_completion=x)