from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import sklearn.metrics
import numpy as np
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from wonderbread.helpers import get_rel_path
sns.set_style("whitegrid", {'axes.grid' : False})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_experimental_results_dir", type=str, default=get_rel_path(__file__, '../../../data/experimental_results'), help="Path to directory containing `all_results.csv` file for experiment", )
    return parser.parse_args()

def get_optimal_label_assignment(pred_labels: List[str], gt_labels: List[str]) -> Dict[str, str]:
    """Given two lists of labels, find the optimal mapping between the two lists."""
    pred_labels = np.array(pred_labels).astype(str)
    gt_labels = np.array(gt_labels).astype(str)
    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)
    
    # Add dummy labels to handle the case where the number of unique labels in the prediction is less than the number of unique labels in the ground truth
    if len(pred_unique) < len(gt_unique):
        pred_unique = np.concatenate([pred_unique, [f"dummy_{i}" for i in range(len(gt_unique) - len(pred_unique))]])
    if len(gt_unique) < len(pred_unique):
        gt_unique = np.concatenate([gt_unique, [f"dummy_{i}" for i in range(len(pred_unique) - len(gt_unique))]])
    
    # Create a cost matrix where each element represents the number of mismatches
    cost_matrix = np.zeros((len(pred_unique), len(gt_unique)))
    for i, pred in enumerate(pred_unique):
        for j, gt in enumerate(gt_unique):
            count_mismatches = np.sum((pred_labels == pred) & (gt_labels != gt))
            cost_matrix[i, j] = count_mismatches
    
    # Use the Hungarian algorithm to find the optimal mapping
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Generate the optimal mapping
    mapping: Dict[str, str] = {}
    for i, j in zip(row_indices, col_indices):
        mapping[str(pred_unique[i])] = str(gt_unique[j])
    
    return mapping

def make_plots(df: pd.DataFrame, path_to_output_dir: str):
    """Create plots of precision and recall + print out some statistics."""
    # Let's see the mean, median, std of "precision" and "recall"
    with open(os.path.join(path_to_output_dir, "metrics.txt"), "w") as f:
        f.write(f"Mean adjusted_rand: {df['adjusted_rand'].mean()}\n")
        f.write(f"Std adjusted_rand: {df['adjusted_rand'].std()}\n")
        f.write(f"Mean v_measure: {df['v_measure'].mean()}\n")
        f.write(f"Std v_measure: {df['v_measure'].std()}\n")

    # Histograms of precision and recall
    plt.figure()
    plt.hist(df["adjusted_rand"], bins=30, range=(0, 1))
    plt.title("adjusted_rand Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "adjusted_rand_hist.pdf"))
    plt.close()

    plt.figure()
    plt.hist(df["v_measure"], bins=30, range=(0, 1))
    plt.title("v_measure Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "v_measure_hist.pdf"))
    plt.close()
        
    # Each line is a different demo, green if correct, red if incorrect
    fig, ax = plt.subplots(figsize=(25, 15))
    # Create a list of dicts, where each inner dicts contains (a) a list of 1s and 0s indicating if the prediction was correct at that timestep, (b) ground truth task_ids for each timestep
    is_correct_data: List[Dict[str, List[int]]] = []
    for __, row in df.iterrows():
        is_corrects, gt_task_ids = [], []
        pred_2_gt: Dict[str, str] = get_optimal_label_assignment(row['pred_task_id'], row['gt_task_id'])
        for idx in range(max(len(row['pred_task_id']), len(row['gt_task_id']))):
            pred = row['pred_task_id'][idx] if idx < len(row['pred_task_id']) else "Z"
            gt = row['gt_task_id'][idx] if idx < len(row['gt_task_id']) else ""
            is_corrects.append(1 if pred_2_gt[str(pred)] == str(gt) else 0)
            gt_task_ids.append(gt)
        is_correct_data.append({
            'is_correct' : is_corrects,
            'gt_task_id' : gt_task_ids,
            'len' : len(is_corrects),
            'transition_idxs' : [],
        })
    is_correct_data = sorted(is_correct_data, key= lambda x: x['len'], reverse=True)
    # Right pad all lists that aren't max length with -1 so they show up as white
    max_inner_len: int = max([ x['len'] for x in is_correct_data ])
    for x_idx, x in enumerate(is_correct_data):
        if x['len'] < max_inner_len:
            is_correct_data[x_idx]['is_correct'] = x['is_correct'] + [-1] * (max_inner_len - x['len'])
            is_correct_data[x_idx]['gt_task_id'] = x['gt_task_id'] + [""] * (max_inner_len - x['len'])
    # Create a colormap with white for -1, green for 1, and red for 0
    cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green'])
    # Plot the data
    plt.imshow([ x['is_correct'] for x in is_correct_data ], 
               cmap=cmap,
               aspect='auto', 
               interpolation='nearest', 
               extent=(0, max_inner_len, 0, len(is_correct_data))) # set plot boundaries to (0, max) instead of (0.5, max-0.5)
    # Mark transition points between gt_task_id's with black vertical lines
    for i in range(len(is_correct_data)):
        for j in range(len(is_correct_data[i]['gt_task_id'])):
            if j != 0 and is_correct_data[i]['gt_task_id'][j] != is_correct_data[i]['gt_task_id'][j-1]: # draw line if gt_task_id changes
                if is_correct_data[i]['gt_task_id'][j] != "": # ignore drawing lines for empty gt (i.e. when prediction is longer than gt)
                    plt.plot([j, j], [len(is_correct_data) - i, len(is_correct_data) - i+1], color='black', linewidth=2)
                    is_correct_data[i]['transition_idxs'].append(j)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Demonstration", fontsize=16, rotation=0, labelpad=80)
    plt.xlabel("Timestep", fontsize=16)
    # Create dummy scatter plots for the legend
    correct_legend = ax.scatter([], [], color='green', label='Correct')
    incorrect_legend = ax.scatter([], [], color='red', label='Incorrect')
    transition_legend = ax.scatter([], [], color='black', label='Transition')
    ax.legend(handles=[correct_legend, incorrect_legend, transition_legend], loc='lower right', fontsize=16)
    plt.savefig(os.path.join(path_to_output_dir, f"is_correct_line_plot.pdf"))
    plt.close()
    
    # Histogram of distance between transition points and errors
    distances: List[int] = []
    for i in range(len(is_correct_data)):
        # For each entry of `0` in is_correct, find the closest idx in `transition_idxs`
        error_idxs = [ idx for idx, x in enumerate(is_correct_data[i]['is_correct']) if x == 0 ]
        if len(is_correct_data[i]['transition_idxs']) == 0: continue
        for error_idx in error_idxs:
            closest_transition_idx: int = min(is_correct_data[i]['transition_idxs'], key=lambda x: abs(x - error_idx))
            distances.append(abs(closest_transition_idx - error_idx))
    plt.figure()
    plt.hist(distances, bins=30)
    plt.ylabel("Count of Incorrect Cluster Assignments")
    plt.xlabel("Distance from Nearest Transition")
    plt.savefig(os.path.join(path_to_output_dir, "hist_distance_to_transition.pdf"))
    plt.close()
    
    # Scatter plot of v-measure v. adjusted_rand
    combos = [
        ("v_measure", "adjusted_rand"),
    ]
    for (x_col, y_col) in combos:
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(os.path.join(path_to_output_dir, f"{x_col}_vs_{y_col}.pdf"))
        plt.close()

def make_tables(df: pd.DataFrame, path_to_output_dir: str):
    """Create LaTeX tables."""
    # Make table with:
    #   - columns: ARI, V-Measure
    #   - rows: ablation
    df = df.groupby(['ablation']).agg({
        'v_measure' : 'mean',
        'adjusted_rand' : 'mean',
        'ablation_human' : 'first',
        'ablation--n_tasks' : 'first',
        'ablation--model' : 'count',
    }).reset_index(drop=False)
    df = df.sort_values(by=['ablation--n_tasks', 'ablation'], ascending=False)
    df = df[['ablation_human', 'ablation--n_tasks', 'adjusted_rand', 'v_measure', 'ablation--model']]
    df = df.rename(columns={ 'ablation_human': 'Ablation', 'ablation--n_tasks' : '# of Tasks', 'adjusted_rand': 'ARI', 'v_measure': 'V-Measure', 'ablation--model' : 'Count' })
    # Make floats pretty
    df['ARI'] = df['ARI'].apply(lambda x: f"{x:.2f}")
    df['V-Measure'] = df['V-Measure'].apply(lambda x: f"{x:.2f}")
    df.to_latex(os.path.join(path_to_output_dir, 'demo_segmentation.tex'), index=False)

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
    experiment_name: str = 'demo_segmentation'
    path_to_output_dir: str = os.path.join(path_to_experimental_results_dir, experiment_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Get paths
    path_to_results_csv = os.path.join(path_to_experimental_results_dir, f"{experiment_name}_all_results.csv")

    # Read in the all_results.csv file
    df_all_results = pd.read_csv(path_to_results_csv)
    
    df_all_results = df_all_results.groupby(['demo_name', 'ablation', 'trial']).agg({
        'pred_task_id' : list,
        'gt_task_id' : list,
    } | {
        col : 'first' for col in df_all_results.columns if col.startswith('ablation--')
    }).reset_index(drop=False)
    
    # Calculate metrics
    accuracies, v_measures, adj_rand_scores = [], [], []
    df_all_results['v_measure'] = df_all_results.apply(lambda row: sklearn.metrics.cluster.v_measure_score(row['gt_task_id'], [ x if not pd.isna(x) else 'Z' for x in row['pred_task_id'] ]), axis=1)
    df_all_results['adjusted_rand'] = df_all_results.apply(lambda row: sklearn.metrics.cluster.adjusted_rand_score(row['gt_task_id'], [ x if not pd.isna(x) else 'Z' for x in row['pred_task_id'] ]), axis=1)

    # Calculate mean results across trials
    print(df_all_results.groupby(['trial']).agg({
        'v_measure' : 'mean',
        'adjusted_rand' : 'mean',
    }))
    
    # Rename ablations to human friendly names
    ablation_2_human: Dict[str, str] = {}
    df_all_results['ablation_human'] = df_all_results.apply(gen_ablation_name, axis=1)

    # Make tables
    make_tables(df_all_results, path_to_output_dir)
    for n_tasks in df_all_results['ablation--n_tasks'].unique():
        df = df_all_results[df_all_results['ablation--n_tasks'] == n_tasks]
        path_to_output_dir_for_n_task = os.path.join(path_to_output_dir, 'n_tasks=' + str(n_tasks))
        os.makedirs(path_to_output_dir_for_n_task, exist_ok=True)
        
        # Make plots
        for ablation in df['ablation'].unique():
            path_to_output_dir_ = os.path.join(path_to_output_dir_for_n_task, ablation)
            os.makedirs(path_to_output_dir_, exist_ok=True)
            df_ = df[df['ablation'] == ablation].copy()
            make_plots(df_, path_to_output_dir_)