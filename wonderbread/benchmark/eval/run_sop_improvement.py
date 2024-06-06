# TODO - update
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import sklearn.metrics
import seaborn as sns

from wonderbread.helpers import get_rel_path
sns.set_style("whitegrid", {'axes.grid' : False})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_experimental_results_dir", type=str, default=get_rel_path(__file__, '../../../data/experimental_results'), help="Path to directory containing `all_results.csv` file for experiment", )
    return parser.parse_args()

def make_plots(df: pd.DataFrame, path_to_output_dir: str):
    """Create plots of precision and recall + print out some statistics."""
    # Let's see the mean, median, std of "precision" and "recall"
    with open(os.path.join(path_to_output_dir, "metrics.txt"), "w") as f:
        f.write("Mean spearman_corr: " + str(df["spearman_corr"].mean()) + "\n")
        f.write("Std spearman_corr: " + str(df["spearman_corr"].std()) + "\n")
        f.write("Mean kendall_corr: " + str(df["kendall_corr"].mean()) + "\n")
        f.write("Std kendall_corr: " + str(df["kendall_corr"].std()) + "\n")

    # Histograms of precision and recall
    plt.figure()
    plt.hist(df["spearman_corr"], bins=30, range=(-1, 1))
    plt.title("spearman_corr Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "spearman_corr_hist.pdf"))

    plt.figure()
    plt.hist(df["kendall_corr"], bins=30, range=(-1, 1))
    plt.title("kendall_corr Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "kendall_corr_hist.pdf"))
    
    # Confusion matrix
    plt.figure()
    classes = sorted(df['gt_ranking'].unique().tolist())
    cnf_matrix = sklearn.metrics.confusion_matrix(df['gt_ranking'], df['pred_ranking'])
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=classes)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(path_to_output_dir, "confusion_matrix.pdf"))

    # Scatter plot of precision vs recall
    combos = [
        ("kendall_corr", "spearman_corr"),
    ]
    for (x_col, y_col) in combos:
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(os.path.join(path_to_output_dir, f"{x_col}_vs_{y_col}.pdf"))

def make_tables(df_all_results: pd.DataFrame, path_to_output_dir: str):
    """Create LaTeX tables."""
    # Make table with:
    #   - columns: Spearman, Kendall
    #   - rows: ablation--model
    df = df_all_results.groupby(['ablation--model']).agg({
        'spearman_corr' : 'mean',
        'kendall_corr' : 'mean',
    }).reset_index(drop=False)
    df = df.sort_values(by='spearman_corr', ascending=False)
    df = df.rename(columns={ 'ablation--model': 'Model', 'spearman_corr': 'Spearman Corr.', 'kendall_corr': 'Kendall Tau' })
    df.to_latex(os.path.join(path_to_output_dir, 'sop_ranking.tex'), index=False)

if __name__ == "__main__":
    args = parse_args()
    is_bucket_rankings_to_3: bool = args.is_bucket_rankings_to_3
    path_to_experimental_results_dir: str = args.path_to_experimental_results_dir
    experiment_name: str = 'sop_ranking'
    path_to_output_dir: str = os.path.join(path_to_experimental_results_dir, experiment_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Get paths
    path_to_results_csv = os.path.join(path_to_experimental_results_dir, f"{experiment_name}_all_results.csv")

    # Read in the all_results.csv file
    df_all_results = pd.read_csv(path_to_results_csv)
    
    if is_bucket_rankings_to_3:
        bucket_rankings = {
            1: 1,
            2: 1,
            3: 2,
            4: 3,
            5: 3,
        }
        df_all_results['gt_ranking'] = df_all_results['gt_ranking'].apply(lambda x : bucket_rankings[x])
        df_all_results['pred_ranking'] = df_all_results['pred_ranking'].apply(lambda x : bucket_rankings[x])
        path_to_output_dir = os.path.join(path_to_output_dir, "bucketed_to_3")
        os.makedirs(path_to_output_dir, exist_ok=True)

    # Make tables
    make_tables(df_all_results, path_to_output_dir)

    # Make plots
    make_plots(df_all_results, path_to_output_dir)