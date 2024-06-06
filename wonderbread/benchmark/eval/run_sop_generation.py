from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from wonderbread.helpers import get_rel_path
from wonderbread.benchmark.tasks.documentation.sop_generation.eval import add_gold_sops, evaluate_sops
import argparse
import seaborn as sns

sns.set_style("whitegrid", {"axes.grid": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_experimental_results_dir", type=str, default=get_rel_path(__file__, '../../../data/experimental_results'), help="Path to directory containing `all_results.csv` file for experiment", )
    parser.add_argument( "--path_to_data_dir", type=str, default=get_rel_path(__file__, "../../../data/gold_demos"), help="Path to directory containing all raw demos", )
    return parser.parse_args()


def make_tables(df_eval: pd.DataFrame, path_to_output_dir: str):
    """Create LaTeX tables of the results."""
    # Rows: ablation
    # Columns: mean precision, mean recall, mean ordering
    df_grouped = (
        df_eval.groupby(
            [
                "ablation",
            ]
        )
        .agg(
            {
                "ablation_human": "first",
                "precision": "mean",
                "recall": "mean",
                "f1": "mean",
                "ordering": "mean",
                "n_lines_pred_sop": "mean",
                "n_lines_gold_sop": "mean",
            }
        )
        .reset_index(drop=False)
    )
    # Make floats look nice
    df_grouped["precision"] = df_grouped["precision"].apply(lambda x: f"{x:.2f}")
    df_grouped["recall"] = df_grouped["recall"].apply(lambda x: f"{x:.2f}")
    df_grouped["f1"] = df_grouped["f1"].apply(lambda x: f"{x:.2f}")
    df_grouped["ordering"] = df_grouped["ordering"].apply(lambda x: f"{x:.2f}")
    df_grouped["n_lines_pred_sop"] = df_grouped["n_lines_pred_sop"].apply(
        lambda x: f"{x:.2f}"
    )
    df_grouped["n_lines_gold_sop"] = df_grouped["n_lines_gold_sop"].apply(
        lambda x: f"{x:.2f}"
    )
    df_grouped = df_grouped.drop(columns=["ablation"])
    df_grouped.to_latex(os.path.join(path_to_output_dir, "results.tex"), index=False)


def make_plots(df_eval: pd.DataFrame, path_to_output_dir: str):
    """Create plots of precision and recall + print out some statistics."""
    # Let's see the mean, median, std of "precision" and "recall"
    with open(os.path.join(path_to_output_dir, "metrics.txt"), "w") as f:
        f.write("Mean Precision: " + str(df_eval["precision"].mean()) + "\n")
        f.write("Std Precision: " + str(df_eval["precision"].std()) + "\n")
        f.write("Mean Recall: " + str(df_eval["recall"].mean()) + "\n")
        f.write("Std Recall: " + str(df_eval["recall"].std()) + "\n")
        f.write("Mean F1: " + str(df_eval["f1"].mean()) + "\n")
        f.write("Std F1: " + str(df_eval["f1"].std()) + "\n")
        f.write("Mean Ordering: " + str(df_eval["ordering"].mean()) + "\n")
        f.write("Std Ordering: " + str(df_eval["ordering"].std()) + "\n")

    # Histograms of precision and recall
    plt.figure()
    plt.hist(df_eval["precision"], bins=30, range=(0, 1))
    plt.title("Precision Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "precision_hist.png"))

    plt.figure()
    plt.hist(df_eval["recall"], bins=30, range=(0, 1))
    plt.title("Recall Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "recall_hist.png"))

    plt.figure()
    plt.hist(df_eval["f1"], bins=30, range=(0, 1))
    plt.title("F1 Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "f1_hist.png"))

    plt.figure()
    plt.hist(df_eval["ordering"], bins=30, range=(0, 1))
    plt.title("Ordering Histogram")
    plt.savefig(os.path.join(path_to_output_dir, "ordering_hist.png"))

    # Scatter plot of precision vs recall
    def make_scatter(df: pd.DataFrame, x_col: str, y_col: str, prefix: str = ""):
        ablations = sorted(df["ablation"].unique().tolist())
        fig, ax = plt.subplots()
        for ablation in ablations:
            df_ablation = df[df["ablation"] == ablation]
            label = df_ablation["ablation_human"].iloc[0]
            model = df_ablation["ablation--model"].iloc[0]
            ablations_for_model = (
                df[df["ablation--model"] == model]["ablation"].unique().tolist()
            )
            if model == "GPT4":
                colors = ["darkgreen", "limegreen", "lime"]
            elif model == "GeminiPro":
                colors = [
                    "orangered",
                    "lightcoral",
                    "red",
                ]
            elif model == "Claude3":
                colors = [
                    "cornflowerblue",
                    "navy",
                    "blue",
                ]
            else:
                colors = cm.rainbow(np.linspace(0, 1, len(ablations_for_model)))
            plt.scatter(
                df_ablation[x_col],
                df_ablation[y_col],
                c=colors[ablations_for_model.index(ablation)],
                label=label,
                s=12,
            )
        plt.xlabel(x_col.title())
        plt.ylabel(y_col.title())
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 7})
        plt.savefig(os.path.join(path_to_output_dir, f"{prefix}{x_col}_vs_{y_col}.pdf"))

    combos = [
        ("precision", "recall"),
        ("n_lines_pred_sop", "precision"),
        ("n_lines_pred_sop", "recall"),
        ("n_lines_gold_sop", "precision"),
        ("n_lines_gold_sop", "recall"),
    ]
    for x_col, y_col in combos:
        make_scatter(df_eval, x_col, y_col)
    # Also make one per model
    for model in df_eval["ablation--model"].unique():
        make_scatter(
            df_eval[df_eval["ablation--model"] == model],
            "precision",
            "recall",
            prefix=f"{model} - ",
        )


def gen_ablation_name(row: pd.Series) -> str:
    human_readable_ablation = (
        row["ablation--model"]
        + " - "
        + "+".join(
            (["TD"] if row["ablation--is_td"] else [])
            + (["KF"] if row["ablation--is_kf"] else [])
            + (["ACT"] if row["ablation--is_act"] else [])
        )
        + (" (pairwise)" if row["ablation--is_pairwise"] else "")
    )
    return human_readable_ablation


if __name__ == "__main__":
    args = parse_args()
    path_to_data_dir: str = args.path_to_data_dir
    path_to_experimental_results_dir: str = args.path_to_experimental_results_dir
    experiment_name: str = "sop_generation"
    path_to_output_dir: str = os.path.join(
        path_to_experimental_results_dir, experiment_name
    )
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Get paths
    path_to_results_csv = os.path.join(
        path_to_experimental_results_dir, f"{experiment_name}_all_results.csv"
    )
    path_to_eval_csv = os.path.join(path_to_output_dir, f"{experiment_name}_eval.csv")

    # Read in the all_results.csv file
    df_orig = pd.read_csv(path_to_results_csv)
    # Rename "sop" to "predicted_sop"
    df_orig.rename(columns={"sop": "pred_sop"}, inplace=True)
    # Add gold sops
    df_orig = add_gold_sops(df_orig, path_to_data_dir)

    # Skip evaluations that have already been done
    df_todos = []
    df_eval = (
        pd.read_csv(path_to_eval_csv)
        if os.path.exists(path_to_eval_csv)
        else pd.DataFrame()
    )

    if df_eval.shape[0] > 0:
        df_eval["demo_name|ablation"] = df_eval["demo_name"] + "|" + df_eval["ablation"]
        done_demos: set = set(df_eval["demo_name|ablation"].tolist())
        for idx, row in df_orig.iterrows():
            if (row["demo_name"] + "|" + row["ablation"]) not in done_demos:
                df_todos.append(row)
        print(
            f"Doing {len(df_todos)} / {df_orig.shape[0]} evals that have NOT already been done."
        )
        df_todos = pd.DataFrame(df_todos)
    else:
        df_todos = df_orig

    # Run the evaluation
    list_of_sop_pairs = df_todos.to_dict(orient="records")
    df_todos_eval = evaluate_sops(list_of_sop_pairs, experiment_name=experiment_name)
    df_eval = pd.concat([df_eval, df_todos_eval], axis=0)
    # Reorder the columns to be in same order
    df_eval = df_eval[
        df_orig.columns.tolist()
        + sorted(list(set(df_eval.columns) - set(df_orig.columns)))
    ]
    df_eval.to_csv(
        os.path.join(path_to_output_dir, f"{experiment_name}_eval.csv"), index=False
    )

    # Read in the eval.csv file
    df_eval = pd.read_csv(
        os.path.join(path_to_output_dir, f"{experiment_name}_eval.csv")
    )
    df_eval["f1"] = (
        2
        * (df_eval["precision"] * df_eval["recall"])
        / (df_eval["precision"] + df_eval["recall"])
    )

    # ! HACK: Remove (pairwise) ablations
    df_eval = df_eval[~df_eval["ablation--is_pairwise"]]

    # Rename ablations to human friendly names
    df_eval["ablation_human"] = df_eval.apply(gen_ablation_name, axis=1)

    # Plot metrics
    make_tables(df_eval, path_to_output_dir)
    make_plots(df_eval, path_to_output_dir)
