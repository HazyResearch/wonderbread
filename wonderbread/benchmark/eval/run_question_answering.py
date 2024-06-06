from typing import Dict, List
import pandas as pd
import os
import argparse
import seaborn as sns
import plotly.graph_objects as go

from wonderbread.helpers import get_rel_path
sns.set_style("whitegrid", {'axes.grid' : False})

DEMO_IMPROVEMENT_QUESTION: str = 'Here are two demonstrations, one of which is more efficient than the other. Please describe ways to improve the less optimal workflow.'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_experimental_results_dir", type=str, default=get_rel_path(__file__, '../../../data/experimental_results'), help="Path to directory containing `all_results.csv` file for experiment", )
    return parser.parse_args()

def make_plots(df: pd.DataFrame, path_to_output_dir: str):
    """Create plots of precision and recall + print out some statistics."""
    # Radar plot
    fig = go.Figure()
    categories: List[str] = [ x for x in df.columns if x.endswith('_score') ]
    category_names: List[str] = [ x.replace('_score', '').title() for x in categories ]

    for model in df['ablation--model'].unique():
        df_ = df[df['ablation--model'] == model]
        avg_scores: Dict = df_[categories].mean().to_dict()        
        fig.add_trace(go.Scatterpolar(
            r=[ avg_scores[cat] for cat in categories ],
            theta=category_names,
            fill='toself',
            name=model,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 3]
        )),
        showlegend=True,
    )
    fig.write_image(os.path.join(path_to_output_dir, "question_answering_radar.png"), scale=5)

def make_tables(df: pd.DataFrame, path_to_output_dir: str):
    """Create LaTeX tables."""
    # Make table with:
    #   - columns: Avg. Score
    #   - rows: ablation--model
    df = df_all_results.groupby(['ablation--model']).agg({
        x : 'mean' for x in df_all_results.columns if x.endswith('_score')
    }).reset_index(drop=False)
    df['average'] = df[[x for x in df.columns if x.endswith('_score')]].mean(axis=1)
    df = df.rename(columns={ 'ablation--model': 'Model' } | { x: x.replace('_score', '') for x in df.columns if x.endswith('_score')})
    # Make floats look nice
    for col in df.columns:
        if col != 'Model':
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
    df.to_latex(os.path.join(path_to_output_dir, 'question_answering_scores.tex'), index=False)

if __name__ == "__main__":
    args = parse_args()
    path_to_experimental_results_dir: str = args.path_to_experimental_results_dir
    experiment_name: str = 'question_answering'
    path_to_output_dir: str = os.path.join(path_to_experimental_results_dir, experiment_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Get paths
    path_to_results_csv = os.path.join(path_to_experimental_results_dir, f"{experiment_name}_all_results.csv")

    # Read in the all_results.csv file
    df_all_results = pd.read_csv(path_to_results_csv)
    
    # ! HACK: Need to reverse scoring for radar plots (i.e. change from lower is better to higher is better)
    for col in df_all_results.columns:
        if col.endswith('_score'):
            df_all_results[col] = 3 - df_all_results[col] + 1 # 3 -> 1, 2 -> 2, 1 -> 3
    
    make_tables(df_all_results, path_to_output_dir)
    make_plots(df_all_results, path_to_output_dir)