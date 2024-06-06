import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Read the CSV files
file1 = 'qna-human-sample-final.csv'
file2 = 'qna-GPT4-sample-final.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

scores = ['completeness', 'soundness', 'clarity', 'compactness']
for score in scores:
    # Extract 'COLUMN TITLE' columns
    SCORE_NAME = score + "_score"
    score1 = df1[SCORE_NAME]
    score2 = df2[SCORE_NAME]

    # Calculate Spearman correlation
    corr, p_value = spearmanr(score1, score2)

    print(f"Spearman correlation between {score} columns: {corr:.4f}")
    print(f"P-value: {p_value:.2e}")

    corr, p_value = pearsonr(score1, score2)
    print(f"Pearson correlation between {score} columns: {corr:.4f}")
    print(f"P-value: {p_value:.2e}")