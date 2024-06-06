import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    print(score2)
    # Create 2D density plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=score1, y=score2, cmap="Blues", shade=True, shade_lowest=False, label='Density')
    plt.plot(score1, score1, c='red', linestyle='--', label='x=y')
    plt.xlabel('Human Score')
    plt.ylabel('Model Score')
    plt.xlim(0.5,3.5)
    plt.ylim(0.5,3.5)
    plt.title(f'2D Density Plot of {score} Scores')
    plt.legend()
    plt.grid(True)

    # Save the figure
    # plt.savefig(f'{score}_density_plot.png')
    # # Close the plot to free up memory
    # plt.close()
    # plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV files
# file1 = 'qna-human-sample-final.csv'
# file2 = 'qna-GPT4-sample-final.csv'

# df1 = pd.read_csv(file1)
# df2 = pd.read_csv(file2)

# scores = ['completeness', 'soundness', 'clarity', 'compactness']
# for score in scores:
#     # Extract 'COLUMN TITLE' columns
#     SCORE_NAME = score + "_score"
#     score1 = df1[SCORE_NAME]
#     score2 = df2[SCORE_NAME]

#     # Create scatter plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(score1, score2, c='blue', alpha=0.6, label='Data Points')
#     plt.plot(score1, score1, c='red', linestyle='--', label='y=x')
#     plt.xlabel('Human Score')
#     plt.ylabel('Model Score')
#     plt.title(f'Scatter Plot of {score} Scores')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
