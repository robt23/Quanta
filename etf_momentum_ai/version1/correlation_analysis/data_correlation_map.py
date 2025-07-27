import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset you just created
df = pd.read_csv("datasets/ml_risk_dataset.csv", index_col=0, parse_dates=True)

# 2. Drop the Label column, since we only want feature–feature correlations
feat_df = df.drop(columns=["Label"])

# 3. Compute the correlation matrix (Pearson by default)
corr = feat_df.corr()

# 4. Plot a heatmap of those correlations
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,        # show the numeric value in each cell
    fmt=".2f",         # two decimal places
    cmap="coolwarm",   # blue = negative corr, red = positive corr
    linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.show()
