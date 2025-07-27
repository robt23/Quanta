import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np

# 1) Load model & data
model = joblib.load("models/lgbm_crash_predictor.joblib")
df    = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0)

# 2) Split X/y
X = df.drop(columns=["Crash"])
y = df["Crash"]

# 3) Compute permutation importances (this may take ~1–2 minutes)
print("Computing permutation importances—this can be slow...")
r = permutation_importance(
    model, X, y,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

# 4) Build a sorted Series of mean importances
importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)

# 5) Show top 15
print("\nTop 15 features by permutation importance (ROC-AUC drop):")
print(importances.head(15).to_string())

# 6) Save full list if you want
importances.to_csv("results/perm_feature_importances.csv", header=["importance_mean"])

# 7) Find high correlations
corr = X.corr().abs()
high_corr = np.where((corr > 0.9) & (corr < 1.0))
pairs = [(X.columns[i], X.columns[j], corr.iloc[i, j])
         for i, j in zip(*high_corr) if i < j]

print("\nHighly correlated feature pairs (corr > 0.90):")
for f1, f2, val in pairs:
    print(f"{f1:20s} | {f2:20s} | {val:.2f}")
