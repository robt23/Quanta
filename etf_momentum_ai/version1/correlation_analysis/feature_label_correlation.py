# feature_engineering.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the original ML dataset
df = pd.read_csv("datasets/ml_risk_dataset.csv", index_col=0, parse_dates=True)

# 2. Compute correlations with the crash Label
corr_with_label = df.corr()["Label"].drop("Label").sort_values()
print("Feature → Label correlations:\n", corr_with_label)

# 3. Split into kept vs. dropped based on |corr| < 0.05
threshold = 0.05
to_drop = corr_with_label[corr_with_label.abs() < threshold].index.tolist()
kept    = corr_with_label[corr_with_label.abs() >= threshold].index.tolist()
print(f"\n▶ Dropping low‐corr features (|corr| < {threshold}): {to_drop}")
print(f"▶ Keeping these features: {kept}")

# 4. Plot distributions of KEPT features
for feature in kept:
    if feature == "Label":
        continue
    plt.figure(figsize=(6,3))
    sns.kdeplot(
        data=df, x=feature,
        hue="Label",
        palette={0:"blue",1:"red"},
        common_norm=False, fill=True, alpha=0.4
    )
    plt.title(f"[KEPT]     {feature}: crash(red) vs no-crash(blue)")
    plt.show()

# 5. Plot distributions of DROPPED features
for feature in to_drop:
    plt.figure(figsize=(6,3))
    sns.kdeplot(
        data=df, x=feature,
        hue="Label",
        palette={0:"blue",1:"red"},
        common_norm=False, fill=True, alpha=0.4
    )
    plt.title(f"[DROPPED] {feature}: crash(red) vs no-crash(blue)")
    plt.show()

# 6. Create pruned copy and add interactions on KEPT only
df_pruned = df[kept].copy()  # this contains Label + the kept originals

# 6a. Example interaction & nonlinear features
df_pruned["Vol20_div_Vol5"] = df_pruned["Vol20"] / (df_pruned["Vol5"] + 1e-6)
df_pruned["VIX_mul_Vol5"]   = df_pruned["VIX"]   *  df_pruned["Vol5"]
df_pruned["Vol20_sq"]       = df_pruned["Vol20"] ** 2
df_pruned["Vol5_sq"]        = df_pruned["Vol5"]  ** 2

# 7. Save the enhanced dataset
out_path = "datasets/ml_risk_dataset_enhanced.csv"
df_pruned.to_csv(out_path)
print(f"Saved enhanced data with {df_pruned.shape[1]-1} features → {out_path}")
