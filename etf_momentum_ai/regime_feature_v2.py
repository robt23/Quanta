import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1) Read the fully enhanced CSV (includes macro/breadth)
df = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0, parse_dates=True)

# 2) Select features to cluster on
#    e.g. all continuous features (drop Crash)
features = df.drop(columns="Crash")

# 3) PCA → 2 dims, then KMeans(n_clusters=3)
Z = PCA(n_components=2).fit_transform(features)
labels = KMeans(n_clusters=3, random_state=42).fit_predict(Z)
df["Regime"] = labels

# 4) Remove any existing Reg_ columns before one-hot encoding
df = df.loc[:, ~df.columns.str.startswith("Reg_")]
df = pd.get_dummies(df, columns=["Regime"], prefix="Reg", drop_first=False)

# 5) Save
df.to_csv("datasets/ml_risk_v2_enhanced_regime.csv")
print("Regime clusters sizes:", pd.Series(labels).value_counts().to_dict())
