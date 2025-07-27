# regime_feature.py

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load your enhanced dataset (with engineered features)
df = pd.read_csv("datasets/ml_risk_dataset_enhanced.csv",
                 index_col=0, parse_dates=True)

# 2. Choose a small set of regime descriptors
#    Here, 20-day vol and VIX → regimes of calm vs. stressed
X_regime = df[["Vol20", "VIX"]]

# 3. Cluster into k=3 regimes (bullish/calm, sideways, stressed)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Regime"] = kmeans.fit_predict(X_regime)

# 4. Visualize the clusters
plt.figure(figsize=(6,6))
for r, color in zip(range(3), ["green","orange","red"]):
    subset = df[df["Regime"]==r]
    plt.scatter(subset["Vol20"], subset["VIX"],
                s=10, c=color, label=f"Regime {r}", alpha=0.3)
plt.xlabel("Vol20")
plt.ylabel("VIX")
plt.legend()
plt.title("Market Regimes via KMeans")
plt.show()

# 5. One-hot encode and save
df = pd.get_dummies(df, columns=["Regime"], prefix="Regime")
df.to_csv("datasets/ml_risk_dataset_regime.csv")
print("Added Regime one-hots → ml_risk_dataset_regime.csv")
