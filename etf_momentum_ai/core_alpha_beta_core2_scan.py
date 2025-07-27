# core_alpha_beta_core2_scan.py
import pandas as pd
import json
from momentum_backtest import run_backtest

# Load best threshold
with open("models/crash_threshold.json") as f:
    best_thr = json.load(f)["threshold"]

results = []
for core in [0.0, 0.25, 0.5]:
    for alpha in [0.5, 1.0, 1.5]:
        for beta in [0.0, 0.5, 1.0]:
            for core2 in [0.0, 0.1, 0.2, 0.3]:
                ann, sr, md = run_backtest(
                    threshold=best_thr,
                    alpha=alpha,
                    core=core,
                    beta=beta,
                    core_2=core2
                )
                results.append({
                    "core": core, "alpha": alpha, "beta": beta, "core2": core2,
                    "AnnRet": ann, "Sharpe": sr, "MaxDD": md
                })

df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
print(df.head(10))
df.to_csv("results/scan_core_alpha_beta_core2.csv", index=False)

# Save best
best = df.iloc[0].to_dict()
with open("models/best_core_alpha_beta_core2.json", "w") as f:
    json.dump(best, f, indent=2)
print("Saved best core/alpha/beta/core2 result to models/best_core_alpha_beta_core2.json")
