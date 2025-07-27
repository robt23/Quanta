import json
import pandas as pd
from momentum_backtest import run_backtest

# 1) Sweep thresholds from 1% to 50%
results = []
for th in [i/100 for i in range(1, 51)]:
    annret, sharpe, maxdd = run_backtest(threshold=th)
    results.append({"threshold": th, "AnnRet": annret, "Sharpe": sharpe, "MaxDD": maxdd})

# 2) Collect into DataFrame
df = pd.DataFrame(results)

# 3) Show top 5 by Sharpe and save new threshold
def show_and_persist(df):
    top5 = df.sort_values("Sharpe", ascending=False).head(5)
    print("Top 5 thresholds by Sharpe:\n", top5.to_string(index=False))
    best = top5.iloc[0]
    best_thr = best["threshold"]
    print(f"\nChosen threshold: {best_thr:.3f} (Sharpe={best['Sharpe']:.3f}, AnnRet={best['AnnRet']:.3%}, MaxDD={best['MaxDD']:.3%})")
    with open("models/crash_threshold.json", "w") as f:
        json.dump({"threshold": best_thr}, f)
    print("\nSaved new threshold to models/crash_threshold.json")

if __name__ == '__main__':
    show_and_persist(df)
    df.to_csv("results/threshold_scan.csv", index=False)
    print("Full scan results written to results/threshold_scan.csv")
