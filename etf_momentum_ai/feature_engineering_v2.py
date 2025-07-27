import pandas as pd

def main():
    # 1) Load base CSV with raw features + Crash label
    df = pd.read_csv("datasets/ml_risk_v2.csv", index_col=0, parse_dates=True)

    # 2) Base features (drop target)
    X = df.drop(columns="Crash").copy()

    # 3) New interactions
    X["Mom20_div_Vol20"] = X["Mom20"] / (X["Vol20"] + 1e-6)
    X["Mom20_mul_Vol5"]  = X["Mom20"] * X["Vol5"]
    X["VolRatio"]        = X["Vol20"] / (X["Vol5"] + 1e-6)
    X["VIX_per_Ret"]     = X["VIX"] / (X["Ret"].abs() + 1e-6)

    # 4) Lagged versions (t-1) for a core set
    lag_feats = ["Vol5", "Vol20", "Mom10", "Mom20", "VIX", "VIXchg", "VolRatio"]
    for feat in lag_feats:
        X[f"{feat}_lag1"] = X[feat].shift(1)

    # 5) Drop rows with NaNs from lagging
    X = X.dropna()

    # 6) Drop the four highly collinear lagged features
    for col in ["Vol5_lag1", "Vol20_lag1", "Mom20_lag1", "VIX_lag1", "Vol20"]:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)

    # 7) Reattach the target
    y = df["Crash"].loc[X.index]

    # 8) Save enhanced dataset
    out = pd.concat([X, y.rename("Crash")], axis=1)
    out.to_csv("datasets/ml_risk_v2_enhanced.csv")
    print(f"Features: {X.shape[1]}, Rows: {len(X)}")

if __name__ == "__main__":
    main()
