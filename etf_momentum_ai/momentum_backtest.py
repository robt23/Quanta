import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import matplotlib.dates as mdates
import json
pd.set_option("future.no_silent_downcasting", True)

# Match the *exact* columns your model was trained on (excluding label 'Crash'):
FEATURE_COLS = [
    "Ret",
    "Vol5","Vol10","Vol100","Mom10","Mom20","Mom50","Mom100","VIX","VIXchg",
    "CreditSpread","TermSpread","VIX_Regime","RSI14","Mom20_div_Vol20","Mom20_mul_Vol5",
    "VolRatio","VIX_per_Ret","Mom10_lag1","VIXchg_lag1","VolRatio_lag1",
    "Reg_0","Reg_1","Reg_2"
]

feat_df = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0, nrows=1)
missing = [col for col in FEATURE_COLS if col not in feat_df.columns]
if missing:
    print("Missing features:", missing)
else:
    print("All features present.")


def run_backtest(threshold=None, start="2015-01-01", end="2025-07-25",
                 plot=False, alpha=1.0, core=0.0, beta=0.0, core_2=0.0):
    """
    core  = always-in equity fraction in Regime 1 overlay
    beta  = fraction of safe bucket into TLT (else GLD)
    core2 = always-in equity fraction in Regime 2
    """
        
    if threshold is None:
        threshold = json.load(open("models/crash_threshold.json"))["threshold"]

    # Load model & features
    model = joblib.load("models/lgbm_crash_predictor.joblib")
    feat_df = (
        pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv",
                    index_col=0, parse_dates=True)
          .drop(columns=["Crash"])
    )

    # Price data
    raw = yf.download(["SPY","TLT","GLD","^VIX"], start=start, end=end,
                      auto_adjust=False)
    price = raw["Adj Close"].copy()

    # Join features & drop NA
    price = price.join(feat_df[FEATURE_COLS], how="left").dropna()

    # Crash probability → flag
    price["CrashProb"] = model.predict_proba(
        price[FEATURE_COLS].fillna(0)
    )[:,1]
    price["CrashFlag"] = price["CrashProb"] >= threshold

    # 1) Probabilistic “hedge” weight
    base = (1 - alpha * price["CrashProb"]).clip(0,1)
    # 2) Mix in a core equity stake
    price["EquityWeight"] = core + (1 - core) * base

    # Momentum filter
    price["MA200"] = price["SPY"].rolling(200).mean()

    # 1) Lagged regime flags
    r0 = price["Reg_0"].shift(1).astype(bool)
    r1 = price["Reg_1"].shift(1).astype(bool)
    r2 = price["Reg_2"].shift(1).astype(bool)

    # 2) Base momentum signal
    mom = (price["SPY"] > price["MA200"]).shift(1).astype(float)

    # 3-way routing
    safe_pos = core_2
    raw_pos = np.where(
        r0,
        mom,                                 # Regime 0: pure momentum
        np.where(
            r1,
            price["EquityWeight"].shift(1) * mom,  # Regime 1: overlay
            safe_pos
        )
    )
    price["Position"] = pd.Series(raw_pos, index=price.index).fillna(0)

    # Returns
    price["Return_SPY"] = price["SPY"].pct_change()
    price["Return_TLT"] = price["TLT"].pct_change()
    price["Return_GLD"] = price["GLD"].pct_change()
    w_eq   = price["Position"].shift(1)
    w_safe = 1 - w_eq
    # beta of the safe bucket goes to TLT, rest to GLD
    price["StrategyReturn"] = (
        w_eq * price["Return_SPY"] +
        w_safe * beta * price["Return_TLT"] +
        w_safe * (1-beta) * price["Return_GLD"]
    )

    # Cumulative PnL & metrics
    cum = (1 + price["StrategyReturn"].dropna()).cumprod()
    days = (cum.index[-1] - cum.index[0]).days
    annret = cum.iloc[-1] ** (365.0/days) - 1
    rets = cum.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252)
    maxdd  = ((cum / cum.cummax()) - 1).min()

    if plot:
        return price, (annret, sharpe, maxdd)
    return annret, sharpe, maxdd

if __name__ == "__main__":
    with open("models/best_core_alpha_beta_core2.json") as f:
        best = json.load(f)
    core = best["core"]
    alpha = best["alpha"]
    beta = best["beta"]
    core_2 = best["core2"]

    price, (ann, sr, md) = run_backtest(plot=True, alpha=alpha, core=core, beta=beta, core_2=core_2)

    # Plot 1: SPY w/ flags & TLT shading
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(price.index, price["SPY"], label="SPY", color="blue")
    flags = price.index[price["CrashFlag"]]
    ax.scatter(flags, price.loc[flags,"SPY"], marker="v",
               color="red", label="CrashFlag")
    in_tlt = price["Position"].shift(1) == 0
    ax.fill_between(price.index, price["SPY"].min(),
                    price["SPY"].max(), where=in_tlt,
                    color="orange", alpha=0.1, label="In TLT")
    ax.legend(); ax.set_title("SPY w/ Crash Flags & TLT Periods")
    plt.show()

    # Plot 2: Equity curves
    price["CumulativeBuyHold"] = (1 + price["Return_SPY"]).cumprod()
    price["CumulativeDual"]    = (1 + price["StrategyReturn"]).cumprod()
    plt.figure(figsize=(12,6))
    plt.plot(price.index, price["CumulativeBuyHold"], label="Buy & Hold")
    plt.plot(price.index, price["CumulativeDual"], label="DualMom+Crash")
    plt.legend(); plt.title("Equity Curves"); plt.show()

    # Print metrics
    days = (price.index[-1] - price.index[0]).days
    print(f"BuyHold → AnnRet: {price['CumulativeBuyHold'].iloc[-1]**(365.0/days)-1:.1%}, "
          f"Sharpe: { (price['Return_SPY'].dropna().mean()/price['Return_SPY'].dropna().std()*np.sqrt(252)):.2f}, "
          f"MaxDD: {((price['CumulativeBuyHold']/price['CumulativeBuyHold'].cummax())-1).min():.1%}")
    print(f"DualMom+Crash → AnnRet: {ann:.1%}, Sharpe: {sr:.2f}, MaxDD: {md:.1%}")
