# data_prep.py

import pandas as pd
import yfinance as yf
import os

# 1) Download historical SPY & VIX data
#    - SPY: our equity index ETF
#    - ^VIX: CBOE Volatility Index (implied vol)
#    We fetch from 2010-01-01 up to yesterday (replace end date as needed).
data = yf.download(
    tickers=["SPY", "^VIX"],
    start="2010-01-01",
    end="2025-07-25",
    auto_adjust=True,       # use adjusted closes (splits/dividends)
    progress=False
)

# 2) Build a simple DataFrame of just the two series
price = pd.DataFrame({
    "SPY_Close": data["Close"]["SPY"],
    "VIX":       data["Close"]["^VIX"]
})

# 3) Compute daily SPY returns for volatility calculations
price["Ret"] = price["SPY_Close"].pct_change()

# 4) Realized volatility over 3 horizons
price["Vol5"]  = price["Ret"].rolling(window=5).std()     # 1‐week vol
price["Vol20"] = price["Ret"].rolling(window=20).std()    # 1‐month vol
price["Vol60"] = price["Ret"].rolling(window=60).std()    # 3‐month vol

# 5) Price momentum over 3 horizons
price["Mom10"] = price["SPY_Close"].pct_change(periods=10)  # 2‐week return
price["Mom20"] = price["SPY_Close"].pct_change(periods=20)  # 1‐month return
price["Mom60"] = price["SPY_Close"].pct_change(periods=60)  # 3‐month return

# 6) VIX percent-change over last 5 days (vol momentum)
price["VIXchg"] = price["VIX"].pct_change(periods=5)

# 7) Build features DataFrame and drop any NaNs from rolling windows
features = price[[
    "Vol5","Vol20","Vol60",
    "Mom10","Mom20","Mom60",
    "VIX","VIXchg"
]].dropna()

# 8) Create the crash label:
#    1 if SPY falls >5% over the next 21 trading days, else 0
labels = (
    (price["SPY_Close"].shift(-21) / price["SPY_Close"] - 1) < -0.05
).astype(int)
labels = labels.loc[features.index]  # align with features

# 9) Combine into one DataFrame and drop any leftover NaNs
df_new = pd.concat([features, labels.rename("Label")], axis=1).dropna()

# 10) Save to CSV for your ML model
df_new.to_csv("datasets/ml_risk_dataset.csv")
print(f"Generated dataset with {len(df_new)} rows and {df_new.shape[1]-1} features → datasets/ml_risk_dataset.csv")
