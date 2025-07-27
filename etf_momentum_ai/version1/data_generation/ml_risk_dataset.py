# ml_risk_dataset.py

import yfinance as yf
import pandas as pd
import numpy as np

# 1. Download price + VIX
tickers = ["SPY", "^VIX"]
start   = "2010-01-01"
end     = "2025-07-25"

raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
df  = raw["Adj Close"].copy()  # only need adjusted close

# 2. Feature: 20-day historical volatility of SPY
#    - pct_change gives daily returns; rolling.std over 20 days gives vol
df["Vol20"] = df["SPY"].pct_change().rolling(window=20).std()

# 3. Feature: 20-day momentum of SPY
#    - pct_change(20) is return over past 20 trading days
df["Mom20"] = df["SPY"].pct_change(20)

# 4. Feature: current VIX level (proxy for option-implied volatility)
df["VIX"] = df["^VIX"]

# 5. Label: will SPY drop >5% in the NEXT 21 days?
#    a) future return over 21 days: shift(-21) so today’s row looks at future
df["FutureRet21"] = df["SPY"].pct_change(21).shift(-21)
#    b) label = 1 if future return < -0.05 (i.e. >5% drop), else 0
df["Label"] = (df["FutureRet21"] < -0.05).astype(int)

# 6. Drop rows with any NaN (first 20 days for Vol20/Mom20, last 21 for future label)
dataset = df[["Vol20", "Mom20", "VIX", "Label"]].dropna()

# 7. Save to CSV for easy inspection
dataset.to_csv("datasets/risk_dataset.csv")
print("Dataset shape:", dataset.shape)
print(dataset.head())
