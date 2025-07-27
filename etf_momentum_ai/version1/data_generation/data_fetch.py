# data_fetch.py
import yfinance as yf
import pandas as pd

# 1. Define tickers & date range
tickers = ["SPY", "TLT", "^VIX"]
start   = "2015-01-01"
end     = "2025-07-25"

# 2. Download Adjusted Close prices
raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
data = raw["Adj Close"]

# 3. Show first/last rows
print("=== Head ===")
print(data.head(), "\n")
print("=== Tail ===")
print(data.tail(), "\n")

# 4. Quick summary stats
print("=== Summary Statistics ===")
print(data.describe())
