import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# Set start and end date
start = "2010-01-01"
end   = "2025-07-25"

# --------------------------
# 1. Download SPY & VIX
# --------------------------
data = yf.download(["SPY", "^VIX"], start=start, end=end,
                   auto_adjust=True, progress=False)

# Handle MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    price = pd.DataFrame({
        "SPY": data["Close"]["SPY"],
        "VIX": data["Close"]["^VIX"]
    })
else:
    price = data[["SPY", "^VIX"]].rename(columns={"SPY": "SPY", "^VIX": "VIX"})

# Daily returns
price["Ret"] = price["SPY"].pct_change()

# --------------------------
# 3. Credit & term spreads
# --------------------------
yield_series = ["BAA", "AAA", "DGS2", "DGS10"]
yields = pdr.DataReader(yield_series, "fred", start=start, end=end)
yields = yields.reindex(price.index).ffill()

price["CreditSpread"] = yields["BAA"] - yields["AAA"]
price["TermSpread"]   = yields["DGS10"] - yields["DGS2"]

# --------------------------
# 4. VIX regime flag
# --------------------------
price["VIX_Regime"] = (price["VIX"] > price["VIX"].rolling(50).mean()).astype(int)

# --------------------------
# 5. Volatility, momentum, VIX change
# --------------------------
price["Vol5"]   = price["Ret"].rolling(5).std()
price["Vol10"]  = price["Ret"].rolling(10).std()
price["Vol20"]  = price["Ret"].rolling(20).std()
price["Vol100"] = price["Ret"].rolling(100).std()

price["Mom10"]  = price["SPY"].pct_change(10)
price["Mom20"]  = price["SPY"].pct_change(20)
price["Mom50"]  = price["SPY"].pct_change(50)
price["Mom100"] = price["SPY"].pct_change(100)

price["VIXchg"] = price["VIX"].pct_change(5)

# --------------------------
# 6. RSI14
# --------------------------
delta = price["SPY"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)

roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / (roll_down + 1e-6)  # prevent division by zero

price["RSI14"] = 100 - (100 / (1 + rs))

# --------------------------
# 7. Crash label
# --------------------------
# Label = 1 if future 10-day return < -3%
labels = ((price["SPY"].shift(-10) / price["SPY"] - 1) < -0.03).astype(int)
labels.name = "Crash"

# --------------------------
# 8. Save features + label
# --------------------------
cols = [
    "Ret", "Vol5", "Vol10", "Vol20", "Vol100",
    "Mom10", "Mom20", "Mom50", "Mom100",
    "VIX", "VIXchg", "CreditSpread", "TermSpread", "VIX_Regime", "RSI14"
]

df = pd.concat([price[cols], labels], axis=1).dropna()

# Save to CSV
df.to_csv("datasets/ml_risk_v2.csv")
print("✅ Saved:", len(df), "rows to datasets/ml_risk_v2.csv")
