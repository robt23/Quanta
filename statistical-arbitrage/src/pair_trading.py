import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cointegration_filter(price_df):

    # Define sectors and tickers
    sectors = {
        'Consumer Discretionary': ["XLY", "IYC", "VCR", "XHB"],
        'Consumer Staples':       ["XLP", "IYK", "VDC"],
        'Energy':                 ["XLE", "IYE", "VDE", "XOP", "AMLP", "OIH"],
        'Financials':             ["XLF", "IYF", "VFH", "KBE", "KRE"],
        'Health Care':            ["XLV", "IYH", "IBB", "XBI", "VHT"],
        'Industrials':            ["XLI", "IYJ", "VIS"],
        'Materials':              ["XLB", "IYM", "VAW", "GDX", "GDXJ"],
        'Information Technology': ["XLK", "IYW", "VGT", "FDN", "IGV"],
        'Communication Services': ["IYZ", "VOX", "XLC"],
        'Utilities':              ["XLU", "IDU", "VPU"],
        'Real Estate':            ["RWR", "XLRE", "VNQ"]
    }

    # Run Engle–Granger cointegration test on every pair within each sector
    results = []
    for sector, tickers in sectors.items():
        # build the column names for close prices
        close_cols = [f"Close_{t}" for t in tickers if f"Close_{t}" in price_df.columns]
        
        # Test all combinations of two ETFs in this sector
        for col1, col2 in combinations(close_cols, 2):
            data = price_df[[col1, col2]].dropna()
            if len(data) < 30:
                # skip pairs with too few overlapping points
                continue
            
            score, pvalue, _ = coint(data[col1], data[col2])
            results.append({
                'sector':   sector,
                'pair':     f"{col1.replace('Close_', '')} & {col2.replace('Close_', '')}",
                'adf_stat': score,
                'p_value':  pvalue
            })

    # Compile results into a DataFrame
    results_df = pd.DataFrame(results).sort_values(['sector', 'p_value'])
    results_df = results_df.reset_index()
    results_df = results_df.drop("index", axis=1)


    # If you want only the likely cointegrated pairs (e.g. p < 0.05):
    sig = results_df[results_df['p_value'] < 0.05]
    sig = sig.reset_index()
    sig = sig.drop("index", axis=1)
    print("\nLikely cointegrated pairs (p < 0.05):")
    print(sig)

    # Save to CSV:
    results_df.to_csv('../../data/all_cointegration_results.csv', index=True)
    sig.to_csv('../../data/significant_cointegrated_pairs.csv', index=True)
    
    return sig


# Function to calculate hedge ratio and spread
def calculate_spreads(price_df):
    
    sig = cointegration_filter(price_df)
    pairs = sig['pair'].str.split(' & ').apply(tuple).tolist()
    
    results = []
    for etf1, etf2 in pairs:
        col1 = f"Close_{etf1}"
        col2 = f"Close_{etf2}"
        
        if col1 not in price_df.columns or col2 not in price_df.columns:
            print(f"Missing columns for pair: {etf1}, {etf2}")
            continue
        
        pair_df = price_df[[col1, col2]].dropna()
        
        # Run OLS: ETF1 ~ ETF2
        X = sm.add_constant(pair_df[col2])
        y = pair_df[col1]
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params[col2]
        
        # Spread = ETF1 - hedge_ratio * ETF2
        spread = y - hedge_ratio * pair_df[col2]
        
        pair_df["spread"] = spread
        
        # Statistic metrics of spread
        mean = np.mean(spread)
        sd = np.std(spread, ddof=1)
        
        results.append({
            "ETF1": etf1,
            "ETF2": etf2,
            "hedge_ratio": hedge_ratio,
            "spread_prices": pair_df,
            "mean": mean,
            "SD": sd
        })
        
        for r in results:
            print(f"{r['ETF1']} & {r['ETF2']} → hedge ratio: {r['hedge_ratio']:.4f}")
            print(r['spread_prices'].head(), "\n")
        
        sns.lineplot(data=r['spread_prices']['spread'])
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.title(f"{r['ETF1']} & {r['ETF2']} Spread Over Time")
        plt.show()
        
    
    return results


def generate_signals(results, entry_z=2.0):
    """
    For each cointegrated pair in results, compute:
      - zscore = (spread - mean) / SD
      - signal =  1 when zscore < -entry_z (long ETF1, short ETF2)
                 -1 when zscore >  entry_z (short ETF1, long ETF2)
                  0 otherwise
      - position_ETF1 = signal
      - position_ETF2 = -hedge_ratio * signal

    Returns a dict keyed by "ETF1 & ETF2" with each value a DataFrame
    containing columns: ['spread','zscore','signal','position_<ETF1>','position_<ETF2>'].
    """
    trades = {}

    for r in results:
        etf1, etf2 = r["ETF1"], r["ETF2"]
        hr, df = r["hedge_ratio"], r["spread_prices"].copy()
        mu, sd = r["mean"], r["SD"]

        # compute z-score
        df["zscore"] = (df["spread"] - mu) / sd

        # generate entry signals
        df["signal"] = 0
        df.loc[df["zscore"] >  entry_z, "signal"] = -1   # short etf1, long etf2
        df.loc[df["zscore"] < -entry_z, "signal"] =  1   # long etf1, short etf2

        # translate signal into per-ETF positions
        df[f"position_{etf1}"] = df["signal"]
        df[f"position_{etf2}"] = -hr * df["signal"]

        trades[f"{etf1} & {etf2}"] = df[[
            "spread", "zscore", "signal", f"position_{etf1}", f"position_{etf2}"
        ]]
        
    for trade in trades:
        print(trades[trade])
    return trades


def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    price_df = pd.read_csv('../../data/master_df.csv', index_col=0, parse_dates=True)
    results = calculate_spreads(price_df)
    trades = generate_signals(results)
    

if __name__ == "__main__":
    main()
