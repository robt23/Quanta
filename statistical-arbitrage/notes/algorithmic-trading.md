# Robert's Notes: "Quantitative Trading: How to Build Your Own Algorithmic Trading Business" by Ernie Chan

## Terminology
- **Stocks**: ownership shares in a company; when you buy a stock, you own a portion of that company and may receive dividends and/or capital gains
- **Bonds**: debt instruments issued by governments or corporations to raise capital; when you buy a bond, you’re lending money to the issuer in exchange for periodic interest payments and repayment of the principal at maturity
- **Securities**: broad term for tradable financial assets, including stocks, bonds, options, mutual funds, and other investment products
- **Derivatives**: financial contracts whose value is derived from an underlying asset, such as a stock, bond, index, commodity, or interest rate; common derivatives include options, futures, and swaps
- **Options**:  type of derivative that gives the holder the right, but not the obligation, to buy (call) or sell (put) an asset at a specific price before or on a specific date
- **Futures**: financial contracts that obligate the buyer to purchase, and the seller to deliver, an asset at a predetermined price and date in the future
- **Longs**: taking a long position means you buy an asset expecting its price will rise in the future, allowing you to sell it at a profit
- **Shorts**: selling an asset you don’t own (typically borrowed), hoping to buy it back later at a lower price and profit from the price decline
- **Index Funds**: passive mutual funds or ETFs designed to track the performance of a specific market index; they offer broad market exposure and low fees
- **Mutual Funds**: pooled investment vehicles managed by professionals, where investors buy shares and the fund invests in a diversified portfolio of stocks, bonds, or other securities
- **ETFs**: like mutual funds, but trade on stock exchanges like individual stocks; ETFs typically track indices and are known for low costs and intraday liquidity
- **Equity**: another word for ownership in an asset, typically a company; in finance, equity refers to stock shares or the net value of an asset after subtracting liabilities
- **Splits**: increases the number of shares a company has without changing the total value of your investment
- **Dividends**: a cash payment (or sometimes additional stock) made by a company to its shareholders, usually from profits
- **Basis Point**: 1 basis point corresponds to 1/100th of a percentage 
point i.e. 0.01%

- Capacity: how much a strategy is able to absort without negatively impacting returns

## Concepts


### Sharpe Ratio
$\text{Sharpe Ratio} = \frac{\text{Average of Excess Returns}}{\text{SD of Excess Returns}} \: \text{where Excess Returns} = \text{Portfolio Returns} − \text{Benchmark Returns}$

- If a strategy trades infrequently, the Sharpe Ratio is probably not high, converse of a strategy that has long or high drawdowns
- <1 ratio is not suitable standalone, annualized ratio for monthly profitability is >2, >3 for daily profitability


### Survivorship Bias
- Historical database that only include stocks that have "survived" bankruptcies, delistings, mergers, or acquisitions
- Can be dangerous when backtesting and cause to misleading results because value might be skewed


### Drawdown
- Strategy suffers drawdown when it loses money recently: at time $t$, difference in current equity and global maximum before $t$
- Maximum drawdown: difference between global max equity (high watermark) and global min equity, given that the min occurred after the max
- Max drawdown duration: longest it has taken for the equity curve to recover losses


### Data-Snooping Bias
- Overfitting model with lots of parameters to historial accidents may fail b/c these events probably won't repeat themselves
- Simple models are less prone to suffering from data-snooping bias


### Backtesting
- The process of testing a trading strategy or model on historical data to see how it would have performed in the past
- Use open/close data vs. high/low data and adjusted for splits and dividends. If a company issues a dividend $\$d$ per share, all prices before $T$ ned to be multiplied by $(\text{Close}(T - 1) - d) / \text{Close}(T - 1)$ where $Close$ is the closing price of a specific day and $T - 1$ is the trading day before $T$.


### Neutral Portfolios
- Investment strategy designed to eliminate or minimize exposure to a specific market risk — such as movements in the overall stock market — while still allowing the investor to profit from relative performance between assets

| Type              | Neutral To                 | Description / Example                                                                 |
|-------------------|-----------------------------|----------------------------------------------------------------------------------------|
| **Market Neutral**| Overall market direction    | Long undervalued stocks and short overvalued ones to eliminate exposure to market moves |
| **Sector Neutral**| Specific sectors or industries | Equal exposure across sectors to avoid bias toward any one industry                   |
| **Dollar Neutral**| Dollar exposure (long = short) | Long $100K in one stock, short $100K in another                                       |
| **Beta Neutral**  | Market beta (target beta = 0) | Adjust positions so overall sensitivity to market is zero (e.g., using CAPM beta)     |


### Performance Measures
- Subtract risk-free rate from returns of dollar-neutral portolio or no?
- "The answer is no. A dollar-neutral portfolio is self-financing, meaning the cash you get from selling short pays for the purchase of the long securities, so the financing cost (due to the spread between the credit and debit interest rates) is small and can be neglected for many backtesting purposes. Meanwhile, the margin balance you have to maintain earns a credit interest close to the risk-free rate $r_F$. So let’s say the strategy return (the portfolio return minus the contribution from the credit interest) is $R$, and the risk-
free rate is $r_F$. Then the excess return used in calculating the Sharpe ratio is $R + rF– rF = R$. So, essentially, you can ignore the risk-free rate in the whole calculation and just focus on the returns due to your stock positions."
- Subtract risk-free rate when calculating Sharpe ratio iff financing cost
