# test_iv_multi.py
import yfinance as yf
import pandas as pd

tickers = ["NVDA","TSLA","MSFT","LLY","ANET","AAPL","VZ","GOOG"]

def get_iv_summary(ticker: str):
    stock = yf.Ticker(ticker)
    expiries = stock.options
    if not expiries:
        print(f"{ticker}: no options expirations found")
        return None
    expiry = expiries[0]  # next available expiry
    try:
        opt = stock.option_chain(expiry)
    except Exception as e:
        print(f"{ticker}: error fetching options for expiry {expiry}: {e}")
        return None

    df = pd.concat([opt.calls, opt.puts], ignore_index=True)
    df = df.dropna(subset=["impliedVolatility"])
    if df.empty:
        print(f"{ticker}: no impliedVolatility data for expiry {expiry}")
        return None

    stats = df["impliedVolatility"].describe()
    # find ATM strike impliedVolatility
    price = stock.info.get("regularMarketPrice", None)
    atm_iv = None
    if price is not None:
        df["dist"] = (df["strike"] - price).abs()
        atm_iv = df.loc[df["dist"].idxmin(), "impliedVolatility"]

    return {
        "ticker": ticker,
        "expiry": expiry,
        "count": int(stats["count"]),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "min": float(stats["min"]),
        "25%": float(stats["25%"]),
        "50%": float(stats["50%"]),
        "75%": float(stats["75%"]),
        "max": float(stats["max"]),
        "atm_iv": float(atm_iv) if atm_iv is not None else None
    }

results = []
for t in tickers:
    res = get_iv_summary(t)
    if res:
        results.append(res)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv("iv_summary_results.csv", index=False)
