# test_iv_fetch.py
import yfinance as yf
import pandas as pd

def get_iv_for_ticker(ticker: str):
    stock = yf.Ticker(ticker)
    expiries = stock.options
    print(f"{ticker} expiries: {expiries[:5]}")
    if not expiries:
        return None
    expiry = expiries[0]  # pick nearest
    opt_chain = stock.option_chain(expiry)
    df_calls = opt_chain.calls
    df_puts  = opt_chain.puts
    # Combine
    df = pd.concat([df_calls, df_puts], ignore_index=True)
    # Drop NaNs in impliedVolatility
    df2 = df.dropna(subset=["impliedVolatility"])
    print(f"{ticker} {expiry} sample impliedVolatility stats:")
    print(df2["impliedVolatility"].describe())
    # Optionally: strike closest to underlying price
    price = stock.info.get("regularMarketPrice", None)
    if price is not None:
        df2["dist"] = (df2["strike"] - price).abs()
        atm = df2.loc[df2["dist"].idxmin()]
        print(f"{ticker} ATM strike {atm['strike']}, impliedVolatility {atm['impliedVolatility']}")
    return df2

for t in ["NVDA", "TSLA"]:
    get_iv_for_ticker(t)
