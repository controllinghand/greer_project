# 7_cc_targets.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine

st.set_page_config(page_title="Covered-Call Target Scan", layout="wide")

@st.cache_data(ttl=3600)
def load_cc_targets(iv_min_atm: float, market_cap_min: float):
    engine = get_engine()
    query = text("""
    WITH latest_price AS (
      SELECT
        p.ticker,
        p.close AS latest_price
      FROM prices p
      JOIN (
        SELECT ticker, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
      ) mp
        ON mp.ticker = p.ticker
       AND mp.max_date = p.date
    ),
    latest_shares AS (
      SELECT
        f.ticker,
        f.shares_outstanding
      FROM financials f
      JOIN (
        SELECT ticker, MAX(report_date) AS max_report
        FROM financials
        GROUP BY ticker
      ) mf
        ON mf.ticker = f.ticker
       AND mf.max_report = f.report_date
    ),
    mc AS (
      SELECT
        ls.ticker,
        (ls.shares_outstanding * lp.latest_price) AS market_cap
      FROM latest_shares ls
      JOIN latest_price lp ON ls.ticker = lp.ticker
    ),
    recent_iv AS (
      SELECT
        ivs.ticker,
        ivs.fetch_date,
        ivs.iv_median,
        ivs.iv_atm
      FROM iv_summary ivs
      WHERE ivs.fetch_date = (
        SELECT MAX(fetch_date)
          FROM iv_summary
         WHERE ticker = ivs.ticker
      )
    )
    SELECT
      r.ticker,
      mc.market_cap,
      r.iv_median,
      r.iv_atm,
      lp.latest_price
    FROM recent_iv r
    JOIN mc ON r.ticker = mc.ticker
    JOIN latest_price lp ON r.ticker = lp.ticker
    WHERE
      mc.market_cap >= :market_cap_min
      AND r.iv_atm >= :iv_min_atm
    ORDER BY
      r.iv_atm DESC,
      mc.market_cap DESC
    """)
    df = pd.read_sql(
        query,
        engine,
        params={
            "market_cap_min": market_cap_min,
            "iv_min_atm": iv_min_atm
        }
    )
    return df

def main():
    st.title("📋 Covered-Call Target Scan")
    st.markdown(
        """
        **Filter criteria:**  
        * Market Cap ≥ your threshold  
        * ATM Implied Volatility (IV ATM) ≥ your threshold  
        * (Optional) IV Median also shown  
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        iv_min_atm = st.number_input(
            "Minimum implied volatility (ATM) (decimal form)",
            value=0.70,
            step=0.05,
            format="%.2f"
        )
    with col2:
        market_cap_min = st.number_input(
            "Minimum market cap",
            value=10_000_000_000,
            step=1_000_000_000,
            format="%d"
        )

    df = load_cc_targets(iv_min_atm=iv_min_atm, market_cap_min=market_cap_min)

    st.subheader(f"🧮 Found {len(df)} potential targets")
    if df.empty:
        st.info("No tickers match the current filter criteria.")
        return

    # Format numeric columns
    df["market_cap"] = df["market_cap"].apply(lambda x: f"${x:,.0f}")
    df["latest_price"] = df["latest_price"].round(2)
    df["iv_median"] = df["iv_median"].round(3)
    df["iv_atm"] = df["iv_atm"].round(3)

    # Optional: ticker filter
    search = st.text_input("Search ticker (optional):").upper().strip()
    if search:
        df = df[df["ticker"].str.contains(search)]

    st.dataframe(df, hide_index=True, use_container_width=True)

    # Download button
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="covered_call_targets.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
