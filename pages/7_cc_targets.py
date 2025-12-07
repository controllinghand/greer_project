# 7_cc_targets_filtered.py  (modified version)

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import date, timedelta

st.set_page_config(page_title="Covered-Call/Put Target Scan", layout="wide")

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
        ivs.expiry,
        ivs.contract_count,
        ivs.iv_median,
        ivs.iv_atm,
        ivs.atm_premium,
        ivs.atm_premium_pct
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
      lp.latest_price as latest_price,
      r.iv_atm,
      r.iv_median,
      r.atm_premium,
      r.atm_premium_pct,
      r.expiry,
      r.contract_count
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
        params={"market_cap_min": market_cap_min,
                "iv_min_atm": iv_min_atm}
    )
    return df

def main():
    st.title("📋 Covered-Call Target Scan")
    st.markdown(
        """
        **Filter criteria:**  
        - Market Cap ≥ your threshold  
        - ATM Implied Volatility (IV ATM) ≥ your threshold  
        - Option contract count ≥ 10  
        - Option expiry ≤ 7 days from today  
        """
    )

    # Input params
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

    if df.empty:
        st.info("No raw candidates found under market cap / IV filters.")
        return

    # Pre-format
    df["market_cap"] = df["market_cap"].apply(lambda x: f"${x:,.0f}")
    df["latest_price"] = df["latest_price"].round(2)
    df["iv_median"] = df["iv_median"].round(3)
    df["iv_atm"] = df["iv_atm"].round(3)
    if "atm_premium" in df.columns:
        df["atm_premium"] = df["atm_premium"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
    if "atm_premium_pct" in df.columns:
        df["atm_premium_pct"] = df["atm_premium_pct"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notnull(x) else ""
        )

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["contract_count"] = df["contract_count"].astype("Int64")

    # ********** NEW FILTERS **********
    # 1) contract_count >= 10
    df = df[df["contract_count"] >= 10]

    # 2) expiry within next 7 days (including today)
    today = date.today()
    max_allowed = today + timedelta(days=7)
    df = df[df["expiry"] <= max_allowed]
    # ********************************

    st.subheader(f"🧮 Found {len(df)} potential targets after contract & expiry filter")
    if df.empty:
        st.info("No tickers match all filter criteria (market cap / IV / contract count / expiry).")
        return

    # Optional ticker search filter
    search = st.text_input("Search ticker (optional):").upper().strip()
    if search:
        df = df[df["ticker"].str.contains(search)]

    # Columns to display
    columns = [
        "ticker",
        "latest_price",
        "market_cap",
        "iv_atm",
        "iv_median"
    ]
    if "atm_premium" in df.columns:
        columns += ["atm_premium", "atm_premium_pct"]
    columns += ["expiry", "contract_count"]

    st.dataframe(df[columns], hide_index=True, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="covered_call_targets_filtered.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
