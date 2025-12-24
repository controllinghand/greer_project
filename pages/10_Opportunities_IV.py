# opportunities-IV.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine  # ✅ Centralized DB connection

# Page config — update tab title
st.set_page_config(page_title="⭐ Opportunities with IV", layout="wide")

# --------------------------------------------------
# Insert custom CSS for styled table
st.markdown("""
<style>
  /* Table styling */
  .op-table {
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
  }
  .op-table th, .op-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }
  .op-table th {
    background-color: #1976D2;
    color: white;
  }
  .op-table tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  .op-table tr:hover {
    background-color: #f1f1f1;
  }
  .star-icon {
    color: #D4AF37; /* gold color for stars */
    font-size: 1.1rem;
  }
  a.ticker-link {
    color: #1976D2;
    text-decoration: none;
    font-weight: bold;
  }
  a.ticker-link:hover {
    text-decoration: underline;
  }
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------


@st.cache_data(ttl=3600)
def load_filtered_companies():
    engine = get_engine()
    query = text(
        """
        WITH live_bull_gaps AS (
          SELECT ticker, date AS entry_date
          FROM public.fair_value_gaps
          WHERE direction = 'bullish'
            AND mitigated = false
        ),
        last_entry AS (
          SELECT ticker, MAX(entry_date) AS last_entry_date
          FROM live_bull_gaps
          GROUP BY ticker
        ),
        latest_prices AS (
          SELECT ticker, close AS current_price, date
          FROM prices
          WHERE (ticker, date) IN (
               SELECT ticker, MAX(date) FROM prices GROUP BY ticker
            )
        ),
        latest_gfv AS (
          SELECT ticker, gfv_price, date
          FROM greer_fair_value_daily
          WHERE (ticker, date) IN (
               SELECT ticker, MAX(date) FROM greer_fair_value_daily GROUP BY ticker
            )
        ),
        recent_iv AS (
          SELECT DISTINCT ON (ivs.ticker)
            ivs.ticker,
            ivs.iv_atm,
            ivs.expiry AS iv_expiry
          FROM iv_summary ivs
          JOIN (
            SELECT ticker, MAX(fetch_date) AS max_fetch_date
            FROM iv_summary
            GROUP BY ticker
          ) mf
            ON mf.ticker = ivs.ticker
           AND mf.max_fetch_date = ivs.fetch_date
          WHERE
            ivs.expiry >= CURRENT_DATE
          ORDER BY
            ivs.ticker,
            ivs.expiry ASC
        )
        SELECT
          l.ticker,
          c.greer_star_rating    AS stars,
          l.greer_value_score    AS greer_value,
          l.greer_yield_score    AS yield_score,
          l.buyzone_flag,
          l.fvg_last_direction,
          le.last_entry_date,
          p.current_price,
          gfv.gfv_price,
          gfv.gfv_price * 0.75   AS gfv_mos,
          riv.iv_atm             AS iv_atm,
          riv.iv_expiry          AS iv_expiry
        FROM last_entry le
        JOIN latest_company_snapshot l ON l.ticker = le.ticker
        JOIN companies c               ON l.ticker = c.ticker
        JOIN latest_prices p           ON p.ticker = l.ticker
        JOIN latest_gfv gfv            ON gfv.ticker = l.ticker
        LEFT JOIN recent_iv riv        ON riv.ticker = l.ticker
        WHERE l.greer_value_score >= 50
          AND l.greer_yield_score >= 3
          AND l.buyzone_flag = TRUE
          AND l.fvg_last_direction = 'bullish'
          AND p.current_price < gfv.gfv_price * 0.75
          AND c.delisted = FALSE
        ORDER BY
          riv.iv_atm DESC NULLS LAST,
          le.last_entry_date DESC;
        """
    )
    return pd.read_sql(query, engine)


def main():
    st.title("⭐ Opportunities with IV")
    st.markdown(
        """
        **Showing companies that currently meet all of the following criteria:**  
        - Greer Value ≥ **50**  
        - Yield Score ≥ **3**  
        - **In** the Buy-Zone  
        - Latest FVG direction is **bullish**  
        - Current Price < GFV Price × **0.75** (25% margin of safety)

        **IV logic (tightened):**
        - Uses latest `fetch_date` per ticker  
        - Picks nearest expiry within **≤ 7 days** (and ≥ today)  
        """,
        unsafe_allow_html=True,
    )

    df = load_filtered_companies()
    if df.empty:
        st.info("No companies currently meet all conditions.")
        return

    # Format numeric columns
    df['last_entry_date'] = pd.to_datetime(df['last_entry_date']).dt.date

    for col in ['current_price', 'gfv_price', 'gfv_mos']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Format iv_atm
    if "iv_atm" in df.columns:
        df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce").round(3)

    # Format iv_expiry
    if "iv_expiry" in df.columns:
        df["iv_expiry"] = pd.to_datetime(df["iv_expiry"], errors="coerce").dt.date

    # Total matching count
    st.subheader(f"📈 {len(df)} matching companies")

    # Build star-rating icons
    def stars_to_html(n):
        try:
            n = int(n)
        except:
            return ""
        return f"<span class='star-icon'>{'★'*n}{'☆'*(3-n)}</span>"

    df = df.copy()
    df['Stars'] = df['stars'].apply(stars_to_html)

    # Make ticker column a clickable link
    def link_ticker(t: str) -> str:
        return f"<a href='/?ticker={t}' class='ticker-link'>{t}</a>"

    df['Ticker'] = df['ticker'].apply(link_ticker)

    # Build display table
    df_display = df[[
        'Ticker', 'Stars', 'greer_value', 'yield_score',
        'iv_atm', 'iv_expiry',
        'current_price', 'gfv_price', 'gfv_mos', 'last_entry_date'
    ]].rename(columns={
        'greer_value': 'Greer Value %',
        'yield_score': 'Yield Score',
        'iv_atm': 'IV ATM',
        'iv_expiry': 'IV Expiry',
        'current_price': 'Current Price',
        'gfv_price': 'GFV',
        'gfv_mos': 'GFV 75% MOS',
        'last_entry_date': 'Last Gap Date'
    })

    html_table = df_display.to_html(
        index=False,
        escape=False,
        classes="op-table"
    )

    st.markdown(html_table, unsafe_allow_html=True)

    st.download_button(
        "Download CSV",
        df_display.to_csv(index=False).encode('utf-8'),
        file_name="greer_gfv_opportunities.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
