# opportunities.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine  # ✅ Centralized DB connection

"""
Greer Value Opportunities Page
---------------------------------
**Source:** `latest_company_snapshot` materialized view & `fair_value_gaps` table  
**Criteria:**
1. Greer Value Score ≥ 50  
2. Yield Score ≥ 3  
3. Currently *in* the Buy‑Zone (`buyzone_flag` = TRUE)  
4. Latest Fair‑Value‑Gap direction = **bullish**  

Surfaces each ticker’s most recent un‑mitigated bullish gap date.
"""

st.set_page_config(page_title="Greer Value Opportunities", layout="wide")

# ----------------------------------------------------------
# Load filtered tickers + last un-mitigated bullish gap date
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_filtered_companies():
    engine = get_engine()
    query = text(
        """
        WITH live_bull_gaps AS (
          SELECT
            ticker,
            date AS entry_date
          FROM public.fair_value_gaps
          WHERE direction = 'bullish'
            AND mitigated = false
        ),
        last_entry AS (
          SELECT
            ticker,
            MAX(entry_date) AS last_entry_date
          FROM live_bull_gaps
          GROUP BY ticker
        )
        SELECT
          l.ticker,
          l.greer_value_score  AS greer_value,
          l.greer_yield_score  AS yield_score,
          l.buyzone_flag,
          l.fvg_last_direction,
          le.last_entry_date
        FROM last_entry le
        JOIN latest_company_snapshot l
          ON l.ticker = le.ticker
        JOIN companies c  -- Join to companies for delisted check
          ON l.ticker = c.ticker
        WHERE l.greer_value_score >= 50
          AND l.greer_yield_score >= 3
          AND l.buyzone_flag = TRUE
          AND l.fvg_last_direction = 'bullish'
          AND c.delisted = FALSE  -- Exclude delisted
        ORDER BY le.last_entry_date DESC;
        """
    )
    return pd.read_sql(query, engine)

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
def main():
    st.title("💎 Greer Value Opportunities")
    st.markdown(
        """
        **Showing companies that currently satisfy:**  
        * Greer Value ≥ **50**  
        * Yield Score ≥ **3**  
        * **In** the Buy‑Zone  
        * Latest FVG direction is **bullish**
        """,
        unsafe_allow_html=True,
    )

    df = load_filtered_companies()

    # format date column
    df['last_entry_date'] = pd.to_datetime(df['last_entry_date']).dt.date

    st.subheader(f"📈 {len(df)} matching companies")
    if df.empty:
        st.info("No companies currently meet all conditions.")
        return

    # ticker filter
    search = st.text_input("Filter by ticker …", "").upper()
    if search:
        df = df[df["ticker"].str.contains(search)]

    st.dataframe(df, hide_index=True, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="greer_value_opportunities.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
