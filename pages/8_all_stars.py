import streamlit as st
import pandas as pd
import numpy as np
from db import get_engine
from datetime import date

# Page config
st.set_page_config(page_title="Greer 3-Star Companies", layout="wide")

st.markdown("<h1>⭐ All 3-Star Companies</h1>", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_3star_companies():
    engine = get_engine()
    df = pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange,
               delisted, delisted_date, greer_star_rating
        FROM companies
        WHERE greer_star_rating >= 3
        ORDER BY ticker;
        """,
        engine
    )
    return df

@st.cache_data(ttl=600)
def fetch_latest_snapshot(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"""
        SELECT *
        FROM latest_company_snapshot
        WHERE ticker IN ({placeholders})
    """
    return pd.read_sql(query, engine, params=tuple(tickers))

@st.cache_data(ttl=600)
def fetch_latest_gfv(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"""
        SELECT DISTINCT ON (ticker)
               ticker, date AS gfv_date, close_price, gfv_price, gfv_status,
               dcf_value, graham_value
        FROM greer_fair_value_daily
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, date DESC
    """
    return pd.read_sql(query, engine, params=tuple(tickers))

# ----------------------------------------------------------
# Load data
stars_df = fetch_3star_companies()
if stars_df.empty:
    st.info("No 3-star companies found (greer_star_rating ≥ 3).")
    st.stop()

tickers = stars_df["ticker"].tolist()
snap_df = fetch_latest_snapshot(tickers)
gfv_df = fetch_latest_gfv(tickers)

# Merge data
df = stars_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df[[
    "ticker", "close_price", "gfv_price", "gfv_status", "gfv_date"
]], how="left", on="ticker")

# Option: show table or cards
show_table = st.checkbox("Show as table (vs cards)", value=True)

if show_table:
    tbl = df[[
        "ticker", "name", "sector", "industry", "exchange",
        "greer_star_rating",
        "greer_value_score", "above_50_count",
        "greer_yield_score",
        "gfv_price", "gfv_status", "close_price"
    ]].rename(columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "exchange": "Exchange",
        "greer_star_rating": "Stars",
        "greer_value_score": "Greer Value %",
        "above_50_count": "GV Above50 Count",
        "greer_yield_score": "Yield Score",
        "gfv_price": "Fair Value (GFV)",
        "gfv_status": "GFV Status",
        "close_price": "Current Price",
    })

    # Convert ticker column to markdown links
    def make_link(t):
        return f'<a href="/?ticker={t}" target="_self">{t}</a>'

    tbl["Ticker"] = tbl["Ticker"].apply(make_link)
    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV of 3-star companies",
        csv,
        "greer_3star_companies.csv",
        mime="text/csv"
    )
else:
    for _, row in df.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        stars = int(row.get("greer_star_rating", 0))

        # Make the ticker + name heading a link
        link = f"/?ticker={ticker}"
        st.markdown(f"## ⭐ <a href='{link}' target='_self'>{ticker}</a> — {name}  {'★'*stars}", unsafe_allow_html=True)

        st.write({
            "Sector": row.get("sector"),
            "Industry": row.get("industry"),
            "Exchange": row.get("exchange"),
            "Current Price": row.get("close_price"),
            "Greer Fair Value": row.get("gfv_price"),
            "GFV Status": row.get("gfv_status"),
            "Greer Value %": row.get("greer_value_score"),
            "Yield Score": row.get("greer_yield_score"),
        })

        st.markdown("---")
