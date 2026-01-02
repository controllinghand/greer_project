# 8_all_stars.py

import streamlit as st
import pandas as pd
from db import get_engine

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Greer 3-Star Companies", layout="wide")
st.markdown("<h1>⭐ All 3-Star Companies</h1>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_3star_companies():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange,
               delisted, delisted_date, greer_star_rating
        FROM companies
        WHERE greer_star_rating >= 3
        ORDER BY ticker;
        """,
        engine,
    )

@st.cache_data(ttl=600)
def fetch_latest_snapshot(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    return pd.read_sql(
        f"""
        SELECT *
        FROM latest_company_snapshot
        WHERE ticker IN ({placeholders})
        """,
        engine,
        params=tuple(tickers),
    )

@st.cache_data(ttl=600)
def fetch_latest_gfv(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    return pd.read_sql(
        f"""
        SELECT DISTINCT ON (ticker)
               ticker,
               close_price,
               gfv_price,
               gfv_status
        FROM greer_fair_value_daily
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, date DESC
        """,
        engine,
        params=tuple(tickers),
    )

@st.cache_data(ttl=600)
def fetch_latest_buyzone(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    return pd.read_sql(
        f"""
        SELECT DISTINCT ON (ticker)
               ticker,
               in_buyzone,
               in_sellzone
        FROM greer_buyzone_daily
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, date DESC
        """,
        engine,
        params=tuple(tickers),
    )

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
stars_df = fetch_3star_companies()
if stars_df.empty:
    st.info("No 3-star companies found (greer_star_rating ≥ 3).")
    st.stop()

tickers = stars_df["ticker"].tolist()
snap_df = fetch_latest_snapshot(tickers)
gfv_df = fetch_latest_gfv(tickers)
bz_df = fetch_latest_buyzone(tickers)

# ----------------------------------------------------------
# Merge
# ----------------------------------------------------------
df = stars_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df, how="left", on="ticker")
df = df.merge(bz_df, how="left", on="ticker")

# Normalize flags
for col in ["in_buyzone", "in_sellzone"]:
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].fillna(False).astype(bool)

# ----------------------------------------------------------
# Top summary metrics
# ----------------------------------------------------------
total_count = len(df)
buyzone_count = int(df["in_buyzone"].sum())
sellzone_count = int(df["in_sellzone"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("⭐ Total 3-Star+ Companies", f"{total_count:,}")
c2.metric("🟢 In BuyZone", f"{buyzone_count:,}")
c3.metric("🔴 In SellZone", f"{sellzone_count:,}")
c4.metric("⚪ Neutral", f"{total_count - buyzone_count - sellzone_count:,}")

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
left, mid, right = st.columns([1.2, 1.2, 2.6])

with left:
    zone_filter = st.selectbox(
        "Filter",
        ["All", "BuyZone only", "SellZone only"],
        index=0,
    )

with mid:
    show_table = st.checkbox("Show as table (vs cards)", value=True)

with right:
    hide_delisted = st.checkbox("Hide delisted", value=True)

if hide_delisted and "delisted" in df.columns:
    df = df[df["delisted"] == False].copy()

if zone_filter == "BuyZone only":
    df = df[df["in_buyzone"]]
elif zone_filter == "SellZone only":
    df = df[df["in_sellzone"]]

if df.empty:
    st.info("No companies match the current filter.")
    st.stop()

# ----------------------------------------------------------
# Render helpers
# ----------------------------------------------------------
def make_link(t):
    return f'<a href="/?ticker={t}" target="_self">{t}</a>'

def badge(v: bool) -> str:
    return "✅" if v else ""

# ----------------------------------------------------------
# Render
# ----------------------------------------------------------
if show_table:
    tbl = df[[
        "ticker", "name", "sector", "industry", "exchange",
        "greer_star_rating",
        "in_buyzone",
        "in_sellzone",
        "greer_value_score",
        "above_50_count",
        "greer_yield_score",
        "gfv_price",
        "gfv_status",
        "close_price",
    ]].rename(columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "exchange": "Exchange",
        "greer_star_rating": "Stars",
        "in_buyzone": "BuyZone",
        "in_sellzone": "SellZone",
        "greer_value_score": "Greer Value %",
        "above_50_count": "GV Above50 Count",
        "greer_yield_score": "Yield Score",
        "gfv_price": "Fair Value (GFV)",
        "gfv_status": "GFV Status",
        "close_price": "Current Price",
    })

    tbl["Ticker"] = tbl["Ticker"].apply(make_link)
    tbl["BuyZone"] = tbl["BuyZone"].apply(badge)
    tbl["SellZone"] = tbl["SellZone"].apply(badge)

    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        "Download CSV",
        tbl.to_csv(index=False).encode("utf-8"),
        "greer_3star_companies.csv",
        mime="text/csv",
    )

else:
    for _, row in df.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        stars = int(row.get("greer_star_rating", 0))

        zone = (
            "🟢 BuyZone" if row["in_buyzone"]
            else "🔴 SellZone" if row["in_sellzone"]
            else "⚪ Neutral"
        )

        st.markdown(
            f"## ⭐ <a href='/?ticker={ticker}' target='_self'>{ticker}</a> — {name}  {'★'*stars}  &nbsp;&nbsp; {zone}",
            unsafe_allow_html=True,
        )

        st.write({
            "Sector": row.get("sector"),
            "Industry": row.get("industry"),
            "Exchange": row.get("exchange"),
            "Greer Value %": row.get("greer_value_score"),
            "Yield Score": row.get("greer_yield_score"),
            "Fair Value": row.get("gfv_price"),
            "GFV Status": row.get("gfv_status"),
            "Current Price": row.get("close_price"),
        })

        st.markdown("---")
