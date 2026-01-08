# 2_Dashboard-mini.py
# dashboard_mini_cols.py

import streamlit as st
import pandas as pd
from db import get_engine

# st.set_page_config(page_title="Mini Dashboard — Ticker Grid", layout="wide")

@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    df = pd.read_sql(
        """
        SELECT c.ticker,
               c.greer_star_rating,
               s.greer_value_score,
               s.above_50_count,
               s.greer_yield_score,
               s.buyzone_flag
        FROM companies c
        JOIN latest_company_snapshot s ON c.ticker = s.ticker
        WHERE c.delisted = FALSE
        ORDER BY c.ticker;
        """,
        engine
    )
    return df

df = load_data()

st.markdown("## Mini Dashboard — All Companies")

# Sidebar filters (optional)
with st.sidebar:
    st.header("Filters")
    min_stars = st.selectbox("Minimum star rating", [0, 1, 2, 3], index=0)
    only_buyzone = st.checkbox("Only show BuyZone = True", value=False)

filtered = df.copy()
if min_stars:
    filtered = filtered[filtered["greer_star_rating"] >= min_stars]
if only_buyzone:
    filtered = filtered[filtered["buyzone_flag"] == True]

st.markdown(f"Showing **{len(filtered)}** companies")

# Render ticker grid using columns
BOXES_PER_ROW = 10  # control how many per row

for i in range(0, len(filtered), BOXES_PER_ROW):
    row = filtered.iloc[i : i + BOXES_PER_ROW]
    cols = st.columns(len(row))
    for col, (_, r) in zip(cols, row.iterrows()):
        t = r["ticker"]
        stars = int(r.get("greer_star_rating", 0))
        gv = r.get("greer_value_score") or 0
        above50 = r.get("above_50_count") or 0

        # color logic for box background (adjust as needed)
        if above50 == 6:
            box_color = "#D4AF37"
        elif gv >= 50:
            box_color = "#4CAF50"
        else:
            box_color = "#F44336"

        link = f"/?ticker={t}"

        col.markdown(
            f"""
            <a href="{link}" style="
               display:block;
               background:{box_color};
               color:#fff;
               padding:4px;
               margin:2px;
               border-radius:4px;
               text-align:center;
               text-decoration:none;
               font-weight:bold;
               font-size:0.9rem;
            ">
             {t}
            </a>
            """,
            unsafe_allow_html=True
        )
