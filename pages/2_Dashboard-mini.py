# 2_Dashboard-mini.py
# dashboard_mini_cols.py

import streamlit as st
import pandas as pd
from db import get_engine

# st.set_page_config(page_title="Mini Dashboard — Ticker Grid", layout="wide")

# --------------------------------------------------
# Load the dashboard snapshot (fast read-only table)
# --------------------------------------------------
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
            ticker,
            greer_star_rating,
            greer_value_score,
            above_50_count,
            greer_yield_score,
            buyzone_flag,
            snapshot_date
        FROM dashboard_snapshot
        ORDER BY ticker;
        """,
        engine
    )

df = load_data()
if df.empty:
    st.warning("dashboard_snapshot is empty. Run run_all.py (or build_dashboard_snapshot.py) to populate it.")
    st.stop()

st.markdown("## Mini Dashboard — All Companies")

last_updated = df["snapshot_date"].max() if "snapshot_date" in df.columns else None
if last_updated is not None:
    st.caption(f"Last updated: {last_updated}")

# Sidebar filters (optional)
with st.sidebar:
    st.header("Filters")
    min_stars = st.selectbox("Minimum star rating", [0, 1, 2, 3], index=0)
    only_buyzone = st.checkbox("Only show BuyZone = True", value=False)

filtered = df.copy()

# NaN-safe star filtering
if min_stars:
    stars_series = pd.to_numeric(filtered["greer_star_rating"], errors="coerce").fillna(0)
    filtered = filtered[stars_series >= min_stars]

if only_buyzone:
    filtered = filtered[filtered["buyzone_flag"] == True]

st.markdown(f"Showing **{len(filtered)}** companies")

# Render ticker grid using columns
BOXES_PER_ROW = 10  # control how many per row

for i in range(0, len(filtered), BOXES_PER_ROW):
    row = filtered.iloc[i : i + BOXES_PER_ROW]
    cols = st.columns(len(row))

    for col, (_, r) in zip(cols, row.iterrows()):
        t = r.get("ticker", "")

        stars_val = r.get("greer_star_rating")
        stars = int(stars_val) if pd.notnull(stars_val) else 0

        gv_val = r.get("greer_value_score")
        gv = float(gv_val) if pd.notnull(gv_val) else 0.0

        above50_val = r.get("above_50_count")
        above50 = int(above50_val) if pd.notnull(above50_val) else 0

        # color logic for box background
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
