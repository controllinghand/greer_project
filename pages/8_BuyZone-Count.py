# 8_BuyZone-Count.py
# ----------------------------------------------------------
# BuyZone Count Dashboard
# - Tracks BuyZone count over time
# - Source: dashboard_summary_daily
# - Uses practical threshold bands for early history
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_engine

st.set_page_config(page_title="BuyZone Count Dashboard", layout="wide")

# ----------------------------------------------------------
# Load history
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_buyzone_history(days: int = 365) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        f"""
        SELECT
            date AS summary_date,
            total_tickers AS total_companies,
            buyzone_count,
            buyzone_pct
        FROM buyzone_breadth
        ORDER BY date DESC
        LIMIT {days};
        """,
        engine
    )

# 1. Keep the 365 limit but ensure it's calculated from the latest date
df = load_buyzone_history(365)

if df.empty:
    st.warning("dashboard_summary_daily is empty.")
    st.stop()

df["summary_date"] = pd.to_datetime(df["summary_date"])
df = df.sort_values("summary_date").reset_index(drop=True)

df["buyzone_count"] = pd.to_numeric(df["buyzone_count"], errors="coerce").fillna(0)
df["buyzone_pct"] = pd.to_numeric(df["buyzone_pct"], errors="coerce").fillna(0)
df["total_companies"] = pd.to_numeric(df["total_companies"], errors="coerce").fillna(0)

df["bz_50dma"] = df["buyzone_count"].rolling(50, min_periods=1).mean()
df["bz_pct_50dma"] = df["buyzone_pct"].rolling(50, min_periods=1).mean()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# 2. Update Thresholds to be DYNAMIC based on the current Universe size
# This ensures that if you track 900 or 1200 companies, the bands move with you.
latest_total = int(latest['total_companies'])

# Dynamic thresholds based on your GOI percentages (10%, 14%, 46%, 66%)
DYN_LOW      = latest_total * 0.10
DYN_NORMAL   = latest_total * 0.14
DYN_ELEVATED = latest_total * 0.46
DYN_STRESS   = latest_total * 0.66

def classify_buyzone_count_dynamic(count: float, total: int) -> tuple[str, str]:
    pct = (count / total) * 100
    if pct < 10.0:
        return "🔴 Extreme Greed", "Opportunity is scarce. Market is likely overheated."
    if pct < 14.0:
        return "🟠 Low Opportunity", "Market is strong; selectivity is required."
    if pct < 46.0:
        return "🔵 Normal Range", "BuyZone count is within a typical working range."
    return "🟢 Broad Opportunity", "A large group of stocks is attractively positioned."

regime_title, regime_note = classify_buyzone_count_dynamic(
    latest["buyzone_count"], 
    int(latest["total_companies"])
)

# ----------------------------------------------------------
# Practical starter thresholds
# ----------------------------------------------------------
#LOW_THRESHOLD = 280
#NORMAL_THRESHOLD = 360
#ELEVATED_THRESHOLD = 440

#def classify_buyzone_count(value: float) -> tuple[str, str]:
#    if value < LOW_THRESHOLD:
#        return "🟢 Low Participation", "Few stocks are in BuyZone. Market stress looks contained."
#    if value < NORMAL_THRESHOLD:
#        return "🔵Normal Range", "BuyZone count is within a typical working range."
#    if value < ELEVATED_THRESHOLD:
#        return "🟠 Elevated Opportunity", "A larger-than-normal group of stocks is entering BuyZone."
#    return "🔴 Broad Stress", "BuyZone participation is unusually high across the universe."

#regime_title, regime_note = classify_buyzone_count(latest["buyzone_count"])

# ----------------------------------------------------------
# Trend helpers
# ----------------------------------------------------------
day_delta = latest["buyzone_count"] - prev["buyzone_count"]

if len(df) >= 6:
    week_ago_value = df.iloc[-6]["buyzone_count"]
else:
    week_ago_value = df.iloc[0]["buyzone_count"]

week_delta = latest["buyzone_count"] - week_ago_value

def trend_label(x: float) -> str:
    if x > 0:
        return "Rising"
    if x < 0:
        return "Falling"
    return "Flat"

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.title("🎯 BuyZone Count Dashboard")
st.info(f"{regime_title} • {regime_note}")
st.caption(
    f"Latest: {latest['summary_date'].date()} • "
    f"Universe: {int(latest['total_companies'])} • "
    f"History rows: {len(df)}"
)

# ----------------------------------------------------------
# Top metrics
# ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "BuyZone Count",
        f"{int(latest['buyzone_count'])}",
        f"{day_delta:+.0f} vs prior day"
    )

with c2:
    st.metric(
        "BuyZone %",
        f"{latest['buyzone_pct']:.1f}%",
        trend_label(day_delta)
    )

with c3:
    st.metric(
        "50-Day Avg Count",
        f"{latest['bz_50dma']:.1f}",
        f"{week_delta:+.0f} vs ~5 days ago"
    )

with c4:
    st.metric(
        "Status",
        regime_title,
        trend_label(week_delta)
    )

st.divider()

# ----------------------------------------------------------
# Main chart
# ----------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["summary_date"],
        y=df["buyzone_count"],
        mode="lines+markers",
        name="BuyZone Count",
        line=dict(color="#1E88E5")
    )
)

fig.add_trace(
    go.Scatter(
        x=df["summary_date"],
        y=df["bz_50dma"],
        mode="lines",
        name="50-Day Avg",
        line=dict(dash="dash", color="#666666")
    )
)

# Dynamic Threshold guide bands
fig.add_hrect(
    y0=0, y1=DYN_LOW,
    fillcolor="red", opacity=0.08, line_width=0,
    annotation_text="Extreme Greed", annotation_position="top left"
)
fig.add_hrect(
    y0=DYN_LOW, y1=DYN_NORMAL,
    fillcolor="orange", opacity=0.08, line_width=0,
    annotation_text="Low Opportunity", annotation_position="top left"
)
fig.add_hrect(
    y0=DYN_NORMAL, y1=DYN_ELEVATED,
    fillcolor="blue", opacity=0.05, line_width=0,
    annotation_text="Normal Range", annotation_position="top left"
)
fig.add_hrect(
    y0=DYN_ELEVATED, y1=latest_total,
    fillcolor="green", opacity=0.08, line_width=0,
    annotation_text="Elevated Opportunity", annotation_position="top left"
)

fig.update_layout(
    title="BuyZone Count Over Time (1-Year Tactical View)",
    xaxis_title="Date",
    yaxis_title="BuyZone Count",
    height=520,
    margin=dict(l=20, r=20, t=60, b=20),
    yaxis=dict(range=[0, latest_total * 1.1]) # Sets scale relative to current universe
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# BuyZone percent chart
# ----------------------------------------------------------
st.subheader("BuyZone % of Universe")

fig_pct = go.Figure()

fig_pct.add_trace(
    go.Scatter(
        x=df["summary_date"],
        y=df["buyzone_pct"],
        mode="lines+markers",
        name="BuyZone %"
    )
)

fig_pct.add_trace(
    go.Scatter(
        x=df["summary_date"],
        y=df["bz_pct_50dma"],
        mode="lines",
        name="50-Day Avg %",
        line=dict(dash="dash")
    )
)

fig_pct.update_layout(
    height=320,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Date",
    yaxis_title="BuyZone %",
)

st.plotly_chart(fig_pct, use_container_width=True)

# ----------------------------------------------------------
# Recent readings
# ----------------------------------------------------------
st.subheader("Recent Readings")

show = df[["summary_date", "buyzone_count", "buyzone_pct", "bz_50dma", "bz_pct_50dma", "total_companies"]].copy()
show.columns = ["Date", "BuyZone Count", "BuyZone %", "50-Day Avg Count", "50-Day Avg %", "Universe"]
show = show.sort_values("Date", ascending=False)

st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Notes
# ----------------------------------------------------------
st.caption(
    "Threshold bands are provisional while history builds. "
    "After 3-6 months of daily snapshots, these should be replaced with percentile-based ranges."
)