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
            summary_date,
            total_companies,
            buyzone_count,
            buyzone_pct
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT {days};
        """,
        engine
    )

df = load_buyzone_history(365)

if df.empty:
    st.warning("dashboard_summary_daily is empty.")
    st.stop()

df["summary_date"] = pd.to_datetime(df["summary_date"])
df = df.sort_values("summary_date").reset_index(drop=True)

df["buyzone_count"] = pd.to_numeric(df["buyzone_count"], errors="coerce").fillna(0)
df["buyzone_pct"] = pd.to_numeric(df["buyzone_pct"], errors="coerce").fillna(0)
df["total_companies"] = pd.to_numeric(df["total_companies"], errors="coerce").fillna(0)

df["bz_7dma"] = df["buyzone_count"].rolling(7, min_periods=1).mean()
df["bz_pct_7dma"] = df["buyzone_pct"].rolling(7, min_periods=1).mean()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ----------------------------------------------------------
# Practical starter thresholds
# ----------------------------------------------------------
LOW_THRESHOLD = 280
NORMAL_THRESHOLD = 360
ELEVATED_THRESHOLD = 440

def classify_buyzone_count(value: float) -> tuple[str, str]:
    if value < LOW_THRESHOLD:
        return "🟢 Low Participation", "Few stocks are in BuyZone. Market stress looks contained."
    if value < NORMAL_THRESHOLD:
        return "🟡 Normal Range", "BuyZone count is within a typical working range."
    if value < ELEVATED_THRESHOLD:
        return "🟠 Elevated Opportunity", "A larger-than-normal group of stocks is entering BuyZone."
    return "🔴 Broad Stress", "BuyZone participation is unusually high across the universe."

regime_title, regime_note = classify_buyzone_count(latest["buyzone_count"])

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
        "7-Day Avg Count",
        f"{latest['bz_7dma']:.1f}",
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
        name="BuyZone Count"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["summary_date"],
        y=df["bz_7dma"],
        mode="lines",
        name="7-Day Avg",
        line=dict(dash="dash")
    )
)

# Threshold guide bands
fig.add_hrect(
    y0=0, y1=LOW_THRESHOLD,
    fillcolor="green", opacity=0.06, line_width=0,
    annotation_text="Low", annotation_position="top left"
)
fig.add_hrect(
    y0=LOW_THRESHOLD, y1=NORMAL_THRESHOLD,
    fillcolor="yellow", opacity=0.06, line_width=0,
    annotation_text="Normal", annotation_position="top left"
)
fig.add_hrect(
    y0=NORMAL_THRESHOLD, y1=ELEVATED_THRESHOLD,
    fillcolor="orange", opacity=0.06, line_width=0,
    annotation_text="Elevated", annotation_position="top left"
)
fig.add_hrect(
    y0=ELEVATED_THRESHOLD, y1=max(int(df["total_companies"].max()), ELEVATED_THRESHOLD + 50),
    fillcolor="red", opacity=0.06, line_width=0,
    annotation_text="Broad Stress", annotation_position="top left"
)

fig.update_layout(
    title="BuyZone Count Over Time",
    xaxis_title="Date",
    yaxis_title="BuyZone Count",
    height=520,
    margin=dict(l=20, r=20, t=60, b=20),
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
        y=df["bz_pct_7dma"],
        mode="lines",
        name="7-Day Avg %",
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

show = df[["summary_date", "buyzone_count", "buyzone_pct", "bz_7dma", "bz_pct_7dma", "total_companies"]].copy()
show.columns = ["Date", "BuyZone Count", "BuyZone %", "7-Day Avg Count", "7-Day Avg %", "Universe"]
show = show.sort_values("Date", ascending=False)

st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Notes
# ----------------------------------------------------------
st.caption(
    "Threshold bands are provisional while history builds. "
    "After 3-6 months of daily snapshots, these should be replaced with percentile-based ranges."
)