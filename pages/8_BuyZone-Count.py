# 8_BuyZone-Count.py
# ----------------------------------------------------------
# BuyZone Count Dashboard
# - Tracks BuyZone count over time
# - Tactical 1-Year View
# - Dynamic: Uses market_regime_thresholds for guide bands
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_engine
from datetime import date, timedelta

# Import the unified brain
from market_cycle_utils import get_market_thresholds

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="BuyZone Count Dashboard", layout="wide")

# ----------------------------------------------------------
# Load history
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_buyzone_history(days: int = 365) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT
            date AS summary_date,
            total_tickers AS total_companies,
            buyzone_count,
            buyzone_pct
        FROM buyzone_breadth
        ORDER BY date DESC
        LIMIT {days};
    """
    return pd.read_sql(query, engine)

# ----------------------------------------------------------
# Classification Helper
# ----------------------------------------------------------
def classify_buyzone_count_dynamic(pct: float, t: dict) -> tuple[str, str]:
    if pct < t['p5']:
        return "🔴 Extreme Greed", "Opportunity is scarce. Market is likely overheated."
    if pct < t['p20']:
        return "🟠 Low Opportunity", "Market is strong; selectivity is required."
    if pct < t['p80']:
        return "🔵 Normal Range", "BuyZone count is within a typical working range."
    if pct < t['p95']:
        return "🟢 Elevated Opportunity", "Broad opportunity is building across the universe."
    return "🟡 Extreme Opportunity", "Panic conditions. Major historical opportunity."

# ----------------------------------------------------------
# Main Dashboard
# ----------------------------------------------------------
def main():
    engine = get_engine()
    
    # 1. Fetch Dynamic Thresholds
    thresholds = get_market_thresholds(engine)
    
    # 2. Load Tactical Data (1 Year)
    df = load_buyzone_history(365)

    if df.empty:
        st.warning("buyzone_breadth is empty.")
        st.stop()

    # Prep DataFrame
    df["summary_date"] = pd.to_datetime(df["summary_date"])
    df = df.sort_values("summary_date").reset_index(drop=True)

    # Ensure numeric types
    for col in ["buyzone_count", "buyzone_pct", "total_companies"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Calculate Moving Averages
    df["bz_50dma"] = df["buyzone_count"].rolling(50, min_periods=1).mean()
    df["bz_pct_50dma"] = df["buyzone_pct"].rolling(50, min_periods=1).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    latest_total = int(latest['total_companies'])
    history_rows = len(df)

    # 3. Dynamic Thresholds (Raw counts for Y-axis)
    DYN_P5  = latest_total * (thresholds['p5'] / 100)
    DYN_P20 = latest_total * (thresholds['p20'] / 100)
    DYN_P80 = latest_total * (thresholds['p80'] / 100)

    regime_title, regime_note = classify_buyzone_count_dynamic(latest["buyzone_pct"], thresholds)

    # 4. Trend Logic
    day_delta = latest["buyzone_count"] - prev["buyzone_count"]
    week_ago_idx = max(0, len(df) - 6)
    week_ago_value = df.iloc[week_ago_idx]["buyzone_count"]
    week_delta = latest["buyzone_count"] - week_ago_value

    def trend_label(x: float) -> str:
        if x > 0: return "Rising"
        if x < 0: return "Falling"
        return "Flat"

    # 5. Header & Metrics
    st.title("🎯 BuyZone Count Dashboard")
    st.info(f"{regime_title} • {regime_note}")

    st.caption(
        f"Latest: {latest['summary_date'].strftime('%Y-%m-%d')} • "
        f"Universe: {latest_total} • "
        f"History rows: {history_rows}"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("BuyZone Count", f"{int(latest['buyzone_count'])}", f"{day_delta:+.0f} vs prior day")
    with c2:
        st.metric("BuyZone %", f"{latest['buyzone_pct']:.1f}%", trend_label(day_delta))
    with c3:
        st.metric("50-Day Avg Count", f"{latest['bz_50dma']:.1f}", f"{week_delta:+.0f} vs ~5 days ago")
    with c4:
        st.metric("Status", regime_title.split(" ")[1], trend_label(week_delta))

    st.divider()

    # 6. Main Tactical Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["summary_date"], y=df["buyzone_count"], mode="lines+markers", name="BuyZone Count", line=dict(color="#1E88E5", width=3)))
    fig.add_trace(go.Scatter(x=df["summary_date"], y=df["bz_50dma"], mode="lines", name="50-Day Avg", line=dict(dash="dash", color="#666666")))

    # Guide Bands
    fig.add_hrect(y0=0, y1=DYN_P5, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Extreme Greed")
    fig.add_hrect(y0=DYN_P5, y1=DYN_P20, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Low Opportunity")
    fig.add_hrect(y0=DYN_P20, y1=DYN_P80, fillcolor="blue", opacity=0.05, line_width=0, annotation_text="Normal Range")
    fig.add_hrect(y0=DYN_P80, y1=latest_total, fillcolor="green", opacity=0.08, line_width=0, annotation_text="Broad Opportunity")

    fig.update_layout(
        title="BuyZone Count Over Time (1-Year Tactical View)",
        xaxis_title="Date", yaxis_title="BuyZone Count", height=520,
        yaxis=dict(range=[0, latest_total * 1.1]),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 7. Percentage Chart
    st.subheader("BuyZone % of Universe")
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Scatter(x=df["summary_date"], y=df["buyzone_pct"], mode="lines+markers", name="BuyZone %", line=dict(color="#1E88E5")))
    fig_pct.add_trace(go.Scatter(x=df["summary_date"], y=df["bz_pct_50dma"], mode="lines", name="50-Day Avg %", line=dict(dash="dash", color="#666666")))
    fig_pct.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Date", yaxis_title="BuyZone %")
    st.plotly_chart(fig_pct, use_container_width=True)

    # 8. Data Table
    st.subheader("Recent Readings")
    show = df[["summary_date", "buyzone_count", "buyzone_pct", "bz_50dma", "bz_pct_50dma", "total_companies"]].copy()
    show.columns = ["Date", "BuyZone Count", "BuyZone %", "50-Day Avg Count", "50-Day Avg %", "Universe"]
    show["Date"] = show["Date"].dt.date
    st.dataframe(show.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)

    st.caption("Threshold bands are dynamically calibrated based on 26 years of historical market percentiles.")

if __name__ == "__main__":
    main()