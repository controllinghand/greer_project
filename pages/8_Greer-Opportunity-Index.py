# 8_Greer-Opportunity-Index.py
# ----------------------------------------------------------
# Greer Opportunity Index Page
# - Displays market-wide breadth (BuyZone %)
# - Dynamic: Fetches thresholds from market_regime_thresholds view
# - Displays historical charts, distributions, and extreme dates
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
import altair as alt
from textwrap import dedent
from datetime import date, timedelta

# Import the unified brain
from market_cycle_utils import get_market_thresholds

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Greer Opportunity Index", layout="wide")

# This controls the chart starting point
HISTORY_START_DATE = "2000-01-01"

# ----------------------------------------------------------
# Zone colors (Keep these fixed for UI consistency)
# ----------------------------------------------------------
COLOR_EXTREME_GREED = "#C62828"        # red
COLOR_LOW_OPPORTUNITY = "#EF6C00"      # orange
COLOR_NORMAL_RANGE = "#1565C0"         # blue
COLOR_ELEVATED_OPPORTUNITY = "#2E7D32"  # green
COLOR_EXTREME_OPPORTUNITY = "#D4AF37"  # gold

# ----------------------------------------------------------
# Database connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    return get_engine()

# ----------------------------------------------------------
# Load full Greer Opportunity Index history
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_goi_history() -> pd.DataFrame:
    engine = get_connection()
    query = """
        SELECT
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM buyzone_breadth
        ORDER BY date
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])
    return df

# ----------------------------------------------------------
# Load recent Greer Opportunity Index history
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_recent_goi_history(days: int = 90) -> pd.DataFrame:
    engine = get_connection()
    cutoff_date = date.today() - timedelta(days=days)

    query = text("""
        SELECT
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM buyzone_breadth
        WHERE date >= :cutoff_date
        ORDER BY date
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"cutoff_date": cutoff_date},
        parse_dates=["date"]
    )
    return df

# ----------------------------------------------------------
# Load historical zone distribution (DYNAMIC CASE logic)
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_zone_distribution(t: dict) -> pd.DataFrame:
    engine = get_connection()

    query = text(f"""
        SELECT
            CASE
                WHEN buyzone_pct < :p5 THEN 'Extreme Greed'
                WHEN buyzone_pct < :p20 THEN 'Low Opportunity'
                WHEN buyzone_pct < :p80 THEN 'Normal Range'
                WHEN buyzone_pct < :p95 THEN 'Elevated Opportunity'
                ELSE 'Extreme Opportunity'
            END AS zone_label,
            COUNT(*) AS days_count
        FROM buyzone_breadth
        WHERE date >= :start_date
        GROUP BY 1
    """)

    df = pd.read_sql(query, engine, params={
        "p5": t['p5'],
        "p20": t['p20'],
        "p80": t['p80'],
        "p95": t['p95'],
        "start_date": HISTORY_START_DATE
    })
    return df

# ----------------------------------------------------------
# Load all-time extreme dates
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_extreme_dates(limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_connection()

    high_query = text("""
        SELECT date, buyzone_count, total_tickers, buyzone_pct
        FROM buyzone_breadth
        WHERE date >= :history_start_date
        ORDER BY buyzone_pct DESC, date DESC
        LIMIT :limit
    """)

    low_query = text("""
        SELECT date, buyzone_count, total_tickers, buyzone_pct
        FROM buyzone_breadth
        WHERE date >= :history_start_date
        ORDER BY buyzone_pct ASC, date ASC
        LIMIT :limit
    """)

    params = {"history_start_date": HISTORY_START_DATE, "limit": limit}
    
    df_high = pd.read_sql(high_query, engine, params=params, parse_dates=["date"])
    # FIXED: Removed the incorrect 'low_q =' assignment inside the function call
    df_low = pd.read_sql(low_query, engine, params=params, parse_dates=["date"])

    return df_high, df_low

# ----------------------------------------------------------
# Return zone label, color, and interpretation (DYNAMIC)
# ----------------------------------------------------------
def get_zone_info(pct: float, t: dict):
    if pct < t['p5']:
        return ("Extreme Greed", COLOR_EXTREME_GREED, "Market is overheated. Opportunity is scarce and risk is elevated.")
    elif pct < t['p20']:
        return ("Low Opportunity", COLOR_LOW_OPPORTUNITY, "Market is relatively strong. Good opportunities are limited.")
    elif pct < t['p80']:
        return ("Normal Range", COLOR_NORMAL_RANGE, "Market is in a typical environment. Selectivity matters most here.")
    elif pct < t['p95']:
        return ("Elevated Opportunity", COLOR_ELEVATED_OPPORTUNITY, "Broad opportunity is building. This is usually a more attractive setup.")
    else:
        return ("Extreme Opportunity", COLOR_EXTREME_OPPORTUNITY, "Panic-style conditions. Historically this has marked major opportunity.")

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------
def get_percentile_rank(series: pd.Series, current_value: float) -> float:
    if series.empty: return 0.0
    rank = (series <= current_value).mean() * 100.0
    return round(rank, 1)

def format_zone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    zone_order = ["Extreme Opportunity", "Elevated Opportunity", "Normal Range", "Low Opportunity", "Extreme Greed"]
    df["zone_label"] = pd.Categorical(df["zone_label"], categories=zone_order, ordered=True)
    df = df.sort_values("zone_label").copy()
    if not df.empty and df["days_count"].sum() > 0:
        df["pct_of_days"] = (100.0 * df["days_count"] / df["days_count"].sum()).round(1)
    return df

def format_history_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not out.empty:
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out["buyzone_pct"] = out["buyzone_pct"].round(2)
    return out

def render_zone_badge(label: str, color: str) -> None:
    text_color = "#000000" if color == COLOR_EXTREME_OPPORTUNITY else "#FFFFFF"
    st.markdown(f"""
        <div style="display: inline-block; padding: 0.45rem 0.85rem; border-radius: 999px; 
        background-color: {color}; color: {text_color}; font-weight: 700; font-size: 0.95rem; 
        margin-top: 0.25rem; margin-bottom: 0.5rem;">{label}</div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# Chart Builder (DYNAMIC BANDS)
# ----------------------------------------------------------
def build_goi_chart(df_hist: pd.DataFrame, t: dict) -> alt.Chart:
    bands = pd.DataFrame([
        {"y0": 0, "y1": t['p5'], "color": COLOR_EXTREME_GREED},
        {"y0": t['p5'], "y1": t['p20'], "color": COLOR_LOW_OPPORTUNITY},
        {"y0": t['p20'], "y1": t['p80'], "color": COLOR_NORMAL_RANGE},
        {"y0": t['p80'], "y1": t['p95'], "color": COLOR_ELEVATED_OPPORTUNITY},
        {"y0": t['p95'], "y1": 100, "color": COLOR_EXTREME_OPPORTUNITY},
    ])

    band_chart = alt.Chart(bands).mark_rect(opacity=0.14).encode(
        y=alt.Y("y0:Q", title="BuyZone %"),
        y2="y1:Q",
        color=alt.Color("color:N", scale=None, legend=None)
    ).properties(height=380)

    line_chart = alt.Chart(df_hist).mark_line(color="#1E88E5", strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("buyzone_pct:Q"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("buyzone_pct:Q", title="BuyZone %", format=".2f"),
            alt.Tooltip("buyzone_count:Q", title="BuyZone Count"),
            alt.Tooltip("total_tickers:Q", title="Tracked Companies"),
        ]
    )

    rule_data = pd.DataFrame({"level": [t['p5'], t['p20'], t['p80'], t['p95']]})
    rule_chart = alt.Chart(rule_data).mark_rule(color="#666666", strokeDash=[6, 4], opacity=0.7).encode(y="level:Q")

    return (band_chart + rule_chart + line_chart).resolve_scale(color="independent")

# ----------------------------------------------------------
# Main page layout
# ----------------------------------------------------------
def main():
    engine = get_connection()
    
    # FETCH DYNAMIC BRAIN
    thresholds = get_market_thresholds(engine)

    st.title("Greer Opportunity Index")
    st.caption("A market-wide breadth signal showing the percentage of tracked companies currently in the Greer BuyZone.")

    df = load_goi_history()
    if df.empty:
        st.warning("No data found in buyzone_breadth.")
        return

    df = df.sort_values("date").copy()
    df_hist = df[df["date"] >= pd.Timestamp(HISTORY_START_DATE)].copy()

    latest = df.iloc[-1]
    current_pct = float(latest["buyzone_pct"])
    current_count = int(latest["buyzone_count"])
    total_tickers = int(latest["total_tickers"])
    current_date = latest["date"]

    zone_label, zone_color, zone_text = get_zone_info(current_pct, thresholds)
    percentile_rank = get_percentile_rank(df_hist["buyzone_pct"], current_pct)

    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    with col1:
        st.metric("Greer Opportunity Index", f"{current_pct:.1f}%")
        render_zone_badge(zone_label, zone_color)
        st.caption(zone_text)
    with col2:
        st.metric("Companies in BuyZone", f"{current_count:,}")
    with col3:
        st.metric("Tracked Companies", f"{total_tickers:,}")
    with col4:
        st.metric("Historical Percentile", f"{percentile_rank:.1f}%")
        st.caption(f"As of {current_date.strftime('%Y-%m-%d')}")

    st.divider()

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("### Historical Index")
        chart = build_goi_chart(df_hist, thresholds)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("### Zone Guide")
        
        # We use a custom markdown block for each to create a "legend" feel
        st.markdown(f"<span style='color:{COLOR_EXTREME_OPPORTUNITY};'>●</span> **Extreme Opportunity** (>= {thresholds['p95']:.1f}%)", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{COLOR_ELEVATED_OPPORTUNITY};'>●</span> **Elevated Opportunity** ({thresholds['p80']:.1f}% to < {thresholds['p95']:.1f}%)", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{COLOR_NORMAL_RANGE};'>●</span> **Normal Range** ({thresholds['p20']:.1f}% to < {thresholds['p80']:.1f}%)", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{COLOR_LOW_OPPORTUNITY};'>●</span> **Low Opportunity** ({thresholds['p5']:.1f}% to < {thresholds['p20']:.1f}%)", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{COLOR_EXTREME_GREED};'>●</span> **Extreme Greed** (< {thresholds['p5']:.1f}%)", unsafe_allow_html=True)
        
        st.markdown("---") # Visual separator
        
        st.markdown("### Historical Percentiles")
        st.write(f"95th (Extreme Opp): **{thresholds['p95']:.1f}%**")
        st.write(f"80th (Elevated Opp): **{thresholds['p80']:.1f}%**")
        st.write(f"50th (Median): **{df_hist['buyzone_pct'].median():.1f}%**")
        st.write(f"20th (Low Opp): **{thresholds['p20']:.1f}%**")
        st.write(f"5th (Extreme Greed): **{thresholds['p5']:.1f}%**")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Historical Zone Distribution")
        dist_df = load_zone_distribution(thresholds)
        zone_df = format_zone_distribution(dist_df)
        st.dataframe(zone_df, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("### Last 90 Days")
        recent_df = load_recent_goi_history(90)
        if not recent_df.empty:
            recent_display = recent_df[["date", "buyzone_count", "total_tickers", "buyzone_pct"]].sort_values("date", ascending=False)
            st.dataframe(format_history_table(recent_display), use_container_width=True, hide_index=True)

    st.divider()

    high_df, low_df = load_extreme_dates(10)
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("### Highest Opportunity Dates")
        st.dataframe(format_history_table(high_df), use_container_width=True, hide_index=True)
    with col_d:
        st.markdown("### Lowest Opportunity Dates")
        st.dataframe(format_history_table(low_df), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### How to Read This Page")
    st.markdown(dedent("""
        The **Greer Opportunity Index** measures the percentage of tracked companies currently in the Greer BuyZone.
        - Higher readings mean opportunity is broadening across the market.
        - Lower readings mean fewer companies are attractively positioned.
        """))

if __name__ == "__main__":
    main()