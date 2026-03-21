# pages/8_Greer-Opportunity-Index.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
import altair as alt
from textwrap import dedent


# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Greer Opportunity Index", layout="wide")


HISTORY_START_DATE = "2012-01-01"

# ----------------------------------------------------------
# Zone thresholds
# Based on historical percentiles from buyzone_breadth
# ----------------------------------------------------------
P5 = 9.8806
P20 = 13.9130
P50 = 28.8986
P80 = 46.0369
P95 = 65.8722

# ----------------------------------------------------------
# Zone cutoffs
# 4 cutoffs create 5 zones
# ----------------------------------------------------------
ZONE_CUT_1 = 10.0
ZONE_CUT_2 = 14.0
ZONE_CUT_3 = 46.0
ZONE_CUT_4 = 66.0

ZONE_EXTREME_GREED = 10.0 
ZONE_LOW_OPPORTUNITY = 14.0 
ZONE_NORMAL = 46.0 
ZONE_ELEVATED = 66.0

# ----------------------------------------------------------
# Zone colors
# ----------------------------------------------------------
COLOR_EXTREME_GREED = "#C62828"       # red
COLOR_LOW_OPPORTUNITY = "#EF6C00"     # orange
COLOR_NORMAL_RANGE = "#1565C0"        # blue
COLOR_ELEVATED_OPPORTUNITY = "#2E7D32"  # green
COLOR_EXTREME_OPPORTUNITY = "#D4AF37" # gold


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
from datetime import date, timedelta
from sqlalchemy import text

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
# Load historical zone distribution
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_zone_distribution() -> pd.DataFrame:
    engine = get_connection()

    query = """
        SELECT
            CASE
                WHEN buyzone_pct < 10 THEN 'Extreme Greed'
                WHEN buyzone_pct < 14 THEN 'Low Opportunity'
                WHEN buyzone_pct < 46 THEN 'Normal Range'
                WHEN buyzone_pct < 66 THEN 'Elevated Opportunity'
                ELSE 'Extreme Opportunity'
            END AS zone_label,
            COUNT(*) AS days_count
        FROM buyzone_breadth
        WHERE date >= DATE '2012-01-01'
        GROUP BY 1
    """

    df = pd.read_sql(query, engine)
    return df


# ----------------------------------------------------------
# Load historical percentile summary
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_percentile_summary() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            percentile_cont(0.05) WITHIN GROUP (ORDER BY buyzone_pct) AS p5,
            percentile_cont(0.20) WITHIN GROUP (ORDER BY buyzone_pct) AS p20,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY buyzone_pct) AS p50,
            percentile_cont(0.80) WITHIN GROUP (ORDER BY buyzone_pct) AS p80,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY buyzone_pct) AS p95
        FROM buyzone_breadth
        WHERE date >= :history_start_date
    """)

    return pd.read_sql(
        query,
        engine,
        params={"history_start_date": HISTORY_START_DATE}
    )


# ----------------------------------------------------------
# Load all-time extreme dates
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_extreme_dates(limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_connection()

    high_query = text("""
        SELECT
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM buyzone_breadth
        WHERE date >= :history_start_date
        ORDER BY buyzone_pct DESC, date DESC
        LIMIT :limit
    """)

    low_query = text("""
        SELECT
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM buyzone_breadth
        WHERE date >= :history_start_date
        ORDER BY buyzone_pct ASC, date ASC
        LIMIT :limit
    """)

    df_high = pd.read_sql(
        high_query,
        engine,
        params={"history_start_date": HISTORY_START_DATE, "limit": limit},
        parse_dates=["date"]
    )

    df_low = pd.read_sql(
        low_query,
        engine,
        params={"history_start_date": HISTORY_START_DATE, "limit": limit},
        parse_dates=["date"]
    )

    return df_high, df_low


# ----------------------------------------------------------
# Return zone label, color, and interpretation from current pct
# ----------------------------------------------------------
def get_zone_info(pct: float):
    if pct < ZONE_CUT_1:
        return (
            "Extreme Greed",
            COLOR_EXTREME_GREED,
            "Market is overheated. Opportunity is scarce and risk is elevated.",
        )
    elif pct < ZONE_CUT_2:
        return (
            "Low Opportunity",
            COLOR_LOW_OPPORTUNITY,
            "Market is relatively strong. Good opportunities are limited.",
        )
    elif pct < ZONE_CUT_3:
        return (
            "Normal Range",
            COLOR_NORMAL_RANGE,
            "Market is in a typical environment. Selectivity matters most here.",
        )
    elif pct < ZONE_CUT_4:
        return (
            "Elevated Opportunity",
            COLOR_ELEVATED_OPPORTUNITY,
            "Broad opportunity is building. This is usually a more attractive setup.",
        )
    else:
        return (
            "Extreme Opportunity",
            COLOR_EXTREME_OPPORTUNITY,
            "Panic-style conditions. Historically this has marked major opportunity.",
        )


# ----------------------------------------------------------
# Build a percentile rank for the current reading
# ----------------------------------------------------------
def get_percentile_rank(series: pd.Series, current_value: float) -> float:
    if series.empty:
        return 0.0

    rank = (series <= current_value).mean() * 100.0
    return round(rank, 1)


# ----------------------------------------------------------
# Format zone distribution table
# ----------------------------------------------------------
def format_zone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    zone_order = [
        "Extreme Opportunity",
        "Elevated Opportunity",
        "Normal Range",
        "Low Opportunity",
        "Extreme Greed",
    ]

    df["zone_label"] = pd.Categorical(df["zone_label"], categories=zone_order, ordered=True)
    df = df.sort_values("zone_label").copy()
    df["pct_of_days"] = (100.0 * df["days_count"] / df["days_count"].sum()).round(1)

    return df


# ----------------------------------------------------------
# Apply friendly formatting to history tables
# ----------------------------------------------------------
def format_history_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["buyzone_pct"] = out["buyzone_pct"].round(2)
    return out


# ----------------------------------------------------------
# Render a colored zone badge
# ----------------------------------------------------------
def render_zone_badge(label: str, color: str) -> None:
    text_color = "#000000" if color == COLOR_EXTREME_OPPORTUNITY else "#FFFFFF"

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background-color: {color};
            color: {text_color};
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------
# Build historical chart with colored zone background bands
# ----------------------------------------------------------
def build_goi_chart(df_hist: pd.DataFrame) -> alt.Chart:
    chart_data = df_hist.copy()

    bands = pd.DataFrame([
        {"y0": 0, "y1": ZONE_CUT_1, "zone": "Extreme Greed", "color": COLOR_EXTREME_GREED},
        {"y0": ZONE_CUT_1, "y1": ZONE_CUT_2, "zone": "Low Opportunity", "color": COLOR_LOW_OPPORTUNITY},
        {"y0": ZONE_CUT_2, "y1": ZONE_CUT_3, "zone": "Normal Range", "color": COLOR_NORMAL_RANGE},
        {"y0": ZONE_CUT_3, "y1": ZONE_CUT_4, "zone": "Elevated Opportunity", "color": COLOR_ELEVATED_OPPORTUNITY},
        {"y0": ZONE_CUT_4, "y1": 80, "zone": "Extreme Opportunity", "color": COLOR_EXTREME_OPPORTUNITY},
    ])

    band_chart = (
        alt.Chart(bands)
        .mark_rect(opacity=0.14)
        .encode(
            y=alt.Y("y0:Q", title="BuyZone %"),
            y2="y1:Q",
            color=alt.Color("color:N", scale=None, legend=None),
        )
        .properties(height=380)
    )

    line_chart = (
        alt.Chart(chart_data)
        .mark_line(color="#1E88E5", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("buyzone_pct:Q", title="BuyZone %"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("buyzone_pct:Q", title="BuyZone %", format=".2f"),
                alt.Tooltip("buyzone_count:Q", title="BuyZone Count"),
                alt.Tooltip("total_tickers:Q", title="Tracked Companies"),
            ],
        )
    )

    rule_data = pd.DataFrame({
        "level": [ZONE_CUT_1, ZONE_CUT_2, ZONE_CUT_3, ZONE_CUT_4]
    })

    rule_chart = (
        alt.Chart(rule_data)
        .mark_rule(color="#666666", strokeDash=[6, 4], opacity=0.7)
        .encode(y="level:Q")
    )

    return (band_chart + rule_chart + line_chart).resolve_scale(color="independent")

# ----------------------------------------------------------
# Main page layout
# ----------------------------------------------------------
def main():
    st.title("Greer Opportunity Index")
    st.caption(
        "A market-wide breadth signal showing the percentage of tracked companies currently in the Greer BuyZone."
    )

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

    zone_label, zone_color, zone_text = get_zone_info(current_pct)
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
        chart = build_goi_chart(df_hist)
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown("### Zone Guide")
        st.markdown(
            f"""
            **Extreme Opportunity**  
            `>= {ZONE_CUT_4:.0f}%`

            **Elevated Opportunity**  
            `{ZONE_CUT_3:.0f}% to < {ZONE_CUT_4:.0f}%`

            **Normal Range**  
            `{ZONE_CUT_2:.0f}% to < {ZONE_CUT_3:.0f}%`

            **Low Opportunity**  
            `{ZONE_CUT_1:.0f}% to < {ZONE_CUT_2:.0f}%`

            **Extreme Greed**  
            `< {ZONE_CUT_1:.0f}%`
            """
        )

        st.markdown("### Historical Percentiles")
        pct_df = load_percentile_summary()
        if not pct_df.empty:
            row = pct_df.iloc[0]
            st.write(f"5th percentile: **{row['p5']:.1f}%**")
            st.write(f"20th percentile: **{row['p20']:.1f}%**")
            st.write(f"50th percentile: **{row['p50']:.1f}%**")
            st.write(f"80th percentile: **{row['p80']:.1f}%**")
            st.write(f"95th percentile: **{row['p95']:.1f}%**")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Historical Zone Distribution")
        zone_df = format_zone_distribution(load_zone_distribution())
        st.dataframe(zone_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### Last 90 Days")
        recent_df = load_recent_goi_history(90).copy()
        if not recent_df.empty:
            recent_display = recent_df[["date", "buyzone_count", "total_tickers", "buyzone_pct"]].copy()
            recent_display["date"] = recent_display["date"].dt.strftime("%Y-%m-%d")
            recent_display["buyzone_pct"] = recent_display["buyzone_pct"].round(2)
            st.dataframe(recent_display.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

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
    st.markdown(
        """
        The **Greer Opportunity Index** measures the percentage of tracked companies currently in the Greer BuyZone.

        - Higher readings mean opportunity is broadening across the market.
        - Lower readings mean fewer companies are attractively positioned.
        - Extreme high readings usually appear during panic or heavy stress events.
        - Extreme low readings usually appear when the market is overheated.

        This page is designed to complement your:
        - **Market Cycle** page for macro context
        - **Sector Cycle** page for sector selection
        - **Company Cycle** pages for individual stock decisions
        """
    )


if __name__ == "__main__":
    main()