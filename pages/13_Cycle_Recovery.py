# 13_Cycle_Recovery.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from market_cycle_utils import classify_phase_with_confidence
from company_cycle_helpers import enrich_company_cycle_dataframe


# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="Cycle Recovery",
    page_icon="🔄",
)

st.title("🔄 Cycle Recovery")
st.caption("Builds YRRG-26 using Recovery phase companies with strong Greer Company Index support.")


# ----------------------------------------------------------
# Insert custom CSS for styled table
# ----------------------------------------------------------
st.markdown("""
<style>
  .cycle-table {
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
  }
  .cycle-table th, .cycle-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
    white-space: nowrap;
  }
  .cycle-table th {
    background-color: #1976D2;
    color: white;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .cycle-table tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  .cycle-table tr:hover {
    background-color: #f1f1f1;
  }
  .phase-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
  }
  .phase-recovery {
    background-color: #D9F2FA;
    color: #0C5460;
  }
  .metric-pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Clamp a numeric value into a range
# ----------------------------------------------------------
def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


# ----------------------------------------------------------
# Convert a numeric score into emoji + percentage text
# ----------------------------------------------------------
def signal_pct(value: float) -> str:
    x = float(value)

    if x <= 24:
        emoji = "🔴"
    elif x <= 49:
        emoji = "🟡"
    elif x <= 74:
        emoji = "🟢"
    else:
        emoji = "🟩"

    return f"{emoji} {x:.1f}%"


# ----------------------------------------------------------
# Render a phase badge
# ----------------------------------------------------------
def phase_chip(phase_name: str) -> str:
    phase = str(phase_name).strip().upper()

    if phase == "RECOVERY":
        return '<span class="phase-chip phase-recovery">🔵 Recovery</span>'

    return f'<span class="phase-chip">{phase.title()}</span>'


# ----------------------------------------------------------
# Load cycle recovery candidates from dashboard snapshot
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_cycle_recovery_data() -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        WITH latest_sector AS (
            SELECT
                sector,
                summary_date,
                buyzone_pct,
                greer_market_index
            FROM sector_summary_daily
            WHERE summary_date = (
                SELECT MAX(summary_date) FROM sector_summary_daily
            )
        )
        SELECT
            d.snapshot_date,
            d.ticker,
            d.name,
            d.sector,
            d.industry,
            d.greer_star_rating,
            d.greer_value_score,
            d.above_50_count,
            d.greer_yield_score,
            d.buyzone_flag,
            d.fvg_last_direction,
            d.current_price,
            d.gfv_price,
            d.gfv_status,
            s.summary_date AS sector_summary_date,
            s.buyzone_pct AS sector_buyzone_pct,
            (100.0 - s.buyzone_pct) AS sector_direction_pct,
            s.greer_market_index AS sector_greer_market_index
        FROM dashboard_snapshot d
        LEFT JOIN latest_sector s
            ON d.sector = s.sector
        ORDER BY
            d.greer_value_score DESC,
            d.greer_yield_score DESC,
            d.ticker
    """)

    df = pd.read_sql(query, engine)

    if df.empty:
        return df

    df = enrich_company_cycle_dataframe(df, classify_phase_with_confidence)
    df["confidence_pct"] = pd.to_numeric(df["confidence"], errors="coerce") * 100.0
    df["confidence_pct"] = df["confidence_pct"].round(1)

    return df


df = load_cycle_recovery_data()

if df.empty:
    st.warning("No data found. Check dashboard_snapshot and sector_summary_daily.")
    st.stop()

snapshot_date = df["snapshot_date"].iloc[0]


# ----------------------------------------------------------
# Page intro
# ----------------------------------------------------------
st.success(
    "This screener is designed to build the YRRG-26 Recovery Growth portfolio."
)

st.markdown(
    """
**Default recovery rules**
- Phase = **RECOVERY**
- Greer Company Index **≥ 65**
- Opportunity **≥ 70**
- Health **≥ 50**
- Direction **≥ 40**
"""
)

st.caption(f"Latest snapshot: {snapshot_date}  •  Companies loaded: {len(df)}")


# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
f1, f2, f3, f4, f5, f6 = st.columns([2, 1, 1, 1, 1, 1])

with f1:
    selected_sectors = st.multiselect(
        "Filter sectors",
        options=sorted(df["sector"].dropna().unique().tolist()),
        default=sorted(df["sector"].dropna().unique().tolist()),
    )

with f2:
    min_gci = st.slider("Min GCI", 0, 100, 65)

with f3:
    min_opportunity = st.slider("Min Opportunity", 0, 100, 70)

with f4:
    min_health = st.slider("Min Health", 0, 100, 50)

with f5:
    min_direction = st.slider("Min Direction", 0, 100, 40)

with f6:
    top_n = st.selectbox(
        "Rows to show",
        options=[25, 50, 100, 250, 500],
        index=2,
    )

advanced1, advanced2, advanced3 = st.columns(3)

with advanced1:
    only_buyzone = st.checkbox("Only BuyZone", value=False)

with advanced2:
    only_bullish_fvg = st.checkbox("Only Bullish FVG", value=False)

with advanced3:
    min_confidence = st.slider("Min Confidence %", 0, 100, 0)


# ----------------------------------------------------------
# Filter dataframe
# ----------------------------------------------------------
filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["sector"].isin(selected_sectors)]
filtered_df = filtered_df[filtered_df["phase"] == "RECOVERY"]
filtered_df = filtered_df[filtered_df["greer_company_index"] >= min_gci]
filtered_df = filtered_df[filtered_df["opportunity_pct"] >= min_opportunity]
filtered_df = filtered_df[filtered_df["health_pct"] >= min_health]
filtered_df = filtered_df[filtered_df["direction_pct"] >= min_direction]
filtered_df = filtered_df[filtered_df["confidence_pct"] >= min_confidence]

if only_buyzone:
    filtered_df = filtered_df[filtered_df["buyzone_flag"] == True]

if only_bullish_fvg:
    filtered_df = filtered_df[
        filtered_df["fvg_last_direction"].fillna("").str.lower() == "bullish"
    ]


# ----------------------------------------------------------
# Sort options
# ----------------------------------------------------------
sort_choice = st.selectbox(
    "Sort by",
    options=[
        "Greer Company Index",
        "Opportunity %",
        "Health %",
        "Direction %",
        "Confidence %",
        "Greer Value Score",
        "Greer Yield Score",
        "Ticker",
    ],
    index=0,
)

sort_map = {
    "Greer Company Index": ("greer_company_index", False),
    "Opportunity %": ("opportunity_pct", False),
    "Health %": ("health_pct", False),
    "Direction %": ("direction_pct", False),
    "Confidence %": ("confidence_pct", False),
    "Greer Value Score": ("greer_value_score", False),
    "Greer Yield Score": ("greer_yield_score", False),
    "Ticker": ("ticker", True),
}

sort_col, ascending = sort_map[sort_choice]

filtered_df = filtered_df.sort_values(
    by=[sort_col, "confidence_pct", "health_pct", "direction_pct", "ticker"],
    ascending=[ascending, False, False, False, True],
    na_position="last",
).head(top_n)


# ----------------------------------------------------------
# Summary metrics
# ----------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Recovery Candidates", len(filtered_df))

with m2:
    avg_gci = filtered_df["greer_company_index"].mean() if not filtered_df.empty else 0.0
    st.metric("Avg GCI", f"{avg_gci:.1f}")

with m3:
    avg_oppty = filtered_df["opportunity_pct"].mean() if not filtered_df.empty else 0.0
    st.metric("Avg Opportunity", f"{avg_oppty:.1f}%")

with m4:
    avg_conf = filtered_df["confidence_pct"].mean() if not filtered_df.empty else 0.0
    st.metric("Avg Confidence", f"{avg_conf:.1f}%")


# ----------------------------------------------------------
# Display recovery screener table
# ----------------------------------------------------------
st.subheader("Recovery Candidates")
st.caption("Companies currently in Recovery phase meeting the YRRG build rules.")

display_df = filtered_df.copy()

display_df["Health"] = display_df["health_pct"].apply(signal_pct)
display_df["Direction"] = display_df["direction_pct"].apply(signal_pct)
display_df["Opportunity"] = display_df["opportunity_pct"].apply(signal_pct)
display_df["Confidence"] = display_df["confidence_pct"].apply(lambda x: f"{x:.1f}%")
display_df["Phase"] = display_df["phase"].apply(phase_chip)
display_df["Current Price"] = pd.to_numeric(display_df["current_price"], errors="coerce").round(2)
display_df["GFV Price"] = pd.to_numeric(display_df["gfv_price"], errors="coerce").round(2)
display_df["Greer Company Index"] = pd.to_numeric(
    display_df["greer_company_index"], errors="coerce"
).round(2)
display_df["GV Score"] = pd.to_numeric(display_df["greer_value_score"], errors="coerce").round(1)
display_df["YS Score"] = pd.to_numeric(display_df["greer_yield_score"], errors="coerce").round(1)
display_df["Sector Direction %"] = pd.to_numeric(
    display_df["sector_direction_pct"], errors="coerce"
).round(1)

display_df = display_df.rename(
    columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "greer_star_rating": "Stars",
        "fvg_last_direction": "FVG Direction",
        "gfv_status": "GFV Status",
        "transition_risk": "Transition Risk",
    }
)

table_cols = [
    "Ticker",
    "Name",
    "Sector",
    "Industry",
    "Health",
    "Direction",
    "Opportunity",
    "Greer Company Index",
    "Phase",
    "Confidence",
    "Transition Risk",
    "GV Score",
    "YS Score",
    "Current Price",
    "GFV Price",
    "GFV Status",
    "FVG Direction",
    "Sector Direction %",
]

st.markdown(
    display_df[table_cols].to_html(
        index=False,
        escape=False,
        classes="cycle-table",
    ),
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# Download filtered candidates
# ----------------------------------------------------------
st.divider()

csv_df = filtered_df[
    [
        "ticker",
        "name",
        "sector",
        "industry",
        "health_pct",
        "direction_pct",
        "opportunity_pct",
        "greer_company_index",
        "phase",
        "confidence_pct",
        "transition_risk",
        "greer_value_score",
        "greer_yield_score",
        "buyzone_flag",
        "fvg_last_direction",
        "current_price",
        "gfv_price",
        "gfv_status",
        "sector_direction_pct",
    ]
].copy()

csv = csv_df.to_csv(index=False)

st.download_button(
    label="Download Recovery Candidates CSV",
    data=csv,
    file_name="cycle_recovery_candidates.csv",
    mime="text/csv",
)


# ----------------------------------------------------------
# Company drilldown
# ----------------------------------------------------------
st.divider()
st.subheader("Recovery Drilldown")

if filtered_df.empty:
    st.info("No companies currently match the selected recovery rules.")
else:
    company_choices = filtered_df.sort_values(["ticker", "name"])[["ticker", "name"]].drop_duplicates()

    selected_label = st.selectbox(
        "Choose a company",
        options=company_choices.apply(lambda r: f"{r['ticker']} — {r['name']}", axis=1).tolist(),
        index=0,
    )

    selected_ticker = selected_label.split(" — ")[0]
    detail = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("Greer Company Index", f"{detail['greer_company_index']:.2f}")

    with d2:
        st.metric("Health", f"{detail['health_pct']:.1f}%")

    with d3:
        st.metric("Direction", f"{detail['direction_pct']:.1f}%")

    with d4:
        st.metric("Opportunity", f"{detail['opportunity_pct']:.1f}%")

    st.markdown(
        f"""
        **{detail['ticker']} — {detail['name']}**  
        Sector: {detail['sector']}  
        Industry: {detail['industry']}  
        Phase: {detail['phase']}  
        Confidence: {detail['confidence_pct']:.1f}%  
        Transition Risk: {detail['transition_risk']}  
        Current Price: {detail['current_price'] if pd.notnull(detail['current_price']) else 'N/A'}  
        GFV Price: {detail['gfv_price'] if pd.notnull(detail['gfv_price']) else 'N/A'}  
        GFV Status: {detail['gfv_status']}  
        FVG Direction: {detail['fvg_last_direction']}  
        """
    )