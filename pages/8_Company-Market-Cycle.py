# 8_Company-Market-Cycle.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_engine
from market_cycle_utils import (
    classify_phase_with_confidence,
    phase_badge_inline,
    phase_interpretation_text,
    phase_confidence_note,
    get_ranked_phase_scores,
)
from company_cycle_helpers import (
    safe_round,
    compute_health_pct,
    compute_direction_pct,
    compute_opportunity_pct,
    compute_company_index,
    compute_company_buyzone_proxy,
    transition_risk_label,
)

st.set_page_config(
    page_title="Greer Company Market Cycle",
    page_icon="🏢"
)

# ----------------------------------------------------------
# Make filter selections blue
# ----------------------------------------------------------
st.markdown("""
<style>
/* Selected multiselect chips */
span[data-baseweb="tag"] {
    background-color: #5BC0DE !important;   /* blue */
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* X icon inside the chip */
span[data-baseweb="tag"] svg {
    fill: white !important;
}

/* Optional: make the multiselect input area a little cleaner */
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load latest company snapshot + latest sector backdrop
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_company_cycle_data() -> pd.DataFrame:
    engine = get_engine()

    query = """
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
            d.greer_star_rating DESC,
            d.greer_value_score DESC,
            d.greer_yield_score DESC,
            d.ticker;
    """

    return pd.read_sql(query, engine)

df = load_company_cycle_data()

if df.empty:
    st.warning("No company data found. Check dashboard_snapshot and sector_summary_daily.")
    st.stop()

snapshot_date = df["snapshot_date"].iloc[0]

# ----------------------------------------------------------
# UI helpers
# ----------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_round(value, digits=2):
    return round(float(value), digits) if pd.notnull(value) else None

def signal_emoji(value: float) -> str:
    x = float(value)
    if x <= 24:
        return "🔴"
    if x <= 49:
        return "🟡"
    if 49 < x < 51:
        return "🔵"
    if x <= 74:
        return "🟢"
    return "🟩"

def signal_pct(value: float) -> str:
    return f"{signal_emoji(value)} {float(value):.1f}%"

def dial_bucket(value: float) -> str:
    x = float(value)
    if x <= 24:
        return "weak"
    if x <= 49:
        return "mixed"
    if 49 < x < 51:
        return "neutral"
    if x <= 74:
        return "strong"
    return "very_strong"

def dial_label(bucket: str) -> str:
    return {
        "weak": "Weak 🔴",
        "mixed": "Mixed 🟡",
        "neutral": "Neutral ⚪",
        "strong": "Strong 🟢",
        "very_strong": "Very Strong 🟩",
    }.get(bucket, "Mixed 🟡")

def phase_label(phase_name: str) -> str:
    p = str(phase_name).strip().upper()

    if p == "EXPANSION":
        return "🟢 Expansion"
    if p == "EUPHORIA":
        return "🟠 Euphoria"
    if p == "RECOVERY":
        return "🔵 Recovery"
    if p == "CONTRACTION":
        return "🔴 Contraction"

    return "⚪ Unknown"

def transition_risk_label(confidence: float) -> str:
    c = float(confidence)

    if c < 0.35:
        return "⚠ Possible Phase Shift"
    if c < 0.55:
        return "👀 Watch"
    return ""

def render_cycle_strip(current_phase: str):
    phase = str(current_phase).strip().upper()

    phases = ["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION"]

    colors = {
        "RECOVERY": "#5BC0DE",
        "EXPANSION": "#5CB85C",
        "EUPHORIA": "#F0AD4E",
        "CONTRACTION": "#D9534F",
    }

    icons = {
        "RECOVERY": "🩹",
        "EXPANSION": "📈",
        "EUPHORIA": "🔥",
        "CONTRACTION": "📉",
    }

    if phase not in phases:
        phase = phases[0]

    st.markdown(
        """
        <style>
          .cycle-wrap{
            display:flex;
            align-items:center;
            gap:10px;
            flex-wrap:wrap;
            margin-top:6px;
            margin-bottom:6px;
          }
          .cycle-step{
            display:flex;
            align-items:center;
            gap:8px;
            padding:8px 12px;
            border-radius:999px;
            border:1px solid rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.65);
            font-size:12px;
            font-weight:700;
            letter-spacing:0.2px;
            color:#1F2937;
          }
          .cycle-step .dot{
            width:10px; height:10px;
            border-radius:50%;
            display:inline-block;
          }
          .cycle-arrow{
            font-size:18px;
            opacity:0.55;
            user-select:none;
          }
          .cycle-active{
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
            border: 2px solid rgba(0,0,0,0.08);
            transform: translateY(-1px);
          }
          .cycle-sub{
            margin-top:2px;
            opacity:0.75;
            font-size:12px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


    parts = ['<div class="cycle-wrap">']
    for i, p in enumerate(phases):
        is_active = (p == phase)
        dot = colors.get(p, "#999999")
        icon = icons.get(p, "•")
        cls = "cycle-step cycle-active" if is_active else "cycle-step"

        parts.append(
            f"""
            <div class="{cls}">
              <span class="dot" style="background:{dot};"></span>
              <span>{icon} {p.title()}</span>
            </div>
            """
        )

        if i < len(phases) - 1:
            parts.append('<div class="cycle-arrow">→</div>')

    parts.append("</div>")
    parts.append(
        f"""
        <div class="cycle-sub">
          You are here: <b>{phase.title()}</b>
        </div>
        """
    )

    st.markdown("".join(parts), unsafe_allow_html=True)

def render_semicircle_gauge(title: str, value: float, subtitle: str, chart_key: str):
    v = clamp(float(value), 0.0, 100.0)
    b = dial_bucket(v)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999999"},
                "bar": {"color": "#777777", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 24], "color": "#D9534F"},
                    {"range": [25, 49], "color": "#F0AD4E"},
                    {"range": [50, 50], "color": "#DDDDDD"},
                    {"range": [51, 74], "color": "#A6D96A"},
                    {"range": [75, 100], "color": "#5CB85C"},
                ],
                "threshold": {
                    "line": {"color": "#666666", "width": 6},
                    "thickness": 0.8,
                    "value": v,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(color="#222222"),
    )

    st.subheader(title)
    st.caption(subtitle)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key=chart_key,
    )
    st.caption(f"Status: **{dial_label(b)}**")



# ----------------------------------------------------------
# Build company metrics
# ----------------------------------------------------------
company_rows = []

for _, row in df.iterrows():
    health_pct = compute_health_pct(
        row["greer_value_score"],
        row["above_50_count"],
    )

    direction_pct = compute_direction_pct(
        row["buyzone_flag"],
        row["fvg_last_direction"],
        row["sector_direction_pct"],
    )

    opportunity_pct = compute_opportunity_pct(
        row["greer_yield_score"],
        row["gfv_status"],
    )

    gci = compute_company_index(
        health_pct,
        direction_pct,
        opportunity_pct,
    )

    company_buyzone_proxy = compute_company_buyzone_proxy(direction_pct)

    phase, confidence = classify_phase_with_confidence(
        health_pct,
        company_buyzone_proxy,
        opportunity_pct,
    )

    company_rows.append(
        {
            "snapshot_date": row["snapshot_date"],
            "ticker": row["ticker"],
            "name": row["name"],
            "sector": row["sector"],
            "industry": row["industry"],
            "greer_star_rating": row["greer_star_rating"],
            "greer_value_score": safe_round(row["greer_value_score"], 1),
            "above_50_count": row["above_50_count"],
            "greer_yield_score": row["greer_yield_score"],
            "buyzone_flag": row["buyzone_flag"],
            "fvg_last_direction": row["fvg_last_direction"],
            "current_price": safe_round(row["current_price"], 2),
            "gfv_price": safe_round(row["gfv_price"], 2),
            "gfv_status": row["gfv_status"],
            "sector_direction_pct": safe_round(row["sector_direction_pct"], 1),
            "sector_greer_market_index": safe_round(row["sector_greer_market_index"], 2),

            "health_pct": health_pct,
            "direction_pct": direction_pct,
            "opportunity_pct": opportunity_pct,
            "greer_company_index": gci,
            "phase": phase,
            "confidence": round(confidence, 4),
            "transition_risk": transition_risk_label(confidence),
        }
    )

company_df = pd.DataFrame(company_rows)

# ----------------------------------------------------------
# Page header
# ----------------------------------------------------------
st.title("🏢 Greer Company Market Cycle")
st.caption(f"Latest company snapshot: {snapshot_date}  •  Companies: {len(company_df)}")

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])

with f1:
    selected_sectors = st.multiselect(
        "Filter sectors",
        options=sorted(company_df["sector"].dropna().unique().tolist()),
        default=sorted(company_df["sector"].dropna().unique().tolist()),
    )

with f2:
    min_stars = st.selectbox(
        "Minimum Stars",
        options=[0, 1, 2, 3],
        index=0,
    )

with f3:
    only_buyzone = st.checkbox("Only BuyZone", value=False)

with f4:
    phase_filter = st.multiselect(
        "Filter phases",
        options=["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION"],
        default=["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION"],
    )

with f5:
    top_n = st.selectbox(
        "Rows to show",
        options=[25, 50, 100, 250, 500],
        index=2,
    )

filtered_df = company_df.copy()

filtered_df = filtered_df[filtered_df["sector"].isin(selected_sectors)]
filtered_df = filtered_df[filtered_df["greer_star_rating"].fillna(0) >= min_stars]
filtered_df = filtered_df[filtered_df["phase"].isin(phase_filter)]

if only_buyzone:
    filtered_df = filtered_df[filtered_df["buyzone_flag"] == True]

# ----------------------------------------------------------
# Summary table
# ----------------------------------------------------------
st.subheader("Company Summary")
st.caption("Quick scan of company-level health, direction, opportunity, phase, confidence, and transition risk.")

summary_display = filtered_df.copy()

summary_display["Health"] = summary_display["health_pct"].apply(signal_pct)
summary_display["Direction"] = summary_display["direction_pct"].apply(signal_pct)
summary_display["Opportunity"] = summary_display["opportunity_pct"].apply(signal_pct)
summary_display["Phase"] = summary_display["phase"].apply(phase_label)
summary_display["Confidence"] = summary_display["confidence"].apply(lambda x: f"{round(float(x) * 100)}%")

summary_display = summary_display.rename(
    columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "greer_star_rating": "Stars",
        "greer_company_index": "Greer Company Index",
        "greer_value_score": "GV Score",
        "greer_yield_score": "YS Score",
        "sector_direction_pct": "Sector Direction %",
        "transition_risk": "Transition Risk",
        "health_pct": "_health_sort",
        "direction_pct": "_direction_sort",
        "opportunity_pct": "_opportunity_sort",
        "confidence": "_confidence_sort",
    }
)

sort_choice = st.selectbox(
    "Sort companies by",
    options=[
        "Greer Company Index",
        "Health %",
        "Direction %",
        "Opportunity %",
        "Confidence %",
        "Stars",
        "Ticker",
    ],
    index=0,
)

sort_map = {
    "Greer Company Index": ("Greer Company Index", False),
    "Health %": ("_health_sort", False),
    "Direction %": ("_direction_sort", False),
    "Opportunity %": ("_opportunity_sort", False),
    "Confidence %": ("_confidence_sort", False),
    "Stars": ("Stars", False),
    "Ticker": ("Ticker", True),
}

sort_col, ascending = sort_map[sort_choice]

summary_display = summary_display[
    [
        "Ticker",
        "Name",
        "Sector",
        "Industry",
        "Stars",
        "Health",
        "Direction",
        "Opportunity",
        "Greer Company Index",
        "Phase",
        "Confidence",
        "Transition Risk",
        "GV Score",
        "YS Score",
        "Sector Direction %",
        "_health_sort",
        "_direction_sort",
        "_opportunity_sort",
        "_confidence_sort",
    ]
]

summary_display = summary_display.sort_values(
    by=[sort_col, "_confidence_sort", "_health_sort", "_direction_sort", "Ticker"],
    ascending=[ascending, False, False, False, True],
    na_position="last",
).head(top_n)

summary_display = summary_display.drop(
    columns=["_health_sort", "_direction_sort", "_opportunity_sort", "_confidence_sort"]
)

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ----------------------------------------------------------
# Company drilldown
# ----------------------------------------------------------
st.subheader("Company Drilldown")
st.caption("Select a company to view phase details and the three cycle pillars.")

company_choices = filtered_df.sort_values(["ticker", "name"])[["ticker", "name"]].drop_duplicates()

selected_label = st.selectbox(
    "Choose a company",
    options=company_choices.apply(lambda r: f"{r['ticker']} — {r['name']}", axis=1).tolist(),
    index=0 if len(company_choices) > 0 else None,
)

if selected_label:
    selected_ticker = selected_label.split(" — ")[0]
    detail = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

    st.subheader(f"🏷️ {detail['ticker']} — {detail['name']}")
    st.caption(
        f"Sector: {detail['sector']}  •  Industry: {detail['industry']}  •  "
        f"Stars: {detail['greer_star_rating']}  •  "
        f"Greer Company Index: {detail['greer_company_index']:.2f}"
    )

    phase_badge_inline(detail["phase"], detail["confidence"], label_prefix="Company Phase")
    st.caption(phase_interpretation_text(detail["phase"], detail["confidence"]))
    render_cycle_strip(detail["phase"])
    st.caption(
        f"Confidence: **{round(detail['confidence'] * 100)}%**  •  "
        f"{phase_confidence_note(detail['confidence'])}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        render_semicircle_gauge(
            "🟢 Company Health",
            detail["health_pct"],
            "Fundamental quality based on Greer Value Score plus consistency support.",
            chart_key=f"{selected_ticker}_health_gauge",
        )

    with c2:
        render_semicircle_gauge(
            "📉 Company Direction",
            detail["direction_pct"],
            "Blend of company BuyZone posture, FVG direction, and sector direction backdrop.",
            chart_key=f"{selected_ticker}_direction_gauge",
        )

    with c3:
        render_semicircle_gauge(
            "💰 Company Opportunity",
            detail["opportunity_pct"],
            "Blend of Greer Yield Score and GFV valuation status.",
            chart_key=f"{selected_ticker}_opportunity_gauge",
        )

    with st.expander(f"Show underlying company cycle components — {selected_ticker}"):
        ranked_scores = get_ranked_phase_scores(
            detail["health_pct"],
            compute_company_buyzone_proxy(detail["direction_pct"]),
            detail["opportunity_pct"],
        )

        st.write(
            {
                "Ticker": detail["ticker"],
                "Name": detail["name"],
                "Sector": detail["sector"],
                "Industry": detail["industry"],
                "Stars": detail["greer_star_rating"],
                "GV Score": detail["greer_value_score"],
                "Above 50 Count": detail["above_50_count"],
                "YS Score": detail["greer_yield_score"],
                "BuyZone Flag": bool(detail["buyzone_flag"]),
                "FVG Last Direction": detail["fvg_last_direction"],
                "GFV Status": detail["gfv_status"],
                "Current Price": detail["current_price"],
                "GFV Price": detail["gfv_price"],
                "Sector Direction %": detail["sector_direction_pct"],
                "Sector Greer Market Index": detail["sector_greer_market_index"],
                "Health %": detail["health_pct"],
                "Direction %": detail["direction_pct"],
                "Opportunity %": detail["opportunity_pct"],
                "Greer Company Index": detail["greer_company_index"],
                "Phase": detail["phase"],
                "Confidence %": round(detail["confidence"] * 100, 1),
                "Transition Risk": detail["transition_risk"],
                "Phase Scores": {
                    k: round(v, 4) for k, v in ranked_scores
                },
            }
        )