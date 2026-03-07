# 5_Sector-Market-Cycle.py
# ----------------------------------------------------------
# Greer Sector Market Cycle Dashboard
# - Reuses the same logic as 4_Market-Cycle.py
# - Shows sector-level Health / Direction / Opportunity gauges
# - Source: sector_summary_daily
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_engine

st.set_page_config(page_title="Greer Sector Market Cycle", layout="wide")

# ----------------------------------------------------------
# Load latest sector summary rows
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_latest_sector_summary() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
          summary_date,
          sector,
          total_companies,

          gv_gold, gv_green, gv_red,
          ys_gold, ys_green, ys_red,
          gfv_gold, gfv_green, gfv_red, gfv_gray,

          buyzone_count,
          buyzone_pct,
          greer_market_index
        FROM sector_summary_daily
        WHERE summary_date = (
            SELECT MAX(summary_date) FROM sector_summary_daily
        )
        ORDER BY sector;
        """,
        engine
    )

df = load_latest_sector_summary()
if df.empty:
    st.warning("sector_summary_daily is empty. Run build_sector_summary_daily.py to populate it.")
    st.stop()

summary_date = df["summary_date"].iloc[0]

# ----------------------------------------------------------
# Load company drilldown for a selected sector
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_sector_companies(selected_sector: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
          ticker,
          name,
          sector,
          industry,
          greer_star_rating,
          greer_value_score,
          above_50_count,
          greer_yield_score,
          buyzone_flag,
          fvg_last_direction,
          current_price,
          gfv_price,
          gfv_status,
          snapshot_date
        FROM dashboard_snapshot
        WHERE sector = %(sector)s
        ORDER BY
          greer_star_rating DESC,
          greer_value_score DESC,
          greer_yield_score DESC,
          ticker;
        """,
        engine,
        params={"sector": selected_sector},
    )

# ----------------------------------------------------------
# Cycle Phase Rules
# ----------------------------------------------------------
def classify_phase(health: float, buyzone: float, opp: float) -> str:
    """
    health  = GV bullish %
    buyzone = BuyZone %
    opp     = Opportunity % (Yield + GFV bullish avg)
    """
    # EUPHORIA: strong fundamentals + strong trend + expensive (low opportunity)
    if health >= 60 and buyzone <= 12 and opp <= 35:
        return "EUPHORIA"

    # EXPANSION: strong fundamentals + strong trend + not super cheap
    if health >= 60 and buyzone <= 20 and opp > 35:
        return "EXPANSION"

    # RECOVERY: lots of pullbacks (high buyzone) but attractive valuations + fundamentals not broken
    if health >= 45 and buyzone >= 25 and opp >= 55:
        return "RECOVERY"

    # CONTRACTION: fundamentals weak and trend weak (buyzones high)
    if health <= 45 and buyzone >= 25:
        return "CONTRACTION"

    return "TRANSITIONAL"

# ----------------------------------------------------------
# UI helpers
# ----------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def phase_badge_inline(phase_name: str):
    p = str(phase_name).strip().upper()

    if p == "EXPANSION":
        st.success("🟢 Sector Phase: **EXPANSION**")
    elif p == "EUPHORIA":
        st.warning("🟠 Sector Phase: **EUPHORIA**")
    elif p == "RECOVERY":
        st.info("🔵 Sector Phase: **RECOVERY**")
    elif p == "CONTRACTION":
        st.error("🔴 Sector Phase: **CONTRACTION**")
    else:
        st.warning("🟡 Sector Phase: **TRANSITIONAL**")

def signal_emoji(value: float) -> str:
    """
    Match the dial color zones:
    0-24   = 🔴
    25-49  = 🟡
    50     = 🔵
    51-74  = 🟢
    75-100 = 🟩
    """
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
    return f"{signal_emoji(value)} {value:.1f}"

def gv_bucket_label(gv_score: float, above_50_count: float) -> str:
    gv = float(gv_score) if pd.notnull(gv_score) else 0.0
    a50 = int(above_50_count) if pd.notnull(above_50_count) else 0

    if a50 == 6:
        return "🟨 Gold"
    if gv >= 50:
        return "🟢 Green"
    return "🔴 Red"

def ys_bucket_label(ys_score: float) -> str:
    ys = int(ys_score) if pd.notnull(ys_score) else 0

    if ys >= 4:
        return "🟨 Gold"
    if ys >= 2:
        return "🟢 Green"
    return "🔴 Red"

def gfv_bucket_label(gfv_status: str) -> str:
    s = str(gfv_status).strip().lower() if pd.notnull(gfv_status) else "gray"

    if s == "gold":
        return "🟨 Gold"
    if s == "green":
        return "🟢 Green"
    if s == "red":
        return "🔴 Red"
    return "⚪ Gray"

def buyzone_label(flag) -> str:
    return "🟢 Yes" if bool(flag) else "⚪ No"

def safe_round(value, digits=2):
    return round(float(value), digits) if pd.notnull(value) else None

def safe_pct_to_fair_value(current_price, gfv_price):
    if pd.isnull(current_price) or pd.isnull(gfv_price):
        return None

    current_price = float(current_price)
    gfv_price = float(gfv_price)

    if current_price == 0:
        return None

    return round(((gfv_price / current_price) - 1.0) * 100.0, 1)

def render_cycle_strip(current_phase: str):
    phase = str(current_phase).strip().upper()

    phases = ["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION", "TRANSITIONAL"]

    colors = {
        "RECOVERY": "#5BC0DE",
        "EXPANSION": "#5CB85C",
        "EUPHORIA": "#F0AD4E",
        "CONTRACTION": "#D9534F",
        "TRANSITIONAL": "#FFD966",
    }

    icons = {
        "RECOVERY": "🩹",
        "EXPANSION": "📈",
        "EUPHORIA": "🔥",
        "CONTRACTION": "📉",
        "TRANSITIONAL": "🟡",
    }

    if phase not in phases:
        phase = "TRANSITIONAL"

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

def phase_interpretation(phase_name: str):
    p = str(phase_name).strip().upper()

    if p == "EXPANSION":
        st.caption(
            "Strong fundamentals and strong market direction. "
            "The majority of companies in this sector are growing and trends remain healthy."
        )

    elif p == "EUPHORIA":
        st.caption(
            "Fundamentals and trends remain strong but valuation opportunities are limited. "
            "This sector may be approaching a late-cycle phase."
        )

    elif p == "RECOVERY":
        st.caption(
            "Valuations are attractive and pullbacks are common, but fundamentals remain intact. "
            "This sector may be emerging from a correction."
        )

    elif p == "CONTRACTION":
        st.caption(
            "Fundamentals are weakening and many companies are in pullback zones. "
            "This sector is likely experiencing broad deterioration."
        )

    else:
        st.caption(
            "Sector signals are mixed. Fundamentals and trends remain reasonable, "
            "but valuation opportunities and momentum are not clearly aligned."
        )

def dial_bucket(value: float) -> str:
    """
    Shared bucket logic for the 3 dials (0-100):
    - 0-24  : Weak
    - 25-49 : Mixed
    - 50    : Neutral-ish
    - 51-74 : Strong
    - 75-100: Very Strong
    """
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
# Build display metrics for each sector
# ----------------------------------------------------------
sector_rows = []

for _, row in df.iterrows():
    total = int(row["total_companies"]) if pd.notnull(row["total_companies"]) else 0

    gv_bullish_pct = (
        ((int(row["gv_green"]) + int(row["gv_gold"])) / total) * 100.0
        if total else 0.0
    )

    buyzone_pct = float(row["buyzone_pct"]) if pd.notnull(row["buyzone_pct"]) else 0.0
    direction_pct = max(0.0, min(100.0, 100.0 - buyzone_pct))

    ys_bullish_pct = (
        ((int(row["ys_green"]) + int(row["ys_gold"])) / total) * 100.0
        if total else 0.0
    )

    gfv_bullish_pct = (
        ((int(row["gfv_green"]) + int(row["gfv_gold"])) / total) * 100.0
        if total else 0.0
    )

    opportunity_pct = (ys_bullish_pct + gfv_bullish_pct) / 2.0
    phase = classify_phase(gv_bullish_pct, buyzone_pct, opportunity_pct)

    sector_rows.append(
        {
            "sector": row["sector"],
            "total_companies": total,
            "health_pct": round(gv_bullish_pct, 1),
            "direction_pct": round(direction_pct, 1),
            "opportunity_pct": round(opportunity_pct, 1),
            "buyzone_pct": round(buyzone_pct, 1),
            "ys_bullish_pct": round(ys_bullish_pct, 1),
            "gfv_bullish_pct": round(gfv_bullish_pct, 1),
            "greer_market_index": round(float(row["greer_market_index"]), 2) if pd.notnull(row["greer_market_index"]) else 0.0,
            "phase": phase,
        }
    )

sector_df = pd.DataFrame(sector_rows)

# ----------------------------------------------------------
# Page Header
# ----------------------------------------------------------
st.title("🏭 Greer Sector Market Cycle")
st.caption(f"Latest sector snapshot: {summary_date}  •  Sectors: {len(sector_df)}")

# ----------------------------------------------------------
# Top Summary Table
# ----------------------------------------------------------
st.subheader("Sector Summary")
st.caption("Quick scan of sector-level health, direction, opportunity, and phase.")

summary_display = sector_df.copy()

summary_display["Health"] = summary_display["health_pct"].apply(signal_pct)
summary_display["Direction"] = summary_display["direction_pct"].apply(signal_pct)
summary_display["Opportunity"] = summary_display["opportunity_pct"].apply(signal_pct)

summary_display = summary_display[
    [
        "sector",
        "total_companies",
        "Health",
        "Direction",
        "Opportunity",
        "greer_market_index",
        "phase",
        "health_pct",
        "direction_pct",
        "opportunity_pct",
    ]
].copy()

summary_display = summary_display.rename(
    columns={
        "sector": "Sector",
        "total_companies": "Tickers",
        "greer_market_index": "Greer Market Index",
        "phase": "Phase",
        "health_pct": "_health_sort",
        "direction_pct": "_direction_sort",
        "opportunity_pct": "_opportunity_sort",
    }
)

summary_display = summary_display.sort_values(
    by=["Greer Market Index", "_health_sort", "_direction_sort"],
    ascending=[False, False, False]
).drop(columns=["_health_sort", "_direction_sort", "_opportunity_sort"])

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ----------------------------------------------------------
# Sector Drilldown
# ----------------------------------------------------------
st.subheader("Sector Drilldown")
st.caption("Select a sector to view the companies that make up that sector.")

drill_c1, drill_c2, drill_c3, drill_c4 = st.columns([2, 1, 1, 1])

with drill_c1:
    drill_sector = st.selectbox(
        "Choose a sector",
        options=sorted(sector_df["sector"].unique().tolist()),
        index=0,
        key="sector_drilldown_select",
    )

with drill_c2:
    only_3_star = st.checkbox("Only 3-Star", value=False)

with drill_c3:
    only_buyzone = st.checkbox("Only BuyZone", value=False)

with drill_c4:
    only_gfv_bullish = st.checkbox("Only GFV Green/Gold", value=False)

companies_df = load_sector_companies(drill_sector)

if companies_df.empty:
    st.info("No companies found for that sector.")
else:
    companies_df = companies_df.copy()

    companies_df["gv_bucket"] = companies_df.apply(
        lambda r: gv_bucket_label(r["greer_value_score"], r["above_50_count"]),
        axis=1
    )

    companies_df["ys_bucket"] = companies_df["greer_yield_score"].apply(ys_bucket_label)
    companies_df["gfv_bucket"] = companies_df["gfv_status"].apply(gfv_bucket_label)
    companies_df["buyzone"] = companies_df["buyzone_flag"].apply(buyzone_label)

    companies_df["gfv_upside_pct"] = companies_df.apply(
        lambda r: safe_pct_to_fair_value(r["current_price"], r["gfv_price"]),
        axis=1
    )

    companies_df["greer_value_score"] = companies_df["greer_value_score"].apply(lambda x: safe_round(x, 1))
    companies_df["current_price"] = companies_df["current_price"].apply(lambda x: safe_round(x, 2))
    companies_df["gfv_price"] = companies_df["gfv_price"].apply(lambda x: safe_round(x, 2))

    if only_3_star:
        companies_df = companies_df[companies_df["greer_star_rating"] == 3]

    if only_buyzone:
        companies_df = companies_df[companies_df["buyzone_flag"] == True]

    if only_gfv_bullish:
        companies_df = companies_df[
            companies_df["gfv_status"].astype("string").str.lower().isin(["green", "gold"])
        ]

    drill_sort = st.selectbox(
        "Sort companies by",
        options=[
            "Star Rating",
            "Greer Value Score",
            "Greer Yield Score",
            "GFV Upside %",
            "Ticker",
        ],
        index=0,
        key="sector_company_sort",
    )

    drill_sort_map = {
        "Star Rating": (["greer_star_rating", "greer_value_score", "greer_yield_score", "ticker"], [False, False, False, True]),
        "Greer Value Score": (["greer_value_score", "greer_star_rating", "ticker"], [False, False, True]),
        "Greer Yield Score": (["greer_yield_score", "greer_value_score", "ticker"], [False, False, True]),
        "GFV Upside %": (["gfv_upside_pct", "greer_value_score", "ticker"], [False, False, True]),
        "Ticker": (["ticker"], [True]),
    }

    drill_sort_cols, drill_sort_asc = drill_sort_map[drill_sort]
    companies_df = companies_df.sort_values(
        by=drill_sort_cols,
        ascending=drill_sort_asc,
        na_position="last"
    )

    st.caption(f"{drill_sector}: {len(companies_df)} companies shown")

    display_df = companies_df[
        [
            "ticker",
            "name",
            "industry",
            "greer_star_rating",
            "greer_value_score",
            "gv_bucket",
            "greer_yield_score",
            "ys_bucket",
            "buyzone",
            "fvg_last_direction",
            "current_price",
            "gfv_price",
            "gfv_bucket",
            "gfv_upside_pct",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "industry": "Industry",
            "greer_star_rating": "Stars",
            "greer_value_score": "GV Score",
            "gv_bucket": "GV",
            "greer_yield_score": "YS Score",
            "ys_bucket": "YS",
            "buyzone": "BuyZone",
            "fvg_last_direction": "FVG Dir",
            "current_price": "Price",
            "gfv_price": "GFV",
            "gfv_bucket": "GFV Status",
            "gfv_upside_pct": "GFV Upside %",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ----------------------------------------------------------
# Optional Filter / Sort
# ----------------------------------------------------------
c1, c2 = st.columns([2, 1])

with c1:
    selected_sectors = st.multiselect(
        "Filter sectors",
        options=sorted(sector_df["sector"].tolist()),
        default=sorted(sector_df["sector"].tolist()),
    )

with c2:
    sort_by = st.selectbox(
        "Sort sector cards by",
        options=[
            "Sector",
            "Greer Market Index",
            "Health %",
            "Direction %",
            "Opportunity %",
        ],
        index=1,
    )

filtered_df = sector_df[sector_df["sector"].isin(selected_sectors)].copy()

sort_map = {
    "Sector": ("sector", True),
    "Greer Market Index": ("greer_market_index", False),
    "Health %": ("health_pct", False),
    "Direction %": ("direction_pct", False),
    "Opportunity %": ("opportunity_pct", False),
}

sort_col, ascending = sort_map[sort_by]
filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

# ----------------------------------------------------------
# Sector Cards
# ----------------------------------------------------------
for _, row in filtered_df.iterrows():
    sector = row["sector"]
    total = int(row["total_companies"])
    health_pct = float(row["health_pct"])
    direction_pct = float(row["direction_pct"])
    opportunity_pct = float(row["opportunity_pct"])
    buyzone_pct = float(row["buyzone_pct"])
    ys_bullish_pct = float(row["ys_bullish_pct"])
    gfv_bullish_pct = float(row["gfv_bullish_pct"])
    phase = row["phase"]
    gmi = float(row["greer_market_index"])

    st.subheader(f"🏷️ {sector}")
    st.caption(f"Universe: {total} tickers  •  Greer Market Index: {gmi:.2f}")

    phase_badge_inline(phase)
    phase_interpretation(phase)
    render_cycle_strip(phase)

    c1, c2, c3 = st.columns(3)

    with c1:
        render_semicircle_gauge(
            "🟢 Sector Health",
            health_pct,
            "Greer Value breadth: % of tickers in this sector that are GV Green + Gold.",
            chart_key=f"{sector}_health_gauge",
        )

    with c2:
        render_semicircle_gauge(
            "📉 Sector Direction",
            direction_pct,
            "Direction strength = 100 − BuyZone %. Higher means fewer pullbacks / stronger trend.",
            chart_key=f"{sector}_direction_gauge",
        )

    with c3:
        render_semicircle_gauge(
            "💰 Sector Opportunity",
            opportunity_pct,
            "Opportunity breadth = avg(% Yield bullish, % GFV bullish). Higher means more undervaluation.",
            chart_key=f"{sector}_opportunity_gauge",
        )

    with st.expander(f"Show underlying sector breadth components — {sector}"):
        st.write(
            {
                "Sector": sector,
                "Total Companies": total,
                "GV Bullish % (Green+Gold)": round(health_pct, 2),
                "BuyZone %": round(buyzone_pct, 2),
                "Direction % (100 - BuyZone)": round(direction_pct, 2),
                "Yield Bullish % (Green+Gold)": round(ys_bullish_pct, 2),
                "GFV Bullish % (Green+Gold)": round(gfv_bullish_pct, 2),
                "Opportunity % (avg Yield+GFV)": round(opportunity_pct, 2),
                "Greer Market Index": round(gmi, 2),
                "Phase": phase,
            }
        )

    st.divider()