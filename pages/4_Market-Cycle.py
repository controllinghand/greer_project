# 4_Market-Cycle.py
# ----------------------------------------------------------
# Greer Market Cycle Dashboard
# - 3 semicircle gauges: Health (GV), Direction (100 - BuyZone), Opportunity (Yield + GFV)
# - Cycle Phase classification: Recovery / Expansion / Euphoria / Contraction
# - Mixed states are expressed through phase confidence instead of a separate Transitional phase
# - Source: dashboard_summary_daily
# ----------------------------------------------------------

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

st.set_page_config(page_title="Greer Market Cycle", layout="wide")

# ----------------------------------------------------------
# Load latest daily summary row
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_latest_summary() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
          summary_date,
          total_companies,

          gv_gold, gv_green, gv_red,
          ys_gold, ys_green, ys_red,
          gfv_gold, gfv_green, gfv_red, gfv_gray,

          buyzone_count,
          buyzone_pct
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT 1;
        """,
        engine
    )

df = load_latest_summary()
if df.empty:
    st.warning("dashboard_summary_daily is empty. Run build_dashboard_summary_daily.py to populate it.")
    st.stop()

row = df.iloc[0]
summary_date = row["summary_date"]
total = int(row["total_companies"])

# ----------------------------------------------------------
# Compute the 3 gauges
# ----------------------------------------------------------
gv_bullish_pct = ((int(row["gv_green"]) + int(row["gv_gold"])) / total * 100.0) if total else 0.0

buyzone_pct = float(row["buyzone_pct"]) if pd.notnull(row["buyzone_pct"]) else 0.0
direction_pct = max(0.0, min(100.0, 100.0 - buyzone_pct))

ys_bullish_pct = ((int(row["ys_green"]) + int(row["ys_gold"])) / total * 100.0) if total else 0.0
gfv_bullish_pct = ((int(row["gfv_green"]) + int(row["gfv_gold"])) / total * 100.0) if total else 0.0
opportunity_pct = (ys_bullish_pct + gfv_bullish_pct) / 2.0

phase, confidence = classify_phase_with_confidence(
    gv_bullish_pct,
    buyzone_pct,
    opportunity_pct,
)

# ----------------------------------------------------------
# UI helpers
# ----------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct_str(x: float) -> str:
    return f"{x:.1f}%"

# ----------------------------------------------------------
# Slick Market Cycle Strip (Recovery → Expansion → Euphoria → Contraction)
# ----------------------------------------------------------
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

def render_semicircle_gauge(title: str, value: float, subtitle: str):
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
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Status: **{dial_label(b)}**")

# ----------------------------------------------------------
# Page Header
# ----------------------------------------------------------
st.title("🧭 Greer Market Cycle")
st.caption(f"Latest snapshot: {summary_date}  •  Universe: {total} tickers")

phase_badge_inline(phase, confidence, label_prefix="Greer Market Cycle Phase")
st.caption(phase_interpretation_text(phase, confidence))
render_cycle_strip(phase)
st.caption(
    f"Confidence: **{round(confidence * 100)}%**  •  "
    f"{phase_confidence_note(confidence)}"
)
st.divider()

# ----------------------------------------------------------
# 3 Semicircle Dials
# ----------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    render_semicircle_gauge(
        "🟢 Market Health",
        gv_bullish_pct,
        "Greer Value breadth: % of tickers that are GV Green + Gold (fundamental strength).",
    )

with c2:
    render_semicircle_gauge(
        "📉 Market Direction",
        direction_pct,
        "Direction strength = 100 − BuyZone %. Higher means fewer pullbacks / stronger trend.",
    )

with c3:
    render_semicircle_gauge(
        "💰 Market Opportunity",
        opportunity_pct,
        "Opportunity breadth = avg(% Yield bullish, % GFV bullish). Higher means more undervaluation.",
    )

st.divider()

# ----------------------------------------------------------
# Optional: show the underlying components for transparency
# ----------------------------------------------------------
with st.expander("Show underlying breadth components"):
    ranked_scores = get_ranked_phase_scores(
        gv_bullish_pct,
        buyzone_pct,
        opportunity_pct,
    )

    st.write(
        {
            "GV Bullish % (Green+Gold)": round(gv_bullish_pct, 2),
            "BuyZone %": round(buyzone_pct, 2),
            "Direction % (100 - BuyZone)": round(direction_pct, 2),
            "Yield Bullish % (Green+Gold)": round(ys_bullish_pct, 2),
            "GFV Bullish % (Green+Gold)": round(gfv_bullish_pct, 2),
            "Opportunity % (avg Yield+GFV)": round(opportunity_pct, 2),
            "Phase": phase,
            "Confidence %": round(confidence * 100, 1),
            "Phase Scores": {
                k: round(v, 4) for k, v in ranked_scores
            },
        }
    )