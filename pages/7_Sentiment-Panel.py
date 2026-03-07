# 7_Sentiment-Panel.py
# ----------------------------------------------------------
# Greer Sentiment Panel
# - Single "terminal" panel combining key market internals
# - Source: dashboard_summary_daily
# ----------------------------------------------------------

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_engine
from market_cycle_utils import (
    classify_phase_with_confidence,
    phase_confidence_note,
)

st.set_page_config(page_title="Greer Sentiment Panel", layout="wide")

# ----------------------------------------------------------
# Load latest summary + short history
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_summary(days: int = 120) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        f"""
        SELECT
          summary_date,
          total_companies,
          gv_gold, gv_green, gv_red,
          ys_gold, ys_green, ys_red,
          gfv_gold, gfv_green, gfv_red, gfv_gray,
          buyzone_count,
          buyzone_pct,
          gv_green_pct,
          ys_green_pct,
          gfv_green_pct,
          greer_market_index
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT {days};
        """,
        engine
    )

df = load_summary(120)
if df.empty:
    st.warning("dashboard_summary_daily is empty. Run build_dashboard_summary_daily.py daily to build history.")
    st.stop()

df["summary_date"] = pd.to_datetime(df["summary_date"])
df = df.sort_values("summary_date")
latest = df.iloc[-1]

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def fg_label(x: float) -> str:
    if x <= 24:
        return "Extreme Fear 😨"
    if x <= 49:
        return "Fear 😟"
    if 49 < x < 51:
        return "Neutral 😐"
    if x <= 74:
        return "Greed 😏"
    return "Extreme Greed 🤑"

def fg_bucket(x: float) -> str:
    if x <= 24:
        return "Extreme Fear"
    if x <= 49:
        return "Fear"
    if x < 75:
        return "Greed"
    return "Extreme Greed"

def sentiment_regime_label(phase: str, confidence: float, buyzone_pct: float) -> tuple[str, str]:
    p = str(phase).strip().upper()

    if p == "EUPHORIA":
        if confidence < 0.35:
            return "🟠 Late Cycle Strength", "Trend Strong • Rotation Risk Rising"
        return "🟠 Late Cycle Strength", "Trend Strong • Opportunity Limited"

    if p == "EXPANSION":
        return "🟢 Trend Strong", "Healthy Breadth • Trend Supportive"

    if p == "RECOVERY":
        if buyzone_pct >= 30:
            return "🔵 Pullback Opportunity", "Value Improving • Stress Elevated"
        return "🔵 Recovery", "Opportunity Building • Trend Repairing"

    if p == "CONTRACTION":
        return "🔴 Defensive Regime", "Stress Rising • Selective Opportunities"

    return "🟡 Mixed Regime", "Signals Mixed"


def temperature_label(x: float) -> str:
    if x < 25:
        return "❄️ Deep Bear"
    if x < 45:
        return "🧊 Cool Market"
    if x < 60:
        return "🌤 Balanced"
    if x < 75:
        return "🔥 Heating Up"
    return "🌋 Overheated"

# ----------------------------------------------------------
# Color intelligence (emoji deltas)
# ----------------------------------------------------------
def metric_signal(value: float, good_above: float, bad_below: float) -> str:
    """
    Streamlit st.metric delta is text; we use emoji as a compact signal.
    """
    if value >= good_above:
        return "🟢 Strong"
    if value <= bad_below:
        return "🔴 Weak"
    return "🟡 Mixed"

def metric_signal_inverse(value: float, good_below: float, bad_above: float) -> str:
    """
    Inverse scoring: lower is better (e.g., BuyZone% is "stress").
    """
    if value <= good_below:
        return "🟢 Calm"
    if value >= bad_above:
        return "🔴 Rising"
    return "🟡 Mixed"



# ----------------------------------------------------------
# Compute panel metrics from latest snapshot
# ----------------------------------------------------------
total = safe_int(latest.get("total_companies"), 0)

gv_gold = safe_float(latest.get("gv_gold"), 0.0)
gv_green_pct = safe_float(latest.get("gv_green_pct"), 0.0)  # already % green

gv_gold_pct = (gv_gold / total * 100.0) if total else 0.0
health = clamp(gv_green_pct + gv_gold_pct, 0.0, 100.0)

buyzone_pct = safe_float(latest.get("buyzone_pct"), 0.0)
direction = clamp(100.0 - buyzone_pct, 0.0, 100.0)

ys_green = safe_float(latest.get("ys_green"), 0.0)
ys_gold = safe_float(latest.get("ys_gold"), 0.0)
yield_bullish_pct = ((ys_green + ys_gold) / total * 100.0) if total else 0.0
opportunity = clamp(yield_bullish_pct, 0.0, 100.0)

phase, confidence = classify_phase_with_confidence(
    health,
    buyzone_pct,
    opportunity,
)

bottom_score = clamp((buyzone_pct + yield_bullish_pct) / 2.0, 0.0, 100.0)

# Fear & Greed Index
stress = buyzone_pct
accel = 0.0
if len(df) >= 6:
    accel = safe_float(df.iloc[-1].get("buyzone_pct"), 0.0) - safe_float(df.iloc[-6].get("buyzone_pct"), 0.0)
accel_norm = clamp((max(accel, 0.0) / 20.0) * 100.0, 0.0, 100.0)

fear_score = clamp(0.45 * stress + 0.30 * accel_norm + 0.25 * opportunity, 0.0, 100.0)
fg_now = clamp(100.0 - fear_score, 0.0, 100.0)
fg_text = fg_label(fg_now)

# Extra breadth snapshot
gfv_green = safe_float(latest.get("gfv_green"), 0.0)
gfv_gold = safe_float(latest.get("gfv_gold"), 0.0)
gfv_bullish_pct = ((gfv_green + gfv_gold) / total * 100.0) if total else 0.0

buyzone_count = safe_int(latest.get("buyzone_count"), 0)

# ----------------------------------------------------------
# Dial (with “animate on bucket change” guard)
# ----------------------------------------------------------
def render_fear_greed_dial(value: float):
    target = clamp(float(value), 0.0, 100.0)

    def make_fig(v: float) -> go.Figure:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=v,
                number={"font": {"size": 52}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999999"},
                    "bar": {"color": "#888888", "thickness": 0.28},
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
        return fig

    ph = st.empty()

    bucket_now = fg_bucket(target)
    last_bucket = st.session_state.get("fg_panel_last_bucket")
    animate = (last_bucket is None) or (bucket_now != last_bucket)

    st.session_state["fg_panel_last_bucket"] = bucket_now

    if not animate:
        ph.plotly_chart(make_fig(target), use_container_width=True, config={"displayModeBar": False})
        return

    start = 50.0
    steps = 14
    duration_ms = 650
    sleep_sec = (duration_ms / 1000.0) / steps

    for i in range(steps + 1):
        v = start + (target - start) * (i / steps)
        ph.plotly_chart(make_fig(v), use_container_width=True, config={"displayModeBar": False})
        time.sleep(sleep_sec)

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.title("🧠 Greer Sentiment Panel")
regime_title, regime_sub = sentiment_regime_label(phase, confidence, buyzone_pct)

st.info(
    f"Market Regime: {regime_title} ({phase.title()} Phase) "
    f"• Confidence {round(confidence * 100)}% "
    f"• {regime_sub}"
)
# ----------------------------------------------------------
# Market Temperature Score
# ----------------------------------------------------------

opportunity_heat = 100.0 - opportunity

temperature = (
    0.30 * direction +
    0.25 * health +
    0.20 * fg_now +
    0.15 * opportunity_heat +
    0.10 * buyzone_pct
)

temperature = clamp(temperature, 0.0, 100.0)
st.metric(
    "Market Temperature",
    f"{temperature:.1f}",
    temperature_label(temperature)
)

st.caption(f"Latest: {latest['summary_date'].date()}  •  Universe: {total} tickers")
st.divider()

# ----------------------------------------------------------
# Panel Row (4 cards)
# ----------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

with c1:
    st.subheader("Fear & Greed")
    st.markdown(f"**Now:** {fg_text}")
    render_fear_greed_dial(fg_now)

with c2:
    st.subheader("Market Cycle")
    st.markdown(f"**Phase:** `{phase}`")
    st.caption(
        f"Confidence: {round(confidence * 100)}% • {phase_confidence_note(confidence)}"
    )

    st.metric(
        "Health (GV bullish)",
        f"{health:.1f}%",
        metric_signal(health, good_above=60.0, bad_below=40.0),
    )
    st.metric(
        "Direction (100 - BZ%)",
        f"{direction:.1f}%",
        metric_signal(direction, good_above=60.0, bad_below=40.0),
    )
    st.metric(
        "Opportunity (Yield bullish)",
        f"{opportunity:.1f}%",
        metric_signal(opportunity, good_above=50.0, bad_below=30.0),
    )

with c3:
    st.subheader("Bottom Detector")
    st.metric(
        "BuyZone % (stress)",
        f"{buyzone_pct:.1f}%",
        metric_signal_inverse(buyzone_pct, good_below=15.0, bad_above=30.0),
    )
    st.metric(
        "Yield Bullish %",
        f"{yield_bullish_pct:.1f}%",
        metric_signal(yield_bullish_pct, good_above=50.0, bad_below=30.0),
    )
    # Bottom score: higher means more "bottom-like" conditions
    st.metric(
        "Bottom Score",
        f"{bottom_score:.2f}",
        metric_signal(bottom_score, good_above=45.0, bad_below=25.0),
    )

with c4:
    st.subheader("Breadth Snapshot")
    st.metric(
        "GV Bullish (G+🥇)",
        f"{health:.1f}%",
        metric_signal(health, good_above=60.0, bad_below=40.0),
    )
    st.metric(
        "GFV Bullish (G+🥇)",
        f"{gfv_bullish_pct:.1f}%",
        metric_signal(gfv_bullish_pct, good_above=35.0, bad_below=20.0),
    )
    st.metric(
        "BuyZone Count",
        f"{buyzone_count}",
        metric_signal_inverse(buyzone_pct, good_below=15.0, bad_above=30.0),
    )

st.divider()

# ----------------------------------------------------------
# Mini trend strip (last 30 days)
# ----------------------------------------------------------
st.subheader("30-Day Trend (quick)")
tail = df.tail(30).copy()
tail["yield_bullish_pct"] = ((tail["ys_green"] + tail["ys_gold"]) / tail["total_companies"]) * 100.0
tail["health"] = tail["gv_green_pct"] + (tail["gv_gold"] / tail["total_companies"]) * 100.0
tail["direction"] = 100.0 - tail["buyzone_pct"]
tail["bottom_score"] = (tail["buyzone_pct"] + tail["yield_bullish_pct"]) / 2.0

st.line_chart(
    tail.set_index("summary_date")[["health", "direction", "yield_bullish_pct", "bottom_score"]],
    height=260,
)