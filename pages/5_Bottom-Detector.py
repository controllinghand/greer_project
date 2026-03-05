# 5_Bottom-Detector.py
# ----------------------------------------------------------
# Greer Bottom Detector
# - Uses BuyZone % + Yield Bullish % to detect "bottom conditions"
# - Adds a semi-circle Bottom Score gauge (0–100)
# - Source: dashboard_summary_daily
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from db import get_engine

st.set_page_config(page_title="Bottom Detector", layout="wide")

# ----------------------------------------------------------
# Load history
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_history(days: int = 365) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        f"""
        SELECT
          summary_date,
          total_companies,

          ys_gold, ys_green, ys_red,
          buyzone_count,
          buyzone_pct
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT {days};
        """,
        engine
    )

hist = load_history(365)
if hist.empty:
    st.warning("dashboard_summary_daily is empty. Run build_dashboard_summary_daily.py daily to build history.")
    st.stop()

hist["summary_date"] = pd.to_datetime(hist["summary_date"])
hist = hist.sort_values("summary_date")

# ----------------------------------------------------------
# Compute Yield Bullish % and Bottom Score
# ----------------------------------------------------------
def safe_pct(n: float, d: float) -> float:
    return (n / d * 100.0) if d else 0.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

hist["yield_bullish_pct"] = hist.apply(
    lambda r: safe_pct((r["ys_green"] + r["ys_gold"]), r["total_companies"]),
    axis=1,
)

# Bottom Score: normalize both to 0–100 and average
# - BuyZone already 0–100
# - Yield bullish already 0–100
hist["bottom_score"] = (hist["buyzone_pct"] + hist["yield_bullish_pct"]) / 2.0

# ----------------------------------------------------------
# Thresholds (tweak later after data builds)
# ----------------------------------------------------------
THRESH_PULLBACK = 30.0        # buyzone_pct >= 30 = broad pullback
THRESH_WASHOUT = 40.0         # buyzone_pct >= 40 = capitulation risk
THRESH_VALUE = 45.0           # yield bullish >= 45 = market broadly undervalued
THRESH_DEEP_VALUE = 55.0      # yield bullish >= 55 = very undervalued

def classify_signal(bz: float, yb: float) -> str:
    # Phase 1: Crash / panic selling (stress high, undervaluation not broad yet)
    if bz >= 45 and yb < 30:
        return "CRASH CONDITIONS"
    # Phase 2: Panic but value emerging
    if bz >= THRESH_WASHOUT and yb >= THRESH_VALUE and yb < THRESH_DEEP_VALUE:
        return "WASHOUT (WATCHLIST)"
    # Phase 3: True bottom forming (broad pullback + deep undervaluation)
    if bz >= THRESH_PULLBACK and yb >= THRESH_DEEP_VALUE and bz < THRESH_WASHOUT:
        return "BOTTOMING CONDITIONS"
    # Phase 4: Maximum opportunity (washout + deep undervaluation)
    if bz >= THRESH_WASHOUT and yb >= THRESH_DEEP_VALUE:
        return "CAPITULATION BUY ZONE"
    return "NORMAL / NO SIGNAL"

def signal_badge_style(label: str) -> tuple[str, str]:
    """
    Returns (emoji, Streamlit color style type) for banner rendering.
    """
    if label == "CAPITULATION BUY ZONE":
        return "🟢", "success"
    if label == "BOTTOMING CONDITIONS":
        return "🟡", "warning"
    if label == "WASHOUT (WATCHLIST)":
        return "🟠", "warning"
    if label == "CRASH CONDITIONS":
        return "🔴", "error"
    return "⚪", "info"

# Latest
latest = hist.iloc[-1]
bz_now = float(latest["buyzone_pct"])
yb_now = float(latest["yield_bullish_pct"])
score_now = float(latest["bottom_score"])
signal_now = classify_signal(bz_now, yb_now)

# Streak: consecutive days in the SAME state category (signal vs no-signal)
def is_signal(label: str) -> bool:
    return label != "NORMAL / NO SIGNAL"

labels = [classify_signal(float(r.buyzone_pct), float(r.yield_bullish_pct)) for r in hist.itertuples()]
streak = 0
for lbl in reversed(labels):
    if is_signal(lbl) == is_signal(signal_now):
        streak += 1
    else:
        break

# Delta vs yesterday
delta_score = None
if len(hist) >= 2:
    delta_score = score_now - float(hist.iloc[-2]["bottom_score"])

# ----------------------------------------------------------
# Gauge (semi-circle) for Bottom Score
# ----------------------------------------------------------
def bottom_score_label(x: float) -> str:
    """
    0–100 Bottom Score interpretation:
    - lower = normal
    - higher = more "bottom-like" conditions (stress + undervaluation breadth)
    """
    if x < 30:
        return "Normal"
    if x < 45:
        return "Early Stress"
    if x < 60:
        return "Fear"
    if x < 75:
        return "Panic"
    return "Capitulation"

def render_bottom_score_gauge(score: float):
    v = clamp(float(score), 0.0, 100.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"font": {"size": 46}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999999"},
                "bar": {"color": "#888888", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#5CB85C"},    # green
                    {"range": [30, 45], "color": "#FFD966"},  # yellow
                    {"range": [45, 60], "color": "#F0AD4E"},  # orange
                    {"range": [60, 75], "color": "#D9534F"},  # red
                    {"range": [75, 100], "color": "#8E44AD"}, # purple
                ],
                "threshold": {
                    "line": {"color": "#333333", "width": 6},
                    "thickness": 0.85,
                    "value": v,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(color="#222222"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("🧯 Greer Bottom Detector — BuyZone + Yield")
st.caption(f"Latest: {latest['summary_date'].date()}  •  Universe: {int(latest['total_companies'])} tickers")

# ----------------------------------------------------------
# Main Gauge (Top Center)
# ----------------------------------------------------------
st.subheader("Bottom Score Gauge")
st.caption(f"Interpretation: **{bottom_score_label(score_now)}**")

center = st.columns([1,2,1])[1]

with center:
    render_bottom_score_gauge(score_now)

st.divider()

# ----------------------------------------------------------
# Signal banner
# ----------------------------------------------------------
delta_str = f"{delta_score:+.2f}" if delta_score is not None else None
arrow = "↑" if (delta_score is not None and delta_score > 0) else ("↓" if (delta_score is not None and delta_score < 0) else "→")
delta_display = f"{arrow} {delta_str}" if delta_str is not None else None

emoji, style = signal_badge_style(signal_now)
banner_text = f"{emoji} **{signal_now}**  •  Score {score_now:.2f}" + (f"  •  {delta_display}" if delta_display else "")

if style == "success":
    st.success(banner_text)
elif style == "warning":
    st.warning(banner_text)
elif style == "error":
    st.error(banner_text)
else:
    st.info(banner_text)

st.caption(f"Current streak: **{streak} day(s)** in this state.")

st.divider()

# ----------------------------------------------------------
# KPI Strip
# ----------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("BuyZone % (higher = more stress)", f"{bz_now:.1f}%")

with c2:
    st.metric("Yield Bullish % (higher = more undervaluation)", f"{yb_now:.1f}%")

with c3:
    st.metric("Bottom Score (avg)", f"{score_now:.2f}")

st.divider()

# ----------------------------------------------------------
# Charts
# ----------------------------------------------------------
chart_df = hist.tail(180).copy()

base = alt.Chart(chart_df).encode(
    x=alt.X("summary_date:T", title=None),
    tooltip=[
        alt.Tooltip("summary_date:T", title="Date"),
        alt.Tooltip("buyzone_pct:Q", title="BuyZone %", format=".1f"),
        alt.Tooltip("yield_bullish_pct:Q", title="Yield Bullish %", format=".1f"),
        alt.Tooltip("bottom_score:Q", title="Bottom Score", format=".2f"),
    ],
)

bz_line = base.mark_line().encode(
    y=alt.Y("buyzone_pct:Q", title="Percent", scale=alt.Scale(domain=[0, 100]))
)
yb_line = base.mark_line(strokeDash=[6, 4]).encode(
    y="yield_bullish_pct:Q"
)

rule_bz1 = alt.Chart(pd.DataFrame({"y": [THRESH_PULLBACK]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
rule_bz2 = alt.Chart(pd.DataFrame({"y": [THRESH_WASHOUT]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
rule_y1 = alt.Chart(pd.DataFrame({"y": [THRESH_VALUE]})).mark_rule(strokeDash=[2, 6]).encode(y="y:Q")
rule_y2 = alt.Chart(pd.DataFrame({"y": [THRESH_DEEP_VALUE]})).mark_rule(strokeDash=[2, 6]).encode(y="y:Q")

st.subheader("BuyZone % (solid) vs Yield Bullish % (dashed)")
st.altair_chart((bz_line + yb_line + rule_bz1 + rule_bz2 + rule_y1 + rule_y2).properties(height=260), use_container_width=True)

score_line = alt.Chart(chart_df).mark_line().encode(
    x=alt.X("summary_date:T", title=None),
    y=alt.Y("bottom_score:Q", title="Bottom Score", scale=alt.Scale(domain=[0, 100])),
    tooltip=[
        alt.Tooltip("summary_date:T", title="Date"),
        alt.Tooltip("bottom_score:Q", title="Bottom Score", format=".2f"),
    ],
)

st.subheader("Bottom Score (average of BuyZone% and Yield Bullish%)")
st.altair_chart(score_line.properties(height=220), use_container_width=True)

# ----------------------------------------------------------
# Explain the model
# ----------------------------------------------------------
with st.expander("How to read this"):
    st.markdown(
        """
**BuyZone %** = market stress / pullback density.  
**Yield Bullish %** = undervaluation breadth (Yield Green + Gold).  

**Bottom Detector logic:**
- When **BuyZone is high** and **Yield Bullish is high**, the market is both stressed and undervalued.
- Those environments often occur near **major lows** or **high-quality entry zones**.

This is not a timing tool to pick the exact bottom day.
It’s a **regime detector** that tells us when the market is offering unusually good long-term entries.
"""
    )