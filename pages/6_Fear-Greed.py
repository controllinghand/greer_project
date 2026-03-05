# 6_Fear-Greed.py
# ----------------------------------------------------------
# Greer Fear & Greed Index (0–100)
# - 0   = Extreme Fear
# - 50  = Neutral
# - 100 = Extreme Greed
# Source: dashboard_summary_daily
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import time
import plotly.graph_objects as go
from db import get_engine

st.set_page_config(page_title="Greer Fear & Greed", layout="wide")

# ----------------------------------------------------------
# Load history
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_history(days: int = 800) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        f"""
        SELECT
          summary_date,
          total_companies,
          ys_gold, ys_green, ys_red,
          buyzone_pct
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT {days};
        """,
        engine
    )

hist = load_history(800)
if hist.empty:
    st.warning("dashboard_summary_daily is empty. Run build_dashboard_summary_daily.py daily to build history.")
    st.stop()

hist["summary_date"] = pd.to_datetime(hist["summary_date"])
hist = hist.sort_values("summary_date")

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def safe_pct(n: float, d: float) -> float:
    return (n / d * 100.0) if d else 0.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def render_fear_greed_dial(value: float):
    """
    Semi-circle dial gauge (0–100)
    Animates ONLY when sentiment bucket changes.
    """
    target = clamp(float(value), 0.0, 100.0)

    def make_fig(v: float) -> go.Figure:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=v,
                number={"font": {"size": 56}},
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
            height=330,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="white",
            font=dict(color="#222222"),
        )
        return fig

    ph = st.empty()

    # ----------------------------------------------------------
    # Very slick guard: animate only on bucket change
    # ----------------------------------------------------------
    bucket_now = fg_bucket(target)
    last_bucket = st.session_state.get("fg_last_bucket")

    animate = (last_bucket is None) or (bucket_now != last_bucket)

    st.session_state["fg_last_bucket"] = bucket_now
    st.session_state["fg_last_value"] = target

    if not animate:
        ph.plotly_chart(make_fig(target), use_container_width=True, config={"displayModeBar": False})
        return

    # ----------------------------------------------------------
    # Animate sweep from 50 → target
    # ----------------------------------------------------------
    start = 50.0
    steps = 18
    duration_ms = 900
    sleep_sec = (duration_ms / 1000.0) / steps

    for i in range(steps + 1):
        v = start + (target - start) * (i / steps)
        ph.plotly_chart(make_fig(v), use_container_width=True, config={"displayModeBar": False})
        time.sleep(sleep_sec)

# ----------------------------------------------------------
# Compute core series
# ----------------------------------------------------------
hist["yield_bullish_pct"] = hist.apply(
    lambda r: safe_pct((r["ys_green"] + r["ys_gold"]), r["total_companies"]),
    axis=1,
)

# 1) Stress (fear) = BuyZone %
hist["stress"] = hist["buyzone_pct"].astype(float)

# 2) Acceleration (fear) = 5-day rise in BuyZone
hist["bz_delta_5d"] = hist["buyzone_pct"].astype(float) - hist["buyzone_pct"].astype(float).shift(5)

# Normalize acceleration to 0–100:
# 0 points = no rise, 20 points in 5 days = max fear accel
hist["accel_norm"] = hist["bz_delta_5d"].fillna(0).apply(
    lambda x: clamp((max(x, 0.0) / 20.0) * 100.0, 0.0, 100.0)
)

# 3) Opportunity (fear proxy) = Yield bullish %
# More undervaluation usually appears during fear, so it raises FearScore
hist["opp_norm"] = hist["yield_bullish_pct"].apply(lambda x: clamp(float(x), 0.0, 100.0))

# ----------------------------------------------------------
# FearScore (0–100) then invert to GreedScore
# ----------------------------------------------------------
# Weights sum to 1.0
hist["fear_score"] = (
    0.45 * hist["stress"]
    + 0.30 * hist["accel_norm"]
    + 0.25 * hist["opp_norm"]
).apply(lambda x: clamp(float(x), 0.0, 100.0))

hist["greer_fear_greed"] = (100.0 - hist["fear_score"]).apply(lambda x: clamp(float(x), 0.0, 100.0))

# ----------------------------------------------------------
# Labeling (Bitcoin Fear & Greed style)
# ----------------------------------------------------------
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
        return "Greed"   # includes Neutral-ish zone around 50
    return "Extreme Greed"

# Latest
latest = hist.iloc[-1]
fg_now = float(latest["greer_fear_greed"])
label_now = fg_label(fg_now)

bucket_now = fg_bucket(fg_now)
prev_bucket = st.session_state.get("fg_prev_bucket_for_msg")

if prev_bucket is not None and prev_bucket != bucket_now:
    st.toast(f"Sentiment shift: {prev_bucket} → {bucket_now}", icon="⚡")

st.session_state["fg_prev_bucket_for_msg"] = bucket_now

# Yesterday / last week (if available)
yesterday = float(hist.iloc[-2]["greer_fear_greed"]) if len(hist) >= 2 else None
last_week = float(hist.iloc[-6]["greer_fear_greed"]) if len(hist) >= 6 else None

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("📊 Greer Fear & Greed Index (Latest)")
st.caption(f"Latest: {latest['summary_date'].date()}  •  Universe: {int(latest['total_companies'])} tickers")

# Dial (Bitcoin-style)
st.markdown(f"### Now: **{label_now}**")
render_fear_greed_dial(fg_now)

# Banner
if fg_now <= 24:
    st.error(f"🔴 Current value: **{fg_now:.0f}**  •  Sentiment: **{label_now}**")
elif fg_now <= 49:
    st.warning(f"🟠 Current value: **{fg_now:.0f}**  •  Sentiment: **{label_now}**")
elif fg_now <= 74:
    st.info(f"🟢 Current value: **{fg_now:.0f}**  •  Sentiment: **{label_now}**")
else:
    st.success(f"🟢 Current value: **{fg_now:.0f}**  •  Sentiment: **{label_now}**")

# Comparison strip like your example
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Current", f"{fg_now:.0f}", label_now)
with c2:
    if yesterday is None:
        st.metric("Yesterday", "—")
    else:
        st.metric("Yesterday", f"{yesterday:.0f}", fg_label(yesterday))
with c3:
    if last_week is None:
        st.metric("Last week", "—")
    else:
        st.metric("Last week", f"{last_week:.0f}", fg_label(last_week))

st.markdown(
    """
**Scale (0–100):**
- **0–24:** Extreme Fear  
- **25–49:** Fear  
- **50:** Neutral  
- **51–74:** Greed  
- **75–100:** Extreme Greed  
"""
)

st.divider()

# Chart
view_days = st.slider("Chart window (days)", min_value=30, max_value=365, value=180, step=15)
chart_df = hist.tail(int(view_days)).copy()

line = alt.Chart(chart_df).mark_line().encode(
    x=alt.X("summary_date:T", title=None),
    y=alt.Y("greer_fear_greed:Q", title="Greer Fear & Greed (0–100)", scale=alt.Scale(domain=[0, 100])),
    tooltip=[
        alt.Tooltip("summary_date:T", title="Date"),
        alt.Tooltip("greer_fear_greed:Q", title="Index", format=".1f"),
        alt.Tooltip("buyzone_pct:Q", title="BuyZone %", format=".1f"),
        alt.Tooltip("yield_bullish_pct:Q", title="Yield Bullish %", format=".1f"),
        alt.Tooltip("bz_delta_5d:Q", title="BuyZone Δ 5D", format="+.1f"),
    ],
)

rule_24 = alt.Chart(pd.DataFrame({"y": [24]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
rule_49 = alt.Chart(pd.DataFrame({"y": [49]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
rule_50 = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
rule_74 = alt.Chart(pd.DataFrame({"y": [74]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")

st.subheader("Greer Fear & Greed History")
st.altair_chart((line + rule_24 + rule_49 + rule_50 + rule_74).properties(height=260), use_container_width=True)

with st.expander("What drives the Greer Fear & Greed Index?"):
    st.markdown(
        """
This index is built from **Greer market internals**, not headlines:

- **BuyZone %** → stress / pullback density (fear increases as BuyZone rises)
- **BuyZone Δ 5D** → acceleration (fear rises fast when stress is rising quickly)
- **Yield Bullish %** → undervaluation breadth (fear environments often create undervaluation)

We compute a **FearScore (0–100)** and then invert it:

**Greer Fear & Greed = 100 − FearScore**

So:
- **low numbers** = fear dominates
- **high numbers** = greed dominates
"""
    )