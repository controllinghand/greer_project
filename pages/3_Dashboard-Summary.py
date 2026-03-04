# 3_Dashboard_Summary.py
# ----------------------------------------------------------
# Dashboard Summary
# - Breadth counts for GV / Yield / GFV / BuyZone
# - Source: dashboard_snapshot
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
from db import get_engine

st.set_page_config(page_title="Dashboard Summary", layout="wide")

# ----------------------------------------------------------
# Load snapshot
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
            ticker,
            greer_star_rating,
            greer_value_score,
            above_50_count,
            greer_yield_score,
            buyzone_flag,
            gfv_status,
            snapshot_date
        FROM dashboard_snapshot
        ORDER BY ticker;
        """,
        engine
    )

df = load_data()
if df.empty:
    st.warning("dashboard_snapshot is empty. Run build_dashboard_snapshot.py (or run_all.py).")
    st.stop()

st.title("📊 Dashboard Summary — Market Breadth")

last_updated = df["snapshot_date"].max() if "snapshot_date" in df.columns else None
if last_updated is not None:
    st.caption(f"Last updated: {last_updated}")

# ----------------------------------------------------------
# History helper
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def load_regime_history(days: int = 120) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        f"""
        SELECT
          summary_date,
          greer_market_index AS regime_score
        FROM dashboard_summary_daily
        ORDER BY summary_date DESC
        LIMIT {days};
        """,
        engine
    )

import altair as alt  # add at top of file with other imports

# ----------------------------------------------------------
# Greer Market Regime Banner + Trend + Chart + Streak
# ----------------------------------------------------------
hist = load_regime_history(180)

if hist.empty:
    st.info("Run build_dashboard_summary_daily.py daily to build regime history.")
else:
    hist["summary_date"] = pd.to_datetime(hist["summary_date"])
    hist = hist.sort_values("summary_date")  # ascending

    # Latest + delta vs yesterday
    latest = hist.iloc[-1]
    today_score = float(latest["regime_score"])

    delta = None
    if len(hist) >= 2:
        yesterday_score = float(hist.iloc[-2]["regime_score"])
        delta = today_score - yesterday_score

    # Regime label function
    def regime_label(score: float) -> str:
        if score >= 70:
            return "BULL"
        if score >= 45:
            return "TRANSITIONAL"
        return "BEAR"

    today_label = regime_label(today_score)

    # Streak: how many consecutive days in same regime label
    hist["regime_label"] = hist["regime_score"].apply(regime_label)
    streak = 0
    for lbl in reversed(hist["regime_label"].tolist()):
        if lbl == today_label:
            streak += 1
        else:
            break

    # Banner style based on regime
    delta_str = f"{delta:+.2f}" if delta is not None else None
    delta_arrow = "↑" if (delta is not None and delta > 0) else ("↓" if (delta is not None and delta < 0) else "→")
    delta_display = f"{delta_arrow} {delta_str}" if delta_str is not None else None

    if today_label == "BULL":
        st.success(f"🟢 Greer Market Regime: {today_label} ({today_score:.2f})  •  Day {streak}" + (f"  •  {delta_display}" if delta_display else ""))
    elif today_label == "TRANSITIONAL":
        st.warning(f"🟡 Greer Market Regime: {today_label} ({today_score:.2f})  •  Day {streak}" + (f"  •  {delta_display}" if delta_display else ""))
    else:
        st.error(f"🔴 Greer Market Regime: {today_label} ({today_score:.2f})  •  Day {streak}" + (f"  •  {delta_display}" if delta_display else ""))

    # 90-day chart with thresholds
    chart_df = hist.tail(90).copy()

    base = alt.Chart(chart_df).encode(
        x=alt.X("summary_date:T", title=None),
        y=alt.Y("regime_score:Q", title="Regime Score", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("summary_date:T", title="Date"),
            alt.Tooltip("regime_score:Q", title="Score", format=".2f"),
            alt.Tooltip("regime_label:N", title="Regime"),
        ],
    )

    line = base.mark_line().encode()

    rule_45 = alt.Chart(pd.DataFrame({"y": [45]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
    rule_70 = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")

    st.altair_chart((line + rule_45 + rule_70).properties(height=220), use_container_width=True)

    st.divider()

# ----------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    min_stars = st.selectbox("Minimum star rating", [0, 1, 2, 3], index=0)
    only_buyzone = st.checkbox("Only show BuyZone = True", value=False)

filtered = df.copy()

if min_stars:
    stars_series = pd.to_numeric(filtered["greer_star_rating"], errors="coerce").fillna(0)
    filtered = filtered[stars_series >= min_stars]

if only_buyzone:
    filtered = filtered[filtered["buyzone_flag"] == True]

total = len(filtered)
st.markdown(f"Showing **{total}** companies")

# ----------------------------------------------------------
# Bucket logic
# ----------------------------------------------------------
def gv_bucket(row) -> str:
    above50_raw = row.get("above_50_count")
    gv_raw = row.get("greer_value_score")

    above50 = 0 if pd.isna(above50_raw) else int(above50_raw)
    gv = 0.0 if pd.isna(gv_raw) else float(gv_raw)

    if above50 == 6:
        return "gold"
    if gv >= 50:
        return "green"
    return "red"

def yield_bucket(row) -> str:
    ys_raw = row.get("greer_yield_score")
    ys = 0 if pd.isna(ys_raw) else int(ys_raw)

    if ys >= 4:
        return "gold"
    if ys >= 2:
        return "green"
    return "red"

def gfv_bucket(row) -> str:
    """
    GFV is already computed in dashboard_snapshot:
    - gold/green/red based on DCF+Graham logic
    - otherwise gray
    """
    s = row.get("gfv_status")
    s = str(s).strip().lower() if pd.notnull(s) else "gray"

    if s in {"gold", "green", "red", "gray"}:
        return s

    # In case older rows used different labels:
    if s in {"undervalued_strong", "strong_buy", "both_above"}:
        return "gold"
    if s in {"undervalued", "buy", "one_above"}:
        return "green"
    if s in {"overvalued", "sell", "both_below"}:
        return "red"

    return "gray"

def summarize(series: pd.Series, include_gray: bool = False) -> dict:
    out = {
        "gold": int((series == "gold").sum()),
        "green": int((series == "green").sum()),
        "red": int((series == "red").sum()),
    }
    if include_gray:
        out["gray"] = int((series == "gray").sum())
    return out

# ----------------------------------------------------------
# Compute buckets + counts
# ----------------------------------------------------------
if total == 0:
    st.info("No rows match your filters.")
    st.stop()

# -----------------------------
# Vectorized buckets (fast + safe)
# -----------------------------
above50 = pd.to_numeric(filtered["above_50_count"], errors="coerce").fillna(0)
gv_score = pd.to_numeric(filtered["greer_value_score"], errors="coerce").fillna(0)

filtered["gv_bucket"] = "red"
filtered.loc[gv_score >= 50, "gv_bucket"] = "green"
filtered.loc[above50 == 6, "gv_bucket"] = "gold"

ys_score = pd.to_numeric(filtered["greer_yield_score"], errors="coerce").fillna(0).astype(int)
filtered["ys_bucket"] = "red"
filtered.loc[ys_score >= 2, "ys_bucket"] = "green"
filtered.loc[ys_score >= 4, "ys_bucket"] = "gold"

# Normalize GFV status
gfv_norm = (
    filtered["gfv_status"]
    .astype("string")
    .fillna("gray")
    .str.strip()
    .str.lower()
)

# Map any legacy labels into the 4 canonical colors
legacy_map = {
    "undervalued_strong": "gold",
    "strong_buy": "gold",
    "both_above": "gold",
    "undervalued": "green",
    "buy": "green",
    "one_above": "green",
    "overvalued": "red",
    "sell": "red",
    "both_below": "red",
}
gfv_norm = gfv_norm.replace(legacy_map)

# Anything not in {gold, green, red, gray} becomes gray
valid = {"gold", "green", "red", "gray"}
filtered["gfv_bucket"] = gfv_norm.where(gfv_norm.isin(valid), "gray")

gv = summarize(filtered["gv_bucket"])
ys = summarize(filtered["ys_bucket"])
gfv = summarize(filtered["gfv_bucket"], include_gray=True)

buyzone_count = int(filtered["buyzone_flag"].fillna(False).sum())
buyzone_pct = (buyzone_count / total * 100.0) if total else 0.0

# ----------------------------------------------------------
# Summary strip (with %)
# ----------------------------------------------------------

def pct(x):
    return f"{(x / total * 100):.1f}%" if total else "0.0%"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Greer Value")
    st.metric("Gold 🥇", f"{gv['gold']} ({pct(gv['gold'])})")
    st.metric("Green 🟢", f"{gv['green']} ({pct(gv['green'])})")
    st.metric("Red 🔴", f"{gv['red']} ({pct(gv['red'])})")

with c2:
    st.subheader("Greer Yield")
    st.metric("Gold 🥇", f"{ys['gold']} ({pct(ys['gold'])})")
    st.metric("Green 🟢", f"{ys['green']} ({pct(ys['green'])})")
    st.metric("Red 🔴", f"{ys['red']} ({pct(ys['red'])})")

with c3:
    st.subheader("Greer Fair Value")
    st.metric("Gold 🥇", f"{gfv['gold']} ({pct(gfv['gold'])})")
    st.metric("Green 🟢", f"{gfv['green']} ({pct(gfv['green'])})")
    st.metric("Red 🔴", f"{gfv['red']} ({pct(gfv['red'])})")
    st.metric("Gray ⚪", f"{gfv['gray']} ({pct(gfv['gray'])})")

with c4:
    st.subheader("BuyZone")
    st.metric("In BuyZone", f"{buyzone_count} ({pct(buyzone_count)})")
    st.metric("Total", total)

st.divider()

# ----------------------------------------------------------
# Market tone (simple)
# ----------------------------------------------------------
gv_green_pct = gv["green"] / total
bz_pct = buyzone_count / total

if gv_green_pct >= 0.60 and bz_pct >= 0.25:
    st.success("Market Tone: Broad Strength + Plenty of BuyZones 🟢")
elif gv_green_pct <= 0.40 and bz_pct >= 0.25:
    st.warning("Market Tone: Weak breadth but lots of BuyZones (possible selloff / opportunity) 🟡")
elif gv_green_pct <= 0.40 and bz_pct < 0.25:
    st.error("Market Tone: Broad Weakness 🔴")
else:
    st.info("Market Tone: Mixed / Transitional ⚪")