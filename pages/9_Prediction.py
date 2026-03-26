# pages/9_Prediction.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Prediction", layout="wide")

PREDICTION_PAGE_DESCRIPTION = """
The Prediction model estimates the probability that a stock will be higher in roughly 60 trading days.

It combines company phase, phase transitions, market opportunity regime (GOI),
BuyZone state, confidence, and fundamentals as a light quality overlay.

Key findings from historical testing:
- The score is not linear. Higher is not always better.
- The strongest high-conviction bucket was around score 90.
- The best scalable bucket was around score 110.
- The best recurring regime was Contraction + Elevated Opportunity.
- Extreme Greed was a dangerous environment for contraction names.

This is a probability tool, not a certainty tool.
"""

# ----------------------------------------------------------
# DB connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    return get_engine()


# ----------------------------------------------------------
# Load live prediction inputs
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_prediction_inputs() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        WITH latest_market AS (
            SELECT
                date,
                buyzone_pct,
                CASE
                    WHEN buyzone_pct >= 66 THEN 'EXTREME_OPPORTUNITY'
                    WHEN buyzone_pct >= 46 THEN 'ELEVATED_OPPORTUNITY'
                    WHEN buyzone_pct >= 14 THEN 'NORMAL'
                    WHEN buyzone_pct >= 10 THEN 'LOW_OPPORTUNITY'
                    ELSE 'EXTREME_GREED'
                END AS goi_zone
            FROM buyzone_breadth
            ORDER BY date DESC
            LIMIT 1
        ),
        company_phase_history AS (
            SELECT
                g.ticker,
                g.date,
                g.phase,
                g.confidence,
                LAG(g.phase) OVER (PARTITION BY g.ticker ORDER BY g.date) AS prior_phase,
                ROW_NUMBER() OVER (PARTITION BY g.ticker ORDER BY g.date DESC) AS rn
            FROM greer_company_index_daily g
        ),
        latest_company_phase AS (
            SELECT
                ticker,
                date AS snapshot_date,
                phase,
                prior_phase,
                confidence
            FROM company_phase_history
            WHERE rn = 1
        )
        SELECT
            ds.ticker,
            ds.name,
            ds.sector,
            ds.industry,
            ds.current_price,
            ds.greer_star_rating,
            ds.greer_value_score,
            ds.greer_yield_score,
            ds.buyzone_flag,
            ds.gfv_price,
            ds.gfv_status,
            lcp.snapshot_date,
            lcp.phase,
            lcp.prior_phase,
            lcp.confidence,
            lm.buyzone_pct AS market_buyzone_pct,
            lm.goi_zone
        FROM dashboard_snapshot ds
        JOIN latest_company_phase lcp
          ON lcp.ticker = ds.ticker
        CROSS JOIN latest_market lm
        ORDER BY ds.ticker
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Prediction scoring helpers
# ----------------------------------------------------------

def round_score_bucket(score: float) -> int:
    return int(round(score / 10.0) * 10)


def get_calibration_bucket(score_bucket: int) -> int | None:
    """
    Maps raw score buckets to the historically calibrated buckets
    that actually have strong backtest meaning.
    """
    if score_bucket >= 125:
        return 130
    if score_bucket >= 105:
        return 110
    if score_bucket >= 85:
        return 90
    return None


def get_prediction_stats_for_bucket(calibration_bucket: int | None) -> tuple[float | None, float | None, str]:
    calibration = {
        90:  (0.734, 0.113, "High Conviction"),
        110: (0.681, 0.083, "Strong"),
        130: (0.646, 0.068, "Constructive"),
    }

    if calibration_bucket in calibration:
        return calibration[calibration_bucket]

    return None, None, "Watchlist"


def build_setup_label(phase: str | None, goi_zone: str | None) -> str:
    phase_label = (phase or "Unknown").title()
    goi_map = {
        "EXTREME_OPPORTUNITY": "Extreme Opportunity",
        "ELEVATED_OPPORTUNITY": "Elevated Opportunity",
        "NORMAL": "Normal",
        "LOW_OPPORTUNITY": "Low Opportunity",
        "EXTREME_GREED": "Extreme Greed",
    }
    goi_label = goi_map.get(goi_zone, goi_zone or "Unknown")
    return f"{phase_label} + {goi_label}"


def calculate_prediction_score(row: pd.Series) -> pd.Series:
    phase = row.get("phase")
    prior_phase = row.get("prior_phase")
    buyzone_flag = row.get("buyzone_flag")
    confidence = float(row.get("confidence") or 0.0)
    gv = row.get("greer_value_score")
    gy = row.get("greer_yield_score")
    goi_zone = row.get("goi_zone")

    phase_score = 30 if phase == "CONTRACTION" else 25 if phase == "RECOVERY" else 20 if phase == "EUPHORIA" else 15 if phase == "EXPANSION" else 0

    if prior_phase == "CONTRACTION" and phase == "EXPANSION":
        transition_score = 25
    elif prior_phase == "CONTRACTION" and phase == "RECOVERY":
        transition_score = 20
    elif prior_phase == "EXPANSION" and phase == "CONTRACTION":
        transition_score = 20
    else:
        transition_score = 0

    buyzone_score = 15 if buyzone_flag is True else 0
    confidence_score = confidence * 20

    fundamentals_score = 0
    if pd.notna(gv) and gv >= 60:
        fundamentals_score += 5
    if pd.notna(gy) and gy >= 3:
        fundamentals_score += 5

    goi_score = 0
    if goi_zone == "ELEVATED_OPPORTUNITY":
        goi_score = 30
    elif goi_zone == "EXTREME_OPPORTUNITY":
        goi_score = 20
    elif goi_zone == "NORMAL":
        goi_score = 10
    elif goi_zone == "LOW_OPPORTUNITY":
        goi_score = 5
    elif goi_zone == "EXTREME_GREED":
        goi_score = -15

    regime_alignment_score = 0
    if phase == "CONTRACTION" and goi_zone == "ELEVATED_OPPORTUNITY":
        regime_alignment_score = 30
    elif phase == "RECOVERY" and goi_zone == "ELEVATED_OPPORTUNITY":
        regime_alignment_score = 25
    elif phase == "CONTRACTION" and goi_zone == "EXTREME_OPPORTUNITY":
        regime_alignment_score = 20
    elif phase == "EUPHORIA" and goi_zone == "EXTREME_GREED":
        regime_alignment_score = -20

    overheat_penalty = 0
    if phase == "EUPHORIA" and goi_zone == "EXTREME_GREED":
        overheat_penalty = -30
    elif phase == "EXPANSION" and goi_zone == "EXTREME_GREED":
        overheat_penalty = -20
    elif phase == "EUPHORIA" and buyzone_flag is False:
        overheat_penalty = -10

    prediction_score = (
        phase_score
        + transition_score
        + buyzone_score
        + confidence_score
        + fundamentals_score
        + goi_score
        + regime_alignment_score
        + overheat_penalty
    )

    score_bucket = round_score_bucket(prediction_score)
    calibration_bucket = get_calibration_bucket(score_bucket)
    expected_win_rate_60d, expected_return_60d, signal_tier = get_prediction_stats_for_bucket(calibration_bucket)
    setup_label = build_setup_label(phase, goi_zone)

    return pd.Series({
        "prediction_score": prediction_score,
        "score_bucket": score_bucket,
        "calibration_bucket": calibration_bucket,
        "expected_win_rate_60d": expected_win_rate_60d,
        "expected_return_60d": expected_return_60d,
        "signal_tier": signal_tier,
        "setup_label": setup_label,
    })


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("🔮 Prediction")
    st.caption("Probability-based 60-day outlook for current company setups.")
    st.markdown(PREDICTION_PAGE_DESCRIPTION)

    df = load_prediction_inputs()

    if df.empty:
        st.warning("No prediction data available.")
        return

    # Current market regime banner
    current_goi_zone = df["goi_zone"].dropna().iloc[0] if not df["goi_zone"].dropna().empty else "UNKNOWN"
    current_goi_label = current_goi_zone.replace("_", " ").title()

    st.info(f"""
Current Market Regime: **{current_goi_label}**

Interpretation:
- Elevated / Extreme Opportunity → Strong environment for entries
- Normal → Selectivity matters
- Extreme Greed → Risk of overextension
""")

    # Optional filters: exclude known weak/noisy setups
    df = df[df["buyzone_flag"].notna()].copy()
    df = df[~((df["phase"] == "CONTRACTION") & (df["goi_zone"] == "EXTREME_GREED"))].copy()

    score_df = df.apply(calculate_prediction_score, axis=1)
    df = pd.concat([df, score_df], axis=1)

    # Top summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best Historical Bucket", "90")
        st.caption("~73.4% 60d win rate")
    with c2:
        st.metric("Best Scalable Bucket", "110")
        st.caption("~68.1% 60d win rate")
    with c3:
        st.metric("Best Core Regime", "Contraction + Elevated")
        st.caption("Strongest recurring setup")

    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Minimum Prediction Score", min_value=0, max_value=150, value=80, step=10)
    with col2:
        selected_phase = st.multiselect(
            "Phase",
            sorted(df["phase"].dropna().unique().tolist()),
            default=sorted(df["phase"].dropna().unique().tolist())
        )
    with col3:
        selected_goi = st.multiselect(
            "GOI Zone",
            sorted(df["goi_zone"].dropna().unique().tolist()),
            default=sorted(df["goi_zone"].dropna().unique().tolist())
        )

    filtered = df[
        (df["prediction_score"] >= min_score) &
        (df["phase"].isin(selected_phase)) &
        (df["goi_zone"].isin(selected_goi))
    ].copy()

    filtered["expected_win_rate_60d_pct"] = (filtered["expected_win_rate_60d"] * 100).round(1)
    filtered["expected_return_60d_pct"] = (filtered["expected_return_60d"] * 100).round(1)
    filtered["confidence_pct"] = (filtered["confidence"] * 100).round(1)
    filtered["prediction_score"] = filtered["prediction_score"].round(1)

    st.markdown("### Live Signals")

    show_cols = [
        "ticker",
        "sector",
        "current_price",
        "prediction_score",
        "score_bucket",
        "calibration_bucket",
        "signal_tier",
        "expected_win_rate_60d_pct",
        "expected_return_60d_pct",
        "setup_label",
        "phase",
        "prior_phase",
        "goi_zone",
        "buyzone_flag",
        "confidence_pct",
        "greer_value_score",
        "greer_yield_score",
    ]

    display_df = filtered[show_cols].sort_values(
        ["expected_win_rate_60d_pct", "prediction_score"],
        ascending=[False, False]
    ).rename(columns={
        "ticker": "Ticker",
        "sector": "Sector",
        "current_price": "Price",
        "prediction_score": "Prediction Score",
        "score_bucket": "Raw Bucket",
        "calibration_bucket": "Calibration Bucket",
        "signal_tier": "Signal Tier",
        "expected_win_rate_60d_pct": "Expected 60d Win Rate %",
        "expected_return_60d_pct": "Expected 60d Return %",
        "setup_label": "Setup",
        "phase": "Phase",
        "prior_phase": "Prior Phase",
        "goi_zone": "GOI Zone",
        "buyzone_flag": "BuyZone",
        "confidence_pct": "Confidence %",
        "greer_value_score": "GV",
        "greer_yield_score": "GY",
    })

    def highlight_rows(row):
        if row["Signal Tier"] == "High Conviction":
            return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
        if row["Signal Tier"] == "Strong":
            return ["background-color: rgba(0, 150, 255, 0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("### Calibration Guide")
    calibration_df = pd.DataFrame([
        {"Calibration Bucket": 90, "Tier": "High Conviction", "Expected 60d Win Rate": "73.4%", "Expected 60d Return": "11.3%"},
        {"Calibration Bucket": 110, "Tier": "Strong", "Expected 60d Win Rate": "68.1%", "Expected 60d Return": "8.3%"},
        {"Calibration Bucket": 130, "Tier": "Constructive", "Expected 60d Win Rate": "64.6%", "Expected 60d Return": "6.8%"},
    ])
    st.dataframe(calibration_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()