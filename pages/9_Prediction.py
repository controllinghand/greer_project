# pages/9_Prediction.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from prediction_utils import calculate_prediction_score


# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

PREDICTION_PAGE_DESCRIPTION = """
The Prediction model estimates the probability that a stock will be higher over a multi-month horizon.

It combines company phase, phase transitions, market opportunity regime (GOI),
BuyZone state, confidence, and fundamentals as a light quality overlay.

Key findings from historical testing:
- The model is NOT a short-term trading system.
- The strongest signals are in the 110 score bucket.
- Performance improves with time (60 → 90 → 120 → 180 days).
- Best results occur when positions are held 120–180 trading days.
- Typical drawdowns of ~10% are normal before a trend develops.
- Tight stop losses reduce performance by cutting winning trends early.

This is a probability tool designed for trend capture, not short-term timing.
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
# Main
# ----------------------------------------------------------
def main():
    st.title("🔮 Prediction")
    st.caption("Probability-based multi-month outlook for current company setups.")
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

    st.warning("""
⚠️ **How to interpret this model**

- This is a medium-term trend system (not a quick trade)
- Many winners experience ~10% drawdowns before moving higher
- A 10% stop loss may exit strong future winners too early
- Best results come from patience and holding through volatility
""")

    # Optional filters: exclude known weak/noisy setups
    df = df[df["buyzone_flag"].notna()].copy()
    df = df[~((df["phase"] == "CONTRACTION") & (df["goi_zone"] == "EXTREME_GREED"))].copy()

    score_df = df.apply(calculate_prediction_score, axis=1)
    df = pd.concat([df, score_df], axis=1)

    # Top summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Optimal Signal Bucket", "110+")
        st.caption("Best balance of return and win rate")
    with c2:
        st.metric("Target Holding Period", "120–180 Days")
        st.caption("Edge improves with time")
    with c3:
        st.metric("Expected Drawdown", "~10% typical")
        st.caption("Temporary dips are normal")

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

    filtered["expected_win_rate_trend_pct"] = (filtered["expected_win_rate_trend"] * 100).round(1)
    filtered["expected_return_trend_pct"] = (filtered["expected_return_trend"] * 100).round(1)
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
        "signal_horizon",
        "expected_win_rate_trend_pct",
        "expected_return_trend_pct",
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
        ["expected_win_rate_trend_pct", "prediction_score"],
        ascending=[False, False]
    ).rename(columns={
        "ticker": "Ticker",
        "sector": "Sector",
        "current_price": "Price",
        "prediction_score": "Prediction Score",
        "score_bucket": "Raw Bucket",
        "calibration_bucket": "Calibration Bucket",
        "signal_tier": "Signal Tier",
        "signal_horizon": "Signal Horizon",
        "expected_win_rate_trend_pct": "Expected Win Rate (Trend) %",
        "expected_return_trend_pct": "Expected Return (Trend) %",
        "setup_label": "Setup",
        "phase": "Phase",
        "prior_phase": "Prior Phase",
        "goi_zone": "GOI Zone",
        "buyzone_flag": "BuyZone",
        "confidence_pct": "Confidence %",
        "greer_value_score": "GV",
        "greer_yield_score": "GY",
    })

    # ----------------------------------------------------------
    # Clean up display formatting
    # ----------------------------------------------------------
    display_df["Price"] = display_df["Price"].round(2)
    display_df["Prediction Score"] = display_df["Prediction Score"].round(1)
    display_df["Expected Win Rate (Trend) %"] = display_df["Expected Win Rate (Trend) %"].round(1)
    display_df["Expected Return (Trend) %"] = display_df["Expected Return (Trend) %"].round(1)

    # Buckets should be integers
    display_df["Raw Bucket"] = display_df["Raw Bucket"].astype("Int64")
    display_df["Calibration Bucket"] = display_df["Calibration Bucket"].astype("Int64")

    # Optional: GV / GY cleanup
    display_df["GV"] = display_df["GV"].round(0)
    display_df["GY"] = display_df["GY"].round(0)

    def highlight_rows(row):
        if row["Signal Tier"] == "Optimal":
            return ["background-color: rgba(0, 150, 255, 0.12)"] * len(row)
        if row["Signal Tier"] == "High Opportunity":
            return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
        if row["Signal Tier"] == "Over-Filtered":
            return ["background-color: rgba(255, 165, 0, 0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style
        .format({
            "Price": "{:.2f}",
            "Prediction Score": "{:.1f}",
            "Expected Win Rate (Trend) %": "{:.1f}",
            "Expected Return (Trend) %": "{:.1f}",
            "GV": "{:.0f}",
            "GY": "{:.0f}",
            "Raw Bucket": "{:.0f}",
            "Calibration Bucket": "{:.0f}",
            "Confidence %": "{:.0f}",
        })
        .apply(highlight_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("### Calibration Guide")
    calibration_df = pd.DataFrame([
        {
            "Calibration Bucket": 110,
            "Tier": "Optimal",
            "Expected 180d Win Rate": "~73%",
            "Expected 180d Return": "~20%",
            "Notes": "Best balance of return and consistency",
        },
        {
            "Calibration Bucket": 90,
            "Tier": "High Opportunity",
            "Expected 180d Win Rate": "~71%",
            "Expected 180d Return": "~19%",
            "Notes": "More signals, slightly more noise",
        },
        {
            "Calibration Bucket": 130,
            "Tier": "Over-Filtered",
            "Expected 180d Win Rate": "~71%",
            "Expected 180d Return": "~18%",
            "Notes": "Too selective, reduces opportunity",
        },
    ])
    st.dataframe(calibration_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()