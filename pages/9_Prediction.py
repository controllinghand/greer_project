# pages/9_Prediction.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from prediction_utils import calculate_prediction_score


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
    # ----------------------------------------------------------
    # Clean up display formatting
    # ----------------------------------------------------------
    display_df["Price"] = display_df["Price"].round(2)

    display_df["Prediction Score"] = display_df["Prediction Score"].round(1)

    display_df["Expected 60d Win Rate %"] = display_df["Expected 60d Win Rate %"].round(1)
    display_df["Expected 60d Return %"] = display_df["Expected 60d Return %"].round(1)

    # Buckets should be integers
    display_df["Raw Bucket"] = display_df["Raw Bucket"].astype("Int64")
    display_df["Calibration Bucket"] = display_df["Calibration Bucket"].astype("Int64")

    # Optional: GV / GY cleanup
    display_df["GV"] = display_df["GV"].round(0)
    display_df["GY"] = display_df["GY"].round(0)

    def highlight_rows(row):
        if row["Signal Tier"] == "High Conviction":
            return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
        if row["Signal Tier"] == "Strong":
            return ["background-color: rgba(0, 150, 255, 0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style
        .format({
            "Price": "{:.2f}",
            "Prediction Score": "{:.1f}",
            "Expected 60d Win Rate %": "{:.1f}",
            "Expected 60d Return %": "{:.1f}",
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
        {"Calibration Bucket": 90, "Tier": "High Conviction", "Expected 60d Win Rate": "73.4%", "Expected 60d Return": "11.3%"},
        {"Calibration Bucket": 110, "Tier": "Strong", "Expected 60d Win Rate": "68.1%", "Expected 60d Return": "8.3%"},
        {"Calibration Bucket": 130, "Tier": "Constructive", "Expected 60d Win Rate": "64.6%", "Expected 60d Return": "6.8%"},
    ])
    st.dataframe(calibration_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()