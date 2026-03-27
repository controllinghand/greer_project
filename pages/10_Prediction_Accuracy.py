# pages/10_Prediction_Accuracy.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine


# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Prediction Accuracy",
    page_icon="🔮",
    layout="wide",
)


# ----------------------------------------------------------
# Page description
# ----------------------------------------------------------
PAGE_DESCRIPTION = """
The Prediction Accuracy dashboard tracks how the live prediction model performs over time.

This model is designed as a trend system, not a short-term trade signal.

It compares:
- Expected probabilities from the model (trend-based expectations)
- Actual outcomes over multiple horizons (20, 60, and 90 trading days)

Key concepts:
- The strongest signals occur in the 110+ bucket
- Performance improves with time (60 → 90 → 120 → 180 days)
- Drawdowns (~10%) are normal before trends develop
- Tight stop-losses can reduce model performance

This page helps answer:
- Are our probabilities holding up?
- Is the model staying calibrated?
- Which setups are strongest?
- How does performance evolve over time?
"""


# ----------------------------------------------------------
# DB connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    return get_engine()


# ----------------------------------------------------------
# Load top summary
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_summary() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            COUNT(*) AS total_predictions,
            MIN(prediction_date) AS first_prediction_date,
            MAX(prediction_date) AS latest_prediction_date,
            COUNT(DISTINCT ticker) AS distinct_tickers
        FROM prediction_tracking
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load daily prediction counts
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_daily_counts() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            prediction_date,
            COUNT(*) AS predictions
        FROM prediction_tracking
        GROUP BY prediction_date
        ORDER BY prediction_date DESC
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load signal tier distribution
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_signal_tier_distribution() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            signal_tier,
            COUNT(*) AS samples
        FROM prediction_tracking
        GROUP BY signal_tier
        ORDER BY samples DESC
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load calibration bucket distribution
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_bucket_distribution() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            calibration_bucket,
            COUNT(*) AS samples
        FROM prediction_tracking
        GROUP BY calibration_bucket
        ORDER BY calibration_bucket
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load setup distribution
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_setup_distribution() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            setup_label,
            COUNT(*) AS samples
        FROM prediction_tracking
        GROUP BY setup_label
        ORDER BY samples DESC
        LIMIT 20
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load 20d accuracy by calibration bucket
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_accuracy_20d_by_bucket() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            calibration_bucket,
            COUNT(*) AS samples,
            AVG(expected_win_rate_60d) AS expected_win_rate_reference,
            AVG(CASE WHEN actual_win_20d THEN 1.0 ELSE 0.0 END) AS actual_win_rate_20d,
            AVG(CASE WHEN actual_win_20d THEN 1.0 ELSE 0.0 END) - AVG(expected_win_rate_60d) AS calibration_drift_20d,
            AVG(actual_return_20d) AS actual_return_20d
        FROM prediction_tracking
        WHERE actual_win_20d IS NOT NULL
          AND calibration_bucket IS NOT NULL
        GROUP BY calibration_bucket
        ORDER BY calibration_bucket
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load 60d accuracy by calibration bucket
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_accuracy_60d_by_bucket() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            calibration_bucket,
            COUNT(*) AS samples,
            AVG(expected_win_rate_60d) AS expected_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) AS actual_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) - AVG(expected_win_rate_60d) AS calibration_drift_60d,
            AVG(actual_return_60d) AS actual_return_60d
        FROM prediction_tracking
        WHERE actual_win_60d IS NOT NULL
          AND calibration_bucket IS NOT NULL
        GROUP BY calibration_bucket
        ORDER BY calibration_bucket
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load 90d accuracy by calibration bucket
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_accuracy_90d_by_bucket() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            calibration_bucket,
            COUNT(*) AS samples,
            AVG(expected_win_rate_60d) AS expected_win_rate_reference,
            AVG(CASE WHEN actual_win_90d THEN 1.0 ELSE 0.0 END) AS actual_win_rate_90d,
            AVG(CASE WHEN actual_win_90d THEN 1.0 ELSE 0.0 END) - AVG(expected_win_rate_60d) AS calibration_drift_90d,
            AVG(actual_return_90d) AS actual_return_90d
        FROM prediction_tracking
        WHERE actual_win_90d IS NOT NULL
          AND calibration_bucket IS NOT NULL
        GROUP BY calibration_bucket
        ORDER BY calibration_bucket
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load 60d accuracy by signal tier
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_accuracy_60d_by_tier() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            signal_tier,
            COUNT(*) AS samples,
            AVG(expected_win_rate_60d) AS expected_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) AS actual_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) - AVG(expected_win_rate_60d) AS calibration_drift_60d,
            AVG(actual_return_60d) AS actual_return_60d
        FROM prediction_tracking
        WHERE actual_win_60d IS NOT NULL
        GROUP BY signal_tier
        ORDER BY samples DESC
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Load 60d accuracy by setup
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_accuracy_60d_by_setup() -> pd.DataFrame:
    engine = get_connection()

    query = text("""
        SELECT
            setup_label,
            COUNT(*) AS samples,
            AVG(expected_win_rate_60d) AS expected_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) AS actual_win_rate_60d,
            AVG(CASE WHEN actual_win_60d THEN 1.0 ELSE 0.0 END) - AVG(expected_win_rate_60d) AS calibration_drift_60d,
            AVG(actual_return_60d) AS actual_return_60d
        FROM prediction_tracking
        WHERE actual_win_60d IS NOT NULL
        GROUP BY setup_label
        HAVING COUNT(*) >= 10
        ORDER BY actual_win_rate_60d DESC
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Format percent columns
# ----------------------------------------------------------
def format_pct_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = (out[col] * 100).round(1)
    return out


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("🔮 Prediction Accuracy")
    st.caption("Track expected vs actual outcomes for the live prediction model.")
    st.markdown(PAGE_DESCRIPTION)

    st.info("""
📊 **Model Horizon Context**

- This is a trend-following system (120–180 trading days)
- 20d and 60d results are early indicators, not final outcomes
- Accuracy should improve as holding period increases
""")

    summary_df = load_summary()

    if summary_df.empty:
        st.warning("No prediction tracking data available yet.")
        return

    summary_row = summary_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Predictions", f"{int(summary_row['total_predictions']):,}")
    with c2:
        st.metric("First Prediction Date", str(summary_row["first_prediction_date"]))
    with c3:
        st.metric("Latest Prediction Date", str(summary_row["latest_prediction_date"]))
    with c4:
        st.metric("Distinct Tickers", f"{int(summary_row['distinct_tickers']):,}")

    st.divider()

    daily_df = load_daily_counts()
    tier_df = load_signal_tier_distribution()
    bucket_df = load_bucket_distribution()
    setup_df = load_setup_distribution()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Daily Prediction Counts")
        st.dataframe(
            daily_df.rename(columns={
                "prediction_date": "Prediction Date",
                "predictions": "Predictions",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        st.markdown("### Signal Tier Distribution")
        st.dataframe(
            tier_df.rename(columns={
                "signal_tier": "Signal Tier",
                "samples": "Samples",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Calibration Bucket Distribution")
        st.dataframe(
            bucket_df.rename(columns={
                "calibration_bucket": "Calibration Bucket",
                "samples": "Samples",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.markdown("### Most Common Setups")
    st.dataframe(
        setup_df.rename(columns={
            "setup_label": "Setup",
            "samples": "Samples",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("## Accuracy vs Expected")
    st.caption("""
Calibration Drift = Actual Win Rate − Expected Win Rate

- Positive → model is underestimating performance
- Negative → model is overestimating performance
""")

    acc20_df = load_accuracy_20d_by_bucket()
    acc60_bucket_df = load_accuracy_60d_by_bucket()
    acc90_df = load_accuracy_90d_by_bucket()
    acc60_tier_df = load_accuracy_60d_by_tier()
    acc60_setup_df = load_accuracy_60d_by_setup()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Early Signal (20d)",
        "Trend Progress (60d / 90d)",
        "By Signal Tier",
        "By Setup",
    ])

    with tab1:
        st.markdown("### 20d Accuracy by Calibration Bucket")
        if acc20_df.empty:
            st.info("No 20-day outcomes available yet.")
        else:
            show_df = format_pct_columns(
                acc20_df,
                ["expected_win_rate_reference", "actual_win_rate_20d", "calibration_drift_20d", "actual_return_20d"]
            ).rename(columns={
                "calibration_bucket": "Calibration Bucket",
                "samples": "Samples",
                "expected_win_rate_reference": "Expected Win Rate (Trend) %",
                "actual_win_rate_20d": "Actual Win Rate 20d %",
                "calibration_drift_20d": "Drift 20d %",
                "actual_return_20d": "Actual Return 20d %",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 60d Accuracy by Calibration Bucket")
        if acc60_bucket_df.empty:
            st.info("No 60-day outcomes available yet.")
        else:
            show_df = format_pct_columns(
                acc60_bucket_df,
                ["expected_win_rate_60d", "actual_win_rate_60d", "calibration_drift_60d", "actual_return_60d"]
            ).rename(columns={
                "calibration_bucket": "Calibration Bucket",
                "samples": "Samples",
                "expected_win_rate_60d": "Expected Win Rate (Trend) %",
                "actual_win_rate_60d": "Actual Win Rate 60d %",
                "calibration_drift_60d": "Drift 60d %",
                "actual_return_60d": "Actual Return 60d %",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.markdown("### 90d Accuracy by Calibration Bucket")
        if acc90_df.empty:
            st.info("No 90-day outcomes available yet.")
        else:
            show_df = format_pct_columns(
                acc90_df,
                ["expected_win_rate_reference", "actual_win_rate_90d", "calibration_drift_90d", "actual_return_90d"]
            ).rename(columns={
                "calibration_bucket": "Calibration Bucket",
                "samples": "Samples",
                "expected_win_rate_reference": "Expected Win Rate (Trend) %",
                "actual_win_rate_90d": "Actual Win Rate 90d %",
                "calibration_drift_90d": "Drift 90d %",
                "actual_return_90d": "Actual Return 90d %",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 60d Accuracy by Signal Tier")
        if acc60_tier_df.empty:
            st.info("No 60-day outcomes available yet.")
        else:
            show_df = format_pct_columns(
                acc60_tier_df,
                ["expected_win_rate_60d", "actual_win_rate_60d", "calibration_drift_60d", "actual_return_60d"]
            ).rename(columns={
                "signal_tier": "Signal Tier",
                "samples": "Samples",
                "expected_win_rate_60d": "Expected Win Rate (Trend) %",
                "actual_win_rate_60d": "Actual Win Rate 60d %",
                "calibration_drift_60d": "Drift 60d %",
                "actual_return_60d": "Actual Return 60d %",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### 60d Accuracy by Setup")
        if acc60_setup_df.empty:
            st.info("No 60-day outcomes available yet.")
        else:
            show_df = format_pct_columns(
                acc60_setup_df,
                ["expected_win_rate_60d", "actual_win_rate_60d", "calibration_drift_60d", "actual_return_60d"]
            ).rename(columns={
                "setup_label": "Setup",
                "samples": "Samples",
                "expected_win_rate_60d": "Expected Win Rate (Trend) %",
                "actual_win_rate_60d": "Actual Win Rate 60d %",
                "calibration_drift_60d": "Drift 60d %",
                "actual_return_60d": "Actual Return 60d %",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### 🧠 Key Insight")
    st.success("""
The Prediction model performs best when treated as a trend system:

- Higher scores (110+) show the strongest consistency
- Win rates improve as holding period increases
- Many winners require patience through early drawdowns
- Over-managing positions (tight stops, early exits) reduces performance

➡️ The edge comes from time in the trade, not timing the trade
""")


if __name__ == "__main__":
    main()