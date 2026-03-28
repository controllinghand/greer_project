# 15_Prediction_History.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine


# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Prediction History", page_icon="🔮", layout="wide")


# ----------------------------------------------------------
# DB connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    return get_engine()


# ----------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------
def fmt_pct(value, multiply=True, decimals=1):
    if pd.isna(value):
        return "—"
    if multiply:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def fmt_int(value):
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


# ----------------------------------------------------------
# Detect prod snapshot mode vs local view mode
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_latest_as_of_date():
    engine = get_connection()

    try:
        query = """
            SELECT MAX(as_of_date) AS as_of_date
            FROM prediction_backtest_overall
        """
        df = pd.read_sql(query, engine)

        if not df.empty and not pd.isna(df.iloc[0]["as_of_date"]):
            return df.iloc[0]["as_of_date"]
    except Exception:
        pass

    return None


# ----------------------------------------------------------
# Load overall summary
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_overall(as_of_date) -> pd.DataFrame:
    engine = get_connection()

    if as_of_date is not None:
        query = text("""
            SELECT *
            FROM prediction_backtest_overall
            WHERE as_of_date = :as_of_date
        """)
        return pd.read_sql(query, engine, params={"as_of_date": as_of_date})

    return pd.read_sql("SELECT * FROM prediction_backtest_overall_v", engine)


# ----------------------------------------------------------
# Load bucket stats
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_bucket_stats(as_of_date) -> pd.DataFrame:
    engine = get_connection()

    if as_of_date is not None:
        query = text("""
            SELECT *
            FROM prediction_backtest_bucket_stats
            WHERE as_of_date = :as_of_date
            ORDER BY calibration_bucket NULLS LAST
        """)
        return pd.read_sql(query, engine, params={"as_of_date": as_of_date})

    return pd.read_sql(
        """
        SELECT *
        FROM prediction_backtest_bucket_stats_v
        ORDER BY calibration_bucket NULLS LAST
        """,
        engine,
    )


# ----------------------------------------------------------
# Load phase x bucket stats
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_phase_bucket_stats(as_of_date) -> pd.DataFrame:
    engine = get_connection()

    if as_of_date is not None:
        query = text("""
            SELECT *
            FROM prediction_backtest_phase_bucket_stats
            WHERE as_of_date = :as_of_date
            ORDER BY phase, calibration_bucket NULLS LAST
        """)
        return pd.read_sql(query, engine, params={"as_of_date": as_of_date})

    return pd.read_sql(
        """
        SELECT *
        FROM prediction_backtest_phase_bucket_stats_v
        ORDER BY phase, calibration_bucket NULLS LAST
        """,
        engine,
    )


# ----------------------------------------------------------
# Load GOI x bucket stats
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_goi_bucket_stats(as_of_date) -> pd.DataFrame:
    engine = get_connection()

    if as_of_date is not None:
        query = text("""
            SELECT *
            FROM prediction_backtest_goi_bucket_stats
            WHERE as_of_date = :as_of_date
            ORDER BY goi_zone, calibration_bucket NULLS LAST
        """)
        return pd.read_sql(query, engine, params={"as_of_date": as_of_date})

    return pd.read_sql(
        """
        SELECT *
        FROM prediction_backtest_goi_bucket_stats_v
        ORDER BY goi_zone, calibration_bucket NULLS LAST
        """,
        engine,
    )


# ----------------------------------------------------------
# Build matrix table
# ----------------------------------------------------------
def build_matrix(df: pd.DataFrame, index_col: str, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index=index_col, columns="bucket_label", values=value_col)

    ordered_cols = [c for c in ["90", "110", "130", "Watchlist"] if c in pivot.columns]
    other_cols = [c for c in pivot.columns if c not in ordered_cols]
    return pivot[ordered_cols + other_cols]


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("🔮 Prediction History")
    st.caption("Historical validation for the Prediction model.")

    as_of_date = load_latest_as_of_date()

    if as_of_date is not None:
        st.caption(f"📦 Prod Snapshot: {as_of_date}")
    else:
        st.caption("🧪 Local Mode (views)")

    overall_df = load_overall(as_of_date)
    bucket_df = load_bucket_stats(as_of_date)
    phase_bucket_df = load_phase_bucket_stats(as_of_date)
    goi_bucket_df = load_goi_bucket_stats(as_of_date)

    if overall_df.empty:
        st.warning("No historical prediction summary data found.")
        return

    overall = overall_df.iloc[0]

    st.markdown(
        """
This page is the historical evidence layer behind the live Prediction model.

It helps answer:
- Which buckets performed best?
- Which market phases produced the strongest results?
- When does the model become selective?
- How do returns improve with time?
"""
    )

    # ----------------------------------------------------------
    # Historical Summary
    # ----------------------------------------------------------
    st.markdown("### Historical Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Backtest Rows", fmt_int(overall["total_rows"]))
    c2.metric("Total Signals", fmt_int(overall["total_signals"]))
    c3.metric("90 Bucket Win Rate", fmt_pct(overall["win_rate_90"]))
    c4.metric("110 Bucket Win Rate", fmt_pct(overall["win_rate_110"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("90 Avg Return (60d)", fmt_pct(overall["avg_return_60d_90"]))
    c6.metric("110 Avg Return (60d)", fmt_pct(overall["avg_return_60d_110"]))
    c7.metric("Watchlist Avg Return (60d)", fmt_pct(overall["avg_return_60d_watchlist"]))

    st.divider()

    # ----------------------------------------------------------
    # Bucket Performance
    # ----------------------------------------------------------
    st.markdown("### Bucket Performance")

    bucket_display = bucket_df.copy()
    bucket_display["rows"] = bucket_display["rows"].apply(fmt_int)
    bucket_display["win_rate_60d"] = bucket_display["win_rate_60d"].apply(lambda x: fmt_pct(x))
    bucket_display["avg_return_60d"] = bucket_display["avg_return_60d"].apply(lambda x: fmt_pct(x))
    bucket_display["avg_return_90d"] = bucket_display["avg_return_90d"].apply(lambda x: fmt_pct(x))
    bucket_display["avg_return_120d"] = bucket_display["avg_return_120d"].apply(lambda x: fmt_pct(x))
    bucket_display["avg_return_180d"] = bucket_display["avg_return_180d"].apply(lambda x: fmt_pct(x))

    st.dataframe(
        bucket_display.rename(columns={
            "bucket_label": "Bucket",
            "rows": "Rows",
            "win_rate_60d": "Win Rate (60d)",
            "avg_return_60d": "Avg Return (60d)",
            "avg_return_90d": "Avg Return (90d)",
            "avg_return_120d": "Avg Return (120d)",
            "avg_return_180d": "Avg Return (180d)",
        })[
            ["Bucket", "Rows", "Win Rate (60d)", "Avg Return (60d)", "Avg Return (90d)", "Avg Return (120d)", "Avg Return (180d)"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ----------------------------------------------------------
    # Phase x Bucket
    # ----------------------------------------------------------
    st.markdown("### Phase × Bucket")

    tab1, tab2 = st.tabs(["Win Rate (60d)", "Avg Return (60d)"])

    with tab1:
        win_matrix = build_matrix(phase_bucket_df, "phase", "win_rate_60d")
        if not win_matrix.empty:
            st.dataframe(
                win_matrix.style.format("{:.1%}").background_gradient(axis=None, cmap="Greens"),
                use_container_width=True,
            )

    with tab2:
        ret_matrix = build_matrix(phase_bucket_df, "phase", "avg_return_60d")
        if not ret_matrix.empty:
            st.dataframe(
                ret_matrix.style.format("{:.1%}").background_gradient(axis=None, cmap="RdYlGn"),
                use_container_width=True,
            )

    st.divider()

    # ----------------------------------------------------------
    # GOI x Bucket
    # ----------------------------------------------------------
    st.markdown("### GOI Zone × Bucket")

    tab3, tab4 = st.tabs(["Win Rate (60d)", "Avg Return (60d)"])

    with tab3:
        goi_win_matrix = build_matrix(goi_bucket_df, "goi_zone", "win_rate_60d")
        if not goi_win_matrix.empty:
            st.dataframe(
                goi_win_matrix.style.format("{:.1%}").background_gradient(axis=None, cmap="Greens"),
                use_container_width=True,
            )

    with tab4:
        goi_ret_matrix = build_matrix(goi_bucket_df, "goi_zone", "avg_return_60d")
        if not goi_ret_matrix.empty:
            st.dataframe(
                goi_ret_matrix.style.format("{:.1%}").background_gradient(axis=None, cmap="RdYlGn"),
                use_container_width=True,
            )

    st.divider()

    st.markdown("### Why This Matters")
    st.markdown(
        """
The live Prediction page shows **today's opportunities**.

This page shows the **historical evidence** behind those signals:
- which buckets were strongest,
- where the model became active,
- and how returns improved across longer horizons.
"""
    )


if __name__ == "__main__":
    main()