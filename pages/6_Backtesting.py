# 6_Backtesting.py
# ----------------------------------------------------------
# Greer Opportunity Backtest Results Page
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from db import get_engine

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
MIN_YOY_DAYS = 90

# ----------------------------------------------------------
# Database connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    return get_engine()

# ----------------------------------------------------------
# Load full backtest results
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_backtest_results():
    engine = get_connection()
    query = """
        SELECT *
        FROM backtest_results
        ORDER BY run_date DESC, ticker ASC
    """
    return pd.read_sql(query, engine)

# ----------------------------------------------------------
# Convert days_held to integer days
# Handles int OR timedelta
# ----------------------------------------------------------
def to_days_int(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.days.astype("int64")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")

# ----------------------------------------------------------
# Add YoY (annualized/CAGR) return %
# Only calculate YoY if days held meets minimum threshold
# ----------------------------------------------------------
def add_yoy_return(df: pd.DataFrame, min_yoy_days: int = MIN_YOY_DAYS) -> pd.DataFrame:
    out = df.copy()

    out["entry_close"] = pd.to_numeric(out["entry_close"], errors="coerce")
    out["last_close"] = pd.to_numeric(out["last_close"], errors="coerce")
    out["pct_return"] = pd.to_numeric(out["pct_return"], errors="coerce")
    out["days_held_days"] = to_days_int(out["days_held"])

    # ----------------------------------------------------------
    # Growth factor
    # ----------------------------------------------------------
    gf = (out["last_close"] / out["entry_close"]).replace([np.inf, -np.inf], np.nan)

    # ----------------------------------------------------------
    # Only annualize rows with enough holding time
    # ----------------------------------------------------------
    valid_yoy_mask = (
        out["days_held_days"].notna() &
        (out["days_held_days"] >= min_yoy_days) &
        out["entry_close"].notna() &
        out["last_close"].notna() &
        (out["entry_close"] > 0) &
        (out["last_close"] > 0)
    )

    out["yoy_return"] = np.nan

    out.loc[valid_yoy_mask, "yoy_return"] = (
        (gf.loc[valid_yoy_mask] ** (365.25 / out.loc[valid_yoy_mask, "days_held_days"])) - 1.0
    ) * 100.0

    out["yoy_return"] = out["yoy_return"].round(2)

    return out

# ----------------------------------------------------------
# Page UI
# ----------------------------------------------------------
st.title("📈 Greer Opportunity Backtest Results")

df = load_backtest_results()

if df.empty:
    st.warning("No backtest results found.")
    st.stop()

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
with st.sidebar:
    st.header("🔍 Filter Results")

    run_dates = sorted(df["run_date"].unique(), reverse=True)
    selected_date = st.selectbox("Select run date", run_dates)

    all_tickers = sorted(df["ticker"].unique())
    selected_tickers = st.multiselect("Filter tickers (optional)", all_tickers)

    min_yoy_days = st.number_input(
        "Minimum days for YoY calculation",
        min_value=30,
        max_value=365,
        value=MIN_YOY_DAYS,
        step=30
    )

# ----------------------------------------------------------
# Apply filters
# ----------------------------------------------------------
filtered_df = df[df["run_date"] == selected_date].copy()

if selected_tickers:
    filtered_df = filtered_df[filtered_df["ticker"].isin(selected_tickers)]

filtered_df = add_yoy_return(filtered_df, min_yoy_days=min_yoy_days)

# ----------------------------------------------------------
# YoY eligible subset
# ----------------------------------------------------------
yoy_df = filtered_df[filtered_df["yoy_return"].notna()].copy()

# ----------------------------------------------------------
# Summary stats
# ----------------------------------------------------------
st.subheader("📊 Summary Stats")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Count", len(filtered_df))
col2.metric("Win Rate", f"{(filtered_df['pct_return'] > 0).mean():.2%}")
col3.metric("Mean Return", f"{filtered_df['pct_return'].mean():.2f}%")
col4.metric("Median Return", f"{filtered_df['pct_return'].median():.2f}%")

# ----------------------------------------------------------
# YoY stats
# ----------------------------------------------------------
st.subheader("📈 YoY (Annualized) Stats")

c1, c2, c3, c4 = st.columns(4)

if yoy_df.empty:
    c1.metric("Mean YoY", "N/A")
    c2.metric("Median YoY", "N/A")
    c3.metric("YoY Win Rate", "N/A")
    c4.metric("Avg Years Held", "N/A")
    st.caption(f"No rows currently meet the {min_yoy_days}-day minimum for annualized YoY calculations.")
else:
    c1.metric("Mean YoY", f"{yoy_df['yoy_return'].mean():.2f}%")
    c2.metric("Median YoY", f"{yoy_df['yoy_return'].median():.2f}%")
    c3.metric("YoY Win Rate", f"{(yoy_df['yoy_return'] > 0).mean():.2%}")
    c4.metric("Avg Years Held", f"{(yoy_df['days_held_days'].mean() / 365.25):.2f}")

st.caption(f"YoY is only calculated for positions held at least {min_yoy_days} days.")

# ----------------------------------------------------------
# Results table
# ----------------------------------------------------------
st.subheader("📋 Backtest Results")

show_cols = [
    "ticker",
    "entry_date",
    "entry_close",
    "last_date",
    "last_close",
    "pct_return",
    "yoy_return",
    "days_held_days",
    "run_date"
]

display_df = filtered_df[show_cols].sort_values("pct_return", ascending=False).copy()

st.dataframe(
    display_df.style.format({
        "entry_close": "${:,.2f}",
        "last_close": "${:,.2f}",
        "pct_return": "{:.2f}%",
        "yoy_return": lambda x: "—" if pd.isna(x) else f"{x:.2f}%",
    }),
    use_container_width=True,
    hide_index=True,
)