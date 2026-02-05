# 6_Backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
from db import get_engine  # ✅ Centralized DB connection

# ----------------------------------------------------------
# Database connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    engine = get_engine()
    return engine

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
    df = pd.read_sql(query, engine)
    return df

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
# ----------------------------------------------------------
# Convert days_held to integer days (handles int OR timedelta)
# ----------------------------------------------------------
def to_days_int(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.days.astype("int64")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")

# ----------------------------------------------------------
# Compute YoY (annualized/CAGR) return %
# ----------------------------------------------------------
def add_yoy_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["days_held_days"] = to_days_int(out["days_held"]).clip(lower=1)

    # Growth factor
    gf = (out["last_close"] / out["entry_close"]).replace([np.inf, -np.inf], np.nan)

    # Annualized (CAGR) YoY %
    out["yoy_return"] = ((gf ** (365.25 / out["days_held_days"])) - 1.0) * 100.0

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

# ----------------------------------------------------------
# Apply filters
# ----------------------------------------------------------
filtered_df = df[df["run_date"] == selected_date]
if selected_tickers:
    filtered_df = filtered_df[filtered_df["ticker"].isin(selected_tickers)]

# Add YoY stats/column
filtered_df = add_yoy_return(filtered_df)

# ----------------------------------------------------------
# Summary stats
# ----------------------------------------------------------
st.subheader("📊 Summary Stats")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Count", len(filtered_df))
col2.metric("Win Rate", f"{(filtered_df['pct_return'] > 0).mean():.2%}")
col3.metric("Mean Return", f"{filtered_df['pct_return'].mean():.2f}%")
col4.metric("Median Return", f"{filtered_df['pct_return'].median():.2f}%")

st.subheader("📈 YoY (Annualized) Stats")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean YoY", f"{filtered_df['yoy_return'].mean():.2f}%")
c2.metric("Median YoY", f"{filtered_df['yoy_return'].median():.2f}%")
c3.metric("YoY Win Rate", f"{(filtered_df['yoy_return'] > 0).mean():.2%}")
c4.metric("Avg Years Held", f"{(filtered_df['days_held_days'].mean() / 365.25):.2f}")

# ----------------------------------------------------------
# Results table
# ----------------------------------------------------------
st.subheader("📋 Backtest Results")

show_cols = [
    "ticker", "entry_date", "entry_close",
    "last_date", "last_close",
    "pct_return", "yoy_return",
    "days_held_days", "run_date"
]

st.dataframe(
    filtered_df[show_cols]
      .sort_values("pct_return", ascending=False)
      .style.format({
          "entry_close": "${:,.2f}",
          "last_close": "${:,.2f}",
          "pct_return": "{:.2f}%",
          "yoy_return": "{:.2f}%",
      }),
    use_container_width=True,
    hide_index=True,
)
