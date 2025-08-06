import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# ----------------------------------------------------------
# Database connection
# ----------------------------------------------------------
@st.cache_resource
def get_connection():
    engine = create_engine("postgresql://greer_user:@localhost:5432/yfinance_db")
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
# Page UI
# ----------------------------------------------------------
st.set_page_config(page_title="Greer Backtest Results", layout="wide")
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
# Results table
# ----------------------------------------------------------
st.subheader("📋 Backtest Results")
st.dataframe(
    filtered_df.sort_values("pct_return", ascending=False).style.format({
        "entry_close": "${:,.2f}",
        "last_close": "${:,.2f}",
        "pct_return": "{:.2f}%",
    }),
    use_container_width=True,
    hide_index=True,
)
