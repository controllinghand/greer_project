# ----------------------------------------------------------
# 9_all_stars_cards.py
# Greer Value Levels — Critical (Level 3) Screener
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from db import get_engine

# ----------------------------------------------------------
# Custom CSS for cards
# ----------------------------------------------------------
st.markdown("""
<style>
  .gv-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    background: #fafafa;
    padding: 16px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }
  .gv-card-header {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .gv-card-meta {
    font-size: 0.9rem;
    color: #555;
    margin-bottom: 8px;
  }
  .value-line {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 12px;
  }
  .metrics-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
  .metric-box {
    flex: 1 1 120px;
    border-radius: 8px;
    padding: 8px 12px;
    color: white;
    font-weight: bold;
    text-align: center;
  }

  .gv-card a {
    color: #1976D2 !important;
    text-decoration: underline !important;
  }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Page Title
# ----------------------------------------------------------
st.markdown("<h1>🔴💲💲💲 Critical Value (Level 3)</h1>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Data Fetching
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_level3_companies():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange,
               delisted, delisted_date, greer_star_rating
        FROM companies
        WHERE greer_star_rating >= 3
        ORDER BY ticker;
        """,
        engine
    )

@st.cache_data(ttl=600)
def fetch_latest_snapshot(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"SELECT * FROM latest_company_snapshot WHERE ticker IN ({placeholders})"
    return pd.read_sql(query, engine, params=tuple(tickers))

@st.cache_data(ttl=600)
def fetch_latest_gfv(tickers):
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"""
    SELECT DISTINCT ON (ticker)
        ticker,
        close_price,
        gfv_price,
        gfv_status
    FROM greer_fair_value_daily
    WHERE ticker IN ({placeholders})
    ORDER BY ticker, date DESC
    """
    return pd.read_sql(query, engine, params=tuple(tickers))

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------
level3_df = fetch_level3_companies()

if level3_df.empty:
    st.info("No Critical Value companies found (Level 3).")
    st.stop()

tickers = level3_df["ticker"].tolist()
snap_df = fetch_latest_snapshot(tickers)
gfv_df = fetch_latest_gfv(tickers)

df = level3_df.merge(snap_df, how="left", on="ticker")
df = df.merge(
    gfv_df[["ticker", "close_price", "gfv_price", "gfv_status"]],
    how="left",
    on="ticker"
)

# ----------------------------------------------------------
# Render Cards
# ----------------------------------------------------------
for _, row in df.iterrows():
    ticker = row["ticker"]
    name = row.get("name", "")
    exchange = row.get("exchange", "")
    sector = row.get("sector", "")
    industry = row.get("industry", "")
    delisted = bool(row.get("delisted", False))

    gv = row.get("greer_value_score")
    yield_score = row.get("greer_yield_score")
    gfv = row.get("gfv_price")
    current_price = row.get("close_price")

    # Value Level (always Level 3 here)
    value_display = "💲💲💲 Critical"

    # Card HTML
    card_html = f"""
    <div class="gv-card">
      <div class="gv-card-header">
        <a href="/?ticker={ticker}" target="_self">
          {ticker} — {name}
        </a>
      </div>

      <div class="gv-card-meta">
        <b>Exchange:</b> {exchange} &nbsp;|&nbsp;
        <b>Sector:</b> {sector} &nbsp;|&nbsp;
        <b>Industry:</b> {industry}
      </div>

      { (f"<div class='gv-card-meta'><b>Delisted:</b> {row.get('delisted_date')}</div>" ) if delisted else "" }

      <div class="value-line">🔴 {value_display}</div>

      <div class="metrics-grid">
        <div class="metric-box" style="background:#2E7D32;">
          Greer Value<br>{ (f'{gv:.2f}%' if gv is not None else '—') }
        </div>

        <div class="metric-box" style="background:#1565C0;">
          Yield Score<br>{ (f'{int(yield_score)}/4' if yield_score is not None else '—') }
        </div>

        <div class="metric-box" style="background:#6A1B9A;">
          GFV<br>{ (f'${float(gfv):,.2f}' if gfv is not None else '—') }
        </div>

        <div class="metric-box" style="background:#111;">
          Price<br>{ (f'${float(current_price):,.2f}' if current_price is not None else '—') }
        </div>
      </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)