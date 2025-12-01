import streamlit as st
import pandas as pd
from db import get_engine

# Page config + custom CSS for cards
st.set_page_config(page_title="Greer 3-Star Companies", layout="wide")

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
  .star-line {
    font-size: 1.2rem;
    color: #D4AF37;
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

  /* --- new: link styling inside cards --- */
  .gv-card a {
    color: #1976D2 !important;
    text-decoration: underline !important;
  }
  .gv-card a:visited {
    color: #551A8B;
  }
  .gv-card a:hover {
    color: #145a9a !important;
  }
  .gv-card a:active {
    color: #0d3a6b;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⭐ All 3-Star Companies</h1>", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_3star_companies():
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

# Load data
stars_df = fetch_3star_companies()
if stars_df.empty:
    st.info("No 3-star companies found (greer_star_rating ≥ 3).")
    st.stop()

tickers = stars_df["ticker"].tolist()
snap_df = fetch_latest_snapshot(tickers)
gfv_df = fetch_latest_gfv(tickers)

df = stars_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df[["ticker","close_price","gfv_price","gfv_status"]], how="left", on="ticker")

# Iterate cards
for _, row in df.iterrows():
    ticker = row["ticker"]
    name = row.get("name", "")
    stars = int(row.get("greer_star_rating", 0))
    exchange = row.get("exchange", "")
    sector = row.get("sector", "")
    industry = row.get("industry", "")
    delisted = bool(row.get("delisted", False))

    gv = row.get("greer_value_score")
    yield_score = row.get("greer_yield_score")
    gfv = row.get("gfv_price")
    current_price = row.get("close_price")

    # Build star icons line
    star_icons = "★" * stars + "☆" * (3 - stars)

    # Build card HTML with link in header
    card_html = f"""
    <div class="gv-card">
      <div class="gv-card-header">
        <a href="/?ticker={ticker}" target="_self" style="text-decoration: none; color: inherit;">
          {ticker} — {name}
        </a>
      </div>
      <div class="gv-card-meta"><b>Exchange:</b> {exchange} &nbsp;|&nbsp; <b>Sector:</b> {sector} &nbsp;|&nbsp; <b>Industry:</b> {industry}</div>
      { (f"<div class='gv-card-meta'><b>Delisted:</b> {row.get('delisted_date')}</div>" ) if delisted else "" }
      <div class="star-line">{star_icons} {stars} Gold Star{'s' if stars>1 else ''}</div>

      <div class="metrics-grid">
        <div class="metric-box" style="background:#D4AF37;">
          Greer Value<br>{ (f'{gv:.2f}%' if gv is not None else '—') }
        </div>
        <div class="metric-box" style="background:#D4AF37;">
          Yield Score<br>{ (f'{int(yield_score)}/4' if yield_score is not None else '—') }
        </div>
        <div class="metric-box" style="background:#D4AF37;">
          GFV<br>{ (f'${float(gfv):,.2f}' if gfv is not None else '—') }
        </div>
        <div class="metric-box" style="background:#111; color:#fff;">
          Current Price<br>{ (f'${float(current_price):,.2f}' if current_price is not None else '—') }
        </div>
      </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
