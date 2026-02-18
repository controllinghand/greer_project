# Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from db import get_engine

# Page config
# st.set_page_config(page_title="Company Dashboard", layout="wide")

st.markdown("""
<style>
  .card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background: #fafafa;
    padding: 16px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .card-header {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .star-rating {
    color: #D4AF37;
    margin-bottom: 8px;
    font-size: 1.1rem;
  }
  .metric-badge {
    padding: 6px 10px;
    border-radius: 4px;
    color: #fff;
    font-weight: bold;
    font-size: 0.9rem;
    margin-right: 6px;
    margin-bottom: 6px;
    display: inline-block;
  }
  .link-ticker {
    text-decoration: none;
    color: inherit;
  }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load the dashboard snapshot (fast read-only table)
# --------------------------------------------------
@st.cache_data(ttl=600)
def load_dashboard_snapshot() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        "SELECT * FROM dashboard_snapshot ORDER BY ticker;",
        engine
    )

df = load_dashboard_snapshot()
if df.empty:
    st.warning("dashboard_snapshot is empty. Run run_all.py (or build_dashboard_snapshot.py) to populate it.")
    st.stop()

st.markdown("# 📊 Company Dashboard — Overview (All Companies)")

last_updated = df["snapshot_date"].max() if "snapshot_date" in df.columns else None
if last_updated is not None:
    st.caption(f"Last updated: {last_updated}")

# Optional filters
with st.sidebar:
    min_stars = st.slider("Min star rating", min_value=0, max_value=3, value=0, step=1)
    only_buyzone = st.checkbox("Only show BuyZone = True", value=False)

filtered = df.copy()
if min_stars > 0:
    filtered = filtered[pd.to_numeric(filtered["greer_star_rating"], errors="coerce").fillna(0) >= min_stars]
if only_buyzone:
    filtered = filtered[filtered["buyzone_flag"] == True]

st.markdown(f"### Showing {len(filtered)} companies")

cards_per_row = 3
for i in range(0, len(filtered), cards_per_row):
    cols = st.columns(cards_per_row)
    for j, (_, row) in enumerate(filtered.iloc[i : i + cards_per_row].iterrows()):
        col = cols[j]
        with col:
            ticker = row.get("ticker", "")
            name = row.get("name", "")

            # --------------------------------------------------
            # NaN-safe parsing for key metrics
            # --------------------------------------------------
            stars_val = row.get("greer_star_rating")
            stars = int(stars_val) if pd.notnull(stars_val) else 0
            stars = max(0, min(3, stars))
            star_icons = "★" * stars + "☆" * (3 - stars)

            gv_val = row.get("greer_value_score")
            gv = float(gv_val) if pd.notnull(gv_val) else 0.0

            above50_val = row.get("above_50_count")
            above50 = int(above50_val) if pd.notnull(above50_val) else 0

            yld_val = row.get("greer_yield_score")
            yld_i = int(yld_val) if pd.notnull(yld_val) else 0  # integer 0–4 expected

            bz = bool(row.get("buyzone_flag"))
            fvg = row.get("fvg_last_direction") or ""
            price = row.get("current_price")
            gfv = row.get("gfv_price")
            gfv_status = str(row.get("gfv_status") or "").lower()

            # GV color logic
            if above50 == 6:
                gv_color = "#D4AF37"
            elif gv >= 50:
                gv_color = "#4CAF50"
            else:
                gv_color = "#F44336"

            # Yield color logic (use yld_i)
            if yld_i == 4:
                yld_color = "#D4AF37"
            elif yld_i == 3:
                yld_color = "#4CAF50"
            elif yld_i in (1, 2):
                yld_color = "#2196F3"
            else:
                yld_color = "#F44336"

            # BuyZone and FVG colors
            bz_color = "#4CAF50" if bz else "#9E9E9E"
            fvg_low = str(fvg).lower()
            if fvg_low == "bullish":
                fvg_color = "#4CAF50"
            elif fvg_low == "bearish":
                fvg_color = "#F44336"
            else:
                fvg_color = "#9E9E9E"

            # GFV status coloring
            if gfv_status == "gold":
                gfv_color = "#D4AF37"
            elif gfv_status == "green":
                gfv_color = "#4CAF50"
            elif gfv_status == "red":
                gfv_color = "#F44336"
            else:
                gfv_color = "#9E9E9E"

            # Render card
            col.markdown(f"""
            <div class="card">
              <div class="card-header">
                <a href="/?ticker={ticker}" class="link-ticker">{ticker} — {name}</a>
              </div>
              <div class="star-rating">{star_icons} {stars} Star{'s' if stars != 1 else ''}</div>

              <div>
                <span class="metric-badge" style="background:{gv_color};">GV: {gv:.1f}%</span>
                <span class="metric-badge" style="background:{yld_color};">Yield: {yld_i}/4</span>
                <span class="metric-badge" style="background:{gfv_color};">GFV: {f'${gfv:,.2f}' if pd.notnull(gfv) else '—'}</span>
                <span class="metric-badge" style="background:{bz_color};">BuyZone: {'Yes' if bz else 'No'}</span>
              </div>

              <div style="margin-top: 8px; font-size:0.9rem; color:#555;">
                Price: {f'${price:,.2f}' if pd.notnull(price) else '—'}
              </div>
            </div>
            """, unsafe_allow_html=True)
