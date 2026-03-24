# opportunities-IV.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine  # ✅ Centralized DB connection
from market_cycle_utils import classify_phase_with_confidence

# Page config — update tab title
# st.set_page_config(page_title="⭐ Opportunities with IV", layout="wide")

# --------------------------------------------------
# Insert custom CSS for styled table
# --------------------------------------------------
st.markdown(
    """
<style>
  .op-table {
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
  }
  .op-table th, .op-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
    vertical-align: middle;
  }
  .op-table th {
    background-color: #1976D2;
    color: white;
    position: sticky;
    top: 0;
  }
  .op-table tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  .op-table tr:hover {
    background-color: #f1f1f1;
  }
  .star-icon {
    color: #D4AF37;
    font-size: 1.1rem;
  }
  a.ticker-link {
    color: #1976D2;
    text-decoration: none;
    font-weight: bold;
  }
  a.ticker-link:hover {
    text-decoration: underline;
  }
  .phase-chip {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .phase-recovery { background: #D9F1FB; color: #0B6E99; }
  .phase-expansion { background: #DFF3E3; color: #1E7A34; }
  .phase-euphoria { background: #FCE8CC; color: #9A5B00; }
  .phase-contraction { background: #F9D6D5; color: #A12622; }
  .phase-unknown { background: #ECECEC; color: #555555; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------------------------------------
# Company cycle scoring helpers
# ----------------------------------------------------------
def compute_health_pct(gv_score, above_50_count) -> float:
    gv = float(gv_score) if pd.notnull(gv_score) else 0.0
    a50 = float(above_50_count) if pd.notnull(above_50_count) else 0.0
    a50_pct = (a50 / 6.0) * 100.0

    score = (gv * 0.85) + (a50_pct * 0.15)
    return round(clamp(score, 0.0, 100.0), 1)


def compute_buyzone_score(buyzone_flag) -> float:
    return 25.0 if bool(buyzone_flag) else 75.0


def compute_fvg_score(fvg_last_direction) -> float:
    s = str(fvg_last_direction).strip().lower() if pd.notnull(fvg_last_direction) else ""

    if s in ["bullish", "up", "green"]:
        return 100.0
    if s in ["bearish", "down", "red"]:
        return 0.0
    return 50.0


def compute_direction_pct(buyzone_flag, fvg_last_direction, sector_direction_pct) -> float:
    buyzone_score = compute_buyzone_score(buyzone_flag)
    fvg_score = compute_fvg_score(fvg_last_direction)
    sector_score = float(sector_direction_pct) if pd.notnull(sector_direction_pct) else 50.0

    score = (
        (buyzone_score * 0.35) +
        (fvg_score * 0.35) +
        (sector_score * 0.30)
    )

    return round(clamp(score, 0.0, 100.0), 1)


def compute_gfv_score(gfv_status) -> float:
    s = str(gfv_status).strip().lower() if pd.notnull(gfv_status) else "gray"

    if s == "gold":
        return 100.0
    if s == "green":
        return 75.0
    if s == "gray":
        return 50.0
    return 0.0


def compute_opportunity_pct(greer_yield_score, gfv_status) -> float:
    ys = float(greer_yield_score) if pd.notnull(greer_yield_score) else 0.0
    ys_pct = (ys / 4.0) * 100.0
    gfv_pct = compute_gfv_score(gfv_status)

    score = (ys_pct * 0.5) + (gfv_pct * 0.5)
    return round(clamp(score, 0.0, 100.0), 1)


def compute_company_index(health_pct, direction_pct, opportunity_pct) -> float:
    score = (float(health_pct) + float(direction_pct) + float(opportunity_pct)) / 3.0
    return round(clamp(score, 0.0, 100.0), 2)


def compute_company_buyzone_proxy(direction_pct: float) -> float:
    return round(clamp(100.0 - float(direction_pct), 0.0, 100.0), 2)


def signal_emoji(value: float) -> str:
    x = float(value)
    if x <= 24:
        return "🔴"
    if x <= 49:
        return "🟡"
    if 49 < x < 51:
        return "🔵"
    if x <= 74:
        return "🟢"
    return "🟩"


def signal_pct(value: float) -> str:
    return f"{signal_emoji(value)} {float(value):.1f}%"


def phase_chip(phase_name: str) -> str:
    phase = str(phase_name).strip().upper()

    if phase == "RECOVERY":
        return "<span class='phase-chip phase-recovery'>🔵 Recovery</span>"
    if phase == "EXPANSION":
        return "<span class='phase-chip phase-expansion'>🟢 Expansion</span>"
    if phase == "EUPHORIA":
        return "<span class='phase-chip phase-euphoria'>🟠 Euphoria</span>"
    if phase == "CONTRACTION":
        return "<span class='phase-chip phase-contraction'>🔴 Contraction</span>"

    return "<span class='phase-chip phase-unknown'>⚪ Unknown</span>"


# --------------------------------------------------
# Load opportunities + company cycle inputs
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_filtered_companies() -> pd.DataFrame:
    engine = get_engine()
    query = text(
        """
        WITH live_bull_gaps AS (
          SELECT ticker, date AS entry_date
          FROM public.fair_value_gaps
          WHERE direction = 'bullish'
            AND mitigated = false
        ),
        last_entry AS (
          SELECT ticker, MAX(entry_date) AS last_entry_date
          FROM live_bull_gaps
          GROUP BY ticker
        ),
        latest_prices AS (
          SELECT ticker, close AS current_price, date
          FROM prices
          WHERE (ticker, date) IN (
               SELECT ticker, MAX(date) FROM prices GROUP BY ticker
            )
        ),
        latest_gfv AS (
          SELECT DISTINCT ON (g.ticker)
            g.ticker,
            g.gfv_price,
            g.gfv_status,
            g.date
          FROM greer_fair_value_daily g
          ORDER BY g.ticker, g.date DESC
        ),
        latest_sector AS (
          SELECT
            sector,
            summary_date,
            buyzone_pct,
            greer_market_index
          FROM sector_summary_daily
          WHERE summary_date = (
            SELECT MAX(summary_date) FROM sector_summary_daily
          )
        ),
        recent_iv AS (
          SELECT DISTINCT ON (ivs.ticker)
            ivs.ticker,
            ivs.iv_atm,
            ivs.expiry AS iv_expiry
          FROM iv_summary ivs
          JOIN (
            SELECT ticker, MAX(fetch_date) AS max_fetch_date
            FROM iv_summary
            GROUP BY ticker
          ) mf
            ON mf.ticker = ivs.ticker
           AND mf.max_fetch_date = ivs.fetch_date
          WHERE ivs.expiry >= CURRENT_DATE
          ORDER BY ivs.ticker, ivs.expiry ASC
        )
        SELECT
          l.ticker,
          c.name,
          c.sector,
          c.industry,
          c.greer_star_rating AS stars,
          l.greer_value_score AS greer_value,
          l.above_50_count,
          l.greer_yield_score AS yield_score,
          l.buyzone_flag,
          l.fvg_last_direction,
          le.last_entry_date,
          p.current_price,
          gfv.gfv_price,
          gfv.gfv_status,
          gfv.gfv_price * 0.75 AS gfv_mos,
          riv.iv_atm AS iv_atm,
          riv.iv_expiry AS iv_expiry,
          s.summary_date AS sector_summary_date,
          s.buyzone_pct AS sector_buyzone_pct,
          (100.0 - s.buyzone_pct) AS sector_direction_pct,
          s.greer_market_index AS sector_greer_market_index
        FROM last_entry le
        JOIN latest_company_snapshot l ON l.ticker = le.ticker
        JOIN companies c               ON l.ticker = c.ticker
        JOIN latest_prices p           ON p.ticker = l.ticker
        JOIN latest_gfv gfv            ON gfv.ticker = l.ticker
        LEFT JOIN recent_iv riv        ON riv.ticker = l.ticker
        LEFT JOIN latest_sector s      ON c.sector = s.sector
        WHERE l.greer_value_score >= 50
          AND l.greer_yield_score >= 3
          AND l.buyzone_flag = TRUE
          AND l.fvg_last_direction = 'bullish'
          AND p.current_price < gfv.gfv_price * 0.75
          AND c.delisted = FALSE
        ORDER BY
          riv.iv_atm DESC NULLS LAST,
          le.last_entry_date DESC,
          l.ticker;
        """
    )

    df = pd.read_sql(query, engine)

    if df.empty:
        return df

    company_rows = []

    for _, row in df.iterrows():
        health_pct = compute_health_pct(
            row["greer_value"],
            row["above_50_count"],
        )

        direction_pct = compute_direction_pct(
            row["buyzone_flag"],
            row["fvg_last_direction"],
            row["sector_direction_pct"],
        )

        opportunity_pct = compute_opportunity_pct(
            row["yield_score"],
            row["gfv_status"],
        )

        gci = compute_company_index(
            health_pct,
            direction_pct,
            opportunity_pct,
        )

        company_buyzone_proxy = compute_company_buyzone_proxy(direction_pct)

        phase, confidence = classify_phase_with_confidence(
            health_pct,
            company_buyzone_proxy,
            opportunity_pct,
        )

        row_dict = row.to_dict()
        row_dict.update(
            {
                "health_pct": health_pct,
                "direction_pct": direction_pct,
                "opportunity_pct": opportunity_pct,
                "greer_company_index": gci,
                "phase": phase,
                "confidence": round(float(confidence) * 100.0, 1),
            }
        )
        company_rows.append(row_dict)

    return pd.DataFrame(company_rows)


# --------------------------------------------------
# Main page
# --------------------------------------------------
def main():
    st.title("⭐ Opportunities with IV")
    st.markdown(
        """
        **Showing companies that currently meet all of the following criteria:**  
        - Greer Value ≥ **50**  
        - Yield Score ≥ **3**  
        - **In** the Buy-Zone  
        - Latest FVG direction is **bullish**  
        - Current Price < GFV Price × **0.75** (25% margin of safety)  

        **New Company Cycle layer:**  
        - Adds **Greer Company Index**, **Health**, **Direction**, **Opportunity**, **Phase**, and **Confidence**  
        - Uses the same company-cycle scoring logic as the dedicated Company Market Cycle page  

        **IV logic:**
        - Uses latest `fetch_date` per ticker  
        - Picks nearest expiry that is **≥ today**  
        """,
        unsafe_allow_html=True,
    )

    df = load_filtered_companies()
    if df.empty:
        st.info("No companies currently meet all conditions.")
        return

    df = df.copy()
    df["sector"] = df["sector"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")
    df["phase"] = df["phase"].fillna("UNKNOWN")

    if "last_entry_date" in df.columns:
        df["last_entry_date"] = pd.to_datetime(df["last_entry_date"], errors="coerce").dt.date

    if "iv_expiry" in df.columns:
        df["iv_expiry"] = pd.to_datetime(df["iv_expiry"], errors="coerce").dt.date

    numeric_cols = [
        "current_price",
        "gfv_price",
        "gfv_mos",
        "iv_atm",
        "greer_value",
        "yield_score",
        "health_pct",
        "direction_pct",
        "opportunity_pct",
        "greer_company_index",
        "confidence",
        "sector_direction_pct",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    with c1:
        sector_options = sorted(df["sector"].dropna().unique().tolist())
        selected_sectors = st.multiselect(
            "Filter sectors",
            options=sector_options,
            default=sector_options,
        )

    with c2:
        min_stars = st.selectbox(
            "Minimum Stars",
            options=[0, 1, 2, 3],
            index=0,
        )

    with c3:
        min_gci = st.slider(
            "Min GCI",
            min_value=0,
            max_value=100,
            value=55,
            step=5,
        )

    with c4:
        only_buyzone = st.checkbox("Only BuyZone", value=True)

    with c5:
        top_n = st.selectbox(
            "Rows to show",
            options=[25, 50, 100, 250, 500],
            index=2,
        )

    a1, a2, a3 = st.columns([2, 1, 1])

    with a1:
        phase_options = ["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION"]
        phase_filter = st.multiselect(
            "Filter phases",
            options=phase_options,
            default=["RECOVERY", "EXPANSION"],
        )

    with a2:
        sort_choice = st.selectbox(
            "Sort by",
            options=[
                "Greer Company Index",
                "Health %",
                "Direction %",
                "Opportunity %",
                "IV ATM",
                "Greer Value %",
                "Yield Score",
                "Last Gap Date",
                "Ticker",
            ],
            index=0,
        )

    with a3:
        sort_desc = st.checkbox("Descending", value=True)

    with st.expander("Advanced company cycle filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            min_health = st.slider("Minimum Health", 0, 100, 0, 5)
        with f2:
            min_direction = st.slider("Minimum Direction", 0, 100, 0, 5)
        with f3:
            min_opportunity = st.slider("Minimum Opportunity", 0, 100, 0, 5)

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["sector"].isin(selected_sectors)]
    filtered_df = filtered_df[filtered_df["stars"].fillna(0) >= min_stars]
    filtered_df = filtered_df[filtered_df["greer_company_index"].fillna(0) >= min_gci]
    filtered_df = filtered_df[filtered_df["health_pct"].fillna(0) >= min_health]
    filtered_df = filtered_df[filtered_df["direction_pct"].fillna(0) >= min_direction]
    filtered_df = filtered_df[filtered_df["opportunity_pct"].fillna(0) >= min_opportunity]

    if phase_filter:
        filtered_df = filtered_df[filtered_df["phase"].isin(phase_filter)]

    if only_buyzone:
        filtered_df = filtered_df[filtered_df["buyzone_flag"] == True]

    if filtered_df.empty:
        st.info("No companies currently match the selected filters.")
        return

    sort_map = {
        "Greer Company Index": "greer_company_index",
        "Health %": "health_pct",
        "Direction %": "direction_pct",
        "Opportunity %": "opportunity_pct",
        "IV ATM": "iv_atm",
        "Greer Value %": "greer_value",
        "Yield Score": "yield_score",
        "Last Gap Date": "last_entry_date",
        "Ticker": "ticker",
    }

    sort_col = sort_map[sort_choice]
    filtered_df = filtered_df.sort_values(
        by=[sort_col, "greer_company_index", "iv_atm", "ticker"],
        ascending=[not sort_desc, False, False, True],
        na_position="last",
    ).head(top_n)

    # --------------------------------------------------
    # Summary metrics
    # --------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Companies", len(filtered_df))
    m2.metric("Avg GCI", f"{filtered_df['greer_company_index'].mean():.1f}")
    m3.metric("Avg Health", f"{filtered_df['health_pct'].mean():.1f}%")
    m4.metric("Avg Opportunity", f"{filtered_df['opportunity_pct'].mean():.1f}%")

    st.subheader(f"📈 {len(filtered_df)} matching companies")

    # --------------------------------------------------
    # Display helpers
    # --------------------------------------------------
    def stars_to_html(n):
        try:
            n = int(n)
        except Exception:
            return ""
        return f"<span class='star-icon'>{'★' * n}{'☆' * (3 - n)}</span>"

    def link_ticker(t: str) -> str:
        return f"<a href='/?ticker={t}' class='ticker-link'>{t}</a>"

    display_df = filtered_df.copy()
    display_df["Ticker"] = display_df["ticker"].apply(link_ticker)
    display_df["Stars"] = display_df["stars"].apply(stars_to_html)
    display_df["Phase"] = display_df["phase"].apply(phase_chip)
    display_df["Health"] = display_df["health_pct"].apply(signal_pct)
    display_df["Direction"] = display_df["direction_pct"].apply(signal_pct)
    display_df["Opportunity"] = display_df["opportunity_pct"].apply(signal_pct)
    display_df["Confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")

    for col in ["greer_company_index", "greer_value", "yield_score", "iv_atm", "current_price", "gfv_price", "gfv_mos"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    df_display = display_df[
        [
            "Ticker",
            "sector",
            "Stars",
            "greer_company_index",
            "Health",
            "Direction",
            "Opportunity",
            "Phase",
            "Confidence",
            "greer_value",
            "yield_score",
            "iv_atm",
            "iv_expiry",
            "current_price",
            "gfv_price",
            "gfv_mos",
            "last_entry_date",
        ]
    ].rename(
        columns={
            "sector": "Sector",
            "greer_company_index": "Greer Company Index",
            "greer_value": "Greer Value %",
            "yield_score": "Yield Score",
            "iv_atm": "IV ATM",
            "iv_expiry": "IV Expiry",
            "current_price": "Current Price",
            "gfv_price": "GFV",
            "gfv_mos": "GFV 75% MOS",
            "last_entry_date": "Last Gap Date",
        }
    )

    html_table = df_display.to_html(
        index=False,
        escape=False,
        classes="op-table",
    )

    st.markdown(html_table, unsafe_allow_html=True)

    csv_export = filtered_df[
        [
            "ticker",
            "name",
            "sector",
            "industry",
            "stars",
            "greer_company_index",
            "health_pct",
            "direction_pct",
            "opportunity_pct",
            "phase",
            "confidence",
            "greer_value",
            "above_50_count",
            "yield_score",
            "buyzone_flag",
            "fvg_last_direction",
            "gfv_status",
            "sector_direction_pct",
            "iv_atm",
            "iv_expiry",
            "current_price",
            "gfv_price",
            "gfv_mos",
            "last_entry_date",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "sector": "Sector",
            "industry": "Industry",
            "stars": "Stars",
            "greer_company_index": "Greer Company Index",
            "health_pct": "Health %",
            "direction_pct": "Direction %",
            "opportunity_pct": "Opportunity %",
            "phase": "Phase",
            "confidence": "Confidence %",
            "greer_value": "Greer Value %",
            "above_50_count": "Above 50 Count",
            "yield_score": "Yield Score",
            "buyzone_flag": "BuyZone Flag",
            "fvg_last_direction": "FVG Last Direction",
            "gfv_status": "GFV Status",
            "sector_direction_pct": "Sector Direction %",
            "iv_atm": "IV ATM",
            "iv_expiry": "IV Expiry",
            "current_price": "Current Price",
            "gfv_price": "GFV",
            "gfv_mos": "GFV 75% MOS",
            "last_entry_date": "Last Gap Date",
        }
    )

    st.download_button(
        "Download CSV",
        csv_export.to_csv(index=False).encode("utf-8"),
        file_name="greer_gfv_opportunities_with_company_index.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
