# opportunities-IV.py

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine  # ✅ Centralized DB connection

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
# Load opportunities snapshot
# --------------------------------------------------
@st.cache_data(ttl=300)
def load_snapshot() -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        SELECT *
        FROM opportunities_iv_snapshot
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM opportunities_iv_snapshot
        )
        ORDER BY greer_company_index DESC, iv_atm DESC NULLS LAST, ticker
        """
    )

    return pd.read_sql(query, engine)


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

    df = load_snapshot()

    if df.empty:
        st.info("No companies currently meet all conditions.")
        return

    df = df.copy()
    df["sector"] = df["sector"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")
    df["phase"] = df["phase"].fillna("UNKNOWN")

    if "last_entry_date" in df.columns:
        df["last_entry_date"] = pd.to_datetime(
            df["last_entry_date"], errors="coerce"
        ).dt.date

    if "iv_expiry" in df.columns:
        df["iv_expiry"] = pd.to_datetime(
            df["iv_expiry"], errors="coerce"
        ).dt.date

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
        "stars",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "buyzone_flag" in df.columns:
        df["buyzone_flag"] = df["buyzone_flag"].fillna(False)

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
    display_df["Confidence"] = display_df["confidence"].apply(
        lambda x: f"{x:.0f}%" if pd.notnull(x) else ""
    )

    for col in [
        "greer_company_index",
        "greer_value",
        "yield_score",
        "iv_atm",
        "current_price",
        "gfv_price",
        "gfv_mos",
    ]:
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

    export_cols = [
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

    existing_export_cols = [col for col in export_cols if col in filtered_df.columns]

    csv_export = filtered_df[existing_export_cols].rename(
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