# 8_all_stars.py
# ----------------------------------------------------------
# ⭐ All 3-Star Companies
# - Uses companies (static metadata)
# - Uses latest_company_snapshot (fast “latest” indicators: GV/Yield/BuyZone/FVG)
# - Uses company_snapshot for latest GFV + close (cheap, single table)
# - Uses company_snapshot for star transition dates (became 3⭐, fell out)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from datetime import date
from db import get_engine

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
# st.set_page_config(page_title="Greer 3-Star Companies", layout="wide")
st.markdown("<h1>⭐ All 3-Star Companies</h1>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_3star_companies():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange,
               delisted, delisted_date, greer_star_rating
        FROM public.companies
        WHERE greer_star_rating >= 3
        ORDER BY ticker;
        """,
        engine,
    )

@st.cache_data(ttl=600)
def fetch_latest_company_snapshot(tickers):
    """
    Pulls latest indicator fields for tickers from materialized view:
      - greer_value_score, above_50_count, greer_yield_score
      - buyzone_flag (+ bz_start_date/bz_end_date)
      - fvg_last_date/fvg_last_direction (active/unmitigated from fair_value_gaps)
      - first_trade_date, is_new_company
    """
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(tickers))
    return pd.read_sql(
        f"""
        SELECT *
        FROM public.latest_company_snapshot
        WHERE ticker IN ({placeholders})
        """,
        engine,
        params=tuple(tickers),
    )

@st.cache_data(ttl=600)
def fetch_latest_gfv_from_company_snapshot(tickers):
    """
    Gets the latest close + gfv_price from company_snapshot (per ticker).
    We compute a simple GFV status on the page (undervalued/overvalued) to avoid
    hitting greer_fair_value_daily.
    """
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(tickers))
    return pd.read_sql(
        f"""
        SELECT DISTINCT ON (ticker)
               ticker,
               snapshot_date::date AS gfv_date,
               close AS close_price,
               gfv_price
        FROM public.company_snapshot
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, snapshot_date DESC;
        """,
        engine,
        params=tuple(tickers),
    )

@st.cache_data(ttl=600)
def fetch_star_transitions(tickers):
    """
    Returns per ticker:
      - entered_3star_date: most recent date it crossed into >=3 (latest run entry)
      - exit_after_enter: first date it crossed from >=3 to <3 AFTER that latest entry (null if active)
      - last_exit_date: most recent exit date overall (kept for history)
      - tracking_start_date: when star tracking began (global)
      - asof_date: latest snapshot date for that ticker (for accurate day counts)
    """
    engine = get_engine()
    if not tickers:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(tickers))

    return pd.read_sql(
        f"""
        WITH tracking AS (
          SELECT MIN(snapshot_date)::date AS tracking_start_date
          FROM public.company_snapshot
          WHERE greer_star_rating IS NOT NULL
        ),
        s AS (
          SELECT
            cs.ticker,
            cs.snapshot_date::date AS d,
            cs.greer_star_rating AS star,
            LAG(cs.greer_star_rating)  OVER (PARTITION BY cs.ticker ORDER BY cs.snapshot_date) AS prev_star
          FROM public.company_snapshot cs
          WHERE cs.greer_star_rating IS NOT NULL
            AND cs.ticker IN ({placeholders})
        ),
        enters AS (
          SELECT ticker, d AS entered
          FROM s
          WHERE star >= 3 AND (prev_star < 3 OR prev_star IS NULL)
        ),
        last_enter AS (
          SELECT ticker, MAX(entered) AS entered_3star_date
          FROM enters
          GROUP BY ticker
        ),
        exits AS (
          SELECT ticker, d AS exited
          FROM s
          WHERE star < 3 AND prev_star >= 3
        ),
        exit_after_enter AS (
          SELECT e.ticker, MIN(e.exited) AS exit_after_enter
          FROM exits e
          JOIN last_enter le ON le.ticker = e.ticker
          WHERE e.exited > le.entered_3star_date
          GROUP BY e.ticker
        ),
        last_exit AS (
          SELECT ticker, MAX(exited) AS last_exit_date
          FROM exits
          GROUP BY ticker
        ),
        asof AS (
          SELECT ticker, MAX(snapshot_date)::date AS asof_date
          FROM public.company_snapshot
          WHERE ticker IN ({placeholders})
          GROUP BY ticker
        )
        SELECT
          le.ticker,
          le.entered_3star_date,
          ea.exit_after_enter,
          lx.last_exit_date,
          (SELECT tracking_start_date FROM tracking) AS tracking_start_date,
          a.asof_date
        FROM last_enter le
        LEFT JOIN exit_after_enter ea ON ea.ticker = le.ticker
        LEFT JOIN last_exit lx       ON lx.ticker = le.ticker
        LEFT JOIN asof a             ON a.ticker = le.ticker;
        """,
        engine,
        params=tuple(tickers) + tuple(tickers),  # placeholders used twice
    )


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def make_link(t):
    return f'<a href="/?ticker={t}" target="_self">{t}</a>'

def badge(v: bool) -> str:
    return "✅" if bool(v) else ""

def fmt_date(x):
    if pd.isna(x) or x is None:
        return ""
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)

def fmt_pct(x):
    if x is None or pd.isna(x):
        return ""
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return str(x)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
stars_df = fetch_3star_companies()
if stars_df.empty:
    st.info("No 3-star companies found (greer_star_rating ≥ 3).")
    st.stop()

tickers = stars_df["ticker"].tolist()

snap_df = fetch_latest_company_snapshot(tickers)
gfv_df = fetch_latest_gfv_from_company_snapshot(tickers)
tr_df = fetch_star_transitions(tickers)

# ----------------------------------------------------------
# Merge
# ----------------------------------------------------------
df = stars_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df, how="left", on="ticker")
df = df.merge(tr_df, how="left", on="ticker")

# ----------------------------------------------------------
# Normalize / Derived fields
# ----------------------------------------------------------
# BuyZone
if "buyzone_flag" not in df.columns:
    df["buyzone_flag"] = False
df["buyzone_flag"] = df["buyzone_flag"].fillna(False).astype(bool)

# FVG (from latest_company_snapshot: active/unmitigated)
if "fvg_last_direction" not in df.columns:
    df["fvg_last_direction"] = ""
df["fvg_last_direction"] = df["fvg_last_direction"].fillna("").astype(str)

if "fvg_last_date" not in df.columns:
    df["fvg_last_date"] = pd.NaT

df["fvg_bullish"] = df["fvg_last_direction"].str.upper().eq("BULLISH")
df["fvg_bearish"] = df["fvg_last_direction"].str.upper().eq("BEARISH")

# GFV status (simple, derived)
df["gfv_status"] = ""
if "close_price" in df.columns and "gfv_price" in df.columns:
    close = pd.to_numeric(df["close_price"], errors="coerce")
    gfv = pd.to_numeric(df["gfv_price"], errors="coerce")
    ratio = close / gfv
    # % vs GFV: positive = above GFV, negative = below GFV
    df["gfv_gap_pct"] = (ratio - 1.0) * 100.0

    df.loc[(close.notna()) & (gfv.notna()) & (close <= gfv), "gfv_status"] = "🟢 Below GFV"
    df.loc[(close.notna()) & (gfv.notna()) & (close > gfv), "gfv_status"] = "🔴 Above GFV"
else:
    df["gfv_gap_pct"] = None

# Days in 3⭐ (latest run)
df["entered_3star_date"] = pd.to_datetime(df.get("entered_3star_date"), errors="coerce").dt.date
df["exit_after_enter"]   = pd.to_datetime(df.get("exit_after_enter"), errors="coerce").dt.date
df["asof_date"]          = pd.to_datetime(df.get("asof_date"), errors="coerce").dt.date

from datetime import date
import pandas as pd

def calc_days(row):
    entered = row.get("entered_3star_date")
    exited  = row.get("exit_after_enter")
    asof    = row.get("asof_date")

    # entered is required
    if pd.isna(entered) or entered is None:
        return None

    # asof fallback
    if pd.isna(asof) or asof is None:
        asof = date.today()

    # if exited is missing, treat as active
    if pd.isna(exited) or exited is None:
        return (asof - entered).days + 1

    # exited is real
    if exited >= entered:
        return (exited - entered).days + 1

    # weird edge case (shouldn't happen)
    return None


df["days_in_3star"] = df.apply(calc_days, axis=1)


# ----------------------------------------------------------
# Show tracking start info
# ----------------------------------------------------------
tracking_start = None
if "tracking_start_date" in df.columns and df["tracking_start_date"].notna().any():
    tracking_start = pd.to_datetime(df["tracking_start_date"].dropna().iloc[0]).date()
    st.caption(f"⭐ Star tracking started on **{tracking_start.isoformat()}** (company_snapshot.greer_star_rating)")

# ----------------------------------------------------------
# Top summary metrics
# ----------------------------------------------------------
total_count = df["ticker"].nunique()
buyzone_count = df.loc[df["buyzone_flag"], "ticker"].nunique()
bullish_fvg_count = df.loc[df["fvg_bullish"], "ticker"].nunique()
bearish_fvg_count = df.loc[df["fvg_bearish"], "ticker"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("⭐ Total 3-Star+ Companies", f"{total_count:,}")
c2.metric("🟢 In BuyZone", f"{buyzone_count:,}")
c3.metric("🟩 Bullish FVG", f"{bullish_fvg_count:,}")
c4.metric("🟥 Bearish FVG", f"{bearish_fvg_count:,}")

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
left, mid, right = st.columns([1.2, 1.2, 2.6])

with left:
    zone_filter = st.selectbox(
        "Filter",
        ["All", "BuyZone only", "Bullish FVG only", "Bearish FVG only"],
        index=0,
    )

with mid:
    show_table = st.checkbox("Show as table (vs cards)", value=True)

with right:
    hide_delisted = st.checkbox("Hide delisted", value=True)

if hide_delisted and "delisted" in df.columns:
    df = df[df["delisted"] == False].copy()

if zone_filter == "BuyZone only":
    df = df[df["buyzone_flag"]]
elif zone_filter == "Bullish FVG only":
    df = df[df["fvg_bullish"]]
elif zone_filter == "Bearish FVG only":
    df = df[df["fvg_bearish"]]

if df.empty:
    st.info("No companies match the current filter.")
    st.stop()

# ----------------------------------------------------------
# Render
# ----------------------------------------------------------
if show_table:
    tbl = df[[
        "ticker", "name", "sector", "industry", "exchange",
        "greer_star_rating",
        "entered_3star_date",
        "days_in_3star",
        "exit_after_enter",
        "last_exit_date",
        "buyzone_flag",
        "bz_start_date",
        "bz_end_date",
        "greer_value_score",
        "above_50_count",
        "greer_yield_score",
        "fvg_last_direction",
        "fvg_last_date",
        "gfv_price",
        "gfv_status",
        "gfv_gap_pct",
        "close_price",
    ]].rename(columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "exchange": "Exchange",
        "greer_star_rating": "Stars",
        "entered_3star_date": "3⭐ Entered",
        "days_in_3star": "Days in 3⭐",
        "exit_after_enter": "Exited (current run)",
        "last_exit_date": "Last Exit (history)",
        "buyzone_flag": "BuyZone",
        "bz_start_date": "BZ Start",
        "bz_end_date": "BZ End",
        "greer_value_score": "Greer Value %",
        "above_50_count": "GV Above50 Count",
        "greer_yield_score": "Yield Score",
        "fvg_last_direction": "FVG Direction",
        "fvg_last_date": "FVG Date",
        "gfv_price": "Fair Value (GFV)",
        "gfv_status": "GFV Status",
        "gfv_gap_pct": "Price vs GFV",
        "close_price": "Current Price",
    })

    # Pretty + links
    tbl["Ticker"] = tbl["Ticker"].apply(make_link)
    tbl["BuyZone"] = tbl["BuyZone"].apply(badge)
    tbl["3⭐ Entered"] = tbl["3⭐ Entered"].apply(fmt_date)
    tbl["Exited (current run)"] = tbl["Exited (current run)"].apply(fmt_date)
    tbl["Last Exit (history)"] = tbl["Last Exit (history)"].apply(fmt_date)
    tbl["BZ Start"] = tbl["BZ Start"].apply(fmt_date)
    tbl["BZ End"] = tbl["BZ End"].apply(fmt_date)
    tbl["FVG Date"] = tbl["FVG Date"].apply(fmt_date)

    # Emoji labeling for direction
    def fmt_dir(x):
        s = str(x or "").upper()
        if s == "BULLISH":
            return "🟩 BULLISH"
        if s == "BEARISH":
            return "🟥 BEARISH"
        return x or ""

    tbl["FVG Direction"] = tbl["FVG Direction"].apply(fmt_dir)

    # % formatting for GFV gap
    tbl["Price vs GFV"] = tbl["Price vs GFV"].apply(fmt_pct)

    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        "Download CSV",
        tbl.to_csv(index=False).encode("utf-8"),
        "greer_3star_companies.csv",
        mime="text/csv",
    )

else:
    for _, row in df.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        stars = int(row.get("greer_star_rating", 0))

        # Zone + FVG
        zone = "🟢 BuyZone" if row.get("buyzone_flag") else "⚪ Neutral"
        fvg_dir = str(row.get("fvg_last_direction", "") or "").upper()
        fvg_date = fmt_date(row.get("fvg_last_date"))
        if fvg_dir == "BULLISH":
            fvg_txt = f"🟩 Bullish ({fvg_date})" if fvg_date else "🟩 Bullish"
        elif fvg_dir == "BEARISH":
            fvg_txt = f"🟥 Bearish ({fvg_date})" if fvg_date else "🟥 Bearish"
        else:
            fvg_txt = ""

        became = fmt_date(row.get("became_3star_date"))
        days_in = row.get("days_in_3star")
        days_in_txt = f"{int(days_in)} days" if pd.notna(days_in) else ""
        fell = fmt_date(row.get("fell_out_3star_date"))

        st.markdown(
            f"## ⭐ <a href='/?ticker={ticker}' target='_self'>{ticker}</a> — {name}  {'★'*stars}  &nbsp;&nbsp; {zone}",
            unsafe_allow_html=True,
        )

        st.write({
            "Became 3⭐ Date": became,
            "Days in 3⭐": days_in_txt,
            "Fell Out Date": fell,
            "Sector": row.get("sector"),
            "Industry": row.get("industry"),
            "Exchange": row.get("exchange"),
            "Greer Value %": row.get("greer_value_score"),
            "Yield Score": row.get("greer_yield_score"),
            "BuyZone Start": fmt_date(row.get("bz_start_date")),
            "BuyZone End": fmt_date(row.get("bz_end_date")),
            "FVG": fvg_txt,
            "Fair Value (GFV)": row.get("gfv_price"),
            "GFV Status": row.get("gfv_status"),
            "Price vs GFV": fmt_pct(row.get("gfv_gap_pct")),
            "Current Price": row.get("close_price"),
        })

        st.markdown("---")
