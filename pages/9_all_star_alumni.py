# ----------------------------------------------------------
# 9_all_star_alumni.py
# ----------------------------------------------------------
# 💲 Critical Value Alumni (Re-Entry Watchlist)
# Companies that were once Level 3 / Critical historically,
# but are NOT currently Level 3 / Critical.
#
# - Uses companies (current metadata + current star rating)
# - Uses company_snapshot (historical star status + transition dates + latest GFV + close)
# - Uses latest_company_snapshot (fast “latest” indicators: GV/Yield/BuyZone/FVG)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from datetime import date
from db import get_engine
from value_utils import get_value_level, value_level_label, value_level_short

# ----------------------------------------------------------
# Page header
# ----------------------------------------------------------
st.markdown("<h1>💲 Critical Value Alumni (Re-Entry Watchlist)</h1>", unsafe_allow_html=True)
st.caption(
    "Companies that **previously reached Level 3 / Critical Value** but are **not currently Level 3**. "
    "These are strong candidates to monitor for re-entry signals."
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


def fmt_dir(x):
    s = str(x or "").upper()
    if s == "BULLISH":
        return "🟩 BULLISH"
    if s == "BEARISH":
        return "🟥 BEARISH"
    return x or ""


# ----------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_alumni_companies():
    """
    Returns companies that:
      - have EVER had greer_star_rating >= 3 in company_snapshot history
      - but have CURRENT greer_star_rating < 3 in companies
      - and are NOT delisted
    """
    engine = get_engine()
    return pd.read_sql(
        """
        WITH ever_3 AS (
          SELECT DISTINCT cs.ticker
          FROM public.company_snapshot cs
          WHERE cs.greer_star_rating >= 3
        )
        SELECT
          c.ticker,
          c.name,
          c.sector,
          c.industry,
          c.exchange,
          c.delisted,
          c.delisted_date,
          c.greer_star_rating AS current_star_rating
        FROM public.companies c
        JOIN ever_3 e ON e.ticker = c.ticker
        WHERE COALESCE(c.greer_star_rating, 0) < 3
          AND c.delisted = FALSE
        ORDER BY c.ticker;
        """,
        engine,
    )


@st.cache_data(ttl=600)
def fetch_latest_company_snapshot(tickers):
    """
    Pulls latest indicator fields for tickers from materialized view:
      - greer_value_score, above_50_count, greer_yield_score
      - buyzone_flag (+ bz_start_date/bz_end_date)
      - fvg_last_date/fvg_last_direction
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
    We compute a simple GFV status on the page (below/above) to avoid
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
def fetch_alumni_level_transitions(tickers):
    """
    Returns per ticker:
      - first_enter_3star_date: first time it crossed into >=3
      - last_enter_3star_date: most recent time it crossed into >=3
      - last_exit_3star_date:  most recent exit from >=3 to <3
      - tracking_start_date:   global earliest snapshot date with star data
      - asof_date:             latest snapshot date for that ticker
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
            LAG(cs.greer_star_rating) OVER (
              PARTITION BY cs.ticker
              ORDER BY cs.snapshot_date
            ) AS prev_star
          FROM public.company_snapshot cs
          WHERE cs.greer_star_rating IS NOT NULL
            AND cs.ticker IN ({placeholders})
        ),
        enters AS (
          SELECT ticker, d AS entered
          FROM s
          WHERE star >= 3 AND (prev_star < 3 OR prev_star IS NULL)
        ),
        first_enter AS (
          SELECT ticker, MIN(entered) AS first_enter_3star_date
          FROM enters
          GROUP BY ticker
        ),
        last_enter AS (
          SELECT ticker, MAX(entered) AS last_enter_3star_date
          FROM enters
          GROUP BY ticker
        ),
        exits AS (
          SELECT ticker, d AS exited
          FROM s
          WHERE star < 3 AND prev_star >= 3
        ),
        last_exit AS (
          SELECT ticker, MAX(exited) AS last_exit_3star_date
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
          fe.first_enter_3star_date,
          le.last_enter_3star_date,
          lx.last_exit_3star_date,
          (SELECT tracking_start_date FROM tracking) AS tracking_start_date,
          a.asof_date
        FROM last_enter le
        LEFT JOIN first_enter fe ON fe.ticker = le.ticker
        LEFT JOIN last_exit  lx ON lx.ticker = le.ticker
        LEFT JOIN asof       a  ON a.ticker  = le.ticker;
        """,
        engine,
        params=tuple(tickers) + tuple(tickers),
    )


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
alumni_df = fetch_alumni_companies()
if alumni_df.empty:
    st.info("No Critical Value alumni found.")
    st.stop()

tickers = alumni_df["ticker"].tolist()

snap_df = fetch_latest_company_snapshot(tickers)
gfv_df = fetch_latest_gfv_from_company_snapshot(tickers)
tr_df = fetch_alumni_level_transitions(tickers)

# ----------------------------------------------------------
# Merge
# ----------------------------------------------------------
df = alumni_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df, how="left", on="ticker")
df = df.merge(tr_df, how="left", on="ticker")

# ----------------------------------------------------------
# Normalize / Derived fields
# ----------------------------------------------------------
df["current_value_level"] = df["current_star_rating"].apply(get_value_level)
df["current_value_label"] = df["current_value_level"].apply(value_level_label)

# BuyZone
if "buyzone_flag" not in df.columns:
    df["buyzone_flag"] = False
df["buyzone_flag"] = df["buyzone_flag"].fillna(False).astype(bool)

# FVG
if "fvg_last_direction" not in df.columns:
    df["fvg_last_direction"] = ""
df["fvg_last_direction"] = df["fvg_last_direction"].fillna("").astype(str)

if "fvg_last_date" not in df.columns:
    df["fvg_last_date"] = pd.NaT

df["fvg_bullish"] = df["fvg_last_direction"].str.upper().eq("BULLISH")
df["fvg_bearish"] = df["fvg_last_direction"].str.upper().eq("BEARISH")

# GFV status
df["gfv_status"] = ""
df["gfv_gap_pct"] = None
if "close_price" in df.columns and "gfv_price" in df.columns:
    close = pd.to_numeric(df["close_price"], errors="coerce")
    gfv = pd.to_numeric(df["gfv_price"], errors="coerce")
    ratio = close / gfv
    df["gfv_gap_pct"] = (ratio - 1.0) * 100.0

    df.loc[(close.notna()) & (gfv.notna()) & (close <= gfv), "gfv_status"] = "🟢 Below GFV"
    df.loc[(close.notna()) & (gfv.notna()) & (close > gfv), "gfv_status"] = "🔴 Above GFV"

# Dates
df["first_enter_3star_date"] = pd.to_datetime(df.get("first_enter_3star_date"), errors="coerce").dt.date
df["last_enter_3star_date"] = pd.to_datetime(df.get("last_enter_3star_date"), errors="coerce").dt.date
df["last_exit_3star_date"] = pd.to_datetime(df.get("last_exit_3star_date"), errors="coerce").dt.date
df["asof_date"] = pd.to_datetime(df.get("asof_date"), errors="coerce").dt.date

today = date.today()

def calc_days_since_exit(row):
    exited = row.get("last_exit_3star_date")
    asof = row.get("asof_date") or today
    if not exited:
        return None
    return (asof - exited).days

df["days_since_exit"] = df.apply(calc_days_since_exit, axis=1)

# ----------------------------------------------------------
# Show tracking start info
# ----------------------------------------------------------
tracking_start = None
if "tracking_start_date" in df.columns and df["tracking_start_date"].notna().any():
    tracking_start = pd.to_datetime(df["tracking_start_date"].dropna().iloc[0]).date()
    st.caption(
        f"💲 Value-level tracking started on **{tracking_start.isoformat()}** "
        f"(mapped from company_snapshot.greer_star_rating)"
    )

# ----------------------------------------------------------
# Top summary metrics
# ----------------------------------------------------------
total_count = df["ticker"].nunique()
buyzone_count = df.loc[df["buyzone_flag"], "ticker"].nunique()
bullish_fvg_count = df.loc[df["fvg_bullish"], "ticker"].nunique()
below_gfv_count = df.loc[df["gfv_status"].astype(str).str.contains("Below GFV"), "ticker"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💲 Alumni Count", f"{total_count:,}")
c2.metric("🟢 In BuyZone", f"{buyzone_count:,}")
c3.metric("🟩 Bullish FVG", f"{bullish_fvg_count:,}")
c4.metric("🟢 Below GFV", f"{below_gfv_count:,}")

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
left, mid, right = st.columns([1.4, 1.2, 2.4])

with left:
    filter_mode = st.selectbox(
        "Filter",
        [
            "All",
            "BuyZone only",
            "Bullish FVG only",
            "Below GFV only",
            "BuyZone + Below GFV",
        ],
        index=0,
    )

with mid:
    show_table = st.checkbox("Show as table (vs cards)", value=True)

with right:
    max_days = st.number_input(
        "Only show alumni exited within last N days (0 = no limit)",
        min_value=0,
        max_value=3650,
        value=0,
        step=30,
    )

if max_days and max_days > 0:
    df = df[(df["days_since_exit"].notna()) & (df["days_since_exit"] <= int(max_days))].copy()

if filter_mode == "BuyZone only":
    df = df[df["buyzone_flag"]]
elif filter_mode == "Bullish FVG only":
    df = df[df["fvg_bullish"]]
elif filter_mode == "Below GFV only":
    df = df[df["gfv_status"] == "🟢 Below GFV"]
elif filter_mode == "BuyZone + Below GFV":
    df = df[(df["buyzone_flag"]) & (df["gfv_status"] == "🟢 Below GFV")]

if df.empty:
    st.info("No companies match the current filter.")
    st.stop()

# ----------------------------------------------------------
# Render
# ----------------------------------------------------------
if show_table:
    tbl = df[[
        "ticker",
        "name",
        "sector",
        "industry",
        "exchange",
        "current_value_level",
        "current_value_label",
        "first_enter_3star_date",
        "last_enter_3star_date",
        "last_exit_3star_date",
        "days_since_exit",
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
        "current_value_level": "Current Level",
        "current_value_label": "Current Value Signal",
        "first_enter_3star_date": "First Level 3 Entry",
        "last_enter_3star_date": "Last Level 3 Entry",
        "last_exit_3star_date": "Last Level 3 Exit",
        "days_since_exit": "Days Since Exit",
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

    tbl["Ticker"] = tbl["Ticker"].apply(make_link)
    tbl["BuyZone"] = tbl["BuyZone"].apply(badge)

    tbl["First Level 3 Entry"] = tbl["First Level 3 Entry"].apply(fmt_date)
    tbl["Last Level 3 Entry"] = tbl["Last Level 3 Entry"].apply(fmt_date)
    tbl["Last Level 3 Exit"] = tbl["Last Level 3 Exit"].apply(fmt_date)

    tbl["BZ Start"] = tbl["BZ Start"].apply(fmt_date)
    tbl["BZ End"] = tbl["BZ End"].apply(fmt_date)
    tbl["FVG Date"] = tbl["FVG Date"].apply(fmt_date)

    tbl["FVG Direction"] = tbl["FVG Direction"].apply(fmt_dir)
    tbl["Price vs GFV"] = tbl["Price vs GFV"].apply(fmt_pct)

    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        "Download CSV",
        tbl.to_csv(index=False).encode("utf-8"),
        "critical_value_alumni.csv",
        mime="text/csv",
    )

else:
    for _, row in df.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        current_level = int(row.get("current_value_level") or 0)
        current_label = row.get("current_value_label", "—")

        zone = "🟢 BuyZone" if row.get("buyzone_flag") else "⚪ Neutral"
        fvg_dir = str(row.get("fvg_last_direction", "") or "").upper()
        fvg_date = fmt_date(row.get("fvg_last_date"))

        if fvg_dir == "BULLISH":
            fvg_txt = f"🟩 Bullish ({fvg_date})" if fvg_date else "🟩 Bullish"
        elif fvg_dir == "BEARISH":
            fvg_txt = f"🟥 Bearish ({fvg_date})" if fvg_date else "🟥 Bearish"
        else:
            fvg_txt = ""

        last_exit = fmt_date(row.get("last_exit_3star_date"))
        days_since = row.get("days_since_exit")
        days_since_txt = f"{int(days_since)} days" if pd.notna(days_since) else ""

        st.markdown(
            f"## 💲 <a href='/?ticker={ticker}' target='_self'>{ticker}</a> — {name} &nbsp;&nbsp; {current_label} &nbsp;&nbsp; {zone}",
            unsafe_allow_html=True,
        )

        st.write({
            "Current Level": current_level,
            "Current Value Signal": current_label,
            "First Level 3 Entry": fmt_date(row.get("first_enter_3star_date")),
            "Last Level 3 Entry": fmt_date(row.get("last_enter_3star_date")),
            "Last Level 3 Exit": last_exit,
            "Days Since Exit": days_since_txt,
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