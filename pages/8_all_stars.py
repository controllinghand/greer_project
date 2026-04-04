# ----------------------------------------------------------
# 8_all_stars.py
# Greer Value Levels
# - Uses companies (static metadata)
# - Uses latest_company_snapshot (fast “latest” indicators: GV/Yield/BuyZone/FVG)
# - Uses company_snapshot for latest GFV + close (cheap, single table)
# - Uses company_snapshot for value-level transition dates
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from datetime import date
from db import get_engine
from value_utils import get_value_level, value_level_label

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
# st.set_page_config(page_title="Greer Value Levels", layout="wide")
st.markdown("<h1>💲 Greer Value Levels</h1>", unsafe_allow_html=True)
st.caption("DEFCON-style value signal system: $ Normal • $$ Elevated • $$$ Critical")

# ----------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_value_level_companies():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange,
               delisted, delisted_date, greer_star_rating
        FROM public.companies
        WHERE greer_star_rating >= 1
        ORDER BY greer_star_rating DESC, ticker;
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
    We compute a simple GFV status on the page.
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
def fetch_level_transitions(tickers):
    """
    Returns per ticker:
      - entered_current_level_date: most recent date it crossed into its current level
      - last_exit_critical_date: most recent exit date from Critical level overall
      - tracking_start_date: when tracking began
      - asof_date: latest snapshot date for that ticker
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
            cs.greer_star_rating AS level,
            LAG(cs.greer_star_rating) OVER (
              PARTITION BY cs.ticker
              ORDER BY cs.snapshot_date
            ) AS prev_level
          FROM public.company_snapshot cs
          WHERE cs.greer_star_rating IS NOT NULL
            AND cs.ticker IN ({placeholders})
        ),
        current_level AS (
          SELECT DISTINCT ON (ticker)
                 ticker,
                 level AS current_level,
                 d AS current_level_date
          FROM s
          ORDER BY ticker, d DESC
        ),
        entered_current AS (
          SELECT
            s.ticker,
            MAX(s.d) AS entered_current_level_date
          FROM s
          JOIN current_level cl
            ON s.ticker = cl.ticker
          WHERE s.level = cl.current_level
            AND (
              s.prev_level IS NULL
              OR s.prev_level <> cl.current_level
            )
          GROUP BY s.ticker
        ),
        critical_exits AS (
          SELECT ticker, d AS exited_critical_date
          FROM s
          WHERE level < 3 AND prev_level >= 3
        ),
        last_critical_exit AS (
          SELECT ticker, MAX(exited_critical_date) AS last_exit_critical_date
          FROM critical_exits
          GROUP BY ticker
        ),
        asof AS (
          SELECT ticker, MAX(snapshot_date)::date AS asof_date
          FROM public.company_snapshot
          WHERE ticker IN ({placeholders})
          GROUP BY ticker
        )
        SELECT
          cl.ticker,
          cl.current_level,
          ec.entered_current_level_date,
          lx.last_exit_critical_date,
          (SELECT tracking_start_date FROM tracking) AS tracking_start_date,
          a.asof_date
        FROM current_level cl
        LEFT JOIN entered_current ec ON ec.ticker = cl.ticker
        LEFT JOIN last_critical_exit lx ON lx.ticker = cl.ticker
        LEFT JOIN asof a ON a.ticker = cl.ticker;
        """,
        engine,
        params=tuple(tickers) + tuple(tickers),  # used in s + asof
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

def fmt_money(x):
    if x is None or pd.isna(x):
        return ""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
levels_df = fetch_value_level_companies()
if levels_df.empty:
    st.info("No value-level companies found.")
    st.stop()

tickers = levels_df["ticker"].tolist()

snap_df = fetch_latest_company_snapshot(tickers)
gfv_df = fetch_latest_gfv_from_company_snapshot(tickers)
tr_df = fetch_level_transitions(tickers)

# ----------------------------------------------------------
# Merge
# ----------------------------------------------------------
df = levels_df.merge(snap_df, how="left", on="ticker")
df = df.merge(gfv_df, how="left", on="ticker")
df = df.merge(tr_df, how="left", on="ticker")

# ----------------------------------------------------------
# Normalize / Derived fields
# ----------------------------------------------------------
df["value_level"] = df["greer_star_rating"].apply(get_value_level)
df["value_level_label"] = df["value_level"].apply(value_level_label)

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
if "close_price" in df.columns and "gfv_price" in df.columns:
    close = pd.to_numeric(df["close_price"], errors="coerce")
    gfv = pd.to_numeric(df["gfv_price"], errors="coerce")
    ratio = close / gfv
    df["gfv_gap_pct"] = (ratio - 1.0) * 100.0

    df.loc[(close.notna()) & (gfv.notna()) & (close <= gfv), "gfv_status"] = "🟢 Below GFV"
    df.loc[(close.notna()) & (gfv.notna()) & (close > gfv), "gfv_status"] = "🔴 Above GFV"
else:
    df["gfv_gap_pct"] = None

# Days in current value level
df["entered_current_level_date"] = pd.to_datetime(
    df.get("entered_current_level_date"), errors="coerce"
).dt.date
df["asof_date"] = pd.to_datetime(df.get("asof_date"), errors="coerce").dt.date

today = date.today()

def calc_days_in_level(row):
    entered = row.get("entered_current_level_date")
    asof = row.get("asof_date") or today

    if not entered:
        return None

    return (asof - entered).days + 1

df["days_in_level"] = df.apply(calc_days_in_level, axis=1)

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
level1_count = df.loc[df["value_level"] == 1, "ticker"].nunique()
level2_count = df.loc[df["value_level"] == 2, "ticker"].nunique()
level3_count = df.loc[df["value_level"] == 3, "ticker"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💲 Total Companies", f"{total_count:,}")
c2.metric("🟢 $ Normal", f"{level1_count:,}")
c3.metric("🟡 $$ Elevated", f"{level2_count:,}")
c4.metric("🔴 $$$ Critical", f"{level3_count:,}")

# ----------------------------------------------------------
# Filters
# ----------------------------------------------------------
left, mid, right, far = st.columns([1.4, 1.2, 1.2, 2.2])

with left:
    level_filter = st.selectbox(
        "Value Level",
        ["All", "Level 3 — Critical", "Level 2 — Elevated", "Level 1 — Normal"],
        index=0,
    )

with mid:
    zone_filter = st.selectbox(
        "Signal Filter",
        ["All", "BuyZone only", "Bullish FVG only", "Bearish FVG only"],
        index=0,
    )

with right:
    show_table = st.checkbox("Show as table (vs cards)", value=True)

with far:
    hide_delisted = st.checkbox("Hide delisted", value=True)

if hide_delisted and "delisted" in df.columns:
    df = df[df["delisted"] == False].copy()

if level_filter == "Level 3 — Critical":
    df = df[df["value_level"] == 3]
elif level_filter == "Level 2 — Elevated":
    df = df[df["value_level"] == 2]
elif level_filter == "Level 1 — Normal":
    df = df[df["value_level"] == 1]

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
        "ticker",
        "name",
        "sector",
        "industry",
        "exchange",
        "value_level",
        "value_level_label",
        "entered_current_level_date",
        "days_in_level",
        "last_exit_critical_date",
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
        "value_level": "Level",
        "value_level_label": "Value Signal",
        "entered_current_level_date": "Entered Level",
        "days_in_level": "Days in Level",
        "last_exit_critical_date": "Last Exit Critical",
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
    tbl["Entered Level"] = tbl["Entered Level"].apply(fmt_date)
    tbl["Last Exit Critical"] = tbl["Last Exit Critical"].apply(fmt_date)
    tbl["BZ Start"] = tbl["BZ Start"].apply(fmt_date)
    tbl["BZ End"] = tbl["BZ End"].apply(fmt_date)
    tbl["FVG Date"] = tbl["FVG Date"].apply(fmt_date)
    tbl["Fair Value (GFV)"] = tbl["Fair Value (GFV)"].apply(fmt_money)
    tbl["Current Price"] = tbl["Current Price"].apply(fmt_money)

    def fmt_dir(x):
        s = str(x or "").upper()
        if s == "BULLISH":
            return "🟩 BULLISH"
        if s == "BEARISH":
            return "🟥 BEARISH"
        return x or ""

    tbl["FVG Direction"] = tbl["FVG Direction"].apply(fmt_dir)
    tbl["Price vs GFV"] = tbl["Price vs GFV"].apply(fmt_pct)

    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        "Download CSV",
        tbl.to_csv(index=False).encode("utf-8"),
        "greer_value_levels.csv",
        mime="text/csv",
    )

else:
    for _, row in df.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        level = int(row.get("value_level", 0))
        level_label = row.get("value_level_label", "")

        zone = "🟢 BuyZone" if row.get("buyzone_flag") else "⚪ Neutral"
        fvg_dir = str(row.get("fvg_last_direction", "") or "").upper()
        fvg_date = fmt_date(row.get("fvg_last_date"))

        if fvg_dir == "BULLISH":
            fvg_txt = f"🟩 Bullish ({fvg_date})" if fvg_date else "🟩 Bullish"
        elif fvg_dir == "BEARISH":
            fvg_txt = f"🟥 Bearish ({fvg_date})" if fvg_date else "🟥 Bearish"
        else:
            fvg_txt = ""

        entered = fmt_date(row.get("entered_current_level_date"))
        days_in = row.get("days_in_level")
        days_in_txt = f"{int(days_in)} days" if pd.notna(days_in) else ""
        last_critical_exit = fmt_date(row.get("last_exit_critical_date"))

        st.markdown(
            f"## {level_label} <a href='/?ticker={ticker}' target='_self'>{ticker}</a> — {name}  &nbsp;&nbsp; {zone}",
            unsafe_allow_html=True,
        )

        st.write({
            "Value Level": level,
            "Entered Level Date": entered,
            "Days in Level": days_in_txt,
            "Last Exit Critical": last_critical_exit,
            "Sector": row.get("sector"),
            "Industry": row.get("industry"),
            "Exchange": row.get("exchange"),
            "Greer Value %": row.get("greer_value_score"),
            "Yield Score": row.get("greer_yield_score"),
            "BuyZone Start": fmt_date(row.get("bz_start_date")),
            "BuyZone End": fmt_date(row.get("bz_end_date")),
            "FVG": fvg_txt,
            "Fair Value (GFV)": fmt_money(row.get("gfv_price")),
            "GFV Status": row.get("gfv_status"),
            "Price vs GFV": fmt_pct(row.get("gfv_gap_pct")),
            "Current Price": fmt_money(row.get("close_price")),
        })

        st.markdown("---")