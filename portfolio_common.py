# portfolio_common.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, bindparam
from db import get_engine
from datetime import date, datetime

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def fmt_money(x):
    try:
        if x is None or pd.isna(x):
            return ""
        return f"${float(x):,.2f}"
    except Exception:
        return ""

def fmt_money0(x):
    try:
        if x is None or pd.isna(x):
            return ""
        return f"${float(x):,.0f}"
    except Exception:
        return ""

def fmt_pct_ratio(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

def safe_upper(s: str) -> str:
    return (s or "").strip().upper()

def to_date(x) -> date | None:
    try:
        if x is None or pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

# ----------------------------------------------------------
# DB: Portfolio helpers
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_portfolio_by_code(code: str) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT portfolio_id, code, name, start_date, starting_cash
        FROM portfolios
        WHERE code = :code
        LIMIT 1
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"code": safe_upper(code)})

@st.cache_data(ttl=300)
def load_nav_series(portfolio_id: int, start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT nav_date, cash, equity_value, nav
        FROM portfolio_nav_daily
        WHERE portfolio_id = :portfolio_id
          AND nav_date >= :start_date
        ORDER BY nav_date ASC
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

@st.cache_data(ttl=300)
def load_nav_series_between(
    portfolio_id: int,
    start_date: date,
    end_date: date | None = None
) -> pd.DataFrame:
    engine = get_engine()

    if end_date is None:
        q = text("""
            SELECT nav_date, cash, equity_value, nav
            FROM portfolio_nav_daily
            WHERE portfolio_id = :portfolio_id
              AND nav_date >= :start_date
            ORDER BY nav_date ASC
        """)
        params = {"portfolio_id": int(portfolio_id), "start_date": start_date}
    else:
        q = text("""
            SELECT nav_date, cash, equity_value, nav
            FROM portfolio_nav_daily
            WHERE portfolio_id = :portfolio_id
              AND nav_date >= :start_date
              AND nav_date <= :end_date
            ORDER BY nav_date ASC
        """)
        params = {"portfolio_id": int(portfolio_id), "start_date": start_date, "end_date": end_date}

    with engine.connect() as conn:
        return pd.read_sql(q, conn, params=params)

# ----------------------------------------------------------
# DB: Events
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_events_stockfund(portfolio_id: int, start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          event_id,
          event_time,
          event_type,
          ticker,
          quantity,
          price,
          fees,
          cash_delta,
          notes
        FROM portfolio_events
        WHERE portfolio_id = :portfolio_id
          AND event_time >= CAST(:start_date AS timestamp)
        ORDER BY event_time ASC, event_id ASC
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

@st.cache_data(ttl=300)
def load_events_optionsfund(portfolio_id: int, start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          event_id,
          event_time,
          event_type,
          ticker,
          quantity,
          price,
          fees,
          option_type,
          strike,
          expiry,
          cash_delta,
          notes
        FROM portfolio_events
        WHERE portfolio_id = :portfolio_id
          AND event_time >= CAST(:start_date AS timestamp)
        ORDER BY event_time ASC, event_id ASC
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

@st.cache_data(ttl=300)
def load_totals_by_type(portfolio_id: int, start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          event_type,
          COUNT(*) AS events,
          SUM(COALESCE(fees,0)) AS total_fees,
          SUM(COALESCE(cash_delta,0)) AS total_cash_delta
        FROM portfolio_events
        WHERE portfolio_id = :portfolio_id
          AND event_time >= CAST(:start_date AS timestamp)
        GROUP BY event_type
        ORDER BY total_cash_delta DESC NULLS LAST
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

@st.cache_data(ttl=300)
def load_totals_by_ticker(portfolio_id: int, start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          ticker,
          COUNT(*) AS events,
          SUM(COALESCE(fees,0)) AS total_fees,
          SUM(COALESCE(cash_delta,0)) AS total_cash_delta
        FROM portfolio_events
        WHERE portfolio_id = :portfolio_id
          AND event_time >= CAST(:start_date AS timestamp)
          AND ticker IS NOT NULL AND ticker <> ''
        GROUP BY ticker
        ORDER BY total_cash_delta DESC NULLS LAST
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

# ----------------------------------------------------------
# DB: Holdings helpers (latest prices + company names)
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_latest_prices_and_names(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "name", "last_close", "last_date"])

    engine = get_engine()
    q = text("""
        WITH latest AS (
            SELECT p.ticker, MAX(p.date) AS last_date
            FROM prices p
            WHERE p.ticker IN :tickers
            GROUP BY p.ticker
        )
        SELECT
            l.ticker,
            c.name,
            p.close AS last_close,
            l.last_date
        FROM latest l
        JOIN prices p
          ON p.ticker = l.ticker
         AND p.date = l.last_date
        LEFT JOIN companies c
          ON c.ticker = l.ticker
        ORDER BY l.ticker
    """).bindparams(bindparam("tickers", expanding=True))

    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"tickers": tickers})

# ----------------------------------------------------------
# Stock-fund analytics (BUY_SHARES / SELL_SHARES only)
# ----------------------------------------------------------
def calc_cashflows_stockfund(events: pd.DataFrame) -> dict:
    if events is None or events.empty:
        return {"fees_total": 0.0, "deposits_net": 0.0, "trade_cashflow": 0.0}

    e = events.copy()
    e["event_type"] = e["event_type"].astype(str).str.upper()
    e["cash_delta"] = pd.to_numeric(e["cash_delta"], errors="coerce").fillna(0.0)
    e["fees"] = pd.to_numeric(e["fees"], errors="coerce").fillna(0.0)

    fees_total = float(e["fees"].sum())

    dep_mask = e["event_type"].isin(["DEPOSIT", "CASH_DEPOSIT", "CONTRIBUTION"])
    wdr_mask = e["event_type"].isin(["WITHDRAW", "WITHDRAWAL", "CASH_WITHDRAWAL", "DISTRIBUTION"])
    deposits_net = float(e.loc[dep_mask, "cash_delta"].sum() + e.loc[wdr_mask, "cash_delta"].sum())

    trade_mask = e["event_type"].isin(["BUY_SHARES", "SELL_SHARES"])
    trade_cashflow = float(e.loc[trade_mask, "cash_delta"].sum())

    return {"fees_total": fees_total, "deposits_net": deposits_net, "trade_cashflow": trade_cashflow}

def calc_pnl_avg_cost(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "shares", "cost_basis", "avg_cost",
                "realized_pl", "realized_cost", "realized_proceeds", "realized_pct",
            ]
        )

    e = events.copy()
    e["event_type"] = e["event_type"].astype(str).str.upper()
    e["ticker"] = e["ticker"].fillna("").astype(str).str.upper().str.strip()
    e = e[(e["ticker"] != "")]
    # Treat assignment events as share buys for holdings/cost basis
    share_buy_types = {"BUY_SHARES", "ASSIGN_PUT"}
    share_sell_types = {"SELL_SHARES"}  # keep simple for now

    e = e[e["event_type"].isin(share_buy_types | share_sell_types)].copy()


    e["event_time"] = pd.to_datetime(e["event_time"], errors="coerce")
    e = e.sort_values(["ticker", "event_time", "event_id"], ascending=True)

    e["quantity"] = pd.to_numeric(e["quantity"], errors="coerce").fillna(0.0)
    e["price"] = pd.to_numeric(e["price"], errors="coerce")
    e["fees"] = pd.to_numeric(e["fees"], errors="coerce").fillna(0.0)
    e["cash_delta"] = pd.to_numeric(e["cash_delta"], errors="coerce")

    missing_price = e["price"].isna() | (e["price"] <= 0)
    can_infer = missing_price & e["cash_delta"].notna() & (e["quantity"].abs() > 0)
    e.loc[can_infer, "price"] = (e.loc[can_infer, "cash_delta"].abs() / e.loc[can_infer, "quantity"].abs())

    e["price"] = e["price"].fillna(0.0)

    out = []
    for ticker, g in e.groupby("ticker", sort=False):
        shares = 0.0
        basis = 0.0

        realized_pl = 0.0
        realized_cost = 0.0
        realized_proceeds = 0.0

        for _, r in g.iterrows():
            qty_raw = float(r["quantity"] or 0.0)
            qty = abs(qty_raw)
            px = float(r["price"] or 0.0)
            fee = float(r["fees"] or 0.0)

            if qty <= 0:
                continue

            if r["event_type"] == "BUY_SHARES":
                shares += qty
                basis += (qty * px) + fee

            elif r["event_type"] == "SELL_SHARES":
                if shares <= 0:
                    shares = 0.0
                    basis = 0.0
                    continue

                sell_qty = min(qty, shares)
                avg_cost = (basis / shares) if shares > 0 else 0.0
                cost_removed = sell_qty * avg_cost

                if pd.notna(r["cash_delta"]):
                    proceeds_net = float(r["cash_delta"])
                else:
                    proceeds_net = (sell_qty * px) - fee

                realized_pl += (proceeds_net - cost_removed)
                realized_cost += cost_removed
                realized_proceeds += proceeds_net

                shares -= sell_qty
                basis -= cost_removed

                if shares < 1e-9:
                    shares = 0.0
                    basis = 0.0

        avg_cost = (basis / shares) if shares > 0 else 0.0
        realized_pct = (realized_pl / realized_cost) if realized_cost > 0 else 0.0

        out.append(
            {
                "ticker": ticker,
                "shares": shares,
                "cost_basis": basis,
                "avg_cost": avg_cost,
                "realized_pl": realized_pl,
                "realized_cost": realized_cost,
                "realized_proceeds": realized_proceeds,
                "realized_pct": realized_pct,
            }
        )

    # Build final DF (guard: out can be empty → pandas makes df with zero columns)
    cols = [
        "ticker", "shares", "cost_basis", "avg_cost",
        "realized_pl", "realized_cost", "realized_proceeds", "realized_pct",
    ]

    if not out:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(out)

    # Guard: if something weird happens and cols are missing, return typed empty
    for c in cols:
        if c not in df.columns:
            return pd.DataFrame(columns=cols)

    df = df[
        (df["shares"].abs() > 1e-9) |
        (df["realized_proceeds"].abs() > 1e-9) |
        (df["realized_pl"].abs() > 1e-9)
    ].copy()

    return df


# ----------------------------------------------------------
# Options-fund analytics: Open equity holdings (assignments)
# - Uses BUY_SHARES / SELL_SHARES to detect current stock
# - Adds latest close + unrealized P&L
# ----------------------------------------------------------
def calc_open_equity_with_unrealized(events: pd.DataFrame) -> pd.DataFrame:
    """
    Returns open equity positions with latest close + unrealized P&L.
    Intended for options-income funds where assignments create stock.

    Output columns:
      ticker, name, shares, avg_cost, cost_basis, last_close, mkt_value, unrealized_pl, unrealized_pct
    """
    if events is None or events.empty:
        return pd.DataFrame(
            columns=["ticker","name","shares","avg_cost","cost_basis","last_close","mkt_value","unrealized_pl","unrealized_pct"]
        )

    # Reuse your canonical avg-cost logic
    pnl = calc_pnl_avg_cost(events)
    if pnl is None or pnl.empty:
        return pd.DataFrame(
            columns=["ticker","name","shares","avg_cost","cost_basis","last_close","mkt_value","unrealized_pl","unrealized_pct"]
        )

    # Keep only OPEN positions
    pnl = pnl.copy()
    pnl["shares"] = pd.to_numeric(pnl["shares"], errors="coerce").fillna(0.0)
    pnl = pnl[pnl["shares"].abs() > 1e-9].copy()
    if pnl.empty:
        return pd.DataFrame(
            columns=["ticker","name","shares","avg_cost","cost_basis","last_close","mkt_value","unrealized_pl","unrealized_pct"]
        )

    tickers = pnl["ticker"].astype(str).str.upper().tolist()

    # Latest prices + names
    px = load_latest_prices_and_names(tickers)
    if px is None or px.empty:
        pnl["name"] = ""
        pnl["last_close"] = 0.0
    else:
        px = px.copy()
        px["ticker"] = px["ticker"].astype(str).str.upper()
        px["last_close"] = pd.to_numeric(px["last_close"], errors="coerce").fillna(0.0)
        px["name"] = px["name"].fillna("").astype(str)

        pnl = pnl.merge(px[["ticker","name","last_close"]], on="ticker", how="left")
        pnl["name"] = pnl["name"].fillna("").astype(str)
        pnl["last_close"] = pd.to_numeric(pnl["last_close"], errors="coerce").fillna(0.0)

    pnl["avg_cost"] = pd.to_numeric(pnl["avg_cost"], errors="coerce").fillna(0.0)
    pnl["cost_basis"] = pd.to_numeric(pnl["cost_basis"], errors="coerce").fillna(0.0)

    pnl["mkt_value"] = pnl["shares"] * pnl["last_close"]
    pnl["unrealized_pl"] = pnl["mkt_value"] - pnl["cost_basis"]
    pnl["unrealized_pct"] = pnl.apply(
        lambda r: (r["unrealized_pl"] / r["cost_basis"]) if r["cost_basis"] else None,
        axis=1
    )

    out = pnl[[
        "ticker","name","shares","avg_cost","cost_basis",
        "last_close","mkt_value","unrealized_pl","unrealized_pct"
    ]].copy()

    return out.sort_values("unrealized_pl").reset_index(drop=True)

# ----------------------------------------------------------
# Header card (stock-only)
# ----------------------------------------------------------
def render_header_stockfund(
    portfolio_code: str,
    portfolio_name: str,
    start_date: date,
    starting_cash: float,
    latest_nav_row: dict | None,
    events_count: int,
    fees_total: float,
    deposits_net: float,
    trade_cashflow: float,
):
    nav_date = latest_nav_row.get("nav_date") if latest_nav_row else None
    nav_val = float(latest_nav_row.get("nav")) if latest_nav_row and latest_nav_row.get("nav") is not None else None
    nav_cash = float(latest_nav_row.get("cash")) if latest_nav_row and latest_nav_row.get("cash") is not None else None
    nav_eq = float(latest_nav_row.get("equity_value")) if latest_nav_row and latest_nav_row.get("equity_value") is not None else None

    nav_gain = (nav_val - float(starting_cash)) if (nav_val is not None and starting_cash is not None) else None
    nav_gain_pct = (nav_gain / float(starting_cash)) if (nav_gain is not None and starting_cash and starting_cash > 0) else None

    net_invested = float(starting_cash or 0.0) + float(deposits_net or 0.0)
    gain_vs_invested = (nav_val - net_invested) if (nav_val is not None) else None
    gain_vs_invested_pct = (gain_vs_invested / net_invested) if (gain_vs_invested is not None and net_invested > 0) else None

    with st.container(border=True):
        top_l, top_r = st.columns([1.6, 1.0], vertical_alignment="top")

        with top_l:
            st.markdown(
                f"### {portfolio_code} — {portfolio_name}\n"
                f"Start **{start_date}** · Starting cash **{fmt_money0(starting_cash)}** · Events **{events_count}**\n\n"
                f"Latest NAV date: **{nav_date if nav_date else '—'}**"
            )

        with top_r:
            st.caption("NAV = cash + equity (EOD)\n\nTrades = BUY_SHARES / SELL_SHARES")

        r1 = st.columns(4)
        with r1[0]:
            st.metric(
                "Latest NAV",
                fmt_money(nav_val) if nav_val is not None else "—",
                f"{fmt_money(nav_gain)} · {fmt_pct_ratio(nav_gain_pct)}" if nav_gain is not None else None,
            )
        with r1[1]:
            st.metric("NAV Cash", fmt_money(nav_cash) if nav_cash is not None else "—")
        with r1[2]:
            st.metric("NAV Equity", fmt_money(nav_eq) if nav_eq is not None else "—")
        with r1[3]:
            st.metric("Net deposits", fmt_money(deposits_net))

        r2 = st.columns(3)
        with r2[0]:
            st.metric("Fees", fmt_money(fees_total))
        with r2[1]:
            st.metric("Trade cashflow", fmt_money(trade_cashflow))
        with r2[2]:
            st.metric(
                "Gain vs invested",
                fmt_money(gain_vs_invested) if gain_vs_invested is not None else "—",
                fmt_pct_ratio(gain_vs_invested_pct) if gain_vs_invested_pct is not None else None,
            )

        st.caption("Source of truth: portfolios + portfolio_events + portfolio_nav_daily")

# ----------------------------------------------------------
# Time-Weighted Return (TWR) helpers
# - Use for funds with external cash flows (e.g., private fund)
# ----------------------------------------------------------
EXTERNAL_CASHFLOW_TYPES_DEFAULT = {
    "DEPOSIT", "CASH_DEPOSIT", "CONTRIBUTION",
    "WITHDRAW", "WITHDRAWAL", "CASH_WITHDRAWAL", "DISTRIBUTION",
}

def _calc_twr_series(
    nav_all: pd.DataFrame,
    events_all: pd.DataFrame,
    external_types: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build a daily Time-Weighted Return (TWR) series from EOD NAV and external cash flows.

    Daily return convention:
      r(D) = (NAV(D) - CF(D)) / NAV(D-1) - 1

    CF(D) is the net external flow on day D (deposits positive, withdrawals negative).
    """
    external_types = external_types or EXTERNAL_CASHFLOW_TYPES_DEFAULT

    if nav_all is None or nav_all.empty:
        return pd.DataFrame(columns=["nav_date", "nav", "cf", "r", "twr_factor"])

    nav = nav_all.copy()
    nav["nav_date"] = pd.to_datetime(nav["nav_date"], errors="coerce").dt.date
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["nav_date", "nav"]).sort_values("nav_date").copy()

    if nav.empty:
        return pd.DataFrame(columns=["nav_date", "nav", "cf", "r", "twr_factor"])

    # External cash flows by date
    cf_by_day = pd.DataFrame(columns=["nav_date", "cf"])
    if events_all is not None and not events_all.empty:
        e = events_all.copy()
        e["event_time"] = pd.to_datetime(e["event_time"], errors="coerce")
        e = e.dropna(subset=["event_time"]).copy()
        e["event_type"] = e["event_type"].astype(str).str.upper()
        e["cash_delta"] = pd.to_numeric(e["cash_delta"], errors="coerce").fillna(0.0)
        e["event_date"] = e["event_time"].dt.date

        ext = e[e["event_type"].isin(set(external_types))].copy()
        if not ext.empty:
            cf_by_day = (
                ext.groupby("event_date", as_index=False)["cash_delta"]
                .sum()
                .rename(columns={"event_date": "nav_date", "cash_delta": "cf"})
            )

    out = nav.merge(cf_by_day, on="nav_date", how="left")
    out["cf"] = pd.to_numeric(out["cf"], errors="coerce").fillna(0.0)

    # Prior-day NAV
    out["nav_prev"] = out["nav"].shift(1)

    # Daily return
    out["r"] = 0.0
    valid = out["nav_prev"].notna() & (out["nav_prev"] != 0)
    out.loc[valid, "r"] = (out.loc[valid, "nav"] - out.loc[valid, "cf"]) / out.loc[valid, "nav_prev"] - 1.0

    # Cumulative TWR factor
    out["twr_factor"] = (1.0 + out["r"]).cumprod()

    return out[["nav_date", "nav", "cf", "r", "twr_factor"]].copy()


def _twr_perf(
    nav_all: pd.DataFrame,
    events_all: pd.DataFrame,
    start_dt: date,
    end_dt: date | None = None,
    external_types: set[str] | None = None,
) -> dict:
    """
    Windowed performance using Time-Weighted Return (TWR).
    Returns a dict similar to _nav_perf but uses TWR instead of naive NAV change.
    """
    twr = _calc_twr_series(nav_all, events_all, external_types=external_types)
    if twr.empty:
        return {"ok": False}

    df = twr.copy().sort_values("nav_date")

    if end_dt is not None:
        df = df[df["nav_date"] <= end_dt]
    if df.empty:
        return {"ok": False}

    df_window = df[df["nav_date"] >= start_dt]
    if df_window.empty:
        return {"ok": False}

    start_day = df_window.iloc[0]["nav_date"]
    prev_rows = df[df["nav_date"] < start_day]

    prev_factor = 1.0
    if not prev_rows.empty:
        prev_factor = float(prev_rows.iloc[-1]["twr_factor"] or 1.0)

    end_factor = float(df_window.iloc[-1]["twr_factor"] or 1.0)
    window_twr = (end_factor / prev_factor) - 1.0 if prev_factor != 0 else None

    start_nav = float(df_window.iloc[0]["nav"])
    end_nav = float(df_window.iloc[-1]["nav"])
    # Net external cash during the window
    cash_flow_net = 0.0
    if events_all is not None and not events_all.empty:
        e = events_all.copy()
        e["event_time"] = pd.to_datetime(e["event_time"], errors="coerce")
        e = e.dropna(subset=["event_time"])
        e["event_type"] = e["event_type"].astype(str).str.upper()
        e["cash_delta"] = pd.to_numeric(e["cash_delta"], errors="coerce").fillna(0.0)
        e["event_date"] = e["event_time"].dt.date

        mask = (
            (e["event_type"].isin(EXTERNAL_CASHFLOW_TYPES_DEFAULT)) &
            (e["event_date"] >= start_dt) &
            ((end_dt is None) | (e["event_date"] <= end_dt))
        )

        cash_flow_net = float(e.loc[mask, "cash_delta"].sum())

    return {
        "ok": True,
        "start_date": df_window.iloc[0]["nav_date"],
        "end_date": df_window.iloc[-1]["nav_date"],
        "start_nav": start_nav,
        "end_nav": end_nav,
        "cash_flow_net": cash_flow_net,
        "chg": (end_nav - start_nav - cash_flow_net),  # ← strategy P&L
        "pct": window_twr,
    }


# ----------------------------------------------------------
# Year summary blocks (YTD + Inception + explicit years)
# ----------------------------------------------------------
def _nav_perf(nav_all: pd.DataFrame, start_dt: date, end_dt: date | None = None) -> dict:
    """
    Compute performance using NAV rows:
      - start = first nav_date >= start_dt
      - end   = last nav_date <= end_dt (or last row overall)
    """
    if nav_all is None or nav_all.empty:
        return {"ok": False}

    df = nav_all.copy()
    df["nav_date"] = pd.to_datetime(df["nav_date"], errors="coerce").dt.date
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")

    if df.empty:
        return {"ok": False}

    if end_dt is None:
        df_end = df
    else:
        df_end = df[df["nav_date"] <= end_dt]

    df_start = df[df["nav_date"] >= start_dt]
    if df_start.empty or df_end.empty:
        return {"ok": False}

    start_row = df_start.iloc[0]
    end_row = df_end.iloc[-1]

    start_nav = float(start_row["nav"])
    end_nav = float(end_row["nav"])
    chg = end_nav - start_nav
    pct = (chg / start_nav) if start_nav != 0 else None

    return {
        "ok": True,
        "start_date": start_row["nav_date"],
        "end_date": end_row["nav_date"],
        "start_nav": start_nav,
        "end_nav": end_nav,
        "chg": chg,
        "pct": pct,
    }


def render_year_summary_blocks(
    nav_all: pd.DataFrame,
    portfolio_start_date: date,
    years: list[int] | None = None,
    events_all: pd.DataFrame | None = None,
    use_twr: bool = False,
):
    """
    Renders:
      - YTD (based on today's year)
      - Inception (from portfolio_start_date)
      - Explicit year blocks (e.g., 2026), in the order provided
    """
    years = years or []

    today = date.today()
    ytd_start = date(today.year, 1, 1)

    if use_twr and events_all is not None:
        ytd = _twr_perf(nav_all, events_all, ytd_start, None)
        inc = _twr_perf(nav_all, events_all, portfolio_start_date, None)
    else:
        ytd = _nav_perf(nav_all, ytd_start, None)
        inc = _nav_perf(nav_all, portfolio_start_date, None)

    with st.container(border=True):
        st.markdown("### 📊 Performance summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                f"{today.year} YTD",
                fmt_money(ytd["end_nav"]) if ytd.get("ok") else "—",
                f"{fmt_money(ytd['chg'])} · {fmt_pct_ratio(ytd['pct'])}" if ytd.get("ok") else None,
            )
        with c2:
            st.metric(
                "Inception",
                fmt_money(inc["end_nav"]) if inc.get("ok") else "—",
                f"{fmt_money(inc['chg'])} · {fmt_pct_ratio(inc['pct'])}" if inc.get("ok") else None,
            )
        with c3:
            st.caption("YTD window")
            st.write(
                f"{ytd.get('start_date', '—')} → {ytd.get('end_date', '—')}"
                if ytd.get("ok") else "—"
            )
        with c4:
            st.caption("Inception window")
            st.write(
                f"{inc.get('start_date', '—')} → {inc.get('end_date', '—')}"
                if inc.get("ok") else "—"
            )

    for yr in years:
        yr_start = date(int(yr), 1, 1)
        yr_end = date(int(yr), 12, 31)

        if use_twr and events_all is not None:
            perf = _twr_perf(nav_all, events_all, yr_start, yr_end)
        else:
            perf = _nav_perf(nav_all, yr_start, yr_end)

        with st.container(border=True):
            st.markdown(f"### 🗓️ {yr} performance")
            a, b, c, d = st.columns(4)
            with a:
                st.metric("Start NAV", fmt_money(perf["start_nav"]) if perf.get("ok") else "—")
            with b:
                st.metric("End NAV", fmt_money(perf["end_nav"]) if perf.get("ok") else "—")
            with c:
                st.metric("Strategy P&L", fmt_money(perf["chg"]) if perf.get("ok") else "—")
            with d:
                st.metric("Return", fmt_pct_ratio(perf["pct"]) if perf.get("ok") else "—")

            st.caption(
                f"Window: {perf.get('start_date','—')} → {perf.get('end_date','—')}"
                if perf.get("ok") else
                "No NAV rows available for that year yet. (Run nav.py after you have prices + events.)"
            )

# ----------------------------------------------------------
# Options-fund helpers (YRI)
# ----------------------------------------------------------
def calc_collateral_from_events(events: pd.DataFrame) -> tuple[float, float]:
    """
    Collateral proxy:
      - CSP collateral = SUM(strike * contracts * 100) for SELL_CSP
      - CC notional    = SUM(strike * contracts * 100) for SELL_CC
    """
    if events is None or events.empty:
        return 0.0, 0.0

    e = events.copy()
    e["quantity"] = pd.to_numeric(e["quantity"], errors="coerce").fillna(0.0)
    e["strike"] = pd.to_numeric(e["strike"], errors="coerce").fillna(0.0)

    csp_mask = e["event_type"].astype(str).str.upper() == "SELL_CSP"
    cc_mask = e["event_type"].astype(str).str.upper() == "SELL_CC"

    csp = float((e.loc[csp_mask, "quantity"] * e.loc[csp_mask, "strike"]).sum() * 100.0)
    cc = float((e.loc[cc_mask, "quantity"] * e.loc[cc_mask, "strike"]).sum() * 100.0)

    return csp, cc

def render_header_optionsfund(
    portfolio_code: str,
    portfolio_name: str,
    start_date: date,
    starting_cash: float,
    latest_nav_row: dict | None,
    credits_gross: float,
    fees_total: float,
    events_count: int,
    csp_collateral: float,
    cc_collateral: float,
):
    nav_date = latest_nav_row.get("nav_date") if latest_nav_row else None
    nav_val = float(latest_nav_row.get("nav")) if latest_nav_row and latest_nav_row.get("nav") is not None else None
    nav_cash = float(latest_nav_row.get("cash")) if latest_nav_row and latest_nav_row.get("cash") is not None else None
    nav_eq = float(latest_nav_row.get("equity_value")) if latest_nav_row and latest_nav_row.get("equity_value") is not None else None

    nav_gain = (nav_val - float(starting_cash)) if (nav_val is not None and starting_cash is not None) else None
    nav_gain_pct = (nav_gain / float(starting_cash)) if (nav_gain is not None and starting_cash and starting_cash > 0) else None

    credits_gross = float(credits_gross or 0.0)
    fees_total = float(fees_total or 0.0)
    credits_net = credits_gross - fees_total
    credits_yield_vs_start = (credits_gross / float(starting_cash)) if starting_cash and starting_cash > 0 else None

    total_collateral = float(csp_collateral or 0.0) + float(cc_collateral or 0.0)
    blended_yield = (credits_net / total_collateral) if total_collateral > 0 else None
    util = (total_collateral / float(starting_cash)) if starting_cash and starting_cash > 0 else None

    with st.container(border=True):
        top_l, top_r = st.columns([1.6, 1.0], vertical_alignment="top")

        with top_l:
            st.markdown(
                f"### {portfolio_code} — {portfolio_name}\n"
                f"Start **{start_date}** · Starting cash **{fmt_money0(starting_cash)}** · Events **{events_count}**\n\n"
                f"Latest NAV date: **{nav_date if nav_date else '—'}**"
            )

        with top_r:
            st.caption("NAV = cash + equity (EOD)\n\nCredits = Σ cash_delta for SELL_CSP/SELL_CC")

        r1 = st.columns(4)
        with r1[0]:
            st.metric(
                "Latest NAV",
                fmt_money(nav_val) if nav_val is not None else "—",
                f"{fmt_money(nav_gain)} · {fmt_pct_ratio(nav_gain_pct)}" if nav_gain is not None else None,
            )
        with r1[1]:
            st.metric("NAV Cash", fmt_money(nav_cash) if nav_cash is not None else "—")
        with r1[2]:
            st.metric("NAV Equity", fmt_money(nav_eq) if nav_eq is not None else "—")
        with r1[3]:
            st.metric(
                "Credits (gross)",
                fmt_money(credits_gross),
                f"{fmt_pct_ratio(credits_yield_vs_start)} of start cash" if credits_yield_vs_start is not None else None,
            )

        r2 = st.columns(3)
        with r2[0]:
            st.metric("Fees", fmt_money(fees_total))
        with r2[1]:
            st.metric("Credits net", fmt_money(credits_net))
        with r2[2]:
            st.metric(
                "Blended yield · Util",
                f"{fmt_pct_ratio(blended_yield)} · {fmt_pct_ratio(util)}" if blended_yield is not None else "—",
                f"Collateral {fmt_money0(total_collateral)}" if total_collateral > 0 else None,
            )

        st.caption("Source of truth: portfolios + portfolio_events + portfolio_nav_daily")
