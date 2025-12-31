# 12_YR3G.py
import streamlit as st
import pandas as pd
from sqlalchemy import text, bindparam
from db import get_engine
from datetime import date

st.set_page_config(page_title="YR3G Results", layout="wide")

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

# ----------------------------------------------------------
# DB: Events
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_events(portfolio_id: int, start_date: date) -> pd.DataFrame:
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
    e = e[e["event_type"].isin(["BUY_SHARES", "SELL_SHARES"])].copy()

    e["event_time"] = pd.to_datetime(e["event_time"], errors="coerce")
    e = e.sort_values(["ticker", "event_time", "event_id"], ascending=True)

    e["quantity"] = pd.to_numeric(e["quantity"], errors="coerce").fillna(0.0)
    e["price"] = pd.to_numeric(e["price"], errors="coerce")
    e["fees"] = pd.to_numeric(e["fees"], errors="coerce").fillna(0.0)
    e["cash_delta"] = pd.to_numeric(e["cash_delta"], errors="coerce")

    # Infer price if missing: abs(cash_delta)/qty
    missing_price = e["price"].isna() | (e["price"] <= 0)
    can_infer = missing_price & e["cash_delta"].notna() & (e["quantity"] > 0)
    e.loc[can_infer, "price"] = (e.loc[can_infer, "cash_delta"].abs() / e.loc[can_infer, "quantity"])
    e["price"] = e["price"].fillna(0.0)

    out = []
    for ticker, g in e.groupby("ticker", sort=False):
        shares = 0.0
        basis = 0.0

        realized_pl = 0.0
        realized_cost = 0.0
        realized_proceeds = 0.0

        for _, r in g.iterrows():
            qty = float(r["quantity"] or 0.0)
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

                # Net proceeds: prefer cash_delta (already net), else compute
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

    df = pd.DataFrame(out)
    df = df[(abs(df["shares"]) > 1e-9) | (abs(df["realized_proceeds"]) > 1e-9) | (abs(df["realized_pl"]) > 1e-9)].copy()
    return df

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
# Main
# ----------------------------------------------------------
def main():
    st.title("🚀 YR3G Results (You Rock 3 Stars Growth Fund)")

    st.markdown(
        """
        Stock-only community page for **YR3G** (buys & sells of shares).  
        Uses **only**: portfolios, portfolio_events, portfolio_nav_daily.
        """
    )

    default_code = "YR3G"

    # Default start date = portfolio start_date if present
    p0 = load_portfolio_by_code(default_code)
    db_start = None
    if not p0.empty and pd.notna(p0.iloc[0]["start_date"]):
        try:
            db_start = pd.to_datetime(p0.iloc[0]["start_date"]).date()
        except Exception:
            db_start = None
    default_start_date = db_start or date(2025, 12, 1)

    with st.sidebar:
        st.header("Controls")
        portfolio_code = st.text_input("Portfolio code", value=default_code).strip().upper()
        start_date = st.date_input("Display start date", value=default_start_date)

    p = load_portfolio_by_code(portfolio_code)
    if p.empty:
        st.error(f"No portfolio found with code '{portfolio_code}'. Create it in Admin Ledger first.")
        return

    portfolio_id = int(p.iloc[0]["portfolio_id"])
    portfolio_name = str(p.iloc[0]["name"])
    starting_cash = float(p.iloc[0]["starting_cash"])
    portfolio_start = p.iloc[0]["start_date"]

    st.caption(
        f"Portfolio: **{portfolio_code} — {portfolio_name}** (portfolio_id={portfolio_id}) · "
        f"Start date in DB: {portfolio_start}"
    )

    nav = load_nav_series(portfolio_id=portfolio_id, start_date=start_date)
    if not nav.empty:
        nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.date
    latest_nav_row = nav.iloc[-1].to_dict() if not nav.empty else None

    events = load_events(portfolio_id=portfolio_id, start_date=start_date)
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"])

    if nav.empty and events.empty:
        st.info("No NAV or ledger events found for this date window yet.")
        return

    flows = calc_cashflows_stockfund(events)
    fees_total = float(flows["fees_total"])
    deposits_net = float(flows["deposits_net"])
    trade_cashflow = float(flows["trade_cashflow"])

    render_header_stockfund(
        portfolio_code=portfolio_code,
        portfolio_name=portfolio_name,
        start_date=start_date,
        starting_cash=starting_cash,
        latest_nav_row=latest_nav_row,
        events_count=int(len(events)) if not events.empty else 0,
        fees_total=fees_total,
        deposits_net=deposits_net,
        trade_cashflow=trade_cashflow,
    )

    st.divider()

    st.subheader("📈 NAV over time")
    if nav.empty:
        st.info("No NAV rows found yet. (Run nav.py / nightly cron.)")
    else:
        nav_curve = nav.copy()
        nav_curve["nav"] = pd.to_numeric(nav_curve["nav"], errors="coerce").fillna(0.0)
        nav_curve = nav_curve.set_index(pd.to_datetime(nav_curve["nav_date"]))
        st.line_chart(nav_curve["nav"])

    st.divider()

    st.subheader("📦 Holdings snapshot (Open positions)")
    pnl = calc_pnl_avg_cost(events)

    open_pos = pnl[pnl["shares"].abs() > 1e-9].copy()
    closed_pos = pnl[(pnl["shares"].abs() <= 1e-9) & (pnl["realized_proceeds"].abs() > 1e-9)].copy()

    if open_pos.empty:
        st.caption("No open positions right now.")
    else:
        tickers = open_pos["ticker"].astype(str).str.upper().tolist()
        px = load_latest_prices_and_names(tickers)

        snap = open_pos.merge(px, on="ticker", how="left")

        snap["last_close"] = pd.to_numeric(snap["last_close"], errors="coerce").fillna(0.0)
        snap["shares"] = pd.to_numeric(snap["shares"], errors="coerce").fillna(0.0)
        snap["cost_basis"] = pd.to_numeric(snap["cost_basis"], errors="coerce").fillna(0.0)
        snap["avg_cost"] = pd.to_numeric(snap["avg_cost"], errors="coerce").fillna(0.0)

        snap["market_value"] = snap["shares"] * snap["last_close"]
        snap["unrealized_pl"] = snap["market_value"] - snap["cost_basis"]
        snap["unrealized_pct"] = (snap["unrealized_pl"] / snap["cost_basis"]).where(snap["cost_basis"] > 0, 0.0)

        total_mv = float(snap["market_value"].sum()) if not snap.empty else 0.0
        snap["weight_pct"] = (snap["market_value"] / total_mv) if total_mv > 0 else 0.0

        snap["realized_pl"] = pd.to_numeric(snap["realized_pl"], errors="coerce").fillna(0.0)
        snap["realized_cost"] = pd.to_numeric(snap["realized_cost"], errors="coerce").fillna(0.0)
        snap["realized_pct"] = (snap["realized_pl"] / snap["realized_cost"]).where(snap["realized_cost"] > 0, 0.0)

        show = snap.copy()
        show["name"] = show["name"].fillna("")
        show["last_date"] = pd.to_datetime(show["last_date"], errors="coerce").dt.date

        show["avg_cost"] = show["avg_cost"].apply(fmt_money)
        show["last_close"] = show["last_close"].apply(fmt_money)
        show["market_value"] = show["market_value"].apply(fmt_money)
        show["cost_basis"] = show["cost_basis"].apply(fmt_money)
        show["unrealized_pl"] = show["unrealized_pl"].apply(fmt_money)
        show["unrealized_pct"] = show["unrealized_pct"].apply(fmt_pct_ratio)
        show["realized_pl"] = show["realized_pl"].apply(fmt_money)
        show["realized_pct"] = show["realized_pct"].apply(fmt_pct_ratio)
        show["weight_pct"] = show["weight_pct"].apply(fmt_pct_ratio)

        st.dataframe(
            show[
                [
                    "ticker",
                    "name",
                    "shares",
                    "avg_cost",
                    "last_close",
                    "last_date",
                    "market_value",
                    "cost_basis",
                    "unrealized_pl",
                    "unrealized_pct",
                    "realized_pl",
                    "realized_pct",
                    "weight_pct",
                ]
            ].sort_values("market_value", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.subheader("✅ Closed positions (Realized only)")
    if closed_pos.empty:
        st.caption("No closed positions with realized activity yet.")
    else:
        # Attach names for nicer output (no prices needed)
        tickers = closed_pos["ticker"].astype(str).str.upper().tolist()
        names = load_latest_prices_and_names(tickers)[["ticker", "name"]].drop_duplicates()

        closed = closed_pos.merge(names, on="ticker", how="left")
        closed["name"] = closed["name"].fillna("")

        closed["realized_pl"] = pd.to_numeric(closed["realized_pl"], errors="coerce").fillna(0.0)
        closed["realized_cost"] = pd.to_numeric(closed["realized_cost"], errors="coerce").fillna(0.0)
        closed["realized_proceeds"] = pd.to_numeric(closed["realized_proceeds"], errors="coerce").fillna(0.0)
        closed["realized_pct"] = (closed["realized_pl"] / closed["realized_cost"]).where(closed["realized_cost"] > 0, 0.0)

        show = closed.copy()
        show["realized_proceeds"] = show["realized_proceeds"].apply(fmt_money)
        show["realized_cost"] = show["realized_cost"].apply(fmt_money)
        show["realized_pl"] = show["realized_pl"].apply(fmt_money)
        show["realized_pct"] = show["realized_pct"].apply(fmt_pct_ratio)

        st.dataframe(
            show[
                ["ticker", "name", "realized_proceeds", "realized_cost", "realized_pl", "realized_pct"]
            ].sort_values("realized_pl", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    # Totals block (open + closed)
    total_realized = float(pnl["realized_pl"].sum()) if not pnl.empty else 0.0
    total_unrealized = 0.0
    holdings_mv = 0.0
    if not open_pos.empty:
        # recompute from open_pos + prices (cheap + safe)
        tickers = open_pos["ticker"].astype(str).str.upper().tolist()
        px = load_latest_prices_and_names(tickers)
        tmp = open_pos.merge(px, on="ticker", how="left")
        tmp["last_close"] = pd.to_numeric(tmp["last_close"], errors="coerce").fillna(0.0)
        tmp["market_value"] = pd.to_numeric(tmp["shares"], errors="coerce").fillna(0.0) * tmp["last_close"]
        tmp["cost_basis"] = pd.to_numeric(tmp["cost_basis"], errors="coerce").fillna(0.0)
        total_unrealized = float((tmp["market_value"] - tmp["cost_basis"]).sum())
        holdings_mv = float(tmp["market_value"].sum())

    total_pl = total_realized + total_unrealized

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Holdings MV", fmt_money(holdings_mv))
    with c2:
        st.metric("Total Unrealized", fmt_money(total_unrealized))
    with c3:
        st.metric("Total Realized", fmt_money(total_realized))
    with c4:
        st.metric("Total P/L", fmt_money(total_pl))

    st.divider()

    st.subheader("🧮 Totals by event type")
    by_type = load_totals_by_type(portfolio_id=portfolio_id, start_date=start_date)
    if by_type.empty:
        st.info("No events found for this window.")
    else:
        show = by_type.copy()
        show["total_fees"] = show["total_fees"].apply(fmt_money)
        show["total_cash_delta"] = show["total_cash_delta"].apply(fmt_money)
        st.dataframe(show, hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("🏷️ Totals by ticker")
    by_ticker = load_totals_by_ticker(portfolio_id=portfolio_id, start_date=start_date)
    if by_ticker.empty:
        st.info("No ticker-tagged events found for this window.")
    else:
        show = by_ticker.copy()
        show["total_fees"] = show["total_fees"].apply(fmt_money)
        show["total_cash_delta"] = show["total_cash_delta"].apply(fmt_money)
        st.dataframe(show, hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("🧾 Ledger events")
    if events.empty:
        st.info("No events logged yet.")
    else:
        ev = events.copy()
        ev["price"] = ev["price"].apply(fmt_money)
        ev["fees"] = ev["fees"].apply(fmt_money)
        ev["cash_delta"] = ev["cash_delta"].apply(fmt_money)

        st.dataframe(
            ev[
                ["event_time", "event_type", "ticker", "quantity", "price", "fees", "cash_delta", "notes"]
            ].sort_values("event_time", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if not events.empty:
            st.download_button(
                "Download events CSV",
                events.to_csv(index=False).encode("utf-8"),
                file_name=f"{portfolio_code.lower()}_events.csv",
                mime="text/csv",
            )
    with c2:
        if not nav.empty:
            st.download_button(
                "Download NAV CSV",
                nav.to_csv(index=False).encode("utf-8"),
                file_name=f"{portfolio_code.lower()}_nav.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
