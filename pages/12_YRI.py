# 12_YRI.py
import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import date

st.set_page_config(page_title="YRI Results", layout="wide")

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
            q,
            conn,
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
            q,
            conn,
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
            q,
            conn,
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
            q,
            conn,
            params={"portfolio_id": int(portfolio_id), "start_date": start_date},
        )

# ----------------------------------------------------------
# Collateral proxy (for blended yield)
# CSP collateral  = SUM(strike * contracts * 100) for SELL_CSP
# CC notional     = SUM(strike * contracts * 100) for SELL_CC
# ----------------------------------------------------------
def calc_collateral_from_events(events: pd.DataFrame) -> tuple[float, float]:
    if events is None or events.empty:
        return 0.0, 0.0

    e = events.copy()
    e["quantity"] = pd.to_numeric(e["quantity"], errors="coerce").fillna(0.0)
    e["strike"] = pd.to_numeric(e["strike"], errors="coerce").fillna(0.0)

    csp_mask = e["event_type"] == "SELL_CSP"
    cc_mask = e["event_type"] == "SELL_CC"

    csp = float((e.loc[csp_mask, "quantity"] * e.loc[csp_mask, "strike"]).sum() * 100.0)
    cc = float((e.loc[cc_mask, "quantity"] * e.loc[cc_mask, "strike"]).sum() * 100.0)

    return csp, cc

# ----------------------------------------------------------
# Header card (NO raw HTML printing)
# Uses st.columns + st.metric so you never see "<div ...>"
# ----------------------------------------------------------
def render_header(
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

        # ---------- Metrics Row 1 ----------
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

        # ---------- Metrics Row 2 ----------
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

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("📣 YRI Results (Community)")

    st.markdown(
        """
        Read-only community page for YRI.  
        Uses **only**: portfolios, portfolio_events, portfolio_nav_daily.
        """
    )

    with st.sidebar:
        st.header("Controls")
        portfolio_code = st.text_input("Portfolio code", value="YRI").strip().upper()
        start_date = st.date_input("Display start date", value=date(2025, 12, 1))

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
        events["expiry"] = pd.to_datetime(events["expiry"]).dt.date

    if nav.empty and events.empty:
        st.info("No NAV or ledger events found for this date window yet.")
        return

    # Credits (premium inflows) = cash_delta on SELL_CSP/SELL_CC
    credits_gross = 0.0
    fees_total = 0.0
    if not events.empty:
        fees_total = float(pd.to_numeric(events["fees"], errors="coerce").fillna(0.0).sum())
        mask_credits = events["event_type"].isin(["SELL_CSP", "SELL_CC"])
        credits_gross = float(pd.to_numeric(events.loc[mask_credits, "cash_delta"], errors="coerce").fillna(0.0).sum())

    # Collateral proxy (for blended yield + utilization)
    csp_collateral, cc_collateral = calc_collateral_from_events(events)

    render_header(
        portfolio_code=portfolio_code,
        portfolio_name=portfolio_name,
        start_date=start_date,
        starting_cash=starting_cash,
        latest_nav_row=latest_nav_row,
        credits_gross=credits_gross,
        fees_total=fees_total,
        events_count=int(len(events)) if not events.empty else 0,
        csp_collateral=csp_collateral,
        cc_collateral=cc_collateral,
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

    st.subheader("📉 NAV vs cumulative credits")
    if nav.empty or events.empty:
        st.caption("Needs both NAV rows and ledger events.")
    else:
        credits = events[events["event_type"].isin(["SELL_CSP", "SELL_CC"])].copy()
        credits["event_date"] = credits["event_time"].dt.date
        credits["cash_delta"] = pd.to_numeric(credits["cash_delta"], errors="coerce").fillna(0.0)

        credits_daily = credits.groupby("event_date", as_index=False)["cash_delta"].sum().sort_values("event_date")
        credits_daily["cum_credits"] = credits_daily["cash_delta"].cumsum()

        nav_join = nav.copy()
        nav_join["nav_date"] = pd.to_datetime(nav_join["nav_date"]).dt.date
        nav_join["nav"] = pd.to_numeric(nav_join["nav"], errors="coerce").fillna(0.0)

        merged = pd.merge(
            nav_join[["nav_date", "nav"]],
            credits_daily[["event_date", "cum_credits"]],
            left_on="nav_date",
            right_on="event_date",
            how="left",
        ).sort_values("nav_date")

        merged["cum_credits"] = merged["cum_credits"].ffill().fillna(0.0)
        merged = merged.set_index(pd.to_datetime(merged["nav_date"]))

        st.line_chart(merged[["nav", "cum_credits"]])

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
        ev["strike"] = ev["strike"].apply(fmt_money)
        ev["cash_delta"] = ev["cash_delta"].apply(fmt_money)

        st.dataframe(
            ev[
                [
                    "event_time",
                    "event_type",
                    "ticker",
                    "quantity",
                    "price",
                    "option_type",
                    "strike",
                    "expiry",
                    "fees",
                    "cash_delta",
                    "notes",
                ]
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
