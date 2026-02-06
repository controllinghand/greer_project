# 12_YR3G-26.py
import streamlit as st
import pandas as pd
from datetime import date

from portfolio_common import (
    load_portfolio_by_code,
    load_nav_series,
    load_nav_series_between,
    load_events_stockfund,
    load_totals_by_type,
    load_totals_by_ticker,
    load_latest_prices_and_names,
    calc_cashflows_stockfund,
    calc_pnl_avg_cost,
    render_header_stockfund,
    render_year_summary_blocks,
    fmt_money,
    fmt_pct_ratio,
)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("🚀 YR3G Results (You Rock 3 Stars Growth Fund est. 2026)")

    st.markdown(
        """
        Stock-only community page for **YR3G** (buys & sells of shares).  
        Uses **only**: portfolios, portfolio_events, portfolio_nav_daily.

        **Strategy:**  
        Holdings are systematically sourced from our **Opportunities_IV** screener and must 
        achieve a **3-Star rating**, representing the highest tier in our quality framework. 
        This structured approach filters the market for companies demonstrating exceptional 
        opportunity characteristics while reinforcing disciplined, data-driven portfolio construction.
        """
    )


    default_code = "YR3G-26"

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
    portfolio_start = pd.to_datetime(p.iloc[0]["start_date"]).date() if pd.notna(p.iloc[0]["start_date"]) else start_date

    st.caption(
        f"Portfolio: **{portfolio_code} — {portfolio_name}** (portfolio_id={portfolio_id}) · "
        f"Start date in DB: {p.iloc[0]['start_date']}"
    )

    nav_all = load_nav_series_between(portfolio_id=portfolio_id, start_date=portfolio_start)
    if not nav_all.empty:
        nav_all["nav_date"] = pd.to_datetime(nav_all["nav_date"]).dt.date
    latest_nav_row = nav_all.iloc[-1].to_dict() if not nav_all.empty else None

    nav = load_nav_series(portfolio_id=portfolio_id, start_date=start_date)
    if not nav.empty:
        nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.date

    events = load_events_stockfund(portfolio_id=portfolio_id, start_date=start_date)
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"])

    if (nav_all.empty and nav.empty) and events.empty:
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

    # 2026 block right after header
    st.divider()
    render_year_summary_blocks(nav_all=nav_all, portfolio_start_date=portfolio_start, years=[2026])

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
            show[["ticker", "name", "realized_proceeds", "realized_cost", "realized_pl", "realized_pct"]]
            .sort_values("realized_pl", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    total_realized = float(pnl["realized_pl"].sum()) if not pnl.empty else 0.0
    total_unrealized = 0.0
    holdings_mv = 0.0
    if not open_pos.empty:
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
            ev[["event_time", "event_type", "ticker", "quantity", "price", "fees", "cash_delta", "notes"]]
            .sort_values("event_time", ascending=False),
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
        if not nav_all.empty:
            st.download_button(
                "Download NAV CSV",
                nav_all.to_csv(index=False).encode("utf-8"),
                file_name=f"{portfolio_code.lower()}_nav.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
