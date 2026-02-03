# 11_YRSI-26.py
import streamlit as st
import pandas as pd
from datetime import date

from portfolio_common import (
    load_portfolio_by_code,
    load_nav_series,
    load_nav_series_between,
    load_events_optionsfund,
    load_totals_by_type,
    load_totals_by_ticker,
    calc_collateral_from_events,
    render_header_optionsfund,
    render_year_summary_blocks,
    fmt_money,
    fmt_money0,
    fmt_pct_ratio,
    calc_open_equity_with_unrealized,
)

# st.set_page_config(page_title="YRSI Results", layout="wide")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("⭐📣 YRSI Results (You Rock Star Income Fund est. 2026)")

    st.markdown(
        """
        Read-only community page for YRSI.  
        Uses **only**: portfolios, portfolio_events, portfolio_nav_daily.
        """
    )

    default_code = "YRSI-26"

    # Prefer DB start_date if present; otherwise default to 2026-01-01
    p0 = load_portfolio_by_code(default_code)
    db_start = None
    if not p0.empty and pd.notna(p0.iloc[0]["start_date"]):
        try:
            db_start = pd.to_datetime(p0.iloc[0]["start_date"]).date()
        except Exception:
            db_start = None

    default_start_date = db_start or date(2026, 1, 1)

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

    events = load_events_optionsfund(portfolio_id=portfolio_id, start_date=start_date)
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"])
        events["expiry"] = pd.to_datetime(events["expiry"], errors="coerce").dt.date

    if (nav_all.empty and nav.empty) and events.empty:
        st.info("No NAV or ledger events found for this date window yet.")
        return

    # Credits (premium inflows) = cash_delta on SELL_CSP/SELL_CC
    credits_gross = 0.0
    fees_total = 0.0
    if not events.empty:
        fees_total = float(pd.to_numeric(events["fees"], errors="coerce").fillna(0.0).sum())
        mask_credits = events["event_type"].astype(str).str.upper().isin(["SELL_CSP", "SELL_CC"])
        credits_gross = float(
            pd.to_numeric(events.loc[mask_credits, "cash_delta"], errors="coerce")
            .fillna(0.0)
            .sum()
        )

    csp_collateral, cc_collateral = calc_collateral_from_events(events)

    render_header_optionsfund(
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

    # 2026 block right after header
    st.divider()
    render_year_summary_blocks(nav_all=nav_all, portfolio_start_date=portfolio_start, years=[2026])
    
    # Debug
    st.write("DEBUG holdings:")
    st.write("portfolio_id:", portfolio_id)
    st.write("events_all rows:", len(events_all))
    st.write("share-like rows:", len(events_all[events_all["event_type"].astype(str).str.upper().isin(list(SHARE_BUY_TYPES | SHARE_SELL_TYPES))]))
    st.write("DUOL rows:", len(events_all[events_all["ticker"].astype(str).str.upper() == "DUOL"]))

    # ----------------------------------------------------------
    # Open Holdings
    # ---------------------------------------------------------
    st.divider()
    st.subheader("📦 Open holdings (assignments)")

    open_eq = calc_open_equity_with_unrealized(events)

    if open_eq.empty:
        st.caption("No open share positions detected.")
    else:
        show = open_eq.copy()

        # Pretty formatting
        show["shares"] = show["shares"].astype(float)
        show["avg_cost"] = show["avg_cost"].apply(fmt_money)
        show["last_close"] = show["last_close"].apply(fmt_money)
        show["cost_basis"] = show["cost_basis"].apply(fmt_money)
        show["mkt_value"] = show["mkt_value"].apply(fmt_money)
        show["unrealized_pl"] = show["unrealized_pl"].apply(fmt_money)
        show["unrealized_pct"] = show["unrealized_pct"].apply(fmt_pct_ratio)

        st.dataframe(
            show[["ticker","name","shares","avg_cost","last_close","mkt_value","unrealized_pl","unrealized_pct"]],
            hide_index=True,
            use_container_width=True,
        )

        # Reconciliation: credits vs unrealized explains why return is smaller
        credits_net = float((credits_gross or 0.0) - (fees_total or 0.0))
        unreal_total = float(pd.to_numeric(open_eq["unrealized_pl"], errors="coerce").fillna(0.0).sum())

        st.caption("Reconciliation (why Credits ≠ Return)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Credits (net)", fmt_money(credits_net))
        with c2:
            st.metric("Unrealized equity P/L", fmt_money(unreal_total))
        with c3:
            st.metric("Credits + Unrealized", fmt_money(credits_net + unreal_total))

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
        credits = events[events["event_type"].astype(str).str.upper().isin(["SELL_CSP", "SELL_CC"])].copy()
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
        if not nav_all.empty:
            st.download_button(
                "Download NAV CSV",
                nav_all.to_csv(index=False).encode("utf-8"),
                file_name=f"{portfolio_code.lower()}_nav.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
