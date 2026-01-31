# admin/12_ROTH-26.py
# ----------------------------------------------------------
# Private Admin ROTH-26 page (Personal Roth IRA Fund)
# - Read-only results for ROTH-26
# - Admin-only via env var: YRC_ADMIN=1
# - Uses: portfolios, portfolio_events, portfolio_nav_daily
# ----------------------------------------------------------

import os
import streamlit as st
import pandas as pd
from datetime import date

from sqlalchemy import text
from db import get_engine

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
)

# ----------------------------------------------------------
# Admin gate
# ----------------------------------------------------------
IS_ADMIN = os.getenv("YRC_ADMIN", "0") == "1"
if not IS_ADMIN:
    st.error("This page is admin-only.")
    st.stop()


# ----------------------------------------------------------
# Helpers: holdings snapshot (positions from events + latest close)
# ----------------------------------------------------------
def load_holdings_snapshot(portfolio_id: int) -> pd.DataFrame:
    """
    Compute current holdings using all share-impact events up to now:
    BUY_SHARES, SELL_SHARES, ASSIGN_PUT, CALL_AWAY.

    Uses latest available close in prices table per ticker.

    Returns columns:
      ticker, shares, close_used, market_value
    """
    engine = get_engine()

    q = text("""
        WITH pos AS (
          SELECT
            UPPER(TRIM(ticker)) AS ticker,
            SUM(
              CASE
                WHEN event_type IN ('BUY_SHARES','SELL_SHARES','ASSIGN_PUT','CALL_AWAY')
                  THEN COALESCE(quantity,0)
                ELSE 0
              END
            ) AS shares
          FROM portfolio_events
          WHERE portfolio_id = :portfolio_id
            AND ticker IS NOT NULL AND ticker <> ''
          GROUP BY 1
          HAVING ABS(SUM(
              CASE
                WHEN event_type IN ('BUY_SHARES','SELL_SHARES','ASSIGN_PUT','CALL_AWAY')
                  THEN COALESCE(quantity,0)
                ELSE 0
              END
          )) > 1e-9
        ),
        px AS (
          SELECT
            p.ticker,
            p.shares,
            (
              SELECT pr.close
              FROM prices pr
              WHERE pr.ticker = p.ticker
              ORDER BY pr.date DESC
              LIMIT 1
            ) AS close_used
          FROM pos p
        )
        SELECT
          ticker,
          shares,
          COALESCE(close_used, 0) AS close_used,
          (shares * COALESCE(close_used, 0)) AS market_value
        FROM px
        ORDER BY market_value DESC;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"portfolio_id": int(portfolio_id)})

    return df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("🔒 ROTH-26 Results (Private – Roth IRA Fund)")

    st.markdown(
        """
        **Admin-only page** for the **ROTH-26** personal fund.  
        Strategy: **QQQ only** (covered calls + cash-secured puts), plus **monthly withdrawals** and an **annual Trad IRA → Roth IRA distribution**.  
        Uses **only**: portfolios, portfolio_events, portfolio_nav_daily.
        """
    )

    default_code = "ROTH-26"

    # ----------------------------------------------------------
    # Resolve default start date from DB
    # ----------------------------------------------------------
    p0 = load_portfolio_by_code(default_code)
    db_start = None
    if not p0.empty and pd.notna(p0.iloc[0]["start_date"]):
        try:
            db_start = pd.to_datetime(p0.iloc[0]["start_date"]).date()
        except Exception:
            db_start = None

    default_start_date = db_start or date(2026, 2, 1)

    # ----------------------------------------------------------
    # Sidebar controls
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        portfolio_code = st.text_input(
            "Portfolio code",
            value=default_code
        ).strip().upper()

        start_date = st.date_input(
            "Display start date",
            value=default_start_date
        )

        st.caption("Tip: NAV rows will appear once nav.py runs on/after the portfolio start date.")

    # ----------------------------------------------------------
    # Load portfolio
    # ----------------------------------------------------------
    p = load_portfolio_by_code(portfolio_code)
    if p.empty:
        st.error(
            f"No portfolio found with code '{portfolio_code}'. "
            "Create it in Admin Ledger first."
        )
        return

    portfolio_id = int(p.iloc[0]["portfolio_id"])
    portfolio_name = str(p.iloc[0]["name"])
    starting_cash = float(p.iloc[0]["starting_cash"])
    portfolio_start = (
        pd.to_datetime(p.iloc[0]["start_date"]).date()
        if pd.notna(p.iloc[0]["start_date"])
        else start_date
    )

    st.caption(
        f"Portfolio: **{portfolio_code} — {portfolio_name}** "
        f"(portfolio_id={portfolio_id}) · "
        f"Start date in DB: {p.iloc[0]['start_date']}"
    )

    # ----------------------------------------------------------
    # NAV + Events
    # ----------------------------------------------------------
    nav_all = load_nav_series_between(
        portfolio_id=portfolio_id,
        start_date=portfolio_start
    )
    if not nav_all.empty:
        nav_all["nav_date"] = pd.to_datetime(nav_all["nav_date"]).dt.date
    latest_nav_row = nav_all.iloc[-1].to_dict() if not nav_all.empty else None

    nav = load_nav_series(
        portfolio_id=portfolio_id,
        start_date=start_date
    )
    if not nav.empty:
        nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.date

    events = load_events_optionsfund(
        portfolio_id=portfolio_id,
        start_date=start_date
    )
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"])
        events["expiry"] = pd.to_datetime(
            events["expiry"],
            errors="coerce"
        ).dt.date

    if (nav_all.empty and nav.empty) and events.empty:
        st.info("No NAV or ledger events found for this date window yet.")
        return

    # ----------------------------------------------------------
    # Credits, fees, collateral
    # ----------------------------------------------------------
    credits_gross = 0.0
    fees_total = 0.0

    if not events.empty:
        fees_total = float(
            pd.to_numeric(events["fees"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        mask_credits = (
            events["event_type"]
            .astype(str)
            .str.upper()
            .isin(["SELL_CSP", "SELL_CC"])
        )
        credits_gross = float(
            pd.to_numeric(
                events.loc[mask_credits, "cash_delta"],
                errors="coerce"
            )
            .fillna(0.0)
            .sum()
        )

    csp_collateral, cc_collateral = calc_collateral_from_events(events)

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Current Holdings (recommended for Roth-26 sanity check)
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📦 Current holdings")

    try:
        holdings = load_holdings_snapshot(portfolio_id)
    except Exception as e:
        holdings = pd.DataFrame()
        st.warning(f"Holdings snapshot failed: {e}")

    if holdings.empty:
        st.info("No current share holdings detected from ledger events.")
    else:
        show = holdings.copy()
        show["close_used"] = show["close_used"].apply(fmt_money)
        show["market_value"] = show["market_value"].apply(fmt_money)
        st.dataframe(show, hide_index=True, use_container_width=True)

    # ----------------------------------------------------------
    # Year summary
    # ----------------------------------------------------------
    st.divider()
    events_all = load_events_optionsfund(portfolio_id=portfolio_id, start_date=portfolio_start)

    render_year_summary_blocks(
        nav_all=nav_all,
        portfolio_start_date=portfolio_start,
        years=[2026],
        events_all=events_all,
        use_twr=False,  # Roth tracking is more "broker-style"; can switch later if you want
    )

    # ----------------------------------------------------------
    # NAV chart
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📈 NAV over time")

    if nav.empty:
        st.info("No NAV rows found yet. (Run nav.py / nightly cron.)")
    else:
        nav_curve = nav.copy()
        nav_curve["nav"] = pd.to_numeric(
            nav_curve["nav"],
            errors="coerce"
        ).fillna(0.0)
        nav_curve = nav_curve.set_index(
            pd.to_datetime(nav_curve["nav_date"])
        )
        st.line_chart(nav_curve["nav"])

    # ----------------------------------------------------------
    # NAV vs credits
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📉 NAV vs cumulative credits")

    if nav.empty or events.empty:
        st.caption("Needs both NAV rows and ledger events.")
    else:
        credits = events[
            events["event_type"]
            .astype(str)
            .str.upper()
            .isin(["SELL_CSP", "SELL_CC"])
        ].copy()

        credits["event_date"] = credits["event_time"].dt.date
        credits["cash_delta"] = pd.to_numeric(
            credits["cash_delta"],
            errors="coerce"
        ).fillna(0.0)

        credits_daily = (
            credits
            .groupby("event_date", as_index=False)["cash_delta"]
            .sum()
            .sort_values("event_date")
        )
        credits_daily["cum_credits"] = credits_daily["cash_delta"].cumsum()

        nav_join = nav.copy()
        nav_join["nav_date"] = pd.to_datetime(nav_join["nav_date"]).dt.date
        nav_join["nav"] = pd.to_numeric(
            nav_join["nav"],
            errors="coerce"
        ).fillna(0.0)

        merged = (
            pd.merge(
                nav_join[["nav_date", "nav"]],
                credits_daily[["event_date", "cum_credits"]],
                left_on="nav_date",
                right_on="event_date",
                how="left",
            )
            .sort_values("nav_date")
        )

        merged["cum_credits"] = merged["cum_credits"].ffill().fillna(0.0)
        merged = merged.set_index(pd.to_datetime(merged["nav_date"]))

        st.line_chart(merged[["nav", "cum_credits"]])

    # ----------------------------------------------------------
    # Totals by event type
    # ----------------------------------------------------------
    st.divider()
    st.subheader("🧮 Totals by event type")

    by_type = load_totals_by_type(
        portfolio_id=portfolio_id,
        start_date=start_date
    )
    if by_type.empty:
        st.info("No events found for this window.")
    else:
        show = by_type.copy()
        show["total_fees"] = show["total_fees"].apply(fmt_money)
        show["total_cash_delta"] = show["total_cash_delta"].apply(fmt_money)
        st.dataframe(show, hide_index=True, use_container_width=True)

    # ----------------------------------------------------------
    # Totals by ticker
    # ----------------------------------------------------------
    st.divider()
    st.subheader("🏷️ Totals by ticker")

    by_ticker = load_totals_by_ticker(
        portfolio_id=portfolio_id,
        start_date=start_date
    )
    if by_ticker.empty:
        st.info("No ticker-tagged events found for this window.")
    else:
        show = by_ticker.copy()
        show["total_fees"] = show["total_fees"].apply(fmt_money)
        show["total_cash_delta"] = show["total_cash_delta"].apply(fmt_money)
        st.dataframe(show, hide_index=True, use_container_width=True)

    # ----------------------------------------------------------
    # Ledger table
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Downloads
    # ----------------------------------------------------------
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
