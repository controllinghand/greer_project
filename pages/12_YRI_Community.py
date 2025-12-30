# 12_YRI_Results.py
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
        return f"${float(x):,.2f}"
    except Exception:
        return ""

def fmt_money0(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return ""

def fmt_pct(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

# ----------------------------------------------------------
# Load trades (read-only)
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_trades(start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          trade_id,
          created_at,
          ticker,
          strategy,
          option_type,
          expiry,
          strike,
          contracts,
          fill_price,
          COALESCE(fees, 0) AS fees,

          COALESCE(premium_total, (contracts * 100 * fill_price)) AS gross_credit,
          COALESCE(premium_total, (contracts * 100 * fill_price)) - COALESCE(fees, 0) AS net_credit,

          COALESCE(cash_secured, 0) AS cash_secured,
          COALESCE(shares_covered, 0) AS shares_covered,
          COALESCE(notional, (strike * contracts * 100)) AS notional,

          notes
        FROM income_fund_trades
        WHERE created_at >= :start_date
        ORDER BY created_at ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start_date": start_date})
    return df

# ----------------------------------------------------------
# Aggregations
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_totals_by_expiry(start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          expiry,
          COUNT(*) AS trades,
          SUM(contracts) AS total_contracts,

          SUM(COALESCE(premium_total, (contracts * 100 * fill_price))) AS gross_credit,
          SUM(COALESCE(fees, 0)) AS total_fees,
          SUM(COALESCE(premium_total, (contracts * 100 * fill_price)) - COALESCE(fees, 0)) AS net_credit,

          SUM(COALESCE(cash_secured, 0)) AS total_cash_secured,
          SUM(COALESCE(shares_covered, 0)) AS total_shares_covered,

          SUM(
            CASE WHEN strategy = 'CC'
                 THEN COALESCE(notional, (strike * contracts * 100))
                 ELSE 0
            END
          ) AS total_cc_notional
        FROM income_fund_trades
        WHERE created_at >= :start_date
        GROUP BY expiry
        ORDER BY expiry ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start_date": start_date})
    return df

@st.cache_data(ttl=300)
def load_totals_by_ticker(start_date: date) -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT
          ticker,
          COUNT(*) AS trades,
          SUM(contracts) AS total_contracts,
          SUM(COALESCE(premium_total, (contracts * 100 * fill_price))) AS gross_credit,
          SUM(COALESCE(fees, 0)) AS total_fees,
          SUM(COALESCE(premium_total, (contracts * 100 * fill_price)) - COALESCE(fees, 0)) AS net_credit,
          SUM(COALESCE(cash_secured, 0)) AS cash_secured,
          SUM(CASE WHEN strategy = 'CC' THEN COALESCE(notional, (strike * contracts * 100)) ELSE 0 END) AS cc_notional
        FROM income_fund_trades
        WHERE created_at >= :start_date
        GROUP BY ticker
        ORDER BY net_credit DESC NULLS LAST
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start_date": start_date})
    return df

# ----------------------------------------------------------
# Header card
# ----------------------------------------------------------
def render_yri_header_card(
    start_date: date,
    starting_cash: float,
    trades_count: int,
    total_gross: float,
    total_fees: float,
    total_net: float,
    csp_collateral: float,
    cc_collateral: float,
):
    total_collateral = float(csp_collateral) + float(cc_collateral)
    util = (total_collateral / float(starting_cash)) if starting_cash and starting_cash > 0 else None
    blended_yield = (float(total_net) / total_collateral) if total_collateral > 0 else None

    st.markdown(
        f"""
        <div style="
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 16px 16px 8px 16px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            margin-bottom: 10px;">
          <div style="display:flex; align-items:center; justify-content:space-between; gap: 12px;">
            <div>
              <div style="font-size: 22px; font-weight: 800; margin-bottom: 2px;">YRI — You Rock Income Fund</div>
              <div style="font-size: 13px; color:#666;">
                Simulation start <b>{start_date}</b> · Starting cash <b>{fmt_money0(starting_cash)}</b> · Trades <b>{trades_count}</b>
              </div>
            </div>
            <div style="font-size: 12px; color:#777; text-align:right;">
              <div><b>Premiums</b> = net credit (after fees)</div>
              <div><b>Yield</b> = net credit / collateral proxy</div>
            </div>
          </div>

          <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 12px;">
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">Gross credit</div>
              <div style="font-size:18px; font-weight:700;">{fmt_money(total_gross)}</div>
            </div>
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">Fees</div>
              <div style="font-size:18px; font-weight:700;">{fmt_money(total_fees)}</div>
            </div>
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">Net credit</div>
              <div style="font-size:18px; font-weight:700;">{fmt_money(total_net)}</div>
            </div>
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">CSP collateral</div>
              <div style="font-size:18px; font-weight:700;">{fmt_money0(csp_collateral)}</div>
            </div>
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">CC notional proxy</div>
              <div style="font-size:18px; font-weight:700;">{fmt_money0(cc_collateral)}</div>
            </div>
            <div style="border:1px solid #f0f0f0; border-radius:12px; padding:10px;">
              <div style="font-size:12px; color:#777;">Blended yield · Utilization</div>
              <div style="font-size:18px; font-weight:700;">
                {fmt_pct(blended_yield)} · {fmt_pct(util)}
              </div>
            </div>
          </div>

          <div style="margin-top:10px; font-size: 12px; color:#666;">
            Collateral proxy = <b>cash-secured</b> (CSP) + <b>strike×100×contracts</b> (CC).
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("📣 YRI Results (You Rock Income Fund)")

    st.markdown(
        """
        This is the **community results** page for YRI.  
        It is **read-only** and summarizes the premiums collected from your logged options trades.
        """
    )

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("YRI Controls")
        start_date = st.date_input("Simulation / Fund start date", value=date(2025, 12, 1))
        starting_cash = st.number_input("Starting cash (for utilization)", value=100_000, step=5_000)
        st.caption("Starting cash is used for utilization %, not full portfolio accounting (yet).")

    trades = load_trades(start_date=start_date)
    if trades.empty:
        st.info("No YRI trades found for this start date yet.")
        return

    # Clean types
    trades["created_at"] = pd.to_datetime(trades["created_at"])
    trades["expiry"] = pd.to_datetime(trades["expiry"]).dt.date

    # ----------------------------------------------------------
    # Per-trade yield logic
    # - CSP: net_credit / cash_secured
    # - CC:  net_credit / notional (collateral proxy)
    # ----------------------------------------------------------
    trades["collateral"] = 0.0
    trades.loc[trades["strategy"] == "CSP", "collateral"] = pd.to_numeric(trades["cash_secured"], errors="coerce").fillna(0.0)
    trades.loc[trades["strategy"] == "CC", "collateral"] = pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0)

    trades["yield_pct"] = None
    mask = pd.to_numeric(trades["collateral"], errors="coerce").fillna(0.0) > 0
    trades.loc[mask, "yield_pct"] = (
        pd.to_numeric(trades.loc[mask, "net_credit"], errors="coerce").fillna(0.0)
        / pd.to_numeric(trades.loc[mask, "collateral"], errors="coerce").fillna(0.0)
    )

    # ----------------------------------------------------------
    # KPIs + Header Card
    # ----------------------------------------------------------
    total_gross = float(pd.to_numeric(trades["gross_credit"], errors="coerce").fillna(0.0).sum())
    total_fees = float(pd.to_numeric(trades["fees"], errors="coerce").fillna(0.0).sum())
    total_net = float(pd.to_numeric(trades["net_credit"], errors="coerce").fillna(0.0).sum())

    csp_collateral = float(pd.to_numeric(trades.loc[trades["strategy"] == "CSP", "cash_secured"], errors="coerce").fillna(0.0).sum())
    cc_collateral = float(pd.to_numeric(trades.loc[trades["strategy"] == "CC", "notional"], errors="coerce").fillna(0.0).sum())

    render_yri_header_card(
        start_date=start_date,
        starting_cash=float(starting_cash),
        trades_count=int(len(trades)),
        total_gross=total_gross,
        total_fees=total_fees,
        total_net=total_net,
        csp_collateral=csp_collateral,
        cc_collateral=cc_collateral,
    )

    st.divider()

    # ----------------------------------------------------------
    # Cumulative net credit chart
    # ----------------------------------------------------------
    st.subheader("📈 Cumulative net credit")
    curve = trades[["created_at", "net_credit"]].copy()
    curve["net_credit"] = pd.to_numeric(curve["net_credit"], errors="coerce").fillna(0.0)
    curve = curve.sort_values("created_at")
    curve["cum_net_credit"] = curve["net_credit"].cumsum()
    curve = curve.set_index("created_at")

    st.line_chart(curve["cum_net_credit"])

    st.divider()

    # ----------------------------------------------------------
    # Totals by expiry
    # ----------------------------------------------------------
    st.subheader("🗓️ Totals by expiry")
    by_exp = load_totals_by_expiry(start_date=start_date)

    by_exp["collateral"] = (
        pd.to_numeric(by_exp["total_cash_secured"], errors="coerce").fillna(0.0)
        + pd.to_numeric(by_exp["total_cc_notional"], errors="coerce").fillna(0.0)
    )
    by_exp["yield_pct"] = None
    mask2 = by_exp["collateral"] > 0
    by_exp.loc[mask2, "yield_pct"] = (
        pd.to_numeric(by_exp.loc[mask2, "net_credit"], errors="coerce").fillna(0.0)
        / pd.to_numeric(by_exp.loc[mask2, "collateral"], errors="coerce").fillna(0.0)
    )

    show_exp = by_exp.copy()
    show_exp["gross_credit"] = show_exp["gross_credit"].apply(fmt_money)
    show_exp["total_fees"] = show_exp["total_fees"].apply(fmt_money)
    show_exp["net_credit"] = show_exp["net_credit"].apply(fmt_money)
    show_exp["total_cash_secured"] = show_exp["total_cash_secured"].apply(fmt_money0)
    show_exp["total_cc_notional"] = show_exp["total_cc_notional"].apply(fmt_money0)
    show_exp["total_shares_covered"] = show_exp["total_shares_covered"].fillna(0).astype(int)
    show_exp["yield_pct"] = show_exp["yield_pct"].apply(fmt_pct)

    st.dataframe(
        show_exp[
            [
                "expiry",
                "trades",
                "total_contracts",
                "gross_credit",
                "total_fees",
                "net_credit",
                "total_cash_secured",
                "total_cc_notional",
                "total_shares_covered",
                "yield_pct",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------------------
    # Totals by ticker
    # ----------------------------------------------------------
    st.subheader("🏷️ Totals by ticker")
    by_ticker = load_totals_by_ticker(start_date=start_date)
    by_ticker["collateral"] = (
        pd.to_numeric(by_ticker["cash_secured"], errors="coerce").fillna(0.0)
        + pd.to_numeric(by_ticker["cc_notional"], errors="coerce").fillna(0.0)
    )
    by_ticker["yield_pct"] = None
    mask3 = by_ticker["collateral"] > 0
    by_ticker.loc[mask3, "yield_pct"] = (
        pd.to_numeric(by_ticker.loc[mask3, "net_credit"], errors="coerce").fillna(0.0)
        / pd.to_numeric(by_ticker.loc[mask3, "collateral"], errors="coerce").fillna(0.0)
    )

    show_t = by_ticker.copy()
    show_t["gross_credit"] = show_t["gross_credit"].apply(fmt_money)
    show_t["total_fees"] = show_t["total_fees"].apply(fmt_money)
    show_t["net_credit"] = show_t["net_credit"].apply(fmt_money)
    show_t["cash_secured"] = show_t["cash_secured"].apply(fmt_money0)
    show_t["cc_notional"] = show_t["cc_notional"].apply(fmt_money0)
    show_t["yield_pct"] = show_t["yield_pct"].apply(fmt_pct)

    st.dataframe(
        show_t[
            [
                "ticker",
                "trades",
                "total_contracts",
                "gross_credit",
                "total_fees",
                "net_credit",
                "cash_secured",
                "cc_notional",
                "yield_pct",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------------------
    # Recent trades (community view)
    # ----------------------------------------------------------
    st.subheader("🧾 Recent trade logs")
    trades_view = trades.copy()

    trades_view["fill_price"] = trades_view["fill_price"].apply(fmt_money)
    trades_view["fees"] = trades_view["fees"].apply(fmt_money)
    trades_view["gross_credit"] = trades_view["gross_credit"].apply(fmt_money)
    trades_view["net_credit"] = trades_view["net_credit"].apply(fmt_money)
    trades_view["cash_secured"] = trades_view["cash_secured"].apply(fmt_money0)
    trades_view["notional"] = trades_view["notional"].apply(fmt_money0)
    trades_view["yield_pct"] = trades_view["yield_pct"].apply(fmt_pct)

    st.dataframe(
        trades_view[
            [
                "created_at",
                "ticker",
                "strategy",
                "option_type",
                "expiry",
                "strike",
                "contracts",
                "fill_price",
                "fees",
                "gross_credit",
                "net_credit",
                "cash_secured",
                "notional",
                "yield_pct",
                "notes",
            ]
        ].sort_values("created_at", ascending=False),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download trades CSV",
        trades.to_csv(index=False).encode("utf-8"),
        file_name="yri_trades.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
