# Wk_IV_Targets.py

import os
import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import date, timedelta

# ----------------------------------------------------------
# Admin flag (set on Render)
# - public service:  YRC_ADMIN=0 (or unset)
# - admin service:   YRC_ADMIN=1
# ----------------------------------------------------------
IS_ADMIN = os.getenv("YRC_ADMIN", "0") == "1"

st.set_page_config(page_title="Weekly IV Targets", layout="wide")

# ----------------------------------------------------------
# Convert numeric star rating into a pretty "⭐⭐⭐" string
# ----------------------------------------------------------
def stars_display(x) -> str:
    try:
        if pd.isna(x):
            return ""
        n = int(x)
        if n <= 0:
            return ""
        return "⭐" * n
    except Exception:
        return ""

# ----------------------------------------------------------
# Load Weekly Targets
# - latest fetch_date per ticker
# - within that fetch_date, prefer nearest expiry (smallest dte then expiry)
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_weekly_targets(iv_min_atm: float, market_cap_min: float, min_star_rating: int) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
    WITH latest_price AS (
      SELECT
        p.ticker,
        p.close AS latest_price
      FROM prices p
      JOIN (
        SELECT ticker, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
      ) mp
        ON mp.ticker = p.ticker
       AND mp.max_date = p.date
    ),
    latest_shares AS (
      SELECT
        f.ticker,
        f.shares_outstanding
      FROM financials f
      JOIN (
        SELECT ticker, MAX(report_date) AS max_report
        FROM financials
        GROUP BY ticker
      ) mf
        ON mf.ticker = f.ticker
       AND mf.max_report = f.report_date
    ),
    mc AS (
      SELECT
        ls.ticker,
        (ls.shares_outstanding * lp.latest_price) AS market_cap
      FROM latest_shares ls
      JOIN latest_price lp ON ls.ticker = lp.ticker
    ),
    latest_fetch AS (
      SELECT
        ticker,
        MAX(fetch_date) AS max_fetch_date
      FROM iv_summary
      GROUP BY ticker
    ),
    recent_iv AS (
      SELECT DISTINCT ON (ivs.ticker)
        ivs.*
      FROM iv_summary ivs
      JOIN latest_fetch lf
        ON lf.ticker = ivs.ticker
       AND lf.max_fetch_date = ivs.fetch_date
      ORDER BY ivs.ticker, ivs.dte ASC NULLS LAST, ivs.expiry ASC
    )
    SELECT
      r.ticker,
      COALESCE(c.greer_star_rating, 0) AS greer_star_rating,
      mc.market_cap,
      lp.latest_price AS latest_price,

      r.fetch_date,
      r.expiry,
      r.dte,
      r.contract_count,
      r.iv_atm,
      r.iv_median,
      r.atm_premium,
      r.atm_premium_pct,
      r.underlying_price,

      r.put_20d_strike,
      r.put_20d_delta,
      r.put_20d_premium,
      r.put_20d_premium_pct,

      r.call_20d_strike,
      r.call_20d_delta,
      r.call_20d_premium,
      r.call_20d_premium_pct
    FROM recent_iv r
    JOIN mc ON r.ticker = mc.ticker
    JOIN latest_price lp ON r.ticker = lp.ticker
    LEFT JOIN companies c ON c.ticker = r.ticker
    WHERE
      mc.market_cap >= :market_cap_min
      AND r.iv_atm >= :iv_min_atm
      AND COALESCE(c.greer_star_rating, 0) >= :min_star_rating
    ORDER BY
      r.iv_atm DESC,
      mc.market_cap DESC
    """)
    df = pd.read_sql(
        query,
        engine,
        params={
            "market_cap_min": market_cap_min,
            "iv_min_atm": iv_min_atm,
            "min_star_rating": min_star_rating
        }
    )
    return df

# ----------------------------------------------------------
# Insert a trade log row for Income Fund tracking
# - Computes premium_total, notional, cash_secured, shares_covered
# ----------------------------------------------------------
def insert_income_trade(
    ticker: str,
    strategy: str,
    expiry: date,
    strike: float,
    option_type: str,
    contracts: int,
    fill_price: float,
    fees: float,
    notes: str
) -> None:
    engine = get_engine()

    # Normalize / validate
    t = (ticker or "").strip().upper()
    if not t:
        raise ValueError("Ticker is required.")

    strategy = (strategy or "").strip().upper()
    option_type = (option_type or "").strip().lower()

    if strategy not in ("CSP", "CC"):
        raise ValueError("Strategy must be CSP or CC.")
    if option_type not in ("put", "call"):
        raise ValueError("Option type must be put or call.")
    if strategy == "CSP" and option_type != "put":
        raise ValueError("CSP trades must have option_type='put'.")
    if strategy == "CC" and option_type != "call":
        raise ValueError("CC trades must have option_type='call'.")

    if contracts <= 0:
        raise ValueError("Contracts must be > 0.")
    if strike <= 0:
        raise ValueError("Strike must be > 0.")
    if fill_price < 0:
        raise ValueError("Fill price must be >= 0.")
    if fees < 0:
        raise ValueError("Fees must be >= 0.")

    premium_total = contracts * 100 * fill_price
    notional = strike * contracts * 100

    cash_secured = 0
    shares_covered = 0
    if strategy == "CSP":
        cash_secured = notional
    else:
        shares_covered = contracts * 100

    ins = text("""
        INSERT INTO income_fund_trades
        (
            ticker,
            strategy,
            expiry,
            strike,
            option_type,
            contracts,
            fill_price,
            fees,
            premium_total,
            notional,
            cash_secured,
            shares_covered,
            notes
        )
        VALUES
        (
            :ticker,
            :strategy,
            :expiry,
            :strike,
            :option_type,
            :contracts,
            :fill_price,
            :fees,
            :premium_total,
            :notional,
            :cash_secured,
            :shares_covered,
            :notes
        )
    """)

    with engine.begin() as conn:
        conn.execute(ins, {
            "ticker": t,
            "strategy": strategy,
            "expiry": expiry,
            "strike": strike,
            "option_type": option_type,
            "contracts": contracts,
            "fill_price": fill_price,
            "fees": fees,
            "premium_total": premium_total,
            "notional": notional,
            "cash_secured": cash_secured,
            "shares_covered": shares_covered,
            "notes": notes
        })

# ----------------------------------------------------------
# Load recent trades (admin only)
# - Uses premium_total if present
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_recent_trades(limit: int = 25) -> pd.DataFrame:
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
          notes
        FROM income_fund_trades
        ORDER BY created_at DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"limit": limit})
    return df

# ----------------------------------------------------------
# Weekly totals by expiry (admin only)
# - Includes secured cash + shares covered
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_weekly_totals(days_forward: int = 14) -> pd.DataFrame:
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
          SUM(COALESCE(shares_covered, 0)) AS total_shares_covered
        FROM income_fund_trades
        WHERE expiry >= CURRENT_DATE
          AND expiry <= (CURRENT_DATE + (:days_forward || ' days')::interval)
        GROUP BY expiry
        ORDER BY expiry ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"days_forward": days_forward})
    return df

def main():
    st.title("📋 Weekly IV Targets (20Δ Wheel Targets)")

    st.markdown(
        """
        **Weekend plan:** Use this page on Sat/Sun to prep your Monday orders for options expiring Friday.

        **Included:**  
        - **~20Δ Put** (Cash-Secured Put candidate)  
        - **~20Δ Call** (Covered Call candidate)  
        - Premium % filters to target **~1%/week**
        """
    )

    # ----------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------
    top1, top2, top3, top4 = st.columns([1.1, 1.2, 1.1, 1.2])
    with top1:
        iv_min_atm = st.number_input(
            "Minimum implied volatility (ATM) (decimal form)",
            value=0.40,
            step=0.05,
            format="%.2f"
        )
    with top2:
        market_cap_min = st.number_input(
            "Minimum market cap",
            value=10_000_000_000,
            step=1_000_000_000,
            format="%d"
        )
    with top3:
        min_star_rating = st.slider("Min ⭐ rating", 0, 5, 0)
    with top4:
        expiry_days = st.slider("Max expiry days from today", 1, 14, 7)

    colA, colB, colC = st.columns([1.35, 1.15, 1.2])
    with colA:
        wheel_mode = st.selectbox("Wheel Mode (what you're preparing for Monday)", ["CASH (Sell CSP)", "SHARES (Sell CC)"])
    with colB:
        min_target_premium_pct = st.number_input(
            "Min target premium % (0.01 = 1%)",
            min_value=0.0,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
    with colC:
        sort_by = st.selectbox("Sort by", ["Action premium %", "IV ATM", "Stars", "Market Cap", "Ticker"])

    df = load_weekly_targets(iv_min_atm=iv_min_atm, market_cap_min=market_cap_min, min_star_rating=min_star_rating)

    if df.empty:
        st.info("No candidates found under market cap / IV / star filters.")
        return

    # ----------------------------------------------------------
    # Add star display + clean types
    # ----------------------------------------------------------
    df["stars"] = df["greer_star_rating"].apply(stars_display)

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["fetch_date"] = pd.to_datetime(df["fetch_date"]).dt.date
    df["contract_count"] = pd.to_numeric(df["contract_count"], errors="coerce").astype("Int64")

    df["latest_price"] = pd.to_numeric(df["latest_price"], errors="coerce").round(2)
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce").round(2)
    df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce").round(3)
    df["iv_median"] = pd.to_numeric(df["iv_median"], errors="coerce").round(3)

    # Market cap formatting
    df["market_cap_raw"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["market_cap"] = df["market_cap_raw"].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")

    # Format helpers
    def fmt_money(x):
        return f"${float(x):.2f}" if pd.notnull(x) else ""

    def fmt_pct(x):
        return f"{float(x)*100:.2f}%" if pd.notnull(x) else ""

    df["put_20d_premium_fmt"] = df["put_20d_premium"].apply(fmt_money)
    df["put_20d_premium_pct_fmt"] = df["put_20d_premium_pct"].apply(fmt_pct)

    df["call_20d_premium_fmt"] = df["call_20d_premium"].apply(fmt_money)
    df["call_20d_premium_pct_fmt"] = df["call_20d_premium_pct"].apply(fmt_pct)

    # ----------------------------------------------------------
    # Filters
    # ----------------------------------------------------------
    df = df[df["contract_count"] >= 10]

    today = date.today()
    max_allowed = today + timedelta(days=expiry_days)
    df = df[df["expiry"] <= max_allowed]

    # ----------------------------------------------------------
    # Wheel mode action columns
    # ----------------------------------------------------------
    if wheel_mode.startswith("CASH"):
        df["action_strategy"] = "CSP"
        df["action_type"] = "put"
        df["action_strike"] = df["put_20d_strike"]
        df["action_delta"] = df["put_20d_delta"]
        df["action_premium"] = df["put_20d_premium"]
        df["action_premium_pct"] = df["put_20d_premium_pct"]
        df["action_premium_fmt"] = df["put_20d_premium_fmt"]
        df["action_premium_pct_fmt"] = df["put_20d_premium_pct_fmt"]
    else:
        df["action_strategy"] = "CC"
        df["action_type"] = "call"
        df["action_strike"] = df["call_20d_strike"]
        df["action_delta"] = df["call_20d_delta"]
        df["action_premium"] = df["call_20d_premium"]
        df["action_premium_pct"] = df["call_20d_premium_pct"]
        df["action_premium_fmt"] = df["call_20d_premium_fmt"]
        df["action_premium_pct_fmt"] = df["call_20d_premium_pct_fmt"]

    # Require action fields
    df = df[df["action_premium_pct"].notna() & df["action_strike"].notna() & df["action_premium"].notna()]

    # Premium target filter
    df = df[df["action_premium_pct"] >= float(min_target_premium_pct)]

    if df.empty:
        st.info("No tickers match all criteria after premium filter.")
        return

    # Optional ticker search filter
    search = st.text_input("Search ticker (optional):").upper().strip()
    if search:
        df = df[df["ticker"].str.contains(search)]

    if df.empty:
        st.info("No tickers match your search.")
        return

    # Sorting
    if sort_by == "Action premium %":
        df = df.sort_values(by="action_premium_pct", ascending=False, na_position="last")
    elif sort_by == "IV ATM":
        df = df.sort_values(by="iv_atm", ascending=False, na_position="last")
    elif sort_by == "Stars":
        df = df.sort_values(by="greer_star_rating", ascending=False, na_position="last")
    elif sort_by == "Market Cap":
        df = df.sort_values(by="market_cap_raw", ascending=False, na_position="last")
    else:
        df = df.sort_values(by="ticker", ascending=True)

    st.subheader(f"🧮 Found {len(df)} targets (Wheel Mode: {wheel_mode})")

    columns = [
        "ticker",
        "stars",
        "latest_price",
        "market_cap",
        "iv_atm",
        "iv_median",
        "expiry",
        "dte",
        "contract_count",
        "put_20d_strike",
        "put_20d_delta",
        "put_20d_premium_fmt",
        "put_20d_premium_pct_fmt",
        "call_20d_strike",
        "call_20d_delta",
        "call_20d_premium_fmt",
        "call_20d_premium_pct_fmt",
        "action_strategy",
        "action_strike",
        "action_delta",
        "action_premium_fmt",
        "action_premium_pct_fmt",
    ]

    st.dataframe(df[columns], hide_index=True, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="weekly_iv_targets_filtered.csv",
        mime="text/csv",
    )

    # ----------------------------------------------------------
    # Admin-only tracking / inserts
    # ----------------------------------------------------------
    if not IS_ADMIN:
        st.divider()
        st.info("Income Fund trade logging is admin-only.")
        return

    st.divider()
    st.subheader("✅ Income Fund Tracking — Log Executed Trade (ADMIN)")

    tickers = df["ticker"].dropna().unique().tolist()
    if not tickers:
        st.warning("No tickers available to log from this filtered list.")
        return

    selected_ticker = st.selectbox("Ticker to log", tickers)
    row = df[df["ticker"] == selected_ticker].head(1).iloc[0]

    default_expiry = row["expiry"]
    default_strike = float(row["action_strike"])
    default_type = str(row["action_type"])
    default_strategy = str(row["action_strategy"])
    default_fill = float(row["action_premium"]) if pd.notnull(row["action_premium"]) else 0.0

    x1, x2, x3, x4, x5 = st.columns([1.1, 1.0, 1.2, 1.0, 2.0])
    with x1:
        strategy = st.selectbox("Strategy", ["CSP", "CC"], index=0 if default_strategy == "CSP" else 1)
    with x2:
        contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
    with x3:
        fill_price = st.number_input("Fill Price (credit)", min_value=0.0, value=default_fill, step=0.01, format="%.2f")
    with x4:
        fees = st.number_input("Fees", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    with x5:
        notes = st.text_input(
            "Notes (optional)",
            value=f"Logged from Weekly IV Targets: {default_strategy} {default_type} exp {default_expiry} strike {default_strike}"
        )

    # enforce option type based on strategy
    option_type = "put" if strategy == "CSP" else "call"

    st.caption(
        f"Defaults: expiry={default_expiry} | type={option_type} | strike={default_strike} | "
        f"est_delta={row['action_delta']:.3f} | est_premium={row['action_premium_fmt']} ({row['action_premium_pct_fmt']})"
    )

    if st.button("📌 Log Trade to income_fund_trades"):
        try:
            insert_income_trade(
                ticker=selected_ticker,
                strategy=strategy,
                expiry=default_expiry,
                strike=float(default_strike),
                option_type=option_type,
                contracts=int(contracts),
                fill_price=float(fill_price),
                fees=float(fees),
                notes=notes
            )
            st.success("Trade logged! (income_fund_trades)")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed to insert trade. Error: {e}")

    st.divider()
    st.subheader("📊 Income Fund Tracking — This Week + Recent Trades (ADMIN)")

    wt1, wt2 = st.columns([1.2, 1.0])
    with wt1:
        days_forward = st.slider("Show totals for expiries within next N days", 7, 28, 14)
    with wt2:
        recent_limit = st.selectbox("Recent trades to show", [10, 25, 50, 100], index=1)

    weekly_totals = load_weekly_totals(days_forward=int(days_forward))
    if weekly_totals.empty:
        st.info("No logged trades found for upcoming expiries in this window.")
    else:
        weekly_totals["gross_credit"] = weekly_totals["gross_credit"].apply(lambda x: f"${float(x):,.2f}")
        weekly_totals["total_fees"] = weekly_totals["total_fees"].apply(lambda x: f"${float(x):,.2f}")
        weekly_totals["net_credit"] = weekly_totals["net_credit"].apply(lambda x: f"${float(x):,.2f}")
        weekly_totals["total_cash_secured"] = weekly_totals["total_cash_secured"].apply(lambda x: f"${float(x):,.0f}")
        weekly_totals["total_shares_covered"] = weekly_totals["total_shares_covered"].fillna(0).astype(int)
        st.write("**Totals by expiry**")
        st.dataframe(weekly_totals, hide_index=True, use_container_width=True)

    recent_trades = load_recent_trades(limit=int(recent_limit))
    if recent_trades.empty:
        st.info("No trades logged yet.")
    else:
        recent_trades["created_at"] = pd.to_datetime(recent_trades["created_at"])
        recent_trades["expiry"] = pd.to_datetime(recent_trades["expiry"]).dt.date
        recent_trades["fill_price"] = recent_trades["fill_price"].apply(lambda x: f"${float(x):.2f}")
        recent_trades["fees"] = recent_trades["fees"].apply(lambda x: f"${float(x):.2f}")
        recent_trades["gross_credit"] = recent_trades["gross_credit"].apply(lambda x: f"${float(x):,.2f}")
        recent_trades["net_credit"] = recent_trades["net_credit"].apply(lambda x: f"${float(x):,.2f}")
        recent_trades["cash_secured"] = recent_trades["cash_secured"].apply(lambda x: f"${float(x):,.0f}")
        recent_trades["shares_covered"] = recent_trades["shares_covered"].fillna(0).astype(int)

        st.write("**Recent trade logs**")
        st.dataframe(
            recent_trades[
                [
                    "created_at", "ticker", "strategy", "option_type", "expiry",
                    "strike", "contracts", "fill_price", "fees",
                    "gross_credit", "net_credit",
                    "cash_secured", "shares_covered",
                    "notes"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
