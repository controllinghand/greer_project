# 13_Admin_Ledger.py
# ----------------------------------------------------------
# Private Admin Ledger page
# - Manual entry into existing tables:
#     portfolios(portfolio_id, code, name, start_date, starting_cash)
#     portfolio_events(event_id, portfolio_id, event_time, event_type, ticker,
#                      quantity, price, fees, option_type, strike, expiry, cash_delta, notes)
#     portfolio_nav_daily(...)  (not written here yet)
# - Admin-only via env var: YRC_ADMIN=1
# ----------------------------------------------------------

import os
import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import datetime, date as dt_date

# ----------------------------------------------------------
# Admin gate (set on Render + local)
# ----------------------------------------------------------
IS_ADMIN = os.getenv("YRC_ADMIN", "0") == "1"

st.set_page_config(page_title="Admin Ledger", layout="wide")

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

def safe_upper(s: str) -> str:
    return (s or "").strip().upper()

# ----------------------------------------------------------
# DB: Load portfolios
# ----------------------------------------------------------
@st.cache_data(ttl=60)
def load_portfolios() -> pd.DataFrame:
    engine = get_engine()
    q = text("""
        SELECT portfolio_id, code, name, start_date, starting_cash
        FROM portfolios
        ORDER BY code ASC
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn)

# ----------------------------------------------------------
# DB: Create/Update portfolio
# ----------------------------------------------------------
def upsert_portfolio(code: str, name: str, start_date: dt_date, starting_cash: float) -> int:
    engine = get_engine()
    q = text("""
        INSERT INTO portfolios (code, name, start_date, starting_cash)
        VALUES (:code, :name, :start_date, :starting_cash)
        ON CONFLICT (code) DO UPDATE
          SET name = EXCLUDED.name,
              start_date = EXCLUDED.start_date,
              starting_cash = EXCLUDED.starting_cash
        RETURNING portfolio_id
    """)
    with engine.begin() as conn:
        pid = conn.execute(q, {
            "code": safe_upper(code),
            "name": (name or "").strip(),
            "start_date": start_date,
            "starting_cash": float(starting_cash)
        }).scalar()
    st.cache_data.clear()
    return int(pid)

# ----------------------------------------------------------
# DB: Insert event into portfolio_events (matches your schema)
# ----------------------------------------------------------
def insert_event(payload: dict) -> None:
    engine = get_engine()
    ins = text("""
        INSERT INTO portfolio_events
        (
          portfolio_id,
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
        )
        VALUES
        (
          :portfolio_id,
          :event_time,
          :event_type,
          :ticker,
          :quantity,
          :price,
          :fees,
          :option_type,
          :strike,
          :expiry,
          :cash_delta,
          :notes
        )
    """)
    with engine.begin() as conn:
        conn.execute(ins, payload)

# ----------------------------------------------------------
# DB: Load recent events
# ----------------------------------------------------------
@st.cache_data(ttl=60)
def load_recent_events(portfolio_id: int, limit: int = 50) -> pd.DataFrame:
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
        ORDER BY event_time DESC, event_id DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"portfolio_id": portfolio_id, "limit": limit})

# ----------------------------------------------------------
# Simple event set (we can expand later)
# Note: quantity is flexible:
# - shares: +100, -100, etc
# - option contracts: 1, 7, etc (you decide convention)
# ----------------------------------------------------------
EVENT_TYPES = [
    "DEPOSIT",
    "WITHDRAWAL",
    "SELL_CSP",
    "ASSIGN_PUT",
    "SELL_CC",
    "CALL_AWAY",
    "BUY_SHARES",
    "SELL_SHARES",
    "DIVIDEND",
    "ADJUSTMENT",
]

def main():
    st.title("🔒 Admin Ledger (Manual Entry)")

    if not IS_ADMIN:
        st.error("This page is admin-only. Set YRC_ADMIN=1 to access.")
        st.stop()

    # ----------------------------------------------------------
    # Sidebar: select + create portfolio
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("Admin Controls")

        portfolios = load_portfolios()

        st.subheader("Select portfolio")
        if portfolios.empty:
            st.info("No portfolios found yet. Create one below.")
            selected_pid = None
        else:
            label_map = {
                int(r["portfolio_id"]): f"{r['code']} — {r['name']}"
                for _, r in portfolios.iterrows()
            }
            selected_pid = st.selectbox(
                "Portfolio",
                options=list(label_map.keys()),
                format_func=lambda pid: label_map[pid],
            )

        st.divider()
        st.subheader("Create / Update portfolio")

        new_code = st.text_input("Code", value="YRI").strip().upper()
        new_name = st.text_input("Name", value="You Rock Income Fund").strip()
        # IMPORTANT: use dt_date (aliased) to avoid shadowing bugs
        new_start = st.date_input("Start date", value=dt_date(2025, 12, 1))
        new_cash = st.number_input("Starting cash", value=100_000.0, step=5_000.0)

        if st.button("➕ Save portfolio"):
            if not new_code or not new_name:
                st.warning("Portfolio code and name are required.")
            elif new_cash <= 0:
                st.warning("Starting cash must be > 0.")
            else:
                pid = upsert_portfolio(new_code, new_name, new_start, float(new_cash))
                st.success(f"Saved {new_code} (portfolio_id={pid}).")
                st.rerun()

    if selected_pid is None:
        st.info("Create/select a portfolio to start logging events.")
        return

    # ----------------------------------------------------------
    # Portfolio summary card
    # ----------------------------------------------------------
    portfolios = load_portfolios()
    p = portfolios[portfolios["portfolio_id"] == selected_pid].iloc[0]

    st.markdown(
        f"""
        <div style="border:1px solid #eaeaea; border-radius:14px; padding:14px; background:#fff; margin-bottom:10px;">
          <div style="font-size:20px; font-weight:800;">{p["code"]} — {p["name"]}</div>
          <div style="color:#666; font-size:13px;">
            Start date <b>{p["start_date"]}</b> · Starting cash <b>{fmt_money0(p["starting_cash"])}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------
    # Entry form
    # ----------------------------------------------------------
    st.subheader("✍️ Add ledger event")

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        event_type = st.selectbox("Event type", EVENT_TYPES, index=0)

        # --- Event time inputs (Streamlit does NOT have datetime_input)
        event_date = st.date_input("Event date", value=dt_date.today(), key="event_date")
        event_clock = st.time_input("Event time", value=datetime.now().time(), key="event_time_clock")
        event_time = datetime.combine(event_date, event_clock)

        ticker = safe_upper(st.text_input("Ticker (optional)", value=""))
        fees = st.number_input("Fees", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        notes = st.text_input("Notes (optional)", value="")

        # core fields in your schema
        quantity = st.number_input("Quantity (shares or contracts)", value=0.0, step=1.0, format="%.2f")
        price = st.number_input("Price (stock or option)", value=0.0, step=0.01, format="%.2f")

        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            option_type = st.selectbox("Option type (if any)", ["", "put", "call"], index=0)
        with opt_col2:
            strike = st.number_input("Strike (if any)", value=0.0, step=0.50, format="%.2f")
        with opt_col3:
            # keep a valid date here; we’ll only store it if option_type is set
            expiry = st.date_input("Expiry (if any)", value=dt_date.today(), key="expiry")

        st.caption("Tip: Leave option fields blank/0 for non-option events. We’ll validate lightly for now.")

        # ----------------------------------------------------------
        # Cash delta logic (you can override)
        # ----------------------------------------------------------
        st.divider()
        st.markdown("**Cash delta** (required): positive = cash in, negative = cash out")

        auto_calc = st.checkbox("Auto-calc cash_delta (basic)", value=True)

        cash_delta_default = 0.0
        if auto_calc:
            # Very simple conventions:
            # - SELL_CSP / SELL_CC: cash in = quantity*100*price - fees (assumes quantity=contracts)
            # - BUY_SHARES: cash out = quantity*price + fees (assumes quantity=shares)
            # - SELL_SHARES: cash in = quantity*price - fees
            # - DEPOSIT / WITHDRAWAL: use "price" as amount (so you can just type the amount once)
            if event_type in ("SELL_CSP", "SELL_CC"):
                cash_delta_default = float(quantity) * 100.0 * float(price) - float(fees)
            elif event_type == "BUY_SHARES":
                cash_delta_default = -(float(quantity) * float(price) + float(fees))
            elif event_type == "SELL_SHARES":
                cash_delta_default = float(quantity) * float(price) - float(fees)
            elif event_type == "DEPOSIT":
                cash_delta_default = float(price)
            elif event_type == "WITHDRAWAL":
                cash_delta_default = -float(price)
            elif event_type == "DIVIDEND":
                cash_delta_default = float(price) - float(fees)
            else:
                # ASSIGN_PUT / CALL_AWAY / ADJUSTMENT etc: manual
                cash_delta_default = 0.0

        cash_delta = st.number_input(
            "cash_delta",
            value=float(cash_delta_default),
            step=1.0,
            format="%.2f",
            help="Final cash impact. Override if needed (recommended for assignments/call-away)."
        )

        # Minimal validation + save
        if st.button("✅ Save event"):
            if cash_delta is None:
                st.error("cash_delta is required.")
                st.stop()

            ot = (option_type or "").strip().lower()
            if ot not in ("", "put", "call"):
                st.error("option_type must be blank, put, or call.")
                st.stop()

            payload = {
                "portfolio_id": int(selected_pid),
                "event_time": event_time,
                "event_type": (event_type or "").strip(),
                "ticker": ticker if ticker else None,
                "quantity": float(quantity) if quantity is not None else None,
                "price": float(price) if price is not None else None,
                "fees": float(fees) if fees is not None else 0.0,
                "option_type": ot if ot else None,
                "strike": float(strike) if (strike is not None and float(strike) > 0) else None,
                "expiry": expiry if ot else None,
                "cash_delta": float(cash_delta),
                "notes": notes if notes else None,
            }

            try:
                insert_event(payload)
                st.success("Saved event to portfolio_events.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Insert failed: {e}")

    with right:
        st.subheader("🧾 Recent ledger events")

        limit = st.selectbox("Show last N events", [25, 50, 100, 200], index=1)
        events = load_recent_events(int(selected_pid), limit=int(limit))

        if events.empty:
            st.info("No events yet for this portfolio.")
        else:
            ev = events.copy()
            ev["event_time"] = pd.to_datetime(ev["event_time"])
            ev["expiry"] = pd.to_datetime(ev["expiry"]).dt.date

            for c in ["price", "fees", "strike", "cash_delta"]:
                if c in ev.columns:
                    ev[c] = ev[c].apply(lambda x: fmt_money(x) if pd.notnull(x) else "")

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
                ],
                hide_index=True,
                use_container_width=True,
            )

            st.caption("If you need to delete a bad entry, you can delete by event_id in SQL (we can add a delete button later).")

if __name__ == "__main__":
    main()
