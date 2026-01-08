# 13_Admin_Ledger.py
# ----------------------------------------------------------
# Private Admin Ledger page (BEST UX)
# - Manual entry into existing tables:
#     portfolios(...)
#     portfolio_events(...)
# - Admin-only via env var: YRC_ADMIN=1
#
# UX goals:
# - You NEVER type negative shares again
# - Shares/contracts inputs are always positive
# - Event type determines direction automatically
# - Auto-calc cash_delta for common events (override allowed)
# - Only show fields relevant to the selected event type
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

if not IS_ADMIN:
    # Optional: hide details; just stop.
    st.error("This page is admin-only.")
    st.stop()

# Remove this line when using st.navigation:
# st.set_page_config(page_title="Admin Ledger", layout="wide")


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

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

# ----------------------------------------------------------
# Event sets
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

OPTION_EVENTS = {"SELL_CSP", "SELL_CC"}
SHARE_EVENTS = {"BUY_SHARES", "SELL_SHARES", "ASSIGN_PUT", "CALL_AWAY"}
CASH_ONLY_EVENTS = {"DEPOSIT", "WITHDRAWAL", "DIVIDEND", "ADJUSTMENT"}

# ----------------------------------------------------------
# Direction rules (SIGN-PROOF)
# - Shares/contracts inputs are ALWAYS positive in UI
# - We apply correct signs for quantity + cash_delta by event_type
# ----------------------------------------------------------
def signed_share_qty(event_type: str, shares_abs: float) -> float:
    """
    Convert positive share input into signed quantity stored in DB.
    """
    et = (event_type or "").strip().upper()
    s = abs(safe_float(shares_abs, 0.0))
    if et in ("BUY_SHARES", "ASSIGN_PUT"):
        return +s
    if et in ("SELL_SHARES", "CALL_AWAY"):
        return -s
    return s

def signed_contract_qty(event_type: str, contracts_abs: float) -> float:
    """
    Contracts are generally stored positive for SELL_CSP/SELL_CC in your system.
    Keep as positive to preserve your existing convention.
    """
    et = (event_type or "").strip().upper()
    c = abs(safe_float(contracts_abs, 0.0))
    if et in ("SELL_CSP", "SELL_CC"):
        return +c
    return c

def calc_cash_delta(event_type: str, qty_signed: float, price: float, fees: float) -> float:
    """
    Basic conventions:
    - SELL_CSP / SELL_CC: +contracts*100*option_price - fees
    - ASSIGN_PUT:         -(abs(shares)*stock_price + fees)
    - CALL_AWAY:          +(abs(shares)*stock_price - fees)
    - BUY_SHARES:         -(abs(shares)*stock_price + fees)
    - SELL_SHARES:        +(abs(shares)*stock_price - fees)
    - DEPOSIT/WITHDRAWAL: use amount as price input (amount field)
    - DIVIDEND:           +(amount - fees)
    - ADJUSTMENT:         manual
    """
    et = (event_type or "").strip().upper()
    p = safe_float(price, 0.0)
    f = safe_float(fees, 0.0)

    if et in ("SELL_CSP", "SELL_CC"):
        contracts = abs(safe_float(qty_signed, 0.0))
        return contracts * 100.0 * p - f

    if et == "ASSIGN_PUT":
        shares = abs(safe_float(qty_signed, 0.0))
        return -(shares * p + f)

    if et == "CALL_AWAY":
        shares = abs(safe_float(qty_signed, 0.0))
        return shares * p - f

    if et == "BUY_SHARES":
        shares = abs(safe_float(qty_signed, 0.0))
        return -(shares * p + f)

    if et == "SELL_SHARES":
        shares = abs(safe_float(qty_signed, 0.0))
        return shares * p - f

    if et == "DEPOSIT":
        return p

    if et == "WITHDRAWAL":
        return -p

    if et == "DIVIDEND":
        return p - f

    # ADJUSTMENT or anything else: default 0 (manual)
    return 0.0

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
# DB: Insert event
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
# Main
# ----------------------------------------------------------
def main():
    st.title("🔒 Admin Ledger (Manual Entry)")

    if not IS_ADMIN:
        st.error("This page is admin-only.")
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

    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        event_type = st.selectbox("Event type", EVENT_TYPES, index=0)
        et_u = (event_type or "").strip().upper()

        # Event time inputs
        c1, c2 = st.columns(2)
        with c1:
            event_date = st.date_input("Event date", value=dt_date.today(), key="event_date")
        with c2:
            event_clock = st.time_input("Event time", value=datetime.now().time(), key="event_time_clock")
        event_time = datetime.combine(event_date, event_clock)

        st.divider()

        # ----------------------------------------------------------
        # Dynamic input panels by event type
        # ----------------------------------------------------------
        ticker = ""
        qty_signed = 0.0
        price = 0.0
        fees = 0.0
        option_type = None
        strike = None
        expiry = None

        notes = st.text_input("Notes (optional)", value="")

        # Fees shown for most things
        fees = st.number_input("Fees", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        if et_u in CASH_ONLY_EVENTS:
            # Cash-only UX: user types amount once
            amount_label = "Amount"
            if et_u == "DEPOSIT":
                amount_label = "Deposit amount"
            elif et_u == "WITHDRAWAL":
                amount_label = "Withdrawal amount"
            elif et_u == "DIVIDEND":
                amount_label = "Dividend amount"
            elif et_u == "ADJUSTMENT":
                amount_label = "Adjustment amount (use + / - via cash_delta)"

            amount = st.number_input(amount_label, value=0.0, step=100.0, format="%.2f")

            # For cash-only, store amount in price (keeps schema simple)
            price = float(amount)
            qty_signed = 0.0
            ticker = ""

        elif et_u in OPTION_EVENTS:
            # Options UX: contracts always positive, auto option_type, auto premium cash_delta
            ticker = safe_upper(st.text_input("Ticker", value=""))

            contracts = st.number_input("Contracts (always positive)", min_value=0.0, value=1.0, step=1.0, format="%.0f")
            opt_price = st.number_input("Option price (premium)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

            if et_u == "SELL_CSP":
                option_type = "put"
            elif et_u == "SELL_CC":
                option_type = "call"

            strike = st.number_input("Strike", min_value=0.0, value=0.0, step=0.50, format="%.2f")
            expiry = st.date_input("Expiry", value=dt_date.today(), key="expiry_opt")

            qty_signed = signed_contract_qty(et_u, contracts)
            price = float(opt_price)

            st.caption("Contracts are stored as a positive count. Premium cash_delta is auto-calculated if enabled below.")

        elif et_u in SHARE_EVENTS:
            # Shares UX: shares always positive, we enforce sign automatically
            ticker = safe_upper(st.text_input("Ticker", value=""))

            shares = st.number_input("Shares (always positive)", min_value=0.0, value=100.0, step=1.0, format="%.0f")
            stock_price = st.number_input("Stock price", min_value=0.0, value=0.0, step=0.01, format="%.2f")

            qty_signed = signed_share_qty(et_u, shares)
            price = float(stock_price)

            # Nice UX hint
            if et_u == "ASSIGN_PUT":
                st.info("ASSIGN_PUT will be saved as a positive share quantity (adds shares).")
            if et_u == "CALL_AWAY":
                st.info("CALL_AWAY will be saved as a negative share quantity (removes shares).")

        else:
            st.warning("Unknown event type configuration.")

        st.divider()
        st.markdown("### 💰 Cash delta")

        auto_calc = st.checkbox("Auto-calc cash_delta", value=True)

        auto_cash = calc_cash_delta(et_u, qty_signed, price, fees) if auto_calc else 0.0
        cash_delta = st.number_input(
            "cash_delta (override allowed)",
            value=float(auto_cash),
            step=1.0,
            format="%.2f",
            help="Positive = cash in, Negative = cash out. Auto-calculated for common events."
        )

        # ----------------------------------------------------------
        # Validation + Save
        # ----------------------------------------------------------
        st.divider()
        st.markdown("### ✅ Review")

        # Quick validation rules
        errors = []

        if et_u in (OPTION_EVENTS | SHARE_EVENTS) and not ticker:
            errors.append("Ticker is required for this event type.")

        if et_u in SHARE_EVENTS and safe_float(price) <= 0:
            errors.append("Stock price must be > 0 for share events.")

        if et_u in OPTION_EVENTS:
            if safe_float(price) <= 0:
                errors.append("Option price (premium) must be > 0 for option sells.")
            if strike is None or safe_float(strike) <= 0:
                errors.append("Strike must be > 0 for option sells.")
            if expiry is None:
                errors.append("Expiry is required for option sells.")

        if et_u == "ADJUSTMENT" and cash_delta == 0:
            st.warning("ADJUSTMENT with cash_delta = 0 will have no impact (that may be intended).")

        # Show computed payload preview (super helpful)
        preview = {
            "portfolio_id": int(selected_pid),
            "event_time": event_time,
            "event_type": et_u,
            "ticker": ticker if ticker else None,
            "quantity": float(qty_signed) if qty_signed is not None else None,
            "price": float(price) if price is not None else None,
            "fees": float(fees) if fees is not None else 0.0,
            "option_type": option_type if option_type else None,
            "strike": float(strike) if (strike is not None and safe_float(strike) > 0) else None,
            "expiry": expiry if option_type else None,
            "cash_delta": float(cash_delta),
            "notes": notes if notes else None,
        }
        st.code(preview, language="python")

        if errors:
            for e in errors:
                st.error(e)

        if st.button("✅ Save event", disabled=bool(errors)):
            try:
                insert_event(preview)
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
            if "expiry" in ev.columns:
                ev["expiry"] = pd.to_datetime(ev["expiry"]).dt.date

            for c in ["price", "fees", "strike", "cash_delta"]:
                if c in ev.columns:
                    ev[c] = ev[c].apply(lambda x: fmt_money(x) if pd.notnull(x) else "")

            st.dataframe(
                ev[
                    [
                        "event_id",
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

            st.caption("Tip: We can add edit/delete buttons next (safe-mode with confirmations).")

if __name__ == "__main__":
    main()
