# add_company.py
import os
import subprocess

import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import text

from db import get_engine  # ✅ centralized DB access

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Add Company", layout="centered")
st.title("➕ Add Company")

# ----------------------------------------------------------
# Admin gate (updates & pipeline for existing rows require admin)
# ----------------------------------------------------------
def _get_admin_code() -> str | None:
    """
    Read admin code from Streamlit secrets if present; otherwise from env var.
    Never raises StreamlitSecretNotFoundError.
    """
    try:
        code = st.secrets.get("admin_code")  # safe even if secrets missing
        if code:
            return str(code)
    except Exception:
        pass
    return os.environ.get("GREER_ADMIN_CODE")

ADMIN_CODE = _get_admin_code()
ADMIN_CONFIGURED = bool(ADMIN_CODE)

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

with st.sidebar:
    st.subheader("Admin")
    if ADMIN_CONFIGURED:
        if not st.session_state["is_admin"]:
            code = st.text_input("Enter admin code to enable updates & pipeline for existing tickers", type="password")
            if code:
                if code == ADMIN_CODE:
                    st.session_state["is_admin"] = True
                    st.success("Admin enabled for this session.")
                else:
                    st.error("Invalid admin code.")
        else:
            st.success("Admin mode is ON.")
            if st.button("Log out admin"):
                st.session_state["is_admin"] = False
    else:
        st.info(
            "Admin code not configured. Updates to existing companies and pipeline for existing tickers "
            "are disabled.\nSet `.streamlit/secrets.toml` admin_code or env var GREER_ADMIN_CODE."
        )

IS_ADMIN = bool(st.session_state["is_admin"])

# ----------------------------------------------------------
# Seed the ticker once from query param / session, then use a stateful input
# ----------------------------------------------------------
incoming = None
try:
    incoming = st.query_params.get("ticker")
except Exception:
    incoming = None
incoming = (incoming or st.session_state.get("pending_add_ticker") or "").upper().strip()

# Seed only if the widget doesn't already have a value
if incoming and not st.session_state.get("add_ticker"):
    st.session_state["add_ticker"] = incoming

# Controlled input: reads/writes st.session_state["add_ticker"]
st.text_input("Ticker", key="add_ticker", placeholder="e.g., AAPL")
ticker = (st.session_state.get("add_ticker") or "").upper().strip()

# ----------------------------------------------------------
# DB helpers
# ----------------------------------------------------------
def company_exists(symbol: str) -> bool:
    if not symbol:
        return False
    engine = get_engine()
    sql = text("SELECT 1 FROM companies WHERE ticker = :t LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(sql, {"t": symbol}).fetchone()
    return bool(row)

def has_financials(symbol: str) -> bool:
    if not symbol:
        return False
    engine = get_engine()
    sql = text("SELECT 1 FROM financials WHERE ticker = :t LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(sql, {"t": symbol}).fetchone()
    return bool(row)

def has_prices(symbol: str) -> bool:
    if not symbol:
        return False
    engine = get_engine()
    sql = text("SELECT 1 FROM prices WHERE ticker = :t LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(sql, {"t": symbol}).fetchone()
    return bool(row)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def validate_on_yfinance(symbol: str) -> dict | None:
    """
    Return basic company info if the symbol looks valid on yfinance,
    else return None. "Supported" = at least some price history OR non-empty info.
    """
    if not symbol:
        return None
    try:
        tk = yf.Ticker(symbol)

        # Try price data first (fast & robust)
        hist = tk.history(period="5d", auto_adjust=False)
        info = {}
        try:
            info = tk.fast_info or {}
        except Exception:
            pass

        # Try richer company info (yfinance API varies by version)
        try:
            meta = tk.get_info() or {}
        except Exception:
            meta = {}

        supported = (hist is not None and not hist.empty) or bool(meta)
        if not supported:
            return None

        # Normalize fields we care about
        name = meta.get("longName") or meta.get("shortName") or info.get("shortName")
        sector = meta.get("sector")
        industry = meta.get("industry")
        exchange = meta.get("exchange") or info.get("exchange")

        return {
            "ticker": symbol,
            "name": name,
            "sector": sector,
            "industry": industry,
            "exchange": exchange,
        }
    except Exception:
        return None

def upsert_company(row: dict, allow_update: bool) -> str:
    """
    Insert or update the companies row.
    - If allow_update=True -> full UPSERT (insert or update).
    - If allow_update=False -> INSERT ... DO NOTHING (no overwrites).
    Returns a short message about what happened.
    """
    engine = get_engine()
    exists = company_exists(row["ticker"])

    if allow_update:
        sql = text("""
            INSERT INTO companies (ticker, name, sector, industry, exchange, delisted, delisted_date, added_at)
            VALUES (:ticker, :name, :sector, :industry, :exchange, FALSE, NULL, NOW())
            ON CONFLICT (ticker) DO UPDATE SET
              name = EXCLUDED.name,
              sector = EXCLUDED.sector,
              industry = EXCLUDED.industry,
              exchange = EXCLUDED.exchange,
              delisted = FALSE,
              delisted_date = NULL
        """)
        with engine.begin() as conn:
            conn.execute(sql, row)
        return "inserted" if not exists else "updated"
    else:
        # Non-admin: allow creating NEW rows but never overwrite existing ones
        sql = text("""
            INSERT INTO companies (ticker, name, sector, industry, exchange, delisted, delisted_date, added_at)
            VALUES (:ticker, :name, :sector, :industry, :exchange, FALSE, NULL, NOW())
            ON CONFLICT (ticker) DO NOTHING
        """)
        with engine.begin() as conn:
            conn.execute(sql, row)
        return "inserted" if not exists else "skipped (exists; update requires admin)"

def _call_step(name: str, cmd: list[str]) -> tuple[str, int]:
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError:
        rc = 127  # script not found
    return (name, rc)

def run_one_ticker_pipeline(symbol: str) -> list[tuple[str, int]]:
    """
    Pipeline that gracefully handles brand-new tickers (prices exist, financials not yet).
    Order:
      1) fetch_company_info
      2) price_loader
      3) fetch_financials
      4) (conditional) greer_value_score + greer_value_yield_score if financials exist
      5) greer_fair_value_gap
      6) greer_buyzone_calculator
      7) refresh_snapshot
      8) greer_fair_value_calculator  <-- make sure this actually runs
    """
    results: list[tuple[str, int]] = []

    # 1) company info
    results.append(_call_step("fetch_company_info", ["python", "fetch_company_info.py", "--tickers", symbol]))
    if results[-1][1] != 0:
        return results

    # 2) prices
    results.append(_call_step("price_loader", ["python", "price_loader.py", "--tickers", symbol]))
    if results[-1][1] != 0:
        return results

    # 3) financials
    results.append(_call_step("fetch_financials", ["python", "fetch_financials.py", "--tickers", symbol]))
    if results[-1][1] != 0:
        return results

    # Check DB for financials after running the fetch
    fin_ok = has_financials(symbol)

    # 4) conditional value/yield steps
    if fin_ok:
        results.append(_call_step("greer_value_score", ["python", "greer_value_score.py", "--tickers", symbol]))
        if results[-1][1] != 0:
            return results
        results.append(_call_step("greer_value_yield_score", ["python", "greer_value_yield_score.py", "--tickers", symbol]))
        if results[-1][1] != 0:
            return results
    else:
        results.append(("greer_value_score (skipped – no financials)", 0))
        results.append(("greer_value_yield_score (skipped – no financials)", 0))

    # 5) FVG (price-only)
    results.append(_call_step("greer_fair_value_gap", ["python", "greer_fair_value_gap.py", "--tickers", symbol]))
    if results[-1][1] != 0:
        return results

    # 6) BuyZone
    results.append(_call_step("greer_buyzone_calculator", ["python", "greer_buyzone_calculator.py", "--tickers", symbol]))
    if results[-1][1] != 0:
        return results

    # 7) refresh snapshot / MVs
    results.append(_call_step("run_opportunities_refresh", ["python", "refresh_snapshot.py"]))
    if results[-1][1] != 0:
        return results

    # 8) Greer Fair Value (always run; handles missing data gracefully)
    results.append(_call_step("greer_fair_value", ["python", "greer_fair_value_calculator.py", "--tickers", symbol]))

    return results


def render_company_preview(info: dict) -> None:
    st.subheader("Company Preview")
    df = pd.DataFrame([info]).T
    df.columns = ["Value"]
    st.dataframe(df)

# ----------------------------------------------------------
# UI Flow
# ----------------------------------------------------------
info: dict | None = None

exists_now = company_exists(ticker) if ticker else False

# Show a notice if user tries to update an existing company without admin
if ticker and exists_now and not IS_ADMIN:
    st.warning(
        "This ticker already exists. You can add new tickers, but **updating existing company details "
        "or running the data pipeline for existing tickers requires admin**."
    )

if ticker:
    with st.spinner("Checking yfinance…"):
        info = validate_on_yfinance(ticker)

    if info is None:
        st.error("❌ Not seeing this ticker on yfinance. Double-check the symbol and try again.")
    else:
        render_company_preview(info)

# Determine if pipeline is allowed for the current typed ticker
pipeline_allowed = bool(ticker) and (IS_ADMIN or not exists_now)

run_now = st.checkbox(
    "Also run the full one-ticker import (prices, financials if available, yields, buyzone, FVG, refresh MVs)",
    value=pipeline_allowed,
    disabled=not pipeline_allowed,
    help="Pipeline can be run by admins anytime, or by anyone for brand-new tickers."
)

btn_label = (
    "Add / Update Company" if IS_ADMIN or not exists_now
    else "Add Company (no update)"
)
if st.button(btn_label, type="primary", use_container_width=True):
    submit_ticker = (st.session_state.get("add_ticker") or "").upper().strip()
    if not submit_ticker:
        st.error("Please enter a ticker.")
        st.stop()

    # Re-evaluate existence & permission on submit
    exists_submit = company_exists(submit_ticker)
    pipeline_allowed_submit = IS_ADMIN or not exists_submit

    # Validate again on submit to be safe
    with st.spinner("Validating ticker…"):
        info = validate_on_yfinance(submit_ticker)

    if info is None:
        st.error("❌ Not seeing this ticker on yfinance. Double-check the symbol and try again.")
        st.stop()

    try:
        allow_update = IS_ADMIN or not exists_submit
        result = upsert_company(info, allow_update=allow_update)

        if result == "updated":
            st.success(f"✅ Updated `{submit_ticker}` in `companies` (admin).")
        elif result == "inserted":
            st.success(f"✅ Inserted `{submit_ticker}` into `companies`.")
        else:
            st.info(f"ℹ️ `{submit_ticker}` already exists — company record left unchanged (admin required to update).")

        # Enforce pipeline rule even if someone managed to toggle the box
        if run_now and not pipeline_allowed_submit:
            st.error("⛔ Running the data pipeline is restricted to admins or brand-new tickers.")
            run_now = False

        if run_now:
            st.info("🚀 Running one-ticker pipeline…")
            results = run_one_ticker_pipeline(submit_ticker)

            # Show each step result
            ok = True
            for step, rc in results:
                if rc == 0:
                    st.write(f"• **{step}** — ✅ OK")
                else:
                    ok = False
                    st.write(f"• **{step}** — ❌ Exit code {rc} (stopped)")
                    break

            # Guidance if no financials yet
            if ok:
                if not has_financials(submit_ticker):
                    if has_prices(submit_ticker):
                        st.warning(
                            "This looks like a **newly listed company** with **prices starting recently** "
                            "but **no financial statements in the database yet**. "
                            "I created a **price-only profile** (FVG & BuyZone). "
                            "Greer Value & Yields will populate automatically once financials are available."
                        )
                    else:
                        st.info(
                            "No price data loaded yet. Once pricing is available, price-only features "
                            "will appear; Greer Value & Yields will follow after financials are published."
                        )

                # Clear caches so Home picks up new data immediately
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.success("🎉 Import finished. The Home page should now reflect what’s available for this ticker.")

        # Clear transient seed after success (optional)
        st.session_state.pop("pending_add_ticker", None)

        st.page_link("Home.py", label="⬅️ Back to Home", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to add company: {e}")

# Friendly hint if no ticker yet
if not ticker:
    st.info("Enter a ticker above to validate and add it.")
