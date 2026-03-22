# ledger_sync_api.py
# ----------------------------------------------------------
# Google Sheet ↔ YouRockClub API
# - Google Sheet -> Portfolio Ledger Sync API
# - Database -> Weekly CSP Targets API
# ----------------------------------------------------------

import os
import pandas as pd
from datetime import date, timedelta
from flask import Flask, request, jsonify
from sqlalchemy import text
from db import get_engine

app = Flask(__name__)

SYNC_SECRET = os.getenv("GSHEET_SYNC_SECRET")

# ----------------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/db")
def health_db():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


print(f"LOADED FILE: {__file__}")
print("REGISTERED ROUTES:")
for rule in app.url_map.iter_rules():
    print(rule)


# ----------------------------------------------------------
# Sheet fund code -> DB portfolio code map
# ----------------------------------------------------------
FUND_CODE_MAP = {
    "YRVI": "YRVI-26",
    "YRSI": "YRSI-26",
    "YR3G": "YR3G-26",
    "YROG": "YROG-26",
    "YRQI": "YRQI-26",
    "SPY": "SPY-26",
    "QQQ": "QQQ-26",
    "GLD": "GLD-26",
    "BTC": "BTC-26",
    "ROTH": "ROTH-26",
}


# ----------------------------------------------------------
# Map sheet fund code to DB portfolio code
# ----------------------------------------------------------
def map_fund_code(sheet_code):
    code = (sheet_code or "").strip().upper()
    return FUND_CODE_MAP.get(code, code)


# ----------------------------------------------------------
# Helper: get portfolio_id from fund code
# ----------------------------------------------------------
def get_portfolio_id(code):
    engine = get_engine()

    q = text("""
        SELECT portfolio_id
        FROM portfolios
        WHERE code = :code
    """)

    with engine.connect() as conn:
        row = conn.execute(q, {"code": code}).fetchone()

    if not row:
        raise Exception(f"Portfolio not found for code={code}")

    return row[0]


# ----------------------------------------------------------
# Helper: prevent duplicate inserts
# ----------------------------------------------------------
def event_exists(portfolio_id, marker):
    engine = get_engine()

    q = text("""
        SELECT 1
        FROM portfolio_events
        WHERE portfolio_id = :pid
          AND notes = :marker
        LIMIT 1
    """)

    with engine.connect() as conn:
        row = conn.execute(q, {"pid": portfolio_id, "marker": marker}).fetchone()

    return row is not None


# ----------------------------------------------------------
# Insert ledger event
# ----------------------------------------------------------
def insert_event(payload):
    engine = get_engine()

    q = text("""
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
        conn.execute(q, payload)


# ----------------------------------------------------------
# Helper: earnings flag for weekly targets
# ----------------------------------------------------------
def earnings_flag(days_to_earnings):
    try:
        if pd.isna(days_to_earnings):
            return "❓"

        v = int(days_to_earnings)

        # Recently reported
        if v < 0:
            if v >= -3:
                return "🟣"
            return "⚫"

        # Upcoming
        if v == 0:
            return "🚨"
        if v <= 3:
            return "🟥"
        if v <= 7:
            return "🟧"

        return "✅"
    except Exception:
        return "❓"


# ----------------------------------------------------------
# Helper: add CSP safety / wheel flags
# ----------------------------------------------------------
def add_put_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out["put_20d_strike"] = pd.to_numeric(out["put_20d_strike"], errors="coerce")
    out["put_20d_delta"] = pd.to_numeric(out["put_20d_delta"], errors="coerce")

    out["put_itm_flag"] = (
        out["put_20d_strike"].notna()
        & out["underlying_price"].notna()
        & (out["put_20d_strike"] > out["underlying_price"])
    )

    out["put_delta_abs"] = out["put_20d_delta"].abs()
    out["delta_mismatch_flag"] = (
        out["put_delta_abs"].notna()
        & (out["put_delta_abs"] > 0.35)
    )

    reasons = []
    for _, row in out.iterrows():
        r = []

        if bool(row.get("put_itm_flag", False)):
            r.append("PUT strike > underlying (PUT ITM)")

        put_delta_abs = row.get("put_delta_abs")
        if pd.notna(put_delta_abs) and float(put_delta_abs) > 0.35:
            r.append(f"PUT |Δ| too high ({float(put_delta_abs):.2f})")

        reasons.append(" · ".join(r) if r else "OK")

    out["wheel_reason"] = reasons
    out["wheel_flag"] = "✅"

    red = out["put_itm_flag"].fillna(False) | out["delta_mismatch_flag"].fillna(False)
    out.loc[red, "wheel_flag"] = "❌"

    out["wheel_fit"] = out["wheel_flag"].map({
        "✅": "Wheel-ready",
        "❌": "Avoid/Review"
    })

    return out


# ----------------------------------------------------------
# Load weekly CSP targets from database
# ----------------------------------------------------------
def load_csp_targets(iv_min_atm: float, market_cap_min: float, min_star_rating: int) -> pd.DataFrame:
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
    market_caps AS (
      SELECT
        ls.ticker,
        (ls.shares_outstanding * lp.latest_price) AS market_cap
      FROM latest_shares ls
      JOIN latest_price lp
        ON lp.ticker = ls.ticker
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

      c.greer_star_rating,
      c.sector,

      snap.greer_value_score,
      snap.greer_yield_score,
      snap.buyzone_flag,

      gfv.gfv_price,
      gfv.gfv_status,

      mc.market_cap,
      lp.latest_price,

      r.fetch_date,
      r.expiry,
      r.dte,
      r.contract_count,
      r.iv_atm,
      r.iv_median,
      r.underlying_price,

      r.put_20d_strike,
      r.put_20d_iv,
      r.put_20d_premium,
      r.put_20d_premium_pct,
      r.put_20d_delta,

      e.earnings_date,
      e.days_to_earnings

    FROM recent_iv r
    JOIN market_caps mc
      ON mc.ticker = r.ticker
    JOIN latest_price lp
      ON lp.ticker = r.ticker
    LEFT JOIN companies c
      ON c.ticker = r.ticker
    LEFT JOIN latest_company_snapshot snap
      ON snap.ticker = r.ticker
    LEFT JOIN greer_fair_value_latest gfv
      ON gfv.ticker = r.ticker
    LEFT JOIN latest_company_earnings e
      ON e.ticker = r.ticker
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
# Main Sync Endpoint
# ----------------------------------------------------------
@app.post("/api/ledger/sync")
def sync_ledger():

    data = request.get_json()

    if not data:
        return jsonify({"error": "missing json"}), 400

    if data.get("secret") != SYNC_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    trade_id = data["trade_id"]
    fund = data["fund"]
    ticker = data["ticker"]
    strategy = data["strategy"]

    db_fund_code = map_fund_code(fund)

    try:
        portfolio_id = get_portfolio_id(db_fund_code)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "fund": fund,
            "db_fund_code": db_fund_code
        }), 404

    sync_type = data["sync_type"]

    if sync_type == "entry":

        marker = f"GSHEET:{trade_id}:ENTRY"

        if event_exists(portfolio_id, marker):
            return jsonify({"status": "already_synced"})

        contracts = float(data.get("contracts") or 0)
        premium = float(data.get("premium") or 0)
        fees = float(data.get("fees") or 0)
        strike = float(data.get("strike") or 0)

        expiry = data["expiry"]
        event_time = data["entry_date"]

        option_type = "put" if strategy == "Put" else "call"
        event_type = "SELL_CSP" if strategy == "Put" else "SELL_CC"

        cash_delta = contracts * 100 * premium - fees

        insert_event({
            "portfolio_id": portfolio_id,
            "event_time": event_time,
            "event_type": event_type,
            "ticker": ticker,
            "quantity": contracts,
            "price": premium,
            "fees": fees,
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry,
            "cash_delta": cash_delta,
            "notes": marker
        })

        return jsonify({"status": "entry_synced"})

    elif sync_type == "result":

        result = data["result"]
        marker = f"GSHEET:{trade_id}:RESULT"

        if event_exists(portfolio_id, marker):
            return jsonify({"status": "already_synced"})

        if result == "Expired":
            return jsonify({"status": "expired_no_event"})

        shares = float(data.get("shares") or 0)
        strike = float(data.get("strike") or 0)
        fees = float(data.get("fees") or 0)
        event_time = data["expiry"]

        if result == "Assigned":
            event_type = "ASSIGN_PUT"
            quantity = shares
            cash_delta = -(shares * strike + fees)

        elif result == "Called Away":
            event_type = "CALL_AWAY"
            quantity = -shares
            cash_delta = shares * strike - fees

        else:
            return jsonify({"error": "unknown result"}), 400

        insert_event({
            "portfolio_id": portfolio_id,
            "event_time": event_time,
            "event_type": event_type,
            "ticker": ticker,
            "quantity": quantity,
            "price": strike,
            "fees": fees,
            "option_type": None,
            "strike": None,
            "expiry": None,
            "cash_delta": cash_delta,
            "notes": marker
        })

        return jsonify({"status": "result_synced"})

    return jsonify({"error": "invalid sync_type"}), 400


# ----------------------------------------------------------
# Weekly CSP Targets Endpoint
# ----------------------------------------------------------
@app.get("/api/targets/csp")
def get_csp_targets():

    secret = request.args.get("secret")
    if secret != SYNC_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    try:
        iv_min_atm = float(request.args.get("iv_min_atm", 0.40))
        market_cap_min = float(request.args.get("market_cap_min", 10_000_000_000))
        min_star_rating = int(request.args.get("min_star_rating", 0))
        expiry_days = int(request.args.get("expiry_days", 7))
        min_target_premium_pct = float(request.args.get("min_target_premium_pct", 0.01))
        earnings_days_hide = int(request.args.get("earnings_days_hide", 7))
        earnings_recent_hide = int(request.args.get("earnings_recent_hide", 0))
        hide_red = request.args.get("hide_red", "true").lower() == "true"
    except Exception as e:
        return jsonify({"error": f"invalid parameters: {str(e)}"}), 400

    df = load_csp_targets(
        iv_min_atm=iv_min_atm,
        market_cap_min=market_cap_min,
        min_star_rating=min_star_rating
    )

    if df.empty:
        return jsonify({"count": 0, "rows": []})

    # ----------------------------------------------------------
    # Clean and convert data
    # ----------------------------------------------------------
    df["fetch_date"] = pd.to_datetime(df["fetch_date"], errors="coerce").dt.date
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    df["days_to_earnings"] = pd.to_numeric(df["days_to_earnings"], errors="coerce").astype("Int64")

    df["greer_star_rating"] = pd.to_numeric(df["greer_star_rating"], errors="coerce").astype("Int64")
    df["greer_value_score"] = pd.to_numeric(df["greer_value_score"], errors="coerce").round(2)
    df["greer_yield_score"] = pd.to_numeric(df["greer_yield_score"], errors="coerce").astype("Int64")

    df["gfv_price"] = pd.to_numeric(df["gfv_price"], errors="coerce").round(2)
    df["latest_price"] = pd.to_numeric(df["latest_price"], errors="coerce").round(2)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce").round(3)
    df["iv_median"] = pd.to_numeric(df["iv_median"], errors="coerce").round(3)
    df["contract_count"] = pd.to_numeric(df["contract_count"], errors="coerce").astype("Int64")

    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce").round(2)
    df["put_20d_strike"] = pd.to_numeric(df["put_20d_strike"], errors="coerce").round(2)
    df["put_20d_iv"] = pd.to_numeric(df["put_20d_iv"], errors="coerce").round(3)
    df["put_20d_premium"] = pd.to_numeric(df["put_20d_premium"], errors="coerce").round(2)
    df["put_20d_premium_pct"] = pd.to_numeric(df["put_20d_premium_pct"], errors="coerce")
    df["put_20d_delta"] = pd.to_numeric(df["put_20d_delta"], errors="coerce").round(3)

    df["earnings_flag"] = df["days_to_earnings"].apply(earnings_flag)

    # ----------------------------------------------------------
    # Filters
    # ----------------------------------------------------------
    df = df[df["contract_count"].fillna(0) >= 10]

    today = date.today()
    max_allowed = today + timedelta(days=expiry_days)
    df = df[df["expiry"].notna()]
    df = df[df["expiry"] <= max_allowed]

    if earnings_days_hide > 0:
        df = df[
            df["days_to_earnings"].isna()
            | (df["days_to_earnings"] < 0)
            | (df["days_to_earnings"] > earnings_days_hide)
        ]

    if earnings_recent_hide > 0:
        df = df[
            df["days_to_earnings"].isna()
            | (df["days_to_earnings"] >= 0)
            | (df["days_to_earnings"] < -earnings_recent_hide)
        ]

    df = df[
        df["put_20d_premium_pct"].notna()
        & df["put_20d_strike"].notna()
        & df["put_20d_premium"].notna()
    ]

    df = df[df["put_20d_premium_pct"] >= min_target_premium_pct]

    if df.empty:
        return jsonify({"count": 0, "rows": []})

    # ----------------------------------------------------------
    # Add wheel safety flags
    # ----------------------------------------------------------
    df = add_put_flags(df)

    if hide_red:
        df = df[df["wheel_flag"] != "❌"]

    if df.empty:
        return jsonify({"count": 0, "rows": []})

    # ----------------------------------------------------------
    # Final sort
    # ----------------------------------------------------------
    df = df.sort_values(
        by=["put_20d_premium_pct", "iv_atm", "market_cap"],
        ascending=[False, False, False],
        na_position="last"
    )

    # ----------------------------------------------------------
    # Final output
    # ----------------------------------------------------------
    output_cols = [
        "ticker",
        "greer_star_rating",
        "greer_value_score",
        "greer_yield_score",
        "gfv_price",
        "gfv_status",
        "buyzone_flag",
        "days_to_earnings",
        "earnings_flag",
        "earnings_date",
        "sector",
        "latest_price",
        "expiry",
        "put_20d_strike",
        "put_20d_premium",
        "put_20d_premium_pct",
        "put_20d_delta",
        "put_20d_iv",
        "iv_atm",
        "iv_median",
        "contract_count",
        "wheel_flag",
        "wheel_fit",
        "wheel_reason",
        "fetch_date"
    ]

    rows = df[output_cols].copy()

    # Convert pandas nulls to JSON-safe None
    rows = rows.astype(object).where(pd.notnull(rows), None)

    return jsonify({
        "count": len(rows),
        "rows": rows.to_dict(orient="records")
    })

print("FINAL ROUTES AFTER REGISTRATION:")
for rule in app.url_map.iter_rules():
    print(rule)

