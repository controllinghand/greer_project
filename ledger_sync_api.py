# ledger_sync_api.py
# ----------------------------------------------------------
# Google Sheet → Portfolio Ledger Sync API
# Inserts events into portfolio_events table
# ----------------------------------------------------------

import os
from flask import Flask, request, jsonify
from sqlalchemy import text
from db import get_engine

app = Flask(__name__)

SYNC_SECRET = os.getenv("GSHEET_SYNC_SECRET")


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

    portfolio_id = get_portfolio_id(fund)

    sync_type = data["sync_type"]

    if sync_type == "entry":

        marker = f"GSHEET:{trade_id}:ENTRY"

        if event_exists(portfolio_id, marker):
            return jsonify({"status": "already_synced"})

        contracts = float(data["contracts"])
        premium = float(data["premium"])
        fees = float(data["fees"])
        strike = float(data["strike"])

        expiry = data["expiry"]
        event_time = data["entry_date"]

        option_type = "put" if strategy == "Put" else "call"

        if strategy == "Put":
            event_type = "SELL_CSP"
        else:
            event_type = "SELL_CC"

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

        shares = float(data["shares"])
        strike = float(data["strike"])
        fees = float(data.get("fees", 0))
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