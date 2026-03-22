# price_loader_tradier.py

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, UTC, date
from sqlalchemy import text
import logging

from db import get_engine
from fetch_iv_summary import TradierClient

def get_effective_end_date() -> date:
    today = datetime.now(UTC).date()

    # Saturday -> Friday
    if today.weekday() == 5:
        return today - timedelta(days=1)

    # Sunday -> Friday
    if today.weekday() == 6:
        return today - timedelta(days=2)

    return today
# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = get_engine()

# ----------------------------------------------------------
# Get latest dates per ticker
# ----------------------------------------------------------
def get_latest_dates():
    query = """
        SELECT ticker, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return dict(zip(df["ticker"], df["max_date"]))

# ----------------------------------------------------------
# Insert prices
# ----------------------------------------------------------
def bulk_insert_prices(rows):
    if not rows:
        return 0

    query = text("""
        INSERT INTO prices (ticker, date, close, high_price, low_price)
        VALUES (:ticker, :date, :close, :high_price, :low_price)
        ON CONFLICT (ticker, date) DO UPDATE SET
            close = EXCLUDED.close,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price
    """)

    with engine.begin() as conn:
        conn.execute(query, rows)

    logger.info(f"Inserted/updated {len(rows)} rows")
    return len(rows)

# ----------------------------------------------------------
# Fetch prices from Tradier
# ----------------------------------------------------------
def fetch_prices_tradier(tradier, ticker, start_date):
    end_date = get_effective_end_date()

    if start_date > end_date:
        logger.info(f"{ticker}: no new data to import (start_date {start_date} > end_date {end_date})")
        return []

    params = {
        "symbol": ticker,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "interval": "daily",
    }

    logger.info(f"{ticker}: Tradier history params={params}")
    data = tradier.get("/markets/history", params=params)

    if not isinstance(data, dict):
        logger.info(f"{ticker}: unexpected non-dict response; treating as no new data")
        return []

    history_obj = data.get("history")
    if not isinstance(history_obj, dict):
        logger.info(f"{ticker}: no history object returned for {params['start']} to {params['end']}")
        return []

    history = history_obj.get("day")
    if not history:
        logger.info(f"{ticker}: no daily history rows returned for {params['start']} to {params['end']}")
        return []

    if isinstance(history, dict):
        history = [history]

    rows = []
    for row in history:
        try:
            rows.append({
                "ticker": ticker,
                "date": row["date"],
                "close": float(row["close"]),
                "high_price": float(row["high"]),
                "low_price": float(row["low"]),
            })
        except Exception as e:
            logger.warning(f"{ticker}: skipping malformed row {row} error={e}")
            continue

    # ✅ DEBUG LOG (now actually runs)
    if rows:
        returned_dates = [r["date"] for r in rows]
        logger.info(
            f"{ticker}: received {len(rows)} row(s), "
            f"min_date={min(returned_dates)}, max_date={max(returned_dates)}"
        )

    return rows
# ----------------------------------------------------------
# Load tickers
# ----------------------------------------------------------
def load_tickers(args):
    if args.tickers:
        return [t.upper() for t in args.tickers]

    if args.file:
        df = pd.read_csv(args.file)
        return df["ticker"].str.upper().tolist()

    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT ticker FROM companies WHERE COALESCE(delisted,false)=false",
            conn
        )

    return df["ticker"].tolist()

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--file")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--start", default="2012-01-01")

    args = parser.parse_args()

    token = os.getenv("TRADIER_PRICE_TOKEN", "").strip()
    env = os.getenv("TRADIER_PRICE_ENV", "live").strip().lower()

    if not token:
        raise SystemExit("Missing TRADIER_PRICE_TOKEN env var")

    logger.info(f"Using Tradier price environment: {env}")
    print(f"Using Tradier price environment: {env}")

    tradier = TradierClient(token=token, env=env)

    tickers = load_tickers(args)
    latest_dates = get_latest_dates()

    logger.info(f"Processing {len(tickers)} tickers")

    total_rows = 0
    no_data_tickers = []

    for ticker in tickers:
        try:
            if args.reload or ticker not in latest_dates:
                start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
            else:
                start_date = latest_dates[ticker] + timedelta(days=1)

            logger.info(f"{ticker}: fetching from {start_date}")

            rows = fetch_prices_tradier(tradier, ticker, start_date)

            if not rows:
                logger.info(f"{ticker}: no new data to import")
                no_data_tickers.append(ticker)
                continue

            inserted = bulk_insert_prices(rows)
            total_rows += inserted

        except Exception as e:
            logger.exception(f"{ticker}: error {e}")

    logger.info(f"Done. Total inserted/updated rows: {total_rows}")
    print(f"Done. Total inserted/updated rows: {total_rows}")

    if no_data_tickers:
        logger.info(f"No new data for {len(no_data_tickers)} ticker(s): {', '.join(no_data_tickers)}")
        print(f"No new data for {len(no_data_tickers)} ticker(s): {', '.join(no_data_tickers)}")

if __name__ == "__main__":
    main()