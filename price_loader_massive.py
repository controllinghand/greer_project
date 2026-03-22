# price_loader_massive.py

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, UTC, date

import pandas as pd
from massive import RESTClient
from sqlalchemy import text

from db import get_engine

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = get_engine()

# ----------------------------------------------------------
# Get latest dates per ticker
# ----------------------------------------------------------
def get_latest_dates() -> dict:
    query = """
        SELECT ticker, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return dict(zip(df["ticker"], df["max_date"]))


# ----------------------------------------------------------
# Weekend-aware effective end date
# ----------------------------------------------------------
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
# Insert prices
# ----------------------------------------------------------
def bulk_insert_prices(rows: list[dict]) -> int:
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
# Load tickers
# ----------------------------------------------------------
def load_tickers(args) -> list[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]

    if args.file:
        df = pd.read_csv(args.file)
        return df["ticker"].dropna().astype(str).str.upper().tolist()

    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT ticker FROM companies WHERE COALESCE(delisted,false)=false ORDER BY ticker",
            conn,
        )

    return df["ticker"].tolist()


# ----------------------------------------------------------
# Fetch prices from Massive aggregates
# ----------------------------------------------------------
def fetch_prices_massive(client: RESTClient, ticker: str, start_date: date, pause_sec: float = 0.0) -> list[dict]:
    end_date = get_effective_end_date()

    if start_date > end_date:
        logger.info(f"{ticker}: no new data to import (start_date {start_date} > end_date {end_date})")
        return []

    logger.info(
        f"{ticker}: Massive list_aggs from {start_date.isoformat()} to {end_date.isoformat()}"
    )

    rows: list[dict] = []

    try:
        aggs = client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.isoformat(),
            to=end_date.isoformat(),
            limit=50000,
        )

        for agg in aggs:
            try:
                ts_ms = getattr(agg, "timestamp", None)
                close = getattr(agg, "close", None)
                high = getattr(agg, "high", None)
                low = getattr(agg, "low", None)

                if ts_ms is None or close is None or high is None or low is None:
                    continue

                bar_date = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date()

                rows.append({
                    "ticker": ticker,
                    "date": bar_date,
                    "close": float(close),
                    "high_price": float(high),
                    "low_price": float(low),
                })
            except Exception as e:
                logger.warning(f"{ticker}: skipping malformed aggregate row error={e}")
                continue

    except Exception as e:
        logger.exception(f"{ticker}: Massive fetch failed: {e}")
        return []

    if pause_sec > 0:
        time.sleep(pause_sec)

    if rows:
        returned_dates = [r["date"] for r in rows]
        logger.info(
            f"{ticker}: received {len(rows)} row(s), "
            f"min_date={min(returned_dates)}, max_date={max(returned_dates)}"
        )
    else:
        logger.info(f"{ticker}: no daily bars returned")

    return rows


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch and store historical prices from Massive")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--file")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--pause", type=float, default=12.5, help="Seconds to pause between tickers on free tier")

    args = parser.parse_args()

    api_key = os.getenv("MASSIVE_PRICE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing MASSIVE_PRICE_API_KEY env var")

    logger.info("Using Massive price loader")
    print("Using Massive price loader")

    # trace=True can be useful while testing, but keep it False by default
    client = RESTClient(api_key=api_key)

    tickers = load_tickers(args)
    latest_dates = get_latest_dates()

    logger.info(f"Processing {len(tickers)} tickers")

    total_rows = 0
    no_data_tickers: list[str] = []

    for ticker in tickers:
        try:
            if args.reload or ticker not in latest_dates:
                start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
            else:
                start_date = latest_dates[ticker] + timedelta(days=1)

            logger.info(f"{ticker}: fetching from {start_date}")

            rows = fetch_prices_massive(
                client=client,
                ticker=ticker,
                start_date=start_date,
                pause_sec=args.pause,
            )

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