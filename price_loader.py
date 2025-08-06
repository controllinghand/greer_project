# price_loader.py

import yfinance as yf
import argparse
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# ----------------------------------------------------------
# Import shared DB connection functions
# ----------------------------------------------------------
from db import get_engine, get_psycopg_connection

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "price_loader.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Get the latest stored date for a ticker
# ----------------------------------------------------------
def get_latest_date(ticker):
    try:
        with get_psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(date) FROM prices WHERE ticker = %s;", (ticker,))
                result = cur.fetchone()
                return result[0]  # None if no data exists
    except Exception as e:
        logger.error(f"Error getting latest date for {ticker}: {e}")
        return None

# ----------------------------------------------------------
# Insert a price row (close, high, low) into the database
# ----------------------------------------------------------
def insert_price(ticker, date, close, high, low):
    try:
        with get_psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prices (ticker, date, close, high_price, low_price)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE
                    SET close = EXCLUDED.close,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price;
                """, (ticker, date, close, high, low))
        print(f"Inserting: {ticker} | {date} | C: {close} | H: {high} | L: {low}")
    except Exception as e:
        logger.error(f"Error inserting price for {ticker} on {date}: {e}")

# ----------------------------------------------------------
# Fetch and store historical price data
# ----------------------------------------------------------
def fetch_and_store_prices(ticker, start_date="2010-01-01", force_reload=False):
    try:
        if force_reload:
            print(f"♻️ Forcing full reload of {ticker} from {start_date}")
            with get_psycopg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM prices WHERE ticker = %s;", (ticker,))
        else:
            latest_date = get_latest_date(ticker)
            if latest_date:
                start_date = (latest_date + timedelta(days=1)).isoformat()
                print(f"📅 Resuming {ticker} from {start_date}")
            else:
                print(f"📥 Fetching {ticker} from scratch since {start_date}")

        data = yf.download(ticker, start=start_date, auto_adjust=False)

        if data.empty:
            print(f"⚠️ No data found for {ticker}")
            return

        for date, row in data.iterrows():
            date_str = date.date() if isinstance(date, datetime) else date
            try:
                close_price = float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close'])
                high_price = float(row['High'].item() if hasattr(row['High'], "item") else row['High'])
                low_price = float(row['Low'].item() if hasattr(row['Low'], "item") else row['Low'])

                insert_price(ticker, date_str, close_price, high_price, low_price)
            except Exception as e:
                print(f"⚠️ Skipping insert for {ticker} on {date_str} due to error: {e}")
                logger.error(f"Skipping insert for {ticker} on {date_str}: {e}")

        print(f"✅ Done storing {ticker} closing prices.")

    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")

# ----------------------------------------------------------
# Load tickers from file, CLI list, or fallback to DB
# ----------------------------------------------------------
def load_tickers(args):
    if args.file:
        print(f"📄 Loading tickers from file: {args.file}")
        df = pd.read_csv(args.file)
        return df["ticker"].dropna().str.upper().unique().tolist()
    elif args.tickers:
        print(f"📥 Loading tickers from command line: {args.tickers}")
        return [t.upper() for t in args.tickers]
    else:
        print("🗃️  Loading tickers from companies table...")
        engine = get_engine()
        with engine.begin() as conn:
            return pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()

# ----------------------------------------------------------
# Main execution block
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and store historical prices.")
    parser.add_argument("--tickers", nargs="+", help="List of tickers (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--file", type=str, help="Path to file with tickers (one per line)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--reload", action="store_true", help="Force full reload from start date")
    args = parser.parse_args()

    tickers = load_tickers(args)
    print(f"✅ Loaded {len(tickers)} tickers")

    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        fetch_and_store_prices(
            ticker=ticker,
            start_date=args.start,
            force_reload=args.reload
        )
