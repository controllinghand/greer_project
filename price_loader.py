# price_loader.py
import yfinance as yf
import argparse
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text

# Import shared DB connection functions
from db import get_engine, get_psycopg_connection

# Logging Setup
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "price_loader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Get latest dates for all tickers in one query
def get_latest_dates():
    engine = get_engine()
    query = text("SELECT ticker, MAX(date) AS latest_date FROM prices GROUP BY ticker")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return dict(zip(df['ticker'], df['latest_date']))

# Fetch and collect price data for a ticker (for parallel execution)
def fetch_prices_for_ticker(ticker, start_date="2010-01-01", force_reload=False, latest_dates=None):
    # Check delisting status
    engine = get_engine()
    query = text("SELECT delisted, delisted_date FROM companies WHERE ticker = :ticker")
    with engine.connect() as conn:
        result = conn.execute(query, {'ticker': ticker}).fetchone()
        if result and result[0]:  # delisted = TRUE
            delisted_date = result[1] or datetime.now().date()
            logger.info(f"⚠️ Skipping {ticker}: delisted on {delisted_date}")
            print(f"⚠️ Skipping {ticker}: delisted on {delisted_date}")
            return ticker, None, None

    try:
        if force_reload:
            logger.info(f"♻️ Forcing full reload of {ticker} from {start_date}")
            return ticker, yf.download(ticker, start=start_date, auto_adjust=False), None

        latest_date = latest_dates.get(ticker)
        if latest_date:
            start_date = (latest_date + timedelta(days=1)).isoformat()
            logger.info(f"📅 Resuming {ticker} from {start_date}")
        else:
            logger.info(f"📥 Fetching {ticker} from scratch since {start_date}")

        data = yf.download(ticker, start=start_date, auto_adjust=False)

        if data.empty:
            logger.warning(f"⚠️ No data found for {ticker}")
            return ticker, None, None

        # Filter out post-delisting data (if delisted_date exists)
        if result and result[1]:
            data = data[data.index.date <= result[1]]

        return ticker, data, force_reload
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return ticker, None, None

# Bulk insert prices with ON CONFLICT
def bulk_insert_prices(prices_list):
    if not prices_list:
        return

    engine = get_engine()
    query = text("""
        INSERT INTO prices (ticker, date, close, high_price, low_price)
        VALUES (:ticker, :date, :close, :high, :low)
        ON CONFLICT (ticker, date) DO UPDATE
        SET close = EXCLUDED.close,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price;
    """)
    with engine.connect() as conn:
        conn.execute(query, prices_list)
        conn.commit()

# Load tickers from file, CLI list, or fallback to DB
def load_tickers(args):
    if args.file:
        print(f"📄 Loading tickers from file: {args.file}")
        df = pd.read_csv(args.file)
        return df["ticker"].dropna().str.upper().unique().tolist()
    elif args.tickers:
        print(f"📥 Loading tickers from command line: {args.tickers}")
        return [t.upper() for t in args.tickers]
    else:
        print("🗃️ Loading tickers from companies table...")
        engine = get_engine()
        query = text("SELECT ticker FROM companies WHERE delisted = FALSE OR delisted IS NULL ORDER BY ticker")
        with engine.connect() as conn:
            return pd.read_sql(query, conn)["ticker"].tolist()

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and store historical prices.")
    parser.add_argument("--tickers", nargs="+", help="List of tickers (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--file", type=str, help="Path to file with tickers (one per line)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--reload", action="store_true", help="Force full reload from start date")
    parser.add_argument("--max_workers", type=int, default=10, help="Max concurrent workers for parallel processing")
    args = parser.parse_args()

    tickers = load_tickers(args)
    print(f"✅ Loaded {len(tickers)} tickers")

    # Get latest dates in one query
    latest_dates = get_latest_dates()

    # Parallel fetch
    prices_to_insert = []
    reload_tickers = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_prices_for_ticker, ticker, args.start, args.reload, latest_dates): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker, data, force_reload = future.result()
                if data is None:
                    continue

                if force_reload:
                    reload_tickers.append(ticker)

                for date, row in data.iterrows():
                    date_str = date.date()
                    try:
                        close_price = float(row['Close'])
                        high_price = float(row['High'])
                        low_price = float(row['Low'])
                        prices_to_insert.append({
                            'ticker': ticker,
                            'date': date_str,
                            'close': close_price,
                            'high': high_price,
                            'low': low_price
                        })
                    except Exception as e:
                        logger.error(f"Skipping insert for {ticker} on {date_str}: {e}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

    # Batch delete for reload tickers
    if reload_tickers:
        engine = get_engine()
        delete_query = text("DELETE FROM prices WHERE ticker = :ticker")
        with engine.connect() as conn:
            for ticker in reload_tickers:
                conn.execute(delete_query, {'ticker': ticker})
            conn.commit()
        print(f"♻️ Deleted old prices for {len(reload_tickers)} reloaded tickers")

    # Bulk insert all collected prices
    if prices_to_insert:
        bulk_insert_prices(prices_to_insert)
        print(f"✅ Inserted/updated {len(prices_to_insert)} price rows")

    print("✅ All tickers processed")