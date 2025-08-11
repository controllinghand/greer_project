# price_loader.py
import yfinance as yf
import argparse
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Ensure yfinance cache directory exists to avoid potential SQLite issues
os.makedirs(os.path.expanduser('~/.cache/yfinance'), exist_ok=True)

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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Get latest dates for all tickers in one query
# ----------------------------------------------------------
def get_latest_dates():
    try:
        engine = get_engine()
        query = text("SELECT ticker, MAX(date) AS latest_date FROM prices GROUP BY ticker")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return dict(zip(df['ticker'], df['latest_date']))
    except OperationalError as e:
        logger.error(f"Failed to fetch latest dates: {e}. Check PostgreSQL connection (host=localhost, port=5432, db=yfinance_db).")
        raise

# ----------------------------------------------------------
# Fetch price data for a single ticker with retries
# ----------------------------------------------------------
def fetch_single_ticker(ticker, start_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, auto_adjust=False, timeout=30, threads=False)
            if not data.empty:
                data.columns = data.columns.droplevel(1)  # Flatten MultiIndex columns
                # Validate latest date is recent (within 3 days, accounting for weekends)
                latest_date = data.index.max()
                today = datetime.now().date()
                if (today - latest_date.date()).days > 3:
                    logger.warning(f"⚠️ Latest data for {ticker} is {latest_date.date()}, older than expected")
            return data
        except Exception as e:
            logger.warning(f"Retry {attempt+1}/{max_retries} for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)  # Increased backoff
    logger.error(f"Failed to fetch data for {ticker} after {max_retries} attempts")
    return None

# ----------------------------------------------------------
# Fetch and collect price data for a ticker (for non-reload cases)
# ----------------------------------------------------------
def fetch_prices_for_ticker(ticker, start_date="2010-01-01", force_reload=False, latest_dates=None):
    try:
        if force_reload:
            logger.info(f"♻️ Forcing full reload of {ticker} from {start_date}")
            data = fetch_single_ticker(ticker, start_date)
            if data is None or data.empty:
                logger.warning(f"⚠️ No data found for {ticker}")
                return ticker, None, force_reload
            return ticker, data, force_reload

        latest_date = latest_dates.get(ticker)
        if latest_date:
            start_date = (latest_date + timedelta(days=1)).isoformat()
            logger.info(f"📅 Resuming {ticker} from {start_date}")
        else:
            logger.info(f"📥 Fetching {ticker} from scratch since {start_date}")

        data = fetch_single_ticker(ticker, start_date)
        if data is None or data.empty:
            logger.warning(f"⚠️ No data found for {ticker}")
            return ticker, None, None

        return ticker, data, force_reload
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return ticker, None, None

# ----------------------------------------------------------
# Bulk insert prices with ON CONFLICT, with error handling
# ----------------------------------------------------------
def bulk_insert_prices(prices_list):
    if not prices_list:
        logger.info("No prices to insert.")
        return
    logger.info(f"Attempting to insert/update {len(prices_list)} price rows")
    try:
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
        logger.info(f"Successfully inserted/updated {len(prices_list)} price rows")
    except OperationalError as e:
        logger.error(f"Database error during insert: {e}. Check PostgreSQL connection (host=localhost, port=5432, db=yfinance_db).")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during insert: {e}")
        raise

# ----------------------------------------------------------
# Load tickers from file, CLI list, or fallback to DB
# ----------------------------------------------------------
def load_tickers(args):
    try:
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
            with engine.connect() as conn:
                return pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    except Exception as e:
        logger.error(f"Failed to load tickers: {e}")
        raise

# ----------------------------------------------------------
# Main execution block
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and store historical prices.")
    parser.add_argument("--tickers", nargs="+", help="List of tickers (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--file", type=str, help="Path to file with tickers (one per line)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--reload", action="store_true", help="Force full reload from start date")
    parser.add_argument("--max_workers", type=int, default=1, help="Max concurrent workers for non-reload (default: 1)")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for reload downloads (default: 50)")
    args = parser.parse_args()

    try:
        tickers = load_tickers(args)
        print(f"✅ Loaded {len(tickers)} tickers")
    except Exception as e:
        print(f"❌ Failed to load tickers: {e}")
        exit(1)

    # Get latest dates in one query
    try:
        latest_dates = get_latest_dates()
    except Exception as e:
        print(f"❌ Failed to fetch latest dates: {e}")
        exit(1)

    # Parallel fetch or batch download
    prices_to_insert = []
    reload_tickers = []
    if args.reload:
        print("📥 Performing batch download for full reload...")
        # Split tickers into smaller batches to avoid overload/rate limits
        for i in range(0, len(tickers), args.batch_size):
            batch_tickers = tickers[i:i + args.batch_size]
            logger.info(f"Processing batch {i//args.batch_size + 1}: {batch_tickers}")
            prices_to_insert = []  # Reset for each batch
            all_data = None
            for attempt in range(3):
                try:
                    all_data = yf.download(tickers=' '.join(batch_tickers), start=args.start, auto_adjust=False,
                                           timeout=30, threads=False)
                    break
                except Exception as e:
                    logger.warning(f"Batch retry {attempt+1}/3 for batch {i//args.batch_size + 1}: {e}")
                    if attempt < 2:
                        time.sleep(10)
            if all_data is None:
                logger.error(f"Failed to fetch batch {i//args.batch_size + 1} after retries")
                continue
            # Delete old prices for this batch's tickers
            try:
                engine = get_engine()
                delete_query = text("DELETE FROM prices WHERE ticker = :ticker")
                with engine.connect() as conn:
                    for ticker in batch_tickers:
                        conn.execute(delete_query, {'ticker': ticker})
                    conn.commit()
                logger.info(f"Deleted old prices for batch {i//args.batch_size + 1} tickers: {batch_tickers}")
                reload_tickers.extend(batch_tickers)
            except OperationalError as e:
                logger.error(f"Database error during delete for batch {i//args.batch_size + 1}: {e}. Check PostgreSQL connection.")
                raise
            for ticker in batch_tickers:
                try:
                    data = all_data.xs(ticker, level=1, axis=1) if len(batch_tickers) > 1 else all_data
                    if data.empty:
                        logger.warning(f"⚠️ No data found for {ticker}")
                        continue
                    # Validate latest date
                    latest_date = data.index.max()
                    today = datetime.now().date()
                    if (today - latest_date.date()).days > 3:
                        logger.warning(f"⚠️ Latest data for {ticker} is {latest_date.date()}, older than expected")
                    for date, row in data.iterrows():
                        date_str = date.date()
                        try:
                            close_price = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
                            high_price = float(row['High'].iloc[0] if isinstance(row['High'], pd.Series) else row['High'])
                            low_price = float(row['Low'].iloc[0] if isinstance(row['Low'], pd.Series) else row['Low'])
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
                    logger.error(f"Error processing batch data for {ticker}: {e}")
            # Insert prices for this batch
            if prices_to_insert:
                try:
                    bulk_insert_prices(prices_to_insert)
                    print(f"✅ Inserted/updated {len(prices_to_insert)} price rows for batch {i//args.batch_size + 1}")
                except Exception as e:
                    print(f"❌ Failed to insert prices for batch {i//args.batch_size + 1}: {e}")
                    exit(1)
            else:
                print(f"⚠️ No new price rows to insert for batch {i//args.batch_size + 1}")
            time.sleep(5)  # Sleep between batches to avoid rate limits
    else:
        # Sequential fetch for non-reload (different start dates)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_ticker = {
                executor.submit(fetch_prices_for_ticker, ticker, args.start, args.reload, latest_dates): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    _, data, force_reload = future.result()
                    if data is None:
                        continue
                    if force_reload:
                        # Delete old prices for this ticker
                        try:
                            engine = get_engine()
                            delete_query = text("DELETE FROM prices WHERE ticker = :ticker")
                            with engine.connect() as conn:
                                conn.execute(delete_query, {'ticker': ticker})
                                conn.commit()
                            logger.info(f"Deleted old prices for {ticker}")
                            reload_tickers.append(ticker)
                        except OperationalError as e:
                            logger.error(f"Database error during delete for {ticker}: {e}. Check PostgreSQL connection.")
                            raise
                    for date, row in data.iterrows():
                        date_str = date.date()
                        try:
                            close_price = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
                            high_price = float(row['High'].iloc[0] if isinstance(row['High'], pd.Series) else row['High'])
                            low_price = float(row['Low'].iloc[0] if isinstance(row['Low'], pd.Series) else row['Low'])
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
        # Insert all prices for non-reload mode
        if prices_to_insert:
            try:
                bulk_insert_prices(prices_to_insert)
                print(f"✅ Inserted/updated {len(prices_to_insert)} price rows")
            except Exception as e:
                print(f"❌ Failed to insert prices: {e}")
                exit(1)
        else:
            print("⚠️ No new price rows to insert/update")

    print("✅ All tickers processed")