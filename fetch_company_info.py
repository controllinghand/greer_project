# fetch_company_info.py
import os
import time
import argparse
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from datetime import datetime, timedelta
import logging

# Import shared DB connections
from db import get_engine

# Logging Setup
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "fetch_company_info.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Initialize DB Connection
engine = get_engine()

# Valid US exchanges and invalid tickers
VALID_EXCHANGES = {'NYQ', 'NMS', 'NGM', 'NCM'}  # NYSE, NASDAQ
INVALID_TICKERS = {'AMRZ', 'B', 'LTM', 'NBIS', 'SAML', 'TBB'}

# Function: Check if ticker is stale (no price updates for 7 days)
def is_ticker_stale(ticker):
    query = text("""
        SELECT MAX(date) AS last_price_date
        FROM prices
        WHERE ticker = :ticker
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {'ticker': ticker}).fetchone()
        last_price_date = result[0] if result else None
        if not last_price_date:
            return True, None  # No prices, consider stale
        is_stale = (datetime.now().date() - last_price_date).days >= 7
        return is_stale, last_price_date

# Function: Remove foreign ticker from all tables
def remove_foreign_ticker(ticker: str):
    tables = ['companies', 'prices', 'financials', 'greer_buyzone_daily', 'fair_value_gaps', 'greer_opportunity_periods']
    for table in tables:
        query = text(f"DELETE FROM {table} WHERE ticker = :ticker")
        with engine.connect() as conn:
            conn.execute(query, {'ticker': ticker})
            conn.commit()
    print(f"Removed foreign ticker {ticker} from all tables.")
    logger.info(f"Removed foreign ticker {ticker} from all tables.")

# Function: Fetch company info for a ticker using yfinance
def fetch_company_info(ticker: str, retries: int = 3, delay: int = 2):
    # Check if ticker is stale
    is_stale, last_price_date = is_ticker_stale(ticker)
    if is_stale:
        return {
            'ticker': ticker,
            'name': '',
            'sector': '',
            'industry': '',
            'exchange': '',
            'delisted': True,
            'delisted_date': last_price_date or datetime.now().date()
        }

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or 'longName' not in info:
                raise ValueError("No info available")

            name = info.get('longName', '')
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            exchange = info.get('exchange', '')

            if exchange in VALID_EXCHANGES:
                return {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'industry': industry,
                    'exchange': exchange,
                    'delisted': False,
                    'delisted_date': None
                }
            # Remove foreign exchange tickers
            remove_foreign_ticker(ticker)
            raise ValueError(f"Non-US exchange: {exchange}")
        except Exception as e:
            logger.error(f"Error fetching info for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    # If all retries fail, mark as delisted if stale
    is_stale, last_price_date = is_ticker_stale(ticker)
    if is_stale:
        return {
            'ticker': ticker,
            'name': '',
            'sector': '',
            'industry': '',
            'exchange': '',
            'delisted': True,
            'delisted_date': last_price_date or datetime.now().date()
        }
    logger.error(f"Failed to fetch info for {ticker}")
    return None

# Function: Load tickers from file or companies table
def load_tickers(file_path=None):
    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().str.upper().unique().tolist()
    else:
        print("🗃️ Loading tickers from companies table...")
        query = text("""
            SELECT ticker
            FROM companies
            WHERE name IS NULL OR name = ''
               OR sector IS NULL OR sector = ''
               OR industry IS NULL OR industry = ''
               OR exchange IS NULL OR exchange = ''
               OR delisted IS NULL OR delisted = FALSE
            ORDER BY ticker
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        tickers = df["ticker"].tolist()
    tickers = [t for t in tickers if t not in INVALID_TICKERS]
    return tickers

# Function: Process tickers in parallel
def process_tickers(tickers, max_workers=2):
    start_time = time.time()
    new_infos = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_company_info, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                info = future.result()
                if info:
                    status = f"delisted on {info['delisted_date']}" if info['delisted'] else f"{info['name']} on {info['exchange']}"
                    new_infos.append(info)
                    print(f"✅ Collected info for {ticker}: {status}")
                    logger.info(f"Collected info for {ticker}: {status}")
                else:
                    print(f"⚠️ Skipping {ticker}: no data.")
                    logger.info(f"Skipping {ticker}: no data")
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")

    # Bulk update companies table
    if new_infos:
        query = text("""
            UPDATE companies
            SET name = :name,
                sector = :sector,
                industry = :industry,
                exchange = :exchange,
                delisted = :delisted,
                delisted_date = :delisted_date
            WHERE ticker = :ticker
        """)
        with engine.connect() as conn:
            conn.execute(query, new_infos)
            conn.commit()
        print(f"✅ Updated {len(new_infos)} companies in the database.")
        logger.info(f"Updated {len(new_infos)} companies in the database")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch company info from yFinance and update PostgreSQL")
    parser.add_argument("--file", type=str, help="Optional path to CSV file of tickers")
    args = parser.parse_args()

    tickers = load_tickers(args.file)
    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=2)