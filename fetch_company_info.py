# fetch_company_info.py
import os
import time
import argparse
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from datetime import datetime
import logging

# Import shared DB connections
from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "fetch_company_info.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Initialize DB Connection
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
# Valid US exchanges and invalid tickers
VALID_EXCHANGES = {'NYQ', 'NMS', 'NGM', 'NCM'}  # NYSE/NASDAQ variants from yfinance
INVALID_TICKERS = {'AMRZ', 'B', 'LTM', 'NBIS', 'SAML', 'TBB'}
INACTIVITY_BUSINESS_DAYS = 10  # how many *business* days without prices => delisted

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def parse_tickers_arg(raw: str | None) -> list[str]:
    """
    Accepts comma and/or whitespace separated tickers.
    Example: "AAPL, MSFT TSLA" -> ["AAPL","MSFT","TSLA"]
    """
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def is_ticker_inactive_by_prices(ticker: str, max_business_days: int = INACTIVITY_BUSINESS_DAYS) -> tuple[bool, datetime | None]:
    """
    Returns (inactive: bool, last_price_date: date|None).
    Inactive means there have been no prices for >= `max_business_days` *business* days.
    If the ticker has never had prices (last_price_date is None), treat as *not inactive* (new listing).
    """
    q = text("SELECT MAX(date) AS last_price_date FROM prices WHERE ticker = :t")
    with engine.connect() as conn:
        row = conn.execute(q, {"t": ticker}).fetchone()
    last_price_date = row[0] if row else None
    if not last_price_date:
        # New/never-priced: don't auto-mark delisted
        return False, None

    today = pd.Timestamp.today().normalize().date()
    # Business days strictly after last_price_date up to today
    bdays_without_prices = pd.bdate_range(
        pd.Timestamp(last_price_date) + pd.Timedelta(days=1),
        pd.Timestamp(today)
    ).size
    return (bdays_without_prices >= max_business_days), last_price_date

def set_delisted(ticker: str, delisted: bool, when=None):
    """
    Update companies.delisted (+ delisted_date when delisted).
    When clearing delisted, delisted_date is reset to NULL.
    """
    q = text("""
        UPDATE companies
        SET delisted = :d,
            delisted_date = CASE
                WHEN :d THEN COALESCE(delisted_date, :w)
                ELSE NULL
            END
        WHERE ticker = :t
    """)
    with engine.begin() as conn:
        conn.execute(q, {"d": delisted, "w": when, "t": ticker})

# Function: Remove foreign ticker from all tables
def remove_foreign_ticker(ticker: str):
    tables = ['companies', 'prices', 'financials', 'greer_buyzone_daily', 'fair_value_gaps', 'greer_opportunity_periods']
    for table in tables:
        q = text(f"DELETE FROM {table} WHERE ticker = :ticker")
        with engine.begin() as conn:
            conn.execute(q, {'ticker': ticker})
    print(f"Removed foreign ticker {ticker} from all tables.")
    logger.info(f"Removed foreign ticker {ticker} from all tables.")

# Function: Fetch company info for a ticker using yfinance
def fetch_company_info(ticker: str, retries: int = 3, delay: int = 2):
    # 1) Price-based inactivity (business days)
    inactive, last_price_date = is_ticker_inactive_by_prices(ticker)
    if inactive:
        # Auto-mark delisted based on inactivity
        set_delisted(ticker, True, last_price_date or datetime.now().date())
        return {
            'ticker': ticker,
            'name': '',
            'sector': '',
            'industry': '',
            'exchange': '',
            'delisted': True,
            'delisted_date': last_price_date or datetime.now().date()
        }

    # 2) Try yfinance lookups
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            # yfinance info access can vary by version; keep your existing approach
            info = stock.info
            if not info or 'longName' not in info:
                raise ValueError("No info available")

            name = info.get('longName', '') or info.get('shortName', '')
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            exchange = info.get('exchange', '')

            if exchange in VALID_EXCHANGES:
                # If prices are flowing / valid exchange, ensure not marked delisted
                set_delisted(ticker, False, None)
                return {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'industry': industry,
                    'exchange': exchange,
                    'delisted': False,
                    'delisted_date': None
                }

            # Non-US exchange: purge from your tables
            remove_foreign_ticker(ticker)
            raise ValueError(f"Non-US exchange: {exchange}")

        except Exception as e:
            logger.error(f"Error fetching info for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    # 3) Final fallback after retries: if now inactive by prices, mark delisted; else return None
    inactive_end, last_price_date_end = is_ticker_inactive_by_prices(ticker)
    if inactive_end:
        set_delisted(ticker, True, last_price_date_end or datetime.now().date())
        return {
            'ticker': ticker,
            'name': '',
            'sector': '',
            'industry': '',
            'exchange': '',
            'delisted': True,
            'delisted_date': last_price_date_end or datetime.now().date()
        }

    logger.error(f"Failed to fetch info for {ticker}")
    return None

# Function: Load tickers from --tickers, file, or companies table (in that order)
def load_tickers(file_path: str | None = None, explicit_tickers: list[str] | None = None):
    if explicit_tickers:
        print(f"🎯 Using explicit tickers: {', '.join(explicit_tickers)}")
        tickers = explicit_tickers
    elif file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    else:
        print("🗃️ Loading tickers from companies table...")
        q = text("""
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
            df = pd.read_sql(q, conn)
        tickers = df["ticker"].tolist()

    # Filter invalids
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
                    status = (
                        f"delisted on {info['delisted_date']}"
                        if info['delisted']
                        else f"{info['name']} on {info['exchange']}"
                    )
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
        q = text("""
            UPDATE companies
            SET name = :name,
                sector = :sector,
                industry = :industry,
                exchange = :exchange,
                delisted = :delisted,
                delisted_date = :delisted_date
            WHERE ticker = :ticker
        """)
        with engine.begin() as conn:
            conn.execute(q, new_infos)
        print(f"✅ Updated {len(new_infos)} companies in the database.")
        logger.info(f"Updated {len(new_infos)} companies in the database")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")
    logger.info(f"Total processing time: {elapsed:.2f} seconds")

# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch company info from yfinance and update PostgreSQL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file of tickers")
    group.add_argument("--tickers", type=str, help='Optional comma/space separated tickers, e.g. "AAPL,MSFT TSLA"')
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    args = parser.parse_args()

    explicit = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit)

    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=args.workers)
