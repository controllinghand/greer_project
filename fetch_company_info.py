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
# Logging Setup (file only)
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
INVALID_TICKERS = {'AMRZ', 'B', 'LTM', 'NBIS', 'SAML', 'TBB'}
INACTIVITY_BUSINESS_DAYS = 10  # default cutoff

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def parse_tickers_arg(raw: str | None) -> list[str]:
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
    q = text("SELECT MAX(date) AS last_price_date FROM prices WHERE ticker = :t")
    with engine.connect() as conn:
        row = conn.execute(q, {"t": ticker}).fetchone()
    last_price_date = row[0] if row else None
    if not last_price_date:
        return False, None

    today = pd.Timestamp.today().normalize().date()
    bdays_without_prices = pd.bdate_range(
        pd.Timestamp(last_price_date) + pd.Timedelta(days=1),
        pd.Timestamp(today)
    ).size
    return (bdays_without_prices >= max_business_days), last_price_date

def set_delisted(ticker: str, delisted: bool, when=None):
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

def _has_recent_pricing(stock: yf.Ticker) -> bool:
    try:
        hist = stock.history(period="5d", auto_adjust=False)
        if hist is not None and not hist.empty:
            return True
        hist = stock.history(period="1mo", auto_adjust=False)
        return hist is not None and not hist.empty
    except Exception as e:
        logger.error(f"_has_recent_pricing error: {e}")
        return False

def _has_company_info(info: dict) -> bool:
    if not info:
        return False
    return bool(info.get("longName") or info.get("shortName"))

def _normalize_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = df.index.astype(str).str.strip().str.lower()
    return df

def _has_financials(stock: yf.Ticker) -> bool:
    try:
        inc = stock.get_income_stmt(freq="yearly")
        if inc is None or inc.empty:
            inc = stock.get_income_stmt(freq="trailing")
        if inc is None or inc.empty:
            inc = getattr(stock, "income_stmt", pd.DataFrame())
        inc = _normalize_index(inc)
        return inc is not None and not inc.empty and len(inc.columns) > 0
    except Exception as e:
        logger.error(f"_has_financials error: {e}")
        return False

# ----------------------------------------------------------
# Fetch company info
# ----------------------------------------------------------
def fetch_company_info(ticker: str, retries: int = 3, delay: int = 2, verbose: bool = False, force: bool = False):
    inactive, last_price_date = is_ticker_inactive_by_prices(ticker)
    if inactive:
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

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            try:
                info = stock.info
            except Exception:
                info = {}

            name = (info.get('longName') or info.get('shortName') or '') if info else ''
            sector = info.get('sector', '') if info else ''
            industry = info.get('industry', '') if info else ''
            exchange = info.get('exchange', '') if info else ''

            info_ok = _has_company_info(info)
            pricing_ok = _has_recent_pricing(stock)
            financials_ok = _has_financials(stock)

            if verbose:
                print(f"[{ticker}] info_ok={info_ok} pricing_ok={pricing_ok} financials_ok={financials_ok} "
                      f"name='{name}' sector='{sector}' industry='{industry}' exchange='{exchange}'")

            if not (info_ok and pricing_ok and financials_ok):
                if force and info_ok:
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
                else:
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    return None

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

        except Exception as e:
            logger.error(f"Error fetching info for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Failed to fetch info for {ticker}")
    return None

# ----------------------------------------------------------
# Load tickers
# ----------------------------------------------------------
def load_tickers(file_path: str | None = None,
                 explicit_tickers: list[str] | None = None,
                 reload_all: bool = False):
    if explicit_tickers:
        print(f"🎯 Using explicit tickers: {', '.join(explicit_tickers)}")
        tickers = explicit_tickers
    elif file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    else:
        with engine.connect() as conn:
            if reload_all:
                print("🗃️ Reload mode: loading ALL tickers from companies…")
                df = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker;", conn)
            else:
                print("🗃️ Loading tickers with missing company fields…")
                df = pd.read_sql(
                    """
                    SELECT ticker
                    FROM companies
                    WHERE name IS NULL OR name = ''
                       OR sector IS NULL OR sector = ''
                       OR industry IS NULL OR industry = ''
                       OR exchange IS NULL OR exchange = ''
                       OR delisted IS NULL
                    ORDER BY ticker
                    """,
                    conn,
                )
        tickers = df["ticker"].astype(str).str.upper().tolist()

    tickers = [t for t in tickers if t not in INVALID_TICKERS]
    return tickers

# ----------------------------------------------------------
# Process tickers
# ----------------------------------------------------------
def process_tickers(tickers, max_workers=2, verbose=False, force=False):
    start_time = time.time()
    new_infos = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_company_info, ticker, verbose=verbose, force=force): ticker for ticker in tickers}
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
                    print(f"⚠️ Skipping {ticker}: requirements not met (pricing/info/financials).")
                    logger.info(f"Skipping {ticker}: gating failed or no data.")
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")

    if new_infos:
        q = text("""
            INSERT INTO companies (ticker, name, sector, industry, exchange, delisted, delisted_date, added_at)
            VALUES (:ticker, :name, :sector, :industry, :exchange, :delisted, :delisted_date, NOW())
            ON CONFLICT (ticker) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                exchange = EXCLUDED.exchange,
                delisted = EXCLUDED.delisted,
                delisted_date = EXCLUDED.delisted_date,
                added_at = COALESCE(companies.added_at, EXCLUDED.added_at)
        """)
        with engine.begin() as conn:
            conn.execute(q, new_infos)
        print(f"✅ Upserted {len(new_infos)} companies in the database.")
        logger.info(f"Upserted {len(new_infos)} companies in the database")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")
    logger.info(f"Total processing time: {elapsed:.2f} seconds")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch company info from yfinance and update PostgreSQL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file of tickers")
    group.add_argument("--tickers", type=str, help='Optional comma/space separated tickers, e.g. "AAPL,MSFT TSLA"')
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--reload", action="store_true", help="Reload company info for ALL tickers (or only --tickers if provided).")
    parser.add_argument("--inactive-days", type=int, default=INACTIVITY_BUSINESS_DAYS,
                        help=f"Business days without DB prices to mark delisted (default: {INACTIVITY_BUSINESS_DAYS}).")
    parser.add_argument("--verbose", action="store_true", help="Print gating and extracted fields for each ticker.")
    parser.add_argument("--force", action="store_true", help="Upsert even if gating fails (use when yfinance is flaky).")

    args = parser.parse_args()

    explicit = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit, reload_all=args.reload)

    INACTIVITY_BUSINESS_DAYS = int(args.inactive_days)

    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=args.workers, verbose=args.verbose, force=args.force)
