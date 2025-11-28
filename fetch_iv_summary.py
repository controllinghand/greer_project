# fetch_iv_summary.py

import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
import logging
import threading

from db import get_engine

# ----------------------------------------------------------
# Logging Setup (file only; no console handler)
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers[:] = []  # avoid duplicate handlers if reloaded

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(log_dir, "fetch_iv_summary.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# ----------------------------------------------------------
# Initialize DB Connection (global engine)
engine = get_engine()

# ----------------------------------------------------------
# Helper functions
def load_tickers(file_path: str | None) -> list[str]:
    if file_path:
        logger.info(f"Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
            .tolist()
        )
        logger.info(f"Loaded {len(tickers)} tickers from file")
        return tickers

    logger.info("Loading tickers from companies table…")
    with engine.connect() as conn:
        tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    tickers = [t.strip().upper() for t in tickers]
    logger.info(f"Loaded {len(tickers)} tickers from companies table")
    return tickers

def run_with_timeout(func, args=(), kwargs=None, timeout_sec=60):
    if kwargs is None:
        kwargs = {}
    result = {}
    exc = {}

    def target():
        try:
            result['value'] = func(*args, **kwargs)
        except Exception as e:
            exc['error'] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_sec} seconds")
    if 'error' in exc:
        raise exc['error']
    return result.get('value')

def fetch_iv_summary_for_ticker(
    ticker: str,
    max_retries: int = 3,
    initial_backoff: float = 30.0,
    timeout_sec: float = 60.0
) -> dict | None:
    today = pd.Timestamp.utcnow().date()
    with engine.connect() as conn:
        exists_row = conn.execute(
            text("SELECT 1 FROM iv_summary WHERE ticker = :ticker AND fetch_date = :today"),
            {"ticker": ticker, "today": today}
        ).fetchone()
    if exists_row:
        logger.info(f"{ticker}: already processed for today ({today}) — skipping.")
        return None

    logger.info(f"{ticker}: starting fetch.")
    stock = yf.Ticker(ticker)
    expiries = stock.options
    if not expiries:
        logger.warning(f"{ticker}: no option expirations found")
        return None

    expiry = expiries[0]
    logger.info(f"{ticker}: will use expiry date {expiry}.")
    attempt = 0
    backoff = initial_backoff

    while attempt < max_retries:
        logger.info(f"{ticker}: attempt {attempt+1}/{max_retries}. Timeout per attempt = {timeout_sec}s.")
        try:
            opt = run_with_timeout(stock.option_chain, args=(expiry,), timeout_sec=timeout_sec)
            logger.info(f"{ticker}: option_chain returned successfully on attempt {attempt+1}.")
            break
        except TimeoutError as te:
            logger.warning(f"{ticker}: fetch timed out on attempt {attempt+1}/{max_retries}; sleeping {backoff}s before retry.")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2
            continue
        except Exception as e:
            err_str = str(e)
            logger.error(f"{ticker}: exception when fetching option chain: {err_str}")
            if ("Too Many Requests" in err_str) or ("Rate limited" in err_str) or ("429" in err_str) or ("YFRateLimitError" in err_str):
                logger.warning(f"{ticker}: detected rate-limit/block on attempt {attempt+1}/{max_retries}. Sleeping {backoff}s before retry.")
                time.sleep(backoff)
                attempt += 1
                backoff *= 2
                continue
            else:
                logger.error(f"{ticker}: non-rate-limit/non-timeout error fetching option chain for expiry {expiry}: {e}")
                return None
    else:
        logger.error(f"{ticker}: exceeded max retries ({max_retries}) due to timeout or rate-limit; skipping ticker.")
        return None

    # Using calls only for ATM call premium
    calls = opt.calls.dropna(subset=["impliedVolatility"])
    if calls.empty:
        logger.warning(f"{ticker}: no call option data for expiry {expiry}")
        return None

    stats = calls["impliedVolatility"].describe()
    logger.info(f"{ticker}: impliedVolatility stats computed.")

    price = stock.info.get("regularMarketPrice", None)
    atm_iv = None
    atm_premium = None
    atm_premium_pct = None

    if price is not None:
        calls["dist"] = (calls["strike"] - price).abs()
        atm_call = calls.loc[calls["dist"].idxmin()]

        atm_iv = float(atm_call["impliedVolatility"])
        # try to get lastPrice (premium)
        last_price = atm_call.get("lastPrice", None)
        if last_price is not None and not np.isnan(last_price):
            atm_premium = float(last_price)
            # premium as percent of underlying price
            atm_premium_pct = atm_premium / price
            logger.info(f"{ticker}: ATM strike {atm_call['strike']}, iv_atm = {atm_iv:.6f}, premium = {atm_premium:.2f}, premium_pct = {atm_premium_pct:.4f}")
        else:
            logger.warning(f"{ticker}: ATM call lastPrice not available (None or NaN).")

    else:
        logger.warning(f"{ticker}: could not retrieve regularMarketPrice for ATM calculation.")

    result = {
        "ticker": ticker,
        "fetch_date": today,
        "expiry": pd.to_datetime(expiry).date(),
        "contract_count": int(stats["count"]),
        "iv_mean": float(stats["mean"]),
        "iv_std": float(stats["std"]),
        "iv_min": float(stats["min"]),
        "iv_25": float(stats["25%"]),
        "iv_median": float(stats["50%"]),
        "iv_75": float(stats["75%"]),
        "iv_max": float(stats["max"]),
        "iv_atm": atm_iv,
        "atm_premium": atm_premium,
        "atm_premium_pct": atm_premium_pct
    }

    logger.info(f"{ticker}: fetched IV summary for expiry {expiry}: {result}")
    return result

def process_tickers(tickers: list[str], max_workers: int = 3, delay_between: float = 2.0):
    start_time = time.time()
    summaries = []

    logger.info(f"process_tickers: starting with {len(tickers)} tickers, workers={max_workers}, delay_between={delay_between}s.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_iv_summary_for_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                res = future.result()
                if res:
                    summaries.append(res)
                    logger.info(f"{ticker}: summary collected successfully.")
                    logger.info(f"{ticker}: waiting {delay_between}s before next ticker.")
                    time.sleep(delay_between)
                else:
                    logger.info(f"{ticker}: summary returned None (skipped or error). No wait.")
            except Exception as e:
                logger.error(f"{ticker}: unexpected error in future result: {e}")
                logger.info(f"{ticker}: waiting {delay_between}s before next ticker.")
                time.sleep(delay_between)

    elapsed = time.time() - start_time
    logger.info(f"process_tickers: completed runs in {elapsed:.2f}s; collected {len(summaries)} summaries.")

    if not summaries:
        logger.info("No summaries to insert. Ending process_tickers without DB insert.")
        return

    # ----------------------------------------------------------
    # IMPORTANT: Make sure iv_summary table has atm_premium (numeric) and atm_premium_pct (numeric) columns!
    insert_query = text("""
        INSERT INTO iv_summary (
            ticker, fetch_date, expiry,
            contract_count, iv_mean, iv_std,
            iv_min, iv_25, iv_median, iv_75,
            iv_max, iv_atm, atm_premium, atm_premium_pct
        )
        VALUES (
            :ticker, :fetch_date, :expiry,
            :contract_count, :iv_mean, :iv_std,
            :iv_min, :iv_25, :iv_median, :iv_75,
            :iv_max, :iv_atm, :atm_premium, :atm_premium_pct
        )
        ON CONFLICT (ticker, fetch_date, expiry) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(insert_query, summaries)
        logger.info(f"Inserted {len(summaries)} records into iv_summary table.")

    elapsed2 = time.time() - start_time
    logger.info(f"Completed DB insert in {elapsed2:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch option-chain IV summary and store in DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV with tickers (expects 'ticker' column)")
    group.add_argument("--tickers", nargs="+", type=str,
                       help='List of tickers (e.g., TSLA AAPL MSFT)')
    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between each ticker submission")
    parser.add_argument("--retries", type=int, default=3, help="Max retries on timeout/rate limit errors")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout seconds for each option-chain fetch")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of tickers per batch")
    parser.add_argument("--pause_between_batches", type=float, default=30.0, help="Pause seconds between batches")
    args = parser.parse_args()

    if args.tickers:
        tickers_list = [t.strip().upper() for t in args.tickers]
        logger.info(f"Using explicit tickers: {tickers_list}")
    elif args.file:
        tickers_list = load_tickers(args.file)
    else:
        tickers_list = load_tickers(None)

    logger.info(f"Starting IV summary fetch for {len(tickers_list)} tickers with {args.workers} workers, delay {args.delay}s, retries {args.retries}, timeout {args.timeout}s, batch_size {args.batch_size}, pause_between_batches {args.pause_between_batches}s.")

    for i in range(0, len(tickers_list), args.batch_size):
        batch = tickers_list[i : i + args.batch_size]
        logger.info(f"Processing batch {i//args.batch_size + 1} (tickers {batch})")
        process_tickers(batch, max_workers=args.workers, delay_between=args.delay)
        if i + args.batch_size < len(tickers_list):
            logger.info(f"Batch {i//args.batch_size + 1} done — sleeping {args.pause_between_batches}s before next batch.")
            time.sleep(args.pause_between_batches)

    logger.info("Finished IV summary fetch.")
