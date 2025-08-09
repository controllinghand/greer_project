# price_loader.py
# ----------------------------------------------------------
# Fetch and store historical prices into Postgres (streaming inserts)
# Default ticker source: companies table in DB
# Overrides supported: --tickers, --file (CSV with column 'ticker')
# Low-RAM friendly defaults: 2 workers, 2k upsert batches
# ----------------------------------------------------------

import argparse
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import text

# ----------------------------------------------------------
# DB helpers
# ----------------------------------------------------------
from db import get_engine, get_psycopg_connection  # noqa: F401 (kept for project consistency)

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "price_loader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Tunables for low-RAM servers
# ----------------------------------------------------------
MAX_RETRIES = 3
RETRY_BASE = 1.5   # seconds (exponential backoff base)
DEFAULT_BATCH_SIZE = 2_000  # rows per upsert batch


# ----------------------------------------------------------
# Normalize yfinance output to standard columns
# ----------------------------------------------------------
def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns (defensive)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize possible lowercase columns (history() path)
    ren = {"close": "Close", "high": "High", "low": "Low"}
    if any(k in df.columns for k in ren):
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    expected = {"Close", "High", "Low"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns from yfinance output: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    df = df.dropna(subset=list(expected), how="all")
    return df


# ----------------------------------------------------------
# SQL Helpers
# ----------------------------------------------------------
def get_latest_dates() -> dict:
    """
    Get the last loaded date for each ticker from the prices table (one query).
    Returns: {ticker -> date}
    """
    engine = get_engine()
    query = text("SELECT ticker, MAX(date) AS latest_date FROM prices GROUP BY ticker;")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return dict(zip(df["ticker"], df["latest_date"]))


def delete_prices_for_tickers(tickers):
    """Delete all historical prices for a list of tickers (used for --reload)."""
    if not tickers:
        return
    engine = get_engine()
    delete_query = text("DELETE FROM prices WHERE ticker = :ticker;")
    with engine.begin() as conn:
        for t in tickers:
            conn.execute(delete_query, {"ticker": t})


# ----------------------------------------------------------
# Robust downloader with retries and fallback to Ticker().history()
# ----------------------------------------------------------
def yf_download_with_retry(ticker, start_date, end_date):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,  # we control threading outside
            )
            return normalize_price_df(df)
        except Exception as e:
            if attempt == MAX_RETRIES:
                try:
                    t = yf.Ticker(ticker)
                    df = t.history(start=start_date, end=end_date, auto_adjust=False)
                    return normalize_price_df(df)
                except Exception as e2:
                    logger.error(f"Final fallback failed for {ticker}: {e2}")
                    return pd.DataFrame()
            sleep_s = (RETRY_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(f"Retry {attempt}/{MAX_RETRIES} for {ticker} after error: {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)


# ----------------------------------------------------------
# Convert df rows to dicts and upsert in batches for a single ticker
# ----------------------------------------------------------
def insert_prices_chunked_for_ticker(ticker: str, df: pd.DataFrame, batch=DEFAULT_BATCH_SIZE):
    if df is None or df.empty:
        return 0
    engine = get_engine()
    query = text("""
        INSERT INTO prices (ticker, date, close, high_price, low_price)
        VALUES (:ticker, :date, :close, :high, :low)
        ON CONFLICT (ticker, date) DO UPDATE
        SET close = EXCLUDED.close,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price;
    """)
    buf, inserted = [], 0
    with engine.begin() as conn:
        for row in df.itertuples(index=True):
            d = row.Index.date()
            try:
                buf.append({
                    "ticker": ticker,
                    "date": d,
                    "close": float(row.Close),
                    "high": float(row.High),
                    "low": float(row.Low),
                })
                if len(buf) >= batch:
                    conn.execute(query, buf)
                    inserted += len(buf)
                    buf.clear()
            except Exception as e:
                logger.error(f"Skipping {ticker} on {d}: {e}")
        if buf:
            conn.execute(query, buf)
            inserted += len(buf)
    return inserted


# ----------------------------------------------------------
# Download prices for a single ticker (resume unless --reload)
# ----------------------------------------------------------
def fetch_prices_for_ticker(
    ticker: str,
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    force_reload: bool = False,
    latest_dates: dict | None = None,
):
    engine = get_engine()

    # Delisting check
    delisted = False
    delisted_date = None
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT delisted, delisted_date FROM companies WHERE ticker = :t"),
            {"t": ticker},
        ).fetchone()
        if res:
            delisted = bool(res[0]) if res[0] is not None else False
            delisted_date = res[1]

    if delisted:
        dd = delisted_date or datetime.now().date()
        logger.info(f"Skipping {ticker}: delisted on {dd}")
        print(f"⚠️ Skipping {ticker}: delisted on {dd}")
        return ticker, pd.DataFrame()

    # Determine effective start
    if not force_reload and latest_dates:
        last = latest_dates.get(ticker)
        if last:
            start_date = (last + timedelta(days=1)).isoformat()
            logger.info(f"📅 Resuming {ticker} from {start_date}")
        else:
            logger.info(f"📥 Fetching {ticker} from scratch since {start_date}")
    else:
        logger.info(f"♻️ Forcing full reload of {ticker} from {start_date}")

    # Fetch
    df = yf_download_with_retry(ticker, start_date, end_date)
    if df is None or df.empty:
        logger.warning(f"No data returned for {ticker}")
        return ticker, pd.DataFrame()

    if delisted_date is not None:
        df = df[df.index.date <= delisted_date]

    if df.empty:
        logger.warning(f"All data filtered out for {ticker}")
        return ticker, pd.DataFrame()

    return ticker, df


# ----------------------------------------------------------
# Load tickers (DB default; --file and --tickers supported overrides)
# ----------------------------------------------------------
def load_tickers(args) -> list[str]:
    # CSV override
    if args.file:
        print(f"📄 Loading tickers from file: {args.file}")
        df = pd.read_csv(args.file)
        return (
            df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
        )
    # CLI override
    if args.tickers:
        print(f"📥 Loading tickers from command line: {args.tickers}")
        return [t.strip().upper() for t in args.tickers if str(t).strip()]

    # Default: DB companies table
    print("🗃️ Loading tickers from companies table (DB default)…")
    engine = get_engine()
    query = text(
        "SELECT ticker FROM companies WHERE delisted = FALSE OR delisted IS NULL ORDER BY ticker;"
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn)["ticker"].tolist()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch and store historical prices (DB default; overrides supported).")
    parser.add_argument("--tickers", nargs="+", help="List of tickers (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--file", type=str, help="Path to CSV with column 'ticker' (override)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--reload", action="store_true", help="Force full reload from start date for each ticker")
    parser.add_argument("--max_workers", type=int, default=2, help="Max concurrent workers (default=2)")
    args = parser.parse_args()

    tickers = load_tickers(args)
    print(f"✅ Loaded {len(tickers)} tickers")

    # Resume map only when not reloading
    latest_dates = {} if args.reload else get_latest_dates()

    processed = 0
    total_inserted = 0
    deleted_for = set()  # avoid double-deleting a ticker

    with ThreadPoolExecutor(max_workers=min(args.max_workers or 2, 8)) as executor:
        futures = {
            executor.submit(
                fetch_prices_for_ticker,
                ticker,
                args.start,
                args.end,
                args.reload,
                latest_dates,
            ): ticker
            for ticker in tickers
        }

        for fut in as_completed(futures):
            tkr = futures[fut]
            try:
                tkr, df = fut.result()
                processed += 1

                if df is None or df.empty:
                    if processed % 50 == 0:
                        print(f"… {processed}/{len(tickers)} processed")
                    continue

                # If --reload, delete this ticker's history just-in-time
                if args.reload and tkr not in deleted_for:
                    delete_prices_for_tickers([tkr])
                    deleted_for.add(tkr)

                inserted = insert_prices_chunked_for_ticker(tkr, df, batch=DEFAULT_BATCH_SIZE)
                total_inserted += inserted
                print(
                    f"✅ {tkr}: inserted/upserted {inserted} rows "
                    f"({df.index.min().date()} → {df.index.max().date()})"
                )

                if processed % 50 == 0:
                    print(f"📈 Progress: {processed}/{len(tickers)} tickers, {total_inserted} rows total")

            except Exception as e:
                logger.error(f"Error processing {tkr}: {e}")

    print(f"✅ All tickers processed. Total rows upserted: {total_inserted}")


if __name__ == "__main__":
    main()
