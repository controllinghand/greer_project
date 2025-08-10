# price_loader.py
# ----------------------------------------------------------
# Fetch and store historical prices into Postgres (streaming inserts)
# Default ticker source: companies table in DB
# Overrides supported: --tickers, --file (CSV with column 'ticker')
# Low-RAM friendly defaults: 2 workers, 2k upsert batches
# Adds: anomaly detection, split-aware jump filter, single-day re-fetch,
#       optional yfinance server-side repair, and --scrub mode to fix DB rows
# ----------------------------------------------------------

import argparse
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date

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
# Anomaly thresholds
# ----------------------------------------------------------
# Day-over-day change threshold beyond which we treat as suspect (abs %)
DEFAULT_JUMP_PCT = 0.25  # 25%
# Agreement tolerance when comparing two sources for the same day
AGREE_TOL = 0.01  # 1%


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
# Load split dates for a ticker (used to exempt legitimate jumps)
# ----------------------------------------------------------
def load_splits_map(ticker: str) -> set[date]:
    try:
        t = yf.Ticker(ticker)
        splits = t.splits
        if splits is None or len(splits) == 0:
            return set()
        return {pd.Timestamp(ts).date() for ts in splits.index}
    except Exception as e:
        logger.warning(f"{ticker}: failed to load splits: {e}")
        return set()


# ----------------------------------------------------------
# Flag anomalies (H/L/C rules + big jump vs prior close, excluding split days)
# ----------------------------------------------------------
def flag_price_anomalies(ticker: str, df: pd.DataFrame, pct_jump: float = DEFAULT_JUMP_PCT) -> pd.Index:
    if df.empty:
        return df.index[:0]

    bad = pd.Index([])

    # Structural checks
    struct_bad = df[(df["Low"] > df["High"]) | (df["Close"] < df["Low"]) | (df["Close"] > df["High"])].index
    bad = bad.union(struct_bad)

    # Big jump checks
    splits = load_splits_map(ticker)
    s = df["Close"].copy()
    prev = s.shift(1).ffill()

    # Explicitly handle inf/-inf instead of using deprecated pd.option_context
    jump = (s / prev - 1.0)
    jump = jump.replace([float("inf"), float("-inf")], pd.NA).abs()

    # Mask out the very first point (no previous)
    if len(jump) > 0:
        jump.iloc[0] = 0.0

    big = df[(jump > pct_jump) & (~df.index.date.astype("O").isin(splits))].index
    bad = bad.union(big)

    return bad



# ----------------------------------------------------------
# Re-fetch a small window around a single suspect day from 2 sources
# ----------------------------------------------------------
def refetch_single_day(ticker: str, day: pd.Timestamp, repair: bool = True) -> pd.DataFrame:
    start = (day - pd.Timedelta(days=2)).date().isoformat()
    end = (day + pd.Timedelta(days=2)).date().isoformat()

    # Source A: download()
    try:
        a = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
            repair=repair,           # server-side repair if available
            raise_errors=False,      # don't raise, we'll handle empties
        )
        a = normalize_price_df(a)
    except Exception:
        a = pd.DataFrame()

    # Source B: Ticker().history()
    try:
        b = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        b = normalize_price_df(b)
    except Exception:
        b = pd.DataFrame()

    if a.empty and b.empty:
        return pd.DataFrame()

    # Prefer a row only if both sources for 'day' agree within tolerance
    rows = []
    if (not a.empty) and (day in a.index):
        rows.append(a.loc[[day]])
    if (not b.empty) and (day in b.index):
        rows.append(b.loc[[day]])

    if len(rows) == 2:
        aa, bb = rows
        try:
            agree = (abs(aa["Close"].iloc[0] / bb["Close"].iloc[0] - 1) < AGREE_TOL)
        except Exception:
            agree = False
        return aa if agree else pd.DataFrame()

    return rows[0] if rows else pd.DataFrame()


# ----------------------------------------------------------
# Drop or auto-repair anomalies, logging what happened
# ----------------------------------------------------------
def drop_or_repair_anomalies(ticker: str, df: pd.DataFrame, auto_repair: bool = True, repair_flag: bool = True) -> pd.DataFrame:
    bad_idx = flag_price_anomalies(ticker, df)
    if not len(bad_idx):
        return df

    fixed = []
    dropped = []
    for day in bad_idx:
        if not auto_repair:
            dropped.append(day)
            continue
        repaired = refetch_single_day(ticker, pd.Timestamp(day), repair=repair_flag)
        if repaired.empty:
            dropped.append(day)
        else:
            fixed.append((day, repaired))

    if fixed:
        # Replace bad with repaired
        for day, rep in fixed:
            df.loc[day, ["Close", "High", "Low"]] = rep.loc[day, ["Close", "High", "Low"]].values
    if dropped:
        df = df.drop(index=pd.Index(dropped))

    if fixed:
        logger.warning(f"{ticker}: repaired {len(fixed)} suspect rows: {[pd.Timestamp(d).date().isoformat() for d, _ in fixed]}")
    if dropped:
        logger.warning(f"{ticker}: dropped {len(dropped)} suspect rows: {[pd.Timestamp(d).date().isoformat() for d in dropped]}")

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
def yf_download_with_retry(ticker, start_date, end_date, repair: bool = True):
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
                repair=repair,
                raise_errors=False,
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
            d = pd.Timestamp(row.Index).date()
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
    auto_repair: bool = True,
    repair_flag: bool = True,
):
    # Delisting check
    engine = get_engine()
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
    df = yf_download_with_retry(ticker, start_date, end_date, repair=repair_flag)
    if df is None or df.empty:
        logger.warning(f"No data returned for {ticker}")
        return ticker, pd.DataFrame()

    if delisted_date is not None:
        df = df[df.index.date <= delisted_date]

    if df.empty:
        logger.warning(f"All data filtered out for {ticker}")
        return ticker, pd.DataFrame()

    # Drop/repair anomalies before returning
    df = drop_or_repair_anomalies(ticker, df, auto_repair=auto_repair, repair_flag=repair_flag)
    if df.empty:
        logger.warning(f"All rows for {ticker} were anomalous and removed")
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
        return [str(t).strip().upper() for t in args.tickers if str(t).strip()]

    # Default: DB companies table
    print("🗃️ Loading tickers from companies table (DB default)…")
    engine = get_engine()
    query = text(
        "SELECT ticker FROM companies WHERE delisted = FALSE OR delisted IS NULL ORDER BY ticker;"
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn)["ticker"].tolist()


# ----------------------------------------------------------
# Scrub existing DB rows for anomalies and auto-repair
# ----------------------------------------------------------
def find_anomalies_in_db(ticker: str | None, start: str | None, end: str | None, pct_jump: float = DEFAULT_JUMP_PCT) -> pd.DataFrame:
    """
    Returns rows (ticker, date) that look suspicious based on > pct_jump day-over-day change.
    If ticker is None, scans all tickers; otherwise limits to that ticker.
    Optional start/end (YYYY-MM-DD) window.
    """
    engine = get_engine()
    where = []
    params = {}

    if ticker:
        where.append("p.ticker = :t")
        params["t"] = ticker
    if start:
        where.append("p.date >= :start")
        params["start"] = start
    if end:
        where.append("p.date <= :end")
        params["end"] = end

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        WITH ranked AS (
          SELECT
            p.ticker,
            p.date,
            p.close,
            LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.date) AS prev_close
          FROM prices p
          {where_sql}
        )
        SELECT ticker, date, close, prev_close
        FROM ranked
        WHERE prev_close IS NOT NULL
          AND ABS(close/prev_close - 1) > :pct
        ORDER BY date DESC, ticker;
    """
    params["pct"] = pct_jump
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df


def scrub_db_anomalies(tickers: list[str] | None, start: str | None, end: str | None, pct_jump: float, repair_flag: bool = True):
    """
    Finds suspect rows in DB and attempts single-day repair using dual-source method.
    """
    engine = get_engine()

    if tickers:
        # Process per-ticker windows
        suspects = []
        for t in tickers:
            df = find_anomalies_in_db(t, start, end, pct_jump)
            if not df.empty:
                suspects.append(df)
        suspect_df = pd.concat(suspects, ignore_index=True) if suspects else pd.DataFrame()
    else:
        # All tickers
        suspect_df = find_anomalies_in_db(None, start, end, pct_jump)

    if suspect_df.empty:
        print("✅ No anomalies found in DB for the given window/filters.")
        return

    print(f"🔍 Found {len(suspect_df)} suspect rows to re-verify/rebuild")

    upsert_q = text("""
        INSERT INTO prices (ticker, date, close, high_price, low_price)
        VALUES (:ticker, :date, :close, :high, :low)
        ON CONFLICT (ticker, date) DO UPDATE
        SET close = EXCLUDED.close,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price;
    """)

    with engine.begin() as conn:
        fixed = 0
        dropped = 0
        for r in suspect_df.itertuples(index=False):
            tkr = r.ticker
            day = pd.Timestamp(r.date)
            repaired = refetch_single_day(tkr, day, repair=repair_flag)
            if repaired.empty:
                # If we cannot repair, we leave the existing row as-is but log it.
                logger.warning(f"{tkr} {day.date()}: could not repair; leaving current DB value")
                dropped += 1
                continue
            # Upsert repaired row
            conn.execute(upsert_q, {
                "ticker": tkr,
                "date": day.date(),
                "close": float(repaired.loc[day, "Close"]),
                "high": float(repaired.loc[day, "High"]),
                "low": float(repaired.loc[day, "Low"]),
            })
            fixed += 1

    print(f"🧼 Scrub complete. Fixed: {fixed}, Unresolved (left as-is): {dropped}")


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
    # ------------------------------------------------------
    # New flags
    # ------------------------------------------------------
    parser.add_argument("--no-auto-repair", action="store_true", help="Do NOT auto-repair suspect rows; drop them instead")
    parser.add_argument("--no-repair-flag", action="store_true", help="Disable yfinance server-side 'repair' param")
    parser.add_argument("--jump", type=float, default=DEFAULT_JUMP_PCT, help="Percent jump threshold for anomaly flagging (default=0.25)")
    parser.add_argument("--scrub", action="store_true", help="Scan DB for anomalies and attempt to repair them (no downloads unless anomalies found)")
    parser.add_argument("--scrub_start", type=str, default=None, help="Scrub window start date (YYYY-MM-DD)")
    parser.add_argument("--scrub_end", type=str, default=None, help="Scrub window end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # If --scrub, run scrubber first (or exclusively if no other flags)
    if args.scrub:
        tickers = None
        if args.file:
            df = pd.read_csv(args.file)
            tickers = df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
        elif args.tickers:
            tickers = [str(t).strip().upper() for t in args.tickers if str(t).strip()]
        print("🧽 Running DB scrub for anomalies …")
        scrub_db_anomalies(
            tickers,
            start=args.scrub_start,
            end=args.scrub_end,
            pct_jump=args.jump,
            repair_flag=(not args.no_repair_flag),
        )
        # After scrub, we continue into normal fetch unless user only wanted scrub.
        # If you prefer scrub-only, uncomment the next line:
        # return

    tickers = load_tickers(args)
    print(f"✅ Loaded {len(tickers)} tickers")

    # Resume map only when not reloading
    latest_dates = {} if args.reload else get_latest_dates()

    processed = 0
    total_inserted = 0
    deleted_for = set()  # avoid double-deleting a ticker

    auto_repair = not args.no_auto_repair
    repair_flag = not args.no_repair_flag

    with ThreadPoolExecutor(max_workers=min(args.max_workers or 2, 8)) as executor:
        futures = {
            executor.submit(
                fetch_prices_for_ticker,
                ticker,
                args.start,
                args.end,
                args.reload,
                latest_dates,
                auto_repair,
                repair_flag,
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
