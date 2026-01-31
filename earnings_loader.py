# earnings_loader.py

import os
import time
import argparse
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
import logging

# ----------------------------------------------------------
# DB helpers
# ----------------------------------------------------------
from db import get_engine

# ----------------------------------------------------------
# Logging Setup (file only; no console handler)
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers[:] = []  # avoid duplicate handlers in reloads

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(log_dir, "earnings_loader.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# ----------------------------------------------------------
# Initialize DB Connection (global engine)
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def parse_tickers_arg(raw: str | None) -> list[str]:
    """
    Parse a --tickers string into a list. Accepts comma and/or whitespace.
    Example: "AAPL, MSFT TSLA" -> ["AAPL","MSFT","TSLA"]
    """
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_tickers(file_path: str | None, explicit_tickers: list[str] | None) -> list[str]:
    """
    Priority:
      1) explicit_tickers (from --tickers)
      2) file_path (CSV with 'ticker' column)
      3) companies table (default)
    """
    if explicit_tickers:
        print(f"🎯 Using explicit tickers: {', '.join(explicit_tickers)}")
        logger.info(f"Using explicit tickers: {explicit_tickers}")
        return explicit_tickers

    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Could not read file {file_path}: {e}")

        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = (
            df[col]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
            .tolist()
        )
        logger.info(f"Loaded {len(tickers)} tickers from file ({file_path})")
        return tickers

    print("🗃️ Loading tickers from companies table…")
    with engine.connect() as conn:
        tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    logger.info(f"Loaded {len(tickers)} tickers from companies table")
    return tickers


def _ts_to_utc_date(ts) -> date | None:
    """
    Convert epoch seconds to a UTC date.
    Yahoo often returns epoch seconds (not ms) in yfinance info.
    """
    try:
        if ts is None:
            return None
        if isinstance(ts, (int, float)) and ts > 10_000:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
    except Exception:
        return None
    return None


def _days_to(d: date | None, today: date) -> int | None:
    if d is None:
        return None
    return (d - today).days


def _apply_guardrails(
    earnings_date: date | None,
    today: date,
    negative_grace_days: int,
    lookahead_days: int,
) -> tuple[date | None, int | None, str | None]:
    """
    Apply sanity rules:
      - allow small negative lag (<= negative_grace_days)
      - null out anything older than that (stale)
      - null out anything too far into the future (> lookahead_days)
    Returns (earnings_date, days_to_earnings, note)
    """
    if earnings_date is None:
        return None, None, None

    dte = _days_to(earnings_date, today)
    if dte is None:
        return None, None, None

    # Too negative => stale / not "next earnings"
    if dte < -abs(int(negative_grace_days)):
        return None, None, f"stale(dte={dte})"

    # Too far out => probably not a real scheduled earnings (or bad value)
    if lookahead_days is not None and lookahead_days > 0 and dte > int(lookahead_days):
        return None, None, f"too_far(dte={dte})"

    return earnings_date, dte, None


# ----------------------------------------------------------
# Function: Fetch next earnings for a ticker via yfinance info timestamps (with guardrails)
# ----------------------------------------------------------
def fetch_next_earnings(
    ticker: str,
    retries: int = 3,
    delay: int = 2,
    negative_grace_days: int = 3,
    lookahead_days: int = 180,
) -> dict:
    """
    Uses Yahoo/yfinance info timestamps:
      - earningsTimestampStart
      - earningsTimestampEnd
      - earningsTimestamp (fallback)

    Guardrails:
      - if days_to_earnings < -negative_grace_days => NULL (stale)
      - if days_to_earnings > lookahead_days => NULL (too far / probably wrong)

    Returns a dict suitable for DB upsert.
    """
    today = date.today()

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            start_ts = info.get("earningsTimestampStart")
            end_ts = info.get("earningsTimestampEnd")
            single_ts = info.get("earningsTimestamp")

            start_date = _ts_to_utc_date(start_ts)
            end_date = _ts_to_utc_date(end_ts)
            single_date = _ts_to_utc_date(single_ts)

            # Prefer start date (most common "next earnings" value when it's correct)
            raw_earnings_date = start_date or single_date

            # Label the raw source
            if start_date:
                source = "yahoo(info earningsTimestampStart/End)"
            elif single_date:
                source = "yahoo(info earningsTimestamp)"
            else:
                source = "yahoo(info) none"

            # Apply sanity guardrails
            earnings_date, dte, note = _apply_guardrails(
                earnings_date=raw_earnings_date,
                today=today,
                negative_grace_days=negative_grace_days,
                lookahead_days=lookahead_days,
            )

            if note:
                # Keep source, append note
                source = f"{source} | {note}"

            return {
                "ticker": ticker,
                "asof_date": today,
                "earnings_date": earnings_date,
                "earnings_start_ts_utc": int(start_ts) if isinstance(start_ts, (int, float)) else None,
                "earnings_end_ts_utc": int(end_ts) if isinstance(end_ts, (int, float)) else None,
                "source": source,
                "days_to_earnings": dte,
            }

        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Failed to fetch earnings for {ticker} after {retries} attempt(s)")
    return {
        "ticker": ticker,
        "asof_date": date.today(),
        "earnings_date": None,
        "earnings_start_ts_utc": None,
        "earnings_end_ts_utc": None,
        "source": "yahoo failed",
        "days_to_earnings": None,
    }


# ----------------------------------------------------------
# Function: Ensure DB objects exist (table + view)
# ----------------------------------------------------------
def ensure_company_earnings_objects() -> None:
    ddl_table = """
    CREATE TABLE IF NOT EXISTS company_earnings (
        ticker                 TEXT NOT NULL,
        asof_date              DATE NOT NULL,
        earnings_date          DATE,
        earnings_start_ts_utc  BIGINT,
        earnings_end_ts_utc    BIGINT,
        source                 TEXT,
        days_to_earnings       INTEGER,
        updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (ticker, asof_date)
    );

    CREATE INDEX IF NOT EXISTS idx_company_earnings_ticker_asof
        ON company_earnings (ticker, asof_date DESC);

    CREATE INDEX IF NOT EXISTS idx_company_earnings_asof
        ON company_earnings (asof_date DESC);
    """

    ddl_view = """
    CREATE OR REPLACE VIEW latest_company_earnings AS
    SELECT DISTINCT ON (ticker)
        ticker,
        asof_date,
        earnings_date,
        earnings_start_ts_utc,
        earnings_end_ts_utc,
        source,
        days_to_earnings,
        updated_at
    FROM company_earnings
    ORDER BY ticker, asof_date DESC;
    """

    with engine.begin() as conn:
        conn.execute(text(ddl_table))
        conn.execute(text(ddl_view))

    logger.info("Ensured company_earnings table + latest_company_earnings view exist")


# ----------------------------------------------------------
# Function: Upsert rows into company_earnings
# ----------------------------------------------------------
def upsert_company_earnings(rows: list[dict]) -> int:
    if not rows:
        return 0

    sql = text(
        """
        INSERT INTO company_earnings (
            ticker,
            asof_date,
            earnings_date,
            earnings_start_ts_utc,
            earnings_end_ts_utc,
            source,
            days_to_earnings,
            updated_at
        )
        VALUES (
            :ticker,
            :asof_date,
            :earnings_date,
            :earnings_start_ts_utc,
            :earnings_end_ts_utc,
            :source,
            :days_to_earnings,
            NOW()
        )
        ON CONFLICT (ticker, asof_date)
        DO UPDATE SET
            earnings_date = EXCLUDED.earnings_date,
            earnings_start_ts_utc = EXCLUDED.earnings_start_ts_utc,
            earnings_end_ts_utc = EXCLUDED.earnings_end_ts_utc,
            source = EXCLUDED.source,
            days_to_earnings = EXCLUDED.days_to_earnings,
            updated_at = NOW();
        """
    )

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


# ----------------------------------------------------------
# Function: Process tickers in parallel and write to DB
# ----------------------------------------------------------
def process_tickers(
    tickers: list[str],
    max_workers: int = 6,
    retries: int = 3,
    delay: int = 2,
    negative_grace_days: int = 3,
    lookahead_days: int = 180,
) -> None:
    start_time = time.time()
    today = date.today()

    rows: list[dict] = []
    missing = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_next_earnings,
                t,
                retries,
                delay,
                negative_grace_days,
                lookahead_days,
            ): t
            for t in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                row = future.result()
                if row.get("earnings_date") is None:
                    missing += 1
                    logger.info(f"{ticker}: no usable earnings date (asof={today}) src={row.get('source')}")
                rows.append(row)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

    n = upsert_company_earnings(rows)

    elapsed = time.time() - start_time
    print(f"✅ Upserted {n} rows into company_earnings (asof={today})")
    print(f"⚠️ Missing/NULL earnings_date: {missing} / {len(tickers)}")
    print(f"Total processing time: {elapsed:.2f} seconds")

    logger.info(f"Upserted {n} rows into company_earnings (asof={today})")
    logger.info(f"Missing/NULL earnings_date: {missing} / {len(tickers)}")
    logger.info(f"Total processing time: {elapsed:.2f} seconds")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch next earnings dates from yfinance and store in PostgreSQL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file of tickers (expects 'ticker' column)")
    group.add_argument(
        "--tickers",
        type=str,
        help='Optional comma/space separated list of tickers (e.g., "AAPL,MSFT TSLA")'
    )

    parser.add_argument("--workers", type=int, default=6, help="Max parallel workers (default: 6)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per ticker (default: 3)")
    parser.add_argument("--delay", type=int, default=2, help="Retry delay seconds (default: 2)")

    # NEW: guardrails
    parser.add_argument(
        "--negative-grace-days",
        type=int,
        default=365,
        help="Allow small negative days_to_earnings (Yahoo lag). Values below -N become NULL. Default: 3",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=180,
        help="Max allowed days_to_earnings in the future. Values above N become NULL. Default: 180",
    )

    parser.add_argument(
        "--no-ddl",
        action="store_true",
        help="Skip creating company_earnings table/view (assumes already exists)."
    )

    args = parser.parse_args()

    explicit_tickers = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit_tickers)

    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    if not args.no_ddl:
        ensure_company_earnings_objects()

    process_tickers(
        tickers,
        max_workers=args.workers,
        retries=args.retries,
        delay=args.delay,
        negative_grace_days=args.negative_grace_days,
        lookahead_days=args.lookahead_days,
    )
