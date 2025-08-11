# list_price_anomalies.py
# ----------------------------------------------------------
# List suspicious daily price jumps from the DB
# Defaults to last 30 days, skips NaNs/newborns, supports CSV export
# ----------------------------------------------------------

import argparse
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Build date window (defaults to last N days)
# ----------------------------------------------------------
def resolve_window(start: str | None, end: str | None, days: int | None):
    # If explicit start/end provided, honor them
    if start or end:
        return start, end
    # Else default to last `days` (30)
    if days is None:
        days = 30
    d_end = date.today().isoformat()
    d_start = (date.today() - timedelta(days=days)).isoformat()
    return d_start, d_end

# ----------------------------------------------------------
# Query anomalies with guards for NaNs/new tickers
# ----------------------------------------------------------
def find_anomalies_in_db(
    tickers: list[str] | None,
    start: str | None,
    end: str | None,
    pct_jump: float = 0.25,
    min_prev_days: int = 3,
) -> pd.DataFrame:
    """
    Flags rows where |close/prev_close - 1| > pct_jump within date window,
    requiring:
      - close IS NOT NULL
      - prev_close IS NOT NULL AND prev_close != 0
      - at least `min_prev_days` prior observations for that ticker (to skip newborns)
    """
    engine = get_engine()
    where = []
    params = {}

    if start:
        where.append("p.date >= :start")
        params["start"] = start
    if end:
        where.append("p.date <= :end")
        params["end"] = end
    if tickers:
        where.append("p.ticker = ANY(:tickers)")
        params["tickers"] = tickers

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        WITH base AS (
          SELECT
            p.ticker,
            p.date,
            p.close,
            -- LAG for previous close
            LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.date) AS prev_close,
            -- Count of observations up to this row (trading age)
            COUNT(p.close) OVER (PARTITION BY p.ticker ORDER BY p.date
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS obs_count
          FROM prices p
          {where_sql}
        )
        SELECT
          ticker,
          date,
          close,
          prev_close,
          ROUND(ABS(close/prev_close - 1.0)*100, 2) AS pct_change
        FROM base
        WHERE close IS NOT NULL
          AND prev_close IS NOT NULL
          AND prev_close <> 0
          AND obs_count > :min_prev_days
          AND ABS(close/prev_close - 1.0) > :pct
        ORDER BY date DESC, ticker;
    """

    params["pct"] = pct_jump
    params["min_prev_days"] = int(min_prev_days)

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    return df

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List price anomalies from the DB (last 30 days by default).")
    parser.add_argument("--tickers", nargs="+", help="Optional list of tickers to filter (e.g. WM AAPL MSFT)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    parser.add_argument("--pct", type=float, default=0.25, help="Jump threshold (e.g., 0.25 for 25%%)")
    parser.add_argument("--min_prev_days", type=int, default=3, help="Require at least this many prior trading days (default: 3)")
    parser.add_argument("--csv", type=str, help="Optional path to write CSV output")
    args = parser.parse_args()

    start, end = resolve_window(args.start, args.end, args.days)
    tickers = [t.upper() for t in args.tickers] if args.tickers else None

    df = find_anomalies_in_db(
        tickers=tickers,
        start=start,
        end=end,
        pct_jump=args.pct,
        min_prev_days=args.min_prev_days,
    )

    print(f"Window: {start} → {end}")
    print(f"Threshold: > {args.pct*100:.2f}% day-over-day | Min prior days: {args.min_prev_days}")
    print(f"Found {len(df)} suspect rows")
    if not df.empty:
        # Optional quick summary per ticker
        counts = df.groupby("ticker", as_index=False).size().rename(columns={"size": "count"})
        print("\nBy ticker:")
        print(counts.to_string(index=False))
        print("\nDetails:")
        print(df.to_string(index=False))

        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\n📄 Wrote CSV: {args.csv}")
