# backtest.py

import argparse
import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy import text
import os
import logging
from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "backtest.log"),
    level=logging.INFO,  # Fixed typo: was 'loggingACON'
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# CLI Arguments
# ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Back-test Greer Opportunity strategy")
parser.add_argument("--since", help="Earliest entry date (YYYY-MM-DD)")
parser.add_argument("--until", help="Latest entry date (YYYY-MM-DD)")
parser.add_argument("--reload", action="store_true", help="Reload: Delete existing results for today before inserting new ones")
args = parser.parse_args()

SINCE = date.fromisoformat(args.since) if args.since else date(1900, 1, 1)
UNTIL = date.fromisoformat(args.until) if args.until else date.today()
TODAY = date.today().isoformat()

# ----------------------------------------------------------
# DB Connection
# ----------------------------------------------------------
engine = get_engine()
print(f"Database connection: {engine.url}")  # Shows host, dbname, user, etc.

try:
    # ----------------------------------------------------------
    # Main Query
    # ----------------------------------------------------------
    SQL = text("""
        WITH daily AS (
            SELECT p.ticker,
                   p.date,
                   p.close,
                   (
                     SELECT gs.greer_score
                     FROM greer_scores gs
                     WHERE gs.ticker = p.ticker AND gs.report_date <= p.date
                     ORDER BY gs.report_date DESC
                     LIMIT 1
                   ) AS greer_val,
                   (
                     SELECT yd.score
                     FROM greer_yields_daily yd
                     WHERE yd.ticker = p.ticker AND yd.date <= p.date
                     ORDER BY yd.date DESC
                     LIMIT 1
                   ) AS yield_val,
                   bz.in_buyzone,
                   fvg.direction
            FROM prices p
            LEFT JOIN greer_buyzone_daily bz ON bz.ticker = p.ticker AND bz.date = p.date
            LEFT JOIN fair_value_gaps fvg ON fvg.ticker = p.ticker AND fvg.date = p.date
            WHERE p.date BETWEEN :since AND :until
        ), qual AS (
            SELECT *,
                   greer_val >= 50        AS gv_ok,
                   yield_val >= 3         AS yld_ok,
                   in_buyzone IS TRUE     AS bz_ok,
                   direction = 'bullish'  AS fvg_ok
            FROM daily
        ), first_hit AS (
            SELECT DISTINCT ON (ticker) ticker, date AS entry_date, close AS entry_close
            FROM qual
            WHERE gv_ok AND yld_ok AND bz_ok AND fvg_ok
            ORDER BY ticker, date
        ), latest_price AS (
            SELECT DISTINCT ON (ticker) ticker, date AS last_date, close AS last_close
            FROM prices
            WHERE close > 0.01 AND date <= :today
            ORDER BY ticker, date DESC
        )
        SELECT fh.ticker,
               fh.entry_date,
               fh.entry_close,
               lp.last_date,
               lp.last_close,
               ROUND((lp.last_close - fh.entry_close) / fh.entry_close * 100, 2) AS pct_return,
               (lp.last_date - fh.entry_date) AS days_held
        FROM first_hit fh
        JOIN latest_price lp USING (ticker)
        ORDER BY pct_return DESC;
    """)

    # ----------------------------------------------------------
    # Execute and Fetch
    # ----------------------------------------------------------
    with engine.begin() as conn:
        # Delete existing results for today if --reload is used
        if args.reload:
            DELETE_SQL = text("""
                DELETE FROM backtest_results
                WHERE run_date = :run_date;
            """)
            conn.execute(DELETE_SQL, {"run_date": TODAY})
            print(f"🧹 Deleted existing backtest_results for run_date={TODAY} (reload mode).")

        # DEBUG SECTION
        test_sql = text("""
            SELECT date, close
            FROM prices
            WHERE ticker = 'SHW'
            ORDER BY date DESC
            LIMIT 1;
        """)
        test_df = pd.read_sql(test_sql, conn)
        print("Script's view of SHW latest price:")
        print(test_df)
        # END DEBUG SECTION
        df = pd.read_sql(SQL, conn, params={"since": SINCE.isoformat(), "until": UNTIL.isoformat(), "today": TODAY})

        if df.empty:
            print("⚠️ No qualifying trades found.")
            logger.info("No qualifying trades found.")
            exit(0)

        # Validate last_close values
        if (df['last_close'] < 1).any():
            print("⚠️ Warning: Some last_closes are suspiciously low—check prices table.")
            logger.warning("Some last_closes < 1 detected in results.")

    # ----------------------------------------------------------
    # Print Results
    # ----------------------------------------------------------
    pd.set_option("display.max_rows", None)
    print(df)

    print("\n📊 Summary metrics")
    print("=========================")
    print(f"Count       : {len(df)}")
    print(f"Win rate    : {(df['pct_return'] > 0).mean():.2%}")  # Fixed typo: was '.2%'
    print(f"Mean %      : {df['pct_return'].mean():.2f}")
    print(f"Median %    : {df['pct_return'].median():.2f}")
    print(f"Std dev %   : {df['pct_return'].std():.2f}")

    # ----------------------------------------------------------
    # Insert Into DB (Upsert)
    # ----------------------------------------------------------
    INSERT_SQL = text("""
        INSERT INTO backtest_results (
            ticker, entry_date, entry_close,
            last_date, last_close, pct_return,
            days_held, run_date
        )
        VALUES (
            :ticker, :entry_date, :entry_close,
            :last_date, :last_close, :pct_return,
            :days_held, :run_date
        )
        ON CONFLICT (ticker, run_date) DO UPDATE SET
            entry_date = EXCLUDED.entry_date,
            entry_close = EXCLUDED.entry_close,
            last_date = EXCLUDED.last_date,
            last_close = EXCLUDED.last_close,
            pct_return = EXCLUDED.pct_return,
            days_held = EXCLUDED.days_held;
    """)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(INSERT_SQL, {
                "ticker": row["ticker"],
                "entry_date": row["entry_date"],  # Fixed typo: was 'entry_SETUP'
                "entry_close": row["entry_close"],
                "last_date": row["last_date"],
                "last_close": row["last_close"],
                "pct_return": row["pct_return"],
                "days_held": row["days_held"],
                "run_date": TODAY
            })

    print(f"\n✅ {len(df)} backtest results inserted/updated in DB.\n")

except Exception as e:
    logger.error(f"❌ Backtest failed: {e}")
    print(f"❌ Backtest failed: {e}")
    exit(1)