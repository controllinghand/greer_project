# prediction_engine_backtest_builder.py

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from prediction_utils import calculate_prediction_score


# ----------------------------------------------------------
# Configure logging
# ----------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_engine_backtest_builder.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Normalize pandas / numpy values before insert
# ----------------------------------------------------------
def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass

    return value


# ----------------------------------------------------------
# Get trading dates to process
# Uses prices table so we only process real market days
# ----------------------------------------------------------
def get_backtest_dates(years: int, end_date: str | None = None) -> list[pd.Timestamp]:
    engine = get_engine()

    if end_date:
        end_dt = pd.to_datetime(end_date).date()
    else:
        end_dt = datetime.today().date()

    approx_start = end_dt - timedelta(days=(years * 370))

    sql = text(
        """
        SELECT DISTINCT date
        FROM prices
        WHERE date >= :start_date
          AND date <= :end_date
        ORDER BY date
        """
    )

    df = pd.read_sql(
        sql,
        engine,
        params={
            "start_date": approx_start,
            "end_date": end_dt,
        },
    )

    dates = pd.to_datetime(df["date"]).tolist()
    logger.info("Found %s trading dates between %s and %s", len(dates), approx_start, end_dt)
    return dates


# ----------------------------------------------------------
# Load historical prediction inputs for one snapshot date
# Mirrors live page logic, but historically
# ----------------------------------------------------------
def load_prediction_inputs_for_date(snapshot_date: pd.Timestamp) -> pd.DataFrame:
    engine = get_engine()

    sql = text(
        """
        WITH market_day AS (
            SELECT
                b.date,
                b.buyzone_pct,
                CASE
                    WHEN b.buyzone_pct >= 66 THEN 'EXTREME_OPPORTUNITY'
                    WHEN b.buyzone_pct >= 46 THEN 'ELEVATED_OPPORTUNITY'
                    WHEN b.buyzone_pct >= 14 THEN 'NORMAL'
                    WHEN b.buyzone_pct >= 10 THEN 'LOW_OPPORTUNITY'
                    ELSE 'EXTREME_GREED'
                END AS goi_zone
            FROM buyzone_breadth b
            WHERE b.date = :snapshot_date
        ),
        company_phase_history AS (
            SELECT
                g.ticker,
                g.date,
                g.phase,
                g.confidence,
                LAG(g.phase) OVER (PARTITION BY g.ticker ORDER BY g.date) AS prior_phase
            FROM greer_company_index_daily g
            WHERE g.date <= :snapshot_date
        ),
        latest_company_phase AS (
            SELECT DISTINCT ON (ticker)
                ticker,
                date AS snapshot_date,
                phase,
                prior_phase,
                confidence
            FROM company_phase_history
            ORDER BY ticker, snapshot_date DESC
        ),
        latest_snapshot AS (
            SELECT DISTINCT ON (s.ticker)
                s.ticker,
                s.close AS current_price,
                s.greer_star_rating,
                s.greer_value_score,
                s.greer_yield_score,
                s.buyzone_flag,
                s.gfv_price,
                s.snapshot_date::date AS snapshot_date
            FROM company_snapshot s
            WHERE s.snapshot_date::date <= :snapshot_date
            ORDER BY s.ticker, s.snapshot_date DESC
        )
        SELECT
            ls.ticker,
            ls.current_price,
            ls.greer_star_rating,
            ls.greer_value_score,
            ls.greer_yield_score,
            ls.buyzone_flag,
            ls.gfv_price,
            lcp.snapshot_date,
            lcp.phase,
            lcp.prior_phase,
            lcp.confidence,
            md.buyzone_pct AS market_buyzone_pct,
            md.goi_zone
        FROM latest_snapshot ls
        JOIN latest_company_phase lcp
          ON lcp.ticker = ls.ticker
        CROSS JOIN market_day md
        ORDER BY ls.ticker
        """
    )

    df = pd.read_sql(sql, engine, params={"snapshot_date": snapshot_date.date()})
    return df


# ----------------------------------------------------------
# Attach forward returns for one snapshot date
# ----------------------------------------------------------
def load_forward_returns_for_date(snapshot_date: pd.Timestamp) -> pd.DataFrame:
    engine = get_engine()

    sql = text(
        """
        SELECT
            snapshot_date,
            ticker,
            entry_price,
            return_60d,
            return_90d,
            return_120d,
            return_180d
        FROM prediction_horizon_returns_fast
        WHERE snapshot_date = :snapshot_date
        """
    )

    return pd.read_sql(sql, engine, params={"snapshot_date": snapshot_date.date()})


# ----------------------------------------------------------
# Build one day's scored rows
# Keeps the same filters as live page / tracker
# ----------------------------------------------------------
def build_rows_for_date(snapshot_date: pd.Timestamp) -> pd.DataFrame:
    inputs_df = load_prediction_inputs_for_date(snapshot_date)

    if inputs_df.empty:
        logger.warning("%s | no historical inputs found", snapshot_date.date())
        return pd.DataFrame()

    # Keep aligned with live model
    inputs_df = inputs_df[inputs_df["buyzone_flag"].notna()].copy()
    inputs_df = inputs_df[
        ~((inputs_df["phase"] == "CONTRACTION") & (inputs_df["goi_zone"] == "EXTREME_GREED"))
    ].copy()

    if inputs_df.empty:
        logger.warning("%s | inputs empty after model filters", snapshot_date.date())
        return pd.DataFrame()

    score_df = inputs_df.apply(calculate_prediction_score, axis=1)
    scored_df = pd.concat([inputs_df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)

    returns_df = load_forward_returns_for_date(snapshot_date)
    scored_df = scored_df.merge(
        returns_df,
        how="left",
        left_on=["snapshot_date", "ticker"],
        right_on=["snapshot_date", "ticker"],
    )

    scored_df["win_60d"] = scored_df["return_60d"] > 0
    scored_df["win_90d"] = scored_df["return_90d"] > 0
    scored_df["win_120d"] = scored_df["return_120d"] > 0
    scored_df["win_180d"] = scored_df["return_180d"] > 0

    out = pd.DataFrame(
        {
            "snapshot_date": scored_df["snapshot_date"],
            "ticker": scored_df["ticker"],
            "current_price": scored_df["current_price"],
            "greer_star_rating": scored_df["greer_star_rating"],
            "greer_value_score": scored_df["greer_value_score"],
            "greer_yield_score": scored_df["greer_yield_score"],
            "buyzone_flag": scored_df["buyzone_flag"],
            "gfv_price": scored_df["gfv_price"],
            "prior_phase": scored_df["prior_phase"],
            "phase": scored_df["phase"],
            "confidence": scored_df["confidence"],
            "market_buyzone_pct": scored_df["market_buyzone_pct"],
            "goi_zone": scored_df["goi_zone"],
            "prediction_score": scored_df["prediction_score"],
            "raw_bucket": scored_df["score_bucket"],
            "calibration_bucket": scored_df["calibration_bucket"],
            "signal_tier": scored_df["signal_tier"],
            "signal_horizon": scored_df["signal_horizon"],
            "expected_win_rate_trend": scored_df["expected_win_rate_trend"],
            "expected_return_trend": scored_df["expected_return_trend"],
            "expected_win_rate_60d": scored_df.get("expected_win_rate_60d"),
            "expected_return_60d": scored_df.get("expected_return_60d"),
            "setup_label": scored_df["setup_label"],
            "return_60d": scored_df["return_60d"],
            "return_90d": scored_df["return_90d"],
            "return_120d": scored_df["return_120d"],
            "return_180d": scored_df["return_180d"],
            "win_60d": scored_df["return_60d"] > 0,
            "win_90d": scored_df["return_90d"] > 0,
            "win_120d": scored_df["return_120d"] > 0,
            "win_180d": scored_df["return_180d"] > 0,
        }
    )

    return out


# ----------------------------------------------------------
# Upsert one day's rows
# ----------------------------------------------------------
def upsert_backtest_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    engine = get_engine()

    rows = []
    for _, row in df.iterrows():
        rows.append({col: clean_value(row[col]) for col in df.columns})

    sql = text(
        """
        INSERT INTO prediction_engine_backtest_daily (
            snapshot_date,
            ticker,
            current_price,
            greer_star_rating,
            greer_value_score,
            greer_yield_score,
            buyzone_flag,
            gfv_price,
            prior_phase,
            phase,
            confidence,
            market_buyzone_pct,
            goi_zone,
            prediction_score,
            raw_bucket,
            calibration_bucket,
            signal_tier,
            signal_horizon,
            expected_win_rate_trend,
            expected_return_trend,
            expected_win_rate_60d,
            expected_return_60d,
            setup_label,
            return_60d,
            return_90d,
            return_120d,
            return_180d,
            win_60d,
            win_90d,
            win_120d,
            win_180d
        )
        VALUES (
            :snapshot_date,
            :ticker,
            :current_price,
            :greer_star_rating,
            :greer_value_score,
            :greer_yield_score,
            :buyzone_flag,
            :gfv_price,
            :prior_phase,
            :phase,
            :confidence,
            :market_buyzone_pct,
            :goi_zone,
            :prediction_score,
            :raw_bucket,
            :calibration_bucket,
            :signal_tier,
            :signal_horizon,
            :expected_win_rate_trend,
            :expected_return_trend,
            :expected_win_rate_60d,
            :expected_return_60d,
            :setup_label,
            :return_60d,
            :return_90d,
            :return_120d,
            :return_180d,
            :win_60d,
            :win_90d,
            :win_120d,
            :win_180d
        )
        ON CONFLICT (snapshot_date, ticker)
        DO UPDATE SET
            current_price = EXCLUDED.current_price,
            greer_star_rating = EXCLUDED.greer_star_rating,
            greer_value_score = EXCLUDED.greer_value_score,
            greer_yield_score = EXCLUDED.greer_yield_score,
            buyzone_flag = EXCLUDED.buyzone_flag,
            gfv_price = EXCLUDED.gfv_price,
            prior_phase = EXCLUDED.prior_phase,
            phase = EXCLUDED.phase,
            confidence = EXCLUDED.confidence,
            market_buyzone_pct = EXCLUDED.market_buyzone_pct,
            goi_zone = EXCLUDED.goi_zone,
            prediction_score = EXCLUDED.prediction_score,
            raw_bucket = EXCLUDED.raw_bucket,
            calibration_bucket = EXCLUDED.calibration_bucket,
            signal_tier = EXCLUDED.signal_tier,
            signal_horizon = EXCLUDED.signal_horizon,
            expected_win_rate_trend = EXCLUDED.expected_win_rate_trend,
            expected_return_trend = EXCLUDED.expected_return_trend,
            expected_win_rate_60d = EXCLUDED.expected_win_rate_60d,
            expected_return_60d = EXCLUDED.expected_return_60d,
            setup_label = EXCLUDED.setup_label,
            return_60d = EXCLUDED.return_60d,
            return_90d = EXCLUDED.return_90d,
            return_120d = EXCLUDED.return_120d,
            return_180d = EXCLUDED.return_180d,
            win_60d = EXCLUDED.win_60d,
            win_90d = EXCLUDED.win_90d,
            win_120d = EXCLUDED.win_120d,
            win_180d = EXCLUDED.win_180d
        """
    )

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


# ----------------------------------------------------------
# Main runner
# ----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical prediction engine backtest rows.")
    parser.add_argument("--years", type=int, default=1, help="Number of years of trading dates to rebuild")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD end date")
    parser.add_argument("--limit-days", type=int, default=None, help="Optional cap for quick runtime tests")
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date override")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting prediction_engine_backtest_builder")

    dates = get_backtest_dates(years=args.years, end_date=args.end_date)

    if args.start_date:
        start_dt = pd.to_datetime(args.start_date)
        dates = [d for d in dates if d >= start_dt]

    if args.limit_days:
        dates = dates[-args.limit_days:]

    total_rows = 0

    for i, snapshot_date in enumerate(dates, start=1):
        loop_start = time.time()

        try:
            day_df = build_rows_for_date(snapshot_date)
            inserted = upsert_backtest_rows(day_df)
            total_rows += inserted

            elapsed = time.time() - loop_start
            logger.info(
                "%s/%s | %s | rows=%s | %.2fs",
                i,
                len(dates),
                snapshot_date.date(),
                inserted,
                elapsed,
            )
        except Exception:
            logger.exception("Failed for snapshot_date=%s", snapshot_date.date())
            raise

    total_elapsed = time.time() - start_time
    logger.info("Finished. dates=%s rows=%s total_seconds=%.2f", len(dates), total_rows, total_elapsed)


if __name__ == "__main__":
    main()