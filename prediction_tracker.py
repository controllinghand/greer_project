# prediction_tracker.py

import logging
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from prediction_utils import calculate_prediction_score


# ----------------------------------------------------------
# Configure logging
# ----------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_tracker.log")

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

    # Convert numpy bools / integers / floats cleanly
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
# Load live prediction inputs for all tickers
# ----------------------------------------------------------
def load_prediction_inputs() -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        WITH latest_market AS (
            SELECT
                date,
                buyzone_pct,
                CASE
                    WHEN buyzone_pct >= 66 THEN 'EXTREME_OPPORTUNITY'
                    WHEN buyzone_pct >= 46 THEN 'ELEVATED_OPPORTUNITY'
                    WHEN buyzone_pct >= 14 THEN 'NORMAL'
                    WHEN buyzone_pct >= 10 THEN 'LOW_OPPORTUNITY'
                    ELSE 'EXTREME_GREED'
                END AS goi_zone
            FROM buyzone_breadth
            ORDER BY date DESC
            LIMIT 1
        ),
        company_phase_history AS (
            SELECT
                g.ticker,
                g.date,
                g.phase,
                g.confidence,
                LAG(g.phase) OVER (PARTITION BY g.ticker ORDER BY g.date) AS prior_phase,
                ROW_NUMBER() OVER (PARTITION BY g.ticker ORDER BY g.date DESC) AS rn
            FROM greer_company_index_daily g
        ),
        latest_company_phase AS (
            SELECT
                ticker,
                date AS snapshot_date,
                phase,
                prior_phase,
                confidence
            FROM company_phase_history
            WHERE rn = 1
        )
        SELECT
            ds.ticker,
            ds.name,
            ds.sector,
            ds.industry,
            ds.current_price,
            ds.greer_star_rating,
            ds.greer_value_score,
            ds.greer_yield_score,
            ds.buyzone_flag,
            ds.gfv_price,
            ds.gfv_status,
            lcp.snapshot_date,
            lcp.phase,
            lcp.prior_phase,
            lcp.confidence,
            lm.buyzone_pct AS market_buyzone_pct,
            lm.goi_zone
        FROM dashboard_snapshot ds
        JOIN latest_company_phase lcp
          ON lcp.ticker = ds.ticker
        CROSS JOIN latest_market lm
        ORDER BY ds.ticker
        """
    )

    df = pd.read_sql(query, engine)
    logger.info("Loaded %s live prediction input rows", len(df))
    return df


# ----------------------------------------------------------
# Build prediction rows for tracking
# ----------------------------------------------------------
def build_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Keep tracking aligned with live page logic
    df = df[df["buyzone_flag"].notna()].copy()
    df = df[~((df["phase"] == "CONTRACTION") & (df["goi_zone"] == "EXTREME_GREED"))].copy()

    score_df = df.apply(calculate_prediction_score, axis=1)
    df = pd.concat([df, score_df], axis=1)

    tracking_df = pd.DataFrame(
        {
            "prediction_date": pd.Timestamp.today().date(),
            "ticker": df["ticker"],
            "current_price": df["current_price"],
            "phase": df["phase"],
            "prior_phase": df["prior_phase"],
            "goi_zone": df["goi_zone"],
            "buyzone_flag": df["buyzone_flag"],
            "confidence": df["confidence"],
            "greer_value_score": df["greer_value_score"],
            "greer_yield_score": df["greer_yield_score"],
            "prediction_score": df["prediction_score"],
            "raw_bucket": df["score_bucket"],
            "calibration_bucket": df["calibration_bucket"],
            "signal_tier": df["signal_tier"],
            "expected_win_rate_60d": df["expected_win_rate_60d"],
            "expected_return_60d": df["expected_return_60d"],
            "setup_label": df["setup_label"],
        }
    )

    logger.info("Prepared %s prediction tracking rows", len(tracking_df))
    return tracking_df


# ----------------------------------------------------------
# Upsert today's prediction rows into prediction_tracking
# ----------------------------------------------------------
def upsert_predictions(df: pd.DataFrame) -> int:
    if df.empty:
        logger.info("No prediction rows to upsert")
        return 0

    engine = get_engine()

    rows = [
        {
            "prediction_date": clean_value(row["prediction_date"]),
            "ticker": clean_value(row["ticker"]),
            "current_price": clean_value(row["current_price"]),
            "phase": clean_value(row["phase"]),
            "prior_phase": clean_value(row["prior_phase"]),
            "goi_zone": clean_value(row["goi_zone"]),
            "buyzone_flag": clean_value(row["buyzone_flag"]),
            "confidence": clean_value(row["confidence"]),
            "greer_value_score": clean_value(row["greer_value_score"]),
            "greer_yield_score": clean_value(row["greer_yield_score"]),
            "prediction_score": clean_value(row["prediction_score"]),
            "raw_bucket": clean_value(row["raw_bucket"]),
            "calibration_bucket": clean_value(row["calibration_bucket"]),
            "signal_tier": clean_value(row["signal_tier"]),
            "expected_win_rate_60d": clean_value(row["expected_win_rate_60d"]),
            "expected_return_60d": clean_value(row["expected_return_60d"]),
            "setup_label": clean_value(row["setup_label"]),
        }
        for _, row in df.iterrows()
    ]

    sql = text("""
        INSERT INTO prediction_tracking (
            prediction_date,
            ticker,
            current_price,
            phase,
            prior_phase,
            goi_zone,
            buyzone_flag,
            confidence,
            greer_value_score,
            greer_yield_score,
            prediction_score,
            raw_bucket,
            calibration_bucket,
            signal_tier,
            expected_win_rate_60d,
            expected_return_60d,
            setup_label
        )
        VALUES (
            :prediction_date,
            :ticker,
            :current_price,
            :phase,
            :prior_phase,
            :goi_zone,
            :buyzone_flag,
            :confidence,
            :greer_value_score,
            :greer_yield_score,
            :prediction_score,
            :raw_bucket,
            :calibration_bucket,
            :signal_tier,
            :expected_win_rate_60d,
            :expected_return_60d,
            :setup_label
        )
        ON CONFLICT (prediction_date, ticker)
        DO UPDATE SET
            current_price = EXCLUDED.current_price,
            phase = EXCLUDED.phase,
            prior_phase = EXCLUDED.prior_phase,
            goi_zone = EXCLUDED.goi_zone,
            buyzone_flag = EXCLUDED.buyzone_flag,
            confidence = EXCLUDED.confidence,
            greer_value_score = EXCLUDED.greer_value_score,
            greer_yield_score = EXCLUDED.greer_yield_score,
            prediction_score = EXCLUDED.prediction_score,
            raw_bucket = EXCLUDED.raw_bucket,
            calibration_bucket = EXCLUDED.calibration_bucket,
            signal_tier = EXCLUDED.signal_tier,
            expected_win_rate_60d = EXCLUDED.expected_win_rate_60d,
            expected_return_60d = EXCLUDED.expected_return_60d,
            setup_label = EXCLUDED.setup_label
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    logger.info("Upserted %s prediction rows", len(rows))
    return len(rows)


# ----------------------------------------------------------
# Update realized 20d outcomes using trading-day offsets
# ----------------------------------------------------------
def update_actuals_20d() -> None:
    engine = get_engine()

    sql = text("""
        UPDATE prediction_tracking pt
        SET
            actual_price_20d = r.future_price,
            actual_return_20d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN (r.future_price - pt.current_price) / pt.current_price
                ELSE NULL
            END,
            actual_win_20d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN r.future_price > pt.current_price
                ELSE NULL
            END
        FROM (
            SELECT
                pt2.tracking_id,
                (
                    SELECT p.close
                    FROM prices p
                    WHERE p.ticker = pt2.ticker
                      AND p.date > pt2.prediction_date
                    ORDER BY p.date
                    OFFSET 19 LIMIT 1
                ) AS future_price
            FROM prediction_tracking pt2
            WHERE pt2.actual_price_20d IS NULL
        ) r
        WHERE pt.tracking_id = r.tracking_id
          AND r.future_price IS NOT NULL
    """)

    with engine.begin() as conn:
        result = conn.execute(sql)

    logger.info("Updated 20d actuals for %s rows", result.rowcount)


# ----------------------------------------------------------
# Update realized 60d outcomes using trading-day offsets
# ----------------------------------------------------------
def update_actuals_60d() -> None:
    engine = get_engine()

    sql = text("""
        UPDATE prediction_tracking pt
        SET
            actual_price_60d = r.future_price,
            actual_return_60d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN (r.future_price - pt.current_price) / pt.current_price
                ELSE NULL
            END,
            actual_win_60d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN r.future_price > pt.current_price
                ELSE NULL
            END
        FROM (
            SELECT
                pt2.tracking_id,
                (
                    SELECT p.close
                    FROM prices p
                    WHERE p.ticker = pt2.ticker
                      AND p.date > pt2.prediction_date
                    ORDER BY p.date
                    OFFSET 59 LIMIT 1
                ) AS future_price
            FROM prediction_tracking pt2
            WHERE pt2.actual_price_60d IS NULL
        ) r
        WHERE pt.tracking_id = r.tracking_id
          AND r.future_price IS NOT NULL
    """)

    with engine.begin() as conn:
        result = conn.execute(sql)

    logger.info("Updated 60d actuals for %s rows", result.rowcount)

# ----------------------------------------------------------
# Update realized 90d outcomes using trading-day offsets
# ----------------------------------------------------------
def update_actuals_90d() -> None:
    engine = get_engine()

    sql = text("""
        UPDATE prediction_tracking pt
        SET
            actual_price_90d = r.future_price,
            actual_return_90d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN (r.future_price - pt.current_price) / pt.current_price
                ELSE NULL
            END,
            actual_win_90d = CASE
                WHEN pt.current_price IS NOT NULL AND pt.current_price <> 0 AND r.future_price IS NOT NULL
                THEN r.future_price > pt.current_price
                ELSE NULL
            END
        FROM (
            SELECT
                pt2.tracking_id,
                (
                    SELECT p.close
                    FROM prices p
                    WHERE p.ticker = pt2.ticker
                      AND p.date > pt2.prediction_date
                    ORDER BY p.date
                    OFFSET 89 LIMIT 1
                ) AS future_price
            FROM prediction_tracking pt2
            WHERE pt2.actual_price_90d IS NULL
        ) r
        WHERE pt.tracking_id = r.tracking_id
          AND r.future_price IS NOT NULL
    """)

    with engine.begin() as conn:
        result = conn.execute(sql)

    logger.info("Updated 90d actuals for %s rows", result.rowcount)


# ----------------------------------------------------------
# Main runner
# ----------------------------------------------------------
def main() -> None:
    start = datetime.now()
    logger.info("Starting prediction tracker")

    live_df = load_prediction_inputs()
    tracking_df = build_prediction_rows(live_df)
    upsert_predictions(tracking_df)

    update_actuals_20d()
    update_actuals_60d()
    update_actuals_90d()

    elapsed = datetime.now() - start
    logger.info("Prediction tracker finished in %s", elapsed)


if __name__ == "__main__":
    main()