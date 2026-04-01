# refresh_prediction_snapshot.py
# ----------------------------------------------------------
# Dynamic Uplift: Now uses market_regime_thresholds view
# ----------------------------------------------------------

import logging
import os
import sys
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from prediction_utils import calculate_prediction_score
# NEW: Import our unified brain
from market_cycle_utils import get_market_thresholds, get_goi_label


# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "refresh_prediction_snapshot.log")

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
# Clean values for SQLAlchemy inserts
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
# Load live prediction inputs
# ----------------------------------------------------------
def load_prediction_inputs() -> pd.DataFrame:
    engine = get_engine()
    
    # NEW: Fetch dynamic thresholds first
    thresholds = get_market_thresholds(engine)

    # UPDATED SQL: We removed the hardcoded CASE statement and 
    # replaced it with a simple select. We'll handle the labeling in Python.
    query = text("""
        WITH latest_market AS (
            SELECT
                date,
                buyzone_pct
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
                LAG(g.phase) OVER (
                    PARTITION BY g.ticker
                    ORDER BY g.date
                ) AS prior_phase,
                ROW_NUMBER() OVER (
                    PARTITION BY g.ticker
                    ORDER BY g.date DESC
                ) AS rn
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
            lm.buyzone_pct AS market_buyzone_pct
        FROM dashboard_snapshot ds
        JOIN latest_company_phase lcp
          ON lcp.ticker = ds.ticker
        CROSS JOIN latest_market lm
        ORDER BY ds.ticker
    """)

    df = pd.read_sql(query, engine)
    
    # NEW: Apply the dynamic GOI label in Python using the unified utility
    if not df.empty:
        df['goi_zone'] = df['market_buyzone_pct'].apply(lambda x: get_goi_label(x, thresholds))
    
    logger.info("Loaded %s prediction input rows with dynamic GOI zones", len(df))
    return df


# ----------------------------------------------------------
# Apply page filtering and prediction scoring
# ----------------------------------------------------------
def build_prediction_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working_df = df.copy()

    # Match current page behavior
    working_df = working_df[working_df["buyzone_flag"].notna()].copy()
    working_df = working_df[
        ~(
            (working_df["phase"] == "CONTRACTION") &
            (working_df["goi_zone"] == "EXTREME_GREED")
        )
    ].copy()

    if working_df.empty:
        logger.warning("All prediction rows filtered out before scoring")
        return pd.DataFrame()

    score_df = working_df.apply(calculate_prediction_score, axis=1)
    working_df = pd.concat([working_df, score_df], axis=1)

    working_df["prediction_date"] = pd.to_datetime(date.today())
    working_df["expected_win_rate_trend_pct"] = (
        working_df["expected_win_rate_trend"] * 100
    ).round(1)
    working_df["expected_return_trend_pct"] = (
        working_df["expected_return_trend"] * 100
    ).round(1)
    working_df["confidence_pct"] = (working_df["confidence"] * 100).round(1)

    snapshot_df = pd.DataFrame({
        "prediction_date": working_df["prediction_date"],
        "ticker": working_df["ticker"],
        "name": working_df["name"],
        "sector": working_df["sector"],
        "industry": working_df["industry"],
        "current_price": working_df["current_price"],
        "prediction_score": working_df["prediction_score"],
        "score_bucket": working_df["score_bucket"],
        "calibration_bucket": working_df["calibration_bucket"],
        "signal_tier": working_df["signal_tier"],
        "signal_horizon": working_df["signal_horizon"],
        "expected_win_rate_trend": working_df["expected_win_rate_trend"],
        "expected_return_trend": working_df["expected_return_trend"],
        "expected_win_rate_trend_pct": working_df["expected_win_rate_trend_pct"],
        "expected_return_trend_pct": working_df["expected_return_trend_pct"],
        "setup_label": working_df["setup_label"],
        "phase": working_df["phase"],
        "prior_phase": working_df["prior_phase"],
        "goi_zone": working_df["goi_zone"],
        "buyzone_flag": working_df["buyzone_flag"],
        "confidence": working_df["confidence"],
        "confidence_pct": working_df["confidence_pct"],
        "greer_value_score": working_df["greer_value_score"],
        "greer_yield_score": working_df["greer_yield_score"],
        "greer_star_rating": working_df["greer_star_rating"],
        "gfv_price": working_df["gfv_price"],
        "gfv_status": working_df["gfv_status"],
        "snapshot_date": working_df["snapshot_date"],
        "market_buyzone_pct": working_df["market_buyzone_pct"],
    })

    logger.info("Built %s prediction snapshot rows", len(snapshot_df))
    return snapshot_df


# ----------------------------------------------------------
# Upsert prediction snapshot rows
# ----------------------------------------------------------
def upsert_prediction_snapshot(df: pd.DataFrame) -> int:
    if df.empty:
        logger.info("No prediction snapshot rows to upsert")
        return 0

    engine = get_engine()

    rows = [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    sql = text("""
        INSERT INTO prediction_snapshot (
            prediction_date,
            ticker,
            name,
            sector,
            industry,
            current_price,
            prediction_score,
            score_bucket,
            calibration_bucket,
            signal_tier,
            signal_horizon,
            expected_win_rate_trend,
            expected_return_trend,
            expected_win_rate_trend_pct,
            expected_return_trend_pct,
            setup_label,
            phase,
            prior_phase,
            goi_zone,
            buyzone_flag,
            confidence,
            confidence_pct,
            greer_value_score,
            greer_yield_score,
            greer_star_rating,
            gfv_price,
            gfv_status,
            snapshot_date,
            market_buyzone_pct,
            updated_at
        )
        VALUES (
            :prediction_date,
            :ticker,
            :name,
            :sector,
            :industry,
            :current_price,
            :prediction_score,
            :score_bucket,
            :calibration_bucket,
            :signal_tier,
            :signal_horizon,
            :expected_win_rate_trend,
            :expected_return_trend,
            :expected_win_rate_trend_pct,
            :expected_return_trend_pct,
            :setup_label,
            :phase,
            :prior_phase,
            :goi_zone,
            :buyzone_flag,
            :confidence,
            :confidence_pct,
            :greer_value_score,
            :greer_yield_score,
            :greer_star_rating,
            :gfv_price,
            :gfv_status,
            :snapshot_date,
            :market_buyzone_pct,
            now()
        )
        ON CONFLICT (prediction_date, ticker)
        DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            current_price = EXCLUDED.current_price,
            prediction_score = EXCLUDED.prediction_score,
            score_bucket = EXCLUDED.score_bucket,
            calibration_bucket = EXCLUDED.calibration_bucket,
            signal_tier = EXCLUDED.signal_tier,
            signal_horizon = EXCLUDED.signal_horizon,
            expected_win_rate_trend = EXCLUDED.expected_win_rate_trend,
            expected_return_trend = EXCLUDED.expected_return_trend,
            expected_win_rate_trend_pct = EXCLUDED.expected_win_rate_trend_pct,
            expected_return_trend_pct = EXCLUDED.expected_return_trend_pct,
            setup_label = EXCLUDED.setup_label,
            phase = EXCLUDED.phase,
            prior_phase = EXCLUDED.prior_phase,
            goi_zone = EXCLUDED.goi_zone,
            buyzone_flag = EXCLUDED.buyzone_flag,
            confidence = EXCLUDED.confidence,
            confidence_pct = EXCLUDED.confidence_pct,
            greer_value_score = EXCLUDED.greer_value_score,
            greer_yield_score = EXCLUDED.greer_yield_score,
            greer_star_rating = EXCLUDED.greer_star_rating,
            gfv_price = EXCLUDED.gfv_price,
            gfv_status = EXCLUDED.gfv_status,
            snapshot_date = EXCLUDED.snapshot_date,
            market_buyzone_pct = EXCLUDED.market_buyzone_pct,
            updated_at = now()
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    logger.info("Upserted %s prediction snapshot rows", len(rows))
    return len(rows)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> None:
    logger.info("Starting prediction snapshot refresh")
    input_df = load_prediction_inputs()
    snapshot_df = build_prediction_snapshot(input_df)
    row_count = upsert_prediction_snapshot(snapshot_df)
    logger.info("Prediction snapshot refresh complete: %s rows", row_count)


if __name__ == "__main__":
    main()