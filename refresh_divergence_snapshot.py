# refresh_divergence_snapshot.py

import logging
import os
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine


# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "refresh_divergence_snapshot.log")

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
# Load divergence snapshot data
# ----------------------------------------------------------
def load_divergence_snapshot() -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        WITH latest_company_index AS (
            SELECT DISTINCT ON (ticker)
                ticker,
                date,
                health_pct,
                direction_pct,
                opportunity_pct,
                greer_company_index,
                phase,
                confidence
            FROM greer_company_index_daily
            ORDER BY ticker, date DESC
        ),
        company_index_50dma AS (
            SELECT
                ticker,
                date,
                AVG(greer_company_index) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                ) AS company_index_50dma
            FROM greer_company_index_daily
        ),
        latest_company_index_50dma AS (
            SELECT DISTINCT ON (ticker)
                ticker,
                date,
                company_index_50dma
            FROM company_index_50dma
            ORDER BY ticker, date DESC
        ),
        latest_snapshot AS (
            SELECT
                ticker,
                greer_value_score,
                greer_yield_score,
                buyzone_flag,
                fvg_last_direction
            FROM latest_company_snapshot
        ),
        base AS (
            SELECT
                c.ticker,
                c.name,
                c.sector,
                c.industry,
                lci.date AS company_index_date,
                ROUND(lci.greer_company_index::numeric, 2) AS greer_company_index,
                ROUND(lci50.company_index_50dma::numeric, 2) AS company_index_50dma,
                ROUND((lci.greer_company_index - lci50.company_index_50dma)::numeric, 2) AS index_vs_50dma,
                lci.phase,
                ROUND(lci.confidence::numeric * 100.0, 0) AS confidence_pct,
                ROUND(lci.health_pct::numeric, 2) AS health_pct,
                ROUND(lci.direction_pct::numeric, 2) AS direction_pct,
                ROUND(lci.opportunity_pct::numeric, 2) AS opportunity_pct,
                ROUND(pvp.current_price::numeric, 2) AS current_price,
                ROUND(pvp.peak_52w_price::numeric, 2) AS peak_52w_price,
                ROUND(pvp.price_vs_peak_pct::numeric, 2) AS price_vs_peak_pct,
                ls.greer_value_score,
                ls.greer_yield_score,
                ls.buyzone_flag,
                ls.fvg_last_direction,
                CASE
                    WHEN lci.greer_company_index >= COALESCE(lci50.company_index_50dma, 0) + 3 THEN 'Rising'
                    WHEN lci.greer_company_index <= COALESCE(lci50.company_index_50dma, 0) - 3 THEN 'Falling'
                    ELSE 'Flat'
                END AS index_trend
            FROM latest_company_index lci
            JOIN latest_company_index_50dma lci50
              ON lci50.ticker = lci.ticker
            JOIN price_vs_peak_view pvp
              ON pvp.ticker = lci.ticker
            JOIN companies c
              ON c.ticker = lci.ticker
            LEFT JOIN latest_snapshot ls
              ON ls.ticker = lci.ticker
            WHERE COALESCE(c.delisted, false) = false
        ),
        classified AS (
            SELECT
                *,
                CASE
                    WHEN greer_company_index >= 75
                     AND price_vs_peak_pct <= -15
                     AND index_trend IN ('Rising', 'Flat')
                    THEN 'High Conviction Pullback'

                    WHEN greer_company_index >= 70
                     AND price_vs_peak_pct <= -35
                    THEN 'Deep Value Divergence'

                    WHEN greer_company_index >= 70
                     AND price_vs_peak_pct <= -15
                    THEN 'Watchlist Divergence'

                    ELSE 'Other'
                END AS divergence_bucket
            FROM base
        )
        SELECT
            CURRENT_DATE AS snapshot_date,
            ticker,
            name,
            sector,
            industry,
            company_index_date,
            greer_company_index,
            company_index_50dma,
            index_vs_50dma,
            phase,
            confidence_pct,
            health_pct,
            direction_pct,
            opportunity_pct,
            current_price,
            peak_52w_price,
            price_vs_peak_pct,
            greer_value_score,
            greer_yield_score,
            buyzone_flag,
            fvg_last_direction,
            index_trend,
            divergence_bucket
        FROM classified
        WHERE divergence_bucket <> 'Other'
        ORDER BY
            CASE divergence_bucket
                WHEN 'High Conviction Pullback' THEN 1
                WHEN 'Deep Value Divergence' THEN 2
                WHEN 'Watchlist Divergence' THEN 3
                ELSE 99
            END,
            greer_company_index DESC,
            price_vs_peak_pct ASC
    """)

    df = pd.read_sql(query, engine)
    logger.info("Loaded %s divergence snapshot rows", len(df))
    return df


# ----------------------------------------------------------
# Upsert divergence snapshot rows
# ----------------------------------------------------------
def upsert_divergence_snapshot(df: pd.DataFrame) -> int:
    if df.empty:
        logger.info("No divergence snapshot rows to upsert")
        return 0

    engine = get_engine()

    rows = [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    sql = text("""
        INSERT INTO divergence_snapshot (
            snapshot_date,
            ticker,
            name,
            sector,
            industry,
            company_index_date,
            greer_company_index,
            company_index_50dma,
            index_vs_50dma,
            phase,
            confidence_pct,
            health_pct,
            direction_pct,
            opportunity_pct,
            current_price,
            peak_52w_price,
            price_vs_peak_pct,
            greer_value_score,
            greer_yield_score,
            buyzone_flag,
            fvg_last_direction,
            index_trend,
            divergence_bucket,
            updated_at
        )
        VALUES (
            :snapshot_date,
            :ticker,
            :name,
            :sector,
            :industry,
            :company_index_date,
            :greer_company_index,
            :company_index_50dma,
            :index_vs_50dma,
            :phase,
            :confidence_pct,
            :health_pct,
            :direction_pct,
            :opportunity_pct,
            :current_price,
            :peak_52w_price,
            :price_vs_peak_pct,
            :greer_value_score,
            :greer_yield_score,
            :buyzone_flag,
            :fvg_last_direction,
            :index_trend,
            :divergence_bucket,
            now()
        )
        ON CONFLICT (snapshot_date, ticker)
        DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            company_index_date = EXCLUDED.company_index_date,
            greer_company_index = EXCLUDED.greer_company_index,
            company_index_50dma = EXCLUDED.company_index_50dma,
            index_vs_50dma = EXCLUDED.index_vs_50dma,
            phase = EXCLUDED.phase,
            confidence_pct = EXCLUDED.confidence_pct,
            health_pct = EXCLUDED.health_pct,
            direction_pct = EXCLUDED.direction_pct,
            opportunity_pct = EXCLUDED.opportunity_pct,
            current_price = EXCLUDED.current_price,
            peak_52w_price = EXCLUDED.peak_52w_price,
            price_vs_peak_pct = EXCLUDED.price_vs_peak_pct,
            greer_value_score = EXCLUDED.greer_value_score,
            greer_yield_score = EXCLUDED.greer_yield_score,
            buyzone_flag = EXCLUDED.buyzone_flag,
            fvg_last_direction = EXCLUDED.fvg_last_direction,
            index_trend = EXCLUDED.index_trend,
            divergence_bucket = EXCLUDED.divergence_bucket,
            updated_at = now()
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    logger.info("Upserted %s divergence snapshot rows", len(rows))
    return len(rows)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> None:
    logger.info("Starting divergence snapshot refresh")
    snapshot_df = load_divergence_snapshot()
    row_count = upsert_divergence_snapshot(snapshot_df)
    logger.info("Divergence snapshot refresh complete: %s rows", row_count)


if __name__ == "__main__":
    main()