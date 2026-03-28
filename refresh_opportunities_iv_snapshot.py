# refresh_opportunities_iv_snapshot.py

import logging
import os
import sys
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine


# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "refresh_opportunities_iv_snapshot.log")

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
# Clean values
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
# Load raw opportunities
# ----------------------------------------------------------
def load_raw_opportunities() -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        WITH live_bull_gaps AS (
          SELECT ticker, date AS entry_date
          FROM public.fair_value_gaps
          WHERE direction = 'bullish'
            AND mitigated = false
        ),
        last_entry AS (
          SELECT ticker, MAX(entry_date) AS last_entry_date
          FROM live_bull_gaps
          GROUP BY ticker
        ),
        latest_prices AS (
          SELECT ticker, close AS current_price, date
          FROM prices
          WHERE (ticker, date) IN (
               SELECT ticker, MAX(date) FROM prices GROUP BY ticker
          )
        ),
        latest_gfv AS (
          SELECT DISTINCT ON (g.ticker)
            g.ticker,
            g.gfv_price,
            g.gfv_status
          FROM greer_fair_value_daily g
          ORDER BY g.ticker, g.date DESC
        ),
        latest_sector AS (
          SELECT
            sector,
            buyzone_pct
          FROM sector_summary_daily
          WHERE summary_date = (
            SELECT MAX(summary_date) FROM sector_summary_daily
          )
        ),
        latest_company_index AS (
          SELECT DISTINCT ON (ticker)
            ticker,
            greer_company_index,
            health_pct,
            direction_pct,
            opportunity_pct,
            phase,
            confidence
          FROM greer_company_index_daily
          ORDER BY ticker, date DESC
        ),
        recent_iv AS (
          SELECT DISTINCT ON (ivs.ticker)
            ivs.ticker,
            ivs.iv_atm,
            ivs.expiry AS iv_expiry
          FROM iv_summary ivs
          JOIN (
            SELECT ticker, MAX(fetch_date) AS max_fetch_date
            FROM iv_summary
            GROUP BY ticker
          ) mf
            ON mf.ticker = ivs.ticker
           AND mf.max_fetch_date = ivs.fetch_date
          WHERE ivs.expiry >= CURRENT_DATE
          ORDER BY ivs.ticker, ivs.expiry ASC
        )
        SELECT
          l.ticker,
          c.name,
          c.sector,
          c.industry,
          c.greer_star_rating AS stars,

          -- 🔥 USE REAL COMPANY CYCLE DATA
          lci.greer_company_index,
          lci.health_pct,
          lci.direction_pct,
          lci.opportunity_pct,
          lci.phase,
          lci.confidence * 100 AS confidence,

          -- valuation + signals
          l.greer_value_score AS greer_value,
          l.greer_yield_score AS yield_score,
          l.buyzone_flag,
          l.fvg_last_direction,

          le.last_entry_date,
          p.current_price,
          gfv.gfv_price,
          gfv.gfv_status,
          gfv.gfv_price * 0.75 AS gfv_mos,

          riv.iv_atm,
          riv.iv_expiry,

          (100.0 - s.buyzone_pct) AS sector_direction_pct

        FROM last_entry le
        JOIN latest_company_snapshot l ON l.ticker = le.ticker
        JOIN companies c ON l.ticker = c.ticker
        JOIN latest_prices p ON p.ticker = l.ticker
        JOIN latest_gfv gfv ON gfv.ticker = l.ticker
        JOIN latest_company_index lci ON lci.ticker = l.ticker

        LEFT JOIN recent_iv riv ON riv.ticker = l.ticker
        LEFT JOIN latest_sector s ON c.sector = s.sector

        WHERE l.greer_value_score >= 50
          AND l.greer_yield_score >= 3
          AND l.buyzone_flag = TRUE
          AND l.fvg_last_direction = 'bullish'
          AND p.current_price < gfv.gfv_price * 0.75
          AND c.delisted = FALSE
    """)

    df = pd.read_sql(query, engine)
    logger.info("Loaded %s raw opportunity rows", len(df))
    return df


# ----------------------------------------------------------
# Enrich (FINAL CLEAN VERSION)
# ----------------------------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Confidence already scaled in SQL → just clean
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(1)

    # Add snapshot date
    df["snapshot_date"] = pd.Timestamp(date.today())

    return df


# ----------------------------------------------------------
# Upsert
# ----------------------------------------------------------
def upsert(df: pd.DataFrame):
    if df.empty:
        return

    engine = get_engine()

    rows = [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    sql = text("""
        INSERT INTO opportunities_iv_snapshot (
            snapshot_date,
            ticker,
            name,
            sector,
            industry,
            stars,
            greer_company_index,
            health_pct,
            direction_pct,
            opportunity_pct,
            phase,
            confidence,
            greer_value,
            yield_score,
            buyzone_flag,
            fvg_last_direction,
            gfv_status,
            sector_direction_pct,
            iv_atm,
            iv_expiry,
            current_price,
            gfv_price,
            gfv_mos,
            last_entry_date,
            updated_at
        )
        VALUES (
            :snapshot_date,
            :ticker,
            :name,
            :sector,
            :industry,
            :stars,
            :greer_company_index,
            :health_pct,
            :direction_pct,
            :opportunity_pct,
            :phase,
            :confidence,
            :greer_value,
            :yield_score,
            :buyzone_flag,
            :fvg_last_direction,
            :gfv_status,
            :sector_direction_pct,
            :iv_atm,
            :iv_expiry,
            :current_price,
            :gfv_price,
            :gfv_mos,
            :last_entry_date,
            now()
        )
        ON CONFLICT (snapshot_date, ticker)
        DO UPDATE SET
            greer_company_index = EXCLUDED.greer_company_index,
            health_pct = EXCLUDED.health_pct,
            direction_pct = EXCLUDED.direction_pct,
            opportunity_pct = EXCLUDED.opportunity_pct,
            phase = EXCLUDED.phase,
            confidence = EXCLUDED.confidence,
            iv_atm = EXCLUDED.iv_atm,
            updated_at = now()
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    logger.info("Upserted %s opportunity rows", len(rows))


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    raw = load_raw_opportunities()
    enriched = enrich(raw)
    upsert(enriched)


if __name__ == "__main__":
    main()