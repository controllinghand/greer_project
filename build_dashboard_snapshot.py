# build_dashboard_snapshot.py
# ----------------------------------------------------------
# build_dashboard_snapshot.py
# - Builds a fast, read-only dashboard snapshot table
# - Source: companies + latest_company_snapshot + latest price + latest GFV
# ----------------------------------------------------------

import logging
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Build snapshot
# ----------------------------------------------------------
def main():
    engine = get_engine()

    ddl = """
    CREATE TABLE IF NOT EXISTS dashboard_snapshot (
        ticker text PRIMARY KEY,
        name text,
        sector text,
        industry text,
        greer_star_rating int,
        greer_value_score numeric,
        above_50_count int,
        greer_yield_score int,
        buyzone_flag boolean,
        fvg_last_direction text,
        current_price numeric,
        gfv_price numeric,
        gfv_status text,
        snapshot_date date NOT NULL DEFAULT CURRENT_DATE
    );
    """

    refresh_sql = """
    TRUNCATE dashboard_snapshot;

    WITH latest_prices AS (
      SELECT DISTINCT ON (ticker)
        ticker,
        close AS current_price
      FROM prices
      ORDER BY ticker, date DESC
    ),
    latest_gfv AS (
      SELECT DISTINCT ON (ticker)
        ticker,
        gfv_price,
        gfv_status
      FROM greer_fair_value_daily
      ORDER BY ticker, date DESC
    )
    INSERT INTO dashboard_snapshot (
      ticker, name, sector, industry,
      greer_star_rating, greer_value_score, above_50_count, greer_yield_score,
      buyzone_flag, fvg_last_direction,
      current_price, gfv_price, gfv_status,
      snapshot_date
    )
    SELECT
      c.ticker,
      c.name,
      c.sector,
      c.industry,
      c.greer_star_rating,
      s.greer_value_score,
      s.above_50_count,
      s.greer_yield_score,
      s.buyzone_flag,
      s.fvg_last_direction,
      p.current_price,
      g.gfv_price,
      g.gfv_status,
      CURRENT_DATE
    FROM companies c
    JOIN latest_company_snapshot s
      ON c.ticker = s.ticker
    LEFT JOIN latest_prices p
      ON c.ticker = p.ticker
    LEFT JOIN latest_gfv g
      ON c.ticker = g.ticker
    WHERE c.delisted = FALSE;
    """

    logger.info("Ensuring dashboard_snapshot exists…")
    with engine.begin() as conn:
        conn.execute(text(ddl))

    logger.info("Refreshing dashboard_snapshot…")
    with engine.begin() as conn:
        conn.execute(text(refresh_sql))

    logger.info("✅ dashboard_snapshot refreshed successfully")

if __name__ == "__main__":
    main()
