# create_greer_opportunities_snapshot.py
from sqlalchemy import text
from db import get_engine
import time

def create_and_refresh_greer_opportunities_snapshot():
    start = time.time()
    engine = get_engine()

    # Create materialized view and indexes
    create_query = text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS greer_opportunities_snapshot AS
        WITH current_ops AS (
          SELECT ticker, MAX(entry_date) AS last_entry_date
          FROM public.greer_opportunity_periods
          WHERE exit_date >= CURRENT_DATE AT TIME ZONE 'America/New_York' - INTERVAL '1 day'
          GROUP BY ticker
        )
        SELECT
          l.ticker,
          l.greer_value_score AS greer_value,
          l.greer_yield_score AS yield_score,
          l.buyzone_flag,
          l.fvg_last_direction,
          o.last_entry_date
        FROM latest_company_snapshot AS l
        JOIN current_ops AS o ON l.ticker = o.ticker
        WHERE l.greer_value_score >= 50
          AND l.greer_yield_score >= 3
          AND l.buyzone_flag IS TRUE
          AND l.fvg_last_direction = 'bullish'
        ORDER BY l.greer_value_score DESC;
        
        -- Unique index for CONCURRENT refresh
        CREATE UNIQUE INDEX IF NOT EXISTS idx_greer_opps_ticker_unique ON greer_opportunities_snapshot (ticker);
        -- Non-unique index for filtering
        CREATE INDEX IF NOT EXISTS idx_greer_opps_ticker ON greer_opportunities_snapshot (ticker);
        -- Trigram index for LIKE searches
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE INDEX IF NOT EXISTS idx_greer_opps_ticker_trgm ON greer_opportunities_snapshot USING GIN (ticker gin_trgm_ops);
    """)

    # Refresh materialized view
    refresh_query = text("REFRESH MATERIALIZED VIEW CONCURRENTLY greer_opportunities_snapshot")

    try:
        # Create view and indexes
        with engine.connect() as conn:
            conn.execute(create_query)
            conn.commit()
        print("Materialized view greer_opportunities_snapshot created or updated with indexes")

        # Refresh view
        with engine.connect() as conn:
            conn.execute(refresh_query)
            conn.commit()
        print(f"Materialized view refreshed in {time.time() - start} seconds")

    except Exception as e:
        print(f"Error: {e}")
        # Fallback to non-concurrent refresh if needed
        try:
            with engine.connect() as conn:
                conn.execute(text("REFRESH MATERIALIZED VIEW greer_opportunities_snapshot"))
                conn.commit()
            print(f"Non-concurrent refresh completed in {time.time() - start} seconds")
        except Exception as e2:
            print(f"Non-concurrent refresh failed: {e2}")

if __name__ == "__main__":
    create_and_refresh_greer_opportunities_snapshot()