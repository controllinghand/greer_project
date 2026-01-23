# greer_opportunity_gfv_periods.py
"""
Rebuild greer_opportunity_gfv_periods from snapshot history with GFV MOS criterion.

IMPORTANT:
- Daily workflow should NOT truncate/rebuild company_snapshot.
- company_snapshot should be written daily by refresh_snapshot.py.
- This script can optionally backfill snapshot history via --reload (destructive) or --start-date.
"""

import argparse
import logging
from sqlalchemy import text

from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

engine = get_engine()

# ----------------------------------------------------------
# Refresh MV
# ----------------------------------------------------------
def refresh_mv(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW public.latest_company_snapshot;"))
    logger.info("🔄 latest_company_snapshot refreshed")

# ----------------------------------------------------------
# Optional: Backfill snapshot (ONLY when needed)
# ----------------------------------------------------------
def backfill_snapshot(conn, start_date: str, reload: bool):
    """
    Backfill company_snapshot from prices + last-known daily values.
    - If reload=True: TRUNCATE and rebuild (DESTRUCTIVE)
    - Else: UPSERT for the chosen date range (safe / repeatable)
    """
    if reload:
        logger.warning("🧨 RELOAD requested: truncating public.company_snapshot (DESTRUCTIVE)")
        conn.execute(text("TRUNCATE public.company_snapshot;"))

    # NOTE: includes greer_star_rating via companies
    conn.execute(
        text("""
        INSERT INTO public.company_snapshot (
          snapshot_date, ticker,
          greer_value_score, greer_yield_score,
          buyzone_flag, fvg_last_direction,
          close, gfv_price,
          greer_star_rating
        )
        WITH daily AS (
          SELECT
            p.ticker,
            p.date::timestamp AS snapshot_date,
            p.close,

            (SELECT gs.greer_score
               FROM public.greer_scores gs
              WHERE gs.ticker = p.ticker
                AND gs.report_date <= p.date
              ORDER BY gs.report_date DESC
              LIMIT 1
            ) AS greer_value_score,

            (SELECT yd.score
               FROM public.greer_yields_daily yd
              WHERE yd.ticker = p.ticker
                AND yd.date <= p.date
              ORDER BY yd.date DESC
              LIMIT 1
            ) AS greer_yield_score,

            COALESCE((
              SELECT gbd.in_buyzone
                FROM public.greer_buyzone_daily gbd
               WHERE gbd.ticker = p.ticker
                 AND gbd.date <= p.date
               ORDER BY gbd.date DESC
               LIMIT 1
            ), FALSE) AS buyzone_flag,

            (SELECT fvg.direction
               FROM public.fair_value_gaps fvg
              WHERE fvg.ticker = p.ticker
                AND fvg.date <= p.date
                AND fvg.mitigated = false
              ORDER BY fvg.date DESC
              LIMIT 1
            ) AS fvg_last_direction,

            (SELECT gfd.gfv_price
               FROM public.greer_fair_value_daily gfd
              WHERE gfd.ticker = p.ticker
                AND gfd.date <= p.date
              ORDER BY gfd.date DESC
              LIMIT 1
            ) AS gfv_price,

            (SELECT c.greer_star_rating
               FROM public.companies c
              WHERE c.ticker = p.ticker
              LIMIT 1
            ) AS greer_star_rating

          FROM public.prices p
          WHERE p.date >= :start_date::date
        )
        SELECT
          snapshot_date,
          ticker,
          greer_value_score,
          greer_yield_score,
          buyzone_flag,
          fvg_last_direction,
          close,
          gfv_price,
          greer_star_rating
        FROM daily
        ON CONFLICT (ticker, snapshot_date) DO UPDATE
          SET
            greer_value_score  = EXCLUDED.greer_value_score,
            greer_yield_score  = EXCLUDED.greer_yield_score,
            buyzone_flag       = EXCLUDED.buyzone_flag,
            fvg_last_direction = EXCLUDED.fvg_last_direction,
            close              = EXCLUDED.close,
            gfv_price          = EXCLUDED.gfv_price,
            greer_star_rating  = EXCLUDED.greer_star_rating;
        """),
        {"start_date": start_date},
    )

    logger.info("🗃️ company_snapshot backfill complete (start_date=%s, reload=%s)", start_date, reload)

# ----------------------------------------------------------
# Rebuild opportunity periods
# ----------------------------------------------------------
def rebuild_periods(conn):
    conn.execute(text("TRUNCATE public.greer_opportunity_gfv_periods;"))

    conn.execute(text("""
    WITH qualifies AS (
      SELECT
        ticker,
        snapshot_date,
        (greer_value_score  >= 50
         AND greer_yield_score >= 3
         AND buyzone_flag IS TRUE
         AND fvg_last_direction = 'bullish'
         AND close < gfv_price * 0.75
        ) AS meets
      FROM public.company_snapshot
      WHERE gfv_price IS NOT NULL
    ),
    flagged AS (
      SELECT
        *,
        LAG(meets) OVER (PARTITION BY ticker ORDER BY snapshot_date) AS prev_meets
      FROM qualifies
    ),
    islands AS (
      SELECT
        ticker,
        snapshot_date,
        meets,
        SUM(
          CASE WHEN meets AND (prev_meets IS FALSE OR prev_meets IS NULL)
               THEN 1 ELSE 0 END
        ) OVER (PARTITION BY ticker ORDER BY snapshot_date) AS grp
      FROM flagged
    )
    INSERT INTO public.greer_opportunity_gfv_periods (ticker, entry_date, exit_date)
    SELECT
      ticker,
      MIN(snapshot_date)::date AS entry_date,
      MAX(snapshot_date)::date AS exit_date
    FROM islands
    WHERE meets
    GROUP BY ticker, grp
    ON CONFLICT (ticker, entry_date) DO UPDATE
      SET exit_date = EXCLUDED.exit_date;
    """))

    logger.info("📜 greer_opportunity_gfv_periods rebuilt")

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backfill-snapshot", action="store_true", help="Backfill company_snapshot from prices (optional).")
    p.add_argument("--reload", action="store_true", help="With --backfill-snapshot: truncate company_snapshot first (DESTRUCTIVE).")
    p.add_argument("--start-date", default="2021-01-01", help="With --backfill-snapshot: start date for backfill (YYYY-MM-DD).")
    p.add_argument("--refresh-mv", action="store_true", help="Refresh latest_company_snapshot before rebuilding periods.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    try:
        with engine.begin() as conn:
            if args.refresh_mv:
                refresh_mv(conn)

            if args.backfill_snapshot:
                backfill_snapshot(conn, start_date=args.start_date, reload=args.reload)

            rebuild_periods(conn)

        logger.info("✅ greer_opportunity_gfv_periods module complete")

    except Exception as e:
        logger.exception("❌ Script execution failed: %s", e)
        raise
