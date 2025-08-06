# greer_opportunity_periods.py
"""
Module to refresh the Greer materialized view, rebuild daily snapshot history,
and compute opportunity entry/exit periods.
"""
import os
from sqlalchemy import create_engine, text

# Database connection URL
DB_URL = os.getenv("DATABASE_URL", "postgresql://greer_user:@localhost:5432/yfinance_db")
engine = create_engine(DB_URL)


def refresh_mv():
    """Refresh the latest_company_snapshot materialized view"""
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW public.latest_company_snapshot;"))
        print("🔄 latest_company_snapshot refreshed")


def rebuild_snapshot():
    """Truncate and repopulate company_snapshot with last-known daily values"""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE public.company_snapshot;"))
        conn.execute(text("""
        INSERT INTO public.company_snapshot (
          snapshot_date, ticker,
          greer_value_score, greer_yield_score,
          buyzone_flag, fvg_last_direction
        )
        WITH daily AS (
          SELECT
            p.ticker,
            p.date::timestamp AS snapshot_date,
            (SELECT gs.greer_score
               FROM greer_scores gs
              WHERE gs.ticker = p.ticker
                AND gs.report_date <= p.date
              ORDER BY gs.report_date DESC
              LIMIT 1
            ) AS greer_value_score,
            (SELECT yd.score
               FROM greer_yields_daily yd
              WHERE yd.ticker = p.ticker
                AND yd.date <= p.date
              ORDER BY yd.date DESC
              LIMIT 1
            ) AS greer_yield_score,
            COALESCE((
              SELECT gbd.in_buyzone
                FROM greer_buyzone_daily gbd
               WHERE gbd.ticker = p.ticker
                 AND gbd.date <= p.date
               ORDER BY gbd.date DESC
               LIMIT 1
            ), FALSE) AS buyzone_flag,
            (SELECT fvg.direction
               FROM fair_value_gaps fvg
              WHERE fvg.ticker = p.ticker
                AND fvg.date <= p.date
                AND fvg.mitigated = false
              ORDER BY fvg.date DESC
              LIMIT 1
            ) AS fvg_last_direction
          FROM prices p
          WHERE p.date >= '2021-01-01'
        )
        SELECT
          snapshot_date,
          ticker,
          greer_value_score,
          greer_yield_score,
          buyzone_flag,
          fvg_last_direction
        FROM daily

        ON CONFLICT (ticker, snapshot_date) DO UPDATE
          SET
            greer_value_score  = EXCLUDED.greer_value_score,
            greer_yield_score  = EXCLUDED.greer_yield_score,
            buyzone_flag       = EXCLUDED.buyzone_flag,
            fvg_last_direction = EXCLUDED.fvg_last_direction;
        """))
        print("🗃️  company_snapshot rebuilt")


def rebuild_periods():
    """Truncate and rebuild greer_opportunity_periods from snapshot history"""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE public.greer_opportunity_periods;"))
        conn.execute(text("""
        WITH qualifies AS (
          SELECT
            ticker,
            snapshot_date,
            (greer_value_score  >= 50
             AND greer_yield_score >= 3
             AND buyzone_flag IS TRUE
             AND fvg_last_direction = 'bullish'
            ) AS meets
          FROM public.company_snapshot
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
        INSERT INTO public.greer_opportunity_periods (ticker, entry_date, exit_date)
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
        print("📜 greer_opportunity_periods rebuilt")


if __name__ == "__main__":
    refresh_mv()
    rebuild_snapshot()
    rebuild_periods()
    print("✅ greer_opportunity_periods module complete")
