# refresh_snapshot.py

import os
import logging
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "refresh_snapshot.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# SQL
# ----------------------------------------------------------
# NOTE:
# - Inserts exactly ONE row per ticker per day (snapshot_date = date_trunc('day', now()))
# - Uses "last known <= today" logic for each metric
# - Pulls close from prices (latest close <= today)
# - Pulls gfv_price from greer_fair_value_daily (latest <= today)
# - Pulls stars from companies (current stars)
INSERT_TODAY_SNAPSHOT_SQL = """
INSERT INTO public.company_snapshot (
  snapshot_date,
  ticker,
  greer_value_score,
  greer_yield_score,
  buyzone_flag,
  fvg_last_direction,
  close,
  gfv_price,
  greer_star_rating
)
WITH universe AS (
  SELECT ticker FROM public.companies WHERE delisted = false
),
asof AS (
  SELECT (CURRENT_DATE)::date AS asof_date
)
SELECT
  date_trunc('day', now())::timestamp AS snapshot_date,
  u.ticker,

  (SELECT gs.greer_score
     FROM public.greer_scores gs
    WHERE gs.ticker = u.ticker
      AND gs.report_date <= (SELECT asof_date FROM asof)
    ORDER BY gs.report_date DESC
    LIMIT 1
  ) AS greer_value_score,

  (SELECT yd.score
     FROM public.greer_yields_daily yd
    WHERE yd.ticker = u.ticker
      AND yd.date <= (SELECT asof_date FROM asof)
    ORDER BY yd.date DESC
    LIMIT 1
  ) AS greer_yield_score,

  COALESCE((
    SELECT gbd.in_buyzone
      FROM public.greer_buyzone_daily gbd
     WHERE gbd.ticker = u.ticker
       AND gbd.date <= (SELECT asof_date FROM asof)
     ORDER BY gbd.date DESC
     LIMIT 1
  ), FALSE) AS buyzone_flag,

  (SELECT fvg.direction
     FROM public.fair_value_gaps fvg
    WHERE fvg.ticker = u.ticker
      AND fvg.date <= (SELECT asof_date FROM asof)
      AND fvg.mitigated = false
    ORDER BY fvg.date DESC
    LIMIT 1
  ) AS fvg_last_direction,

  (SELECT p.close
     FROM public.prices p
    WHERE p.ticker = u.ticker
      AND p.date <= (SELECT asof_date FROM asof)
    ORDER BY p.date DESC
    LIMIT 1
  ) AS close,

  (SELECT gfd.gfv_price
     FROM public.greer_fair_value_daily gfd
    WHERE gfd.ticker = u.ticker
      AND gfd.date <= (SELECT asof_date FROM asof)
    ORDER BY gfd.date DESC
    LIMIT 1
  ) AS gfv_price,

  (SELECT c.greer_star_rating
     FROM public.companies c
    WHERE c.ticker = u.ticker
    LIMIT 1
  ) AS greer_star_rating

FROM universe u
ON CONFLICT (ticker, snapshot_date) DO UPDATE
SET
  greer_value_score  = EXCLUDED.greer_value_score,
  greer_yield_score  = EXCLUDED.greer_yield_score,
  buyzone_flag       = EXCLUDED.buyzone_flag,
  fvg_last_direction = EXCLUDED.fvg_last_direction,
  close              = EXCLUDED.close,
  gfv_price          = EXCLUDED.gfv_price,
  greer_star_rating  = EXCLUDED.greer_star_rating;
"""

REFRESH_MATVIEW_SQL = "REFRESH MATERIALIZED VIEW CONCURRENTLY public.latest_company_snapshot;"

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
print("🔄 Writing today’s company_snapshot + refreshing latest_company_snapshot …")
logger.info("Starting refresh_snapshot.py")

engine = get_engine()
try:
  with engine.begin() as conn:
      # 1) Make sure matview is fresh for the website pages
      conn.execute(text(REFRESH_MATVIEW_SQL))
      logger.info("latest_company_snapshot refreshed")

      # 2) Write today's snapshot (history-safe)
      conn.execute(text(INSERT_TODAY_SNAPSHOT_SQL))
      logger.info("company_snapshot upserted for today")

  print("✅ Daily snapshot written + latest_company_snapshot refreshed")
  logger.info("refresh_snapshot.py completed successfully")

except Exception as e:
  print(f"❌ Error in refresh_snapshot.py: {e}")
  logger.exception("Error in refresh_snapshot.py")
  raise
