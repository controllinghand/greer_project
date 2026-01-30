# nav.py
"""
Nightly NAV builder for all portfolios.

Writes to:
  portfolio_nav_daily (portfolio_id, nav_date, cash, equity_value, nav)

Logic:
- cash EOD = sum(cash_delta) for events with event_time < (nav_date + 1 day)
- shares EOD per ticker = sum(quantity) for share-impact events with event_time < (nav_date + 1 day)
- equity_value = sum(shares * close) using prices on nav_date, or last close before nav_date
- nav = cash + equity_value

Runs for every portfolio from its start_date through CURRENT_DATE.
"""

import logging
import os
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "nav.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# SQL (only binds :portfolio_id)
# ----------------------------------------------------------
NAV_SQL = text("""
WITH p AS (
  SELECT portfolio_id, start_date
  FROM portfolios
  WHERE portfolio_id = :portfolio_id
),
days AS (
  SELECT generate_series(
    (SELECT start_date FROM p)::date,
    CURRENT_DATE::date,
    interval '1 day'
  )::date AS nav_date
),
cash_eod AS (
  SELECT
    d.nav_date,
    (SELECT COALESCE(starting_cash, 0) FROM portfolios WHERE portfolio_id = (SELECT portfolio_id FROM p))
    + COALESCE(SUM(e.cash_delta), 0) AS cash
  FROM days d
  LEFT JOIN portfolio_events e
    ON e.portfolio_id = (SELECT portfolio_id FROM p)
   AND e.event_time < (d.nav_date + interval '1 day')
  GROUP BY d.nav_date
),
shares_eod AS (
  SELECT
    d.nav_date,
    e.ticker,
    COALESCE(SUM(
      CASE
        -- options-style share events: rely on stored quantity sign (ASSIGN_PUT usually +, CALL_AWAY usually -)
        WHEN e.event_type IN ('ASSIGN_PUT', 'CALL_AWAY') THEN COALESCE(e.quantity, 0)

        -- stock-only events: BUY adds, SELL subtracts
        WHEN e.event_type = 'BUY_SHARES'  THEN COALESCE(e.quantity, 0)
        WHEN e.event_type = 'SELL_SHARES' THEN COALESCE(e.quantity, 0)

        ELSE 0
      END
    ), 0) AS shares
  FROM days d
  LEFT JOIN portfolio_events e
    ON e.portfolio_id = (SELECT portfolio_id FROM p)
   AND e.event_time < (d.nav_date + interval '1 day')
  WHERE e.ticker IS NOT NULL AND e.ticker <> ''
  GROUP BY d.nav_date, e.ticker
),
equity_eod AS (
  SELECT
    s.nav_date,
    COALESCE(SUM(
      s.shares *
      COALESCE(
        pr.close,
        (
          SELECT pr2.close
          FROM prices pr2
          WHERE pr2.ticker = s.ticker
            AND pr2.date < s.nav_date
          ORDER BY pr2.date DESC
          LIMIT 1
        ),
        0
      )
    ), 0) AS equity_value
  FROM shares_eod s
  LEFT JOIN prices pr
    ON pr.ticker = s.ticker
   AND pr.date = s.nav_date
  GROUP BY s.nav_date
),
final_nav AS (
  SELECT
    (SELECT portfolio_id FROM p) AS portfolio_id,
    c.nav_date,
    c.cash,
    COALESCE(e.equity_value, 0) AS equity_value,
    (c.cash + COALESCE(e.equity_value, 0)) AS nav
  FROM cash_eod c
  LEFT JOIN equity_eod e
    ON e.nav_date = c.nav_date
)
INSERT INTO portfolio_nav_daily (portfolio_id, nav_date, cash, equity_value, nav)
SELECT portfolio_id, nav_date, cash, equity_value, nav
FROM final_nav
ON CONFLICT (portfolio_id, nav_date)
DO UPDATE SET
  cash = EXCLUDED.cash,
  equity_value = EXCLUDED.equity_value,
  nav = EXCLUDED.nav;
""")

COUNT_DAYS_SQL = text("""
SELECT (CURRENT_DATE::date - start_date::date + 1) AS days
FROM portfolios
WHERE portfolio_id = :portfolio_id
""")

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def load_portfolios():
    engine = get_engine()
    q = text("SELECT portfolio_id, code, name, start_date FROM portfolios ORDER BY portfolio_id;")
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return rows

def run_nav_for_portfolio(portfolio_id: int, code: str):
    engine = get_engine()
    try:
        with engine.begin() as conn:
            # Upsert NAV rows
            conn.execute(NAV_SQL, {"portfolio_id": int(portfolio_id)})

            # How many calendar days in range (for a friendly log message)
            days = conn.execute(COUNT_DAYS_SQL, {"portfolio_id": int(portfolio_id)}).scalar() or 0

        msg = f"✅ NAV complete for {code} (pid={portfolio_id}). Days in range: {int(days)}"
        print(msg)
        logger.info(msg)
        return True
    except Exception as e:
        msg = f"❌ NAV failed for {code} (pid={portfolio_id}): {e}"
        print(msg)
        logger.exception(msg)
        return False

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    portfolios = load_portfolios()
    if not portfolios:
        print("No portfolios found. Nothing to do.")
        return

    ok = 0
    bad = 0

    for p in portfolios:
        pid = int(p["portfolio_id"])
        code = (p["code"] or f"pid_{pid}").strip()
        success = run_nav_for_portfolio(pid, code)
        if success:
            ok += 1
        else:
            bad += 1

    print(f"\nNAV finished. Portfolios ok={ok}, failed={bad}")
    logger.info(f"NAV finished. Portfolios ok={ok}, failed={bad}")

if __name__ == "__main__":
    main()
