# run_opportunities.py
from datetime import datetime, timedelta
from sqlalchemy import text
from pytz import timezone
from db import get_engine
import time
import gc

engine = get_engine()
batch_size = timedelta(days=7)

max_date_query = text("SELECT COALESCE(MAX(snapshot_date)::date, '1900-01-01'::date) FROM company_snapshot")
with engine.connect() as conn:
    max_date = conn.execute(max_date_query).scalar() or '1900-01-01'
max_date = datetime.strptime(str(max_date), '%Y-%m-%d').replace(tzinfo=timezone('America/New_York'))
end_date = datetime.now(tz=timezone('America/New_York'))

start = time.time()
current_start = max_date
while current_start < end_date:
    current_end = min(current_start + batch_size, end_date)
    query = text("""
        SET jit = OFF;
        INSERT INTO public.company_snapshot (
          snapshot_date, ticker, greer_value_score, greer_yield_score, buyzone_flag, fvg_last_direction
        )
        SELECT
          p.date::timestamp, p.ticker,
          gs.greer_score, yd.score,
          COALESCE(gbd.in_buyzone, FALSE),
          fvg.direction
        FROM prices p
        LEFT JOIN LATERAL (
          SELECT greer_score
          FROM greer_scores
          WHERE ticker = p.ticker AND report_date <= p.date
          ORDER BY report_date DESC LIMIT 1
        ) gs ON true
        LEFT JOIN LATERAL (
          SELECT score
          FROM greer_yields_daily
          WHERE ticker = p.ticker AND date <= p.date
          ORDER BY date DESC LIMIT 1
        ) yd ON true
        LEFT JOIN LATERAL (
          SELECT in_buyzone
          FROM greer_buyzone_daily
          WHERE ticker = p.ticker AND date = p.date  -- Exact date match
          ORDER BY date DESC LIMIT 1
        ) gbd ON true
        LEFT JOIN LATERAL (
          SELECT direction
          FROM fair_value_gaps
          WHERE ticker = p.ticker AND date <= p.date
          ORDER BY date DESC LIMIT 1
        ) fvg ON true
        WHERE p.date BETWEEN :start AND :end
        ON CONFLICT (ticker, snapshot_date) DO UPDATE SET
            greer_value_score = EXCLUDED.greer_value_score,
            greer_yield_score = EXCLUDED.greer_yield_score,
            buyzone_flag = EXCLUDED.buyzone_flag,
            fvg_last_direction = EXCLUDED.fvg_last_direction;
        SET jit = ON;
    """)
    with engine.connect() as conn:
        conn.execute(query, {'start': current_start.date(), 'end': current_end.date()})
        conn.commit()
    print(f"Batch INSERT for {current_start.date()} to {current_end.date()} completed")
    current_start = current_end + timedelta(days=1)
    gc.collect()
print(f"Total Batch INSERT completed in {time.time() - start} seconds")