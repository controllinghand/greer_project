# update_greer_opportunities.py
from sqlalchemy import text
from db import get_engine
import time

start = time.time()
engine = get_engine()
query = text("""
    INSERT INTO greer_opportunity_periods (ticker, entry_date, exit_date)
    SELECT
      ticker,
      date AS entry_date,
      date + INTERVAL '1 day' AS exit_date
    FROM prices
    WHERE date >= CURRENT_DATE AT TIME ZONE 'America/New_York' - INTERVAL '1 day'
    ON CONFLICT (ticker, entry_date) DO NOTHING;
""")
with engine.connect() as conn:
    conn.execute(query)
    conn.commit()
print(f"greer_opportunity_periods updated in {time.time() - start} seconds")