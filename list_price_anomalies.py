# list_price_anomalies.py
from db import get_engine
import pandas as pd
from sqlalchemy import text

def find_anomalies_in_db(ticker: str | None, start: str | None, end: str | None, pct_jump: float = 0.25) -> pd.DataFrame:
    engine = get_engine()
    where = []
    params = {}
    if ticker:
        where.append("p.ticker = :t")
        params["t"] = ticker
    if start:
        where.append("p.date >= :start")
        params["start"] = start
    if end:
        where.append("p.date <= :end")
        params["end"] = end

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        WITH ranked AS (
          SELECT
            p.ticker,
            p.date,
            p.close,
            LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.date) AS prev_close
          FROM prices p
          {where_sql}
        )
        SELECT ticker, date, close, prev_close,
               ROUND(ABS(close/prev_close - 1.0)*100, 2) AS pct_change
        FROM ranked
        WHERE prev_close IS NOT NULL
          AND ABS(close/prev_close - 1) > :pct
        ORDER BY date DESC, ticker;
    """
    params["pct"] = pct_jump
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)

if __name__ == "__main__":
    df = find_anomalies_in_db(None, "2025-05-10", None, pct_jump=0.25)
    print(f"Found {len(df)} suspect rows")
    print(df.to_string(index=False))
