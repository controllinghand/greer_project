# update_all_star_ratings.py

import logging
from psycopg2.extras import DictCursor
from db import get_psycopg_connection  # adjust import to match your project

logging.basicConfig(
    filename="update_all_star_ratings.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def compute_star_rating(gv_above50_count, yield_score, gfv_status):
    """
    Compute a 0–3 star rating based on your rules:

      - +1 star if above50_count == 6  (Greer Value star)
      - +1 star if yield_score == 4    (Yield star)
      - +1 star if gfv_status == 'gold' (Greer Fair Value star)

    Returns integer 0..3.
    """
    stars = 0
    if gv_above50_count == 6:
        stars += 1
    if yield_score == 4:
        stars += 1
    # assuming gfv_status is stored as lowercase or uppercase? adjust as needed
    if gfv_status is not None and str(gfv_status).lower() == 'gold':
        stars += 1
    return stars

def fetch_latest_yield_scores(cur):
    """
    Returns dict: ticker -> latest yield_score (int)
    from greer_yields_daily.
    """
    cur.execute("""
        SELECT ticker, score
        FROM greer_yields_daily y
        WHERE (ticker, date) IN (
            SELECT ticker, MAX(date)
            FROM greer_yields_daily
            GROUP BY ticker
        );
    """)
    return {row['ticker']: row['score'] for row in cur.fetchall()}

def fetch_latest_gfv_status(cur):
    """
    Returns dict: ticker -> latest gfv_status (text)
    from greer_fair_value_daily.
    """
    cur.execute("""
        SELECT ticker, gfv_status
        FROM greer_fair_value_daily f
        WHERE (ticker, date) IN (
            SELECT ticker, MAX(date)
            FROM greer_fair_value_daily
            GROUP BY ticker
        );
    """)
    return {row['ticker']: row['gfv_status'] for row in cur.fetchall()}

def update_all_star_ratings():
    conn = get_psycopg_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            yield_map = fetch_latest_yield_scores(cur)
            gfv_map = fetch_latest_gfv_status(cur)

            # Fetch all companies with their latest greer_scores above_50_count
            cur.execute("""
                SELECT gs.ticker, gs.above_50_count
                FROM greer_scores gs
                WHERE (gs.ticker, gs.report_date) IN (
                    SELECT ticker, MAX(report_date)
                    FROM greer_scores
                    GROUP BY ticker
                );
            """)
            rows = cur.fetchall()

            updates = []
            for r in rows:
                ticker = r['ticker']
                gv_count = r['above_50_count']
                yield_score = yield_map.get(ticker)
                gfv_status = gfv_map.get(ticker)

                stars = compute_star_rating(
                    gv_above50_count = gv_count,
                    yield_score = yield_score,
                    gfv_status = gfv_status
                )

                updates.append((stars, ticker))

            logging.info("Updating star rating for %d tickers", len(updates))

            cur.executemany("""
                UPDATE public.companies
                SET greer_star_rating = %s
                WHERE ticker = %s;
            """, updates)

        conn.commit()
        logging.info("Star-ratings updated successfully.")
    except Exception:
        conn.rollback()
        logging.exception("Error updating star ratings.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    update_all_star_ratings()
