# greer_value_yield_score.py

import psycopg2
import argparse
import pandas as pd
import os
import logging
from datetime import date
from sqlalchemy import create_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "greer_value_yield_score.log"),
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# DB Connection
# ----------------------------------------------------------
DB_CONN_STRING = "host=localhost dbname=yfinance_db user=greer_user port=5432"

# ----------------------------------------------------------
# Load financials by ticker
# ----------------------------------------------------------
def get_financials(ticker):
    with psycopg2.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXTRACT(YEAR FROM report_date) AS fiscal_year,
                       net_income,
                       free_cash_flow,
                       total_revenue,
                       shares_outstanding,
                       book_value_per_share
                FROM financials
                WHERE ticker = %s
                ORDER BY fiscal_year
            """, (ticker,))
            rows = cur.fetchall()

    return [
        {
            "year": int(row[0]),
            "net_income": row[1],
            "fcf": row[2],
            "revenue": row[3],
            "shares": row[4],
            "book": row[5],
            "eps": (row[1] / row[4]) if row[1] is not None and row[4] else None
        }
        for row in rows
    ]

# ----------------------------------------------------------
# Get the closing price for the last trading day of a fiscal year
# ----------------------------------------------------------
def get_price_on_year_end(ticker, year):
    with psycopg2.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT close FROM prices
                WHERE ticker = %s AND date <= %s
                ORDER BY date DESC LIMIT 1;
            """, (ticker, f"{year}-12-31"))
            row = cur.fetchone()
            return row[0] if row else None


# ----------------------------------------------------------
# Insert result into greer_yields_daily
# ----------------------------------------------------------
def insert_yield_row(ticker, row):
    with psycopg2.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO greer_yields_daily (
                    ticker,
                    date,
                    eps_yield, fcf_yield, revenue_yield, book_yield,
                    avg_eps_yield, avg_fcf_yield, avg_revenue_yield, avg_book_yield,
                    tvpct, tvavg, tvavg_trend, score
                )
                VALUES (
                    %s,
                    %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (ticker, date) DO UPDATE
                  SET eps_yield        = EXCLUDED.eps_yield,
                      fcf_yield        = EXCLUDED.fcf_yield,
                      revenue_yield    = EXCLUDED.revenue_yield,
                      book_yield       = EXCLUDED.book_yield,
                      avg_eps_yield    = EXCLUDED.avg_eps_yield,
                      avg_fcf_yield    = EXCLUDED.avg_fcf_yield,
                      avg_revenue_yield= EXCLUDED.avg_revenue_yield,
                      avg_book_yield   = EXCLUDED.avg_book_yield,
                      tvpct            = EXCLUDED.tvpct,
                      tvavg            = EXCLUDED.tvavg,
                      tvavg_trend      = EXCLUDED.tvavg_trend,
                      score            = EXCLUDED.score;
            """, (
                ticker,
                f"{row['year']}-12-31",      # use fiscal‐year end as the date
                row['eps_yield'],
                row['fcf_yield'],
                row['rev_yield'],
                row['book_yield'],
                row['eps_avg'],
                row['fcf_avg'],
                row['rev_avg'],
                row['book_avg'],
                row['tvpct'],
                row['tvavg'],
                row['tvavg_trend'],
                row['score']
            ))


# ----------------------------------------------------------
# Calculate historical yields and store them
# ----------------------------------------------------------
def calculate_yields(ticker):
    try:
        fundamentals = get_financials(ticker)

        eps_list = []
        fcf_list = []
        rev_list = []
        book_list = []
        tvavg_prev = None

        for f in fundamentals:
            price = get_price_on_year_end(ticker, f["year"])
            if not price or not f["shares"]:
                continue

            eps = f["eps"] if f["eps"] else 0
            fcf_per_share = f["fcf"] / f["shares"] if f["fcf"] and f["shares"] else 0
            rev_per_share = f["revenue"] / f["shares"] if f["revenue"] and f["shares"] else 0
            book_per_share = f["book"]

            eps_yield = (eps / float(price)) * 100 if eps else 0
            fcf_yield = (fcf_per_share / float(price)) * 100 if fcf_per_share else 0
            rev_yield = (rev_per_share / float(price)) * 100 if rev_per_share else 0
            book_yield = (book_per_share / float(price)) * 100 if book_per_share else 0

            eps_list.append(eps_yield)
            fcf_list.append(fcf_yield)
            rev_list.append(rev_yield)
            book_list.append(book_yield)

            eps_avg = sum(eps_list) / len(eps_list)
            fcf_avg = sum(fcf_list) / len(fcf_list)
            rev_avg = sum(rev_list) / len(rev_list)
            book_avg = sum(book_list) / len(book_list)

            score = sum([
                1 if eps_yield > eps_avg else 0,
                1 if fcf_yield > fcf_avg else 0,
                1 if rev_yield > rev_avg else 0,
                1 if book_yield > book_avg else 0,
            ])

            tvpct = eps_yield + fcf_yield + rev_yield + book_yield
            tvavg = eps_avg + fcf_avg + rev_avg + book_avg
            tvavg_trend = None if tvavg_prev is None else tvavg > tvavg_prev
            tvavg_prev = tvavg

            result = {
                "year": f["year"],
                "eps": eps,
                "fcf": f["fcf"],
                "revenue": f["revenue"],
                "shares": f["shares"],
                "book": f["book"],
                "close": float(price),
                "eps_yield": eps_yield,
                "fcf_yield": fcf_yield,
                "rev_yield": rev_yield,
                "book_yield": book_yield,
                "eps_avg": eps_avg,
                "fcf_avg": fcf_avg,
                "rev_avg": rev_avg,
                "book_avg": book_avg,
                "tvpct": tvpct,
                "tvavg": tvavg,
                "tvavg_trend": tvavg_trend,
                "score": score
            }

            insert_yield_row(ticker, result)
            print(f"✅ {ticker} {f['year']} → Score {score} | TVPCT={tvpct:.2f}% vs TVAVG={tvavg:.2f}%")

        return {
            "eps": eps_list,
            "fcf": fcf_list,
            "rev": rev_list,
            "book": book_list
        }

    except Exception as e:
        logger.error(f"❌ Error processing {ticker}: {e}")
        return {"eps": [], "fcf": [], "rev": [], "book": []}

# ----------------------------------------------------------
# Calculate daily Greer Yields using most recent financials available at each date
# ----------------------------------------------------------
def calculate_daily_yields(ticker):
    import decimal
    from decimal import Decimal

    with psycopg2.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            # Load all close prices for the ticker
            cur.execute("""
                SELECT date, close FROM prices
                WHERE ticker = %s
                ORDER BY date
            """, (ticker,))
            price_rows = cur.fetchall()

            # Load financials (sorted by report_date)
            cur.execute("""
                SELECT report_date, net_income, free_cash_flow, total_revenue, shares_outstanding, book_value_per_share
                FROM financials
                WHERE ticker = %s
                ORDER BY report_date
            """, (ticker,))
            fin_rows = cur.fetchall()

            if not price_rows or not fin_rows:
                print(f"⚠️ Missing data for {ticker}")
                return

            fin_idx = 0
            current_fin = fin_rows[fin_idx]
            first_fin_date = fin_rows[0][0]  # Only start after this

            # Running lists for yield averages
            eps_list, fcf_list, rev_list, book_list = [], [], [], []
            tvavg_prev = None

            for p in price_rows:
                date, close = p
                if date < first_fin_date:
                    continue  # Skip if before we have financial data

                close = float(close)

                # If next financials are now available, switch to them
                if fin_idx + 1 < len(fin_rows) and date >= fin_rows[fin_idx + 1][0]:
                    fin_idx += 1
                    current_fin = fin_rows[fin_idx]

                report_date, net_income, fcf, revenue, shares, book = current_fin
                if not shares or shares == 0:
                    continue

                # Calculate per-share values
                eps = (net_income / shares) if net_income else 0
                fcf_ps = (fcf / shares) if fcf else 0
                rev_ps = (revenue / shares) if revenue else 0
                book_ps = book  # already per share

                # Calculate yields
                eps_yield = (eps / close) * 100 if eps else 0
                fcf_yield = (fcf_ps / close) * 100 if fcf_ps else 0
                rev_yield = (rev_ps / close) * 100 if rev_ps else 0
                book_yield = (book_ps / close) * 100 if book_ps else 0
                tvpct = eps_yield + fcf_yield + rev_yield + book_yield

                # Running averages
                eps_list.append(eps_yield)
                fcf_list.append(fcf_yield)
                rev_list.append(rev_yield)
                book_list.append(book_yield)

                eps_avg = sum(eps_list) / len(eps_list)
                fcf_avg = sum(fcf_list) / len(fcf_list)
                rev_avg = sum(rev_list) / len(rev_list)
                book_avg = sum(book_list) / len(book_list)

                tvavg = eps_avg + fcf_avg + rev_avg + book_avg
                tvavg_trend = None if tvavg_prev is None else tvpct > tvavg_prev
                tvavg_prev = tvavg

                # Score calculation
                score = sum([
                    1 if eps_yield > eps_avg else 0,
                    1 if fcf_yield > fcf_avg else 0,
                    1 if rev_yield > rev_avg else 0,
                    1 if book_yield > book_avg else 0,
                ])

                # Insert row
                cur.execute("""
                    INSERT INTO greer_yields_daily (
                        ticker, date,
                        eps_yield, fcf_yield, revenue_yield, book_yield,
                        avg_eps_yield, avg_fcf_yield, avg_revenue_yield, avg_book_yield,
                        tvpct, tvavg, tvavg_trend, score
                    ) VALUES (%s, %s,
                              %s, %s, %s, %s,
                              %s, %s, %s, %s,
                              %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO NOTHING
                """, (
                    ticker, date,
                    eps_yield, fcf_yield, rev_yield, book_yield,
                    eps_avg, fcf_avg, rev_avg, book_avg,
                    tvpct, tvavg, tvavg_trend, score
                ))

            print(f"✅ Finished daily yield calc for {ticker}")

# ----------------------------------------------------------
# Realtime snapshot using latest data
# ----------------------------------------------------------
def calculate_latest_yield(ticker, hist):
    with psycopg2.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            # 1️⃣ Fetch the earliest price for this ticker
            cur.execute("""
                SELECT MIN(date) FROM prices WHERE ticker = %s
            """, (ticker,))
            first_price = cur.fetchone()[0]

            # 2️⃣ Fetch most recent fundamentals
            cur.execute("""
                SELECT report_date, net_income, free_cash_flow, total_revenue,
                       shares_outstanding, book_value_per_share
                FROM financials
                WHERE ticker = %s
                ORDER BY report_date DESC
                LIMIT 1
            """, (ticker,))
            row = cur.fetchone()
            if not row:
                msg = f"No fundamentals found for {ticker}."
                print(f"❌ {msg}")
                logger.error(msg)
                return

            report_date, ni, fcf, rev, shares, book = row
            eps    = ni / shares if ni and shares else 0
            fcf_ps = fcf / shares if fcf and shares else 0
            rev_ps = rev / shares if rev and shares else 0

            # 3️⃣ Fetch latest price
            cur.execute("""
                SELECT close FROM prices
                WHERE ticker = %s
                ORDER BY date DESC
                LIMIT 1
            """, (ticker,))
            price_row = cur.fetchone()
            if not price_row:
                msg = f"No price found for {ticker}."
                print(f"❌ {msg}")
                logger.error(msg)
                return
            price = float(price_row[0])

    # If no annual history at all
    if not hist["eps"]:
        # Detect “new” by comparing first price vs fiscal year-end
        fy_end = date(report_date.year, 1, 1).replace(month=12, day=31)
        if first_price and first_price > fy_end:
            msg = f"{ticker} first traded on {first_price} — new company, skipping snapshot."
        else:
            msg = f"No historical yield data for {ticker} — insufficient history."
        print(f"⚠️ {msg}")
        logger.warning(msg)
        return

    # Calculate latest yields
    eps_y = (eps / price) * 100
    fcf_y = (fcf_ps / price) * 100
    rev_y = (rev_ps / price) * 100
    book_y = (book / price) * 100 if book else 0

    # Compute historical averages
    eps_avg  = sum(hist["eps"])  / len(hist["eps"])
    fcf_avg  = sum(hist["fcf"])  / len(hist["fcf"])
    rev_avg  = sum(hist["rev"])  / len(hist["rev"])
    book_avg = sum(hist["book"]) / len(hist["book"])

    # Score & totals
    score = sum([
        1 if eps_y   > eps_avg  else 0,
        1 if fcf_y   > fcf_avg  else 0,
        1 if rev_y   > rev_avg  else 0,
        1 if book_y  > book_avg else 0,
    ])
    tvpct = eps_y + fcf_y + rev_y + book_y
    tvavg = eps_avg + fcf_avg + rev_avg + book_avg

    # Print snapshot
    print("\n📍 Latest Snapshot:")
    print(f"EPS Yield:     {eps_y:.2f}% vs Avg {eps_avg:.2f}%")
    print(f"FCF Yield:     {fcf_y:.2f}% vs Avg {fcf_avg:.2f}%")
    print(f"Revenue Yield: {rev_y:.2f}% vs Avg {rev_avg:.2f}%")
    print(f"Book Yield:    {book_y:.2f}% vs Avg {book_avg:.2f}%")
    print(f"Total Yield:   {tvpct:.2f}% vs TVAVG {tvavg:.2f}% → Score: {score}/4")


# ----------------------------------------------------------
# Load tickers from file or DB
# ----------------------------------------------------------
def load_tickers(file_path=None):
    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        return df["ticker"].dropna().str.upper().unique().tolist()
    else:
        print("💃️  Loading tickers from companies table...")
        engine = create_engine("postgresql://greer_user@localhost:5432/yfinance_db")
        with engine.begin() as conn:
            return pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()

# ----------------------------------------------------------
# Entry
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and score Greer Value Yields")
    parser.add_argument("--file", type=str, help="Path to ticker file")
    args = parser.parse_args()

    tickers = load_tickers(args.file)

    if not tickers:
        print("⚠️ No tickers loaded — check your file or DB.")
    else:
        print(f"✅ Loaded {len(tickers)} tickers\n")

        for ticker in tickers:
            print(f"\n📊 Processing {ticker}...")
            hist = calculate_yields(ticker)
            calculate_daily_yields(ticker)      # ← re-added
            calculate_latest_yield(ticker, hist)
