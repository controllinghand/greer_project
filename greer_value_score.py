# greer_value_score.py
"""
Greer Value Score + Daily Scores

Default behavior (FAST / cron-friendly):
- Updates greer_scores for the LATEST report_date only (per ticker)
- Updates greer_scores_daily incrementally (skips if already up to date)

Optional (SLOW / backfill):
- --rebuild-history : rebuild greer_scores for ALL report_date rows (per ticker)
- --rebuild-daily   : rebuild greer_scores_daily from scratch for each ticker (dangerously slow)

Notes:
- This script reads financials + prices from Postgres and writes into:
  - public.greer_scores
  - public.greer_scores_daily
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
from typing import Optional

# ----------------------------------------------------------
# Shared DB utility
# ----------------------------------------------------------
from db import get_engine, get_psycopg_connection

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "greer_value_score.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Create shared DB connections
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Function: Calculate Greer Value % based on metric trend
# ----------------------------------------------------------
def calculate_greer_value(
    data: pd.DataFrame,
    metric: str,
    weighting: str = "Exponential",
    exp_alpha: float = 0.6,
) -> float:
    if data.empty or len(data) < 2:
        return np.nan

    changes = []
    prev_value = None
    for value in data[metric]:
        if prev_value is not None and not np.isnan(value) and not np.isnan(prev_value):
            if metric == "DILUTED_SHARES_OUTSTANDING":
                changes.append(1.0 if value < prev_value and value > 0 else 0.0)
            else:
                changes.append(1.0 if value > prev_value and value > 0 else 0.0)
        prev_value = value

    if not changes:
        return np.nan

    def simple_avg(arr):
        return np.mean(arr) * 100.0

    def linear_weighted_avg(arr):
        weights = np.arange(1, len(arr) + 1)
        return np.sum(np.array(arr) * weights) / np.sum(weights) * 100.0

    def exp_weighted_avg(arr, alpha):
        weights = np.array([alpha ** (len(arr) - 1 - i) for i in range(len(arr))])
        return np.sum(np.array(arr) * weights) / np.sum(weights) * 100.0

    return (
        exp_weighted_avg(changes, exp_alpha)
        if weighting == "Exponential"
        else linear_weighted_avg(changes)
        if weighting == "Linear"
        else simple_avg(changes)
    )

# ----------------------------------------------------------
# Function: Compute Greer Value Score from multiple metrics
# ----------------------------------------------------------
def compute_greer_value_score(ticker: str, data: pd.DataFrame) -> dict:
    metrics = [
        "BOOK_VALUE_PER_SHARE",
        "FREE_CASH_FLOW",
        "NET_MARGIN",
        "TOTAL_REVENUE",
        "NET_INCOME",
        "DILUTED_SHARES_OUTSTANDING",
    ]

    results = {}
    valid_metrics = 0
    total_pct = 0.0
    above_50_count = 0

    for metric in metrics:
        pct = calculate_greer_value(data, metric)
        results[metric] = {"pct": pct}
        if not np.isnan(pct):
            valid_metrics += 1
            total_pct += pct
            if pct >= 50:
                above_50_count += 1

    greer_value = total_pct / valid_metrics if valid_metrics > 0 else np.nan
    results["GreerValue"] = {"score": greer_value, "above_50_count": above_50_count}
    return {ticker: results}

# ----------------------------------------------------------
# Function: Load historical financial data for a ticker
# ----------------------------------------------------------
def load_data_from_db(ticker: str) -> pd.DataFrame:
    query = """
        SELECT report_date,
               book_value_per_share,
               free_cash_flow,
               net_margin,
               total_revenue,
               net_income,
               shares_outstanding
        FROM public.financials
        WHERE ticker = %s
        ORDER BY report_date ASC
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    df.rename(
        columns={
            "report_date": "dates",
            "book_value_per_share": "BOOK_VALUE_PER_SHARE",
            "free_cash_flow": "FREE_CASH_FLOW",
            "net_margin": "NET_MARGIN",
            "total_revenue": "TOTAL_REVENUE",
            "net_income": "NET_INCOME",
            "shares_outstanding": "DILUTED_SHARES_OUTSTANDING",
        },
        inplace=True,
    )
    return df

# ----------------------------------------------------------
# Convert any numpy types to native Python types
# ----------------------------------------------------------
def convert_numpy_types(row_dict: dict) -> dict:
    out = {}
    for k, v in row_dict.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

# ----------------------------------------------------------
# Insert or update Greer scores into DB
# NOTE: no commit here; caller commits once per ticker
# ----------------------------------------------------------
def insert_greer_score(conn, row: dict, report_date):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.greer_scores (
                ticker, report_date, greer_score, above_50_count,
                book_pct, fcf_pct, margin_pct, revenue_pct, income_pct, shares_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_date) DO UPDATE
            SET
                greer_score = EXCLUDED.greer_score,
                above_50_count = EXCLUDED.above_50_count,
                book_pct = EXCLUDED.book_pct,
                fcf_pct = EXCLUDED.fcf_pct,
                margin_pct = EXCLUDED.margin_pct,
                revenue_pct = EXCLUDED.revenue_pct,
                income_pct = EXCLUDED.income_pct,
                shares_pct = EXCLUDED.shares_pct,
                created_at = CURRENT_TIMESTAMP;
            """,
            (
                row["Ticker"],
                report_date,
                row["Greer Score"],
                row["Above 50%"],
                row["Book%"],
                row["FCF%"],
                row["Margin%"],
                row["Revenue%"],
                row["Income%"],
                row["Shares%"],
            ),
        )

# ----------------------------------------------------------
# Helper: Load tickers from file or companies table
# ----------------------------------------------------------
def load_tickers(file_path: Optional[str] = None) -> list[str]:
    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().str.upper().unique().tolist()
    else:
        print("🗃️  Loading tickers from companies table...")
        with engine.begin() as conn:
            tickers = pd.read_sql("SELECT ticker FROM public.companies ORDER BY ticker", conn)[
                "ticker"
            ].tolist()
    return tickers

# ----------------------------------------------------------
# parse --tickers helper
# ----------------------------------------------------------
def _parse_tickers_arg(raw: Optional[str]) -> list[str]:
    """Accept comma and/or whitespace separated tickers."""
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

# ----------------------------------------------------------
# Daily Scores (incremental, skip-if-current)
# ----------------------------------------------------------
def calculate_daily_scores(ticker: str, rebuild_daily: bool = False):
    """
    Default: incremental update
    - If already up to date, returns immediately
    - Otherwise inserts missing dates only (still 1 INSERT per day; acceptable for daily delta)

    Optional: rebuild_daily=True (SLOW)
    - Deletes all rows for ticker from greer_scores_daily, then rebuilds from first_fin_date onward
    """
    try:
        with get_psycopg_connection() as conn:
            with conn.cursor() as cur:

                # ----------------------------------------------------------
                # Optional destructive rebuild per ticker
                # ----------------------------------------------------------
                if rebuild_daily:
                    print(f"🧨 {ticker}: rebuild-daily ON: deleting existing greer_scores_daily rows")
                    cur.execute("DELETE FROM public.greer_scores_daily WHERE ticker = %s", (ticker,))
                    conn.commit()

                # ----------------------------------------------------------
                # FAST EXIT: skip if already up to date
                # ----------------------------------------------------------
                cur.execute(
                    "SELECT MAX(date) FROM public.greer_scores_daily WHERE ticker = %s",
                    (ticker,),
                )
                last_done = cur.fetchone()[0]

                cur.execute("SELECT MAX(date) FROM public.prices WHERE ticker = %s", (ticker,))
                last_price = cur.fetchone()[0]

                if (
                    not rebuild_daily
                    and last_done is not None
                    and last_price is not None
                    and last_done >= last_price
                ):
                    print(
                        f"⏩ {ticker}: daily scores already up to date (last_done={last_done}, last_price={last_price})"
                    )
                    return

                # ----------------------------------------------------------
                # Load financials (sorted by report_date)
                # ----------------------------------------------------------
                cur.execute(
                    """
                    SELECT report_date, book_value_per_share, free_cash_flow, net_margin,
                           total_revenue, net_income, shares_outstanding
                    FROM public.financials
                    WHERE ticker = %s
                    ORDER BY report_date
                    """,
                    (ticker,),
                )
                fin_rows = cur.fetchall()
                if not fin_rows:
                    msg = f"No financial data found for {ticker}"
                    print(f"⚠️ {msg}")
                    logger.warning(msg)
                    return

                first_fin_date = fin_rows[0][0]
                fin_dates = [r[0] for r in fin_rows]

                # ----------------------------------------------------------
                # Only load the price dates we actually need
                # ----------------------------------------------------------
                if rebuild_daily or last_done is None:
                    cur.execute(
                        """
                        SELECT date
                        FROM public.prices
                        WHERE ticker = %s
                        ORDER BY date
                        """,
                        (ticker,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT date
                        FROM public.prices
                        WHERE ticker = %s AND date > %s
                        ORDER BY date
                        """,
                        (ticker, last_done),
                    )

                price_dates = [r[0] for r in cur.fetchall()]
                if not price_dates:
                    # Nothing new since last_done
                    print(f"⏩ {ticker}: no new price dates beyond last_done={last_done}")
                    return

                # drop price dates before first financial date
                price_dates = [d for d in price_dates if d >= first_fin_date]
                if not price_dates:
                    print(f"⏩ {ticker}: all new price dates are before first_fin_date={first_fin_date}")
                    return

                # ----------------------------------------------------------
                # Compute score once per fin_idx (segment caching)
                # ----------------------------------------------------------
                score_cache: dict[int, dict] = {}

                def score_for_fin_idx(idx: int) -> dict:
                    if idx in score_cache:
                        return score_cache[idx]

                    df_subset = pd.DataFrame(
                        fin_rows[: idx + 1],
                        columns=[
                            "dates",
                            "BOOK_VALUE_PER_SHARE",
                            "FREE_CASH_FLOW",
                            "NET_MARGIN",
                            "TOTAL_REVENUE",
                            "NET_INCOME",
                            "DILUTED_SHARES_OUTSTANDING",
                        ],
                    )
                    df_subset["dates"] = pd.to_datetime(df_subset["dates"])

                    result = compute_greer_value_score(ticker, df_subset)[ticker]
                    row = {
                        "greer_score": result["GreerValue"]["score"],
                        "above_50_count": result["GreerValue"]["above_50_count"],
                        "book_pct": result["BOOK_VALUE_PER_SHARE"]["pct"],
                        "fcf_pct": result["FREE_CASH_FLOW"]["pct"],
                        "margin_pct": result["NET_MARGIN"]["pct"],
                        "revenue_pct": result["TOTAL_REVENUE"]["pct"],
                        "income_pct": result["NET_INCOME"]["pct"],
                        "shares_pct": result["DILUTED_SHARES_OUTSTANDING"]["pct"],
                    }
                    row = convert_numpy_types(row)
                    score_cache[idx] = row
                    return row

                # ----------------------------------------------------------
                # For each price date, map to current fin_idx
                # ----------------------------------------------------------
                fin_idx = 0
                # if we start after some dates, move fin_idx to the right place quickly
                start_date = price_dates[0]
                while fin_idx + 1 < len(fin_dates) and start_date >= fin_dates[fin_idx + 1]:
                    fin_idx += 1

                rows_inserted = 0
                for d in price_dates:
                    while fin_idx + 1 < len(fin_dates) and d >= fin_dates[fin_idx + 1]:
                        fin_idx += 1

                    s = score_for_fin_idx(fin_idx)

                    cur.execute(
                        """
                        INSERT INTO public.greer_scores_daily (
                            ticker, date, greer_score, above_50_count,
                            book_pct, fcf_pct, margin_pct, revenue_pct, income_pct, shares_pct
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO NOTHING
                        """,
                        (
                            ticker,
                            d,
                            s["greer_score"],
                            s["above_50_count"],
                            s["book_pct"],
                            s["fcf_pct"],
                            s["margin_pct"],
                            s["revenue_pct"],
                            s["income_pct"],
                            s["shares_pct"],
                        ),
                    )
                    rows_inserted += 1

                conn.commit()
                print(f"✅ {ticker}: inserted {rows_inserted} daily score rows")

    except Exception as e:
        msg = f"Error processing daily scores for {ticker}: {e}"
        print(f"❌ {msg}")
        logger.exception(msg)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Greer Value Analyzer from DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file containing tickers (column: 'ticker')")
    group.add_argument("--tickers", type=str, help='Optional comma/space separated tickers, e.g. "AAPL,MSFT TSLA"')

    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Rebuild greer_scores for ALL report_date rows (slow). Default updates latest only.",
    )
    parser.add_argument(
        "--rebuild-daily",
        action="store_true",
        help="Rebuild greer_scores_daily from scratch per ticker (VERY slow). Default is incremental + skip-if-current.",
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    explicit = _parse_tickers_arg(args.tickers)
    tickers = explicit if explicit else load_tickers(args.file)

    mode = "REBUILD-HISTORY" if args.rebuild_history else "DAILY(latest-only)"
    daily_mode = "REBUILD-DAILY" if args.rebuild_daily else "INCREMENTAL-DAILY"
    print(f"\n✅ Loaded {len(tickers)} tickers")
    print(f"Mode: {mode} | Daily: {daily_mode}\n")

    summary_rows = []

    # Use one psycopg connection for all greer_scores work (commit per ticker)
    with get_psycopg_connection() as pg_conn:
        for ticker in tickers:
            try:
                print(f"📊 Processing {ticker}...")

                df_full = load_data_from_db(ticker)
                if df_full.empty:
                    print(f"⚠️ No financial data for {ticker} in DB.")
                    continue

                # Optional: write out per-ticker CSV for debugging
                # df_full.to_csv(f"data/{ticker}_data.csv", index=False)

                report_dates = df_full["dates"].sort_values().unique()

                if args.rebuild_history:
                    print(f"🧱 {ticker}: rebuild-history ON ({len(report_dates)} report_date rows)")
                    for report_date in report_dates:
                        df_subset = df_full[df_full["dates"] <= report_date]
                        result = compute_greer_value_score(ticker, df_subset)[ticker]
                        r = result

                        row = {
                            "Ticker": ticker,
                            "Greer Score": round(r["GreerValue"]["score"], 2)
                            if not np.isnan(r["GreerValue"]["score"])
                            else None,
                            "Above 50%": r["GreerValue"]["above_50_count"],
                            "Book%": round(r["BOOK_VALUE_PER_SHARE"]["pct"], 2)
                            if not np.isnan(r["BOOK_VALUE_PER_SHARE"]["pct"])
                            else None,
                            "FCF%": round(r["FREE_CASH_FLOW"]["pct"], 2)
                            if not np.isnan(r["FREE_CASH_FLOW"]["pct"])
                            else None,
                            "Margin%": round(r["NET_MARGIN"]["pct"], 2)
                            if not np.isnan(r["NET_MARGIN"]["pct"])
                            else None,
                            "Revenue%": round(r["TOTAL_REVENUE"]["pct"], 2)
                            if not np.isnan(r["TOTAL_REVENUE"]["pct"])
                            else None,
                            "Income%": round(r["NET_INCOME"]["pct"], 2)
                            if not np.isnan(r["NET_INCOME"]["pct"])
                            else None,
                            "Shares%": round(r["DILUTED_SHARES_OUTSTANDING"]["pct"], 2)
                            if not np.isnan(r["DILUTED_SHARES_OUTSTANDING"]["pct"])
                            else None,
                        }

                        # keep summary (note: one row per report_date in rebuild mode)
                        summary_rows.append(row)

                        insert_greer_score(pg_conn, convert_numpy_types(row), report_date)

                    pg_conn.commit()
                    print(f"✅ {ticker}: greer_scores rebuilt (history)")

                else:
                    latest_report_date = report_dates[-1]
                    df_subset = df_full[df_full["dates"] <= latest_report_date]
                    r = compute_greer_value_score(ticker, df_subset)[ticker]

                    row = {
                        "Ticker": ticker,
                        "Greer Score": round(r["GreerValue"]["score"], 2)
                        if not np.isnan(r["GreerValue"]["score"])
                        else None,
                        "Above 50%": r["GreerValue"]["above_50_count"],
                        "Book%": round(r["BOOK_VALUE_PER_SHARE"]["pct"], 2)
                        if not np.isnan(r["BOOK_VALUE_PER_SHARE"]["pct"])
                        else None,
                        "FCF%": round(r["FREE_CASH_FLOW"]["pct"], 2)
                        if not np.isnan(r["FREE_CASH_FLOW"]["pct"])
                        else None,
                        "Margin%": round(r["NET_MARGIN"]["pct"], 2)
                        if not np.isnan(r["NET_MARGIN"]["pct"])
                        else None,
                        "Revenue%": round(r["TOTAL_REVENUE"]["pct"], 2)
                        if not np.isnan(r["TOTAL_REVENUE"]["pct"])
                        else None,
                        "Income%": round(r["NET_INCOME"]["pct"], 2)
                        if not np.isnan(r["NET_INCOME"]["pct"])
                        else None,
                        "Shares%": round(r["DILUTED_SHARES_OUTSTANDING"]["pct"], 2)
                        if not np.isnan(r["DILUTED_SHARES_OUTSTANDING"]["pct"])
                        else None,
                    }

                    summary_rows.append(row)
                    insert_greer_score(pg_conn, convert_numpy_types(row), latest_report_date)
                    pg_conn.commit()
                    print(f"✅ {ticker}: upserted latest greer_scores (as of {latest_report_date})")

                # Daily scores: incremental by default, rebuild optional
                calculate_daily_scores(ticker, rebuild_daily=args.rebuild_daily)

            except Exception as e:
                logger.exception("❌ Error processing %s: %s", ticker, e)
                print(f"❌ Error processing {ticker}: {e}")

    # Summary output
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("greer_summary.csv", index=False)

    print("\n✅ Saved summary to greer_summary.csv")
    print("🎯 Unique tickers processed:", summary_df["Ticker"].nunique())

if __name__ == "__main__":
    main()
