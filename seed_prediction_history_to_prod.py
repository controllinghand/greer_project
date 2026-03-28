# seed_prediction_history_to_prod.py

import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text


# ----------------------------------------------------------
# Build engine
# ----------------------------------------------------------
def get_engine(db_url: str):
    return create_engine(db_url)


# ----------------------------------------------------------
# Load local summary views
# ----------------------------------------------------------
def load_local_summaries(local_engine):
    overall_df = pd.read_sql("SELECT * FROM prediction_backtest_overall_v", local_engine)
    bucket_df = pd.read_sql("SELECT * FROM prediction_backtest_bucket_stats_v", local_engine)
    phase_df = pd.read_sql("SELECT * FROM prediction_backtest_phase_bucket_stats_v", local_engine)
    goi_df = pd.read_sql("SELECT * FROM prediction_backtest_goi_bucket_stats_v", local_engine)
    return overall_df, bucket_df, phase_df, goi_df


# ----------------------------------------------------------
# Delete existing snapshot for as_of_date
# ----------------------------------------------------------
def clear_prod_snapshot(prod_engine, as_of_date: str):
    with prod_engine.begin() as conn:
        conn.execute(text("DELETE FROM prediction_backtest_overall WHERE as_of_date = :d"), {"d": as_of_date})
        conn.execute(text("DELETE FROM prediction_backtest_bucket_stats WHERE as_of_date = :d"), {"d": as_of_date})
        conn.execute(text("DELETE FROM prediction_backtest_phase_bucket_stats WHERE as_of_date = :d"), {"d": as_of_date})
        conn.execute(text("DELETE FROM prediction_backtest_goi_bucket_stats WHERE as_of_date = :d"), {"d": as_of_date})


# ----------------------------------------------------------
# Upload snapshot to prod
# ----------------------------------------------------------
def upload_snapshot(prod_engine, as_of_date: str, overall_df, bucket_df, phase_df, goi_df):
    overall_df = overall_df.copy()
    bucket_df = bucket_df.copy()
    phase_df = phase_df.copy()
    goi_df = goi_df.copy()

    overall_df["as_of_date"] = as_of_date
    bucket_df["as_of_date"] = as_of_date
    phase_df["as_of_date"] = as_of_date
    goi_df["as_of_date"] = as_of_date

    overall_df = overall_df[
        [
            "as_of_date",
            "total_rows",
            "total_signals",
            "bucket_90_rows",
            "bucket_110_rows",
            "bucket_130_rows",
            "win_rate_90",
            "win_rate_110",
            "win_rate_130",
            "avg_return_60d_90",
            "avg_return_60d_110",
            "avg_return_60d_130",
            "avg_return_60d_watchlist",
        ]
    ]

    bucket_df = bucket_df[
        [
            "as_of_date",
            "bucket_label",
            "calibration_bucket",
            "rows",
            "win_rate_60d",
            "avg_return_60d",
            "avg_return_90d",
            "avg_return_120d",
            "avg_return_180d",
        ]
    ]

    phase_df = phase_df[
        [
            "as_of_date",
            "phase",
            "bucket_label",
            "calibration_bucket",
            "rows",
            "win_rate_60d",
            "avg_return_60d",
            "avg_return_90d",
            "avg_return_120d",
            "avg_return_180d",
        ]
    ]

    goi_df = goi_df[
        [
            "as_of_date",
            "goi_zone",
            "bucket_label",
            "calibration_bucket",
            "rows",
            "win_rate_60d",
            "avg_return_60d",
            "avg_return_90d",
            "avg_return_120d",
            "avg_return_180d",
        ]
    ]

    overall_df.to_sql("prediction_backtest_overall", prod_engine, if_exists="append", index=False, method="multi")
    bucket_df.to_sql("prediction_backtest_bucket_stats", prod_engine, if_exists="append", index=False, method="multi")
    phase_df.to_sql("prediction_backtest_phase_bucket_stats", prod_engine, if_exists="append", index=False, method="multi")
    goi_df.to_sql("prediction_backtest_goi_bucket_stats", prod_engine, if_exists="append", index=False, method="multi")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Seed prediction history summary tables to prod.")
    parser.add_argument("--local-db-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--prod-db-url", default=os.getenv("PROD_DATABASE_URL"))
    parser.add_argument("--as-of-date", required=True)
    args = parser.parse_args()

    if not args.local_db_url:
        print("Missing local DATABASE_URL")
        sys.exit(1)

    if not args.prod_db_url:
        print("Missing PROD_DATABASE_URL")
        sys.exit(1)

    local_engine = get_engine(args.local_db_url)
    prod_engine = get_engine(args.prod_db_url)

    overall_df, bucket_df, phase_df, goi_df = load_local_summaries(local_engine)

    clear_prod_snapshot(prod_engine, args.as_of_date)
    upload_snapshot(prod_engine, args.as_of_date, overall_df, bucket_df, phase_df, goi_df)

    print(f"Prediction history snapshot uploaded for {args.as_of_date}")


if __name__ == "__main__":
    main()