#!/bin/bash

set -euo pipefail

# ----------------------------------------------------------
# monthly_prediction_history_refresh.sh
# Refresh local prediction-history inputs from prod,
# rebuild local backtest history, then seed prod summaries.
# ----------------------------------------------------------

echo "========================================"
echo "Monthly Prediction History Refresh"
echo "========================================"

# ----------------------------------------------------------
# Load environment variables from .env
# ----------------------------------------------------------
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# ----------------------------------------------------------
# Required environment variables
# ----------------------------------------------------------
: "${DATABASE_URL:?DATABASE_URL is required}"
: "${PROD_DATABASE_URL:?PROD_DATABASE_URL is required}"

# Optional override
AS_OF_DATE="${1:-$(date +%F)}"

echo "Using snapshot date: $AS_OF_DATE"
echo

# ----------------------------------------------------------
# Backup local DB
# ----------------------------------------------------------
BACKUP_FILE="logs/local_backup_before_prediction_refresh_${AS_OF_DATE}.sql"
mkdir -p logs

echo "1) Backing up local database..."
pg_dump "$DATABASE_URL" > "$BACKUP_FILE"
echo "Local backup saved to: $BACKUP_FILE"
echo

# ----------------------------------------------------------
# Refresh selected local tables from prod
# ----------------------------------------------------------
echo "2) Truncating local source tables..."
psql "$DATABASE_URL" -c "
TRUNCATE TABLE
    company_snapshot,
    greer_company_index_daily,
    buyzone_breadth
RESTART IDENTITY CASCADE;
"
echo

echo "3) Copying prod data into local tables..."
pg_dump "$PROD_DATABASE_URL" --data-only -t company_snapshot | psql "$DATABASE_URL"
pg_dump "$PROD_DATABASE_URL" --data-only -t greer_company_index_daily | psql "$DATABASE_URL"
pg_dump "$PROD_DATABASE_URL" --data-only -t buyzone_breadth | psql "$DATABASE_URL"
echo "Prod-backed local tables refreshed."
echo

# ----------------------------------------------------------
# Rebuild local historical backtest table
# ----------------------------------------------------------
echo "4) Truncating local prediction_engine_backtest_daily..."
psql "$DATABASE_URL" -c "TRUNCATE TABLE prediction_engine_backtest_daily;"
echo

echo "5) Rebuilding local prediction_engine_backtest_daily..."
python prediction_engine_backtest_builder.py --start-date 2021-01-04 --end-date "$AS_OF_DATE"
echo "Local backtest rebuild complete."
echo

# ----------------------------------------------------------
# Optional sanity checks
# ----------------------------------------------------------
echo "6) Running quick sanity checks..."
psql "$DATABASE_URL" -c "
SELECT COUNT(*) AS rows, MIN(snapshot_date) AS min_date, MAX(snapshot_date) AS max_date
FROM prediction_engine_backtest_daily;
"

psql "$DATABASE_URL" -c "
SELECT calibration_bucket, COUNT(*) AS rows
FROM prediction_engine_backtest_daily
GROUP BY 1
ORDER BY 1;
"
echo

# ----------------------------------------------------------
# Seed prod summary snapshot
# ----------------------------------------------------------
echo "7) Seeding prod prediction history snapshot..."
python seed_prediction_history_to_prod.py --as-of-date "$AS_OF_DATE"
echo

echo "========================================"
echo "Monthly Prediction History Refresh Complete"
echo "Snapshot Date: $AS_OF_DATE"
echo "========================================"