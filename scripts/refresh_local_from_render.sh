#!/usr/bin/env bash
# ----------------------------------------------------------
# refresh_local_from_render.sh
# Clone/refresh local Postgres DB from Render Postgres
#
# Features:
# - Loads .env.local automatically (project root)
# - Optional safety guard before wiping local DB
# - Optional SKIP_DUMP=1 to reuse an existing dump file
# - Dumps prod (Render) to a local .dump file (custom format)
# - Ensures local role + database exist
# - Drops + recreates public schema
# - Restores dump with parallel jobs
# - Refreshes materialized views
# - Runs basic sanity checks
# ----------------------------------------------------------

set -euo pipefail

# -----------------------------
# Load .env.local (if present)
# -----------------------------
if [[ -f ".env.local" ]]; then
  # Export all vars defined in .env.local (no quotes required)
  set -a
  # shellcheck disable=SC1091
  source ".env.local"
  set +a
fi

# -----------------------------
# Config (override via env vars)
# -----------------------------
: "${RENDER_DB:?Set RENDER_DB to your Render EXTERNAL Postgres URL (include sslmode=require)}"

LOCAL_HOST="${LOCAL_HOST:-localhost}"
LOCAL_PORT="${LOCAL_PORT:-5432}"

# Local "admin" role that can CREATE ROLE / CREATE DATABASE (yours is 'seanleegreer')
LOCAL_ADMIN_USER="${LOCAL_ADMIN_USER:-$USER}"

# Local app role + db name
LOCAL_ROLE="${LOCAL_ROLE:-greer_user}"
LOCAL_DB_NAME="${LOCAL_DB_NAME:-yfinance_db}"

# Parallel restore jobs
JOBS="${JOBS:-8}"

# Dump output path
DUMP_FILE="${DUMP_FILE:-render_prod.dump}"

# Exclude extension that causes restore warnings locally
EXCLUDE_EXTENSIONS="${EXCLUDE_EXTENSIONS:-pg_stat_statements}"

# Matviews to refresh (space-separated)
MATVIEWS="${MATVIEWS:-latest_company_snapshot greer_opportunities_snapshot}"

# If you want to skip huge table DATA next time, set:
#   EXCLUDE_TABLE_DATA="prices option_prices_daily"
EXCLUDE_TABLE_DATA="${EXCLUDE_TABLE_DATA:-}"

# Optional: skip dumping and reuse existing dump file
SKIP_DUMP="${SKIP_DUMP:-0}"

# Optional: safety guard to prevent accidental wipes
# To require explicit confirmation, set REQUIRE_WIPE_CONFIRM=1 and run with:
#   I_UNDERSTAND_THIS_WIPES_LOCAL_DB=1 ./scripts/refresh_local_from_render.sh
REQUIRE_WIPE_CONFIRM="${REQUIRE_WIPE_CONFIRM:-0}"
I_UNDERSTAND_THIS_WIPES_LOCAL_DB="${I_UNDERSTAND_THIS_WIPES_LOCAL_DB:-0}"

# -----------------------------
# Helpers
# -----------------------------
need_cmd () {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Missing required command: $1" >&2
    exit 1
  }
}

log () {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

psql_admin () {
  PGPASSWORD="${LOCAL_ADMIN_PASSWORD:-}" \
  psql -h "$LOCAL_HOST" -p "$LOCAL_PORT" -U "$LOCAL_ADMIN_USER" "$@"
}

psql_local () {
  # Uses LOCAL_ROLE to connect; password optional depending on your local auth.
  PGPASSWORD="${LOCAL_PASSWORD:-}" \
  psql -h "$LOCAL_HOST" -p "$LOCAL_PORT" -U "$LOCAL_ROLE" -d "$LOCAL_DB_NAME" "$@"
}

# -----------------------------
# Preconditions
# -----------------------------
need_cmd psql
need_cmd pg_dump
need_cmd pg_restore
need_cmd createdb

log "Using pg_dump: $(pg_dump --version)"
log "Using pg_restore: $(pg_restore --version)"

# -----------------------------
# Safety guard (optional)
# -----------------------------
if [[ "$REQUIRE_WIPE_CONFIRM" == "1" && "$I_UNDERSTAND_THIS_WIPES_LOCAL_DB" != "1" ]]; then
  echo "Refusing to run because REQUIRE_WIPE_CONFIRM=1 and I_UNDERSTAND_THIS_WIPES_LOCAL_DB!=1"
  echo "Run: I_UNDERSTAND_THIS_WIPES_LOCAL_DB=1 ./scripts/refresh_local_from_render.sh"
  exit 1
fi

# -----------------------------
# 1) Dump Render -> local file
# -----------------------------
if [[ "$SKIP_DUMP" == "1" ]]; then
  if [[ ! -f "$DUMP_FILE" ]]; then
    echo "ERROR: SKIP_DUMP=1 but dump file not found: $DUMP_FILE" >&2
    exit 1
  fi
  log "SKIP_DUMP=1 -> Using existing dump file: $DUMP_FILE"
else
  log "Dumping Render DB to: $DUMP_FILE"

  EXCLUDE_EXT_ARGS=()
  IFS=',' read -r -a EXT_ARR <<< "$EXCLUDE_EXTENSIONS"
  for ext in "${EXT_ARR[@]}"; do
    ext="$(echo "$ext" | xargs)" # trim
    [[ -n "$ext" ]] && EXCLUDE_EXT_ARGS+=( "--exclude-extension=$ext" )
  done

  pg_dump "$RENDER_DB" \
    -Fc \
    --no-owner --no-privileges \
    "${EXCLUDE_EXT_ARGS[@]}" \
    -f "$DUMP_FILE"

  log "Dump complete: $(ls -lh "$DUMP_FILE" | awk '{print $5, $9}')"
fi

# -----------------------------
# 2) Ensure local role exists
# -----------------------------
log "Ensuring local role exists: $LOCAL_ROLE"
psql_admin -d postgres -v ON_ERROR_STOP=1 -c "
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${LOCAL_ROLE}') THEN
    CREATE ROLE ${LOCAL_ROLE} WITH LOGIN CREATEDB;
  END IF;
END
\$\$;
"

# -----------------------------
# 3) Ensure local database exists
# -----------------------------
log "Ensuring local database exists: $LOCAL_DB_NAME"
DB_EXISTS="$(psql_admin -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${LOCAL_DB_NAME}'" || true)"
if [[ "$DB_EXISTS" != "1" ]]; then
  log "Creating database ${LOCAL_DB_NAME} owned by ${LOCAL_ROLE}"
  createdb -h "$LOCAL_HOST" -p "$LOCAL_PORT" -U "$LOCAL_ADMIN_USER" -O "$LOCAL_ROLE" "$LOCAL_DB_NAME"
else
  log "Database exists. Ensuring owner is ${LOCAL_ROLE}"
  psql_admin -d postgres -v ON_ERROR_STOP=1 -c "ALTER DATABASE ${LOCAL_DB_NAME} OWNER TO ${LOCAL_ROLE};"
fi

# -----------------------------
# 4) Drop + recreate public schema
# -----------------------------
log "Resetting local public schema"
psql_admin -d "$LOCAL_DB_NAME" -v ON_ERROR_STOP=1 -c "
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
ALTER SCHEMA public OWNER TO ${LOCAL_ROLE};
"

# -----------------------------
# 5) Restore
# -----------------------------
log "Restoring dump into local DB (jobs=$JOBS)"

EXCLUDE_TABLE_ARGS=()
if [[ -n "${EXCLUDE_TABLE_DATA:-}" ]]; then
  for t in $EXCLUDE_TABLE_DATA; do
    EXCLUDE_TABLE_ARGS+=( "--exclude-table-data=$t" )
  done
fi

RESTORE_ARGS=(
  --dbname "postgresql://${LOCAL_ROLE}@${LOCAL_HOST}:${LOCAL_PORT}/${LOCAL_DB_NAME}"
  --no-owner --no-privileges
  --jobs "$JOBS"
)

if (( ${#EXCLUDE_TABLE_ARGS[@]} )); then
  RESTORE_ARGS+=( "${EXCLUDE_TABLE_ARGS[@]}" )
fi

pg_restore "${RESTORE_ARGS[@]}" "$DUMP_FILE"

log "Restore complete"

# -----------------------------
# 6) Refresh matviews (best effort)
# -----------------------------
log "Refreshing materialized views (best effort)"
for mv in $MATVIEWS; do
  if psql_local -tAc "SELECT 1 FROM pg_matviews WHERE schemaname='public' AND matviewname='${mv}'" | grep -q 1; then
    log "Refreshing matview: $mv (concurrently if possible)"
    if ! psql_local -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY ${mv};" 2>/dev/null; then
      psql_local -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW ${mv};"
    fi
  else
    log "Matview not found: $mv (skipping)"
  fi
done

# -----------------------------
# 7) Sanity checks
# -----------------------------
log "Sanity checks"
psql_local -c "SELECT count(*) AS companies FROM companies;"
psql_local -c "SELECT max(date) AS latest_price_date FROM prices;" || true
psql_local -c "SELECT max(date) AS latest_yield_date FROM greer_yields_daily;" || true
psql_local -c "SELECT max(date) AS latest_buyzone_date FROM greer_buyzone_daily;" || true
psql_local -c "SELECT count(*) AS latest_snapshot_rows FROM latest_company_snapshot;" || true

log "Done ✅"