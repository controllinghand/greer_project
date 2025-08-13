# README.md
# Greer Project Database Utilities

This folder contains scripts for **dumping** and **restoring** the PostgreSQL database schema for the Greer Project.  
They are designed to be **data-free** exports so you can regenerate the actual rows with your ETL scripts.

---

## 📜 export_schema.py

Dump the database **schema only** (DDL) using connection info from `db.py`.

### Typical Usage

1. Dump schema for `public` schema only (clean, idempotent):

   ```bash
   python export_schema.py --clean
   ```

2. Dump all non-system schemas:

   ```bash
   python export_schema.py --all-schemas --clean
   ```

3. Include cluster globals (roles/tablespaces):

   ```bash
   python export_schema.py --with-globals --clean
   ```

4. Timestamped archive of schema (keeps previous dumps):

   ```bash
   python export_schema.py --clean --timestamp
   ```

**Notes:**
- Default output is `db/schema/greer_schema.sql` (stable file name for Git).
- The `--clean` flag adds `DROP ... IF EXISTS` so re-runs overwrite cleanly.
- `--no-owner` and `--no-privileges` (default) make dumps portable.
- Use `--verbose` for debug logging.

---

## ♻️ restore_schema.py

Restore a **schema-only** SQL file (and optionally **globals**) to a PostgreSQL database.

### Typical Usage

1. Restore into an existing DB:

   ```bash
   python restore_schema.py --schema db/schema/greer_schema.sql
   ```

2. Create DB if missing, then restore:

   ```bash
   python restore_schema.py --create-db --schema db/schema/greer_schema.sql
   ```

3. Drop & recreate DB (**dangerous**), then restore:

   ```bash
   python restore_schema.py --drop-db --force --create-db --schema db/schema/greer_schema.sql
   ```

4. Apply globals (roles/tablespaces) before restoring schema:

   ```bash
   python restore_schema.py --globals db/schema/pg_globals.sql --create-db --schema db/schema/greer_schema.sql
   ```

**Notes:**
- `export_schema.py --clean` already includes `DROP ... IF EXISTS` making re-runs idempotent.
- `--drop-db` will delete the entire database; requires `--force` as a safety check.
- Use `--verbose` for debug logging.

---

## 💡 Workflow Example

```bash
# 1) Dump current schema
python export_schema.py --all-schemas --clean

# 2) Commit changes to Git
git add db/schema/greer_schema.sql
git commit -m "Update DB schema"

# 3) Restore on another environment
python restore_schema.py --create-db --schema db/schema/greer_schema.sql
```

---

## 🛠 Requirements

- PostgreSQL client tools: `pg_dump`, `pg_dumpall`, `psql`, `createdb`, `dropdb`
- Environment variable `DATABASE_URL` or default in `db.py`
- Python 3.10+ and packages in `requirements.txt`

---

## 🔧 Optional Makefile Shortcuts

You can add a `Makefile` at the root of your repo to speed up common commands:

```makefile
.PHONY: dump-schema restore-schema restore-fresh

dump-schema:
\tpython export_schema.py --all-schemas --clean

restore-schema:
\tpython restore_schema.py --schema db/schema/greer_schema.sql

restore-fresh:
\tpython restore_schema.py --drop-db --force --create-db --schema db/schema/greer_schema.sql
```

Then just run:
```bash
make dump-schema
make restore-schema
make restore-fresh
```

---

## 🔄 Post-Restore: Refresh Materialized Views

After restoring the schema and repopulating base tables, refresh all materialized views used by the Greer Project:

```sql
-- Refresh latest snapshot view
REFRESH MATERIALIZED VIEW CONCURRENTLY latest_company_snapshot;

-- Refresh opportunities (if applicable)
REFRESH MATERIALIZED VIEW CONCURRENTLY greer_opportunity_periods;

-- Add any other materialized views here
```

You can run these from `psql`:

```bash
psql -d yfinance_db -f db/refresh_views.sql
```

Or create a helper script `refresh_views.sql` containing all your `REFRESH MATERIALIZED VIEW` commands, so you can refresh them in one step.
