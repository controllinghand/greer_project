# restore_schema.py
# ----------------------------------------------------------
# Restore PostgreSQL schema-only DDL (and optional globals)
# using db.py configuration.
#
# Typical Usage:
# ----------------------------------------------------------
# 1) Restore into an existing DB (objects dropped/created as per dump's --clean flags):
#     python restore_schema.py --schema db/schema/greer_schema.sql
#
# 2) Create the DB if missing, then restore:
#     python restore_schema.py --create-db --schema db/schema/greer_schema.sql
#
# 3) Drop & recreate the DB (DANGEROUS), then restore:
#     python restore_schema.py --drop-db --force --create-db --schema db/schema/greer_schema.sql
#
# 4) Apply cluster globals first (roles/tablespaces), then schema:
#     python restore_schema.py --globals db/schema/pg_globals.sql --create-db --schema db/schema/greer_schema.sql
#
# Notes:
# - The schema dump created with `export_schema.py --clean` already includes
#   DROP ... IF EXISTS statements, making re-runs idempotent without needing --drop-db.
# - Use --verbose for debug logging.
# ----------------------------------------------------------

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from db import get_engine  # Uses your centralized DB config


# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    """
    Configure logging to console (INFO by default, DEBUG if verbose).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------
# Resolve Binaries
# ----------------------------------------------------------
def find_binary(name: str) -> str:
    """
    Ensure a required binary is on PATH and return the absolute path.
    """
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(
            f"Required binary '{name}' not found on PATH. "
            f"Install PostgreSQL client tools (psql, createdb, dropdb) or add them to PATH."
        )
    return path


# ----------------------------------------------------------
# Helpers to run shell commands
# ----------------------------------------------------------
def run_cmd(cmd: list[str], password: str | None) -> None:
    """
    Run a command and raise with stderr if it fails.
    """
    logging.debug("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    if proc.stdout.strip():
        logging.debug(proc.stdout.strip())


# ----------------------------------------------------------
# Database existence check
# ----------------------------------------------------------
def database_exists(dbname: str, host: str | None, port: int | None, user: str | None, password: str | None) -> bool:
    """
    Check if the target database exists by querying the postgres catalog.
    """
    psql = find_binary("psql")
    cmd = [psql, "-d", "postgres", "-tAc", f"select 1 from pg_database where datname='{dbname}'"]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if res.returncode != 0:
        logging.warning("Could not check db existence (%s). Proceeding: %s", res.returncode, res.stderr.strip())
        return False
    return res.stdout.strip() == "1"


# ----------------------------------------------------------
# Create / Drop DB
# ----------------------------------------------------------
def create_database(dbname: str, host: str | None, port: int | None, user: str | None, password: str | None) -> None:
    createdb = find_binary("createdb")
    cmd = [createdb, dbname]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]
    run_cmd(cmd, password)
    logging.info("Created database: %s", dbname)


def drop_database(dbname: str, host: str | None, port: int | None, user: str | None, password: str | None) -> None:
    dropdb = find_binary("dropdb")
    cmd = [dropdb, dbname]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]
    run_cmd(cmd, password)
    logging.info("Dropped database: %s", dbname)


# ----------------------------------------------------------
# Restore SQL files
# ----------------------------------------------------------
def restore_sql_file(sql_file: Path, dbname: str, host: str | None, port: int | None, user: str | None, password: str | None) -> None:
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")
    psql = find_binary("psql")
    cmd = [psql, "-d", dbname, "-f", str(sql_file)]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]
    logging.info("Restoring: %s", sql_file)
    run_cmd(cmd, password)
    logging.info("Restore complete: %s", sql_file)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore PostgreSQL schema/global SQL using db.py connection.")
    p.add_argument(
        "--schema",
        default="db/schema/greer_schema.sql",
        help="Path to schema SQL file (default: db/schema/greer_schema.sql).",
    )
    p.add_argument(
        "--globals",
        default=None,
        help="Optional path to globals SQL (pg_dumpall --globals-only) to run BEFORE schema.",
    )
    p.add_argument(
        "--create-db",
        action="store_true",
        help="Create the target database if it does not exist.",
    )
    p.add_argument(
        "--drop-db",
        action="store_true",
        help="Drop the target database BEFORE restore (DANGEROUS). Requires --force.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Required if using --drop-db to skip safety check.",
    )
    p.add_argument(
        "--db-name",
        default=None,
        help="Override database name (default: from db.py / DATABASE_URL).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return p.parse_args()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        engine = get_engine()
        url = engine.url

        dbname = args.db_name or url.database
        user = url.username
        password = url.password
        host = url.host
        port = url.port

        if args.drop_db:
            if not args.force:
                raise RuntimeError("Refusing to drop the database without --force. Aborting.")
            if database_exists(dbname, host, port, user, password):
                drop_database(dbname, host, port, user, password)
            else:
                logging.info("Database %s does not exist; nothing to drop.", dbname)

        if args.create_db:
            if not database_exists(dbname, host, port, user, password):
                create_database(dbname, host, port, user, password)
            else:
                logging.info("Database %s already exists.", dbname)

        if args.globals:
            restore_sql_file(Path(args.globals), "postgres", host, port, user, password)

        restore_sql_file(Path(args.schema), dbname, host, port, user, password)

        logging.info("All done.")
        return 0

    except Exception as e:
        logging.error("Restore failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
