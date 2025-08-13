# export_schema.py
# ----------------------------------------------------------
# Dump PostgreSQL schema-only DDL (no data) for Git versioning.
# Uses db.py (get_engine / DB_URL) to obtain connection details,
# then shells out to pg_dump / pg_dumpall.
#
# Typical Usage:
# ----------------------------------------------------------
# 1) Dump schema for the public schema only (clean, idempotent):
#     python export_schema.py --clean
#
# 2) Dump all non-system schemas (clean, idempotent):
#     python export_schema.py --all-schemas --clean
#
# 3) Include cluster globals (roles/tablespaces):
#     python export_schema.py --with-globals --clean
#
# 4) Timestamped archive of schema (keeps previous dumps):
#     python export_schema.py --clean --timestamp
#
# Notes:
# - Default output is db/schema/greer_schema.sql (stable file name for Git).
# - The --clean flag adds DROP ... IF EXISTS so re-runs overwrite cleanly.
# - Use --no-owner and --no-privileges (default) for portability across environments.
# - Use --verbose for debug logging.
# ----------------------------------------------------------

import argparse
import datetime as dt
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from db import get_engine, DB_URL  # Reuse your centralized DB config


# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    """
    Configure logging to console (INFO by default, DEBUG if verbose).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


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
            f"Install PostgreSQL client tools or add them to PATH."
        )
    return path


# ----------------------------------------------------------
# Build pg_dump Command
# ----------------------------------------------------------
def build_pg_dump_cmd(
    dbname: str,
    user: str | None,
    host: str | None,
    port: int | None,
    schemas: list[str] | None,
    clean: bool,
    all_schemas: bool,
    include_priv_owners: bool
) -> list[str]:
    """
    Build the pg_dump command for a schema-only export.
    """
    cmd = [find_binary("pg_dump")]

    # Connection bits
    cmd += ["-d", dbname]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]

    # Schema only, no privileges/owners for Git-friendliness (unless overridden)
    cmd += ["--schema-only"]
    if clean:
        cmd += ["--clean", "--if-exists"]
    if not include_priv_owners:
        cmd += ["--no-owner", "--no-privileges"]

    # Choose schemas
    if not all_schemas and schemas:
        for s in schemas:
            cmd += ["-n", s]
    elif all_schemas:
        cmd += ["-N", "pg_*", "-N", "information_schema"]

    return cmd


# ----------------------------------------------------------
# Run Command Helper
# ----------------------------------------------------------
def run_cmd(cmd: list[str], outfile: Path, password: str | None) -> None:
    """
    Run a command and write output to outfile.
    """
    logging.debug("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    outfile.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    with outfile.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")


# ----------------------------------------------------------
# Dump Schema
# ----------------------------------------------------------
def dump_schema(
    output_path: Path,
    schemas: list[str] | None,
    all_schemas: bool,
    clean: bool,
    include_priv_owners: bool
) -> Path:
    engine = get_engine()
    url = engine.url

    dbname = url.database
    user = url.username
    password = url.password
    host = url.host
    port = url.port

    cmd = build_pg_dump_cmd(
        dbname=dbname,
        user=user,
        host=host,
        port=port,
        schemas=schemas,
        clean=clean,
        all_schemas=all_schemas,
        include_priv_owners=include_priv_owners
    )

    logging.info("Writing schema to: %s", output_path)
    run_cmd(cmd, output_path, password)
    logging.info("Schema export complete.")
    return output_path


# ----------------------------------------------------------
# Dump Globals
# ----------------------------------------------------------
def dump_globals(output_path: Path) -> Path:
    engine = get_engine()
    url = engine.url

    user = url.username
    password = url.password
    host = url.host
    port = url.port

    cmd = [find_binary("pg_dumpall"), "--globals-only"]
    if host:
        cmd += ["-h", host]
    if port:
        cmd += ["-p", str(port)]
    if user:
        cmd += ["-U", user]

    logging.info("Writing globals to: %s", output_path)
    run_cmd(cmd, output_path, password)
    logging.info("Globals export complete.")
    return output_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PostgreSQL schema-only DDL using db.py configuration."
    )
    parser.add_argument("--output-dir", default="db/schema")
    parser.add_argument("--file", default="greer_schema.sql")
    parser.add_argument("--timestamp", action="store_true")
    parser.add_argument("--schemas", default="public")
    parser.add_argument("--all-schemas", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--include-priv-owners", action="store_true")
    parser.add_argument("--with-globals", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        filename = args.file
        if args.timestamp:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(filename).stem
            suffix = Path(filename).suffix or ".sql"
            filename = f"{stem}_{stamp}{suffix}"

        schema_out = outdir / filename

        schema_list = None
        if not args.all_schemas:
            schema_list = [s.strip() for s in args.schemas.split(",") if s.strip()]

        dump_schema(
            output_path=schema_out,
            schemas=schema_list,
            all_schemas=args.all_schemas,
            clean=args.clean,
            include_priv_owners=args.include_priv_owners,
        )

        if args.with_globals:
            dump_globals(outdir / "pg_globals.sql")

        logging.info("Done.")
        return 0

    except Exception as e:
        logging.error("Export failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
