# run_all.py
"""
Master orchestration script that triggers all build and analysis steps.
"""
import subprocess
import logging
import os
from sqlalchemy import text

# Import shared DB connection
from db import get_engine

# Import Greer opportunity periods module
from greer_opportunity_periods import (
    refresh_mv as opp_refresh_mv,
    rebuild_snapshot as opp_rebuild_snapshot,
    rebuild_periods as opp_rebuild_periods
)

# ----------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "run_all.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Commands to run sequentially (default: load tickers from DB)
# ----------------------------------------------------------
commands = [
    ["python", "fetch_financials.py"],
    ["python", "price_loader.py"],
    ["python", "greer_value_score.py"],
    ["python", "greer_value_yield_score.py"],
    ["python", "greer_fair_value_gap.py"],
    ["python", "greer_buyzone_calculator.py"],
]

# ----------------------------------------------------------
# Run each ETL command
# ----------------------------------------------------------
for cmd in commands:
    cmd_str = ' '.join(cmd)
    print(f"\n🚀 Running: {cmd_str}")
    logger.info(f"Running: {cmd_str}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        error_msg = f"❌ Error while running: {cmd_str} — aborting"
        print(error_msg)
        logger.error(error_msg)
        exit(result.returncode)

# ----------------------------------------------------------
# Refresh snapshot view
# ----------------------------------------------------------
print("\n🔄 Refreshing latest_company_snapshot …")
logger.info("Refreshing materialized view: latest_company_snapshot")

engine = get_engine()
try:
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY latest_company_snapshot;"))
    print("✅ latest_company_snapshot refreshed")
    logger.info("latest_company_snapshot refreshed")
except Exception as e:
    error_msg = f"⚠️ Could not refresh snapshot view: {e}"
    print(error_msg)
    logger.error(error_msg)

# ----------------------------------------------------------
# Rebuild Greer Opportunity Periods
# ----------------------------------------------------------
print("\n📜 Rebuilding Greer opportunity periods …")
logger.info("Rebuilding Greer opportunity periods")
try:
    opp_refresh_mv()
    opp_rebuild_snapshot()
    opp_rebuild_periods()
    print("✅ greer_opportunity_periods complete")
    logger.info("greer_opportunity_periods complete")
except Exception as e:
    error_msg = f"❌ Error in greer_opportunity_periods: {e}"
    print(error_msg)
    logger.error(error_msg)
    exit(1)

# ----------------------------------------------------------
# Run backtest
# ----------------------------------------------------------
print("\n📈 Running backtest.py …")
logger.info("Running backtest.py")

result = subprocess.run(["python", "backtest.py"])
if result.returncode != 0:
    error_msg = "❌ Error while running backtest.py — aborting"
    print(error_msg)
    logger.error(error_msg)
    exit(result.returncode)

print("✅ Backtest complete")
logger.info("Backtest completed successfully")
