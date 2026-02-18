# run_all.py
"""
Master orchestration script that triggers all ETL, snapshot, and analysis steps.
"""

import subprocess
import logging
import os
import sys

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
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Helper: Run command with logging + hard fail
# ----------------------------------------------------------
def run_cmd(cmd):
    cmd_str = " ".join(cmd)
    print(f"\n🚀 Running: {cmd_str}")
    logger.info("Running: %s", cmd_str)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info("[stdout]\n%s", result.stdout)
    if result.stderr:
        logger.info("[stderr]\n%s", result.stderr)

    if result.returncode != 0:
        msg = f"❌ Error while running: {cmd_str}"
        print(msg)
        logger.error(msg)
        sys.exit(result.returncode)

# ----------------------------------------------------------
# Core ETL Commands
# ----------------------------------------------------------
etl_commands = [
    ["python", "fetch_company_info.py"],
    ["python", "earnings_loader.py"],
    ["python", "fetch_financials.py"],
    ["python", "price_loader.py"],
    ["python", "fetch_iv_summary.py"],
    ["python", "greer_value_score.py"],
    ["python", "greer_value_yield_score.py"],
    ["python", "greer_fair_value_gap.py"],
    ["python", "greer_buyzone_calculator.py"],
    ["python", "greer_fair_value_calculator.py"],
    ["python", "update_all_star_ratings.py"],
    ["python", "nav.py"],
]

# ----------------------------------------------------------
# Run ETL Scripts
# ----------------------------------------------------------
for cmd in etl_commands:
    run_cmd(cmd)

# ----------------------------------------------------------
# Daily Snapshot + MV Refresh
# ----------------------------------------------------------
print("\n🔄 Writing today’s snapshot + refreshing latest_company_snapshot …")
logger.info("Running refresh_snapshot.py")
run_cmd(["python", "refresh_snapshot.py"])

# ----------------------------------------------------------
# Build the Dashboard Snapshot
# ----------------------------------------------------------
print("\n⚡ Building dashboard_snapshot (fast dashboard table)…")
logger.info("Running build_dashboard_snapshot.py")
run_cmd(["python", "build_dashboard_snapshot.py"])

# ----------------------------------------------------------
# Opportunity Periods
# ----------------------------------------------------------
print("\n📜 Rebuilding Greer opportunity periods …")
logger.info("Running run_opportunities.py")
run_cmd(["python", "run_opportunities.py"])

# ----------------------------------------------------------
# GFV Opportunity Periods (NO MV refresh here)
# ----------------------------------------------------------
print("\n📜 Rebuilding Greer GFV opportunity periods …")
logger.info("Running greer_opportunity_gfv_periods.py")
run_cmd(["python", "greer_opportunity_gfv_periods.py"])

# ----------------------------------------------------------
# Update Aggregated Opportunities
# ----------------------------------------------------------
print("\n📜 Updating Greer opportunity periods …")
logger.info("Running update_greer_opportunities.py")
run_cmd(["python", "update_greer_opportunities.py"])

# ----------------------------------------------------------
# Snapshot Opportunities
# ----------------------------------------------------------
print("\n🔄 Creating/Refreshing Greer opportunities snapshot …")
logger.info("Running create_greer_opportunities_snapshot.py")
run_cmd(["python", "create_greer_opportunities_snapshot.py"])

# ----------------------------------------------------------
# Backtest
# ----------------------------------------------------------
print("\n📈 Running backtest.py …")
logger.info("Running backtest.py")
run_cmd(["python", "backtest.py"])

print("\n✅ run_all.py completed successfully")
logger.info("run_all.py completed successfully")
