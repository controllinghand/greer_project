# run_all.py
"""
Master orchestration script that triggers all ETL, snapshot, and analysis steps.
"""

import subprocess
import logging
import os

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
# Core ETL Commands (Python-based)
# ----------------------------------------------------------
etl_commands = [
    ["python", "fetch_company_info.py"],
    ["python", "fetch_financials.py"],
    ["python", "price_loader.py"],
    ["python", "fetch_iv_summary.py"],  # <-- added
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
# Run Snapshot Refresh Script
# ----------------------------------------------------------
print("\n🔄 Refreshing materialized view: latest_company_snapshot …")
logger.info("Refreshing latest_company_snapshot")
snapshot_result = subprocess.run(["python", "refresh_snapshot.py"])
if snapshot_result.returncode != 0:
    print("❌ Failed to refresh snapshot view")
    logger.error("Failed to refresh snapshot view")
    exit(snapshot_result.returncode)
print("✅ Snapshot view refreshed")
logger.info("Snapshot view refresh successful")

# ----------------------------------------------------------
# Run Opportunity Periods Script
# ----------------------------------------------------------
print("\n📜 Rebuilding Greer opportunity periods …")
logger.info("Running run_opportunities.py")
opp_result = subprocess.run(["python", "run_opportunities.py"])
if opp_result.returncode != 0:
    print("❌ Failed to rebuild opportunity periods")
    logger.error("Failed to rebuild opportunity periods")
    exit(opp_result.returncode)
print("✅ Opportunity periods rebuilt")
logger.info("Opportunity periods rebuild successful")

# ----------------------------------------------------------
# Run GFV Opportunity Periods Script
# ----------------------------------------------------------
print("\n📜 Rebuilding Greer GFV opportunity periods …")
logger.info("Running greer_opportunity_gfv_periods.py")
gfv_opp_result = subprocess.run(["python", "greer_opportunity_gfv_periods.py", "--refresh-mv"])
if gfv_opp_result.returncode != 0:
    print("❌ Failed to rebuild GFV opportunity periods")
    logger.error("Failed to rebuild GFV opportunity periods")
    exit(gfv_opp_result.returncode)
print("✅ GFV opportunity periods rebuilt")
logger.info("GFV opportunity periods rebuild successful")

# ----------------------------------------------------------
# Run Update Greer Opportunities Script
# ----------------------------------------------------------
print("\n📜 Updating Greer opportunity periods …")
logger.info("Running update_greer_opportunities.py")
update_opp_result = subprocess.run(["python", "update_greer_opportunities.py"])
if update_opp_result.returncode != 0:
    print("❌ Failed to update opportunity periods")
    logger.error("Failed to update opportunity periods")
    exit(update_opp_result.returncode)
print("✅ Opportunity periods updated")
logger.info("Opportunity periods update successful")

# ----------------------------------------------------------
# Run Create Greer Opportunities Snapshot Script
# ----------------------------------------------------------
print("\n🔄 Creating/Refreshing Greer opportunities snapshot …")
logger.info("Running create_greer_opportunities_snapshot.py")
create_snapshot_result = subprocess.run(["python", "create_greer_opportunities_snapshot.py"])
if create_snapshot_result.returncode != 0:
    print("❌ Failed to create/refresh opportunities snapshot")
    logger.error("Failed to create/refresh opportunities snapshot")
    exit(create_snapshot_result.returncode)
print("✅ Opportunities snapshot created/refreshed")
logger.info("Opportunities snapshot create/refresh successful")

# ----------------------------------------------------------
# Run Backtest Script
# ----------------------------------------------------------
print("\n📈 Running backtest.py …")
logger.info("Running backtest.py")
backtest_result = subprocess.run(["python", "backtest.py"])
if backtest_result.returncode != 0:
    print("❌ Failed to run backtest")
    logger.error("Failed to run backtest")
    exit(backtest_result.returncode)
print("✅ Backtest completed")
logger.info("Backtest completed successfully")