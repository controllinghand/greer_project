# run_opportunities.py

import logging
from greer_opportunity_periods import (
    refresh_mv,
    rebuild_snapshot,
    rebuild_periods
)

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
logging.basicConfig(
    filename="logs/run_opportunities.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

print("📜 Rebuilding Greer opportunity periods …")
logger.info("Starting rebuild of Greer opportunity periods")

try:
    refresh_mv()
    rebuild_snapshot()
    rebuild_periods()
    print("✅ greer_opportunity_periods complete")
    logger.info("Greer opportunity periods rebuild complete")
except Exception as e:
    print(f"❌ Error in greer_opportunity_periods: {e}")
    logger.error(f"Error in greer_opportunity_periods: {e}")
