# refresh_snapshot.py

import logging
from sqlalchemy import text
from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
logging.basicConfig(
    filename="logs/refresh_snapshot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

print("🔄 Refreshing materialized view: latest_company_snapshot …")
logger.info("Starting snapshot refresh...")

engine = get_engine()
try:
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY latest_company_snapshot;"))
    print("✅ latest_company_snapshot refreshed")
    logger.info("Snapshot refreshed successfully.")
except Exception as e:
    print(f"❌ Error refreshing snapshot: {e}")
    logger.error(f"Error refreshing snapshot: {e}")

