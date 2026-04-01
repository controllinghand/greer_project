# ----------------------------------------------------------
# build_dashboard_summary_daily.py
# - Computes daily market breadth + Greer Market Regime Score
# - Source: dashboard_snapshot
# - Inserts into: dashboard_summary_daily (1 row per day)
# - Dynamic: Fetches GOI thresholds from market_regime_thresholds view
# ----------------------------------------------------------

import logging
import pandas as pd
from sqlalchemy import text
from db import get_engine
# Import the unified brain
from market_cycle_utils import get_market_thresholds, get_goi_label

# ----------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def safe_pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0

def compute_regime_score(gv_bullish_pct: float, buyzone_pct: float) -> float:
    """
    Greer Market Regime Score (0-100)
    gv_bullish_pct = GV Green % + GV Gold %
    buyzone_pct    = % of tickers in BuyZone
    """
    score = (
        0.6 * gv_bullish_pct
        + 0.4 * (100.0 - buyzone_pct)
    )
    return round(float(score), 2)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    engine = get_engine()

    # 1. Fetch dynamic thresholds from the DB View
    logger.info("Fetching dynamic market thresholds from DB view...")
    try:
        thresholds = get_market_thresholds(engine)
        logger.info(f"Thresholds Loaded: {thresholds}")
    except Exception as e:
        logger.error(f"Could not fetch thresholds: {e}. Using hardcoded fallbacks.")
        thresholds = {"p5": 11.0, "p20": 17.0, "p80": 54.0, "p95": 74.0}

    # 2. Ensure table exists (Added goi_status column)
    ddl = """
    CREATE TABLE IF NOT EXISTS dashboard_summary_daily (
      summary_date date PRIMARY KEY,

      total_companies int NOT NULL,
      min_stars int NOT NULL DEFAULT 0,

      gv_gold int NOT NULL,
      gv_green int NOT NULL,
      gv_red int NOT NULL,

      ys_gold int NOT NULL,
      ys_green int NOT NULL,
      ys_red int NOT NULL,

      gfv_gold int NOT NULL,
      gfv_green int NOT NULL,
      gfv_red int NOT NULL,
      gfv_gray int NOT NULL,

      buyzone_count int NOT NULL,

      gv_green_pct numeric NOT NULL,
      ys_green_pct numeric NOT NULL,
      gfv_green_pct numeric NOT NULL,
      buyzone_pct numeric NOT NULL,

      greer_market_index numeric NOT NULL,
      goi_status text,

      created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_dashboard_summary_daily_date
      ON dashboard_summary_daily(summary_date);
    """

    logger.info("Ensuring dashboard_summary_daily exists…")
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # 3. Load Snapshot Data
    logger.info("Loading dashboard_snapshot…")
    df = pd.read_sql(
        """
        SELECT
          ticker,
          greer_star_rating,
          greer_value_score,
          above_50_count,
          greer_yield_score,
          buyzone_flag,
          gfv_status,
          snapshot_date
        FROM dashboard_snapshot
        """,
        engine
    )

    if df.empty:
        logger.warning("dashboard_snapshot is empty. Nothing to summarize.")
        return

    summary_date = pd.to_datetime(df["snapshot_date"].max()).date()

    # 4. Process Buckets
    above50 = pd.to_numeric(df["above_50_count"], errors="coerce").fillna(0)
    gv_score = pd.to_numeric(df["greer_value_score"], errors="coerce").fillna(0)
    ys_score = pd.to_numeric(df["greer_yield_score"], errors="coerce").fillna(0).astype(int)

    gv_bucket = pd.Series("red", index=df.index)
    gv_bucket.loc[gv_score >= 50] = "green"
    gv_bucket.loc[above50 == 6] = "gold"

    ys_bucket = pd.Series("red", index=df.index)
    ys_bucket.loc[ys_score >= 2] = "green"
    ys_bucket.loc[ys_score >= 4] = "gold"

    gfv_norm = df["gfv_status"].astype("string").fillna("gray").str.strip().str.lower()
    valid = {"gold", "green", "red", "gray"}
    gfv_bucket = gfv_norm.where(gfv_norm.isin(valid), "gray")

    buyzone = df["buyzone_flag"].astype("boolean").fillna(False)

    # 5. Aggregate
    total = int(len(df))
    gv_gold = int((gv_bucket == "gold").sum())
    gv_green = int((gv_bucket == "green").sum())
    gv_red = int((gv_bucket == "red").sum())
    ys_gold = int((ys_bucket == "gold").sum())
    ys_green = int((ys_bucket == "green").sum())
    ys_red = int((ys_bucket == "red").sum())
    gfv_gold = int((gfv_bucket == "gold").sum())
    gfv_green = int((gfv_bucket == "green").sum())
    gfv_red = int((gfv_bucket == "red").sum())
    gfv_gray = int((gfv_bucket == "gray").sum())
    buyzone_count = int(buyzone.sum())

    gv_green_pct = safe_pct(gv_green, total)
    gv_gold_pct = safe_pct(gv_gold, total)
    gv_bullish_pct = gv_green_pct + gv_gold_pct
    ys_green_pct = safe_pct(ys_green, total)
    gfv_green_pct = safe_pct(gfv_green, total)
    buyzone_pct = safe_pct(buyzone_count, total)

    # 6. Logic: Score & Dynamic Regime Label
    regime_score = compute_regime_score(gv_bullish_pct, buyzone_pct)
    goi_status = get_goi_label(buyzone_pct, thresholds)

    # 7. Upsert to DB
    upsert = """
    INSERT INTO dashboard_summary_daily (
      summary_date,
      total_companies, min_stars,
      gv_gold, gv_green, gv_red,
      ys_gold, ys_green, ys_red,
      gfv_gold, gfv_green, gfv_red, gfv_gray,
      buyzone_count,
      gv_green_pct, ys_green_pct, gfv_green_pct, buyzone_pct,
      greer_market_index, goi_status
    )
    VALUES (
      :summary_date,
      :total_companies, :min_stars,
      :gv_gold, :gv_green, :gv_red,
      :ys_gold, :ys_green, :ys_red,
      :gfv_gold, :gfv_green, :gfv_red, :gfv_gray,
      :buyzone_count,
      :gv_green_pct, :ys_green_pct, :gfv_green_pct, :buyzone_pct,
      :greer_market_index, :goi_status
    )
    ON CONFLICT (summary_date) DO UPDATE SET
      total_companies = EXCLUDED.total_companies,
      gv_gold = EXCLUDED.gv_gold,
      gv_green = EXCLUDED.gv_green,
      gv_red = EXCLUDED.gv_red,
      ys_gold = EXCLUDED.ys_gold,
      ys_green = EXCLUDED.ys_green,
      ys_red = EXCLUDED.ys_red,
      gfv_gold = EXCLUDED.gfv_gold,
      gfv_green = EXCLUDED.gfv_green,
      gfv_red = EXCLUDED.gfv_red,
      gfv_gray = EXCLUDED.gfv_gray,
      buyzone_count = EXCLUDED.buyzone_count,
      gv_green_pct = EXCLUDED.gv_green_pct,
      ys_green_pct = EXCLUDED.ys_green_pct,
      gfv_green_pct = EXCLUDED.gfv_green_pct,
      buyzone_pct = EXCLUDED.buyzone_pct,
      greer_market_index = EXCLUDED.greer_market_index,
      goi_status = EXCLUDED.goi_status,
      created_at = now();
    """

    params = {
        "summary_date": summary_date,
        "total_companies": total,
        "min_stars": 0,
        "gv_gold": gv_gold,
        "gv_green": gv_green,
        "gv_red": gv_red,
        "ys_gold": ys_gold,
        "ys_green": ys_green,
        "ys_red": ys_red,
        "gfv_gold": gfv_gold,
        "gfv_green": gfv_green,
        "gfv_red": gfv_red,
        "gfv_gray": gfv_gray,
        "buyzone_count": buyzone_count,
        "gv_green_pct": gv_green_pct,
        "ys_green_pct": ys_green_pct,
        "gfv_green_pct": gfv_green_pct,
        "buyzone_pct": buyzone_pct,
        "greer_market_index": regime_score,
        "goi_status": goi_status,
    }

    logger.info(f"Inserting dashboard_summary_daily for {summary_date} (Status: {goi_status})…")
    with engine.begin() as conn:
        conn.execute(text(upsert), params)

    logger.info("✅ dashboard_summary_daily updated successfully")

if __name__ == "__main__":
    main()