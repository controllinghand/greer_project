# greer_company_index_calculator.py

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine
from market_cycle_utils import classify_phase_with_confidence


# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "greer_company_index_calculator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
print_logger = logging.getLogger("stdout")
print_logger.setLevel(logging.INFO)
print_handler = logging.StreamHandler()
print_handler.setFormatter(logging.Formatter("%(message)s"))
if not print_logger.handlers:
    print_logger.addHandler(print_handler)


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------
def transition_risk_label(confidence: float) -> str:
    c = float(confidence)
    if c < 0.35:
        return "⚠ High Shift Risk"
    if c < 0.60:
        return "👀 Watch Transition"
    return "✅ Stable Trend"


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_health_pct(gv_score, above_50_count) -> float:
    gv = float(gv_score) if pd.notnull(gv_score) else 0.0
    a50 = float(above_50_count) if pd.notnull(above_50_count) else 0.0
    a50_pct = (a50 / 6.0) * 100.0
    score = (gv * 0.85) + (a50_pct * 0.15)
    return round(clamp(score, 0.0, 100.0), 2)


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_buyzone_score(in_buyzone) -> float:
    return 25.0 if bool(in_buyzone) else 75.0


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_fvg_score(fvg_direction) -> float:
    direction = str(fvg_direction).strip().lower() if pd.notnull(fvg_direction) else ""
    if direction in ["bullish", "up", "green"]:
        return 100.0
    if direction in ["bearish", "down", "red"]:
        return 0.0
    return 50.0


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_direction_pct(in_buyzone, fvg_direction, sector_direction_pct) -> float:
    buyzone_score = compute_buyzone_score(in_buyzone)
    fvg_score = compute_fvg_score(fvg_direction)
    sector_score = float(sector_direction_pct) if pd.notnull(sector_direction_pct) else 50.0

    score = (
        (buyzone_score * 0.35)
        + (fvg_score * 0.35)
        + (sector_score * 0.30)
    )

    return round(clamp(score, 0.0, 100.0), 2)


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_gfv_score(gfv_status) -> float:
    status = str(gfv_status).strip().lower() if pd.notnull(gfv_status) else "gray"

    if status == "gold":
        return 100.0
    if status == "green":
        return 75.0
    if status == "gray":
        return 50.0
    return 0.0


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_opportunity_pct(greer_yield_score, gfv_status) -> float:
    ys = float(greer_yield_score) if pd.notnull(greer_yield_score) else 0.0
    ys_pct = (ys / 4.0) * 100.0
    gfv_pct = compute_gfv_score(gfv_status)

    score = (ys_pct * 0.50) + (gfv_pct * 0.50)
    return round(clamp(score, 0.0, 100.0), 2)


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_company_index(health_pct, direction_pct, opportunity_pct) -> float:
    score = (float(health_pct) + float(direction_pct) + float(opportunity_pct)) / 3.0
    return round(clamp(score, 0.0, 100.0), 2)


# ----------------------------------------------------------
# Company index component formulas
# ----------------------------------------------------------
def compute_company_buyzone_proxy(direction_pct: float) -> float:
    return round(clamp(100.0 - float(direction_pct), 0.0, 100.0), 2)


# ----------------------------------------------------------
# Database setup
# ----------------------------------------------------------
def ensure_table_exists(engine) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS greer_company_index_daily (
        date                        date            NOT NULL,
        ticker                      text            NOT NULL,
        sector                      text,
        health_pct                  numeric(6,2),
        direction_pct               numeric(6,2),
        opportunity_pct             numeric(6,2),
        greer_company_index         numeric(6,2),
        phase                       text,
        confidence                  numeric(6,4),
        transition_risk             text,
        sector_health_pct           numeric(6,2),
        sector_direction_pct        numeric(6,2),
        sector_opportunity_pct      numeric(6,2),
        sector_greer_market_index   numeric(6,2),
        created_at                  timestamp       DEFAULT now(),
        updated_at                  timestamp       DEFAULT now(),
        PRIMARY KEY (date, ticker)
    );

    CREATE INDEX IF NOT EXISTS idx_greer_company_index_daily_ticker
        ON greer_company_index_daily (ticker, date DESC);

    CREATE INDEX IF NOT EXISTS idx_greer_company_index_daily_date
        ON greer_company_index_daily (date DESC);
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))

    logger.info("Ensured greer_company_index_daily table exists.")


# ----------------------------------------------------------
# Ticker loading
# ----------------------------------------------------------
def load_tickers_from_file(file_path: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {file_path}")

    df = pd.read_csv(path, header=None)
    tickers = (
        df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    tickers = [t for t in tickers if t]
    return sorted(set(tickers))


# ----------------------------------------------------------
# Ticker loading
# ----------------------------------------------------------
def load_tickers_from_db(engine) -> list[str]:
    query = """
    SELECT ticker
    FROM companies
    WHERE COALESCE(delisted, false) = false
    ORDER BY ticker;
    """

    df = pd.read_sql(query, engine)
    if df.empty:
        return []

    return df["ticker"].dropna().astype(str).str.upper().tolist()


# ----------------------------------------------------------
# Ticker loading
# ----------------------------------------------------------
def get_tickers(engine, args) -> list[str]:
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers if str(t).strip()]
        return sorted(set(tickers))

    if args.file:
        return load_tickers_from_file(args.file)

    tickers = load_tickers_from_db(engine)
    return sorted(set(tickers))


# ----------------------------------------------------------
# Company metadata
# ----------------------------------------------------------
def get_company_sector(engine, ticker: str) -> str | None:
    query = """
    SELECT sector
    FROM companies
    WHERE ticker = %(ticker)s
    LIMIT 1;
    """

    df = pd.read_sql(query, engine, params={"ticker": ticker})
    if df.empty:
        return None

    sector = df.loc[0, "sector"]
    return sector if pd.notnull(sector) else None


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_gv_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        date::date AS date,
        greer_score,
        above_50_count
    FROM greer_scores_daily
    WHERE ticker = %(ticker)s
      AND (%(start_date)s IS NULL OR date >= %(start_date)s)
    ORDER BY date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_yield_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        date::date AS date,
        score AS greer_yield_score
    FROM greer_yields_daily
    WHERE ticker = %(ticker)s
      AND (%(start_date)s IS NULL OR date >= %(start_date)s)
    ORDER BY date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_buyzone_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        date::date AS date,
        in_buyzone
    FROM greer_buyzone_daily
    WHERE ticker = %(ticker)s
      AND (%(start_date)s IS NULL OR date >= %(start_date)s)
    ORDER BY date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_gfv_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        date::date AS date,
        gfv_status
    FROM greer_fair_value_daily
    WHERE ticker = %(ticker)s
      AND (%(start_date)s IS NULL OR date >= %(start_date)s)
    ORDER BY date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_fvg_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        date::date AS date,
        direction
    FROM fair_value_gaps
    WHERE ticker = %(ticker)s
      AND (%(start_date)s IS NULL OR date >= %(start_date)s)
    ORDER BY date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical data loaders
# ----------------------------------------------------------
def load_sector_history(engine, sector: str, start_date=None) -> pd.DataFrame:
    query = """
    SELECT
        summary_date::date AS date,
        sector,
        total_companies,
        gv_gold,
        gv_green,
        gv_red,
        ys_gold,
        ys_green,
        ys_red,
        gfv_gold,
        gfv_green,
        gfv_red,
        gfv_gray,
        buyzone_pct,
        greer_market_index
    FROM sector_summary_daily
    WHERE sector = %(sector)s
      AND (%(start_date)s IS NULL OR summary_date >= %(start_date)s)
    ORDER BY summary_date;
    """

    df = pd.read_sql(
        query,
        engine,
        params={"sector": sector, "start_date": start_date},
        parse_dates=["date"],
    )
    return df


# ----------------------------------------------------------
# Historical series builder
# ----------------------------------------------------------
def build_date_spine(
    gv_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    buyzone_df: pd.DataFrame,
    gfv_df: pd.DataFrame,
) -> pd.DataFrame:
    all_dates = pd.concat(
        [
            gv_df[["date"]] if not gv_df.empty else pd.DataFrame(columns=["date"]),
            yield_df[["date"]] if not yield_df.empty else pd.DataFrame(columns=["date"]),
            buyzone_df[["date"]] if not buyzone_df.empty else pd.DataFrame(columns=["date"]),
            gfv_df[["date"]] if not gfv_df.empty else pd.DataFrame(columns=["date"]),
        ],
        ignore_index=True,
    )

    if all_dates.empty:
        return pd.DataFrame(columns=["date"])

    spine = (
        all_dates.drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return spine


# ----------------------------------------------------------
# Historical series builder
# ----------------------------------------------------------
def prepare_sector_metrics(sector_df: pd.DataFrame) -> pd.DataFrame:
    if sector_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sector_health_pct",
                "sector_direction_pct",
                "sector_opportunity_pct",
                "sector_greer_market_index",
                "sector_phase",
                "sector_confidence",
            ]
        )

    df = sector_df.copy()

    def calc_row(row):
        total = int(row["total_companies"]) if pd.notnull(row["total_companies"]) else 0

        if total > 0:
            gv_bullish_pct = ((int(row["gv_green"]) + int(row["gv_gold"])) / total) * 100.0
            ys_bullish_pct = ((int(row["ys_green"]) + int(row["ys_gold"])) / total) * 100.0
            gfv_bullish_pct = ((int(row["gfv_green"]) + int(row["gfv_gold"])) / total) * 100.0
        else:
            gv_bullish_pct = 0.0
            ys_bullish_pct = 0.0
            gfv_bullish_pct = 0.0

        buyzone_pct = float(row["buyzone_pct"]) if pd.notnull(row["buyzone_pct"]) else 50.0
        sector_direction_pct = round(100.0 - buyzone_pct, 2)
        sector_health_pct = round(gv_bullish_pct, 2)
        sector_opportunity_pct = round((ys_bullish_pct + gfv_bullish_pct) / 2.0, 2)
        sector_gmi = float(row["greer_market_index"]) if pd.notnull(row["greer_market_index"]) else None

        phase, confidence = classify_phase_with_confidence(
            sector_health_pct,
            buyzone_pct,
            sector_opportunity_pct,
        )

        return pd.Series(
            {
                "sector_health_pct": sector_health_pct,
                "sector_direction_pct": sector_direction_pct,
                "sector_opportunity_pct": sector_opportunity_pct,
                "sector_greer_market_index": round(sector_gmi, 2) if sector_gmi is not None else None,
                "sector_phase": phase,
                "sector_confidence": round(float(confidence), 4),
            }
        )

    metrics = df.apply(calc_row, axis=1)
    df = pd.concat([df[["date"]], metrics], axis=1)

    return df


# ----------------------------------------------------------
# Historical series builder
# ----------------------------------------------------------
def build_company_index_history(engine, ticker: str, start_date=None) -> pd.DataFrame:
    sector = get_company_sector(engine, ticker)
    if not sector:
        logger.warning("%s: no sector found in companies table.", ticker)
        return pd.DataFrame()

    gv_df = load_gv_history(engine, ticker, start_date=start_date)
    yield_df = load_yield_history(engine, ticker, start_date=start_date)
    buyzone_df = load_buyzone_history(engine, ticker, start_date=start_date)
    gfv_df = load_gfv_history(engine, ticker, start_date=start_date)
    fvg_df = load_fvg_history(engine, ticker, start_date=start_date)
    sector_df = load_sector_history(engine, sector, start_date=start_date)

    spine = build_date_spine(gv_df, yield_df, buyzone_df, gfv_df)
    if spine.empty:
        logger.warning("%s: no daily component history found.", ticker)
        return pd.DataFrame()

    df = spine.copy()
    df["ticker"] = ticker
    df["sector"] = sector

    if not gv_df.empty:
        df = df.merge(gv_df, on="date", how="left")

    if not yield_df.empty:
        df = df.merge(yield_df, on="date", how="left")

    if not buyzone_df.empty:
        df = df.merge(buyzone_df, on="date", how="left")

    if not gfv_df.empty:
        df = df.merge(gfv_df, on="date", how="left")

    sector_metrics_df = prepare_sector_metrics(sector_df)
    if not sector_metrics_df.empty:
        df = df.merge(sector_metrics_df, on="date", how="left")
    else:
        df["sector_health_pct"] = None
        df["sector_direction_pct"] = 50.0
        df["sector_opportunity_pct"] = None
        df["sector_greer_market_index"] = None
        df["sector_phase"] = None
        df["sector_confidence"] = None

    if not fvg_df.empty:
        df = pd.merge_asof(
            df.sort_values("date"),
            fvg_df.sort_values("date").rename(columns={"direction": "fvg_direction"}),
            on="date",
            direction="backward",
        )
    else:
        df["fvg_direction"] = None

    df["health_pct"] = df.apply(
        lambda row: compute_health_pct(row.get("greer_score"), row.get("above_50_count")),
        axis=1,
    )

    df["direction_pct"] = df.apply(
        lambda row: compute_direction_pct(
            row.get("in_buyzone"),
            row.get("fvg_direction"),
            row.get("sector_direction_pct"),
        ),
        axis=1,
    )

    df["opportunity_pct"] = df.apply(
        lambda row: compute_opportunity_pct(
            row.get("greer_yield_score"),
            row.get("gfv_status"),
        ),
        axis=1,
    )

    df["greer_company_index"] = df.apply(
        lambda row: compute_company_index(
            row.get("health_pct"),
            row.get("direction_pct"),
            row.get("opportunity_pct"),
        ),
        axis=1,
    )

    def classify_company_row(row):
        buyzone_proxy = compute_company_buyzone_proxy(row["direction_pct"])
        phase, confidence = classify_phase_with_confidence(
            row["health_pct"],
            buyzone_proxy,
            row["opportunity_pct"],
        )

        return pd.Series(
            {
                "phase": phase,
                "confidence": round(float(confidence), 4),
                "transition_risk": transition_risk_label(confidence),
            }
        )

    phase_df = df.apply(classify_company_row, axis=1)
    df = pd.concat([df, phase_df], axis=1)

    output_cols = [
        "date",
        "ticker",
        "sector",
        "health_pct",
        "direction_pct",
        "opportunity_pct",
        "greer_company_index",
        "phase",
        "confidence",
        "transition_risk",
        "sector_health_pct",
        "sector_direction_pct",
        "sector_opportunity_pct",
        "sector_greer_market_index",
    ]

    df = df[output_cols].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


# ----------------------------------------------------------
# Database writes
# ----------------------------------------------------------
def upsert_company_index_rows(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    records = df.to_dict(orient="records")

    upsert_sql = """
    INSERT INTO greer_company_index_daily (
        date,
        ticker,
        sector,
        health_pct,
        direction_pct,
        opportunity_pct,
        greer_company_index,
        phase,
        confidence,
        transition_risk,
        sector_health_pct,
        sector_direction_pct,
        sector_opportunity_pct,
        sector_greer_market_index,
        updated_at
    )
    VALUES (
        :date,
        :ticker,
        :sector,
        :health_pct,
        :direction_pct,
        :opportunity_pct,
        :greer_company_index,
        :phase,
        :confidence,
        :transition_risk,
        :sector_health_pct,
        :sector_direction_pct,
        :sector_opportunity_pct,
        :sector_greer_market_index,
        now()
    )
    ON CONFLICT (date, ticker)
    DO UPDATE SET
        sector = EXCLUDED.sector,
        health_pct = EXCLUDED.health_pct,
        direction_pct = EXCLUDED.direction_pct,
        opportunity_pct = EXCLUDED.opportunity_pct,
        greer_company_index = EXCLUDED.greer_company_index,
        phase = EXCLUDED.phase,
        confidence = EXCLUDED.confidence,
        transition_risk = EXCLUDED.transition_risk,
        sector_health_pct = EXCLUDED.sector_health_pct,
        sector_direction_pct = EXCLUDED.sector_direction_pct,
        sector_opportunity_pct = EXCLUDED.sector_opportunity_pct,
        sector_greer_market_index = EXCLUDED.sector_greer_market_index,
        updated_at = now();
    """

    with engine.begin() as conn:
        conn.execute(text(upsert_sql), records)

    return len(records)


# ----------------------------------------------------------
# Optional cleanup
# ----------------------------------------------------------
def delete_ticker_history(engine, ticker: str, start_date=None) -> None:
    if start_date:
        delete_sql = """
        DELETE FROM greer_company_index_daily
        WHERE ticker = :ticker
          AND date >= :start_date;
        """
        params = {"ticker": ticker, "start_date": start_date}
    else:
        delete_sql = """
        DELETE FROM greer_company_index_daily
        WHERE ticker = :ticker;
        """
        params = {"ticker": ticker}

    with engine.begin() as conn:
        conn.execute(text(delete_sql), params)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Greer Company Index daily history.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="One or more tickers to process, e.g. --tickers AAPL NVDA MSFT",
    )
    parser.add_argument(
        "--file",
        help="Optional CSV/txt file containing tickers, one per line.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Delete existing rows for ticker(s) before rebuilding.",
    )
    return parser.parse_args()


# ----------------------------------------------------------
# Main runner
# ----------------------------------------------------------
def main():
    args = parse_args()
    engine = get_engine()

    ensure_table_exists(engine)

    tickers = get_tickers(engine, args)
    if not tickers:
        print_logger.info("No tickers found to process.")
        return

    print_logger.info("Processing %s tickers", len(tickers))
    logger.info("Processing %s tickers", len(tickers))

    total_rows = 0

    for ticker in tickers:
        try:
            print_logger.info("Building Greer Company Index history for %s", ticker)
            logger.info("Building Greer Company Index history for %s", ticker)

            if args.full_refresh:
                delete_ticker_history(engine, ticker, start_date=args.start_date)

            df = build_company_index_history(
                engine,
                ticker,
                start_date=args.start_date,
            )

            if df.empty:
                print_logger.info("%s: no rows generated", ticker)
                logger.warning("%s: no rows generated", ticker)
                continue

            row_count = upsert_company_index_rows(engine, df)
            total_rows += row_count

            print_logger.info("%s: upserted %s rows", ticker, row_count)
            logger.info("%s: upserted %s rows", ticker, row_count)

        except Exception as exc:
            print_logger.info("%s: FAILED - %s", ticker, exc)
            logger.exception("%s: FAILED - %s", ticker, exc)

    print_logger.info("Done. Total rows upserted: %s", total_rows)
    logger.info("Done. Total rows upserted: %s", total_rows)


if __name__ == "__main__":
    main()