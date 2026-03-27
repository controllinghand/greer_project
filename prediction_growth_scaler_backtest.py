# prediction_growth_scaler_backtest.py

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine


# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
INITIAL_CAPITAL = 10_000.0
STOP_LOSS_PCT = -0.10
MAX_HOLD_DAYS = 60
TARGET_BUCKET = 90

GAIN_STEPS = [
    (0.10, 0.05),
    (0.20, 0.10),
    (0.30, 0.15),
    (0.40, 0.20),
    (0.50, 0.25),
    (0.60, 0.30),
    (0.70, 0.35),
    (0.80, 0.40),
]

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_growth_scaler_backtest.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Data model
# ----------------------------------------------------------
@dataclass
class TradeResult:
    snapshot_date: Any
    ticker: str
    entry_price: float
    initial_capital: float
    initial_shares: float
    calibration_bucket: int
    prediction_score: float
    signal_tier: str
    setup_label: str
    phase: str | None
    prior_phase: str | None
    goi_zone: str | None
    buyzone_flag: bool | None
    raw_return_60d: float | None
    raw_end_value_60d: float | None
    scaler_return_60d: float | None
    scaler_end_value_60d: float | None
    stopped_out: bool
    stop_date: Any
    stop_price: float | None
    steps_triggered: int
    realized_cash_from_sales: float
    final_shares_remaining: float
    final_mark_price: float | None
    final_exit_date: Any
    final_day_num: int | None


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------
def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    return value


# ----------------------------------------------------------
# Load base trades for backtest
# ----------------------------------------------------------
def load_base_trades() -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        WITH scored AS (
            SELECT
                ps.snapshot_date,
                ps.ticker,
                ps.prediction_score,
                ps.phase,
                ps.prior_phase,
                ps.goi_zone,
                ps.buyzone_flag,
                ps.return_60d,
                CASE
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 125 THEN 130
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 105 THEN 110
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 85 THEN 90
                    ELSE NULL
                END AS calibration_bucket,
                CASE
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 125 THEN 'Constructive'
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 105 THEN 'Strong'
                    WHEN ROUND(ps.prediction_score / 10.0) * 10 >= 85 THEN 'High Conviction'
                    ELSE 'Watchlist'
                END AS signal_tier,
                CASE
                    WHEN ps.phase IS NULL OR ps.goi_zone IS NULL THEN 'Unknown'
                    ELSE INITCAP(ps.phase) || ' + ' ||
                         CASE
                             WHEN ps.goi_zone = 'EXTREME_OPPORTUNITY' THEN 'Extreme Opportunity'
                             WHEN ps.goi_zone = 'ELEVATED_OPPORTUNITY' THEN 'Elevated Opportunity'
                             WHEN ps.goi_zone = 'NORMAL' THEN 'Normal'
                             WHEN ps.goi_zone = 'LOW_OPPORTUNITY' THEN 'Low Opportunity'
                             WHEN ps.goi_zone = 'EXTREME_GREED' THEN 'Extreme Greed'
                             ELSE ps.goi_zone
                         END
                END AS setup_label
            FROM prediction_scores_local_v3 ps
        )
        SELECT
            s.snapshot_date,
            s.ticker,
            pr.price AS entry_price,
            s.prediction_score,
            s.calibration_bucket,
            s.signal_tier,
            s.setup_label,
            s.phase,
            s.prior_phase,
            s.goi_zone,
            s.buyzone_flag,
            s.return_60d
        FROM scored s
        JOIN prediction_research_v2 pr
          ON pr.snapshot_date = s.snapshot_date
         AND pr.ticker = s.ticker
        WHERE s.calibration_bucket = :target_bucket
          AND s.return_60d IS NOT NULL
          AND pr.price IS NOT NULL
          AND s.buyzone_flag IS NOT NULL
        ORDER BY s.snapshot_date, s.ticker
        """
    )

    df = pd.read_sql(query, engine, params={"target_bucket": TARGET_BUCKET})
    logger.info("Loaded %s base trades for bucket %s", len(df), TARGET_BUCKET)
    return df


# ----------------------------------------------------------
# Load forward price path for one trade
# ----------------------------------------------------------
def load_forward_prices(ticker: str, snapshot_date: Any) -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        SELECT
            date,
            close
        FROM prices
        WHERE ticker = :ticker
          AND date > :snapshot_date
        ORDER BY date
        LIMIT :max_hold_days
        """
    )

    df = pd.read_sql(
        query,
        engine,
        params={
            "ticker": ticker,
            "snapshot_date": snapshot_date,
            "max_hold_days": MAX_HOLD_DAYS,
        },
    )
    return df


# ----------------------------------------------------------
# Simulate one trade using Growth Scaler
# ----------------------------------------------------------
def simulate_trade(row: pd.Series) -> TradeResult:
    snapshot_date = row["snapshot_date"]
    ticker = row["ticker"]
    entry_price = float(row["entry_price"])
    prediction_score = float(row["prediction_score"])
    calibration_bucket = int(row["calibration_bucket"])
    signal_tier = row["signal_tier"]
    setup_label = row["setup_label"]
    phase = row["phase"]
    prior_phase = row["prior_phase"]
    goi_zone = row["goi_zone"]
    buyzone_flag = row["buyzone_flag"]
    raw_return_60d = float(row["return_60d"])

    forward_df = load_forward_prices(ticker, snapshot_date)

    initial_shares = INITIAL_CAPITAL / entry_price
    remaining_shares = initial_shares
    realized_cash = 0.0
    stopped_out = False
    stop_date = None
    stop_price = None
    steps_triggered = 0
    next_step_idx = 0
    final_mark_price = None
    final_exit_date = None
    final_day_num = None

    if forward_df.empty:
        raw_end_value_60d = INITIAL_CAPITAL * (1.0 + raw_return_60d)
        return TradeResult(
            snapshot_date=snapshot_date,
            ticker=ticker,
            entry_price=entry_price,
            initial_capital=INITIAL_CAPITAL,
            initial_shares=initial_shares,
            calibration_bucket=calibration_bucket,
            prediction_score=prediction_score,
            signal_tier=signal_tier,
            setup_label=setup_label,
            phase=phase,
            prior_phase=prior_phase,
            goi_zone=goi_zone,
            buyzone_flag=buyzone_flag,
            raw_return_60d=raw_return_60d,
            raw_end_value_60d=raw_end_value_60d,
            scaler_return_60d=None,
            scaler_end_value_60d=None,
            stopped_out=False,
            stop_date=None,
            stop_price=None,
            steps_triggered=0,
            realized_cash_from_sales=0.0,
            final_shares_remaining=remaining_shares,
            final_mark_price=None,
            final_exit_date=None,
            final_day_num=None,
        )

    for i, price_row in enumerate(forward_df.itertuples(index=False), start=1):
        day_close = float(price_row.close)
        day_date = price_row.date
        gain_pct = (day_close - entry_price) / entry_price

        # Stop-loss first, based on daily close
        if gain_pct <= STOP_LOSS_PCT and remaining_shares > 0:
            realized_cash += remaining_shares * day_close
            stopped_out = True
            stop_date = day_date
            stop_price = day_close
            remaining_shares = 0.0
            final_mark_price = day_close
            final_exit_date = day_date
            final_day_num = i
            break

        # Only execute the next pending gain step, at most one step per day
        if next_step_idx < len(GAIN_STEPS) and remaining_shares > 0:
            trigger_gain, sell_pct = GAIN_STEPS[next_step_idx]
            if gain_pct >= trigger_gain:
                shares_to_sell = remaining_shares * sell_pct
                realized_cash += shares_to_sell * day_close
                remaining_shares -= shares_to_sell
                steps_triggered += 1
                next_step_idx += 1

        final_mark_price = day_close
        final_exit_date = day_date
        final_day_num = i

    scaler_end_value_60d = realized_cash + (remaining_shares * final_mark_price if final_mark_price is not None else 0.0)
    scaler_return_60d = (scaler_end_value_60d / INITIAL_CAPITAL) - 1.0 if scaler_end_value_60d is not None else None
    raw_end_value_60d = INITIAL_CAPITAL * (1.0 + raw_return_60d)

    return TradeResult(
        snapshot_date=snapshot_date,
        ticker=ticker,
        entry_price=entry_price,
        initial_capital=INITIAL_CAPITAL,
        initial_shares=initial_shares,
        calibration_bucket=calibration_bucket,
        prediction_score=prediction_score,
        signal_tier=signal_tier,
        setup_label=setup_label,
        phase=phase,
        prior_phase=prior_phase,
        goi_zone=goi_zone,
        buyzone_flag=buyzone_flag,
        raw_return_60d=raw_return_60d,
        raw_end_value_60d=raw_end_value_60d,
        scaler_return_60d=scaler_return_60d,
        scaler_end_value_60d=scaler_end_value_60d,
        stopped_out=stopped_out,
        stop_date=stop_date,
        stop_price=stop_price,
        steps_triggered=steps_triggered,
        realized_cash_from_sales=realized_cash,
        final_shares_remaining=remaining_shares,
        final_mark_price=final_mark_price,
        final_exit_date=final_exit_date,
        final_day_num=final_day_num,
    )


# ----------------------------------------------------------
# Create output table
# ----------------------------------------------------------
def recreate_output_table() -> None:
    engine = get_engine()

    drop_sql = text("DROP TABLE IF EXISTS prediction_growth_scaler_backtest_60d")

    create_sql = text(
        """
        CREATE TABLE prediction_growth_scaler_backtest_60d (
            snapshot_date date,
            ticker text,
            entry_price numeric,
            initial_capital numeric,
            initial_shares numeric,
            calibration_bucket integer,
            prediction_score numeric,
            signal_tier text,
            setup_label text,
            phase text,
            prior_phase text,
            goi_zone text,
            buyzone_flag boolean,
            raw_return_60d numeric,
            raw_end_value_60d numeric,
            scaler_return_60d numeric,
            scaler_end_value_60d numeric,
            stopped_out boolean,
            stop_date date,
            stop_price numeric,
            steps_triggered integer,
            realized_cash_from_sales numeric,
            final_shares_remaining numeric,
            final_mark_price numeric,
            final_exit_date date,
            final_day_num integer
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(drop_sql)
        conn.execute(create_sql)

    logger.info("Recreated prediction_growth_scaler_backtest_60d")


# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------
def save_results(results: list[TradeResult]) -> None:
    if not results:
        logger.warning("No backtest results to save")
        return

    engine = get_engine()

    rows = [
        {
            "snapshot_date": clean_value(r.snapshot_date),
            "ticker": clean_value(r.ticker),
            "entry_price": clean_value(r.entry_price),
            "initial_capital": clean_value(r.initial_capital),
            "initial_shares": clean_value(r.initial_shares),
            "calibration_bucket": clean_value(r.calibration_bucket),
            "prediction_score": clean_value(r.prediction_score),
            "signal_tier": clean_value(r.signal_tier),
            "setup_label": clean_value(r.setup_label),
            "phase": clean_value(r.phase),
            "prior_phase": clean_value(r.prior_phase),
            "goi_zone": clean_value(r.goi_zone),
            "buyzone_flag": clean_value(r.buyzone_flag),
            "raw_return_60d": clean_value(r.raw_return_60d),
            "raw_end_value_60d": clean_value(r.raw_end_value_60d),
            "scaler_return_60d": clean_value(r.scaler_return_60d),
            "scaler_end_value_60d": clean_value(r.scaler_end_value_60d),
            "stopped_out": clean_value(r.stopped_out),
            "stop_date": clean_value(r.stop_date),
            "stop_price": clean_value(r.stop_price),
            "steps_triggered": clean_value(r.steps_triggered),
            "realized_cash_from_sales": clean_value(r.realized_cash_from_sales),
            "final_shares_remaining": clean_value(r.final_shares_remaining),
            "final_mark_price": clean_value(r.final_mark_price),
            "final_exit_date": clean_value(r.final_exit_date),
            "final_day_num": clean_value(r.final_day_num),
        }
        for r in results
    ]

    insert_sql = text(
        """
        INSERT INTO prediction_growth_scaler_backtest_60d (
            snapshot_date,
            ticker,
            entry_price,
            initial_capital,
            initial_shares,
            calibration_bucket,
            prediction_score,
            signal_tier,
            setup_label,
            phase,
            prior_phase,
            goi_zone,
            buyzone_flag,
            raw_return_60d,
            raw_end_value_60d,
            scaler_return_60d,
            scaler_end_value_60d,
            stopped_out,
            stop_date,
            stop_price,
            steps_triggered,
            realized_cash_from_sales,
            final_shares_remaining,
            final_mark_price,
            final_exit_date,
            final_day_num
        )
        VALUES (
            :snapshot_date,
            :ticker,
            :entry_price,
            :initial_capital,
            :initial_shares,
            :calibration_bucket,
            :prediction_score,
            :signal_tier,
            :setup_label,
            :phase,
            :prior_phase,
            :goi_zone,
            :buyzone_flag,
            :raw_return_60d,
            :raw_end_value_60d,
            :scaler_return_60d,
            :scaler_end_value_60d,
            :stopped_out,
            :stop_date,
            :stop_price,
            :steps_triggered,
            :realized_cash_from_sales,
            :final_shares_remaining,
            :final_mark_price,
            :final_exit_date,
            :final_day_num
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)

    logger.info("Saved %s backtest rows", len(rows))


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> None:
    start = datetime.now()
    logger.info("Starting prediction growth scaler backtest")

    base_df = load_base_trades()
    recreate_output_table()

    results: list[TradeResult] = []
    total = len(base_df)

    for i, row in enumerate(base_df.itertuples(index=False), start=1):
        result = simulate_trade(pd.Series(row._asdict()))
        results.append(result)

        if i % 5000 == 0 or i == total:
            logger.info("Processed %s / %s trades", i, total)

    save_results(results)

    elapsed = datetime.now() - start
    logger.info("Backtest finished in %s", elapsed)


if __name__ == "__main__":
    main()