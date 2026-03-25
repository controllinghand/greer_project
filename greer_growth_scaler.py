# greer_growth_scaler.py

import os
import math
import logging
from decimal import Decimal
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import text
from db import get_engine


# ----------------------------------------------------------
# Configure logging
# ----------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "greer_growth_scaler.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Growth fund codes
# ----------------------------------------------------------
GROWTH_FUNDS = ("YR3G-26", "YROG-26", "YRRG-26", "YRVG-26")


# ----------------------------------------------------------
# Safely convert values to float
# ----------------------------------------------------------
def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default

# ----------------------------------------------------------
# Insert simulated SELL_SHARES event into portfolio_events
# Returns the new event_id
# ----------------------------------------------------------
def insert_portfolio_sell_event(
    engine,
    portfolio_id: int,
    ticker: str,
    shares_to_sell: float,
    market_price: float,
    notes: str,
) -> int:
    quantity = -abs(float(shares_to_sell))
    price = float(market_price)
    fees = 0.0
    cash_delta = abs(quantity) * price

    query = text("""
        INSERT INTO portfolio_events (
            portfolio_id,
            event_time,
            event_type,
            ticker,
            quantity,
            price,
            fees,
            cash_delta,
            notes
        )
        VALUES (
            :portfolio_id,
            NOW(),
            'SELL_SHARES',
            :ticker,
            :quantity,
            :price,
            :fees,
            :cash_delta,
            :notes
        )
        RETURNING event_id
    """)

    with engine.begin() as conn:
        event_id = conn.execute(
            query,
            {
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "cash_delta": cash_delta,
                "notes": notes,
            },
        ).scalar_one()

    return int(event_id)

# ----------------------------------------------------------
# Mark trade signal executed and attach execution_event_id
# ----------------------------------------------------------
def mark_signal_executed(engine, signal_id: int, execution_event_id: int) -> None:
    query = text("""
        UPDATE growth_trade_signals
        SET status = 'EXECUTED',
            execution_event_id = :execution_event_id,
            updated_at = NOW()
        WHERE signal_id = :signal_id
    """)

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "signal_id": signal_id,
                "execution_event_id": execution_event_id,
            },
        )

# ----------------------------------------------------------
# Load current active growth stock positions
# Notes:
# - avg_cost_basis is weighted from BUY_SHARES only
# - current_shares is net quantity from portfolio_events
# - first_buy_date is earliest BUY_SHARES date
# ----------------------------------------------------------
def load_growth_positions(engine) -> pd.DataFrame:
    query = text("""
        WITH base AS (
            SELECT
                p.portfolio_id,
                p.code AS portfolio_code,
                e.ticker,
                MIN(CASE WHEN e.event_type = 'BUY_SHARES' THEN e.event_time::date END) AS first_buy_date,
                SUM(CASE WHEN e.event_type = 'BUY_SHARES' THEN ABS(e.quantity) ELSE 0 END) AS total_buy_shares,
                SUM(CASE WHEN e.event_type = 'BUY_SHARES' THEN (ABS(e.quantity) * e.price) + COALESCE(e.fees, 0) ELSE 0 END) AS total_buy_cost,
                SUM(
                    CASE
                        WHEN e.event_type = 'BUY_SHARES' THEN ABS(e.quantity)
                        WHEN e.event_type = 'SELL_SHARES' THEN -ABS(e.quantity)
                        ELSE 0
                    END
                ) AS current_shares
            FROM portfolio_events e
            JOIN portfolios p
              ON p.portfolio_id = e.portfolio_id
            WHERE p.code IN :fund_codes
              AND e.ticker IS NOT NULL
              AND e.event_type IN ('BUY_SHARES', 'SELL_SHARES')
            GROUP BY p.portfolio_id, p.code, e.ticker
        ),
        latest_prices AS (
            SELECT DISTINCT ON (ticker)
                ticker,
                date AS price_date,
                close AS current_price
            FROM prices
            WHERE ticker IN (SELECT ticker FROM base)
            ORDER BY ticker, date DESC
        )
        SELECT
            b.portfolio_id,
            b.portfolio_code,
            b.ticker,
            b.first_buy_date,
            b.total_buy_shares,
            b.total_buy_cost,
            b.current_shares,
            CASE
                WHEN COALESCE(b.total_buy_shares, 0) > 0
                THEN b.total_buy_cost / b.total_buy_shares
                ELSE 0
            END AS avg_cost_basis,
            lp.price_date,
            lp.current_price
        FROM base b
        LEFT JOIN latest_prices lp
          ON lp.ticker = b.ticker
        WHERE COALESCE(b.current_shares, 0) > 0
        ORDER BY b.portfolio_code, b.ticker
    """)

    return pd.read_sql(query, engine, params={"fund_codes": GROWTH_FUNDS})


# ----------------------------------------------------------
# Load active growth rules and steps
# ----------------------------------------------------------
def load_rules(engine) -> Dict[int, Dict[str, Any]]:
    query = text("""
        SELECT
            gfr.rule_id,
            gfr.portfolio_id,
            gfr.portfolio_code,
            gfr.stop_loss_pct,
            gfr.runner_floor_pct,
            gfr.auto_execute_enabled,
            grs.step_order,
            grs.gain_trigger_pct,
            grs.sell_pct
        FROM growth_fund_rules gfr
        JOIN growth_rule_steps grs
          ON grs.rule_id = gfr.rule_id
        WHERE gfr.is_active = TRUE
          AND grs.is_active = TRUE
        ORDER BY gfr.portfolio_code, grs.step_order
    """)

    df = pd.read_sql(query, engine)

    rules: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        portfolio_id = int(row["portfolio_id"])
        if portfolio_id not in rules:
            rules[portfolio_id] = {
                "portfolio_code": row["portfolio_code"],
                "stop_loss_pct": to_float(row["stop_loss_pct"]),
                "runner_floor_pct": to_float(row["runner_floor_pct"]),
                "auto_execute_enabled": bool(row["auto_execute_enabled"]),
                "steps": [],
            }

        rules[portfolio_id]["steps"].append(
            {
                "step_order": int(row["step_order"]),
                "gain_trigger_pct": to_float(row["gain_trigger_pct"]),
                "sell_pct": to_float(row["sell_pct"]),
            }
        )

    return rules


# ----------------------------------------------------------
# Load last executed signal by portfolio/ticker
# ----------------------------------------------------------
def load_last_executed_steps(engine) -> Dict[tuple, float]:
    query = text("""
        SELECT
            portfolio_id,
            ticker,
            MAX(trigger_gain_pct) AS last_executed_gain_pct
        FROM growth_trade_signals
        WHERE status = 'EXECUTED'
          AND signal_type = 'PROFIT_SCALE'
        GROUP BY portfolio_id, ticker
    """)

    df = pd.read_sql(query, engine)

    result: Dict[tuple, float] = {}
    for _, row in df.iterrows():
        result[(int(row["portfolio_id"]), row["ticker"])] = to_float(row["last_executed_gain_pct"])
    return result


# ----------------------------------------------------------
# Check whether a position already has an open signal
# ----------------------------------------------------------
def load_open_signals(engine) -> Dict[tuple, Dict[str, Any]]:
    query = text("""
        SELECT
            signal_id,
            portfolio_id,
            ticker,
            signal_type,
            trigger_gain_pct,
            expected_shares_after,
            status
        FROM growth_trade_signals
        WHERE status IN ('OPEN', 'SENT')
    """)

    df = pd.read_sql(query, engine)

    result: Dict[tuple, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        result[(int(row["portfolio_id"]), row["ticker"])] = {
            "signal_id": int(row["signal_id"]),
            "signal_type": row["signal_type"],
            "trigger_gain_pct": to_float(row["trigger_gain_pct"]),
            "expected_shares_after": to_float(row["expected_shares_after"]),
            "status": row["status"],
        }
    return result


# ----------------------------------------------------------
# Attempt to auto-mark open signals as executed
# If actual shares are now <= expected_shares_after, assume
# the user entered the sell in portfolio_events
# ----------------------------------------------------------
def reconcile_open_signals(engine, positions_df: pd.DataFrame) -> None:
    current_shares_map = {
        (int(row["portfolio_id"]), row["ticker"]): to_float(row["current_shares"])
        for _, row in positions_df.iterrows()
    }

    query = text("""
        SELECT
            signal_id,
            portfolio_id,
            ticker,
            expected_shares_after,
            status
        FROM growth_trade_signals
        WHERE status IN ('OPEN', 'SENT')
    """)

    with engine.begin() as conn:
        results = conn.execute(query).mappings().all()

        for row in results:
            key = (int(row["portfolio_id"]), row["ticker"])
            actual_shares = current_shares_map.get(key)
            expected_shares_after = to_float(row["expected_shares_after"])

            if actual_shares is None:
                continue

            if actual_shares <= expected_shares_after + 0.01:
                conn.execute(
                    text("""
                        UPDATE growth_trade_signals
                        SET status = 'EXECUTED',
                            updated_at = NOW()
                        WHERE signal_id = :signal_id
                    """),
                    {"signal_id": row["signal_id"]},
                )
                logger.info("Marked signal %s as EXECUTED", row["signal_id"])


# ----------------------------------------------------------
# Calculate next pending rule step
# ----------------------------------------------------------
def get_next_step(steps: List[Dict[str, Any]], last_executed_gain_pct: Optional[float]) -> Optional[Dict[str, Any]]:
    last_level = last_executed_gain_pct if last_executed_gain_pct is not None else 0.0

    for step in steps:
        if step["gain_trigger_pct"] > last_level:
            return step

    return None


# ----------------------------------------------------------
# Upsert current growth position state
# ----------------------------------------------------------
def upsert_growth_position_state(engine, payload: Dict[str, Any]) -> None:
    query = text("""
        INSERT INTO growth_position_state (
            portfolio_id,
            portfolio_code,
            ticker,
            first_buy_date,
            initial_shares,
            current_shares,
            avg_cost_basis,
            cost_basis_value,
            current_price,
            market_value,
            unrealized_pl,
            unrealized_pl_pct,
            stop_price,
            runner_floor_shares,
            capital_recovered_amt,
            capital_recovered_pct,
            last_executed_gain_pct,
            next_gain_trigger_pct,
            next_sell_pct,
            position_status,
            last_price_date,
            updated_at
        )
        VALUES (
            :portfolio_id,
            :portfolio_code,
            :ticker,
            :first_buy_date,
            :initial_shares,
            :current_shares,
            :avg_cost_basis,
            :cost_basis_value,
            :current_price,
            :market_value,
            :unrealized_pl,
            :unrealized_pl_pct,
            :stop_price,
            :runner_floor_shares,
            :capital_recovered_amt,
            :capital_recovered_pct,
            :last_executed_gain_pct,
            :next_gain_trigger_pct,
            :next_sell_pct,
            :position_status,
            :last_price_date,
            NOW()
        )
        ON CONFLICT (portfolio_id, ticker)
        DO UPDATE SET
            first_buy_date = EXCLUDED.first_buy_date,
            initial_shares = EXCLUDED.initial_shares,
            current_shares = EXCLUDED.current_shares,
            avg_cost_basis = EXCLUDED.avg_cost_basis,
            cost_basis_value = EXCLUDED.cost_basis_value,
            current_price = EXCLUDED.current_price,
            market_value = EXCLUDED.market_value,
            unrealized_pl = EXCLUDED.unrealized_pl,
            unrealized_pl_pct = EXCLUDED.unrealized_pl_pct,
            stop_price = EXCLUDED.stop_price,
            runner_floor_shares = EXCLUDED.runner_floor_shares,
            capital_recovered_amt = EXCLUDED.capital_recovered_amt,
            capital_recovered_pct = EXCLUDED.capital_recovered_pct,
            last_executed_gain_pct = EXCLUDED.last_executed_gain_pct,
            next_gain_trigger_pct = EXCLUDED.next_gain_trigger_pct,
            next_sell_pct = EXCLUDED.next_sell_pct,
            position_status = EXCLUDED.position_status,
            last_price_date = EXCLUDED.last_price_date,
            updated_at = NOW()
    """)

    with engine.begin() as conn:
        conn.execute(query, payload)


# ----------------------------------------------------------
# Insert a new trade signal
# ----------------------------------------------------------
def insert_trade_signal(engine, payload: Dict[str, Any]) -> int:
    query = text("""
        INSERT INTO growth_trade_signals (
            portfolio_id,
            portfolio_code,
            ticker,
            signal_type,
            trigger_gain_pct,
            sell_pct,
            shares_before,
            shares_to_sell,
            expected_shares_after,
            trigger_price,
            market_price,
            avg_cost_basis,
            status,
            notes,
            created_at,
            updated_at
        )
        VALUES (
            :portfolio_id,
            :portfolio_code,
            :ticker,
            :signal_type,
            :trigger_gain_pct,
            :sell_pct,
            :shares_before,
            :shares_to_sell,
            :expected_shares_after,
            :trigger_price,
            :market_price,
            :avg_cost_basis,
            'OPEN',
            :notes,
            NOW(),
            NOW()
        )
        RETURNING signal_id
    """)

    with engine.begin() as conn:
        signal_id = conn.execute(query, payload).scalar_one()
        return int(signal_id)

# ----------------------------------------------------------
# Auto-execute one growth trade signal into portfolio_events
# ----------------------------------------------------------
def auto_execute_signal(engine, signal_id: int, payload: Dict[str, Any]) -> int:
    if payload["signal_type"] == "STOP_LOSS":
        note = f"AUTO: Growth stop-loss -{payload['trigger_gain_pct']:.0f}% rule"
    else:
        note = f"AUTO: Growth profit-scale +{payload['trigger_gain_pct']:.0f}% rule"

    event_id = insert_portfolio_sell_event(
        engine=engine,
        portfolio_id=int(payload["portfolio_id"]),
        ticker=str(payload["ticker"]),
        shares_to_sell=float(payload["shares_to_sell"]),
        market_price=float(payload["market_price"]),
        notes=note,
    )

    mark_signal_executed(engine, signal_id, event_id)
    return event_id

# ----------------------------------------------------------
# Main rule evaluation logic
# ----------------------------------------------------------
def evaluate_growth_rules(engine) -> None:
    positions_df = load_growth_positions(engine)

    if positions_df.empty:
        logger.info("No active growth positions found")
        return

    reconcile_open_signals(engine, positions_df)

    rules = load_rules(engine)
    open_signals_map = load_open_signals(engine)

    for _, row in positions_df.iterrows():
        portfolio_id = int(row["portfolio_id"])
        portfolio_code = row["portfolio_code"]
        ticker = row["ticker"]

        if portfolio_id not in rules:
            logger.warning("No active growth rule profile found for %s", portfolio_code)
            continue

        rule = rules[portfolio_id]

        avg_cost_basis = to_float(row["avg_cost_basis"])
        current_price = to_float(row["current_price"])
        current_shares = to_float(row["current_shares"])
        initial_shares = to_float(row["total_buy_shares"])
        first_buy_date = row["first_buy_date"]
        price_date = row["price_date"]

        if avg_cost_basis <= 0 or current_price <= 0 or current_shares <= 0 or initial_shares <= 0:
            logger.warning("Skipping %s %s due to incomplete data", portfolio_code, ticker)
            continue

        executed = False

        # ----------------------------------------------------------
        # Calculate current state before any new signal/execution
        # ----------------------------------------------------------
        stop_price = avg_cost_basis * (1.0 - (rule["stop_loss_pct"] / 100.0))
        runner_floor_shares = initial_shares * (rule["runner_floor_pct"] / 100.0)

        market_value = current_shares * current_price
        cost_basis_value = current_shares * avg_cost_basis
        unrealized_pl = market_value - cost_basis_value
        unrealized_pl_pct = ((current_price - avg_cost_basis) / avg_cost_basis) * 100.0
        gain_pct = unrealized_pl_pct

        key = (portfolio_id, ticker)

        last_executed_map = load_last_executed_steps(engine)
        last_executed_gain_pct = last_executed_map.get(key)
        next_step = get_next_step(rule["steps"], last_executed_gain_pct)

        capital_recovered_amt = max(0.0, (initial_shares - current_shares) * avg_cost_basis)
        capital_recovered_pct = (
            (capital_recovered_amt / (initial_shares * avg_cost_basis)) * 100.0
            if initial_shares > 0 else 0.0
        )

        position_status = "ACTIVE"
        if current_price <= stop_price:
            position_status = "STOP_TRIGGERED"
        elif current_shares <= runner_floor_shares + 0.01:
            position_status = "RUNNER"

        next_gain_trigger_pct = next_step["gain_trigger_pct"] if next_step else None
        next_sell_pct = next_step["sell_pct"] if next_step else None

        open_signal = open_signals_map.get(key)

        # ----------------------------------------------------------
        # If an open signal already exists, do not create another one
        # ----------------------------------------------------------
        if open_signal:
            logger.info("Open signal already exists for %s %s", portfolio_code, ticker)

        else:
            # ----------------------------------------------------------
            # Stop-loss alert
            # ----------------------------------------------------------
            if current_price <= stop_price:
                payload = {
                    "portfolio_id": portfolio_id,
                    "portfolio_code": portfolio_code,
                    "ticker": ticker,
                    "signal_type": "STOP_LOSS",
                    "trigger_gain_pct": rule["stop_loss_pct"],
                    "sell_pct": 100.0,
                    "shares_before": current_shares,
                    "shares_to_sell": current_shares,
                    "expected_shares_after": 0.0,
                    "trigger_price": stop_price,
                    "market_price": current_price,
                    "avg_cost_basis": avg_cost_basis,
                    "notes": "Growth stop-loss rule hit",
                }

                signal_id = insert_trade_signal(engine, payload)

                if rule["auto_execute_enabled"]:
                    execution_event_id = auto_execute_signal(engine, signal_id, payload)
                    executed = True
                    logger.info(
                        "Auto-executed STOP_LOSS for %s %s with event_id=%s",
                        portfolio_code,
                        ticker,
                        execution_event_id,
                    )

                logger.info("Created STOP_LOSS signal for %s %s", portfolio_code, ticker)

            # ----------------------------------------------------------
            # Runner protection: stop scaling once shares are already at floor
            # ----------------------------------------------------------
            elif position_status == "RUNNER":
                logger.info("Runner floor reached for %s %s", portfolio_code, ticker)

            # ----------------------------------------------------------
            # Profit-scale alert
            # ----------------------------------------------------------
            elif next_step and gain_pct >= next_step["gain_trigger_pct"]:
                shares_to_sell = math.floor(current_shares * (next_step["sell_pct"] / 100.0))
                shares_to_sell = max(1, shares_to_sell)

                expected_shares_after = current_shares - shares_to_sell
                if expected_shares_after < 0:
                    expected_shares_after = 0

                payload = {
                    "portfolio_id": portfolio_id,
                    "portfolio_code": portfolio_code,
                    "ticker": ticker,
                    "signal_type": "PROFIT_SCALE",
                    "trigger_gain_pct": next_step["gain_trigger_pct"],
                    "sell_pct": next_step["sell_pct"],
                    "shares_before": current_shares,
                    "shares_to_sell": shares_to_sell,
                    "expected_shares_after": expected_shares_after,
                    "trigger_price": avg_cost_basis * (1.0 + next_step["gain_trigger_pct"] / 100.0),
                    "market_price": current_price,
                    "avg_cost_basis": avg_cost_basis,
                    "notes": "Growth profit-scale rule hit",
                }

                signal_id = insert_trade_signal(engine, payload)

                if rule["auto_execute_enabled"]:
                    execution_event_id = auto_execute_signal(engine, signal_id, payload)
                    executed = True
                    logger.info(
                        "Auto-executed PROFIT_SCALE for %s %s with event_id=%s",
                        portfolio_code,
                        ticker,
                        execution_event_id,
                    )

                logger.info("Created PROFIT_SCALE signal for %s %s", portfolio_code, ticker)

        # ----------------------------------------------------------
        # If a trade was auto-executed, reload fresh position data
        # so growth_position_state reflects the updated shares/step
        # on the same run.
        # ----------------------------------------------------------
        if executed:
            refreshed_positions = load_growth_positions(engine)
            refreshed_row = refreshed_positions[
                (refreshed_positions["portfolio_id"] == portfolio_id) &
                (refreshed_positions["ticker"] == ticker)
            ]

            if not refreshed_row.empty:
                refreshed = refreshed_row.iloc[0]

                avg_cost_basis = to_float(refreshed["avg_cost_basis"])
                current_price = to_float(refreshed["current_price"])
                current_shares = to_float(refreshed["current_shares"])
                initial_shares = to_float(refreshed["total_buy_shares"])
                first_buy_date = refreshed["first_buy_date"]
                price_date = refreshed["price_date"]

                stop_price = avg_cost_basis * (1.0 - (rule["stop_loss_pct"] / 100.0))
                runner_floor_shares = initial_shares * (rule["runner_floor_pct"] / 100.0)

                market_value = current_shares * current_price
                cost_basis_value = current_shares * avg_cost_basis
                unrealized_pl = market_value - cost_basis_value
                unrealized_pl_pct = ((current_price - avg_cost_basis) / avg_cost_basis) * 100.0

                last_executed_map = load_last_executed_steps(engine)
                last_executed_gain_pct = last_executed_map.get(key)
                next_step = get_next_step(rule["steps"], last_executed_gain_pct)

                capital_recovered_amt = max(0.0, (initial_shares - current_shares) * avg_cost_basis)
                capital_recovered_pct = (
                    (capital_recovered_amt / (initial_shares * avg_cost_basis)) * 100.0
                    if initial_shares > 0 else 0.0
                )

                position_status = "ACTIVE"
                if current_price <= stop_price:
                    position_status = "STOP_TRIGGERED"
                elif current_shares <= runner_floor_shares + 0.01:
                    position_status = "RUNNER"

                next_gain_trigger_pct = next_step["gain_trigger_pct"] if next_step else None
                next_sell_pct = next_step["sell_pct"] if next_step else None

                logger.info(
                    "STATE UPDATED | %s %s | shares=%s | next_trigger=%s",
                    portfolio_code,
                    ticker,
                    current_shares,
                    next_gain_trigger_pct,
                )

        # ----------------------------------------------------------
        # Final state upsert
        # ----------------------------------------------------------
        upsert_growth_position_state(
            engine,
            {
                "portfolio_id": portfolio_id,
                "portfolio_code": portfolio_code,
                "ticker": ticker,
                "first_buy_date": first_buy_date,
                "initial_shares": initial_shares,
                "current_shares": current_shares,
                "avg_cost_basis": avg_cost_basis,
                "cost_basis_value": round(cost_basis_value, 2),
                "current_price": current_price,
                "market_value": round(market_value, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_pl_pct": round(unrealized_pl_pct, 4),
                "stop_price": round(stop_price, 6),
                "runner_floor_shares": round(runner_floor_shares, 4),
                "capital_recovered_amt": round(capital_recovered_amt, 2),
                "capital_recovered_pct": round(capital_recovered_pct, 4),
                "last_executed_gain_pct": last_executed_gain_pct,
                "next_gain_trigger_pct": next_gain_trigger_pct,
                "next_sell_pct": next_sell_pct,
                "position_status": position_status,
                "last_price_date": price_date,
            },
        )


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
def main() -> None:
    logger.info("Starting greer_growth_scaler.py")
    engine = get_engine()
    evaluate_growth_rules(engine)
    logger.info("Completed greer_growth_scaler.py")


if __name__ == "__main__":
    main()
