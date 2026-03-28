# prediction_utils.py
# ----------------------------------------------------------
# Prediction Scoring Model (DOCUMENTATION ONLY - DO NOT MODIFY LOGIC BELOW)
# ----------------------------------------------------------
"""
RAW PREDICTION SCORE MODEL

The prediction_score is a rules-based composite score that measures
how favorable a company setup is at a given point in time.

This model is NOT machine learning — it is a structured scoring system
built from multiple independent signals.

----------------------------------------------------------
FORMULA
----------------------------------------------------------

prediction_score =
    phase_score
  + transition_score
  + buyzone_score
  + confidence_score
  + fundamentals_score
  + goi_score
  + regime_alignment_score
  + overheat_penalty

----------------------------------------------------------
COMPONENT DEFINITIONS
----------------------------------------------------------

1. Phase Score (Macro Timing)
- CONTRACTION → +30
- RECOVERY    → +25
- EUPHORIA    → +20
- EXPANSION   → +15

→ Rewards earlier cycle entry points

----------------------------------------------------------

2. Transition Score (Inflection Detection)
- CONTRACTION → EXPANSION → +25
- CONTRACTION → RECOVERY  → +20
- EXPANSION → CONTRACTION → +20

→ Captures major turning points in the cycle

----------------------------------------------------------

3. BuyZone Score (Technical Entry Timing)
- BuyZone = TRUE → +15
- Otherwise → 0

→ Ensures timing alignment with pullbacks

----------------------------------------------------------

4. Confidence Score (Signal Strength)
- confidence * 20

Examples:
- 1.0 → +20
- 0.5 → +10

→ Scales conviction of the setup

----------------------------------------------------------

5. Fundamentals Score (Quality + Valuation)
- GV ≥ 60 → +5
- GY ≥ 3  → +5

Max = +10

→ Rewards strong companies at good valuations

----------------------------------------------------------

6. GOI Score (Market Opportunity Environment)
- ELEVATED_OPPORTUNITY → +30
- EXTREME_OPPORTUNITY  → +20
- NORMAL               → +10
- LOW_OPPORTUNITY      → +5
- EXTREME_GREED        → -15

→ Adds macro tailwind / headwind

----------------------------------------------------------

7. Regime Alignment Score (Signal Synergy)
- CONTRACTION + ELEVATED_OPPORTUNITY → +30
- RECOVERY + ELEVATED_OPPORTUNITY    → +25
- CONTRACTION + EXTREME_OPPORTUNITY  → +20
- EUPHORIA + EXTREME_GREED           → -20

→ Rewards "perfect storm" setups

----------------------------------------------------------

8. Overheat Penalty (Risk Control)
- EUPHORIA + EXTREME_GREED → -30
- EXPANSION + EXTREME_GREED → -20
- EUPHORIA + no BuyZone → -10

→ Prevents chasing overheated markets

----------------------------------------------------------
INTERPRETATION
----------------------------------------------------------

The raw score represents:

    "How ideal is this setup right now?"

Higher is better, but the relationship is NOT linear.

----------------------------------------------------------
BUCKET SYSTEM
----------------------------------------------------------

Raw scores are converted into:

1. Raw Bucket (rounded to nearest 10)
2. Calibration Bucket:
    ≥125 → 130
    ≥105 → 110
    ≥85  → 90

Calibration buckets map to historical:
- Expected win rate
- Expected return

----------------------------------------------------------
EXAMPLE (OKTA)
----------------------------------------------------------

Phase = CONTRACTION           +30
Transition = EXPANSION→CONTRACTION +20
BuyZone = TRUE                +15
Confidence = 1.0              +20
GV ≥ 60                       +5
GY ≥ 3                        +5
GOI = NORMAL                  +10
Alignment                     +0
Penalty                       +0

TOTAL = 105

----------------------------------------------------------
DESIGN PRINCIPLES
----------------------------------------------------------

- Reward early cycle entries
- Reward inflection points
- Reward alignment across signals
- Penalize overheated environments
- Keep model interpretable and explainable

----------------------------------------------------------
IMPORTANT NOTE
----------------------------------------------------------

DO NOT modify scoring logic without:
1. Backtesting impact
2. Updating calibration buckets
3. Updating this documentation

This file is the source of truth for how prediction scores are calculated.
"""

import pandas as pd


# ----------------------------------------------------------
# Prediction scoring helpers
# ----------------------------------------------------------
def round_score_bucket(score: float) -> int:
    return int(round(score / 10.0) * 10)


def get_calibration_bucket(score_bucket: int) -> int | None:
    """
    Maps raw score buckets to historically calibrated buckets.
    """
    if score_bucket >= 125:
        return 130
    if score_bucket >= 105:
        return 110
    if score_bucket >= 85:
        return 90
    return None


def get_prediction_stats_for_bucket(
    calibration_bucket: int | None,
) -> tuple[float | None, float | None, str, str]:
    """
    Returns:
        expected_win_rate_trend,
        expected_return_trend,
        signal_tier,
        signal_horizon
    """

    calibration = {
        110: (0.729, 0.204, "Optimal", "Trend (120–180d)"),
        90:  (0.715, 0.198, "High Opportunity", "Trend (120–180d)"),
        130: (0.713, 0.181, "Over-Filtered", "Trend (120–180d)"),
    }

    if calibration_bucket in calibration:
        return calibration[calibration_bucket]

    return None, None, "Watchlist", "Monitor"


def build_setup_label(phase: str | None, goi_zone: str | None) -> str:
    phase_label = (phase or "Unknown").title()
    goi_map = {
        "EXTREME_OPPORTUNITY": "Extreme Opportunity",
        "ELEVATED_OPPORTUNITY": "Elevated Opportunity",
        "NORMAL": "Normal",
        "LOW_OPPORTUNITY": "Low Opportunity",
        "EXTREME_GREED": "Extreme Greed",
    }
    goi_label = goi_map.get(goi_zone, goi_zone or "Unknown")
    return f"{phase_label} + {goi_label}"


def calculate_prediction_score(row: pd.Series) -> pd.Series:
    phase = row.get("phase")
    prior_phase = row.get("prior_phase")
    buyzone_flag = row.get("buyzone_flag")
    confidence = float(row.get("confidence") or 0.0)
    gv = row.get("greer_value_score")
    gy = row.get("greer_yield_score")
    goi_zone = row.get("goi_zone")

    phase_score = (
        30 if phase == "CONTRACTION"
        else 25 if phase == "RECOVERY"
        else 20 if phase == "EUPHORIA"
        else 15 if phase == "EXPANSION"
        else 0
    )

    if prior_phase == "CONTRACTION" and phase == "EXPANSION":
        transition_score = 25
    elif prior_phase == "CONTRACTION" and phase == "RECOVERY":
        transition_score = 20
    elif prior_phase == "EXPANSION" and phase == "CONTRACTION":
        transition_score = 20
    else:
        transition_score = 0

    buyzone_score = 15 if buyzone_flag == True else 0
    confidence_score = confidence * 20

    fundamentals_score = 0
    if pd.notna(gv) and gv >= 60:
        fundamentals_score += 5
    if pd.notna(gy) and gy >= 3:
        fundamentals_score += 5

    goi_score = 0
    if goi_zone == "ELEVATED_OPPORTUNITY":
        goi_score = 30
    elif goi_zone == "EXTREME_OPPORTUNITY":
        goi_score = 20
    elif goi_zone == "NORMAL":
        goi_score = 10
    elif goi_zone == "LOW_OPPORTUNITY":
        goi_score = 5
    elif goi_zone == "EXTREME_GREED":
        goi_score = -15

    regime_alignment_score = 0
    if phase == "CONTRACTION" and goi_zone == "ELEVATED_OPPORTUNITY":
        regime_alignment_score = 30
    elif phase == "RECOVERY" and goi_zone == "ELEVATED_OPPORTUNITY":
        regime_alignment_score = 25
    elif phase == "CONTRACTION" and goi_zone == "EXTREME_OPPORTUNITY":
        regime_alignment_score = 20
    elif phase == "EUPHORIA" and goi_zone == "EXTREME_GREED":
        regime_alignment_score = -20

    overheat_penalty = 0
    if phase == "EUPHORIA" and goi_zone == "EXTREME_GREED":
        overheat_penalty = -30
    elif phase == "EXPANSION" and goi_zone == "EXTREME_GREED":
        overheat_penalty = -20
    elif phase == "EUPHORIA" and buyzone_flag == False:
        overheat_penalty = -10

    prediction_score = (
        phase_score
        + transition_score
        + buyzone_score
        + confidence_score
        + fundamentals_score
        + goi_score
        + regime_alignment_score
        + overheat_penalty
    )

    score_bucket = round_score_bucket(prediction_score)
    calibration_bucket = get_calibration_bucket(score_bucket)

    (
        expected_win_rate_trend,
        expected_return_trend,
        signal_tier,
        signal_horizon,
    ) = get_prediction_stats_for_bucket(calibration_bucket)

    setup_label = build_setup_label(phase, goi_zone)

    return pd.Series({
        "prediction_score": prediction_score,
        "score_bucket": score_bucket,
        "calibration_bucket": calibration_bucket,
        "expected_win_rate_trend": expected_win_rate_trend,
        "expected_return_trend": expected_return_trend,
        "expected_win_rate_60d": expected_win_rate_trend,
        "expected_return_60d": expected_return_trend,
        "signal_tier": signal_tier,
        "signal_horizon": signal_horizon,
        "setup_label": setup_label,
    })