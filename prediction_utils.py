# prediction_utils.py

import pandas as pd


# ----------------------------------------------------------
# Prediction scoring helpers
# ----------------------------------------------------------
def round_score_bucket(score: float) -> int:
    return int(round(score / 10.0) * 10)


def get_calibration_bucket(score_bucket: int) -> int | None:
    """
    Maps raw score buckets to the historically calibrated buckets
    that actually have strong backtest meaning.
    """
    if score_bucket >= 125:
        return 130
    if score_bucket >= 105:
        return 110
    if score_bucket >= 85:
        return 90
    return None


def get_prediction_stats_for_bucket(calibration_bucket: int | None) -> tuple[float | None, float | None, str]:
    calibration = {
        90:  (0.734, 0.113, "High Conviction"),
        110: (0.681, 0.083, "Strong"),
        130: (0.646, 0.068, "Constructive"),
    }

    if calibration_bucket in calibration:
        return calibration[calibration_bucket]

    return None, None, "Watchlist"


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

    phase_score = 30 if phase == "CONTRACTION" else 25 if phase == "RECOVERY" else 20 if phase == "EUPHORIA" else 15 if phase == "EXPANSION" else 0

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
    expected_win_rate_60d, expected_return_60d, signal_tier = get_prediction_stats_for_bucket(calibration_bucket)
    setup_label = build_setup_label(phase, goi_zone)

    return pd.Series({
        "prediction_score": prediction_score,
        "score_bucket": score_bucket,
        "calibration_bucket": calibration_bucket,
        "expected_win_rate_60d": expected_win_rate_60d,
        "expected_return_60d": expected_return_60d,
        "signal_tier": signal_tier,
        "setup_label": setup_label,
    })