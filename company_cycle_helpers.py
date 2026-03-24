# ----------------------------------------------------------
# company_cycle_helpers.py
# Shared helper functions for Greer Company Cycle scoring
# ----------------------------------------------------------

import pandas as pd


# ----------------------------------------------------------
# Clamp a numeric value into a range
# ----------------------------------------------------------
def clamp(value, min_value: float = 0.0, max_value: float = 100.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = min_value

    return max(min_value, min(value, max_value))


# ----------------------------------------------------------
# Safely round a numeric value
# ----------------------------------------------------------
def safe_round(value, digits: int = 2):
    try:
        if pd.isnull(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------
# Convert Greer Value + above_50_count into Health %
# ----------------------------------------------------------
def compute_health_pct(gv_score, above_50_count) -> float:
    gv = float(gv_score) if pd.notnull(gv_score) else 0.0
    a50 = float(above_50_count) if pd.notnull(above_50_count) else 0.0
    a50_pct = (a50 / 6.0) * 100.0

    score = (gv * 0.85) + (a50_pct * 0.15)
    return round(clamp(score), 1)


# ----------------------------------------------------------
# Convert buyzone flag into a direction component
# Lower value means more pullback pressure
# Higher value means better trend direction
# ----------------------------------------------------------
def compute_buyzone_score(buyzone_flag) -> float:
    return 25.0 if bool(buyzone_flag) else 75.0


# ----------------------------------------------------------
# Convert FVG direction into a direction component
# ----------------------------------------------------------
def compute_fvg_score(fvg_last_direction) -> float:
    s = str(fvg_last_direction).strip().lower() if pd.notnull(fvg_last_direction) else ""

    if s in ["bullish", "up", "green"]:
        return 100.0
    if s in ["bearish", "down", "red"]:
        return 0.0
    return 50.0


# ----------------------------------------------------------
# Build Direction % from buyzone, FVG, and sector direction
# ----------------------------------------------------------
def compute_direction_pct(buyzone_flag, fvg_last_direction, sector_direction_pct) -> float:
    buyzone_score = compute_buyzone_score(buyzone_flag)
    fvg_score = compute_fvg_score(fvg_last_direction)
    sector_score = float(sector_direction_pct) if pd.notnull(sector_direction_pct) else 50.0

    score = (
        (buyzone_score * 0.35) +
        (fvg_score * 0.35) +
        (sector_score * 0.30)
    )

    return round(clamp(score), 1)


# ----------------------------------------------------------
# Convert GFV status into an opportunity component
# ----------------------------------------------------------
def compute_gfv_score(gfv_status) -> float:
    s = str(gfv_status).strip().lower() if pd.notnull(gfv_status) else "gray"

    if s == "gold":
        return 100.0
    if s == "green":
        return 75.0
    if s == "gray":
        return 50.0
    return 0.0


# ----------------------------------------------------------
# Build Opportunity % from Yield Score and GFV status
# ----------------------------------------------------------
def compute_opportunity_pct(greer_yield_score, gfv_status) -> float:
    ys = float(greer_yield_score) if pd.notnull(greer_yield_score) else 0.0
    ys_pct = (ys / 4.0) * 100.0
    gfv_pct = compute_gfv_score(gfv_status)

    score = (ys_pct * 0.5) + (gfv_pct * 0.5)
    return round(clamp(score), 1)


# ----------------------------------------------------------
# Build Greer Company Index from three pillars
# ----------------------------------------------------------
def compute_company_index(health_pct, direction_pct, opportunity_pct) -> float:
    score = (
        float(health_pct) +
        float(direction_pct) +
        float(opportunity_pct)
    ) / 3.0

    return round(clamp(score), 2)


# ----------------------------------------------------------
# Existing phase classifier expects "buyzone pressure"
# not direction. Convert direction back into inverse pressure.
# ----------------------------------------------------------
def compute_company_buyzone_proxy(direction_pct: float) -> float:
    return round(clamp(100.0 - float(direction_pct)), 2)


# ----------------------------------------------------------
# Convert confidence score into a readable label
# ----------------------------------------------------------
def transition_risk_label(confidence) -> str:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "Unknown"

    if c >= 0.85:
        return ""
    if c >= 0.60:
        return "👀 Watch"
    if c >= 0.30:
        return "⚠ Possible Phase Shift"
    return "🔁 High Transition Risk"


# ----------------------------------------------------------
# Compute all company cycle fields for a single row
# Returns a dict so pages can merge easily
# ----------------------------------------------------------
def build_company_cycle_metrics(
    gv_score,
    above_50_count,
    buyzone_flag,
    fvg_last_direction,
    sector_direction_pct,
    greer_yield_score,
    gfv_status,
    classify_phase_with_confidence,
) -> dict:
    health_pct = compute_health_pct(gv_score, above_50_count)

    direction_pct = compute_direction_pct(
        buyzone_flag,
        fvg_last_direction,
        sector_direction_pct,
    )

    opportunity_pct = compute_opportunity_pct(
        greer_yield_score,
        gfv_status,
    )

    greer_company_index = compute_company_index(
        health_pct,
        direction_pct,
        opportunity_pct,
    )

    company_buyzone_proxy = compute_company_buyzone_proxy(direction_pct)

    phase, confidence = classify_phase_with_confidence(
        health_pct,
        company_buyzone_proxy,
        opportunity_pct,
    )

    return {
        "health_pct": health_pct,
        "direction_pct": direction_pct,
        "opportunity_pct": opportunity_pct,
        "greer_company_index": greer_company_index,
        "phase": phase,
        "confidence": round(float(confidence), 4),
        "transition_risk": transition_risk_label(confidence),
    }


# ----------------------------------------------------------
# Enrich a dataframe with company cycle fields
# Expects the dataframe to contain the needed source columns
# ----------------------------------------------------------
def enrich_company_cycle_dataframe(df: pd.DataFrame, classify_phase_with_confidence) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows = []

    for _, row in df.iterrows():
        metrics = build_company_cycle_metrics(
            gv_score=row.get("greer_value_score"),
            above_50_count=row.get("above_50_count"),
            buyzone_flag=row.get("buyzone_flag"),
            fvg_last_direction=row.get("fvg_last_direction"),
            sector_direction_pct=row.get("sector_direction_pct"),
            greer_yield_score=row.get("greer_yield_score"),
            gfv_status=row.get("gfv_status"),
            classify_phase_with_confidence=classify_phase_with_confidence,
        )
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.copy(), metrics_df], axis=1)