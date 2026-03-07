# market_cycle_utils.py
# ----------------------------------------------------------
# Shared market cycle helpers
# Centralizes phase classification, confidence scoring,
# badge rendering, and interpretation text so all dashboards
# use the same logic.
# ----------------------------------------------------------

import streamlit as st


# ----------------------------------------------------------
# Normalize a value into a 0.0 to 1.0 scale
# ----------------------------------------------------------
def normalize_pct(value: float) -> float:
    x = float(value)
    return max(0.0, min(100.0, x)) / 100.0


# ----------------------------------------------------------
# Convert raw metrics into weighted phase scores
# ----------------------------------------------------------
def get_phase_scores(health: float, buyzone: float, opp: float) -> dict:
    """
    health  = GV bullish %
    buyzone = BuyZone %
    opp     = Opportunity % (Yield + GFV bullish avg)

    Returns weighted scores for the 4 primary phases.
    Higher score wins.
    """

    h = normalize_pct(health)
    b = normalize_pct(buyzone)
    o = normalize_pct(opp)

    # Direction is the inverse of BuyZone pressure
    d = 1.0 - b

    scores = {
        # Recovery:
        # pullbacks are elevated, opportunity is attractive,
        # fundamentals are not broken
        "RECOVERY": (
            0.45 * o +
            0.35 * b +
            0.20 * h
        ),

        # Expansion:
        # strong fundamentals, strong direction, decent opportunity
        "EXPANSION": (
            0.45 * h +
            0.40 * d +
            0.15 * o
        ),

        # Euphoria:
        # strong fundamentals, strong direction, low opportunity
        "EUPHORIA": (
            0.45 * h +
            0.35 * d +
            0.25 * (1.0 - o)
        ),

        # Contraction:
        # weak fundamentals, elevated pullbacks, weak direction
        "CONTRACTION": (
            0.45 * (1.0 - h) +
            0.35 * b +
            0.20 * (1.0 - d)
        ),
    }

    return scores


# ----------------------------------------------------------
# Classify the phase and compute confidence
# ----------------------------------------------------------
def classify_phase_with_confidence(health: float, buyzone: float, opp: float) -> tuple[str, float]:
    """
    Returns:
        phase_name  : one of RECOVERY / EXPANSION / EUPHORIA / CONTRACTION
        confidence  : 0.0 to 1.0

    Confidence is based on how far ahead the winning phase is
    versus the second-place phase.
    """

    scores = get_phase_scores(health, buyzone, opp)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_phase, top_score = ranked[0]
    _, second_score = ranked[1]

    # Gap between top two drives confidence
    gap = max(0.0, top_score - second_score)

    # Slightly more forgiving scaling so confidence is more useful
    confidence = min(1.0, gap / 0.12)

    return top_phase, confidence


# ----------------------------------------------------------
# Convert numeric confidence into a label
# ----------------------------------------------------------
def phase_confidence_label(confidence: float) -> str:
    c = float(confidence)

    if c >= 0.67:
        return "High"
    if c >= 0.34:
        return "Medium"
    return "Low"


# ----------------------------------------------------------
# Return a note when signals are mixed
# ----------------------------------------------------------
def phase_confidence_note(confidence: float) -> str:
    label = phase_confidence_label(confidence)

    if label == "High":
        return "Signals are aligned."
    if label == "Medium":
        return "Some signals are mixed."
    return "Signals are mixed and the market may be transitioning."


# ----------------------------------------------------------
# Render a phase badge with confidence
# ----------------------------------------------------------
def phase_badge_inline(phase_name: str, confidence: float, label_prefix: str = "Market Cycle Phase"):
    p = str(phase_name).strip().upper()
    conf_label = phase_confidence_label(confidence)
    conf_pct = round(float(confidence) * 100)

    if p == "EXPANSION":
        st.success(f"🟢 {label_prefix}: **EXPANSION**  •  Confidence: **{conf_label} ({conf_pct}%)**")
    elif p == "EUPHORIA":
        st.warning(f"🟠 {label_prefix}: **EUPHORIA**  •  Confidence: **{conf_label} ({conf_pct}%)**")
    elif p == "RECOVERY":
        st.info(f"🔵 {label_prefix}: **RECOVERY**  •  Confidence: **{conf_label} ({conf_pct}%)**")
    elif p == "CONTRACTION":
        st.error(f"🔴 {label_prefix}: **CONTRACTION**  •  Confidence: **{conf_label} ({conf_pct}%)**")
    else:
        st.warning(f"🟡 {label_prefix}: **UNKNOWN**  •  Confidence: **{conf_label} ({conf_pct}%)**")


# ----------------------------------------------------------
# Return interpretation text for the current phase
# ----------------------------------------------------------
def phase_interpretation_text(phase_name: str, confidence: float) -> str:
    p = str(phase_name).strip().upper()
    note = phase_confidence_note(confidence)

    if p == "EXPANSION":
        return (
            "Strong fundamentals and strong market direction. "
            "The majority of companies are acting well and trends remain healthy. "
            f"{note}"
        )

    if p == "EUPHORIA":
        return (
            "Fundamentals and direction remain strong, but valuation opportunity is fading. "
            "This is often a later-cycle environment where upside may still exist, "
            "but risk of overextension is higher. "
            f"{note}"
        )

    if p == "RECOVERY":
        return (
            "Valuations are attractive and pullbacks are still common, "
            "but fundamentals are not broken. "
            "This phase often appears after corrections when conditions begin improving. "
            f"{note}"
        )

    if p == "CONTRACTION":
        return (
            "Fundamentals are weakening and pullback pressure is elevated. "
            "This phase reflects broad deterioration and a more defensive environment. "
            f"{note}"
        )

    return f"Signals are mixed. {note}"


# ----------------------------------------------------------
# Optional debug helper for transparency
# ----------------------------------------------------------
def get_ranked_phase_scores(health: float, buyzone: float, opp: float) -> list[tuple[str, float]]:
    scores = get_phase_scores(health, buyzone, opp)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)