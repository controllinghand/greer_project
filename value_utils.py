# ----------------------------------------------------------
# value_utils.py
# ----------------------------------------------------------
# Shared helpers for Greer Value Level UI mapping
# ----------------------------------------------------------

import pandas as pd


# ----------------------------------------------------------
# Map greer_star_rating to Value Level
# - Keeps DB compatibility while UI uses Value Levels
# ----------------------------------------------------------
def get_value_level(value) -> int:
    if pd.isnull(value):
        return 0

    try:
        s = int(value)
    except Exception:
        return 0

    if s >= 3:
        return 3
    if s == 2:
        return 2
    if s == 1:
        return 1
    return 0


# ----------------------------------------------------------
# Return full Value Level label for UI
# ----------------------------------------------------------
def value_level_label(level: int) -> str:
    if level == 3:
        return "🔴 $$$ Critical"
    if level == 2:
        return "🟡 $$ Elevated"
    if level == 1:
        return "🟢 $ Normal"
    return "—"


# ----------------------------------------------------------
# Short label (used across UI)
# ----------------------------------------------------------
def value_level_short(level: int) -> str:
    if level == 3:
        return "Level 3"
    if level == 2:
        return "Level 2"
    if level == 1:
        return "Level 1"
    return "No Level"
    
# ----------------------------------------------------------
# Return short Value Level label
# ----------------------------------------------------------
def value_level_short(level: int) -> str:
    if level == 3:
        return "Level 3"
    if level == 2:
        return "Level 2"
    if level == 1:
        return "Level 1"
    return "No Level"


# ----------------------------------------------------------
# Convenience helper:
# go directly from greer_star_rating -> full UI label
# ----------------------------------------------------------
def value_signal_from_stars(value) -> str:
    return value_level_label(get_value_level(value))


# ----------------------------------------------------------
# Optional helper for color styling
# ----------------------------------------------------------
def value_level_color(level: int) -> str:
    if level == 3:
        return "#F44336"
    if level == 2:
        return "#FFA726"
    if level == 1:
        return "#66BB6A"
    return "#9E9E9E"


# ----------------------------------------------------------
# Optional helper for sorting/filtering display names
# ----------------------------------------------------------
def value_level_name(level: int) -> str:
    if level == 3:
        return "Critical"
    if level == 2:
        return "Elevated"
    if level == 1:
        return "Normal"
    return "None"


# ----------------------------------------------------------
# Return both level + label (common pattern)
# ----------------------------------------------------------
def value_level_and_label(value) -> tuple[int, str]:
    level = get_value_level(value)
    return level, value_level_label(level)