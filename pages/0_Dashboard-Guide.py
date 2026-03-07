# 0_Dashboard-Guide.py
# ----------------------------------------------------------
# Dashboard Guide / Definitions Page
# Explains the purpose of each dashboard and how to read
# the Greer Market Cycle phases.
# ----------------------------------------------------------

import streamlit as st

st.set_page_config(page_title="Dashboard Guide", layout="wide")

# ----------------------------------------------------------
# Page Title
# ----------------------------------------------------------
st.title("📘 Dashboard Guide")
st.caption("A quick reference for understanding the You Rock dashboards and what each signal means.")

# ----------------------------------------------------------
# Intro
# ----------------------------------------------------------
st.markdown("""
Welcome to the **You Rock Dashboard Guide**.

This page explains the purpose of each dashboard, how to interpret the main signals,
and how the **Greer Market Cycle** phases fit together.

The goal is simple:

- Turn a lot of data into a few clear signals
- Help Rockers understand what the market is doing
- Make the dashboards easier to act on
""")

st.divider()

# ----------------------------------------------------------
# Quick Dashboard Index
# ----------------------------------------------------------
st.header("🧭 Dashboard Index")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment & Cycle")
    st.markdown("""
- **🧠 Sentiment Panel**  
  Shows the emotional tone of the market. Useful for seeing whether investors are fearful, confident, or stretched.

- **🧭 Market Cycle**  
  Tracks the overall market phase using Greer-style signals such as health, buyzones, and opportunity.

- **🏙️ Sector Market Cycle**  
  Applies the same cycle logic to individual sectors so you can see which parts of the market are recovering, expanding, euphoric, or contracting.

- **📊 Fear & Greed**  
  Measures whether the market is leaning too fearful or too greedy. Often useful as a contrarian reference.
    """)

with col2:
    st.subheader("Breadth & Opportunity")
    st.markdown("""
- **🚨 Bottom Detector**  
  Looks for signs that selling may be washed out and conditions are improving.

- **📈 Market Breadth**  
  Shows how many stocks are participating in the move. Healthy rallies usually have broad participation.

- **📋 Dashboard**  
  Full dashboard view with the main indicators and market internals in one place.

- **🧩 Dashboard Mini**  
  A simplified version of the dashboard for quick monitoring.
    """)

st.divider()

# ----------------------------------------------------------
# Market Cycle Section
# ----------------------------------------------------------
st.header("🔄 Greer Market Cycle")

st.markdown("""
The classic psychology chart has many emotional stages, but for automation that gets too messy.

The You Rock system compresses the market into **4 machine-readable phases**:

**Recovery → Expansion → Euphoria → Contraction**

This makes it easier to classify sectors and dashboards consistently.
""")

phase_col1, phase_col2, phase_col3, phase_col4 = st.columns(4)

with phase_col1:
    st.info("🔵 **Recovery**")
    st.markdown("""
**Meaning:** Market stabilizing after weakness  
**Traits:** Pullbacks common, opportunity improving  
**Rockers:** **Start accumulating**
""")

with phase_col2:
    st.success("🟢 **Expansion**")
    st.markdown("""
**Meaning:** Healthy uptrend  
**Traits:** Strong health, trend working  
**Rockers:** **Ride trends**
""")

with phase_col3:
    st.warning("🟠 **Euphoria**")
    st.markdown("""
**Meaning:** Late-stage strength  
**Traits:** Strong trend, stretched valuations  
**Rockers:** **Take profits / sell covered calls**
""")

with phase_col4:
    st.error("🔴 **Contraction**")
    st.markdown("""
**Meaning:** Weak market / bear pressure  
**Traits:** Weak health, elevated stress  
**Rockers:** **Look for BuyZones**
""")

st.markdown("### Cycle Flow")
st.markdown("""
**Recovery → Expansion → Euphoria → Contraction → Recovery**
""")

st.divider()

# ----------------------------------------------------------
# Phase Definitions Table
# ----------------------------------------------------------
st.header("📖 Phase Definitions")

st.markdown("""
These phases are based on three main ingredients, and **confidence** measures how strongly the signals agree:

- **Health** = GV bullish %
- **BuyZone** = BuyZone %
- **Opportunity** = Yield + GFV bullish average
""")



st.markdown("""
| Phase | Health | BuyZone | Opportunity | Interpretation | Typical Action |
|---|---:|---:|---:|---|---|
| 🔵 **Recovery** | Medium to Strong | High | High | Pullbacks are present, but value is improving | Start accumulating |
| 🟢 **Expansion** | Strong | Low to Moderate | Moderate | Healthy uptrend with good participation | Ride trends |
| 🟠 **Euphoria** | Strong | Very Low | Low | Market strong but becoming expensive | Take profits / sell covered calls |
| 🔴 **Contraction** | Weak | High | Mixed | Weak internals and pressure increasing | Look for BuyZones / stay selective |
""")

st.divider()

# ----------------------------------------------------------
# Confidence Explanation
# ----------------------------------------------------------
st.header("🎯 About Confidence")

st.markdown("""
The dashboards now use **4 primary phases**:

**Recovery → Expansion → Euphoria → Contraction**

Instead of showing a separate **Transitional** phase, the system always assigns the
**closest main phase** and then shows a **confidence score**.

In plain English:

- **High confidence** = the signals strongly agree on the current phase
- **Medium confidence** = the phase is still valid, but some signals are mixed
- **Low confidence** = the market may be between phases or close to shifting
""")

st.markdown("""
### How to interpret confidence

- **High confidence**
  - phase is clear
  - signals are aligned
  - regime is stable

- **Medium confidence**
  - phase is likely correct
  - some indicators are starting to disagree
  - worth monitoring for change

- **Low confidence**
  - signals are mixed
  - the market may be transitioning
  - highest chance of a phase shift
""")

st.markdown("""
### Confidence Guide

| Confidence | Meaning | What It Suggests |
|---|---|---|
| **High** | Signals strongly agree | Regime is stable |
| **Medium** | Some signals are mixed | Watch for rotation or weakening |
| **Low** | Signals are conflicted | Possible phase shift / transition risk |
""")

st.info("""
Recommended approach:
Focus on the **main phase first**, then use **confidence** to judge how stable or fragile
that phase is.
""")

st.divider()

# ----------------------------------------------------------
# How to Read the Dashboards
# ----------------------------------------------------------
st.header("🧠 How Rockers Should Use This")

st.markdown("""
Think of the dashboards as a layered system:

1. **Sentiment** tells you how people feel  
2. **Breadth** tells you how many stocks are participating  
3. **Cycle** tells you where the market probably is  
4. **Opportunity** tells you whether prices are attractive  

That means:

- A sector in **Recovery** may be early and worth building into
- A sector in **Expansion** may be one to keep riding
- A sector in **Euphoria** may still be strong, but risk is higher
- A sector in **Contraction** may need patience until BuyZones improve
""")

st.divider()

# ----------------------------------------------------------
# Dashboard-by-Dashboard Help
# ----------------------------------------------------------
st.header("🗂️ Dashboard Definitions")

with st.expander("🧠 Sentiment Panel"):
    st.markdown("""
Use this to gauge the emotional mood of the market.

Helpful for identifying:
- excessive optimism
- excessive fear
- whether crowd behavior is stretched
""")

with st.expander("🧭 Market Cycle"):
    st.markdown("""
Shows the overall market phase using the Greer framework.

Best for answering:
- Are we recovering?
- Are we in a healthy uptrend?
- Are we stretched?
- Are we breaking down?
- How confident is the current regime read?
""")

with st.expander("🏙️ Sector Market Cycle"):
    st.markdown("""
Applies market cycle logic to sectors.

Best for answering:
- Which sectors are leading?
- Which sectors are weak?
- Where is new opportunity forming?
- Which sectors have the highest transition risk?
""")

with st.expander("📊 Fear & Greed"):
    st.markdown("""
A quick sentiment check.

Best used as a reference, not a standalone buy/sell tool.
Extreme fear can create opportunity.
Extreme greed can signal caution.
""")

with st.expander("🚨 Bottom Detector"):
    st.markdown("""
Looks for conditions that often show up near washout lows.

Useful for:
- spotting possible turning points
- identifying markets where selling may be exhausted
""")

with st.expander("📈 Market Breadth"):
    st.markdown("""
Measures participation under the surface.

A strong market usually has broad support.
A weak market often narrows to fewer leaders.
""")

with st.expander("📋 Dashboard"):
    st.markdown("""
The full-featured view that combines multiple indicators in one place.
Use this when you want the most complete market picture.
""")

with st.expander("🧩 Dashboard Mini"):
    st.markdown("""
A simplified quick-look version of the full dashboard.
Use this when you want a fast read without all the detail.
""")

st.divider()

# ----------------------------------------------------------
# Simple Legend
# ----------------------------------------------------------
st.header("🎨 Color Legend")

legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)

with legend_col1:
    st.info("🔵 Recovery")
with legend_col2:
    st.success("🟢 Expansion")
with legend_col3:
    st.warning("🟠 Euphoria")
with legend_col4:
    st.error("🔴 Contraction")

st.caption("These colors are used throughout the dashboard suite for consistent interpretation.")