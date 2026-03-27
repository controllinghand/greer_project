# 0_Dashboard-Guide.py
# ----------------------------------------------------------
# Dashboard Guide / Definitions Page
# Explains the purpose of each dashboard and how to read
# the Greer dashboard ecosystem.
# ----------------------------------------------------------

import streamlit as st

st.set_page_config(page_title="Dashboard Guide", layout="wide")

# ----------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------
def phase_card(color_box: str, title: str, body: str):
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(0,0,0,0.08);
            border-left: 8px solid {color_box};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: rgba(255,255,255,0.55);
            min-height: 200px;
        ">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 0.95rem; line-height: 1.55;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, text: str):
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: rgba(255,255,255,0.55);
            min-height: 145px;
        ">
            <div style="font-size: 1.0rem; font-weight: 700; margin-bottom: 0.45rem;">
                {title}
            </div>
            <div style="font-size: 0.95rem; line-height: 1.55;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# Page Title
# ----------------------------------------------------------
st.title("📘 Dashboard Guide")
st.caption(
    "A practical guide to the You Rock dashboard system — what each page does, how the signals connect, "
    "and how Rockers can use them together."
)

# ----------------------------------------------------------
# Intro
# ----------------------------------------------------------
st.markdown("""
Welcome to the **You Rock Dashboard Guide**.

The goal of the dashboard suite is simple:

- Turn a lot of market data into a few clear signals
- Show where the market is strong, weak, or attractive
- Help Rockers move from **macro context** to **sector focus** to **company-level action**

This is not just a collection of pages.  
It is a **top-down decision system**.
""")

st.divider()

# ----------------------------------------------------------
# System Map
# ----------------------------------------------------------
st.header("🧠 How the System Works")

st.markdown("""
The You Rock dashboards are designed to answer a sequence of questions:

**1. Is opportunity broad right now?**  
Use **🎯 Greer Opportunity Index (GOI)**

**2. What phase is the market in?**  
Use **🧭 Market Cycle**

**3. Where is money flowing?**  
Use **🏭 Sector Market Cycle**

**4. Which stocks are actionable?**  
Use **🏢 Company Market Cycle**

**5. Is market stress rising or falling?**  
Use **🎯 BuyZone Count**
""")

st.markdown("### The Decision Flow")
st.markdown("""
**GOI → Market Cycle → Sector Cycle → Company Cycle → Execution**
""")

flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)

with flow_col1:
    metric_card(
        "🎯 GOI",
        "Macro opportunity map. Shows how much of the market is currently in BuyZone."
    )

with flow_col2:
    metric_card(
        "🧭 Market Cycle",
        "Tells you the current regime: Recovery, Expansion, Euphoria, or Contraction."
    )

with flow_col3:
    metric_card(
        "🏭 Sector Cycle",
        "Shows which sectors are leading, lagging, recovering, or stretched."
    )

with flow_col4:
    metric_card(
        "🏢 Company Cycle",
        "Turns market context into stock-level action using Health, Direction, and Opportunity."
    )

with flow_col5:
    metric_card(
        "⚙️ Execution",
        "Use the system read to decide whether to accumulate, ride, trim, or stay selective."
    )

st.divider()

# ----------------------------------------------------------
# Quick Dashboard Index
# ----------------------------------------------------------
st.header("🧭 Dashboard Index")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cycle, Opportunity & Sentiment")
    st.markdown("""
- **🎯 Greer Opportunity Index (GOI)**  
  Measures how much opportunity exists across the market based on BuyZone breadth.

- **🧭 Market Cycle**  
  Identifies the current market phase and confidence level.

- **🏭 Sector Market Cycle**  
  Applies the cycle framework to sectors so you can see rotation and leadership.

- **🏢 Company Market Cycle**  
  Breaks down each company into Health, Direction, and Opportunity.

- **🧠 Sentiment Panel**  
  Provides a higher-level read on how the market feels internally.

- **📊 Fear & Greed**  
  A contrarian sentiment reference for identifying fear or excess optimism.
    """)

with col2:
    st.subheader("Breadth, Internals & Monitoring")
    st.markdown("""
- **🎯 BuyZone Count**  
  Tracks the number of companies in BuyZone and highlights rising or falling stress.

- **🚨 Bottom Detector**  
  Looks for signs that markets may be washed out and stabilizing.

- **🌐 Market Breadth**  
  Shows how broad participation is beneath the surface.

- **📋 Dashboard**  
  The full dashboard view with the main internal signals in one place.

- **🧩 Dashboard Mini**  
  A quick-look version of the full dashboard for fast monitoring.
    """)

st.divider()

# ----------------------------------------------------------
# Core Signal Building Blocks
# ----------------------------------------------------------
st.header("🧱 Core Signal Building Blocks")

b1, b2, b3, b4 = st.columns(4)

with b1:
    metric_card(
        "🟢 Health",
        "Measures fundamental strength. This is driven primarily by Greer Value strength and related quality support."
    )

with b2:
    metric_card(
        "📉 Direction",
        "Measures technical posture and market pressure. BuyZone and related trend signals help determine whether conditions are improving or weakening."
    )

with b3:
    metric_card(
        "💰 Opportunity",
        "Measures valuation attractiveness. Yield-based signals and fair-value signals help show whether a stock or market is cheap, fair, or stretched."
    )

with b4:
    metric_card(
        "🎯 Confidence",
        "Measures how strongly the signals agree. High confidence means the regime is clear. Low confidence means transition risk is higher."
    )

st.divider()

# ----------------------------------------------------------
# GOI Section
# ----------------------------------------------------------
st.header("🎯 Greer Opportunity Index (GOI)")

st.markdown("""
The **Greer Opportunity Index** is a market-wide opportunity gauge.

It measures the **percentage of tracked companies currently in BuyZone**.

### In plain English
- **High GOI** = opportunity is broad
- **Low GOI** = opportunity is limited
- **Extreme high readings** often happen during stress, panic, or washout periods
- **Extreme low readings** often happen when the market is crowded or overheated

### Why it matters
GOI helps answer the first and most important question:

**Is this a market where I should be hunting aggressively, or staying selective?**
""")

goi_col1, goi_col2 = st.columns(2)

with goi_col1:
    st.markdown("""
### Typical Read
- **Extreme Opportunity**  
  Broad panic-style setup. Historically rare and often attractive.

- **Elevated Opportunity**  
  Opportunity is building across the market.

- **Normal Range**  
  Typical environment. Stock selection matters most.

- **Low Opportunity**  
  Fewer attractive setups available.

- **Extreme Greed**  
  Very few names in BuyZone. Risk of crowding and overstretch is higher.
    """)

with goi_col2:
    st.info("""
Best used for:
- macro opportunity context
- sizing aggression vs selectivity
- recognizing rare market extremes
- complementing Market Cycle and Sector Cycle
""")

st.divider()

# ----------------------------------------------------------
# Market Cycle Section
# ----------------------------------------------------------
st.header("🔄 Greer Market Cycle")

st.markdown("""
The classic market psychology chart is useful, but too fuzzy for automation.

The You Rock system compresses the market into **4 machine-readable phases**:

**Recovery → Expansion → Euphoria → Contraction**

This allows the system to classify the market, sectors, and even companies in a consistent way.
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
The dashboards always assign the **closest major phase**:

**Recovery → Expansion → Euphoria → Contraction**

Then they attach a **confidence score**.

That means the system does not need a separate “transitional” bucket to be useful.
Instead, transition risk shows up through **lower confidence**.
""")

conf1, conf2, conf3 = st.columns(3)

with conf1:
    metric_card(
        "High Confidence",
        "Signals strongly agree. The current phase is clear and the regime is relatively stable."
    )

with conf2:
    metric_card(
        "Medium Confidence",
        "The current phase is still valid, but some signals are starting to disagree. Rotation or weakening may be starting."
    )

with conf3:
    metric_card(
        "Low Confidence",
        "Signals are mixed. The market may be between phases or close to shifting. Transition risk is highest here."
    )

st.info("""
Recommended approach:  
**Read the phase first, then use confidence to judge how stable or fragile that phase is.**
""")

st.divider()

# ----------------------------------------------------------
# How Rockers Should Use This
# ----------------------------------------------------------
st.header("🧠 How Rockers Should Use This")

st.markdown("""
Think of the system as a **decision funnel**.

### Step 1 — Check Opportunity
Use **GOI**  
Ask: **Is the market giving me broad opportunity?**

### Step 2 — Identify the Regime
Use **Market Cycle**  
Ask: **Are we early, trending, stretched, or weak?**

### Step 3 — Find the Best Hunting Grounds
Use **Sector Market Cycle**  
Ask: **Which sectors are leading or recovering?**

### Step 4 — Execute at the Company Level
Use **Company Market Cycle**  
Ask: **Which stocks have strong Health, good Direction, and real Opportunity?**

### Step 5 — Manage Risk
Use **BuyZone Count** and **Breadth tools**  
Ask: **Is stress building or fading under the surface?**
""")

st.markdown("""
### Example Flow

- **High GOI + Recovery** → Start building positions
- **Expansion + strong sectors** → Let winners run
- **Euphoria + low opportunity** → Trim, de-risk, harvest gains
- **Contraction + rising BuyZone Count** → Stay selective, wait for better setups
""")

st.divider()

# ----------------------------------------------------------
# Dashboard Definitions
# ----------------------------------------------------------
st.header("🗂️ Dashboard Definitions")

with st.expander("🎯 Greer Opportunity Index (GOI)"):
    st.markdown("""
Measures the percentage of tracked companies currently in BuyZone.

**Best for answering:**
- Is opportunity broad or limited?
- Are we closer to panic or complacency?
- How unusual is the current setup historically?

**Simple read:**
- High GOI = broad opportunity
- Low GOI = limited opportunity
    """)

with st.expander("🧭 Market Cycle"):
    st.markdown("""
Shows the overall market phase using the Greer framework.

**Best for answering:**
- Are we recovering?
- Are we in a healthy uptrend?
- Are we stretched?
- Are we under pressure?
- How confident is the current regime read?
""")

with st.expander("🏭 Sector Market Cycle"):
    st.markdown("""
Applies the market cycle logic to sectors.

**Best for answering:**
- Which sectors are leading?
- Which sectors are weak?
- Where is new opportunity forming?
- Which sectors have the highest transition risk?
""")

with st.expander("🏢 Company Market Cycle"):
    st.markdown("""
The most actionable cycle dashboard.

Each company is evaluated across:
- **Health** = fundamental quality
- **Direction** = trend + BuyZone posture + backdrop
- **Opportunity** = valuation attractiveness

**Best for answering:**
- Which stocks are actionable now?
- Which names have the best overall profile?
- Which companies are strongest inside the best sectors?
""")

with st.expander("🧠 Sentiment Panel"):
    st.markdown("""
Use this to gauge the emotional tone of the market.

Helpful for identifying:
- excessive optimism
- excessive fear
- whether crowd behavior is stretched
""")

with st.expander("📊 Fear & Greed"):
    st.markdown("""
A quick sentiment reference.

Best used as a contrarian signal, not a standalone buy/sell tool.

- Extreme fear can create opportunity
- Extreme greed can signal caution
""")

with st.expander("🎯 BuyZone Count"):
    st.markdown("""
Tracks the number of companies currently in BuyZone.

Think of it as:

**How many stocks are under pressure right now?**

**Best for answering:**
- Is market stress rising?
- Is stress broad or narrow?
- Are pullbacks spreading or improving?
""")

with st.expander("🚨 Bottom Detector"):
    st.markdown("""
Looks for conditions that often appear near washout lows.

Useful for:
- spotting possible turning points
- identifying when selling may be exhausted
- confirming that fear is reaching extremes
""")

with st.expander("🌐 Market Breadth"):
    st.markdown("""
Measures participation beneath the surface.

A healthy market usually has broad support.  
A weak market often narrows to fewer leaders.

Useful for:
- confirming trend quality
- spotting narrowing leadership
- checking internal participation
""")

with st.expander("📋 Dashboard"):
    st.markdown("""
The full-featured system dashboard.

Use this when you want the most complete market picture in one place.
It brings together the main internal signals and context metrics.
""")

with st.expander("🧩 Dashboard Mini"):
    st.markdown("""
A simplified quick-look version of the full dashboard.

Use this for a fast daily read when you want signal clarity without the full detail.
""")

st.divider()

# ----------------------------------------------------------
# Practical Reading Guide
# ----------------------------------------------------------
st.header("🛠️ Practical Reading Guide")

guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.markdown("""
### When the system is bullish
Typical profile:
- GOI is not overheated
- Market Cycle is in **Recovery** or **Expansion**
- Strong sectors are emerging
- Company Cycle leaders show strong Health + Direction

**Rockers:** build, add, and let leaders work
""")

with guide_col2:
    st.markdown("""
### When the system is defensive
Typical profile:
- GOI is low or stress is unstable
- Market Cycle is in **Euphoria** or **Contraction**
- BuyZone Count is rising fast
- Breadth is weakening
- Confidence is falling

**Rockers:** trim risk, be selective, wait for better asymmetry
""")

st.divider()

# ----------------------------------------------------------
# Color / Status Legend
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

st.caption(
    "These colors are used throughout the dashboard suite so Rockers can move between pages "
    "without relearning the system each time."
)

st.divider()

# ----------------------------------------------------------
# Final Note
# ----------------------------------------------------------
st.header("🎸 Final Takeaway")

st.markdown("""
The You Rock dashboard system is built to help Rockers answer three big questions:

### 1. Where is the market now?
Use **GOI**, **Market Cycle**, and **Sentiment**

### 2. Where should I focus?
Use **Sector Market Cycle**

### 3. What should I act on?
Use **Company Market Cycle** and supporting internals

The edge is not in one page by itself.  
The edge comes from using the pages **together**.
""")