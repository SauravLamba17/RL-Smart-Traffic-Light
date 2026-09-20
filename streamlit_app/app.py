"""
Smart Traffic Light Controller — Streamlit Analytics Dashboard
Main entry point. Run with:
  streamlit run streamlit_app/app.py
"""
import streamlit as st
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Smart Traffic Light — RL Project",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-light.png", width=72)
    st.title("Smart Traffic Light")
    st.caption("Reinforcement Learning Project")
    st.divider()
    st.markdown("**Navigate using the pages above ↑**")
    st.divider()
    st.markdown("**Open Source Project**")
    st.caption("Tabular Q-Learning · Multi-Tier Priority")

# ── Home page ────────────────────────────────────────────────────────────────
st.title("🚦 Smart Traffic Light Controller")
st.subheader("Using Reinforcement Learning (Q-Learning)")

col1, col2, col3 = st.columns(3)
col1.metric("RL Algorithm",    "Tabular Q-Learning")
col2.metric("Wait Reduction",  "~28%",  delta="vs Fixed Timer")
col3.metric("Emergency Gains", "~75%",  delta="Faster clearance")

st.divider()

st.markdown("""
## 🎯 Project Overview

Traditional traffic lights operate on **fixed timer cycles** — they switch phases every N seconds
regardless of actual traffic conditions. This wastes time when one direction is empty and congests
the other.

Our project replaces the fixed timer with a **Reinforcement Learning (Q-Learning) agent** that
observes the intersection state (queue lengths, emergency vehicles, current phase) and learns
to make smarter switching decisions that **minimize vehicle wait time** while **prioritizing
ambulances and fire trucks**.

---

## 🏗️ System Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│  Intersection Env   │◄──────►│  Q-Learning RL Agent     │
│  (4-lane simulator) │  state │  (13,824 state table)    │
│  • Vehicle physics  │        │  • ε-greedy exploration  │
│  • Emergency veh.   │ reward │  • Temporal-difference   │
│  • Collision safety │        │    learning (α=0.1, γ=0.9)│
└─────────────────────┘        └──────────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Fixed-Timer         │  ← Baseline for comparison
│ Controller          │
└─────────────────────┘
```

---

## 📊 Navigate the Dashboard

| Page | What you'll find |
|------|-----------------|
| 🏋️ **Training** | Reward convergence, wait time improvement over 2,500 episodes |
| 📊 **Benchmark** | Fixed Timer vs RL Agent — head-to-head comparison |
| 🚑 **Emergency** | Ambulance & Fire Truck priority response analysis |
| 📂 **Dataset** | Kaggle dataset EDA and simulation calibration proof |
""")

st.info("👈 Use the sidebar or the pages listed above to explore the full analysis.")
