"""Page 1: Project Overview & Methodology."""
import streamlit as st

st.set_page_config(page_title="Overview | RL Traffic", page_icon="🏗️", layout="wide")

with st.sidebar:
    st.caption("Smart Traffic Light · RL Project")

st.title("🏗️ Project Overview & Methodology")

st.markdown("""
## Problem Statement

Urban traffic intersections managed by **fixed-timer controllers** waste green time when
lanes are empty and cause unnecessary congestion when one direction is heavily loaded.
Emergency vehicles (ambulances, fire trucks) have no priority — they wait in queue like
regular cars, losing critical response time.

## Our Solution

A **Tabular Q-Learning agent** that:
1. Observes the 4-lane intersection state (queue sizes, current phase, emergency presence)
2. Decides **when to switch** the traffic signal phase
3. Learns to minimize average vehicle wait time over 2,500 training episodes
4. Prioritizes Tier 1 (Fire Trucks) and Tier 2 (Ambulances) with preemption logic
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("RL Formulation")
    st.markdown("""
    | Component | Definition |
    |-----------|-----------|
    | **State S** | `(q_ns, q_ew, phase, emergency_ns, emergency_ew)` discretized to 13,824 states |
    | **Actions A** | `{0 = Keep current phase, 1 = Switch phase}` |
    | **Reward R** | `+10` per cleared car, `+100` per cleared ambulance, `+200` per cleared fire truck; penalties for waiting, phase switches, imbalance |
    | **Algorithm** | Tabular Q-Learning (ε-greedy, α=0.1, γ=0.9) |
    | **Training** | 2,500 episodes × 1,500 steps = 3.75M total steps |
    """)

with col2:
    st.subheader("Environment Design")
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Lanes | 4 (N, S, E, W) |
    | Simulation FPS | 60 |
    | Min green time | 150 frames (2.5s) |
    | Yellow duration | 60 frames (1.0s) |
    | Arrival rate (normal) | 0.025/step ≈ 1.5 veh/min/lane |
    | Emergency probability | 2.5% of spawns |
    | Collision avoidance | All-red clearance phase |
    | Vehicle physics | Car-following model with braking |
    """)

st.divider()
st.subheader("Training Pipeline")
st.markdown("""
```
Data Collection (Kaggle)
       │
       ▼
Calibrate arrival_rates & timing parameters
       │
       ▼
Train Q-Learning Agent (2,500 episodes)
       │  epsilon: 1.0 → 0.05  (exploration → exploitation)
       ▼
Save Q-table to models/trained_q_table.pkl
       │
       ▼
Benchmark Evaluation (50 episodes)
Fixed Timer vs RL Agent — same seed, same traffic
       │
       ▼
Live Pygame Simulation + Streamlit Dashboard
```
""")

st.divider()
st.subheader("Innovation Highlights")

c1, c2, c3 = st.columns(3)
c1.info("**Multi-Tier Priority**\n\nTwo distinct emergency tiers with different preemption budgets — Fire Truck overrides Ambulance in conflict scenarios.")
c2.info("**Collision-Safe Design**\n\nAll-red clearance phase waits for committed vehicles to exit the intersection before granting cross-traffic green.")
c3.info("**Real Data Calibration**\n\nSimulation parameters derived from Kaggle traffic datasets — not arbitrary values. RL improvement matches published real-world studies (~35% peak hours).")
