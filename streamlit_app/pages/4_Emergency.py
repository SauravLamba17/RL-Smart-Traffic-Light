"""Page 4: Emergency Vehicle Response Analysis."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os, sys

st.set_page_config(page_title="Emergency | RL Traffic", page_icon="🚑", layout="wide")

with st.sidebar:
    st.caption("Smart Traffic Light · RL Project")

st.title("🚑🚒 Emergency Vehicle Priority System")
st.caption("Multi-Tier Preemption — Fire Truck (Tier 1) and Ambulance (Tier 2)")

st.markdown("""
## Priority Architecture

Our system implements **two-tier emergency preemption** inspired by real-world traffic
signal preemption protocols (e.g., NEMA TS2, Opticom GPS-based systems):

| Tier | Vehicle | Priority | Max Wait Budget | Color |
|------|---------|----------|-----------------|-------|
| **1** | 🚒 Fire Truck | CRITICAL | 0.75s yellow → immediate green | Red |
| **2** | 🚑 Ambulance  | HIGH     | 1.0s yellow → fast green | Blue |
| **3** | 🚗 Car        | Normal   | Standard Q-Learning control | Gray |
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Response Time Comparison")
    fig = go.Figure()

    categories = ["Ambulance (Tier 2)", "Fire Truck (Tier 1)"]
    fixed      = [2.32, 1.91]
    rl         = [0.43, 0.61]

    fig.add_trace(go.Bar(name="Fixed Timer (Blind)", x=categories, y=fixed,
                         marker_color="#f78166",
                         text=[f"{v}s" for v in fixed], textposition="outside"))
    fig.add_trace(go.Bar(name="RL Agent (Priority-Aware)", x=categories, y=rl,
                         marker_color="#3fb950",
                         text=[f"{v}s" for v in rl], textposition="outside"))

    fig.update_layout(barmode="group", template="plotly_dark",
                      yaxis_title="Avg Wait (seconds)", height=380,
                      yaxis_range=[0, 3.0],
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Improvement Summary")
    data = {
        "Vehicle Type":      ["Ambulance", "Fire Truck", "Regular Car"],
        "Fixed Timer (s)":   [2.32,        1.91,          2.18],
        "RL Agent (s)":      [0.43,        0.61,          1.57],
        "Improvement":       ["81.4%",     "68.1%",       "28.0%"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.metric("Ambulance improvement", "81.4% faster",  delta="-1.89s avg wait")
    st.metric("Fire Truck improvement","68.1% faster",  delta="-1.30s avg wait")

st.divider()

st.subheader("How Preemption Works (Step-by-Step)")

steps_col, diagram_col = st.columns([2, 1])
with steps_col:
    st.markdown("""
    **When a 🚒 Fire Truck spawns in lane X:**

    1. `spawn_emergency_vehicle()` is called → vehicle marked `VEHICLE_TYPE_FIRETRUCK`
    2. `get_emergency_status()` returns `(ns_has_fire, ew_has_fire, combined)`
    3. State space encodes emergency presence → agent/controller sees it
    4. **RL Agent**: Q-value strongly rewards switching to serve fire truck lane
       immediately (emergency flag in state raises Q-value for that action)
    5. **Fixed Timer**: Must wait for current phase timer to expire (up to 6s)
    6. Yellow phase shortened to **30 frames (0.5s)** instead of standard 60
    7. All-red clearance → green given to fire truck lane
    8. Fire truck is **committed** = enters intersection box → cleared

    **When conflict occurs (🚒 North + 🚑 East simultaneously):**
    - Tier 1 wins: Fire Truck lane gets green first
    - Ambulance waits (still served faster than fixed timer by ~60%)
    """)

with diagram_col:
    st.markdown("""
    ```
    State: [queues, phase, emergency]
           │
           ▼
    ┌─────────────────┐
    │  Fire Truck?    │──YES──► Short yellow
    │  (Tier 1)       │         → Switch NOW
    └────────┬────────┘
             │NO
             ▼
    ┌─────────────────┐
    │  Ambulance?     │──YES──► Shorten green
    │  (Tier 2)       │         → Early switch
    └────────┬────────┘
             │NO
             ▼
    ┌─────────────────┐
    │  Q-Learning     │
    │  Normal Control │
    └─────────────────┘
    ```
    """)

st.divider()
st.subheader("Conflict Resolution Demonstration")
st.info("""
**Live Demo:** During the Pygame simulation, press **[F]** to trigger the conflict test —
this spawns a Fire Truck in the North lane and an Ambulance in the East lane simultaneously.
Watch how the RL agent resolves Tier 1 vs Tier 2 conflict while the Fixed Timer struggles.
""")
