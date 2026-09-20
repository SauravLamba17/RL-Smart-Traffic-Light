"""
Page 2: RL Training Analysis
Shows reward convergence, wait time improvement, epsilon decay over 2500 episodes.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Training | RL Traffic", page_icon="🏋️", layout="wide")

with st.sidebar:
    st.caption("Smart Traffic Light · RL Project")

st.title("🏋️ RL Training Analysis")
st.caption("Q-Learning Agent — 2,500 Episodes × 1,500 Steps")

LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "training_log.csv")

if not os.path.exists(LOG_PATH):
    st.warning("Training log not found. Run `python main.py --train` first.")
    st.stop()

df = pd.read_csv(LOG_PATH)
df.columns = [c.strip() for c in df.columns]

# ── Use exact column names from the CSV ──────────────────────────────────────
# Columns: episode, reward, avg_wait_time, avg_ambulance_wait, avg_firetruck_wait,
#          throughput_per_min, total_spawned, total_cleared, ambulance_cleared,
#          firetruck_cleared, epsilon, mean_td_error
ep_col         = "episode"          if "episode"          in df.columns else df.columns[0]
reward_col     = "reward"           if "reward"           in df.columns else None
wait_col       = "avg_wait_time"    if "avg_wait_time"    in df.columns else None
throughput_col = "throughput_per_min" if "throughput_per_min" in df.columns else None
eps_col        = "epsilon"          if "epsilon"          in df.columns else None

# Rolling smoothing window (~5% of dataset)
window = max(1, len(df) // 20)
if reward_col:
    df["_reward_smooth"] = df[reward_col].rolling(window, min_periods=1).mean()
if wait_col:
    df["_wait_smooth"] = df[wait_col].rolling(window, min_periods=1).mean()

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
if reward_col:
    c1.metric("Final Reward (avg)",   f"{df['_reward_smooth'].iloc[-1]:.1f}")
    c2.metric("Best Reward",          f"{df[reward_col].max():.1f}")
if wait_col:
    c3.metric("Final Avg Wait",       f"{df['_wait_smooth'].iloc[-1]:.2f}s")
if throughput_col:
    c4.metric("Final Throughput",     f"{df[throughput_col].iloc[-1]:.1f} veh/min")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Reward Convergence",
    "⏱️ Wait Time Progress",
    "🚑 Emergency Wait",
    "🔍 Epsilon Decay"
])

with tab1:
    if reward_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[ep_col], y=df[reward_col],
            mode="lines", name="Episode Reward",
            line=dict(color="#4a90d9", width=1), opacity=0.35))
        fig.add_trace(go.Scatter(
            x=df[ep_col], y=df["_reward_smooth"],
            mode="lines", name=f"Smoothed ({window}-ep avg)",
            line=dict(color="#3fb950", width=2.5)))
        fig.update_layout(
            title="Cumulative Reward Convergence",
            xaxis_title="Episode", yaxis_title="Reward",
            template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Reward increases as the agent learns to minimize wait times and prioritize emergency vehicles.")

with tab2:
    if wait_col:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df[ep_col], y=df[wait_col],
            mode="lines", name="Episode Avg Wait",
            line=dict(color="#f78166", width=1), opacity=0.35))
        fig2.add_trace(go.Scatter(
            x=df[ep_col], y=df["_wait_smooth"],
            mode="lines", name=f"Smoothed ({window}-ep avg)",
            line=dict(color="#58a6ff", width=2.5)))
        fig2.update_layout(
            title="Average Vehicle Wait Time During Training",
            xaxis_title="Episode", yaxis_title="Wait Time (s)",
            template="plotly_dark", height=420)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    amb_col  = "avg_ambulance_wait" if "avg_ambulance_wait"  in df.columns else None
    fire_col = "avg_firetruck_wait" if "avg_firetruck_wait"  in df.columns else None
    if amb_col or fire_col:
        fig3 = go.Figure()
        if amb_col:
            fig3.add_trace(go.Scatter(
                x=df[ep_col],
                y=df[amb_col].rolling(window, min_periods=1).mean(),
                mode="lines", name="Ambulance Wait (smoothed)",
                line=dict(color="#50beff", width=2)))
        if fire_col:
            fig3.add_trace(go.Scatter(
                x=df[ep_col],
                y=df[fire_col].rolling(window, min_periods=1).mean(),
                mode="lines", name="Fire Truck Wait (smoothed)",
                line=dict(color="#ff4b4b", width=2)))
        fig3.update_layout(
            title="Emergency Vehicle Wait Time During Training",
            xaxis_title="Episode", yaxis_title="Wait Time (s)",
            template="plotly_dark", height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Both ambulance and fire truck wait times decrease as the agent learns priority preemption.")
    else:
        st.info("Emergency wait columns not found in training log.")

with tab4:
    if eps_col:
        fig4 = px.line(
            df, x=ep_col, y=eps_col,
            title="Epsilon Decay (Exploration → Exploitation)",
            labels={ep_col: "Episode", eps_col: "Epsilon"},
            template="plotly_dark",
            color_discrete_sequence=["#d2a520"])
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)
        st.info("Epsilon starts at 1.0 (full exploration) and decays to 0.05 — "
                "the agent gradually stops making random decisions.")

st.divider()
with st.expander("📋 Raw Training Log (first 50 rows)"):
    st.dataframe(df.drop(columns=["_reward_smooth","_wait_smooth"],
                         errors="ignore").head(50),
                 use_container_width=True)
