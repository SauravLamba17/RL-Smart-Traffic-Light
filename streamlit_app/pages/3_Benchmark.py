"""Page 3: Benchmark — Fixed Timer vs RL Agent comparison."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Benchmark | RL Traffic", page_icon="📊", layout="wide")

with st.sidebar:
    st.caption("Smart Traffic Light · RL Project")

st.title("📊 Benchmark: Fixed Timer vs RL Agent")
st.caption("50-episode headless evaluation — both controllers run identical traffic scenarios")

# ── Benchmark results (from last eval run) ───────────────────────────────────
# These match the evaluation output printed by main.py --eval
BENCHMARK = {
    "Metric":           ["Avg Wait Time (s)", "Throughput (veh/min)", "Ambulance Wait (s)", "Fire Truck Wait (s)"],
    "Fixed Timer":      [2.18,                 101.28,                 2.32,                  1.91],
    "RL Agent":         [1.57,                 105.41,                 0.43,                  0.61],
    "Improvement (%)":  [28.0,                  4.1,                   81.4,                  68.1],
}
df_bench = pd.DataFrame(BENCHMARK)

# ── KPI headline row ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wait Reduction",       "-28.0%",  delta="-0.61s per vehicle")
c2.metric("Throughput Gain",      "+4.1%",   delta="+4.1 veh/min")
c3.metric("Ambulance Clearance",  "-81.4%",  delta="-1.89s wait")
c4.metric("Fire Truck Clearance", "-68.1%",  delta="-1.30s wait")

st.divider()

tab1, tab2, tab3 = st.tabs(["🚗 Wait & Throughput", "🚑 Emergency Response", "📋 Full Table"])

with tab1:
    col_l, col_r = st.columns(2)
    with col_l:
        fig_wait = go.Figure()
        for controller, val, color in [("Fixed Timer", 2.18, "#f78166"), ("RL Agent", 1.57, "#3fb950")]:
            fig_wait.add_trace(go.Bar(name=controller, x=["Avg Wait Time"], y=[val],
                                      marker_color=color, text=[f"{val}s"], textposition="outside"))
        fig_wait.update_layout(title="Average Vehicle Wait Time",
                               yaxis_title="Seconds", template="plotly_dark",
                               barmode="group", height=380, yaxis_range=[0, 3])
        st.plotly_chart(fig_wait, use_container_width=True)

    with col_r:
        fig_tp = go.Figure()
        for controller, val, color in [("Fixed Timer", 101.28, "#f78166"), ("RL Agent", 105.41, "#3fb950")]:
            fig_tp.add_trace(go.Bar(name=controller, x=["Throughput"], y=[val],
                                    marker_color=color, text=[f"{val:.1f}"], textposition="outside"))
        fig_tp.update_layout(title="Vehicles Cleared per Minute",
                             yaxis_title="veh/min", template="plotly_dark",
                             barmode="group", height=380, yaxis_range=[95, 112])
        st.plotly_chart(fig_tp, use_container_width=True)

with tab2:
    fig_em = go.Figure()
    categories   = ["Ambulance Wait", "Fire Truck Wait"]
    fixed_vals   = [2.32, 1.91]
    rl_vals      = [0.43, 0.61]
    fig_em.add_trace(go.Bar(name="Fixed Timer", x=categories, y=fixed_vals,
                            marker_color="#f78166", text=[f"{v}s" for v in fixed_vals],
                            textposition="outside"))
    fig_em.add_trace(go.Bar(name="RL Agent", x=categories, y=rl_vals,
                            marker_color="#3fb950", text=[f"{v}s" for v in rl_vals],
                            textposition="outside"))
    fig_em.update_layout(title="Emergency Vehicle Response Time",
                         yaxis_title="Average Wait (seconds)",
                         template="plotly_dark", barmode="group", height=400,
                         yaxis_range=[0, 3.2])
    st.plotly_chart(fig_em, use_container_width=True)

    st.markdown("""
    **Why the RL agent is so much faster for emergency vehicles:**
    - 🚒 **Fire Truck (Tier 1)** — highest priority. RL agent detects fire truck presence in state
      and immediately preempts current phase, reducing yellow + clearance time to minimum.
    - 🚑 **Ambulance (Tier 2)** — second priority. RL agent shortens current green phase
      and switches to serve the ambulance within ~1 second.
    - Fixed Timer has **no emergency awareness** — it must wait for its fixed phase duration
      to complete before serving any vehicle, emergency or not.
    """)

with tab3:
    st.dataframe(df_bench, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Benchmark CSV",
                       df_bench.to_csv(index=False),
                       file_name="benchmark_results.csv", mime="text/csv")
