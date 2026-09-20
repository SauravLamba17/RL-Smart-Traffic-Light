# 🚦 Smart Traffic Light Controller Using Reinforcement Learning

> Adaptive traffic signal control using **Tabular Q-Learning** with **multi-tier emergency vehicle priority preemption** (Fire Trucks + Ambulances). Built with Python, Pygame, and Streamlit. Calibrated against real-world Kaggle traffic datasets.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pygame](https://img.shields.io/badge/Pygame-CE_2.5-orange.svg)](https://pyga.me)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)

---

## 📖 About

Urban intersections managed by fixed-timer controllers waste green time on empty lanes and force emergency vehicles to wait like regular cars. This project replaces the fixed timer with a **Q-Learning RL agent** that observes queue lengths, detects emergency vehicles, and learns optimal signal switching to:

- **Reduce average vehicle wait time by ~28%** vs fixed timer baseline
- **Clear ambulances 81% faster** (Tier 2 priority)
- **Clear fire trucks 68% faster** (Tier 1 — highest priority)
- **Handle conflicting emergencies** (fire truck on N-S vs ambulance on E-W simultaneously)

## 🎯 Benchmark Results

| Metric | Fixed Timer | RL Agent | Improvement |
|:-------|:----------:|:--------:|:-----------:|
| Avg Wait Time | 2.18s | 1.57s | **-28.0%** |
| Throughput | 101.3 veh/min | 105.4 veh/min | **+4.1%** |
| 🚑 Ambulance Wait | 2.32s | 0.43s | **-81.4%** |
| 🚒 Fire Truck Wait | 1.91s | 0.61s | **-68.1%** |

*Evaluated over 50 episodes with synchronized random seeds.*

---

## 🏗️ Architecture

```
RL Based SmartTrafficLight/
├── environment/
│   ├── intersection_env.py    # RL gym environment (state, reward, step)
│   ├── vehicle.py             # Vehicle physics & car-following model
│   └── renderer.py            # Pygame intersection renderer
├── controllers/
│   ├── rl_agent.py            # Tabular Q-Learning agent (13,824 states × 2 actions)
│   └── fixed_timer_controller.py  # Baseline fixed-timer controller
├── training/
│   ├── train.py               # Training loop (2,500 episodes × 1,500 steps)
│   └── logger.py              # CSV metrics logger
├── dashboard/
│   ├── launch_screen.py       # Animated intro/scenario selector screen
│   └── comparison_view.py     # Side-by-side live simulation dashboard
├── streamlit_app/
│   ├── app.py                 # Web analytics dashboard (Streamlit)
│   └── pages/                 # Training, Benchmark, Emergency, Dataset pages
├── analytics/
│   └── plot_results.py        # Matplotlib benchmark charts
├── data/
│   ├── kaggle_analysis.py     # Kaggle dataset calibration script
│   └── calibration_report.csv # Real-world vs simulation comparison
├── models/
│   └── trained_q_table.pkl    # Pre-trained Q-table (portable)
├── run.py                     # One-command launcher (Pygame + Streamlit)
├── main.py                    # CLI entry point
└── START_SIMULATION.bat       # Windows double-click launcher
```

---

## 🧠 RL Formulation

### State Space (13,824 states)
```
S = (queue_NS, queue_EW, phase, emergency_combined, timer_bucket)
```
- **Queue buckets**: 4 levels per direction (0, 1-2, 3-5, 6+)
- **Phase**: NS-Green or EW-Green
- **Emergency**: 9-value encoding of (NS_priority × EW_priority) — detects conflicting emergencies
- **Timer**: 3 buckets (fresh / mid / long-green)

### Action Space
- `0` = Keep current phase
- `1` = Switch phase (triggers yellow → all-red clearance → new green)

### Reward Function
```
R = +10·(cars_cleared) + 100·(ambulances_cleared) + 200·(firetrucks_cleared)
    - 0.05·(cars_waiting) - 1.5·(amb_waiting) - 3.0·(fire_waiting)
    - 1.0·(switch_penalty) - 0.05·|queue_imbalance|
```

### Training: Tabular Q-Learning
- α = 0.1, γ = 0.9, ε: 1.0 → 0.05 (decay 0.998)
- 2,500 episodes × 1,500 steps = 3.75M total steps (~2.5 minutes)

---

## 🚀 Quick Start

### One Command (recommended)
```bash
python run.py
```
This starts **both** the Streamlit analytics dashboard (browser) and the Pygame simulation (desktop window) simultaneously.

### Or individually
```bash
# Pygame simulation with animated intro
python main.py --simulate

# Streamlit analytics dashboard
streamlit run streamlit_app/app.py

# Train the agent from scratch
python main.py --train

# Run benchmark evaluation
python main.py --eval

# Generate analytics plots
python main.py --plot

# Full pipeline (train → eval → plot → simulate)
python main.py --all
```

### Windows: Double-click
Just run **`START_SIMULATION.bat`**

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/RL-Smart-Traffic-Light.git
cd RL-Smart-Traffic-Light
pip install -r requirements.txt
python run.py
```

### Requirements
- Python 3.10+
- pygame-ce >= 2.5.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- streamlit >= 1.30.0
- pandas >= 2.0.0
- plotly >= 5.18.0

---

## 🎮 Keyboard Controls (Pygame Simulation)

| Key | Action |
|:---:|:-------|
| `1-4` | Dispatch 🚑 Ambulance to N/S/E/W lane |
| `5-8` | Dispatch 🚒 Fire Truck to N/S/E/W lane |
| `F` | 🔥 Conflict test: Fire Truck (N) + Ambulance (E) |
| `SPACE` | Pause / Resume |
| `↑ / ↓` | Speed: 1x / 2x / 4x / 8x |
| `T` | Cycle traffic scenarios |
| `R` | Reset with new random seed |
| `ESC` | Exit |

---

## 📊 Dataset Sources

| Dataset | Source | Used For |
|---------|--------|----------|
| Urban Traffic Light Control | [Kaggle](https://www.kaggle.com/datasets/thedevastator/urban-traffic-light-control-dataset) | Queue lengths, phase data, RL state validation |
| Traffic System Dataset | [Kaggle](https://www.kaggle.com/datasets/thedevastator/traffic-system-dataset) | Arrival rates, wait times, simulation calibration |

Our simulation parameters are calibrated against these datasets — the RL improvement (~35% at peak hours) matches published real-world studies.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
