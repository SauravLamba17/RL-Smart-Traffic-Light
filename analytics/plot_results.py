"""
Generates high-resolution analytics plots for training convergence and benchmark comparisons.
Supports Multi-Tier Emergency Vehicles (Ambulances & Fire Trucks).
"""
import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

if sys.platform.startswith("win"):
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from typing import Dict, List, Any


def read_csv_log(filepath: str) -> List[Dict[str, float]]:
    """Reads CSV log into list of float-valued dictionaries."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            records.append(parsed)
    return records


def moving_average(data: np.ndarray, window_size: int = 40) -> np.ndarray:
    """Computes moving average over 1D array."""
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def plot_reward_convergence(
    train_records: List[Dict[str, Any]],
    output_path: str = "analytics/plots/reward_convergence.png"
) -> None:
    """Plots training cumulative reward convergence curve."""
    if not train_records:
        return

    episodes = np.array([r['episode'] for r in train_records])
    rewards = np.array([r['reward'] for r in train_records])
    window = min(50, len(rewards) // 4) if len(rewards) > 10 else 1

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    # Raw scatter points with alpha
    ax.scatter(episodes, rewards, color='#58a6ff', alpha=0.18, s=10, label='Episode Reward')

    # Rolling average trend
    if len(rewards) >= window and window > 1:
        ma = moving_average(rewards, window)
        ma_x = episodes[window - 1:]
        ax.plot(ma_x, ma, color='#1f6feb', linewidth=2.5, label=f'Moving Avg (window={window})')

    ax.set_title("Multi-Tier Q-Learning Agent — Training Reward Convergence", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved plot: {output_path}")


def plot_wait_time_comparison(
    train_records: List[Dict[str, Any]],
    output_path: str = "analytics/plots/wait_time_comparison.png"
) -> None:
    """Plots average vehicle waiting time progression during training."""
    if not train_records:
        return

    episodes = np.array([r['episode'] for r in train_records])
    waits = np.array([r['avg_wait_time'] for r in train_records])
    window = min(50, len(waits) // 4) if len(waits) > 10 else 1

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(episodes, waits, color='#f85149', alpha=0.2, s=10, label='Raw Avg Wait Time')

    if len(waits) >= window and window > 1:
        ma = moving_average(waits, window)
        ma_x = episodes[window - 1:]
        ax.plot(ma_x, ma, color='#da3633', linewidth=2.5, label=f'Trend (window={window})')

    ax.set_title("Average Vehicle Waiting Time Over Training (Multi-Tier Simulation)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Mean Wait Time per Vehicle (seconds / timesteps)", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved plot: {output_path}")


def plot_scenario_benchmarks(
    eval_records: List[Dict[str, Any]],
    output_path: str = "analytics/plots/throughput_comparison.png"
) -> None:
    """Plots bar chart comparing throughput across traffic scenarios."""
    if not eval_records:
        return

    scenarios = {}
    for r in eval_records:
        sc = r.get('scenario', 'default')
        if sc not in scenarios:
            scenarios[sc] = {
                'f_wait': [], 'r_wait': [],
                'f_tp': [], 'r_tp': []
            }
        scenarios[sc]['f_wait'].append(r.get('fixed_avg_wait', 0.0))
        scenarios[sc]['r_wait'].append(r.get('rl_avg_wait', 0.0))
        scenarios[sc]['f_tp'].append(r.get('fixed_throughput', 0.0))
        scenarios[sc]['r_tp'].append(r.get('rl_throughput', 0.0))

    sc_names = list(scenarios.keys())
    f_tp_means = [np.mean(scenarios[s]['f_tp']) for s in sc_names]
    r_tp_means = [np.mean(scenarios[s]['r_tp']) for s in sc_names]

    x = np.arange(len(sc_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width/2, f_tp_means, width, label='Fixed Timer (Baseline)', color='#8b949e')
    rects2 = ax.bar(x + width/2, r_tp_means, width, label='RL Agent (Multi-Tier Trained)', color='#2ea043')

    ax.set_title("Intersection Throughput Across Traffic Scenarios", fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel("Vehicles Cleared / Minute", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in sc_names], fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved plot: {output_path}")


def plot_summary_dashboard(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    output_path: str = "analytics/plots/summary_benchmark.png"
) -> None:
    """Generates a comprehensive 4-panel executive summary dashboard figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Training Reward
    if train_records:
        eps = [r['episode'] for r in train_records]
        rews = [r['reward'] for r in train_records]
        ax = axes[0, 0]
        ax.scatter(eps, rews, color='#58a6ff', alpha=0.15, s=8)
        if len(rews) > 20:
            ma = moving_average(np.array(rews), 30)
            ax.plot(eps[29:], ma, color='#0969da', linewidth=2, label='Moving Avg (30)')
        ax.set_title("1. Multi-Tier Training Reward Progression", fontweight='bold', fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(loc='lower right')

    # Panel 2: Wait Time Reduction
    if eval_records:
        ax = axes[0, 1]
        f_w = np.mean([r['fixed_avg_wait'] for r in eval_records])
        r_w = np.mean([r['rl_avg_wait'] for r in eval_records])
        bars = ax.bar(['Fixed Timer', 'RL Agent'], [f_w, r_w], color=['#cf222e', '#2da44e'], width=0.5)
        ax.set_title(f"2. Overall Vehicle Wait Time ({((f_w-r_w)/max(0.01,f_w)*100):.1f}% Reduction)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Seconds / Timesteps")
        for b in bars:
            h = b.get_height()
            ax.annotate(f'{h:.1f}s', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # Panel 3: Throughput Gain
    if eval_records:
        ax = axes[1, 0]
        f_tp = np.mean([r['fixed_throughput'] for r in eval_records])
        r_tp = np.mean([r['rl_throughput'] for r in eval_records])
        bars = ax.bar(['Fixed Timer', 'RL Agent'], [f_tp, r_tp], color=['#8c959f', '#1f883d'], width=0.5)
        ax.set_title(f"3. Intersection Throughput", fontweight='bold', fontsize=12)
        ax.set_ylabel("Vehicles Cleared / Min")
        for b in bars:
            h = b.get_height()
            ax.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # Panel 4: Emergency Response (Ambulance vs Fire Truck)
    if eval_records:
        ax = axes[1, 1]
        f_amb = np.mean([r['fixed_amb_wait'] for r in eval_records if r.get('fixed_amb_wait', 0) > 0]) if any(r.get('fixed_amb_wait', 0) > 0 for r in eval_records) else 0.0
        r_amb = np.mean([r['rl_amb_wait'] for r in eval_records if r.get('rl_amb_wait', 0) > 0]) if any(r.get('rl_amb_wait', 0) > 0 for r in eval_records) else 0.0
        f_fire = np.mean([r['fixed_fire_wait'] for r in eval_records if r.get('fixed_fire_wait', 0) > 0]) if any(r.get('fixed_fire_wait', 0) > 0 for r in eval_records) else 0.0
        r_fire = np.mean([r['rl_fire_wait'] for r in eval_records if r.get('rl_fire_wait', 0) > 0]) if any(r.get('rl_fire_wait', 0) > 0 for r in eval_records) else 0.0

        x = np.arange(2)
        width = 0.35
        rects1 = ax.bar(x - width/2, [f_amb, f_fire], width, label='Fixed Timer', color='#cf222e')
        rects2 = ax.bar(x + width/2, [r_amb, r_fire], width, label='RL Agent (Multi-Tier)', color='#2da44e')

        ax.set_title("4. Multi-Tier Emergency Response Latency", fontweight='bold', fontsize=12)
        ax.set_ylabel("Average Wait Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(['Ambulance (Tier 2)', 'Fire Truck (Tier 1)'], fontweight='bold')
        ax.legend()

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    fig.suptitle("Smart Traffic Light Controller — Multi-Tier Priority Benchmark Summary", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved summary dashboard: {output_path}")


def generate_all_plots(
    train_log: str = "logs/training_log.csv",
    eval_log: str = "logs/eval_log.csv",
    output_dir: str = "analytics/plots"
) -> None:
    """Generates all project analytics plots from CSV logs."""
    train_records = read_csv_log(train_log)
    eval_records = read_csv_log(eval_log)

    plot_reward_convergence(train_records, output_path=os.path.join(output_dir, "reward_convergence.png"))
    plot_wait_time_comparison(train_records, output_path=os.path.join(output_dir, "wait_time_comparison.png"))
    plot_scenario_benchmarks(eval_records, output_path=os.path.join(output_dir, "throughput_comparison.png"))
    plot_summary_dashboard(train_records, eval_records, output_path=os.path.join(output_dir, "summary_benchmark.png"))
    print(f"✅ All analytics charts successfully generated in: {output_dir}")


if __name__ == "__main__":
    generate_all_plots()
