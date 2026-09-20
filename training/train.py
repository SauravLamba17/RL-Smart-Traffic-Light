"""
Training pipeline and benchmark evaluation suite for Q-Learning traffic controller.
Supports Multi-Tier Emergency Vehicle Preemption (Ambulances & Fire Trucks).
"""
import os
import sys
import time
import argparse
import numpy as np

if sys.platform.startswith("win"):
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from typing import Dict, Any, Tuple
from environment.intersection_env import IntersectionEnv
from controllers.rl_agent import QLearningAgent
from controllers.fixed_timer_controller import FixedTimerController
from training.logger import MetricsLogger


def train_agent(
    episodes: int = 2500,
    steps_per_episode: int = 1500,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.998,
    model_save_path: str = "models/trained_q_table.pkl",
    log_filepath: str = "logs/training_log.csv",
    eval_interval: int = 200,
    seed: int = 42,
) -> Tuple[QLearningAgent, MetricsLogger]:
    """
    Trains the Q-Learning agent across multiple simulated traffic episodes
    with mixed civilian cars, ambulances (Tier 2), and fire trucks (Tier 1).
    """
    print("=" * 70)
    print("🚦 STARTING MULTI-TIER REINFORCEMENT LEARNING TRAINING")
    print(f"• Episodes: {episodes} | Steps per Episode: {steps_per_episode}")
    print(f"• Alpha (LR): {learning_rate} | Gamma (Discount): {discount_factor}")
    print(f"• Epsilon: {epsilon_start} -> {epsilon_min} (decay: {epsilon_decay})")
    print(f"• Target Model Path: {model_save_path}")
    print("=" * 70)

    env = IntersectionEnv(max_steps=steps_per_episode, seed=seed)
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    logger = MetricsLogger(log_filepath=log_filepath)

    start_time = time.time()

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep * 17
        if ep % 4 == 0:
            env.set_traffic_scenario("rush_hour")
        elif ep % 4 == 1:
            env.set_traffic_scenario("asymmetric_ns")
        elif ep % 4 == 2:
            env.set_traffic_scenario("asymmetric_ew")
        else:
            env.set_traffic_scenario("balanced")

        # Mix standard traffic with emergency drill episodes (dense emergency state training)
        if ep % 3 == 0:
            env.emergency_prob = 0.08  # High emergency density drill
        else:
            env.emergency_prob = 0.025  # Normal scenario

        state, info = env.reset(seed=ep_seed)
        episode_reward = 0.0
        td_errors = []

        while True:
            action = agent.get_action(state, env=env, training=True)
            next_state, reward, terminated, truncated, step_info = env.step(action)

            done = terminated or truncated
            td_err = agent.update(state, action, reward, next_state, done=done)
            td_errors.append(abs(td_err))

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        agent.episodes_trained = ep

        # Log metrics
        avg_wait = step_info['avg_wait_time']
        avg_amb_wait = step_info['avg_ambulance_wait_time']
        avg_fire_wait = step_info['avg_firetruck_wait_time']
        throughput = step_info['throughput_per_min']

        logger.log({
            'episode': ep,
            'reward': round(episode_reward, 2),
            'avg_wait_time': round(avg_wait, 2),
            'avg_ambulance_wait': round(avg_amb_wait, 2),
            'avg_firetruck_wait': round(avg_fire_wait, 2),
            'throughput_per_min': round(throughput, 2),
            'total_spawned': step_info['total_spawned'],
            'total_cleared': step_info['total_cleared'],
            'ambulance_cleared': step_info['ambulance_cleared'],
            'firetruck_cleared': step_info['firetruck_cleared'],
            'epsilon': round(agent.epsilon, 4),
            'mean_td_error': round(float(np.mean(td_errors)), 4) if td_errors else 0.0,
        })

        # Periodic status logging
        if ep % eval_interval == 0 or ep == episodes:
            summary = logger.get_summary(last_n=eval_interval)
            elapsed = time.time() - start_time
            print(
                f"[Ep {ep:4d}/{episodes}] "
                f"Reward: {summary.get('mean_reward', 0.0):7.1f} | "
                f"Avg Wait: {summary.get('mean_avg_wait_time', 0.0):5.1f}s | "
                f"Throughput: {summary.get('mean_throughput_per_min', 0.0):5.1f} veh/min | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Save trained model and logs
    agent.save_model(model_save_path)
    saved_log = logger.save_to_csv(log_filepath)
    total_duration = time.time() - start_time

    print("=" * 70)
    print(f"✅ TRAINING COMPLETE in {total_duration:.2f} seconds")
    print(f"💾 Model saved to: {model_save_path}")
    print(f"📊 Training log saved to: {saved_log}")
    print("=" * 70)

    return agent, logger


def evaluate_agent(
    model_path: str = "models/trained_q_table.pkl",
    episodes: int = 50,
    steps_per_episode: int = 1500,
    seed: int = 999,
    output_csv: str = "logs/eval_log.csv",
) -> Dict[str, Any]:
    """
    Rigorously benchmarks Fixed-Timer Controller vs Trained RL Agent
    with multi-tier emergency evaluation across synchronized random seeds.
    """
    print("=" * 70)
    print("🔬 RUNNING MULTI-TIER BENCHMARK EVALUATION (Fixed Timer vs RL Agent)")
    print(f"• Evaluation Episodes: {episodes} | Steps per Episode: {steps_per_episode}")
    print(f"• Loading Model: {model_path}")
    print("=" * 70)

    agent = QLearningAgent()
    loaded = agent.load_model(model_path)
    if not loaded:
        print(f"⚠️ Warning: Model {model_path} not found or incompatible. Training new model...")
        agent, _ = train_agent(episodes=2500, model_save_path=model_path)

    baseline_controller = FixedTimerController(phase_duration=360)
    eval_logger = MetricsLogger(log_filepath=output_csv)

    fixed_waits, rl_waits = [], []
    fixed_throughputs, rl_throughputs = [], []
    fixed_cleared, rl_cleared = [], []
    fixed_amb_waits, rl_amb_waits = [], []
    fixed_fire_waits, rl_fire_waits = [], []

    scenarios = ["balanced", "rush_hour", "asymmetric_ns", "asymmetric_ew"]

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep * 31
        scenario = scenarios[ep % len(scenarios)]

        # --- 1. Run Fixed Timer Baseline ---
        env_fixed = IntersectionEnv(max_steps=steps_per_episode, seed=ep_seed)
        env_fixed.set_traffic_scenario(scenario)
        state_f, _ = env_fixed.reset(seed=ep_seed)

        while True:
            action_f = baseline_controller.get_action(env_fixed, state_f)
            state_f, _, term_f, trunc_f, info_f = env_fixed.step(action_f)
            if term_f or trunc_f:
                break

        # --- 2. Run Trained RL Agent (Greedy exploitation) ---
        env_rl = IntersectionEnv(max_steps=steps_per_episode, seed=ep_seed)
        env_rl.set_traffic_scenario(scenario)
        state_rl, _ = env_rl.reset(seed=ep_seed)

        while True:
            action_rl = agent.get_action(state_rl, env=env_rl, training=False)
            state_rl, _, term_rl, trunc_rl, info_rl = env_rl.step(action_rl)
            if term_rl or trunc_rl:
                break

        # Record metrics
        f_wait = info_f['avg_wait_time']
        r_wait = info_rl['avg_wait_time']
        f_tp = info_f['throughput_per_min']
        r_tp = info_rl['throughput_per_min']
        f_clr = info_f['total_cleared']
        r_clr = info_rl['total_cleared']
        f_amb_w = info_f['avg_ambulance_wait_time']
        r_amb_w = info_rl['avg_ambulance_wait_time']
        f_fire_w = info_f['avg_firetruck_wait_time']
        r_fire_w = info_rl['avg_firetruck_wait_time']

        fixed_waits.append(f_wait)
        rl_waits.append(r_wait)
        fixed_throughputs.append(f_tp)
        rl_throughputs.append(r_tp)
        fixed_cleared.append(f_clr)
        rl_cleared.append(r_clr)

        if f_amb_w > 0:
            fixed_amb_waits.append(f_amb_w)
        if r_amb_w > 0:
            rl_amb_waits.append(r_amb_w)

        if f_fire_w > 0:
            fixed_fire_waits.append(f_fire_w)
        if r_fire_w > 0:
            rl_fire_waits.append(r_fire_w)

        eval_logger.log({
            'episode': ep,
            'scenario': scenario,
            'fixed_avg_wait': round(f_wait, 2),
            'rl_avg_wait': round(r_wait, 2),
            'wait_reduction_pct': round(((f_wait - r_wait) / max(0.001, f_wait)) * 100, 2),
            'fixed_throughput': round(f_tp, 2),
            'rl_throughput': round(r_tp, 2),
            'fixed_amb_wait': round(f_amb_w, 2),
            'rl_amb_wait': round(r_amb_w, 2),
            'fixed_fire_wait': round(f_fire_w, 2),
            'rl_fire_wait': round(r_fire_w, 2),
        })

    eval_logger.save_to_csv(output_csv)

    # Compute Statistics
    mean_fixed_wait = float(np.mean(fixed_waits))
    mean_rl_wait = float(np.mean(rl_waits))
    wait_reduction = ((mean_fixed_wait - mean_rl_wait) / max(0.001, mean_fixed_wait)) * 100

    mean_fixed_tp = float(np.mean(fixed_throughputs))
    mean_rl_tp = float(np.mean(rl_throughputs))
    tp_gain = ((mean_rl_tp - mean_fixed_tp) / max(0.001, mean_fixed_tp)) * 100

    mean_fixed_clr = float(np.mean(fixed_cleared))
    mean_rl_clr = float(np.mean(rl_cleared))

    mean_fixed_amb = float(np.mean(fixed_amb_waits)) if fixed_amb_waits else 0.0
    mean_rl_amb = float(np.mean(rl_amb_waits)) if rl_amb_waits else 0.0
    amb_gain = (((mean_fixed_amb - mean_rl_amb) / max(0.001, mean_fixed_amb)) * 100) if mean_fixed_amb > 0 else 0.0

    mean_fixed_fire = float(np.mean(fixed_fire_waits)) if fixed_fire_waits else 0.0
    mean_rl_fire = float(np.mean(rl_fire_waits)) if rl_fire_waits else 0.0
    fire_gain = (((mean_fixed_fire - mean_rl_fire) / max(0.001, mean_fixed_fire)) * 100) if mean_fixed_fire > 0 else 0.0

    print("\n" + "=" * 75)
    print("📊 MULTI-TIER BENCHMARK RESULTS (Fixed-Timer Baseline vs Q-Learning Agent)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Fixed Timer':<14} | {'RL Agent':<14} | {'Improvement':<12}")
    print("-" * 75)
    print(f"{'Average Vehicle Wait Time':<32} | {mean_fixed_wait:<14.2f}s| {mean_rl_wait:<14.2f}s| {wait_reduction:>+10.1f}%")
    print(f"{'Throughput (vehicles/min)':<32} | {mean_fixed_tp:<14.2f} | {mean_rl_tp:<14.2f} | {tp_gain:>+10.1f}%")
    print(f"{'Total Cleared Vehicles':<32} | {mean_fixed_clr:<14.1f} | {mean_rl_clr:<14.1f} | {((mean_rl_clr-mean_fixed_clr)/max(1,mean_fixed_clr)*100):>+10.1f}%")
    print(f"{'🚑 Ambulance Wait (Tier 2)':<32} | {mean_fixed_amb:<14.2f}s| {mean_rl_amb:<14.2f}s| {amb_gain:>+10.1f}%")
    print(f"{'🚒 Fire Truck Wait (Tier 1)':<32} | {mean_fixed_fire:<14.2f}s| {mean_rl_fire:<14.2f}s| {fire_gain:>+10.1f}%")
    print("=" * 75)

    return {
        'fixed_wait': mean_fixed_wait,
        'rl_wait': mean_rl_wait,
        'wait_reduction_pct': wait_reduction,
        'fixed_throughput': mean_fixed_tp,
        'rl_throughput': mean_rl_tp,
        'fixed_ambulance_wait': mean_fixed_amb,
        'rl_ambulance_wait': mean_rl_amb,
        'ambulance_gain_pct': amb_gain,
        'fixed_firetruck_wait': mean_fixed_fire,
        'rl_firetruck_wait': mean_rl_fire,
        'firetruck_gain_pct': fire_gain,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Multi-Tier RL Traffic Signal Agent")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=500, help="Steps per episode")
    parser.add_argument("--model", type=str, default="models/trained_q_table.pkl", help="Model file path")
    args = parser.parse_args()

    if args.train:
        train_agent(episodes=args.episodes, steps_per_episode=args.steps, model_save_path=args.model)
    elif args.eval:
        evaluate_agent(model_path=args.model, episodes=min(args.episodes, 50), steps_per_episode=args.steps)
    else:
        train_agent(episodes=args.episodes, steps_per_episode=args.steps, model_save_path=args.model)
        evaluate_agent(model_path=args.model)
