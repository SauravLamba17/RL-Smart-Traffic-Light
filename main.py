"""
Main entry point for the Smart Traffic Light Controller RL Project.
Unified CLI for Training, Headless Evaluation, Analytics Plotting, and Live Simulation.

Project: Smart Traffic Light Controller Using Reinforcement Learning
"""
import sys
import os
import argparse

# Enable UTF-8 encoding for Windows standard output
if sys.platform.startswith("win"):
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train import train_agent, evaluate_agent
from analytics.plot_results import generate_all_plots
from dashboard.comparison_view import run_comparison_dashboard
from dashboard.launch_screen import run_launch_screen


def main():
    parser = argparse.ArgumentParser(
        description="Smart Traffic Light Controller — Reinforcement Learning (Q-Learning) System"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Launch the interactive side-by-side simulation (Fixed Timer vs RL Agent)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the Tabular Q-Learning agent and save model"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run statistical headless evaluation benchmark against the fixed-timer baseline"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate high-resolution analytics plots (Reward, Wait Time, Throughput, Summary)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute complete pipeline: Train -> Evaluate -> Generate Plots -> Launch Simulation"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2500,
        help="Number of training episodes (default: 2500)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of simulation steps per episode (default: 1500)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained_q_table.pkl",
        help="Path to save or load the trained Q-table model (default: models/trained_q_table.pkl)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=["balanced", "rush_hour", "asymmetric_ns", "asymmetric_ew", "light"],
        help="Skip launch screen and jump directly to a traffic scenario"
    )
    parser.add_argument(
        "--no-intro",
        action="store_true",
        help="Skip the animated launch/intro screen and use balanced scenario"
    )

    args = parser.parse_args()

    # Default to simulate if no flags provided
    if not (args.simulate or args.train or args.eval or args.plot or args.all):
        if not os.path.exists(args.model):
            print("🚀 First-time run detected: No trained model found.")
            print("Running quick training pipeline to initialize model and analytics...")
            args.all = True
        else:
            args.simulate = True

    if args.all:
        print("\n🚀 EXECUTING COMPLETE PIPELINE")
        train_agent(episodes=args.episodes, steps_per_episode=args.steps, model_save_path=args.model)
        evaluate_agent(model_path=args.model, episodes=50, steps_per_episode=args.steps)
        generate_all_plots()
        print("\n🎮 Launching Live Side-by-Side Simulation...")
        run_comparison_dashboard(model_path=args.model)

    elif args.train:
        train_agent(episodes=args.episodes, steps_per_episode=args.steps, model_save_path=args.model)

    elif args.eval:
        evaluate_agent(model_path=args.model, episodes=50, steps_per_episode=args.steps)

    elif args.plot:
        generate_all_plots()

    elif args.simulate:
        # If --scene or --no-intro passed, skip the intro screen
        if args.scene or getattr(args, 'no_intro', False):
            scenario = args.scene or "balanced"
        else:
            result = run_launch_screen()
            if result is None:
                return          # User closed the intro window
            scenario, _ = result
        run_comparison_dashboard(model_path=args.model, initial_scenario=scenario)


if __name__ == "__main__":
    main()
