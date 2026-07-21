#!/usr/bin/env python3
"""
Quick run script for fast simulation testing.

Usage:
    python quick_run.py                    # Run 1 episode with default agent
    python quick_run.py --episodes 5       # Run 5 episodes
    python quick_run.py --plot             # Run and show time-space diagram
    python quick_run.py --save-data        # Save experiment data to CSV
"""

import argparse
from typing import Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

from config import build_simulation_elements
from simulator.simulator import Simulator
from simulator.trajectory import plot_time_space_diagram, save_experiment_data


def quick_run(
    episodes: int = 1,
    plot: bool = False,
    save_data: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True
):
    """
    Run simulation quickly without checkpointing or wandb logging.

    Args:
        episodes: Number of episodes to run
        plot: Whether to show time-space diagram after simulation
        save_data: Whether to save experiment data to CSV files
        output_dir: Directory for saving data (default: experiment_data/<agent_name>)
        verbose: Whether to print episode details

    Returns:
        Dict of averaged metrics
    """
    # Build simulation elements from config.yaml
    blueprint, agent, run_config, _ = build_simulation_elements(config_path="config_quick_run.yaml")

    if verbose:
        print(f"Environment: {run_config.get('env_name', 'unknown')}")
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Episodes: {episodes}")
        print(f"Duration: {run_config['episode_duration']}s")
        print("-" * 50)

    print(run_config)

    # Collect metrics across episodes
    all_metrics: Dict[str, list] = defaultdict(list)
    last_simulator = None

    for ep in range(episodes):
        # Create fresh simulator for each episode
        simulator = Simulator(blueprint, agent, run_config)
        stop_bus_hold_action: Dict[Tuple[str, str, str], float] = {}

        # Main simulation loop
        for t in range(run_config['episode_duration']):
            snapshot = simulator.step(t, stop_bus_hold_action)
            stop_bus_hold_action = agent.calculate_hold_time(snapshot)
            snapshot.record_holding_time(stop_bus_hold_action)

        # Get metrics
        metrics, _ = simulator.get_metrics()
        for name, value in metrics.items():
            all_metrics[name].append(value)

        if verbose:
            print(f"Episode {ep}: {metrics}")

        # Reset agent for next episode
        agent.reset(ep)
        last_simulator = simulator

    # Calculate averages
    avg_metrics = {name: np.mean(values) for name, values in all_metrics.items()}

    if verbose:
        print("-" * 50)
        print("Average metrics:")
        for name, value in avg_metrics.items():
            print(f"  {name}: {value:.4f}")

    # Plot time-space diagram if requested (from last episode)
    if plot and last_simulator is not None:
        print("\nGenerating time-space diagram...")
        try:
            agent_name = agent.__class__.__name__
        except:
            agent_name = "agent"

        # Get metrics from last episode for display
        last_metrics, _ = last_simulator.get_metrics()

        plot_time_space_diagram(
            last_simulator.total_buses,
            save_path=f"time_space_diagram_{agent_name}.png",
            title=f"Time-Space Diagram - {agent_name}",
            metrics=last_metrics,
            time_unit='minutes',
            distance_unit='kilometers',
            show_stats=True,
            show_legend=True,
            left_paxs=last_simulator.left_paxs
        )

    # Save experiment data if requested
    if save_data and last_simulator is not None:
        try:
            agent_name = agent.__class__.__name__
        except:
            agent_name = "agent"

        # Set output directory
        if output_dir is None:
            output_dir = f"experiment_data/{agent_name}"

        last_metrics, _ = last_simulator.get_metrics()

        save_experiment_data(
            buses=last_simulator.total_buses,
            output_dir=output_dir,
            prefix=f"ep{episodes-1}_",
            metrics=last_metrics,
            config=run_config,
            left_paxs=last_simulator.left_paxs,
            rejection_events=last_simulator.rejection_events
        )

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Quick simulation run')
    parser.add_argument('--episodes', '-e', type=int, default=1,
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Show time-space diagram after simulation')
    parser.add_argument('--save-data', '-s', action='store_true',
                        help='Save experiment data to CSV files')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory for saving data (default: experiment_data/<agent>)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    quick_run(
        episodes=args.episodes,
        plot=args.plot,
        save_data=args.save_data,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
