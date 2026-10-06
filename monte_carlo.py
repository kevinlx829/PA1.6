import argparse
import csv
import math
import os
import time
from statistics import mean
from typing import Dict, List

import numpy as np

from scenarios.runner import simulate_scenario
from utils.config import load_config


def calculate_metrics(log: Dict[str, List[float]], setpoint: float, deadband: float) -> Dict[str, float]:
    temperatures = log["T_true"]
    heater_states = log["heater"]
    lower = setpoint - deadband / 2.0
    upper = setpoint + deadband / 2.0
    errors = [temperature - setpoint for temperature in temperatures]

    return {
        "rmse_C": math.sqrt(mean(error * error for error in errors)),
        "outside_deadband_pct": 100.0 * sum(
            temperature < lower or temperature > upper for temperature in temperatures
        ) / len(temperatures),
        "heater_duty_pct": 100.0 * mean(heater_states),
        "min_temperature_C": min(temperatures),
        "max_temperature_C": max(temperatures),
    }


def run_monte_carlo(
    scenario_path: str, runs: int, base_seed: int | None = None
) -> List[Dict[str, float | int]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    scenario = load_config(scenario_path)
    first_seed = scenario.sim.seed if base_seed is None else base_seed
    results = []
    for run_index in range(runs):
        seed = first_seed + run_index
        metrics = calculate_metrics(
            simulate_scenario(scenario, seed),
            scenario.controller.setpoint,
            scenario.controller.deadband,
        )
        results.append({"run": run_index + 1, "seed": seed, **metrics})
    return results


def write_results(scenario_path: str, results: List[Dict[str, float | int]]) -> str:
    output_dir = os.path.join("outputs", "monte_carlo")
    os.makedirs(output_dir, exist_ok=True)
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(output_dir, f"{scenario_name}-{timestamp}.csv")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def print_summary(results: List[Dict[str, float | int]]) -> None:
    print(f"Results from {len(results)} runs (mean and empirical 95% percentile range):")
    for metric in (
        "rmse_C",
        "outside_deadband_pct",
        "heater_duty_pct",
        "min_temperature_C",
        "max_temperature_C",
    ):
        values = [result[metric] for result in results]
        lower, upper = np.percentile(values, [2.5, 97.5])
        print(f"  {metric}: mean={mean(values):.3f}, 95% range=[{lower:.3f}, {upper:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated thermostat simulations with different random seeds")
    parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    parser.add_argument("--runs", type=int, default=100, help="Number of simulations (default: 100)")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the first run (defaults to the seed in the scenario file)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    results = run_monte_carlo(args.scenario, args.runs, args.seed)
    output_path = write_results(args.scenario, results)
    print_summary(results)
    print(f"Per-run results written to {output_path}")


if __name__ == "__main__":
    main()
