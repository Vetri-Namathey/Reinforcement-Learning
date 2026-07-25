"""
Baseline heuristic schedulers for the Spatial-Phase RL Scheduler.

Runs three spatially-naive schedulers on the same ThermalGridEnv the
MaskablePPO agent trains on, over the real Azure LLM Inference Trace, and
reports the four core evaluation metrics: P99 TTFT, SLA-attained goodput,
thermal throttling time ratio (TTTR), and spatial temperature variance.

Usage (run from the project root):
    python -m src.baselines.run_baselines                 # full raw Azure conv trace
    python -m src.baselines.run_baselines --jobs 1000     # truncated quick run
    python -m src.baselines.run_baselines --workload code
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.baselines.least_occupied import LeastOccupiedScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.thermal_greedy import ThermalGreedyScheduler
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import ThermalGridEnv

SLA_TTFT_SECONDS = 5.0         # TTFT SLA: job must be placed within 5 s
MAX_STEPS_GUARD = 500_000      # hard cap against a non-terminating policy


def compute_metrics(env: ThermalGridEnv, terminal_info: dict) -> dict:
    """Assemble the four core metrics from env state + terminal info."""
    ttfts = np.array([j["ttft"] for j in env.successful_jobs])
    sim_time = max(env.current_time, 1e-9)

    sla_attained = int(np.sum(ttfts <= SLA_TTFT_SECONDS)) if len(ttfts) else 0
    return {
        "p99_ttft":              terminal_info.get("p99_ttft", 0.0),
        "sla_goodput":           sla_attained / sim_time,
        "sla_attainment_pct":    100.0 * sla_attained / max(len(ttfts), 1),
        "tttr":                  terminal_info.get("final_tttr", 0.0),
        "mean_spatial_variance": terminal_info.get("mean_spatial_variance", 0.0),
        "jobs_placed":           len(ttfts),
        "sim_time":              sim_time,
        "peak_temp":             terminal_info.get("max_temperature", 0.0),
    }


def run_episode(scheduler, fused_df, gpu_model: str = "H100") -> dict:
    """Run one scheduler to episode termination on the fused dataset."""
    env = ThermalGridEnv(fused_dataset=fused_df, gpu_model=gpu_model)
    obs, info = env.reset()

    t0 = time.time()
    terminated = False
    steps = 0
    while not terminated and steps < MAX_STEPS_GUARD:
        action = scheduler.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    wall = time.time() - t0
    if not terminated:
        print(f"  [!] {scheduler.name}: hit {MAX_STEPS_GUARD}-step guard "
              f"before termination")

    metrics = compute_metrics(env, info)
    metrics["steps"] = steps
    metrics["wall_seconds"] = wall
    return metrics


def print_report(all_metrics: dict) -> None:
    """Print the metric comparison table."""
    print("\n" + "=" * 78)
    print("  BASELINE COMPARISON - 4 CORE EVALUATION METRICS")
    print(f"  (SLA: TTFT <= {SLA_TTFT_SECONDS:.1f}s | throttle: any cell above "
          f"its own per-device threshold)")
    print("=" * 78)
    header = (f"  {'Metric':<32}" +
              "".join(f"{name:>15}" for name in all_metrics))
    print(header)
    print("  " + "-" * 76)

    rows = [
        ("P99 TTFT (s)",              "p99_ttft",              "{:.3f}"),
        ("SLA goodput (jobs/s)",      "sla_goodput",           "{:.3f}"),
        ("SLA attainment (%)",        "sla_attainment_pct",    "{:.1f}"),
        ("TTTR (throttle ratio)",     "tttr",                  "{:.4f}"),
        ("Mean spatial var (C^2)",    "mean_spatial_variance", "{:.3f}"),
        ("Jobs placed",               "jobs_placed",           "{:d}"),
        ("Peak temp (C)",             "peak_temp",             "{:.1f}"),
        ("Sim time (s)",              "sim_time",              "{:.1f}"),
        ("Env steps",                 "steps",                 "{:d}"),
        ("Wall time (s)",             "wall_seconds",          "{:.1f}"),
    ]
    for label, key, fmt in rows:
        row = f"  {label:<32}"
        for name in all_metrics:
            row += f"{fmt.format(all_metrics[name][key]):>15}"
        print(row)
    print("=" * 78)
    print("  Compare against the trained MaskablePPO agent on the same trace.")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline schedulers")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Truncate trace to N jobs (default: full trace)")
    parser.add_argument("--workload", choices=["conv", "code"], default="conv",
                        help="Azure trace variant (default: conv)")
    parser.add_argument("--gpu-model", default="H100",
                        help="Homogeneous grid GPU model (default: H100)")
    args = parser.parse_args()

    fuser = WorkloadFuser(gpu_models=args.gpu_model, seed=42)
    fused_df = fuser.generate_fused_dataset(
        num_requests=args.jobs, source="azure", workload=args.workload
    )
    print(f"Loaded {len(fused_df)} fused jobs "
          f"({fused_df.attrs['source']}, GPU={fused_df.attrs['gpu_model']}, "
          f"span={fused_df['arrival_time'].max():.0f}s)")

    schedulers = [
        RoundRobinScheduler(),
        LeastOccupiedScheduler(),
        ThermalGreedyScheduler(),
    ]

    all_metrics = {}
    for sched in schedulers:
        print(f"\nRunning {sched.name} to termination...")
        m = run_episode(sched, fused_df, gpu_model=args.gpu_model)
        all_metrics[sched.name] = m
        print(f"  Done: {m['steps']} steps, {m['jobs_placed']} jobs, "
              f"{m['wall_seconds']:.1f}s wall")

    print_report(all_metrics)


if __name__ == "__main__":
    main()
