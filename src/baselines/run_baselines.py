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

from src.baselines.best_fit import BestFitScheduler
from src.baselines.least_occupied import LeastOccupiedScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.thermal_greedy import ThermalGreedyScheduler
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import (
    ThermalGridEnv,
    make_checkerboard_layout,
    make_split_layout,
)
from src.models.train_agent import DEFAULT_CANDIDATE_K, DEFAULT_LAYOUT, DEFAULT_SLOTS

SLA_TTFT_SECONDS = 5.0         # TTFT SLA: job must be placed within 5 s
MAX_STEPS_GUARD = 500_000      # hard cap against a non-terminating policy

# ── Shared evaluation surface ────────────────────────────────────────────
# This module is both a CLI and the harness every other evaluator runs on:
# generate_report, held_out_eval, campus_deployment_eval, separation_gate and
# the tools/ scripts all import run_episode from here. The helpers below were
# previously copy-pasted into each of them -- four copies of the layout lookup,
# seven of the candidate_k coercion, six scheduler lists, and two *disagreeing*
# copies of the lower-is-better metric set. One definition each, so a fix lands
# everywhere at once.

LAYOUT_FNS = {"homogeneous": None, "checkerboard": make_checkerboard_layout,
              "split": make_split_layout}

# Metrics where a smaller number is the better one, so the sign of a "win"
# flips. Anything not listed is treated as higher-is-better.
LOWER_IS_BETTER = frozenset({"p99_ttft", "tttr", "peak_temp", "sim_time",
                             "mean_spatial_variance", "mean_spatial_var",
                             "throttled_work_ratio"})


def build_layout(name: str, grid_size: int = 6, gpu_model: str = "H100",
                 secondary: str = "A100"):
    """Resolve a layout name to a (grid, grid) matrix of GPU model names."""
    fn = LAYOUT_FNS[name]
    return None if fn is None else fn(gpu_model, secondary, grid_size=grid_size)


def layout_fleet(name: str, gpu_model: str = "H100", secondary: str = "A100") -> list[str]:
    """The GPU models a layout actually places, for the WorkloadFuser."""
    return [gpu_model] if LAYOUT_FNS[name] is None else [gpu_model, secondary]


def resolve_candidate_k(value: int | None) -> int | None:
    """Coerce a --candidate-k CLI value to the env's parameter.

    argparse cannot express "an int, or None" cleanly, so 0 is the sentinel for
    the unrestricted 36-cell action space and every CLI spells it the same way.
    """
    return value if value and value > 0 else None


def all_schedulers(model=None, rl_name: str = "Spatial-Phase RL") -> list:
    """The four heuristics, plus the trained policy when a model is given.

    Schedulers are stateful (Round-Robin holds a cursor), so this returns fresh
    instances on every call -- reusing one across episodes leaks state between
    them and was a real source of irreproducible numbers.
    """
    scheds = [RoundRobinScheduler(), LeastOccupiedScheduler(),
              ThermalGreedyScheduler(), BestFitScheduler()]
    if model is not None:
        from src.baselines.trained_policy import TrainedPolicyScheduler
        scheds.append(TrainedPolicyScheduler(model, name=rl_name))
    return scheds


def compute_instrumentation(actions: list, power_draw_history: list, queue_lengths: list,
                            num_cells: int) -> dict:
    """Compute action distribution, mean power, and queue depth percentiles."""
    # Action distribution: count placements per cell (exclude WAIT actions)
    cell_placements = np.zeros(num_cells, dtype=int)
    for action in actions:
        if action < num_cells:
            cell_placements[action] += 1
    
    total_placements = int(np.sum(cell_placements))
    placement_spread = float(np.std(cell_placements)) if total_placements > 0 else 0.0
    
    # Mean power draw: time-averaged across all steps
    mean_power = float(np.mean(power_draw_history)) if power_draw_history else 0.0
    
    # Queue depth percentiles
    queue_p50 = float(np.percentile(queue_lengths, 50)) if queue_lengths else 0.0
    queue_p95 = float(np.percentile(queue_lengths, 95)) if queue_lengths else 0.0
    queue_p99 = float(np.percentile(queue_lengths, 99)) if queue_lengths else 0.0
    
    return {
        "cell_placements": cell_placements,
        "total_placements": total_placements,
        "placement_spread": placement_spread,
        "mean_power": mean_power,
        "queue_p50": queue_p50,
        "queue_p95": queue_p95,
        "queue_p99": queue_p99,
    }


def compute_metrics(env: ThermalGridEnv, terminal_info: dict,
                    peak_temp: float | None = None) -> dict:
    """Assemble the four core metrics from env state + terminal info.

    `peak_temp` is the max temperature observed over the whole episode and must
    be tracked by the caller. terminal_info["max_temperature"] is the *final*
    step's max, which is not the peak -- by termination the grid has largely
    cooled, so reading it here reported a number well below the real peak.
    """
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
        "peak_temp":             (
            float(peak_temp) if peak_temp is not None
            else terminal_info.get("max_temperature", 0.0)
        ),
        # The quantity the RL objective actually optimises. It was previously
        # not collected here at all, which is why the headline "mean episode
        # reward" comparison could not be regenerated from this repo.
        "total_reward":          float(env.total_reward),
        "reward_place":          float(env.reward_totals["place"]),
        "reward_sla":            float(env.reward_totals["sla"]),
        "reward_thermal":        float(env.reward_totals["thermal"]),
    }


def run_episode(scheduler, fused_df, gpu_model: str = "H100", grid_size: int = 6,
                gpu_layout_matrix=None, active_cell_mask=None,
                max_jobs_per_cell: int = 1, candidate_k: int | None = None) -> dict:
    """Run one scheduler to episode termination on the fused dataset.

    Also collects instrumentation: action distribution, power draw history,
    and queue depth trace.

    ``candidate_k`` MUST match the value the evaluated policy was trained with.
    The shortlist is part of the MDP, not a display filter: it changes the legal
    action set the policy's log-probabilities and advantages were computed over.
    Scoring a k=6 checkpoint on a k=None environment compares two different
    decision problems. It is an argument rather than a fixed constant because
    the same harness scores heuristics, which have no training config -- but for
    a like-for-like table every scheduler in one run must use the same value.
    """
    env = ThermalGridEnv(fused_dataset=fused_df, gpu_model=gpu_model, grid_size=grid_size,
                         gpu_layout_matrix=gpu_layout_matrix, active_cell_mask=active_cell_mask,
                         max_jobs_per_cell=max_jobs_per_cell, candidate_k=candidate_k)
    obs, info = env.reset()

    t0 = time.time()
    terminated = False
    steps = 0
    
    # Instrumentation collectors
    actions_taken = []
    power_draw_history = []
    queue_lengths = []
    peak_temp = 0.0

    while not terminated and steps < MAX_STEPS_GUARD:
        action = scheduler.select_action(obs)
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # Record AFTER the step so we capture the effect of the action
        # (e.g., the power spike from a placement, updated queue after ingestion)
        mean_power = float(np.mean(obs["power_draw"]))
        power_draw_history.append(mean_power)
        queue_lengths.append(info["queue_length"])
        peak_temp = max(peak_temp, info["max_temperature"])

    wall = time.time() - t0
    if not terminated:
        print(f"  [!] {scheduler.name}: hit {MAX_STEPS_GUARD}-step guard "
              f"before termination")

    metrics = compute_metrics(env, info, peak_temp=peak_temp)
    metrics["steps"] = steps
    metrics["wall_seconds"] = wall
    
    # Compute instrumentation
    instr = compute_instrumentation(actions_taken, power_draw_history, queue_lengths, env.NUM_CELLS)
    metrics.update(instr)
    
    return metrics


COLUMN_WIDTH = 18


def _col(name: str) -> str:
    """Right-align a scheduler name into one fixed-width table column."""
    return f"{name[:COLUMN_WIDTH - 1]:>{COLUMN_WIDTH}}"


def print_report(all_metrics: dict) -> None:
    """Print the metric comparison table."""
    print("\n" + "=" * 78)
    print("  BASELINE COMPARISON - 4 CORE EVALUATION METRICS")
    print(f"  (SLA: TTFT <= {SLA_TTFT_SECONDS:.1f}s | throttle: any cell above "
          f"its own per-device threshold)")
    print("=" * 78)
    # Column names are truncated to the column width: a name longer than the
    # field (e.g. "Spatial-Phase RL (campus deploy)") otherwise overflows and
    # runs into its neighbour, making the table unreadable.
    width = 78
    print(f"  {'Metric':<32}" + "".join(_col(name) for name in all_metrics))
    print("  " + "-" * width)

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
            row += f"{fmt.format(all_metrics[name][key]):>{COLUMN_WIDTH}}"
        print(row)
    print("=" * 78)
    
    # Instrumentation section
    print("\n" + "=" * 78)
    print("  INSTRUMENTATION - LOAD DISTRIBUTION & QUEUEING")
    print("=" * 78)
    instr_rows = [
        ("Mean power draw (W)",       "mean_power",            "{:.1f}"),
        ("Cell placement spread",    "placement_spread",      "{:.2f}"),
        ("Queue depth p50",          "queue_p50",             "{:.1f}"),
        ("Queue depth p95",          "queue_p95",             "{:.1f}"),
        ("Queue depth p99",          "queue_p99",             "{:.1f}"),
    ]
    print(f"  {'Metric':<32}" + "".join(_col(name) for name in all_metrics))
    print("  " + "-" * 78)
    
    for label, key, fmt in instr_rows:
        row = f"  {label:<32}"
        for name in all_metrics:
            row += f"{fmt.format(all_metrics[name][key]):>{COLUMN_WIDTH}}"
        print(row)
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline schedulers")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Truncate trace to N jobs (default: full trace)")
    parser.add_argument("--workload", choices=["conv", "code"], default="conv",
                        help="Azure trace variant (default: conv)")
    parser.add_argument("--gpu-model", default="H100",
                        help="Homogeneous grid GPU model (default: H100)")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--layout", choices=["homogeneous", "checkerboard", "split"],
                        default=DEFAULT_LAYOUT,
                        help="Grid hardware layout. Must match how the agent was trained.")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                        help=f"Concurrent jobs per cell, continuous batching "
                             f"(default: {DEFAULT_SLOTS}). Pass 1 to reproduce the "
                             "single-job-per-GPU environment.")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                        help=f"Shortlist width, i.e. how many cells a scheduler may "
                             f"choose between (default: {DEFAULT_CANDIDATE_K}, the value "
                             "the shipped checkpoint was trained on). Pass 0 for the "
                             "unrestricted 36-cell action space.")
    args = parser.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)

    layout_matrix = build_layout(args.layout, args.grid_size, args.gpu_model)
    gpu_models = layout_fleet(args.layout, args.gpu_model)
    fuser = WorkloadFuser(gpu_models=gpu_models, seed=42)
    fused_df = fuser.generate_fused_dataset(
        num_requests=args.jobs, source="azure", workload=args.workload
    )
    print(f"Config: layout={args.layout} slots/cell={args.slots} "
          f"candidate_k={candidate_k if candidate_k else 'unrestricted (36)'}")
    print(f"Loaded {len(fused_df)} fused jobs "
          f"({fused_df.attrs['source']}, GPU={fused_df.attrs['gpu_model']}, "
          f"span={fused_df['arrival_time'].max():.0f}s)")

    schedulers = all_schedulers()

    all_metrics = {}
    for sched in schedulers:
        print(f"\nRunning {sched.name} to termination...")
        m = run_episode(sched, fused_df, gpu_model=args.gpu_model,
                        grid_size=args.grid_size, max_jobs_per_cell=args.slots,
                        gpu_layout_matrix=layout_matrix, candidate_k=candidate_k)
        all_metrics[sched.name] = m
        print(f"  Done: {m['steps']} steps, {m['jobs_placed']} jobs, "
              f"{m['wall_seconds']:.1f}s wall")

    print_report(all_metrics)


if __name__ == "__main__":
    main()
