"""
measure_reward_balance.py -- re-derive the THERMAL_PENALTY_COEFF calibration.

THERMAL_PENALTY_COEFF decides the throughput/heat trade-off the whole scheduler
rests on. Its calibration comment records the sweep that picked 3.0 -- thermal at
~4% of placement reward at coeff 0.5, ~24% at 3.0, ~39% at 5.0 -- but that sweep
was run on a 6x6 / 2-slot / 1.4x-arrival grid, and the shipped configuration is
3 slots at 1.0x. Both knobs change how hot the grid gets and therefore how often
the penalty fires at all, so the 24% figure was an assertion about a
configuration nobody runs.

This script measures the real split. It sweeps the coefficient on whatever
configuration is passed, and reports each reward term as a share of the gross
placement reward, using the per-term totals the environment now accumulates.

The number that matters is the *thermal* column. If it reads ~0%, the penalty
never fires in the evaluated regime and the coefficient is inert -- which is a
finding about the evaluation load, not a reason to raise the coefficient.

Run: python -m tools.measure_reward_balance
     python -m tools.measure_reward_balance --accel 1.4 --slots 2   # old sweep
"""
from __future__ import annotations

import argparse

from src.baselines.best_fit import BestFitScheduler
from src.baselines.run_baselines import build_layout, resolve_candidate_k, run_episode
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import ThermalGridEnv
from src.models.train_agent import (
    DEFAULT_ARRIVAL_ACCEL,
    DEFAULT_CANDIDATE_K,
    DEFAULT_SLOTS,
    EVAL_JOBS,
    EVAL_OFFSET,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the reward-term balance")
    ap.add_argument("--jobs", type=int, default=EVAL_JOBS)
    ap.add_argument("--offset", type=int, default=EVAL_OFFSET)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--accel", type=float, default=DEFAULT_ARRIVAL_ACCEL)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--coeffs", type=float, nargs="+", default=[0.5, 1.0, 3.0, 5.0, 10.0])
    ap.add_argument("--grid-size", type=int, default=6)
    args = ap.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)

    layout = build_layout("checkerboard", args.grid_size)
    df = WorkloadFuser(gpu_models=["H100", "A100"], seed=42).generate_fused_dataset(
        num_requests=args.jobs, source="azure", workload="conv",
        offset=args.offset, arrival_accel=args.accel)

    print("=" * 78)
    print("  REWARD-TERM BALANCE  (Best-Fit reference policy)")
    print(f"  jobs={args.jobs} offset={args.offset} slots/cell={args.slots} "
          f"accel={args.accel}x candidate_k={candidate_k}")
    print(f"  shipped THERMAL_PENALTY_COEFF = {ThermalGridEnv.THERMAL_PENALTY_COEFF}")
    print("=" * 78)
    print(f"  {'coeff':>7}{'placement':>12}{'sla':>10}{'thermal':>10}"
          f"{'thermal %':>11}{'net':>10}{'peak C':>9}{'TTTR':>8}")
    print("  " + "-" * 74)

    original = ThermalGridEnv.THERMAL_PENALTY_COEFF
    try:
        for c in args.coeffs:
            # The coefficient is a class attribute read at reward time, so
            # setting it here applies to the episode built below.
            ThermalGridEnv.THERMAL_PENALTY_COEFF = c
            m = run_episode(BestFitScheduler(), df, gpu_model="H100",
                            grid_size=args.grid_size, gpu_layout_matrix=layout,
                            max_jobs_per_cell=args.slots, candidate_k=candidate_k)
            place, sla, th = m["reward_place"], m["reward_sla"], m["reward_thermal"]
            share = 100.0 * th / place if place else 0.0
            mark = "  <-- shipped" if c == original else ""
            print(f"  {c:7.1f}{place:12.0f}{sla:10.1f}{th:10.1f}{share:10.1f}%"
                  f"{m['total_reward']:10.0f}{m['peak_temp']:9.1f}{m['tttr']:8.3f}{mark}")
    finally:
        ThermalGridEnv.THERMAL_PENALTY_COEFF = original

    print("\n  'thermal %' is the thermal penalty as a share of gross placement")
    print("  reward over the episode. Near 0% means the grid never crosses a")
    print("  throttle point under this load, so the coefficient cannot shape")
    print("  behaviour no matter how it is set -- raise --accel or lower --slots")
    print("  to reach the regime the penalty was designed for.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
