"""
stress_sweep.py -- does the ranking hold when the fleet is actually loaded?

The shipped evaluation runs at the trace's recorded arrival rate, which was
recorded for a fleet far larger than 36 GPUs. The result is a grid that spends
most of its time below saturation: ~30 of 36 cells legal per decision, and P99
TTFT identical to three decimals across four of the five schedulers. A regime
where the primary latency metric cannot separate the policies is not a regime
that tests placement quality on that metric.

Thermal risk and scheduling mistakes both concentrate near saturation, so that is
where a scheduler's value shows up -- and it is exactly the region the headline
number does not cover. This sweep walks the load up and reports where the
policies start to separate, and whether the ranking at the recorded rate survives.

Read the output as: the recorded rate (accel 1.0) is the reported operating
point; the higher rows are the risk profile a real deployment would care about.
A ranking that flips as load rises means the headline comparison does not
generalise past the point it was measured at -- worth knowing either way.

Run: python -m tools.stress_sweep
     python -m tools.stress_sweep --accels 1.0 1.5 2.0 3.0 --metric tttr
     python -m tools.stress_sweep --no-rl          # heuristics only, no checkpoint
"""
from __future__ import annotations

import argparse

from configs import MODELS_DIR
from src.baselines.run_baselines import (
    LOWER_IS_BETTER,
    all_schedulers,
    build_layout,
    resolve_candidate_k,
    run_episode,
)
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.models.train_agent import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SLOTS,
    EVAL_JOBS,
    EVAL_OFFSET,
)

# Reported alongside the headline metric so the load level is legible: how hot it
# got, how long it throttled, and how much work was actually admitted.
CONTEXT = [("peak_temp", "peak C", "{:.1f}"), ("tttr", "TTTR", "{:.3f}"),
           ("p99_ttft", "P99 TTFT", "{:.3f}")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Load stress sweep")
    ap.add_argument("--accels", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0],
                    help="Arrival-rate multipliers to sweep (default: 1.0 1.5 2.0 3.0). "
                         "1.0 is the trace's own recorded rate.")
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--jobs", type=int, default=EVAL_JOBS)
    ap.add_argument("--offset", type=int, default=EVAL_OFFSET)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--metric", default="total_reward")
    ap.add_argument("--model", default="spatial_thermal_ppo")
    ap.add_argument("--no-rl", action="store_true",
                    help="Skip the trained checkpoint (heuristics only).")
    args = ap.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)
    lower_better = args.metric in LOWER_IS_BETTER

    layout = build_layout("checkerboard", args.grid_size)

    model = None
    if not args.no_rl:
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(str(MODELS_DIR / args.model))
        except Exception as exc:
            print(f"[!!] Could not load {args.model}: {exc}\n     Continuing without it.")

    def schedulers():
        return all_schedulers(model)

    names = [s.name for s in schedulers()]

    print("=" * 96)
    print("  LOAD STRESS SWEEP")
    print(f"  jobs={args.jobs} offset={args.offset} slots/cell={args.slots} "
          f"candidate_k={candidate_k}  metric={args.metric}")
    print("=" * 96)
    header = f"  {'accel':>6}" + "".join(f"{n[:13]:>15}" for n in names) + f"{'winner':>17}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for accel in args.accels:
        df = WorkloadFuser(gpu_models=["H100", "A100"], seed=42).generate_fused_dataset(
            num_requests=args.jobs, source="azure", workload="conv",
            offset=args.offset, arrival_accel=accel)
        res = {s.name: run_episode(s, df, gpu_model="H100", grid_size=args.grid_size,
                                   gpu_layout_matrix=layout,
                                   max_jobs_per_cell=args.slots,
                                   candidate_k=candidate_k)
               for s in schedulers()}
        vals = {n: res[n][args.metric] for n in names}
        winner = (min if lower_better else max)(vals, key=vals.get)
        spread = max(vals.values()) - min(vals.values())
        denom = abs(max(vals.values())) or 1.0
        print(f"  {accel:6.1f}" + "".join(f"{vals[n]:15.1f}" for n in names)
              + f"{winner[:16]:>17}")
        rows.append((accel, res, winner, 100 * spread / denom))

    # ── how loaded each rung actually was ────────────────────────────────
    print("\n  Load context (Best-Fit, the reference policy):")
    print(f"  {'accel':>6}" + "".join(f"{lab:>12}" for _, lab, _ in CONTEXT)
          + f"{'spread %':>11}")
    print("  " + "-" * 47)
    for accel, res, _, spread in rows:
        ref = res["Best-Fit"]
        print(f"  {accel:6.1f}" + "".join(f"{fmt.format(ref[k]):>12}"
                                          for k, _, fmt in CONTEXT)
              + f"{spread:10.1f}%")

    winners = {w for _, _, w, _ in rows}
    print()
    if len(winners) == 1:
        print(f"  The ranking is stable: {winners.pop()} wins at every load level")
        print(f"  tested on {args.metric}, so the headline comparison is not an")
        print("  artifact of the recorded rate.")
    else:
        print(f"  The ranking CHANGES with load ({', '.join(sorted(winners))} each win")
        print("  somewhere). The recorded-rate result does not generalise to")
        print("  saturation and should be reported with its operating point.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
