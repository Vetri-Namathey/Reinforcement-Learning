"""
vram_feasibility_audit.py -- how many scheduled placements are physically
impossible on real hardware?

The action mask admits a job whenever a cell has a free slot. Slots are not the
only resource: each resident request holds a KV cache proportional to its total
token count, and three long-context requests can exhaust a 24 GB board while
three slots are still nominally free. A schedule the simulator calls legal can
therefore be one a real inference server would reject or OOM on.

This script quantifies that gap instead of leaving it as a footnote. It runs the
same workload twice -- once with the shipped slot-only admission rule, once with
KV-cache capacity enforced as well -- and reports what changes.

Read the output as: if `placements blocked` is 0, the slot rule happens to be
safe for this workload and fleet, and the gap is latent rather than active. If it
is non-zero, that many placements in the reported results were physically
infeasible, and the fix is to run with vram_aware=True.

Run: python -m tools.vram_feasibility_audit
     python -m tools.vram_feasibility_audit --slots 6      # stress the memory
"""
from __future__ import annotations

import argparse

import numpy as np

from configs import get_hardware_profile, kv_cache_gib, kv_capacity_gib
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
    ap = argparse.ArgumentParser(description="KV-cache feasibility audit")
    ap.add_argument("--jobs", type=int, default=EVAL_JOBS)
    ap.add_argument("--offset", type=int, default=EVAL_OFFSET)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--accel", type=float, default=DEFAULT_ARRIVAL_ACCEL)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--grid-size", type=int, default=6)
    args = ap.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)

    layout = build_layout("checkerboard", args.grid_size)
    df = WorkloadFuser(gpu_models=["H100", "A100"], seed=42).generate_fused_dataset(
        num_requests=args.jobs, source="azure", workload="conv",
        offset=args.offset, arrival_accel=args.accel)

    need = np.array([kv_cache_gib(p, d) for p, d
                     in zip(df["prefill_tokens"], df["decode_tokens"], strict=False)])

    print("=" * 78)
    print("  KV-CACHE FEASIBILITY AUDIT")
    print(f"  jobs={args.jobs} slots/cell={args.slots} accel={args.accel}x "
          f"candidate_k={candidate_k}")
    print("=" * 78)
    print("  Per-device KV budget (VRAM x usable fraction):")
    for m in ("H100", "A100", "RTX6000"):
        prof = get_hardware_profile(m)
        cap = kv_capacity_gib(prof)
        print(f"    {m:<9} {prof['vram_gb']:5.0f} GB board -> {cap:6.2f} GiB cache "
              f"-> holds {int(cap / need.mean())} jobs at this workload's mean")
    print(f"\n  Per-job KV footprint over {args.jobs} jobs (GiB):")
    print(f"    mean {need.mean():.3f}   p50 {np.percentile(need, 50):.3f}   "
          f"p99 {np.percentile(need, 99):.3f}   max {need.max():.3f}")

    worst = need.max() * args.slots
    tight = min(kv_capacity_gib(get_hardware_profile(m)) for m in ("H100", "A100"))
    print(f"\n  Worst case: {args.slots} x {need.max():.3f} = {worst:.2f} GiB on a "
          f"cell whose smallest budget is {tight:.2f} GiB "
          f"-> {'OVERCOMMIT POSSIBLE' if worst > tight else 'always fits'}")

    # ── the two admission rules, same workload ───────────────────────────
    print("\n  Running Best-Fit under both admission rules ...")
    results = {}
    for label, vram in (("slot-only (shipped)", False), ("slot + KV capacity", True)):
        env_kw = dict(gpu_model="H100", grid_size=args.grid_size,
                      gpu_layout_matrix=layout, max_jobs_per_cell=args.slots,
                      candidate_k=candidate_k)
        # run_episode builds the env itself, so the flag is applied by patching
        # the constructor default for the duration of this run.
        original = ThermalGridEnv.__init__

        def patched(self, *a, _v=vram, **kw):
            kw.setdefault("vram_aware", _v)
            original(self, *a, **kw)

        ThermalGridEnv.__init__ = patched
        try:
            results[label] = run_episode(BestFitScheduler(), df, **env_kw)
        finally:
            ThermalGridEnv.__init__ = original

    print(f"\n  {'rule':<24}{'placed':>9}{'makespan':>11}{'peak C':>9}"
          f"{'TTTR':>8}{'reward':>10}")
    print("  " + "-" * 71)
    for label, m in results.items():
        print(f"  {label:<24}{m['jobs_placed']:9d}{m['sim_time']:11.1f}"
              f"{m['peak_temp']:9.1f}{m['tttr']:8.3f}{m['total_reward']:10.0f}")

    a, b = results["slot-only (shipped)"], results["slot + KV capacity"]
    blocked = a["jobs_placed"] - b["jobs_placed"]
    print(f"\n  placements blocked by the KV rule: {blocked}")
    if blocked == 0 and abs(a["sim_time"] - b["sim_time"]) < 1e-6:
        print("  The two rules produce an identical schedule on this workload:")
        print("  Azure conv requests are short enough that 3 of them never")
        print("  exhaust even the smallest board here. The slot-only results")
        print("  are therefore physically feasible as reported -- the gap is")
        print("  latent, and would open on long-context traffic or more slots.")
        print("  Confirm by re-running with --slots 6 or a code-workload trace.")
    else:
        print("  The rules disagree: the shipped slot-only numbers include")
        print("  placements a real serving runtime would have refused.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
