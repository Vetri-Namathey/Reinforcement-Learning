"""
held_out_eval.py -- the multi-slice comparison behind the headline claim.

The README reports "across eight held-out slices ... RN -X% vs Best-Fit (paired
t = ...)". Nothing in this repo used to compute that: generate_report.py scores a
single slice, so the t-statistic was quoted as a conclusion without the paired
samples or their variance ever being committed. This script is that missing step.

Method
------
N disjoint slices of EVAL_JOBS rows each, laid end to end starting at
EVAL_OFFSET, which is where the training pool stops. Every scheduler sees the
same N slices, so the samples are *paired*: slice-to-slice traffic differences
(the dominant variance term, +/-1300 reward between windows) cancel in the
difference, which a two-sample test on the raw means would not do.

For each baseline, the reported statistic is a paired t-test on the per-slice
differences (RL - baseline). With N=8 there are 7 degrees of freedom, so |t| must
exceed 2.36 for p < 0.05 -- the script prints the verdict rather than leaving the
reader to look it up.

Every scheduler runs at the same candidate_k as the shipped checkpoint was
trained on. Scoring the agent on a wider action space than it was optimised over
is a different decision problem, not a harder version of the same one.

Output: artifacts/report/held_out_slices.json -- per-slice raw returns for every
scheduler, so the test can be re-derived without re-running the episodes.

Run: python -m src.models.held_out_eval
     python -m src.models.held_out_eval --slices 8 --metric total_reward
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from configs import ARTIFACTS_DIR, MODELS_DIR
from src.baselines.run_baselines import (
    LOWER_IS_BETTER,
    all_schedulers,
    build_layout,
    layout_fleet,
    resolve_candidate_k,
    run_episode,
)
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.models.train_agent import (
    DEFAULT_ARRIVAL_ACCEL,
    DEFAULT_CANDIDATE_K,
    DEFAULT_LAYOUT,
    DEFAULT_SLOTS,
    EVAL_JOBS,
    EVAL_OFFSET,
    TRAIN_POOL_JOBS,
)

REPORT_DIR = ARTIFACTS_DIR / "report"
RL_NAME = "Spatial-Phase RL"





def main() -> int:
    ap = argparse.ArgumentParser(description="Paired multi-slice held-out evaluation")
    ap.add_argument("--slices", type=int, default=8,
                    help="Number of disjoint held-out slices (default: 8).")
    ap.add_argument("--jobs", type=int, default=EVAL_JOBS,
                    help=f"Jobs per slice (default: {EVAL_JOBS}).")
    ap.add_argument("--start-offset", type=int, default=EVAL_OFFSET,
                    help=f"Trace row the first slice starts at (default: {EVAL_OFFSET}, "
                         f"where the {TRAIN_POOL_JOBS}-job training pool ends).")
    ap.add_argument("--metric", default="total_reward",
                    help="Metric to test (default: total_reward, the RL objective).")
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--gpu-model", default="H100")
    ap.add_argument("--layout", choices=["homogeneous", "checkerboard", "split"],
                    default=DEFAULT_LAYOUT)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--accel", type=float, default=DEFAULT_ARRIVAL_ACCEL)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--model", default="spatial_thermal_ppo")
    args = ap.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)

    if args.start_offset < TRAIN_POOL_JOBS:
        print(f"[!!] start-offset {args.start_offset} < {TRAIN_POOL_JOBS}: the first "
              "slices overlap the training pool. These are not held-out results.")

    layout = build_layout(args.layout, args.grid_size, args.gpu_model)
    gpus = layout_fleet(args.layout, args.gpu_model)

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(MODELS_DIR / args.model))

    def fresh_schedulers():
        return all_schedulers(model, rl_name=RL_NAME)

    names = [s.name for s in fresh_schedulers()]
    offsets = [args.start_offset + i * args.jobs for i in range(args.slices)]

    print("=" * 78)
    print("  HELD-OUT MULTI-SLICE EVALUATION")
    print(f"  {args.slices} disjoint slices of {args.jobs} jobs, rows "
          f"{offsets[0]}..{offsets[-1] + args.jobs}")
    print(f"  layout={args.layout}  slots/cell={args.slots}  accel={args.accel}x  "
          f"candidate_k={candidate_k if candidate_k else 'unrestricted'}")
    print(f"  metric under test: {args.metric}")
    print("=" * 78)

    per_slice: dict[str, list[float]] = {n: [] for n in names}
    full_records = []

    for i, off in enumerate(offsets):
        fuser = WorkloadFuser(gpu_models=gpus, seed=42)
        df = fuser.generate_fused_dataset(num_requests=args.jobs, source="azure",
                                          workload="conv", offset=off,
                                          arrival_accel=args.accel)
        row = {"slice": i, "offset": off}
        for sched in fresh_schedulers():
            m = run_episode(sched, df, gpu_model=args.gpu_model, grid_size=args.grid_size,
                            gpu_layout_matrix=layout, max_jobs_per_cell=args.slots,
                            candidate_k=candidate_k)
            per_slice[sched.name].append(float(m[args.metric]))
            row[sched.name] = {k: float(v) for k, v in m.items()
                               if isinstance(v, (int, float))}
        full_records.append(row)
        vals = "  ".join(f"{n[:11]}={per_slice[n][-1]:8.1f}" for n in names)
        print(f"  slice {i} (rows {off}-{off + args.jobs}):  {vals}")

    # ── paired test, RL against every baseline ───────────────────────────
    rl = np.array(per_slice[RL_NAME])
    lower_better = args.metric in LOWER_IS_BETTER
    print("\n" + "=" * 78)
    print(f"  PAIRED COMPARISON on {args.metric}  (n={args.slices}, "
          f"df={args.slices - 1}, {'lower' if lower_better else 'higher'} is better)")
    print("=" * 78)
    print(f"  {'baseline':<22}{'mean':>10}{'RL mean':>10}{'diff':>10}{'delta %':>10}"
          f"{'t':>8}{'p':>9}  verdict")
    print("  " + "-" * 82)

    comparisons = {}
    for name in names:
        if name == RL_NAME:
            continue
        base = np.array(per_slice[name])
        diff = rl - base                      # paired differences
        t, p = stats.ttest_rel(rl, base)
        # A percentage needs a positive denominator to mean anything. Round-Robin's
        # mean return is negative, so a ratio against it is noise; the absolute
        # paired difference is printed alongside and is always valid.
        pct = (100.0 * diff.mean() / abs(base.mean())
               if base.mean() > 0 else float("nan"))
        if lower_better:
            pct = -pct
        better = (diff.mean() < 0) if lower_better else (diff.mean() > 0)
        if p >= 0.05:
            verdict = "no difference (p>=0.05)"
        else:
            verdict = "RL better" if better else "RL worse"
        comparisons[name] = {
            "baseline_mean": float(base.mean()), "rl_mean": float(rl.mean()),
            "mean_paired_diff": float(diff.mean()), "sd_paired_diff": float(diff.std(ddof=1)),
            "delta_pct": float(pct), "t_stat": float(t), "p_value": float(p),
            "significant_at_05": bool(p < 0.05), "verdict": verdict,
        }
        pct_s = "      n/a" if np.isnan(pct) else f"{pct:+9.1f}%"
        print(f"  {name[:21]:<22}{base.mean():10.1f}{rl.mean():10.1f}"
              f"{diff.mean():+10.1f}{pct_s}{t:8.2f}{p:9.4f}  {verdict}")

    print(f"\n  Reading this: with n={args.slices} the 5% two-sided "
          f"critical value is |t| = {stats.t.ppf(0.975, args.slices - 1):.2f}.")
    print("  A |t| below that means the slices do not separate the two schedulers,")
    print("  which is a real finding -- not a win for either side.")
    print("  'delta %' is omitted where the baseline mean is not positive:")
    print("  a ratio against a negative denominator is not interpretable, so")
    print("  read the 'diff' column (mean paired difference) there instead.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "held_out_slices.json"
    payload = {
        "config": {
            "slices": args.slices, "jobs_per_slice": args.jobs,
            "offsets": offsets, "metric": args.metric, "layout": args.layout,
            "slots": args.slots, "arrival_accel": args.accel,
            "candidate_k": candidate_k, "candidate_k_trained": DEFAULT_CANDIDATE_K,
            "action_space_match": candidate_k == DEFAULT_CANDIDATE_K,
            "model": args.model, "lower_is_better": lower_better,
        },
        "per_slice_metric": per_slice,
        "comparisons": comparisons,
        "per_slice_full": full_records,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved -> {out}")
    print("  The per-slice returns are committed, so the t-test above can be")
    print("  re-derived without re-running a single episode.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
