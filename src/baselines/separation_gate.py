"""Separation gate -- run BEFORE spending any time on training.

Two independent things must hold before a training run is worth starting, and
they pull in opposite directions:

  1. SEPARATION  -- the heuristic baselines must actually disagree. If four
     different rules all score the same, the environment has no decision
     structure and no learned policy can beat them either.

  2. BRANCHING   -- the scheduler must have real choices. Raising load creates
     separation, but push it too far and every cell is full: the mask leaves
     0-1 legal actions and no policy can express a preference among one option.

Checking only (1) is how a configuration with 53.5% single-option decisions got
picked as "passing". The gate now requires both.

Usage (from the project root):
    python -m src.baselines.separation_gate --jobs 500 --slots 3 --accel 1.0
    python -m src.baselines.separation_gate --jobs 500 --slots 1 --accel 1.0
"""
import argparse
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

from src.baselines.least_occupied import LeastOccupiedScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.run_baselines import run_episode
from src.baselines.thermal_greedy import ThermalGreedyScheduler
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import ThermalGridEnv, make_checkerboard_layout
from src.models.train_agent import DEFAULT_ARRIVAL_ACCEL, DEFAULT_LAYOUT, DEFAULT_SLOTS

CONSEQUENTIAL = [("p99_ttft", "P99 TTFT (s)"), ("tttr", "TTTR"),
                 ("sla_attainment_pct", "SLA attain (%)"), ("sim_time", "Makespan (s)")]
DESCRIPTIVE = [("mean_spatial_variance", "Spatial var (C^2)"), ("peak_temp", "Peak temp (C)")]

MIN_SEPARATION = 0.10      # >=10% spread on some consequential metric
MIN_MEAN_OPTIONS = 5.0     # average legal cells per decision
MAX_FRAC_STUCK = 0.25      # share of decisions offering <=1 cell


def measure_branching(df, grid_size, lay, slots):
    """Average number of legal cells the scheduler can choose between."""
    env = ThermalGridEnv(fused_dataset=df, gpu_model="H100", grid_size=grid_size,
                         gpu_layout_matrix=lay, max_jobs_per_cell=slots)
    obs, _ = env.reset()
    probe = ThermalGreedyScheduler()
    opts, terminated, steps = [], False, 0
    while not terminated and steps < 300_000:
        if env.queue:
            opts.append(int(obs["action_mask"][:env.NUM_CELLS].sum()))
        obs, _, terminated, _, _ = env.step(probe.select_action(obs))
        steps += 1
    return np.array(opts) if opts else np.array([0])


def gate(label, jobs, layout_name, grid_size, offset=4000,
         slots=DEFAULT_SLOTS, accel=DEFAULT_ARRIVAL_ACCEL):
    gpus = ["H100"] if layout_name == "homogeneous" else ["H100", "A100"]
    lay = None if layout_name == "homogeneous" else make_checkerboard_layout(
        "H100", "A100", grid_size=grid_size)
    fuser = WorkloadFuser(gpu_models=gpus, seed=42)
    df = fuser.generate_fused_dataset(num_requests=jobs, source="azure", workload="conv",
                                      offset=offset, arrival_accel=accel)

    scheds = [RoundRobinScheduler(), LeastOccupiedScheduler(), ThermalGreedyScheduler()]
    try:
        from src.baselines.best_fit import BestFitScheduler
        scheds.append(BestFitScheduler())
    except Exception:
        pass

    res = {s.name: run_episode(s, df, gpu_model="H100", grid_size=grid_size,
                               gpu_layout_matrix=lay, max_jobs_per_cell=slots)
           for s in scheds}
    names = list(res)

    print("\n" + "=" * 80)
    print(f"  {label}")
    print(f"  jobs={jobs}  grid={grid_size}x{grid_size}  layout={layout_name}  "
          f"slots/cell={slots}  accel={accel}x  rows {offset}-{offset + jobs}")
    print("=" * 80)
    print(f"  {'metric':22s}" + "".join(f"{n[:13]:>15s}" for n in names) + f"{'SPREAD':>10s}")
    print("  " + "-" * (22 + 15 * len(names) + 10))

    def row(key, lab):
        vals = [res[n][key] for n in names]
        hi, lo = max(vals), min(vals)
        sp = 0.0 if hi == 0 else (hi - lo) / abs(hi)
        print(f"  {lab:22s}" + "".join(f"{v:15.3f}" for v in vals)
              + f"{sp * 100:9.1f}%" + ("  <--" if sp >= MIN_SEPARATION else ""))
        return sp

    print("  CONSEQUENTIAL  (a gap here is a reportable result)")
    cons = [row(k, l) for k, l in CONSEQUENTIAL]
    print("  DESCRIPTIVE    (a gap here alone proves nothing)")
    desc = [row(k, l) for k, l in DESCRIPTIVE]

    opts = measure_branching(df, grid_size, lay, slots)
    mean_opts = float(opts.mean())
    frac_stuck = float((opts <= 1).mean())
    best = max(cons)

    rich = mean_opts >= MIN_MEAN_OPTIONS and frac_stuck <= MAX_FRAC_STUCK
    separated = best >= MIN_SEPARATION
    verdict = "PASS" if (rich and separated) else "FAIL"

    print()
    ok_b = "OK " if rich else "BAD"
    ok_s = "OK " if separated else "BAD"
    print(f"  [{ok_b}] BRANCHING   mean {mean_opts:.2f} legal cells/decision "
          f"(need >= {MIN_MEAN_OPTIONS:.0f}), "
          f"<=1 option on {frac_stuck * 100:.1f}% (need <= {MAX_FRAC_STUCK * 100:.0f}%)")
    print(f"  [{ok_s}] SEPARATION  best consequential spread {best * 100:.1f}% "
          f"(need >= {MIN_SEPARATION * 100:.0f}%)")
    print(f"\n  ->  GATE {verdict}")

    if not rich:
        print("      Action space is collapsed -- too few legal cells per decision.")
        print("      Raise --slots or lower --accel.")
    if not separated:
        print(f"      Heuristics agree where it counts (best descriptive spread was "
              f"{max(desc) * 100:.1f}%, but nothing depends on it).")
        print("      Raise --accel or lower --slots.")
    return best, mean_opts, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=500)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT,
                    help="Grid layout. Defaults to the experiment layout; pass "
                         "homogeneous to see the collapsed-signal contrast.")
    ap.add_argument("--offset", type=int, default=4000)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--accel", type=float, default=DEFAULT_ARRIVAL_ACCEL)
    a = ap.parse_args()
    gate("SEPARATION GATE", a.jobs, a.layout, a.grid, a.offset, a.slots, a.accel)
