"""
campus_deployment_eval.py
Generalization test for the trained checkpoint: no retraining, changed world.

The campus fleet differs from the training fleet along three axes at once, and
scoring all three together gives one number that cannot be attributed to any of
them. This script therefore runs them as a ladder of ablations, each adding one
shift to the previous:

    baseline   36-cell H100/A100 checkerboard, candidate_k=6   (training world)
    hardware   36-cell H100/RTX6000 checkerboard               (+ unseen device)
    topology   6 active cells of a H100/A100 grid              (+ unseen layout)
    campus     6 active cells, 4x A100 + 2x RTX6000            (both shifts)

Why the hardware axis is the interesting one: SpatialCNN's `tdp_normalized`
channel is TDP_cell / 700. Across every training run it took exactly two values,
1.0 (H100) and 0.571 (A100). RTX6000 presents 300/700 = 0.429 -- a third value on
an input the network has only ever seen as a near-binary indicator. The hardware
rung isolates whether that alone degrades the policy.

candidate_k is held at the trained value on every rung. Leaving it unset gave the
policy a 36-wide action space it was never optimized over, which is a fourth,
silent shift -- see the note on run_baselines.run_episode.

Run: python -m src.models.campus_deployment_eval
     python -m src.models.campus_deployment_eval --stage campus
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from configs import MODELS_DIR
from src.baselines.run_baselines import (
    all_schedulers,
    print_report,
    resolve_candidate_k,
    run_episode,
)
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import make_checkerboard_layout
from src.models.train_agent import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SLOTS,
    EVAL_JOBS,
    EVAL_OFFSET,
)

GRID = 6


def campus_layout():
    """The real fleet: 4 A100 + 2 RTX6000, the other 30 cells inactive."""
    layout = np.full((GRID, GRID), "A100", dtype=object)
    layout[1, 0] = layout[1, 1] = "RTX6000"
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[0, 0:4] = True   # 4 A100 cells
    mask[1, 0:2] = True   # 2 RTX6000 cells
    return layout, mask


def small_mask():
    """Six active cells in the same positions, on a full training-style grid.

    Isolates the topology shift: same devices the agent trained on, but only six
    of the thirty-six cells are usable.
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[0, 0:4] = True
    mask[1, 0:2] = True
    return mask


def stages():
    """(name, description, layout, active_mask, fleet) for each rung."""
    campus_lay, campus_mask = campus_layout()
    return {
        "baseline": (
            "training world (36 cells, H100/A100 checkerboard)",
            make_checkerboard_layout("H100", "A100", grid_size=GRID), None,
            ["H100", "A100"]),
        "hardware": (
            "+ unseen device (36 cells, H100/RTX6000 checkerboard)",
            make_checkerboard_layout("H100", "RTX6000", grid_size=GRID), None,
            ["H100", "RTX6000"]),
        "topology": (
            "+ unseen topology (6 of 36 cells active, H100/A100)",
            make_checkerboard_layout("H100", "A100", grid_size=GRID), small_mask(),
            ["H100", "A100"]),
        "campus": (
            "both shifts (6 cells: 4x A100 + 2x RTX6000)",
            campus_lay, campus_mask, ["H100", "A100", "RTX6000"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Campus generalization ablation ladder")
    ap.add_argument("--stage", choices=[*stages(), "all"], default="all",
                    help="Which rung to run (default: all, in ladder order).")
    ap.add_argument("--jobs", type=int, default=EVAL_JOBS,
                    help=f"Jobs on the full 36-cell grid (default: {EVAL_JOBS}). Rungs "
                         "with fewer active cells scale this down unless --no-scale-jobs.")
    ap.add_argument("--no-scale-jobs", action="store_true",
                    help="Send the full job count to every rung. That makes the "
                         "reduced-topology rungs a 6x overload rather than a topology "
                         "test, and every policy collapses to the same numbers.")
    ap.add_argument("--offset", type=int, default=EVAL_OFFSET)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                    help=f"Held at the trained value ({DEFAULT_CANDIDATE_K}) by default. "
                         "Pass 0 to reproduce the unrestricted-action-space runs, "
                         "which are not comparable to how the agent was trained.")
    args = ap.parse_args()
    candidate_k = resolve_candidate_k(args.candidate_k)

    model = MaskablePPO.load(str(MODELS_DIR / "spatial_thermal_ppo"))
    chosen = list(stages()) if args.stage == "all" else [args.stage]

    print(f"Campus generalization ladder | trace from row {args.offset} (held-out) | "
          f"slots/cell={args.slots} | candidate_k="
          f"{candidate_k if candidate_k else 'unrestricted'}")
    if candidate_k != DEFAULT_CANDIDATE_K:
        print(f"  [!!] candidate_k != {DEFAULT_CANDIDATE_K}: the agent is scored on a "
              "different action space than it was trained on.")

    for name in chosen:
        desc, layout, active_mask, fleet = stages()[name]
        n_active = int(active_mask.sum()) if active_mask is not None else GRID * GRID
        # Offered load must be held per-cell, not per-fleet. Sending 500 jobs to
        # 6 cells is the same traffic on a sixth of the hardware: a 6x load
        # increase riding along with the topology change, which saturates the
        # grid and makes every policy score identically (P99 ~415 s, SLA ~5%).
        # That is a third confound in a test whose whole point is isolating one
        # shift at a time.
        n_jobs = (args.jobs if args.no_scale_jobs
                  else max(50, round(args.jobs * n_active / (GRID * GRID))))

        # Traffic must be unseen too, so physics is resolved against this
        # rung's own fleet on the held-out slice.
        fuser = WorkloadFuser(gpu_models=fleet, seed=42)
        fused_df = fuser.generate_fused_dataset(
            num_requests=n_jobs, source="azure", workload="conv", offset=args.offset,
        )
        print(f"\n{'=' * 78}\n  STAGE '{name}' -- {desc}\n"
              f"  fleet={'/'.join(fleet)}  active cells={n_active}  jobs={n_jobs}"
              f"  ({n_jobs / n_active:.1f} jobs/cell)\n{'=' * 78}")

        schedulers = all_schedulers(model, rl_name=f"Spatial-Phase RL ({name})")
        all_metrics = {
            s.name: run_episode(
                s, fused_df, gpu_model=fleet[0], grid_size=GRID,
                gpu_layout_matrix=layout, active_cell_mask=active_mask,
                max_jobs_per_cell=args.slots, candidate_k=candidate_k,
            )
            for s in schedulers
        }
        print_report(all_metrics)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
