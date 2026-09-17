"""
campus_deployment_eval.py
Runs the already-trained checkpoint + all 3 baselines on a 6-cell
environment shaped exactly like the real campus fleet (4 A100, 2 RTX6000).
No retraining — this tests generalization of the learned policy.
Run: python -m src.models.campus_deployment_eval
"""
import numpy as np
from sb3_contrib import MaskablePPO
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.baselines.run_baselines import run_episode, print_report
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.least_occupied import LeastOccupiedScheduler
from src.baselines.thermal_greedy import ThermalGreedyScheduler
from src.baselines.trained_policy import TrainedPolicyScheduler
from src.models.train_agent import DEFAULT_SLOTS, EVAL_JOBS, EVAL_OFFSET
from configs import MODELS_DIR

if __name__ == "__main__":
    # 6 real cells: 4 A100 + 2 RTX6000, rest of the 6x6 grid inactive
    layout = np.full((6, 6), "A100", dtype=object)
    layout[1, 0] = layout[1, 1] = "RTX6000"

    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0:4] = True   # 4 A100 cells
    mask[1, 0:2] = True   # 2 RTX6000 cells

    fuser = WorkloadFuser(gpu_models=["H100", "A100", "RTX6000"], seed=42)
    # Held-out slice: this is a generalization test, so the traffic must be
    # unseen too -- not just the hardware and topology.
    fused_df = fuser.generate_fused_dataset(
        num_requests=EVAL_JOBS, source="azure", workload="conv", offset=EVAL_OFFSET,
    )
    print(f"Campus deployment eval | trace rows {EVAL_OFFSET}..{EVAL_OFFSET + EVAL_JOBS} "
          f"(held-out) | 6 active cells: 4x A100 + 2x RTX6000")

    model = MaskablePPO.load(str(MODELS_DIR / "spatial_thermal_ppo"))
    schedulers = [
        RoundRobinScheduler(),
        LeastOccupiedScheduler(),
        ThermalGreedyScheduler(),
        TrainedPolicyScheduler(model, name="Spatial-Phase RL (campus deploy)"),
    ]

    all_metrics = {}
    for sched in schedulers:
        all_metrics[sched.name] = run_episode(
            sched, fused_df, gpu_model="A100",
            gpu_layout_matrix=layout, active_cell_mask=mask,
            max_jobs_per_cell=DEFAULT_SLOTS,
        )

    print_report(all_metrics)
