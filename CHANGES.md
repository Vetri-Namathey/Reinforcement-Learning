# Implementation Changes & New Features

> Detailed explanation of every significant code change made across the project modules.  
> Covers **what changed**, **why it changed**, and **how the pieces connect**.

---

## 1. Baselines — Unified Harness & New Instrumentation (`src/baselines/`)

### 1.1 `run_baselines.py` — Major Overhaul

**What changed**

| Before | After |
|---|---|
| Reported only 4 core metrics | Reports **8 core + 5 instrumentation** metrics |
| No action-distribution tracking | Tracks per-cell placement counts + spread (std-dev) |
| No queue-depth telemetry | Records queue length at every step → P50/P95/P99 |
| No power-draw trace | Records mean power draw per step |
| Single print block | Two-table output: Core Metrics + Instrumentation |

**New functions added**

`compute_instrumentation(actions, power_draw_history, queue_lengths, num_cells)` — Collects action distribution (placements per cell), placement spread (std-dev of placements — a proxy for load-balance quality), time-averaged mean power draw, and queue depth percentiles (P50, P95, P99).

`compute_metrics(env, terminal_info)` — Assembles the four core evaluation metrics from env state and terminal info: `p99_ttft`, `sla_goodput`, `sla_attainment_pct`, `tttr`, `mean_spatial_variance`, `jobs_placed`, `peak_temp`, `sim_time`.

`run_episode(scheduler, fused_df, gpu_model, grid_size, gpu_layout_matrix, active_cell_mask)`:
- Accepts `gpu_layout_matrix` and `active_cell_mask` kwargs — enables campus deployment evaluator to reuse the same harness with a real non-square active cell mask.
- Runs instrumentation collectors inline during the episode loop.
- Merges core metrics + instrumentation into one returned dict.

`print_report(all_metrics)` — Prints two aligned tables: Core Evaluation Metrics and Instrumentation.

**CLI additions**

```
--workload   conv | code          (Azure trace variant)
--gpu-model  H100 | A100 | ...   (homogeneous grid GPU)
--grid-size  int                 (grid dimension, default 6)
```

---

### 1.2 Baseline Schedulers — Hardened `select_action` Interface

All three schedulers (`round_robin.py`, `least_occupied.py`, `thermal_greedy.py`) were updated for correctness:

**Fix:** Each scheduler now reads `num_cells = obs["occupancy"].size` directly from the observation to compute the WAIT action index at runtime — no class-level import dependency.

| Scheduler | Strategy | Key obs field used |
|---|---|---|
| `RoundRobinScheduler` | Cycles through valid cells sequentially | `action_mask`, `occupancy` |
| `LeastOccupiedScheduler` | Picks cell with lowest occupancy timer | `action_mask`, `occupancy` |
| `ThermalGreedyScheduler` | Picks coolest available cell | `action_mask`, `temperatures` |

All three gracefully emit the WAIT action when the entire action mask (excluding WAIT) is zero.

---

## 2. Campus Deployment Evaluator (`src/models/campus_deployment_eval.py`) — New File

**Purpose:** Zero-retraining generalization test — runs the trained checkpoint and all 3 baselines on a non-square active cell mask matching the real campus GPU fleet.

**Campus fleet modeled:**

```
Layout: 6x6 grid, only 6 cells active
  Row 0: [A100, A100, A100, A100, -, -]   <- 4 A100 cells
  Row 1: [RTX6000, RTX6000, -, -, -, -]   <- 2 RTX6000 cells
```

**`TrainedPolicyScheduler` wrapper:**

Wraps a trained MaskablePPO model so it runs through `run_episode()` identically to the heuristic baselines. The action mask is passed in full (all 37 elements including the WAIT slot) to `model.predict()`. Slicing off the WAIT element would produce a shape mismatch.

**How it ties in:** `campus_deployment_eval.py` imports `run_episode` and `print_report` from `run_baselines` — no duplicated episode logic. The trained RL policy is compared on the same metric set, same harness, as all three heuristics.

**Run:**
```bash
python -m src.models.campus_deployment_eval
```

---

## 3. RL Training Pipeline (`src/models/train_agent.py`) — Ablation Extensions

### 3.1 `InvalidActionPenaltyWrapper`

Enables vanilla PPO ablation — instead of masking invalid actions structurally, this wrapper lets the policy attempt any action and applies a −100 penalty for bad placements:

```python
INVALID_ACTION_PENALTY = -100.0
# Applied when: action != WAIT and target cell is already occupied
```

### 3.2 `make_env` — Masking/Penalty Branch

```python
if use_masking:
    env = ActionMasker(env, mask_fn)   # MaskablePPO path
else:
    env = InvalidActionPenaltyWrapper(env)  # vanilla PPO ablation path
```

The `enforce_action_mask` flag is forwarded to `ThermalGridEnv` so the environment permits invalid actions to proceed (for penalty-mode) rather than silently clamping them.

### 3.3 Reward Variant Support — 4 Variants

| Variant | Description |
|---|---|
| `full` | Placement bonus − SLA penalty − thermal penalty (quadratic) |
| `no_thermal` | Removes the thermal penalty term entirely |
| `no_sla` | Removes the SLA queue-wait penalty term |
| `linear_thermal` | Uses linear thermal penalty instead of quadratic |

Each variant saves to a separate model file, e.g. `spatial_thermal_ppo_no_thermal.zip`.

### 3.4 `--throttle-override` CLI Flag

Pins the thermal throttle threshold to a uniform °C value across all cells, overriding per-device values. Used to force real thermal crossings at lower temperatures for ablation comparison.

### 3.5 Model Path Naming

```python
model_suffix   = f"_{reward_variant}" if reward_variant != "full" else ""
masking_suffix = "" if use_masking else "_no_masking"
# → e.g. spatial_thermal_ppo_no_thermal_no_masking.zip
```

---

## 4. SpatialCNN Feature Extractor (`src/models/spatial_cnn.py`) — 5-Channel Input

### Channel Layout

| Ch | Source | Normalization | Purpose |
|---|---|---|---|
| 0 | `temperatures` | T / temp_scale | Thermal state of each cell |
| 1 | `power_draw` | P / TDP_REFERENCE_WATTS | Absolute watts (fleet-wide scale) |
| 2 | `occupancy` | busy_s / 60 | Availability |
| 3 | `tdp_normalized` | TDP_cell / TDP_REF | Per-cell hardware capability |
| 4 | `utilization` | P / TDP_cell | Per-cell saturation / throttle risk |

### Why channels 1 and 4 both encode power

They are intentionally **different normalizations**:

- **Ch 1 (absolute watts):** Normalized by the fleet-wide `TDP_REFERENCE_WATTS` (700 W) to preserve absolute magnitude — this is what drives heat generation and conduction to neighbors in the Laplacian model.
- **Ch 4 (utilization):** Normalized per-cell by that cell's own TDP, so a 400 W A100 reads 1.0 (saturated) while a 400 W H100 reads 0.57 (has headroom). This is what the policy needs to assess throttle risk per device.

If both channels used per-cell normalization they would collapse to the same representation, discarding the absolute scale the Laplacian physically depends on.

### CNN Architecture

```
Grid stream (5 x 6 x 6):
    Conv2d(5→32, 3x3, same padding) → ReLU
    Conv2d(32→64, 3x3, same padding) → ReLU
    Flatten → Linear(2304→256) → ReLU
    → 256-dim grid features

Queue stream ([queue_length, head_job_wait_time] / [100, 60]):
    Linear(2→32) → ReLU
    → 32-dim queue features

Final output: concat(grid, queue) → 288-dim
```

The `action_mask` key is deliberately **not** fed into the network — masking is enforced in MaskablePPO's action distribution layer, not learned from observation.

---

## 5. Diagnostics Pipeline (`src/models/extract_diagnostics.py`) — New File

Reads TensorBoard `.tfevents.*` files and exports PPO training metrics to CSV.

**Metrics extracted:**

| TensorBoard key | Meaning |
|---|---|
| `rollout/ep_rew_mean` | Episode reward (learning curve) |
| `rollout/ep_len_mean` | Episode length |
| `train/explained_variance` | Value function quality |
| `train/clip_fraction` | Fraction of updates hitting the PPO clip |
| `train/approx_kl` | Policy divergence (stability) |
| `train/entropy_loss` | Policy entropy (exploration) |

**Run name → experiment key mapping:**

| TensorBoard run dir | Experiment key |
|---|---|
| `MaskablePPO_1/` | `MaskablePPO` (full reward + masking) |
| `PPO_1/` | `PPO` (vanilla PPO, penalty-based) |
| `MaskablePPO_no_thermal/` | `MaskablePPO_no_thermal` |
| `MaskablePPO_no_sla/` | `MaskablePPO_no_sla` |
| `MaskablePPO_linear_thermal/` | `MaskablePPO_linear_thermal` |

Output: one CSV per experiment in `artifacts/diagnostics/`.

```bash
python -m src.models.extract_diagnostics --logdir artifacts/logs --output artifacts/diagnostics
```

---

## 6. Ablation Plotting (`src/models/plot_ablation_results.py`) — New File

Reads CSVs from `extract_diagnostics.py` and generates four plots to `artifacts/plots/`.

| Output file | Content |
|---|---|
| `01_masking_ablation_learning_curve.png` | MaskablePPO vs vanilla PPO learning curves |
| `02_reward_ablation_learning_curves.png` | All 4 reward variants compared |
| `03_ppo_diagnostics_dashboard.png` | 2x2: explained variance + clip (dual-axis), KL divergence, entropy |
| `summary_metrics.csv` | Best reward + final diagnostics per experiment |

```bash
python -m src.models.plot_ablation_results --diagnostics-dir artifacts/diagnostics --output-dir artifacts/plots
```

---

## 7. End-to-End Data Flow

```
Azure LLM Trace CSV
        |
        v
 AzureTraceLoader          configs/hardware_profiles.py
        |                          |
        +------- WorkloadFuser ----+
                      |
                 fused_df (token counts + efficiency fracs — physics deferred)
                      |
                      v
              ThermalGridEnv  <-- physics_engine.py (Laplacian, Newton, inject_heat)
              (Gymnasium)     <-- action_masking.py (compute_action_mask)
                      |
                 Dict observation:
                   temperatures, power_draw, occupancy,
                   tdp_normalized, utilization,
                   action_mask, queue_stats
                      |
           +----------+---------------------------+
           |                                      |
     SpatialCNN (5-ch CNN + queue MLP)     Baseline schedulers
     + MaskablePPO / PPO (ablation)         (RoundRobin / LeastOccupied
           |                                 / ThermalGreedy)
           +----------+---------------------------+
                      |
                 run_episode()   <-- unified harness (run_baselines.py)
                      |
                 print_report()  <-- two-table output
                      |
          campus_deployment_eval.py   <-- reuses harness, real fleet mask
                      |
            extract_diagnostics.py   <-- TensorBoard -> CSV
                      |
            plot_ablation_results.py <-- CSV -> PNG plots
```

---

## 8. Key Invariants

| Invariant | Where enforced |
|---|---|
| Physics resolved at placement time | `ThermalGridEnv.step()` → `resolve_job_physics(job, cell_profile)` |
| Action mask shape = NUM_CELLS + 1 (37) | All schedulers pass full mask to `model.predict()` |
| Per-cell throttle thresholds | `THROTTLE_TEMP_GRID` is a (6,6) matrix |
| Queue never drops jobs | WAIT action + action masking; SLA penalty accrues |
| Fleet-wide TDP reference for power normalization | `SpatialCNN` ch 1 uses `TDP_REFERENCE_WATTS` = 700 W |
