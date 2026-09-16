# Spatial-Phase RL Scheduler for LLM Inference

> A Reinforcement Learning agent that acts as a spatial traffic controller for LLM inference workloads across a 6×6 GPU grid, maximizing throughput while preventing thermal hotspots via Laplacian heat diffusion — on homogeneous or mixed H100/A100/RTX6000 hardware.

---

## Problem Statement

Large Language Model (LLM) inference creates severe **thermal stacking** in data centers due to an inherent workload asymmetry:

| Phase    | Compute Profile         | Thermal Signature          |
|----------|-------------------------|----------------------------|
| Prefill  | Compute-bound (FLOP-heavy) | 200–700 W — approaches GPU TDP |
| Decode   | Memory-bound (autoregressive) | 70–350 W — cooler, latency-sensitive |

Current schedulers (round-robin, least-loaded) are **spatially blind**. They have no awareness of the physical topology of a GPU cluster, causing localized thermal hotspots that trigger:
- Hardware thermal throttling (P-state downclocking)
- Cascading SLA violations (tail-latency spikes)
- Reduced hardware lifespan from sustained thermal stress

## Solution Architecture

We train a **MaskablePPO** agent (`sb3-contrib`) with a custom CNN feature extractor that observes the full thermal state of a 6×6 physical GPU grid and learns a placement policy:

```
Incoming LLM Query → RL Agent → Place on GPU (row, col) → Thermal Physics Update → Reward Signal
```

The agent learns to **spatially distribute** prefill-heavy (hot) and decode-heavy (cool) workloads to prevent any cell from exceeding its device's throttling threshold. The grid may be **homogeneous** (all H100, or all A100) or **heterogeneous** (a mix of both, in any layout) — every physical constant that differs by device (TDP, idle power, throttle point, thermal resistance) is a per-cell matrix, not a global scalar.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FUSED DATASET PIPELINE                        │
│                                                                      │
│  ┌─────────────────────┐       ┌─────────────────────────────┐       │
│  │   Azure LLM Trace   │       │   GPU Hardware Profiles     │       │
│  │   (Software Layer)   │       │   (Hardware Physics Layer)  │       │
│  │                     │       │                             │       │
│  │  • arrival_time     │       │  H100: 700W TDP, 87°C thr.  │       │
│  │  • prefill_tokens   │──────▶│  A100: 400W TDP, 85°C thr.  │       │
│  │  • decode_tokens    │ FUSE  │  W/token, ms/token ranges   │       │
│  │                     │       │  per phase, per device      │       │
│  └─────────────────────┘       └─────────────────────────────┘       │
│                                         │                            │
│                                         ▼                            │
│                    ┌────────────────────────────────────┐            │
│                    │   Fused Job DataFrame (deferred)     │            │
│                    │                                      │            │
│                    │  job_id, arrival_time                │            │
│                    │  prefill_tokens, decode_tokens        │            │
│                    │  prefill_watt_frac, decode_watt_frac  │  <- token │
│                    │  prefill_lat_frac,  decode_lat_frac   │     counts│
│                    │  (+ reference *_watts/*_duration       │     + per│
│                    │     columns, diagnostics only)         │     -job │
│                    └────────────────────────────────────┘   fractions │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼   watts/durations resolved per (job, cell)
┌──────────────────────────────────────────────────────────────────────┐
│                    THERMAL GRID ENVIRONMENT (Gymnasium)               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  6×6 GPU Grid  (homogeneous or mixed H100/A100)              │     │
│  │                                                              │     │
│  │  ┌────┬────┬────┬────┬────┬────┐   Observation (Dict):       │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • temperatures   [6×6]    │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • power_draw     [6×6]    │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • occupancy      [6×6]    │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • tdp_normalized [6×6]    │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • utilization    [6×6]    │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • action_mask    [37]     │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • queue_stats    [2]      │     │
│  │  ├────┼────┼────┼────┼────┼────┤   Action Space:              │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • Discrete(37)             │     │
│  │  ├────┼────┼────┼────┼────┼────┤     (36 GPU cells + Wait)   │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   Physics Engine:            │     │
│  │  └────┴────┴────┴────┴────┴────┘   • 2D Laplacian Diffusion   │     │
│  │                                     • Per-cell Newton Cooling │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Persistent backlog queue on real Azure arrival timestamps —        │
│  jobs wait when the grid is busy and are NEVER dropped.             │
│                                                                      │
│  Reward = +10 (placed) − 0.1·queue_len − 0.5·Σ max(0, T−T_thr[i,j])²  │
└──────────────────────────────────────────────────────────────────────┘
```

`tdp_normalized` and `utilization` exist specifically for heterogeneity: without them, a cell drawing 400 W is indistinguishable from any other, even though that's 57% of an H100's ceiling but 100% of an A100's.

---

## The Fused Dataset Pipeline

The system cannot rely on a single data source. We **fuse** two orthogonal signal layers, and resolve them **at placement time**, not at fusion time.

### Layer 1 — Software Traffic (Azure LLM Inference Trace 2023 — REAL DATA)

Microsoft's [Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) provides production-scale LLM request patterns. The pipeline downloads the **real trace CSVs** into `data/AzurePublicDataset/` on first use (via `AzureTraceLoader`):

| Trace file | Workload | Requests | Span |
|---|---|---|---|
| `AzureLLMInferenceTrace_conv.csv` | Conversational chat | 19,366 | ~1 hour |
| `AzureLLMInferenceTrace_code.csv` | Coding copilot | 8,819 | ~1 hour |

Raw schema → normalized schema:
- `TIMESTAMP` → **arrival_time** (seconds since trace start)
- `ContextTokens` → **prefill_tokens** (prompt length, determines compute intensity)
- `GeneratedTokens` → **decode_tokens** (generation length, determines memory-bandwidth pressure)

A synthetic Azure-like generator remains as an offline fallback (`source="synthetic"`).

### Layer 2 — Hardware Physics (GPU Hardware Profile Registry)

Hardware physics constants live in `configs/hardware_profiles.py` (`GPU_HARDWARE_PROFILES`), keyed by **GPU model**, calibrated from MLPerf Inference power results and the Splitwise (ISCA 2024) phase-split power measurements:

| Constant                   | H100 Profile            | A100 Profile            |
|----------------------------|-------------------------|-------------------------|
| Prefill watts per token    | 0.30 – 0.55 W/token    | 0.15 – 0.30 W/token    |
| Decode watts per token     | 0.50 – 1.00 W/token    | 0.25 – 0.50 W/token    |
| Prefill power envelope     | 200 – 700 W (clamp)    | 120 – 400 W (clamp)    |
| Decode power envelope      | 100 – 350 W (clamp)    | 70 – 220 W (clamp)     |
| Prefill latency per token  | 0.25 – 0.45 ms/token   | 0.50 – 0.90 ms/token   |
| Decode latency per token   | 15 – 30 ms/token       | 30 – 60 ms/token       |
| Idle power draw            | 76 W                   | 55 W                   |
| TDP                        | 700 W                  | 400 W                  |
| Throttle temperature*      | 87 °C                  | 85 °C                  |
| Power → heat coefficient   | 0.020 °C/(W·s)          | 0.035 °C/(W·s)          |

\* Approximate vendor thermal-slowdown points — verify against the current SXM datasheet before citing in published results; they directly scale the thermal reward penalty.

Ambient temperature is fixed at 25.0 °C (data-center inlet spec).

### Deferred Physics

A job's wattage and duration are not properties of the job alone — they depend on **which device it lands on**, which isn't known until the scheduler places it. So the fused dataset carries token counts plus four hardware-agnostic **efficiency fractions** in `[0, 1]` (`prefill_watt_frac`, `decode_watt_frac`, `prefill_lat_frac`, `decode_lat_frac`) instead of pre-computed watts:

```python
# configs/hardware_profiles.resolve_job_physics(job, profile) — called at placement time
watts    = tokens × lerp(profile.watts_per_token_range, frac) + idle_power   # clamped
duration = tokens × lerp(profile.latency_per_token_range, frac)
```

The same job resolves differently depending on where it's placed:

```python
>>> resolve_job_physics(job, get_hardware_profile("H100"))
{'prefill_watts': 260.6, 'prefill_duration': 0.150, ...}
>>> resolve_job_physics(job, get_hardware_profile("A100"))
{'prefill_watts': 154.5, 'prefill_duration': 0.300, ...}
```

The fused frame's `*_watts`/`*_duration` columns are reference values on the trace's primary GPU — diagnostics only. `ThermalGridEnv.step()` ignores them and calls `resolve_job_physics()` against the target cell's own profile.

### GPU Support Guard

Every GPU model present on the grid must (a) have a calibrated profile and (b) be declared supported by the fused dataset (`validate_grid_support`, set-based — not the old rigid single-string equality). This is what makes a **mixed H100/A100 grid** possible while still rejecting a grid populated with hardware the dataset was never prepared for:

1. `AzureTraceLoader` tags every trace with `attrs["gpu_model"]` (its serving fleet).
2. `WorkloadFuser(gpu_models=[...])` accepts one or several GPU models; `apply_hardware_physics()` raises `GPUMismatchError` if the trace's serving fleet isn't among them.
3. `ThermalGridEnv(fused_dataset, gpu_layout_matrix=...)` raises `GPUMismatchError` if any device on the grid isn't in the dataset's `supported_gpus` set.

---

## 6×6 Spatial Thermodynamic Grid

The environment models a 6×6 array of GPUs — homogeneous or mixed — as a 2D thermal grid.

### Heat Diffusion Model

We apply a **discrete 2D Laplacian** at each timestep to simulate conductive heat transfer between adjacent GPUs, plus per-cell power injection and per-cell Newton cooling:

```
∂T/∂t = η[i,j] · P[i,j] + α · ∇²T − γ · (T − T_ambient)
```

Where:
- `η[i,j]` = per-cell power→heat coefficient (device-specific thermal resistance)
- `P[i,j]` = per-cell power draw (W)
- `α` = thermal diffusivity coefficient — kept scalar even on mixed grids, modeling interface conductance between adjacent packages sharing a chassis
- `∇²T` = discrete Laplacian (4-neighbor stencil: up, down, left, right)
- `γ` = cooling coefficient (rack airflow pulling toward ambient 25 °C)

`src/environment/physics_engine.py` implements each term as an independent, testable pure function (`inject_heat`, `apply_laplacian_diffusion`, `apply_newtonian_cooling`).

### Scheduling Model: Backlog Queue + Action Masking

Jobs are ingested into a **persistent backlog queue** on their real Azure arrival timestamps (simulation clock advances 0.1 s per step). The agent places the head-of-queue job on a GPU cell or takes the **Wait** action (index 36). Invalid choices are prevented structurally by an **action mask** (`src/environment/action_masking.py`): occupied cells are masked out, and all placements are masked when the queue is empty. Jobs are never dropped — they wait and accrue SLA penalty. Feasibility is occupancy-only; there is currently no job-to-device capability check (VRAM, power envelope), so an oversized job can legally land on an under-powered cell and get silently clamped to its wattage envelope — a known gap, not a hidden invariant.

### Reward Signal

`R = R_place − R_sla − R_thermal`, computed per step:

| Component  | Value                                          | Purpose                                        |
|------------|-------------------------------------------------|------------------------------------------------|
| R_place    | +10 if a job was placed                          | Encourage scheduling over idling               |
| R_sla      | 0.1 × queue_length                               | Latency pressure — queued jobs hurt every step |
| R_thermal  | 0.5 × Σ max(0, T[i,j] − T_throttle[i,j])²      | Quadratic penalty, own device's threshold      |

`R_sla` charges queue **occupancy**, not accumulated wait. Summed over an episode
this equals `0.1/dt ×` (total queueing seconds), so the episode penalty is linear
in total delay — the standard Little's-law formulation. Charging
`Σ(now − arrival)` over the queue every step instead double-integrates: a job
waiting *T* seconds costs O(*T*²), which on a 500-job episode reaches ~10⁶ and
drowns the +10 placement term by five orders of magnitude, making "never
schedule anything" the optimal policy.

`T_throttle[i,j]` is per-cell (87 °C on H100, 85 °C on A100 by default). Pass `throttle_temp_override=80.0` to `ThermalGridEnv` to pin a uniform threshold for A/B comparison against older results.

#### ⚠ Known limitation — `R_thermal` is currently inert on H100

A cell reaches steady state at `T = T_ambient + η·P/γ`, so crossing its throttle
point needs a **sustained** draw of `P* = (T_throttle − 25)·γ/η`:

| GPU | η | T_throttle | `P*` needed | decode cap | prefill cap | reachable? |
|---|---|---|---|---|---|---|
| H100 | 0.020 | 87 °C | **434 W** | 350 W | 700 W | prefill bursts only |
| A100 | 0.035 | 85 °C | **240 W** | 220 W | 400 W | prefill bursts only |
| RTX6000 | 0.037 | 89 °C | **242 W** | 180 W | 300 W | prefill bursts only |

No device's *decode* ceiling reaches its own `P*`, and prefill — the only phase
that exceeds it — lasts ~0.35 s median against a thermal time constant of
`1/γ ≈ 7 s`, so it moves the die only a few °C. Measured on 500 held-out jobs:

| Grid | peak T | TTTR |
|---|---|---|
| H100 | 77–78 °C | 0.000 |
| A100 | 83.5–85.6 °C | 0.000–0.002 |
| RTX6000 | 74–75 °C | 0.000 |

**Consequence:** on an H100 grid `R_thermal` is identically 0, so the `full`,
`no_thermal` and `linear_thermal` reward variants are numerically identical and
the reward ablation measures nothing. This is *not* a calibration bug — the
constants match `Clg_Data/physics_validation_results.csv` (γ_fit 0.117–0.140) and
the physics is right. The scenario is simply thermally under-loaded: one job per
cell puts a 700 W-TDP H100 at 15–50 % utilization, where it will never get hot.

Until this is addressed, report thermal results on the **A100** grid (the only
one that crosses its threshold at all), and treat any H100 thermal claim as
unsupported. Root fix is to let a cell host concurrent jobs so power sums — real
GPUs use continuous batching — which is an environment change, not a constant tweak.

---

## Project Structure

```
RL/
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── pytest.ini                    # testpaths = tests
├── .gitignore
├── data/
│   ├── AzurePublicDataset/       # Real Azure trace CSVs (auto-downloaded on first use)
│   │   ├── AzureLLMInferenceTrace_conv.csv
│   │   └── AzureLLMInferenceTrace_code.csv
│   └── local_gpu_trace.csv       # Written by tools/gpu_profiler.py (gitignored)
├── configs/                      # GPU hardware physics + project path constants
│   ├── __init__.py               #   PROJECT_ROOT, DATA_DIR, MODELS_DIR, ... + re-exports
│   └── hardware_profiles.py      #   GPU_HARDWARE_PROFILES, resolve_job_physics, guards
├── src/
│   ├── data_pipeline/            # Module 1: Data ingestion & fusion
│   │   ├── trace_loader.py       #   AzureTraceLoader
│   │   └── workload_fuser.py     #   WorkloadFuser
│   ├── environment/               # Module 2: Gymnasium physics environment
│   │   ├── thermal_env.py        #   ThermalGridEnv + layout helpers
│   │   ├── physics_engine.py     #   Laplacian diffusion, Newton cooling, heat injection
│   │   └── action_masking.py     #   compute_action_mask
│   ├── models/                    # Module 3: RL architecture & training
│   │   ├── spatial_cnn.py        #   SpatialCNN feature extractor (5-channel CNN + queue MLP)
│   │   ├── train_agent.py        #   MaskablePPO/PPO training pipeline + full ablation CLI
│   │   ├── campus_deployment_eval.py # Zero-retraining eval on real campus fleet (6-cell, mixed GPUs)
│   │   ├── extract_diagnostics.py#   TensorBoard scalar extraction -> CSV
│   │   ├── plot_ablation_results.py  # Ablation comparison plots + summary table
│   │   └── generate_report.py    #   Auto-generates full showcase Markdown & plots
│   └── baselines/                 # Module 4: Comparative schedulers
│       ├── round_robin.py        #   Spatially-blind cyclic placement
│       ├── least_occupied.py     #   Load-aware, thermally blind
│       ├── thermal_greedy.py     #   1-step greedy coldest-cell picker
│       ├── trained_policy.py     #   Wraps a MaskablePPO checkpoint in the scheduler API
│       └── run_baselines.py      #   Unified benchmark harness: 8 core + 5 instrumentation metrics
├── tests/
│   └── system_architecture_test.py   # Full integration/physics suite (pytest-collectible)
├── tools/                         # Standalone hardware utilities (not part of the RL pipeline)
│   ├── gpu_profiler.py           #   NVML telemetry capture -> data/local_gpu_trace.csv
│   └── gpu_plot.py               #   Plots that capture's power vs. temperature
├── Clg_Data/                      # Campus GPU telemetry & sim-to-real validation (Stage 8)
│   ├── identify_clg_gpus.py      #   Device-ID diagnostic (SM-clock ceiling + correlation)
│   ├── finalize_clg_dataset.py   #   Builds gpu_telemetry_identified.csv from raw exports
│   ├── validate_physics_against_real_gpu.py  # Fits the thermal model to real hardware
│   ├── gpu_telemetry_identified.csv          # Derived: 6 GPUs x 4 metrics (tracked)
│   └── physics_validation_results.csv        # Derived: per-GPU fit results (tracked)
└── artifacts/                     # Generated outputs (gitignored except .gitkeep)
    ├── models/                    #   spatial_thermal_ppo*.zip, vec_normalize*.pkl
    ├── logs/                      #   TensorBoard event files (one dir per variant)
    ├── diagnostics/               #   Extracted PPO diagnostics CSVs
    ├── plots/                     #   Standalone ablation figures
    └── report/                    #   Auto-generated evaluation markdown and plots
```

The 26 raw week-wise Grafana exports under `Clg_Data/` are **gitignored** — bulky
and only consumed by `finalize_clg_dataset.py`. The two derived CSVs are tracked,
so a fresh clone can still re-run the physics validation (Stage 8c) without them.

Every path in the codebase is anchored to `configs.PROJECT_ROOT` (resolved via `pathlib.Path(__file__)`, not the process's working directory), so every entry point below works identically regardless of where it's invoked from.

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

The real Azure trace auto-downloads on first use into `data/AzurePublicDataset/`.

---

### TL;DR — the minimum path to a full set of results

Run these **in order**. Each one depends on the outputs of the one before it.

```bash
pytest                                                              # 1. verify the build      (~1 s)
python -m src.models.train_agent --timesteps 100000 --eval          # 2. train the agent       (~20-40 min)
python -m src.models.extract_diagnostics                            # 3. logs  -> CSVs         (~5 s)
python -m src.models.generate_report --jobs 500                     # 4. CSVs + model -> report (~2 min)
```

Everything lands in `artifacts/report/` — open `artifacts/report/REPORT.md`.

> **Two commands are not enough.** `train_agent` writes TensorBoard event files, not
> CSVs, and `generate_report` reads **CSVs**. If you skip step 3, the report still
> succeeds but silently prints `[!!] No diagnostics CSVs found` and emits only plots
> 01–04. Plots 05 (learning curves), 06 (ablation table) and 07 (PPO diagnostics)
> will be missing.

---

### The pipeline, stage by stage

#### Stage 1 — Verify the build (no model needed)

```bash
pytest                                        # integration + element-wise physics invariants
python tests/system_architecture_test.py      # same assertions, verbose diagnostic output
```

Asserts the physics calibration, per-cell TDP/throttle bounds, the GPU support
guard, and job conservation. **If this fails, stop** — every downstream number is
meaningless. It is the canary for physics-constant regressions.

#### Stage 2 — Baselines only (no model needed)

```bash
python -m src.baselines.run_baselines --jobs 500
python -m src.baselines.run_baselines --workload code --gpu-model A100
```

Runs the three heuristics and prints the 8 core metrics + 5 instrumentation
metrics as a table. Use this to sanity-check the environment **before** spending
GPU-hours on training — if the baselines all score identically here, the reward
has no signal to learn and training will not help.

#### Stage 3 — Train the agent (produces the checkpoint everything else needs)

```bash
python -m src.models.train_agent --timesteps 100000 --eval
```

Writes `artifacts/models/spatial_thermal_ppo.zip` + `vec_normalize.pkl`, and
TensorBoard events to `artifacts/logs/MaskablePPO_<n>/`.

**Train/eval trace split.** Training draws a **fresh random 500-job window out of
trace rows 0–4000 on every episode reset**; `--eval` runs 3 greedy rollouts on a
fixed, disjoint slice at rows **4000–4500**. The constants live at the top of
`train_agent.py` (`TRAIN_POOL_JOBS`, `EPISODE_JOBS`, `EVAL_OFFSET`, `EVAL_JOBS`).

Window randomization matters: with a single fixed window the environment is
fully deterministic, so PPO memorizes one trajectory. The reward curve still
rises, but nothing generalizes and held-out performance collapses. Without
`--eval` you get a checkpoint but no read on whether it generalizes.

Variants:

```bash
python -m src.models.train_agent --timesteps 100000 --layout checkerboard --eval   # mixed H100/A100
python -m src.models.train_agent --timesteps 5000 --eval                           # fast smoke test
```

> Use `--timesteps 5000` only to check the plumbing. It is **not** a training run —
> do not extract diagnostics from it or it will overwrite your real learning curve.

#### Stage 4 — Ablations (optional; each writes its own checkpoint)

```bash
# Reward-term ablation — 4 variants
python -m src.models.train_agent --timesteps 100000 --reward-variant full
python -m src.models.train_agent --timesteps 100000 --reward-variant no_thermal
python -m src.models.train_agent --timesteps 100000 --reward-variant no_sla
python -m src.models.train_agent --timesteps 100000 --reward-variant linear_thermal

# Masking ablation — structural masking vs penalty-based constraint
python -m src.models.train_agent --timesteps 100000 --no-masking
```

`--use-masking` is the **default**; passing it explicitly is a no-op. Only
`--no-masking` changes behaviour (vanilla PPO + `InvalidActionPenaltyWrapper`).

```bash
# Lower the throttle threshold to force thermal crossings (affects training AND eval)
python -m src.models.train_agent --timesteps 100000 --throttle-override 72.0 --eval
```

> Runs using `--throttle-override` are **not** comparable with runs at the real
> per-device thresholds (87 °C H100 / 85 °C A100). Never mix them in one plot,
> and state the override in any result you report from it.

#### Stage 5 — Extract diagnostics (needs Stage 3/4 logs)

```bash
python -m src.models.extract_diagnostics
# explicit form:
python -m src.models.extract_diagnostics --logdir artifacts/logs --output artifacts/diagnostics
```

Converts TensorBoard event files → `artifacts/diagnostics/*.csv`. When several
runs share a variant name the **longest** run wins, ties going to the **newest**,
so a short smoke test cannot clobber a full-length run and a stale run from
before a physics/reward change gets superseded by an equal-length rerun.

> ⚠ **`artifacts/logs/` accumulates every run ever done, including runs from
> before the physics and reward fixes.** The selection rule above handles the
> `MaskablePPO` key automatically, but the four ablation keys
> (`no_thermal`, `no_sla`, `linear_thermal`, `PPO`) currently resolve to 1-point
> smoke runs recorded under the old reward — `PPO` in particular still shows
> `-213800`, produced by the since-fixed `InvalidActionPenaltyWrapper`. Re-run
> Stage 4 before quoting any ablation number. To start clean:
> `rm -rf artifacts/logs artifacts/diagnostics`

#### Stage 6 — Generate the report (needs Stage 3 model + Stage 5 CSVs)

```bash
python -m src.models.generate_report --jobs 500
```

The single flagship command. Runs all 3 baselines **plus** the trained RL agent
through one identical harness and emits to `artifacts/report/`:

| Output | Source | Needs |
|---|---|---|
| `01_baseline_comparison_bar.png` | live episodes | Stage 3 checkpoint |
| `02_placement_heatmaps.png` | live episodes | Stage 3 checkpoint |
| `03_temperature_timeline.png` | live episodes | Stage 3 checkpoint |
| `04_queue_depth_timeline.png` | live episodes | Stage 3 checkpoint |
| `05_learning_curves.png` | diagnostics CSVs | **Stage 5** |
| `06_ablation_summary_table.png` | diagnostics CSVs | **Stage 5** |
| `07_training_diagnostics.png` | diagnostics CSVs | **Stage 5** |
| `results.json`, `REPORT.md` | both | both |

**On `--jobs` and `--offset`:** the report evaluates on trace rows
`offset .. offset + jobs`, defaulting to `--offset 4000` — disjoint from the
0–4000 training pool. `--jobs 100` is a fast smoke run; `--jobs 500` matches the
training episode length and is the number to report. Any `--offset` below 4000
overlaps the training pool; the run still works but prints a warning and records
`"trace_split": "training-overlap"` in `results.json`, so a training-set number
can't be mistaken for a generalization result later.

```bash
python -m src.models.generate_report --jobs 100               # fast check, held-out
python -m src.models.generate_report --jobs 500               # the one to report
python -m src.models.generate_report --jobs 500 --offset 0    # training-set performance (warns)
```

#### Stage 7 — Generalization tests (needs Stage 3 checkpoint)

```bash
python -m src.models.campus_deployment_eval
```

Loads the checkpoint **without retraining** and runs it on a 6-cell active mask
shaped like the real campus fleet (4 A100 + 2 RTX6000) — hardware and topology
the agent never saw. Compared against all 3 baselines through the same harness.

```bash
python -m src.models.plot_ablation_results --diagnostics-dir artifacts/diagnostics --output-dir artifacts/plots
```

Standalone ablation plots into `artifacts/plots/`. Overlaps with report plots
05–07; use it when you want the ablation figures without re-running episodes.

---

### Stage 8 (independent) — Campus GPU telemetry & sim-to-real validation

A **separate pipeline** from the Azure/RL track. It takes real Grafana exports
from the college cluster (6 physical GPUs — 4× A100, 2× RTX6000 — 7 days,
10-minute sampling) and checks the simulator's thermal model against them.
Nothing in Stages 1–7 depends on it; it exists to answer *"are the physics
constants real?"*

Run in order — later stages consume earlier outputs:

```bash
# 8a. (optional, diagnostic) show how device type + file pairing are inferred
python -X utf8 Clg_Data/identify_clg_gpus.py

# 8b. BUILD the labelled dataset   -> Clg_Data/gpu_telemetry_identified.csv
#     Needs the raw Grafana exports, which are gitignored (local only).
python -X utf8 Clg_Data/finalize_clg_dataset.py

# 8c. VALIDATE physics vs reality  -> Clg_Data/physics_validation_results.csv
#     Runs from the tracked derived CSV, so this works in a fresh clone.
python -X utf8 Clg_Data/validate_physics_against_real_gpu.py
```

`-X utf8` is required: the Grafana exports are a mix of UTF-16 TSV and UTF-8 CSV.

#### How it's structured

| Stage | Script | What it does |
|---|---|---|
| 8a | `identify_clg_gpus.py` | Diagnostic. Labels each device by **SM-clock ceiling** (< 1500 MHz → A100, else RTX6000) and cross-correlates power/util against clocks to pair files belonging to the same physical GPU. Prints only. |
| 8b | `finalize_clg_dataset.py` | The builder. Same logic as 8a, plus greedy matching of the 8 temperature files down to 6 GPUs (2 are duplicate re-exports), resamples to 15 min, writes a 24-column wide CSV (6 GPUs × power/clock/util/temp). |
| 8c | `validate_physics_against_real_gpu.py` | Fits the simulator's thermal model to the real power→temperature relationship and reports what the data can and cannot identify. |

The device-ID trick is sound: clock ceiling is a much cleaner discriminator than
peak wattage, and every file parses through an encoding/separator sweep rather
than trusting the filename. Worth keeping in the report.

#### ⚠ What this dataset can and cannot prove

The thermal model is `dT/dt = η·P − γ·(T − T_ambient)`, whose step solution is
`T ← T_eq + (T − T_eq)·exp(−γ·dt)` with `T_eq = T_ambient + (η/γ)·P`.

Telemetry is sampled every **600 s**; the simulator's thermal time constant is
`1/γ ≈ 7 s`. So `exp(−γ·dt) ≈ 1e-55` — **the transient is dead at every sample**.
Each reading is an independent steady state and the model collapses to a line:

> **γ is not identifiable from this data — at any resampling.** Only the ratio
> **η/γ** (°C per watt) is. An earlier version of 8c reported "fitted γ = 0.117–0.140"
> from an optimizer started at γ = 0.14; that was the optimizer sitting still in a
> flat direction, not a measurement. Sweeping γ across four orders of magnitude
> with η/γ held fixed gives a **bit-identical RMSE**. Measuring γ needs sub-minute
> (ideally ~1 s) sampling.

Script 8c now fits the identifiable model with **both** `η/γ` and `T_ambient`
free (the old version hardcoded ambient = 25 °C):

| GPU | η/γ (°C/W) | fitted ambient | R² | RMSE | RMSE @ ambient=25 | mean util |
|---|---|---|---|---|---|---|
| A100_1 | 0.135 | 30.9 °C | 0.900 | **4.25** | 7.27 | 22.0 % |
| A100_2 | 0.215 | 23.6 °C | 0.721 | 5.09 | 5.29 | 1.2 % ⚠ |
| A100_3 | 0.186 | 27.6 °C | 0.702 | 8.39 | 8.78 | 32.5 % |
| A100_4 | 0.121 | 30.3 °C | 0.855 | **5.24** | 7.44 | 34.3 % |
| RTX6000_1 | 0.137 | 35.9 °C | **0.985** | **1.80** | 11.08 | 22.5 % |
| RTX6000_2 | 0.136 | 35.3 °C | 0.608 | 1.50 | 10.43 | 0.3 % ⚠ |

⚠ = low confidence (near-idle: power barely varies, so the slope is poorly conditioned).

**This is a genuinely good result.** RTX6000_1 hits R² = 0.985 and **1.80 °C RMSE**
against a week of real hardware — the simulator's steady-state thermal model is
a strong fit to reality. Note the ambient column: the two RTX6000s sit in a
~35 °C part of the room, and freeing that parameter cut RTX6000_1's error from
11.08 °C to 1.80 °C — a **6× improvement**. Pinning ambient at 25 °C was the
dominant error term, not the model.

#### Consequence: two profile constants are ~2× too high

`η_implied = (η/γ)_measured × γ_sim`, using only high-confidence devices:

| Device | measured η/γ | implied η at γ=0.14 | `hardware_profiles.py` | off by |
|---|---|---|---|---|
| A100 | 0.147 | **0.0206** | 0.0350 | **1.70×** |
| RTX6000 | 0.137 | **0.0192** | 0.0370 | **1.93×** |

Both inflated because the old fit forced a low ambient, tilting the line steeper.
The constants are **deliberately left uncorrected** — lowering η makes the grid
cooler and `R_thermal` even more inert (see the reward-signal limitation above),
so it should be changed together with the thermal-load redesign, not before.
Both entries in `configs/hardware_profiles.py` carry a note saying so.

> **`Clg_Data/` is gitignored** (`.gitignore:33`) and **zero files in it are
> tracked** — the raw telemetry, all five scripts and both derived CSVs exist
> only in your working copy. The whole folder is 1.3 MB. Since
> `hardware_profiles.py` cites `physics_validation_results.csv` as provenance,
> that citation currently points at a file no clone contains. Consider tracking
> at least the scripts and the two derived CSVs.

---

### Which command answers which question?

| Question | Command | Stage |
|---|---|---|
| Is the build sane? | `pytest` | 1 |
| Does the environment have a learnable signal? | `run_baselines --jobs 500` | 2 |
| Does the agent learn? | `train_agent --timesteps 100000 --eval` | 3 |
| Does the reward shaping matter? | `train_agent --reward-variant ...` | 4 |
| Does masking beat penalties? | `train_agent --no-masking` | 4 |
| Did PPO train stably (EV, KL, clip)? | `extract_diagnostics` → plot 07 | 5 |
| **Does RL beat the baselines?** | `generate_report --jobs 500` | 6 |
| Does it transfer to unseen hardware? | `campus_deployment_eval` | 7 |
| Are the physics constants real? | `validate_physics_against_real_gpu` | 8 |

---

### Ablation Study CLI Notes

| Flag | Options | Description |
|---|---|---|
| `--reward-variant` | `full`, `no_thermal`, `no_sla`, `linear_thermal` | Reward function variant |
| `--use-masking` | (flag) | MaskablePPO with structural action masking (default) |
| `--no-masking` | (flag) | Vanilla PPO + `InvalidActionPenaltyWrapper` (−100 per bad placement) |
| `--throttle-override` | float (e.g. 72.0) | Pin uniform throttle threshold across all cells |
| `--layout` | `homogeneous`, `checkerboard`, `split` | Grid hardware layout |
| `--grid-size` | int | Grid dimension (default 6 → 6×6) |

Each ablation variant saves to a uniquely named model file:
```
artifacts/models/
  spatial_thermal_ppo.zip                      ← full reward, masked
  spatial_thermal_ppo_no_thermal.zip
  spatial_thermal_ppo_no_sla.zip
  spatial_thermal_ppo_linear_thermal.zip
  spatial_thermal_ppo_no_masking.zip           ← vanilla PPO ablation
```

### Metric Reference

Reported by `run_baselines.py`, `campus_deployment_eval.py` and `generate_report.py`:

| Metric | Meaning |
|---|---|
| **P99 TTFT (s)** | 99th-percentile queueing delay from arrival to placement. Excludes prefill execution. |
| **SLA goodput / attainment** | Jobs placed within `SLA_TTFT_SECONDS` (5 s), as a rate and as a percentage. |
| **TTTR** | Thermal throttle time ratio — fraction of steps with any cell above its own threshold. |
| **Mean spatial variance (°C²)** | Grid temperature variance averaged over the episode. Lower = more even heat spread. |
| **Peak temp (°C)** | Hottest cell seen at any point in the episode. |
| **Placement spread** | Std-dev of per-cell placement counts. Lower = better load balance. |
| **Queue depth P50/P95/P99** | Backlog distribution — latency pressure. |
| **Mean power draw (W)** | Fleet-wide average across the episode. |

### Programmatic Usage

```python
# Programmatic usage — homogeneous grid
from src.data_pipeline import WorkloadFuser
from src.environment import ThermalGridEnv

fuser = WorkloadFuser(gpu_models="H100", seed=42)
jobs = fuser.generate_fused_dataset(num_requests=500, source="azure", workload="conv")
env = ThermalGridEnv(fused_dataset=jobs, gpu_model="H100")

# Programmatic usage — heterogeneous grid
from src.environment import make_checkerboard_layout

fuser = WorkloadFuser(gpu_models=["H100", "A100"], seed=42)
jobs = fuser.generate_fused_dataset(num_requests=500, source="azure", workload="conv")
env = ThermalGridEnv(fused_dataset=jobs, gpu_layout_matrix=make_checkerboard_layout())
```

---

## Technology Stack

| Component         | Technology                      |
|-------------------|---------------------------------|
| RL Framework      | Gymnasium (Farama Foundation)   |
| RL Algorithm      | MaskablePPO (`sb3-contrib`) with a custom CNN feature extractor (PyTorch) |
| Numerical Compute | NumPy                           |
| Data Pipeline     | Pandas                          |
| Testing           | pytest                          |
| Language          | Python 3.10+                    |

---

## Academic Context

**Project Type:** B.Tech CSE-AI, Semester 7 Capstone Project  
**Domain:** Reinforcement Learning × Systems Architecture × Thermal Management  
**Keywords:** LLM Inference Scheduling, Spatial RL, Thermal-Aware Placement, Laplacian Heat Diffusion, Prefill-Decode Disaggregation, Heterogeneous GPU Scheduling

---

## References

1. Microsoft Azure LLM Inference Trace — [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)
2. MLPerf Inference Benchmark — [mlcommons.org](https://mlcommons.org/benchmarks/inference-datacenter/)
3. NVIDIA H100 Tensor Core GPU Architecture — [NVIDIA Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
4. Gymnasium Documentation — [Farama Foundation](https://gymnasium.farama.org/)
5. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving" (OSDI 2024)
6. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024)

---

## License

This project is developed for academic purposes. All referenced datasets and benchmarks are used under their respective licenses.
