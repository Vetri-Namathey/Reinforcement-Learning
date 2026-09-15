# Spatial-Phase RL Scheduler for LLM Inference

> A Reinforcement Learning agent that acts as a spatial traffic controller for LLM inference workloads across a 6×6 GPU grid, maximizing throughput while preventing thermal hotspots via Laplacian heat diffusion — on homogeneous or mixed H100/A100/RTX6000 hardware.

> **See [`CHANGES.md`](./CHANGES.md) for a detailed breakdown of all recent implementation changes and new features.**

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
│  │   (Software Layer)  │       │   (Hardware Physics Layer)  │       │
│  │                     │       │                             │       │
│  │  • arrival_time     │       │  H100: 700W TDP, 87°C thr.  │       │
│  │  • prefill_tokens   │──────▶│  A100: 400W TDP, 85°C thr. │       │
│  │  • decode_tokens    │ FUSE  │  W/token, ms/token ranges   │       │
│  │                     │       │  per phase, per device      │       │
│  └─────────────────────┘       └─────────────────────────────┘       │
│                                         │                            │
│                                         ▼                            │
│                    ┌────────────────────────────────────┐            │
│                    │   Fused Job DataFrame (deferred)   │            │
│                    │                                    │            │
│                    │  job_id, arrival_time              │            │
│                    │  prefill_tokens, decode_tokens     │            │
│                    │  prefill_watt_frac, decode_watt_frac│  <- token │
│                    │  prefill_lat_frac,  decode_lat_frac│     counts │
│                    │  (+ reference *_watts/*_duration   │     + per  │
│                    │     columns, diagnostics only)     │     -job   │
│                    └────────────────────────────────────┘  fractions │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼   watts/durations resolved per (job, cell)
┌───────────────────────────────────────────────────────────────────────┐
│                    THERMAL GRID ENVIRONMENT (Gymnasium)               │
│                                                                       │
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
│  │  ├────┼────┼────┼────┼────┼────┤   Action Space:             │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • Discrete(37)            │     │
│  │  ├────┼────┼────┼────┼────┼────┤     (36 GPU cells + Wait)   │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   Physics Engine:           │     │
│  │  └────┴────┴────┴────┴────┴────┘   • 2D Laplacian Diffusion  │     │
│  │                                    • Per-cell Newton Cooling │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Persistent backlog queue on real Azure arrival timestamps —          │
│  jobs wait when the grid is busy and are NEVER dropped.               │
│                                                                       │
│  Reward= +10 (placed) − 0.1·Σ queue_wait − 0.5·Σ max(0, T−T_thr[i,j])²│
└───────────────────────────────────────────────────────────────────────┘
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
| R_sla      | 0.1 × Σ queue wait times (s)                    | Latency pressure — queued jobs hurt every step |
| R_thermal  | 0.5 × Σ max(0, T[i,j] − T_throttle[i,j])²      | Quadratic penalty, own device's threshold      |

`T_throttle[i,j]` is per-cell (87 °C on H100, 85 °C on A100 by default). Pass `throttle_temp_override=80.0` to `ThermalGridEnv` to pin a uniform threshold for A/B comparison against older results.

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
│   └── local_gpu_trace.csv       # Captured local GPU telemetry (gitignored, hardware-specific)
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
│       └── run_baselines.py      #   Unified benchmark harness: 8 core + 5 instrumentation metrics
├── tests/                         # Module 5: Test suite & hardware telemetry
│   ├── system_architecture_test.py   # Full integration/physics suite (pytest-collectible)
│   ├── gpu_profiler.py            #   NVML real-time hardware telemetry capture
│   └── gpu_plot.py                #   Power vs. temperature visualization
├── CHANGES.md                     # Detailed explanation of all implementation changes
└── artifacts/                     # Generated outputs (gitignored except .gitkeep)
    ├── models/                    #   spatial_thermal_ppo*.zip, vec_normalize*.pkl
    ├── logs/                      #   TensorBoard event files (one dir per variant)
    ├── diagnostics/               #   Extracted PPO diagnostics CSVs
    ├── plots/                     #   01_masking_ablation, 02_reward_ablation, 03_ppo_diagnostics
    └── report/                    #   Auto-generated evaluation markdown and plots
```

Every path in the codebase is anchored to `configs.PROJECT_ROOT` (resolved via `pathlib.Path(__file__)`, not the process's working directory), so every entry point below works identically regardless of where it's invoked from.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full integration/physics test suite (auto-downloads the real Azure trace on first run)
pytest
# — or, as a standalone diagnostic script with the same assertions —
python tests/system_architecture_test.py

# Run all three baseline schedulers on the real Azure trace
# Now reports 8 core metrics + 5 instrumentation metrics (power, queue depth, placement spread)
python -m src.baselines.run_baselines --jobs 1000
python -m src.baselines.run_baselines --workload code --gpu-model A100

# Train MaskablePPO (homogeneous H100 grid by default)
python -m src.models.train_agent --timesteps 100000 --eval

# Train on a mixed H100/A100 grid
python -m src.models.train_agent --timesteps 100000 --layout checkerboard --eval

# Masking ablation: MaskablePPO (structural masking) vs vanilla PPO (penalty-based)
python -m src.models.train_agent --timesteps 100000 --use-masking --eval
python -m src.models.train_agent --timesteps 100000 --no-masking --eval

# Reward function ablation (4 variants)
python -m src.models.train_agent --timesteps 100000 --reward-variant full --eval
python -m src.models.train_agent --timesteps 100000 --reward-variant no_thermal --eval
python -m src.models.train_agent --timesteps 100000 --reward-variant no_sla --eval
python -m src.models.train_agent --timesteps 100000 --reward-variant linear_thermal --eval

# Force lower throttle threshold to exercise thermal crossings in ablation
python -m src.models.train_agent --timesteps 100000 --throttle-override 72.0 --eval

# Evaluate trained policy on real campus fleet (zero retraining, mixed GPU types)
# 4 A100 + 2 RTX6000, active_cell_mask applied, compared against all 3 baselines
python -m src.models.campus_deployment_eval

# Extract PPO diagnostics from TensorBoard event files to CSV
python -m src.models.extract_diagnostics --logdir artifacts/logs --output artifacts/diagnostics

# Generate ablation comparison plots from extracted CSVs
python -m src.models.plot_ablation_results --diagnostics-dir artifacts/diagnostics --output-dir artifacts/plots

# Generate the full Results Showcase (Markdown + 7 Plots + JSON) automatically
python -m src.models.generate_report --jobs 500
```

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

### Campus Deployment Eval Notes

`campus_deployment_eval.py` loads the saved `spatial_thermal_ppo` checkpoint **without retraining** and runs it through the identical `run_episode()` harness as the baseline schedulers — on a 6-cell active mask shaped like the real campus fleet (4 A100 + 2 RTX6000). This tests policy generalization to an asymmetric, lower-capacity, mixed-GPU grid the agent was never trained on.

### Baselines Instrumentation Notes

`run_baselines.py` now captures per-step power draw and queue depth throughout each episode, reporting:
- **Placement spread** (std-dev of per-cell placement counts) — lower is better load balance
- **Queue depth P50/P95/P99** — latency pressure distribution
- **Mean power draw (W)** — fleet-wide average across the episode

The integration test extracts 100 real Azure requests, fuses them for both H100 and A100, verifies the GPU support guard (rejecting an unsupported grid/dataset pairing and an uncalibrated GPU), builds a checkerboard H100/A100 grid, confirms the same job resolves to different watts/durations depending on the target cell, then runs a 200-step masked random-action simulation and asserts every physical invariant **element-wise** — including that no A100 cell ever exceeds 400 W, which a scalar check against the grid's 700 W maximum would silently miss.

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

**Project Type:** Final-Year B.Tech / M.Tech Capstone Project  
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
