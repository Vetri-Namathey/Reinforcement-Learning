# Spatial-Phase RL Scheduler for LLM Inference

> A Reinforcement Learning agent that acts as a spatial traffic controller for LLM inference workloads across a 6×6 GPU grid, maximizing throughput while preventing thermal hotspots via Laplacian heat diffusion.

---

## Problem Statement

Large Language Model (LLM) inference creates severe **thermal stacking** in data centers due to an inherent workload asymmetry:

| Phase    | Compute Profile         | Thermal Signature          |
|----------|-------------------------|----------------------------|
| Prefill  | Compute-bound (FLOP-heavy) | 400–700 W — approaches GPU TDP |
| Decode   | Memory-bound (autoregressive) | 150–250 W — cooler, latency-sensitive |

Current schedulers (round-robin, least-loaded) are **spatially blind**. They have no awareness of the physical topology of a GPU cluster, causing localized thermal hotspots that trigger:
- Hardware thermal throttling (P-state downclocking)
- Cascading SLA violations (tail-latency spikes)
- Reduced hardware lifespan from sustained thermal stress

## Solution Architecture

We train a **Deep RL agent** (PPO / SAC) that observes the full thermal state of a 6×6 physical GPU grid and learns a placement policy:

```
Incoming LLM Query → RL Agent → Place on GPU (row, col) → Thermal Physics Update → Reward Signal
```

The agent learns to **spatially distribute** prefill-heavy (hot) and decode-heavy (cool) workloads to prevent any cell from exceeding the 80 °C throttling threshold.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FUSED DATASET PIPELINE                        │
│                                                                      │
│  ┌─────────────────────┐       ┌─────────────────────────────┐       │
│  │   Azure LLM Trace   │       │   MLPerf / NVIDIA Telemetry │       │
│  │   (Software Layer)   │       │   (Hardware Physics Layer)  │       │
│  │                     │       │                             │       │
│  │  • arrival_time     │       │  • W/token (prefill)        │       │
│  │  • prefill_tokens   │──────▶│  • W/token (decode)         │       │
│  │  • decode_tokens    │ FUSE  │  • ms/token (prefill)       │       │
│  │                     │       │  • ms/token (decode)        │       │
│  └─────────────────────┘       └─────────────────────────────┘       │
│                                         │                            │
│                                         ▼                            │
│                          ┌──────────────────────────┐                │
│                          │   Fused Job DataFrame     │                │
│                          │                          │                │
│                          │  job_id                  │                │
│                          │  prefill_watts   (400-700W)│               │
│                          │  prefill_duration (ms)    │                │
│                          │  decode_watts    (150-250W)│               │
│                          │  decode_duration  (ms)    │                │
│                          └──────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    THERMAL GRID ENVIRONMENT (Gymnasium)               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  6×6 GPU Grid  (36 × H100 GPUs)                             │     │
│  │                                                              │     │
│  │  ┌────┬────┬────┬────┬────┬────┐   State Space:              │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • temperatures  [6×6]    │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • power_draw    [6×6]    │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • occupancy     [6×6]    │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • action_mask   [37]     │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • queue_stats   [2]      │     │
│  │  ├────┼────┼────┼────┼────┼────┤   Action Space:             │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • Discrete(37)            │     │
│  │  ├────┼────┼────┼────┼────┼────┤     (36 GPU cells + Wait)  │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   Physics Engine:           │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • 2D Laplacian Diffusion  │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • Ambient Cooling (25°C) │     │
│  │  └────┴────┴────┴────┴────┴────┘   • Power → ΔT Conversion   │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Persistent backlog queue on real Azure arrival timestamps —        │
│  jobs wait when the grid is busy and are NEVER dropped.             │
│                                                                      │
│  Reward = +10 (placed) − 0.1·Σ queue_wait − 0.5·Σ max(0, T−80)²   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## The Fused Dataset Pipeline

The system cannot rely on a single data source. We **fuse** two orthogonal signal layers:

### Layer 1 — Software Traffic (Azure LLM Inference Trace 2023 — REAL DATA)

Microsoft's [Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) provides production-scale LLM request patterns. The pipeline downloads the **real trace CSVs** into `data/` on first use (via `AzureTraceLoader`):

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

Hardware physics constants live in a registry (`GPU_HARDWARE_PROFILES`) keyed by **GPU model**, calibrated from MLPerf Inference power results and the Splitwise (ISCA 2024) phase-split power measurements:

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

Ambient temperature is fixed at 25.0 °C (data-center inlet spec).

### GPU-Matching Invariant

**The two layers must describe the same GPU.** The Azure 2023 trace was served on Azure's H100 (DGX-H100) production fleet (per the companion Splitwise paper), so it may only be fused with the H100 hardware profile. The pipeline enforces this end-to-end:

1. `AzureTraceLoader` tags every trace with `attrs["gpu_model"] = "H100"`.
2. `WorkloadFuser.apply_hardware_physics()` raises `GPUMismatchError` if the trace's GPU differs from the hardware profile's GPU (or is untagged).
3. `ThermalGridEnv(fused_dataset, gpu_model=...)` raises `GPUMismatchError` if the fused dataset was built for a different GPU than the grid simulates.

Fusing an H100 traffic trace with A100 (or local consumer-GPU) power constants would produce physically meaningless wattages and durations — the guard makes that a hard error instead of a silent modeling bug.

### The Fusion

```python
fused_job.prefill_watts = azure_trace.prefill_tokens × mlperf.watts_per_prefill_token + idle_power
fused_job.prefill_duration = azure_trace.prefill_tokens × mlperf.latency_per_prefill_token
fused_job.decode_watts  = azure_trace.decode_tokens × mlperf.watts_per_decode_token + idle_power
fused_job.decode_duration  = azure_trace.decode_tokens × mlperf.latency_per_decode_token
```

The output is a single DataFrame where every row is a **physically-grounded LLM inference job** ready for the RL environment.

---

## 6×6 Spatial Thermodynamic Grid

The environment models a 6×6 array of NVIDIA H100 GPUs as a 2D thermal grid.

### Heat Diffusion Model

We apply a **discrete 2D Laplacian** at each timestep to simulate conductive heat transfer between adjacent GPUs:

```
∂T/∂t = α · ∇²T + Q(power) − γ · (T − T_ambient)
```

Where:
- `α` = thermal diffusivity coefficient (heat spreading rate)
- `∇²T` = discrete Laplacian (4-neighbor stencil: up, down, left, right)
- `Q(power)` = heat injection from GPU workload (proportional to wattage)
- `γ` = cooling coefficient (rack airflow pulling toward ambient 25 °C)

### Scheduling Model: Backlog Queue + Action Masking

Jobs are ingested into a **persistent backlog queue** on their real Azure arrival timestamps (simulation clock advances 0.1 s per step). The agent places the head-of-queue job on a GPU cell or takes the **Wait** action (index 36). Invalid choices are prevented structurally by an **action mask** in the observation: occupied cells are masked out, and all placements are masked when the queue is empty. Jobs are never dropped — they wait and accrue SLA penalty.

### Reward Signal

`R = R_place − R_sla − R_thermal`, computed per step:

| Component  | Value                              | Purpose                                        |
|------------|------------------------------------|------------------------------------------------|
| R_place    | +10 if a job was placed            | Encourage scheduling over idling               |
| R_sla      | 0.1 × Σ queue wait times (s)      | Latency pressure — queued jobs hurt every step |
| R_thermal  | 0.5 × Σ max(0, T−80)² over cells | Quadratic penalty for cells above throttle     |

Overlap placements are no longer penalized (−100 removed) because action masking makes them impossible — the flaw of wasting samples on learning "don't stack jobs" is eliminated from the MDP.

---

## Project Structure

```
RL/
├── README.md                        # This file — system architecture documentation
├── data_fusion_pipeline.py          # Real Azure trace loader + GPU-matched fusion
├── thermal_grid_env.py              # Gymnasium Env: 6×6 spatial thermodynamic grid
├── system_architecture_test.py      # Integration test: proves end-to-end data flow
├── data/                            # Real Azure LLM Inference Trace 2023 CSVs
│   ├── AzureLLMInferenceTrace_conv.csv   # (auto-downloaded on first use)
│   └── AzureLLMInferenceTrace_code.csv
└── test/
    ├── gpu_profiler.py              # Local GPU telemetry profiling utilities
    ├── gpu_plot.py                  # Visualization of GPU metrics
    └── local_gpu_trace.csv          # Captured local GPU trace data
```

---

## Quickstart

```bash
# Install dependencies
pip install gymnasium numpy pandas

# Run the integration test (auto-downloads the real Azure trace on first run)
python system_architecture_test.py
```

The test extracts 100 real Azure requests, fuses them with H100 physics, verifies the GPU-matching guard rejects mismatched fusions (H100 trace × A100 profile), then runs 50 random-action steps to verify the full pipeline: Azure tokens → H100 watts → Grid physics → Temperature updates.

```python
# Programmatic usage
from data_fusion_pipeline import WorkloadFuser
from thermal_grid_env import ThermalGridEnv

fuser = WorkloadFuser(gpu_model="H100", seed=42)
jobs = fuser.generate_fused_dataset(num_requests=500, source="azure", workload="conv")
env = ThermalGridEnv(fused_dataset=jobs, gpu_model="H100")  # GPU match enforced
```

---

## Technology Stack

| Component         | Technology                      |
|-------------------|---------------------------------|
| RL Framework      | Gymnasium (Farama Foundation)   |
| RL Algorithms     | Stable-Baselines3 (planned)    |
| Numerical Compute | NumPy                           |
| Data Pipeline     | Pandas                          |
| Language          | Python 3.10+                    |

---

## Academic Context

**Project Type:** Final-Year B.Tech / M.Tech Capstone Project  
**Domain:** Reinforcement Learning × Systems Architecture × Thermal Management  
**Keywords:** LLM Inference Scheduling, Spatial RL, Thermal-Aware Placement, Laplacian Heat Diffusion, Prefill-Decode Disaggregation

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
