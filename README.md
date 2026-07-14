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
│  │  ├────┼────┼────┼────┼────┼────┤                             │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   Action Space:             │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • Discrete(36)            │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│     (select GPU cell)      │     │
│  │  ├────┼────┼────┼────┼────┼────┤                             │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   Physics Engine:           │     │
│  │  ├────┼────┼────┼────┼────┼────┤   • 2D Laplacian Diffusion  │     │
│  │  │ 25°│ 25°│ 25°│ 25°│ 25°│ 25°│   • Ambient Cooling (25°C) │     │
│  │  └────┴────┴────┴────┴────┴────┘   • Power → ΔT Conversion   │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Reward = +10 (valid) − 100 (overlap) − α·Σ max(0, T − 80)²        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## The Fused Dataset Pipeline

The system cannot rely on a single data source. We **fuse** two orthogonal signal layers:

### Layer 1 — Software Traffic (Azure LLM Inference Trace)

Microsoft's [Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset) provides production-scale LLM request patterns:
- **Arrival timestamps** — inter-arrival distributions for realistic traffic modeling
- **Prefill token counts** — input/prompt lengths (determines compute intensity)
- **Decode token counts** — output/generation lengths (determines memory-bandwidth pressure)

### Layer 2 — Hardware Physics (MLPerf / NVIDIA Telemetry)

MLPerf Inference benchmarks and NVIDIA's published H100 power profiles provide the mathematical constants to convert token counts into physical quantities:

| Constant                   | Value (Mock H100)       | Source            |
|----------------------------|-------------------------|-------------------|
| Prefill watts per token    | 0.30 – 0.55 W/token    | MLPerf Inference  |
| Decode watts per token     | 0.10 – 0.20 W/token    | MLPerf Inference  |
| Prefill latency per token  | 0.015 – 0.025 ms/token | NVIDIA Triton     |
| Decode latency per token   | 0.020 – 0.035 ms/token | NVIDIA Triton     |
| Idle power draw            | 50 W                   | NVIDIA SMI        |
| Ambient temperature        | 25.0 °C                | Data center spec  |

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

### Reward Signal

The RL agent receives a **composite reward** per step:

| Component        | Value       | Purpose                                     |
|------------------|-------------|---------------------------------------------|
| Valid placement  | +10         | Encourage scheduling over idling            |
| Job overlap      | −100        | Penalize placing on an occupied GPU         |
| Thermal penalty  | −α·(T−80)² | Quadratic penalty for cells above throttle  |

---

## Project Structure

```
RL/
├── README.md                        # This file — system architecture documentation
├── data_fusion_pipeline.py          # Fused Dataset: Azure traffic × MLPerf physics
├── thermal_grid_env.py              # Gymnasium Env: 6×6 spatial thermodynamic grid
├── system_architecture_test.py      # Integration test: proves end-to-end data flow
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

# Run the integration test
python system_architecture_test.py
```

The test generates 100 fused jobs, feeds them into the thermal grid, and runs 50 random-action steps to verify the full pipeline: Azure tokens → MLPerf watts → Grid physics → Temperature updates.

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
