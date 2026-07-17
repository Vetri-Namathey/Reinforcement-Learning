"""
data_fusion_pipeline.py
=======================
Fused Dataset Pipeline: Merges the REAL Azure LLM Inference Trace 2023
(software traffic layer) with GPU hardware physics profiles derived from
MLPerf Inference / Splitwise power measurements (hardware physics layer)
to produce physically-grounded LLM inference jobs for the RL environment.

GPU-Matching Invariant
----------------------
The two layers MUST describe the same GPU model. The Azure 2023 trace was
collected from Azure's production LLM inference fleet, whose serving
hardware is the NVIDIA H100 (DGX-H100) platform per the companion paper
"Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
(Patel et al., ISCA 2024). Fusing that trace with hardware constants from
a different GPU (e.g. A100 or a local consumer RTX) would produce
physically meaningless watts/durations, so the pipeline enforces the match
and raises `GPUMismatchError` on violation.

Each output row represents a single LLM inference request with:
  - Prefill wattage and duration (compute-bound, hot phase)
  - Decode wattage and duration (memory-bound, cool phase)
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


class GPUMismatchError(ValueError):
    """Raised when the trace layer and hardware layer describe different GPUs."""


# ══════════════════════════════════════════════════════════════════════
# Layer 2 — GPU Hardware Physics Profiles
# ══════════════════════════════════════════════════════════════════════
# Constants are calibrated from public sources:
#   - Power envelopes:  Splitwise (Patel et al., ISCA 2024) phase-split
#     power measurements; MLPerf Inference v4.0 datacenter power results.
#   - Latency/token:    NVIDIA Triton / TensorRT-LLM published throughput
#     for 70B-class models (per-GPU share).
#   - Idle power / TDP: NVIDIA product specifications (nvidia-smi).
#
# Per-token wattage is a linear surrogate: phase_watts = tokens × W/token
# + idle, clamped to the phase's measured power envelope (saturation).
GPU_HARDWARE_PROFILES = {
    "H100": {
        "gpu_model": "H100",
        "tdp_watts": 700.0,               # H100 SXM TDP
        "idle_power": 76.0,               # nvidia-smi idle, H100 SXM

        # Power surrogate (Watts per token)
        "prefill_watts_per_token": (0.30, 0.55),   # compute-bound, near-TDP
        "decode_watts_per_token":  (0.50, 1.00),   # memory-bound, cooler

        # Measured phase power envelopes (clamp bounds, Watts)
        "prefill_watts_bounds": (200.0, 700.0),
        "decode_watts_bounds":  (100.0, 350.0),

        # Latency per token (seconds)
        "prefill_latency_per_token": (0.25e-3, 0.45e-3),  # ~0.25-0.45 ms/token
        "decode_latency_per_token":  (15e-3, 30e-3),      # ~33-65 tokens/s

        "source": "Splitwise (ISCA 2024) + MLPerf Inference v4.0 H100 power",
    },
    "A100": {
        "gpu_model": "A100",
        "tdp_watts": 400.0,               # A100 SXM TDP
        "idle_power": 55.0,

        "prefill_watts_per_token": (0.15, 0.30),
        "decode_watts_per_token":  (0.25, 0.50),

        "prefill_watts_bounds": (120.0, 400.0),
        "decode_watts_bounds":  (70.0, 220.0),

        # A100 is ~2x slower per token than H100 for LLM inference
        "prefill_latency_per_token": (0.50e-3, 0.90e-3),
        "decode_latency_per_token":  (30e-3, 60e-3),

        "source": "Splitwise (ISCA 2024) A100 phase power measurements",
    },
}


def get_hardware_profile(gpu_model: str) -> dict:
    """Look up a hardware physics profile by GPU model name."""
    key = gpu_model.upper()
    if key not in GPU_HARDWARE_PROFILES:
        raise KeyError(
            f"No hardware profile for GPU '{gpu_model}'. "
            f"Available: {list(GPU_HARDWARE_PROFILES)}"
        )
    return GPU_HARDWARE_PROFILES[key]


# ══════════════════════════════════════════════════════════════════════
# Layer 1 — Real Azure LLM Inference Trace 2023
# ══════════════════════════════════════════════════════════════════════

# GPU model of the fleet the Azure 2023 trace was served on (Splitwise, ISCA 2024)
AZURE_TRACE_GPU_MODEL = "H100"

AZURE_TRACE_BASE_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
)
AZURE_TRACE_FILES = {
    "code": "AzureLLMInferenceTrace_code.csv",   # coding copilot workload
    "conv": "AzureLLMInferenceTrace_conv.csv",   # conversational chat workload
}


class AzureTraceLoader:
    """
    Loads the real Microsoft Azure LLM Inference Trace 2023.

    Trace schema (raw CSV):
        TIMESTAMP        — request arrival wall-clock time
        ContextTokens    — prompt length  → prefill_tokens
        GeneratedTokens  — output length  → decode_tokens

    The loader normalizes arrival times to seconds-since-trace-start and
    tags the resulting DataFrame with the GPU model of the serving fleet
    (df.attrs["gpu_model"]) so downstream fusion can enforce GPU matching.
    """

    def __init__(self, workload: str = "conv", data_dir: str | None = None):
        """
        Args:
            workload: "conv" (chat) or "code" (coding copilot) trace.
            data_dir: Directory holding the trace CSVs. Defaults to
                      <repo>/data next to this file.
        """
        if workload not in AZURE_TRACE_FILES:
            raise ValueError(
                f"Unknown workload '{workload}'. Choose from {list(AZURE_TRACE_FILES)}"
            )
        self.workload = workload
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )
        self.csv_path = os.path.join(self.data_dir, AZURE_TRACE_FILES[workload])
        self.gpu_model = AZURE_TRACE_GPU_MODEL

    def ensure_downloaded(self) -> str:
        """Download the trace CSV from the AzurePublicDataset repo if missing."""
        if not os.path.exists(self.csv_path):
            os.makedirs(self.data_dir, exist_ok=True)
            url = f"{AZURE_TRACE_BASE_URL}/{AZURE_TRACE_FILES[self.workload]}"
            print(f"  Downloading Azure trace: {url}")
            urllib.request.urlretrieve(url, self.csv_path)
        return self.csv_path

    def load(self, max_requests: int | None = None) -> pd.DataFrame:
        """
        Load and normalize the trace.

        Args:
            max_requests: Optionally truncate to the first N requests
                          (chronological order preserved).

        Returns:
            DataFrame with columns [arrival_time, prefill_tokens, decode_tokens],
            arrival_time in seconds from trace start, tagged with
            attrs["gpu_model"] and attrs["source"].
        """
        self.ensure_downloaded()
        raw = pd.read_csv(self.csv_path)

        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Azure trace missing columns: {missing}")

        timestamps = pd.to_datetime(raw["TIMESTAMP"])
        arrival_time = (timestamps - timestamps.min()).dt.total_seconds()

        trace_df = pd.DataFrame({
            "arrival_time":   np.round(arrival_time.values, 4),
            "prefill_tokens": raw["ContextTokens"].astype(np.int64).values,
            "decode_tokens":  raw["GeneratedTokens"].astype(np.int64).values,
        }).sort_values("arrival_time", ignore_index=True)

        if max_requests is not None:
            trace_df = trace_df.head(max_requests).copy()

        trace_df.attrs["gpu_model"] = self.gpu_model
        trace_df.attrs["source"] = f"AzureLLMInferenceTrace2023/{self.workload}"
        return trace_df


# ══════════════════════════════════════════════════════════════════════
# Fusion — Software Traffic × Hardware Physics
# ══════════════════════════════════════════════════════════════════════

class WorkloadFuser:
    """
    Fuses an LLM traffic trace (Layer 1) with GPU hardware physics
    constants (Layer 2) into a unified, physically-grounded job DataFrame.

        Layer 1 (Software):  trace → arrival_time, prefill_tokens, decode_tokens
        Layer 2 (Hardware):  GPU profile → watts/token, latency/token
        Output:              fused job → prefill_watts, prefill_duration,
                                          decode_watts,  decode_duration

    The fuser is bound to one GPU model. Fusion refuses traces collected
    on a different GPU (GPUMismatchError) — watts and latencies are only
    valid for the hardware they were measured on.
    """

    def __init__(self, gpu_model: str = AZURE_TRACE_GPU_MODEL, seed: int = 42):
        """
        Args:
            gpu_model: GPU whose hardware physics profile to fuse with.
            seed: Random seed for reproducible per-job constant sampling.
        """
        self.hardware_profile = get_hardware_profile(gpu_model)
        self.gpu_model = self.hardware_profile["gpu_model"]
        self.rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────────
    # GPU-matching guard
    # ──────────────────────────────────────────────────────────────

    def _validate_gpu_match(self, trace_df: pd.DataFrame) -> None:
        """Ensure the trace's GPU matches this fuser's hardware profile."""
        trace_gpu = trace_df.attrs.get("gpu_model")
        if trace_gpu is None:
            raise GPUMismatchError(
                "Trace DataFrame carries no attrs['gpu_model'] tag — cannot "
                "verify it was collected on the same GPU as the hardware "
                f"profile ('{self.gpu_model}'). Tag the trace before fusing."
            )
        if trace_gpu.upper() != self.gpu_model.upper():
            raise GPUMismatchError(
                f"GPU mismatch: trace was served on '{trace_gpu}' but the "
                f"hardware physics profile is for '{self.gpu_model}'. "
                "Both fusion layers must describe the same GPU."
            )

    # ──────────────────────────────────────────────────────────────
    # Layer 1 fallback: Synthetic Azure-like trace (offline mode)
    # ──────────────────────────────────────────────────────────────

    TRACE_PROFILE = {
        "prefill_tokens_range":  (128, 2048),
        "decode_tokens_range":   (32, 512),
        "arrival_rate_lambda":   0.5,          # Poisson requests/sec
    }

    def generate_azure_trace(self, num_requests: int = 100) -> pd.DataFrame:
        """
        Generate a SYNTHETIC Azure-like trace (offline fallback when the
        real CSV is unavailable). Mimics the real trace's schema and is
        tagged with the same fleet GPU model.
        """
        inter_arrivals = self.rng.exponential(
            scale=1.0 / self.TRACE_PROFILE["arrival_rate_lambda"],
            size=num_requests,
        )
        arrival_times = np.cumsum(inter_arrivals)

        prefill_min, prefill_max = self.TRACE_PROFILE["prefill_tokens_range"]
        decode_min, decode_max = self.TRACE_PROFILE["decode_tokens_range"]

        trace_df = pd.DataFrame({
            "arrival_time":   np.round(arrival_times, 4),
            "prefill_tokens": self.rng.integers(prefill_min, prefill_max, num_requests),
            "decode_tokens":  self.rng.integers(decode_min, decode_max, num_requests),
        })
        trace_df.attrs["gpu_model"] = AZURE_TRACE_GPU_MODEL
        trace_df.attrs["source"] = "synthetic"
        return trace_df

    # ──────────────────────────────────────────────────────────────
    # Layer 2: Hardware Physics Application
    # ──────────────────────────────────────────────────────────────

    def apply_hardware_physics(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map software-layer token counts to hardware-layer physical
        quantities using the GPU-matched hardware profile.

        For each job:
            prefill_watts    = prefill_tokens × W/token + idle   (clamped)
            prefill_duration = prefill_tokens × latency/token
            decode_watts     = decode_tokens  × W/token + idle   (clamped)
            decode_duration  = decode_tokens  × latency/token

        Raises:
            GPUMismatchError: if trace_df was collected on a different GPU.
        """
        self._validate_gpu_match(trace_df)

        n = len(trace_df)
        hw = self.hardware_profile

        # Sample per-job constants (models batching & binning variance)
        prefill_wpt = self.rng.uniform(*hw["prefill_watts_per_token"], size=n)
        decode_wpt  = self.rng.uniform(*hw["decode_watts_per_token"], size=n)
        prefill_lpt = self.rng.uniform(*hw["prefill_latency_per_token"], size=n)
        decode_lpt  = self.rng.uniform(*hw["decode_latency_per_token"], size=n)

        idle = hw["idle_power"]

        # ── Fusion: Software × Hardware ──
        prefill_watts    = trace_df["prefill_tokens"].values * prefill_wpt + idle
        prefill_duration = trace_df["prefill_tokens"].values * prefill_lpt
        decode_watts     = trace_df["decode_tokens"].values * decode_wpt + idle
        decode_duration  = trace_df["decode_tokens"].values * decode_lpt

        # Clamp to the GPU's measured phase power envelopes (saturation)
        prefill_watts = np.clip(prefill_watts, *hw["prefill_watts_bounds"])
        decode_watts  = np.clip(decode_watts, *hw["decode_watts_bounds"])

        fused_df = pd.DataFrame({
            "job_id":           np.arange(n),
            "arrival_time":     trace_df["arrival_time"].values,
            "prefill_tokens":   trace_df["prefill_tokens"].values,
            "decode_tokens":    trace_df["decode_tokens"].values,
            "prefill_watts":    np.round(prefill_watts, 2),
            "prefill_duration": np.round(prefill_duration, 6),
            "decode_watts":     np.round(decode_watts, 2),
            "decode_duration":  np.round(decode_duration, 6),
        })
        fused_df.attrs["gpu_model"] = self.gpu_model
        fused_df.attrs["source"] = trace_df.attrs.get("source", "unknown")
        return fused_df

    # ──────────────────────────────────────────────────────────────
    # Public API: End-to-End Fused Dataset Generation
    # ──────────────────────────────────────────────────────────────

    def generate_fused_dataset(
        self,
        num_requests: int | None = 100,
        source: str = "azure",
        workload: str = "conv",
    ) -> pd.DataFrame:
        """
        Full pipeline: Load/generate trace → Apply hardware physics → Fused jobs.

        Args:
            num_requests: Number of jobs (None = full real trace).
            source: "azure" for the real Azure 2023 trace (downloads on
                    first use), "synthetic" for the offline fallback.
            workload: Real-trace variant, "conv" or "code" (source="azure").

        Returns:
            Fused DataFrame, attrs-tagged with the matched gpu_model:
                [job_id, arrival_time, prefill_tokens, decode_tokens,
                 prefill_watts, prefill_duration, decode_watts, decode_duration]
        """
        if source == "azure":
            try:
                loader = AzureTraceLoader(workload=workload)
                trace_df = loader.load(max_requests=num_requests)
            except (OSError, urllib.error.URLError) as exc:
                print(f"  [!] Real Azure trace unavailable ({exc}); "
                      f"falling back to synthetic trace.")
                trace_df = self.generate_azure_trace(num_requests or 100)
        elif source == "synthetic":
            trace_df = self.generate_azure_trace(num_requests or 100)
        else:
            raise ValueError(f"Unknown source '{source}' (use 'azure' or 'synthetic')")

        return self.apply_hardware_physics(trace_df)

    # ──────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────

    def print_dataset_summary(self, fused_df: pd.DataFrame) -> None:
        """Print a diagnostic summary of the fused dataset."""
        print("\n" + "=" * 70)
        print("  FUSED DATASET SUMMARY")
        print(f"  Trace: {fused_df.attrs.get('source', 'unknown')}  x  "
              f"Hardware: {self.gpu_model} ({self.hardware_profile['source']})")
        print(f"  GPU match: trace fleet = hardware profile = {self.gpu_model}")
        print("=" * 70)
        print(f"  Total jobs:           {len(fused_df)}")
        print(f"  Time span:            {fused_df['arrival_time'].min():.2f}s -> "
              f"{fused_df['arrival_time'].max():.2f}s")
        print(f"  --- Prefill Phase ---")
        print(f"  Tokens:               {fused_df['prefill_tokens'].min()} -> "
              f"{fused_df['prefill_tokens'].max()}")
        print(f"  Watts:                {fused_df['prefill_watts'].min():.1f}W -> "
              f"{fused_df['prefill_watts'].max():.1f}W")
        print(f"  Duration:             {fused_df['prefill_duration'].min()*1e3:.2f}ms -> "
              f"{fused_df['prefill_duration'].max()*1e3:.2f}ms")
        print(f"  --- Decode Phase ----")
        print(f"  Tokens:               {fused_df['decode_tokens'].min()} -> "
              f"{fused_df['decode_tokens'].max()}")
        print(f"  Watts:                {fused_df['decode_watts'].min():.1f}W -> "
              f"{fused_df['decode_watts'].max():.1f}W")
        print(f"  Duration:             {fused_df['decode_duration'].min():.2f}s -> "
              f"{fused_df['decode_duration'].max():.2f}s")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fuser = WorkloadFuser(gpu_model="H100", seed=42)
    df = fuser.generate_fused_dataset(num_requests=10, source="azure", workload="conv")
    fuser.print_dataset_summary(df)
    print("\nSample rows:")
    print(df.head().to_string(index=False))

    # GPU-matching guard demo: H100 trace x A100 physics must fail
    print("\nGPU mismatch guard check:")
    trace = AzureTraceLoader(workload="conv").load(max_requests=5)
    try:
        WorkloadFuser(gpu_model="A100").apply_hardware_physics(trace)
        print("  [FAIL] mismatch was not detected!")
    except GPUMismatchError as e:
        print(f"  [OK] GPUMismatchError raised: {e}")
