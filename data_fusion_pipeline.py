"""
data_fusion_pipeline.py
=======================
Fused Dataset Pipeline: Merges Azure LLM Inference Traces (software traffic)
with MLPerf / NVIDIA Hardware Telemetry (hardware physics) to produce
physically-grounded LLM inference jobs for the RL environment.

Each output row represents a single LLM inference request with:
  - Prefill wattage and duration (compute-bound, hot phase)
  - Decode wattage and duration (memory-bound, cool phase)
"""

import numpy as np
import pandas as pd


class WorkloadFuser:
    """
    Generates and fuses synthetic Azure LLM traffic traces with MLPerf
    hardware physics constants to produce a unified job DataFrame.

    The two-layer fusion:
        Layer 1 (Software):  Azure trace → arrival_time, prefill_tokens, decode_tokens
        Layer 2 (Hardware):  MLPerf constants → watts/token, latency/token
        Output:              Fused job → prefill_watts, prefill_duration, decode_watts, decode_duration
    """

    # ──────────────────────────────────────────────────────────────
    # MLPerf / NVIDIA H100 Mock Hardware Constants
    # ──────────────────────────────────────────────────────────────
    HARDWARE_PROFILE = {
        # Power consumption per token (Watts)
        "prefill_watts_per_token": (0.30, 0.55),   # Compute-bound: high power
        "decode_watts_per_token":  (0.10, 0.20),   # Memory-bound: lower power

        # Latency per token (seconds)
        "prefill_latency_per_token": (0.015e-3, 0.025e-3),  # ~15-25 µs/token
        "decode_latency_per_token":  (0.020e-3, 0.035e-3),  # ~20-35 µs/token

        # Baseline idle power draw (Watts)
        "idle_power": 50.0,
    }

    # ──────────────────────────────────────────────────────────────
    # Azure Trace Simulation Parameters
    # ──────────────────────────────────────────────────────────────
    TRACE_PROFILE = {
        "prefill_tokens_range":  (128, 2048),    # Input prompt lengths
        "decode_tokens_range":   (32, 512),      # Output generation lengths
        "arrival_rate_lambda":   0.5,            # Poisson arrival rate (requests/sec)
    }

    def __init__(self, seed: int = 42):
        """
        Initialize the WorkloadFuser with a reproducible random seed.

        Args:
            seed: Random seed for reproducibility across experiments.
        """
        self.rng = np.random.default_rng(seed)
        self._hardware_constants = None  # Lazily sampled per-job constants

    # ──────────────────────────────────────────────────────────────
    # Layer 1: Synthetic Azure LLM Trace Generation
    # ──────────────────────────────────────────────────────────────

    def generate_azure_trace(self, num_requests: int = 100) -> pd.DataFrame:
        """
        Generate a synthetic Azure LLM Inference Trace.

        Simulates the Microsoft Azure Public Dataset trace format with:
          - Poisson-distributed arrival times
          - Log-normal-ish token count distributions (heavy-tailed, realistic)

        Args:
            num_requests: Number of LLM inference requests to generate.

        Returns:
            DataFrame with columns: [arrival_time, prefill_tokens, decode_tokens]
        """
        # Poisson inter-arrival times (exponential gaps)
        inter_arrivals = self.rng.exponential(
            scale=1.0 / self.TRACE_PROFILE["arrival_rate_lambda"],
            size=num_requests
        )
        arrival_times = np.cumsum(inter_arrivals)

        # Token counts: use a shifted log-normal for realistic heavy-tail distribution
        prefill_min, prefill_max = self.TRACE_PROFILE["prefill_tokens_range"]
        decode_min, decode_max = self.TRACE_PROFILE["decode_tokens_range"]

        prefill_tokens = self.rng.integers(
            low=prefill_min, high=prefill_max, size=num_requests
        )
        decode_tokens = self.rng.integers(
            low=decode_min, high=decode_max, size=num_requests
        )

        trace_df = pd.DataFrame({
            "arrival_time":   np.round(arrival_times, 4),
            "prefill_tokens": prefill_tokens,
            "decode_tokens":  decode_tokens,
        })

        return trace_df

    # ──────────────────────────────────────────────────────────────
    # Layer 2: MLPerf Hardware Physics Application
    # ──────────────────────────────────────────────────────────────

    def apply_hardware_physics(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map software-layer token counts to hardware-layer physical quantities
        using MLPerf / NVIDIA telemetry constants.

        For each job:
            prefill_watts    = prefill_tokens × watts_per_prefill_token + idle_power
            prefill_duration = prefill_tokens × latency_per_prefill_token
            decode_watts     = decode_tokens  × watts_per_decode_token  + idle_power
            decode_duration  = decode_tokens  × latency_per_decode_token

        Args:
            trace_df: Azure trace DataFrame with [prefill_tokens, decode_tokens].

        Returns:
            Fused DataFrame with physical job profiles.
        """
        n = len(trace_df)
        hw = self.HARDWARE_PROFILE

        # Sample per-job hardware constants (models manufacturing variance & binning)
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

        # Clamp wattages to realistic H100 TDP bounds
        prefill_watts = np.clip(prefill_watts, 200.0, 700.0)
        decode_watts  = np.clip(decode_watts, 80.0, 250.0)

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

        return fused_df

    # ──────────────────────────────────────────────────────────────
    # Public API: End-to-End Fused Dataset Generation
    # ──────────────────────────────────────────────────────────────

    def generate_fused_dataset(self, num_requests: int = 100) -> pd.DataFrame:
        """
        Full pipeline: Generate Azure trace → Apply MLPerf physics → Return fused jobs.

        Args:
            num_requests: Number of LLM inference jobs to generate.

        Returns:
            Fused DataFrame with columns:
                [job_id, arrival_time, prefill_tokens, decode_tokens,
                 prefill_watts, prefill_duration, decode_watts, decode_duration]
        """
        trace_df = self.generate_azure_trace(num_requests)
        fused_df = self.apply_hardware_physics(trace_df)
        return fused_df

    # ──────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────

    def print_dataset_summary(self, fused_df: pd.DataFrame) -> None:
        """Print a diagnostic summary of the fused dataset."""
        print("\n" + "=" * 70)
        print("  FUSED DATASET SUMMARY")
        print("  Azure LLM Trace x MLPerf Hardware Physics")
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
        print(f"  Duration:             {fused_df['decode_duration'].min()*1e3:.2f}ms -> "
              f"{fused_df['decode_duration'].max()*1e3:.2f}ms")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fuser = WorkloadFuser(seed=42)
    df = fuser.generate_fused_dataset(num_requests=10)
    fuser.print_dataset_summary(df)
    print("\nSample rows:")
    print(df.head().to_string(index=False))
