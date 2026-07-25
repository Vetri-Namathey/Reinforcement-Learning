"""
Fused Dataset Pipeline: merges the real Azure LLM Inference Trace 2023
(software traffic layer) with GPU hardware physics profiles (hardware
physics layer) into physically-grounded LLM inference jobs.

Deferred-physics model
-----------------------
Token counts are the *primary* job payload. Watts and durations are NOT
properties of a job -- they are properties of a (job, GPU) pair, and on a
heterogeneous grid the GPU is not known until the scheduler places the job.
So this module emits raw tokens plus GPU-agnostic per-job efficiency
fractions, and `resolve_job_physics()` (configs.hardware_profiles) turns
those into watts/durations against whichever hardware profile the target
cell carries.

The fused frame still carries reference watt/duration columns computed on
the primary GPU, but those are diagnostics only -- the environment ignores
them and resolves physics at placement time.

GPU-Matching Invariant (set-based): every GPU model present on the grid
must have a calibrated hardware profile, and must be in the fused
dataset's declared `supported_gpus` set.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Sequence

import numpy as np
import pandas as pd

from configs import (
    GPUMismatchError,
    PHYSICS_FRACTION_COLUMNS,
    get_hardware_profile,
    normalize_gpu_models,
)
from configs.hardware_profiles import lerp
from src.data_pipeline.trace_loader import AZURE_TRACE_GPU_MODEL, AzureTraceLoader


class WorkloadFuser:
    """
    Fuses an LLM traffic trace with GPU hardware physics constants into a
    unified job DataFrame.

    Under the deferred-physics model the fuser is bound to a *set* of GPU
    models -- every device the resulting dataset may legally be scheduled on.
    Token counts and per-job efficiency fractions are hardware-agnostic; the
    watt/duration columns it emits are reference values on the primary GPU
    and are not consumed by the environment.

    The trace's provenance GPU must be among the calibrated set: watts and
    latencies are only meaningful for hardware whose constants were measured.
    """

    def __init__(
        self,
        gpu_models: str | Sequence[str] = AZURE_TRACE_GPU_MODEL,
        seed: int = 42,
        *,
        gpu_model: str | Sequence[str] | None = None,
    ):
        """
        Args:
            gpu_models: One model or a sequence of models this dataset may be
                scheduled on (e.g. "H100" or ["H100", "A100"]).
            seed: RNG seed for per-job efficiency sampling.
            gpu_model: Backward-compatible alias for gpu_models.
        """
        if gpu_model is not None:
            gpu_models = gpu_model

        self.gpu_models = normalize_gpu_models(gpu_models)
        self.gpu_model = self.gpu_models[0]          # primary / reference device
        self.hardware_profile = get_hardware_profile(self.gpu_model)
        self.hardware_profiles = {
            m: get_hardware_profile(m) for m in self.gpu_models
        }
        self.rng = np.random.default_rng(seed)

    def _validate_gpu_match(self, trace_df: pd.DataFrame, required_gpus=None) -> None:
        """
        Verify the trace's provenance fleet is calibrated and that every
        required GPU has a hardware profile. Membership, not equality: a
        trace may legally be fused for several devices at once.

        Raises:
            GPUMismatchError: on an untagged trace, an uncalibrated required
                GPU, or a trace whose serving fleet is outside the set.
        """
        trace_gpu = trace_df.attrs.get("gpu_model")
        if trace_gpu is None:
            raise GPUMismatchError(
                "Trace DataFrame carries no attrs['gpu_model'] tag -- cannot "
                "verify which fleet it was collected on (calibrated set: "
                f"{self.gpu_models}). Tag the trace before fusing."
            )

        required = normalize_gpu_models(
            required_gpus if required_gpus is not None else self.gpu_models
        )
        supported = {m.upper() for m in self.gpu_models}

        missing = sorted({m.upper() for m in required} - supported)
        if missing:
            raise GPUMismatchError(
                f"GPU support mismatch: {missing} requested but this fuser was "
                f"built for {sorted(supported)}. Construct WorkloadFuser with "
                "every GPU the grid will contain."
            )

        if trace_gpu.upper() not in supported:
            raise GPUMismatchError(
                f"GPU provenance mismatch: trace was served on '{trace_gpu}' but "
                f"the calibrated hardware set is {sorted(supported)}. The serving "
                "fleet must be among the calibrated devices -- token/watt "
                "constants are only valid for measured hardware."
            )

    TRACE_PROFILE = {
        "prefill_tokens_range":  (128, 2048),
        "decode_tokens_range":   (32, 512),
        "arrival_rate_lambda":   0.5,          # Poisson requests/sec
    }

    def generate_azure_trace(self, num_requests: int = 100) -> pd.DataFrame:
        """
        Generate a synthetic Azure-like trace (offline fallback when the
        real CSV is unavailable).

        Tagged with this fuser's primary GPU, not a hardcoded constant: a
        synthetic trace has no real serving fleet, so pinning it to H100
        made every non-H100 dataset unbuildable.
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
        trace_df.attrs["gpu_model"] = self.gpu_model
        trace_df.attrs["source"] = "synthetic"
        return trace_df

    def apply_hardware_physics(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach GPU-agnostic per-job efficiency fractions and reference
        physics columns.

        Per-job variance (batching, binning) is captured as a fraction in
        [0, 1] per phase rather than as a resolved wattage, so a job keeps
        its relative efficiency wherever it lands:
            watts    = tokens x lerp(profile.W/token, frac) + idle  (clamped)
            duration = tokens x lerp(profile.latency/token, frac)

        The emitted *_watts/*_duration columns are that formula evaluated on
        the primary GPU. They are diagnostics: ThermalGridEnv re-resolves
        them per cell at placement time.

        Raises:
            GPUMismatchError: if trace_df's fleet is outside the calibrated set.
        """
        self._validate_gpu_match(trace_df)

        n = len(trace_df)

        # Hardware-agnostic per-job efficiency draws, shared across profiles
        fractions = {
            name: self.rng.uniform(0.0, 1.0, size=n)
            for name in PHYSICS_FRACTION_COLUMNS
        }

        fused_df = pd.DataFrame({
            "job_id":         np.arange(n),
            "arrival_time":   trace_df["arrival_time"].values,
            "prefill_tokens": trace_df["prefill_tokens"].values,
            "decode_tokens":  trace_df["decode_tokens"].values,
            **{name: np.round(values, 6) for name, values in fractions.items()},
        })

        reference = self.reference_physics(fused_df, self.gpu_model)
        for column, values in reference.items():
            fused_df[column] = values

        fused_df.attrs["gpu_model"] = self.gpu_model
        fused_df.attrs["supported_gpus"] = frozenset(self.gpu_models)
        fused_df.attrs["source"] = trace_df.attrs.get("source", "unknown")
        fused_df.attrs["physics"] = "deferred"
        return fused_df

    def reference_physics(self, fused_df: pd.DataFrame, gpu_model: str) -> dict:
        """
        Vectorized resolution of every job against one GPU profile.

        Used for the fused frame's diagnostic columns and for offline
        what-if analysis ("what would this trace draw on A100s?").
        """
        profile = get_hardware_profile(gpu_model)
        idle = profile["idle_power"]

        prefill_tokens = fused_df["prefill_tokens"].values.astype(np.float64)
        decode_tokens = fused_df["decode_tokens"].values.astype(np.float64)

        prefill_wpt = lerp(profile["prefill_watts_per_token"],
                           fused_df["prefill_watt_frac"].values)
        decode_wpt  = lerp(profile["decode_watts_per_token"],
                           fused_df["decode_watt_frac"].values)
        prefill_lpt = lerp(profile["prefill_latency_per_token"],
                           fused_df["prefill_lat_frac"].values)
        decode_lpt  = lerp(profile["decode_latency_per_token"],
                           fused_df["decode_lat_frac"].values)

        prefill_watts = np.clip(prefill_tokens * prefill_wpt + idle,
                                *profile["prefill_watts_bounds"])
        decode_watts = np.clip(decode_tokens * decode_wpt + idle,
                               *profile["decode_watts_bounds"])

        return {
            "prefill_watts":    np.round(prefill_watts, 2),
            "prefill_duration": np.round(prefill_tokens * prefill_lpt, 6),
            "decode_watts":     np.round(decode_watts, 2),
            "decode_duration":  np.round(decode_tokens * decode_lpt, 6),
        }

    def generate_fused_dataset(
        self,
        num_requests: int | None = 100,
        source: str = "azure",
        workload: str = "conv",
    ) -> pd.DataFrame:
        """
        Full pipeline: load/generate trace -> attach fractions + reference physics.

        Args:
            num_requests: Number of jobs (None = full real trace).
            source: "azure" for the real trace (downloads on first use),
                    "synthetic" for the offline fallback.
            workload: Real-trace variant, "conv" or "code".
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

    def print_dataset_summary(self, fused_df: pd.DataFrame) -> None:
        """Print a diagnostic summary of the fused dataset."""
        supported = sorted(fused_df.attrs.get("supported_gpus", {self.gpu_model}))

        print("\n" + "=" * 70)
        print("  FUSED DATASET SUMMARY")
        print(f"  Trace: {fused_df.attrs.get('source', 'unknown')}  x  "
              f"Hardware: {'/'.join(supported)}")
        print(f"  Physics: {fused_df.attrs.get('physics', 'static')} "
              f"(resolved per-cell at placement time)")
        print(f"  Supported GPUs: {supported}  |  reference device: {self.gpu_model}")
        print("=" * 70)
        print(f"  Total jobs:           {len(fused_df)}")
        print(f"  Time span:            {fused_df['arrival_time'].min():.2f}s -> "
              f"{fused_df['arrival_time'].max():.2f}s")
        print(f"  Prefill tokens:       {fused_df['prefill_tokens'].min()} -> "
              f"{fused_df['prefill_tokens'].max()}")
        print(f"  Decode tokens:        {fused_df['decode_tokens'].min()} -> "
              f"{fused_df['decode_tokens'].max()}")

        for model in supported:
            phys = self.reference_physics(fused_df, model)
            print(f"  --- Resolved on {model} " + "-" * (48 - len(model)))
            print(f"    Prefill watts:      {phys['prefill_watts'].min():.1f}W -> "
                  f"{phys['prefill_watts'].max():.1f}W")
            print(f"    Prefill duration:   {phys['prefill_duration'].min()*1e3:.2f}ms -> "
                  f"{phys['prefill_duration'].max()*1e3:.2f}ms")
            print(f"    Decode watts:       {phys['decode_watts'].min():.1f}W -> "
                  f"{phys['decode_watts'].max():.1f}W")
            print(f"    Decode duration:    {phys['decode_duration'].min():.2f}s -> "
                  f"{phys['decode_duration'].max():.2f}s")
        print("=" * 70)


if __name__ == "__main__":
    from configs import get_hardware_profile as _get_profile
    from configs import resolve_job_physics as _resolve

    fuser = WorkloadFuser(gpu_models=["H100", "A100"], seed=42)
    df = fuser.generate_fused_dataset(num_requests=10, source="azure", workload="conv")
    fuser.print_dataset_summary(df)
    print("\nSample rows (tokens + hardware-agnostic fractions):")
    print(df[["job_id", "prefill_tokens", "decode_tokens",
              "prefill_watt_frac", "decode_watt_frac"]].head().to_string(index=False))

    print("\nSame job resolved on different hardware:")
    job = df.iloc[0]
    for model in ("H100", "A100"):
        phys = _resolve(job, _get_profile(model))
        print(f"  {model}: prefill {phys['prefill_watts']:6.1f}W / "
              f"{phys['prefill_duration']*1e3:6.2f}ms | "
              f"decode {phys['decode_watts']:6.1f}W / "
              f"{phys['decode_duration']:5.2f}s")

    print("\nA100 synthetic dataset (regression check for the H100 hardcode):")
    a100_df = WorkloadFuser(gpu_models="A100", seed=1).generate_fused_dataset(
        num_requests=5, source="synthetic"
    )
    print(f"  [OK] Built {len(a100_df)} jobs, "
          f"supported={sorted(a100_df.attrs['supported_gpus'])}")
