

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class GPUMismatchError(ValueError):
    """Raised when a required GPU is uncalibrated or unsupported by the dataset."""


GPU_HARDWARE_PROFILES = {
    "H100": {
        "gpu_model": "H100",
        "tdp_watts": 700.0,
        "idle_power": 76.0,
        "throttle_temp": 87.0,
        "power_to_heat": 0.020,

        "prefill_watts_per_token": (0.30, 0.55),
        "decode_watts_per_token":  (0.50, 1.00),

        "prefill_watts_bounds": (200.0, 700.0),
        "decode_watts_bounds":  (100.0, 350.0),

        "prefill_latency_per_token": (0.25e-3, 0.45e-3),
        "decode_latency_per_token":  (15e-3, 30e-3),

        "source": "Splitwise (ISCA 2024) + MLPerf Inference v4.0 H100 power",
    },
    "A100": {
        "gpu_model": "A100",
        "tdp_watts": 400.0,
        "idle_power": 55.0,
        "throttle_temp": 85.0,
        "power_to_heat": 0.035,

        "prefill_watts_per_token": (0.15, 0.30),
        "decode_watts_per_token":  (0.25, 0.50),

        "prefill_watts_bounds": (120.0, 400.0),
        "decode_watts_bounds":  (70.0, 220.0),

        "prefill_latency_per_token": (0.50e-3, 0.90e-3),
        "decode_latency_per_token":  (30e-3, 60e-3),

        "source": "Splitwise (ISCA 2024) A100 phase power measurements. "
                "power_to_heat=0.035 is NOT from Splitwise (a power study gives "
                "no thermal resistance) and is unsourced; our own 4-device "
                "campus A100 fit gives eta/gamma=0.147 degC/W -> eta=0.021 at "
                "gamma=0.14, i.e. this constant is ~1.7x too high. Left "
                "uncorrected for the same reason as RTX6000 — see that note.",
    },
    "RTX6000": {
        "gpu_model": "RTX6000",
        "tdp_watts": 300.0,
        "idle_power": 30.0,
        "throttle_temp": 89.0,
        "power_to_heat": 0.037,

        "prefill_watts_per_token": (0.12, 0.25),
        "decode_watts_per_token":  (0.20, 0.45),
        "prefill_watts_bounds": (80.0, 300.0),
        "decode_watts_bounds":  (50.0, 180.0),
        "prefill_latency_per_token": (0.60e-3, 1.00e-3),
        "decode_latency_per_token":  (35e-3, 70e-3),

        "source": "tdp_watts/idle_power from vendor spec. prefill/decode "
                "token-rate and latency bounds scaled from H100/A100 "
                "(Splitwise, ISCA 2024) by RTX6000/A100 TDP ratio — not "
                "independently measured. "
                "power_to_heat=0.037 was taken from an earlier campus-telemetry "
                "fit that pinned ambient=25C; re-fitting with ambient free "
                "(Clg_Data/validate_physics_against_real_gpu.py) gives "
                "eta/gamma=0.137 degC/W -> eta=0.019 at gamma=0.14, i.e. this "
                "constant is ~1.9x too high. NOT YET CORRECTED: lowering it "
                "makes the grid cooler and the thermal reward even more inert, "
                "so it is deliberately left pending the thermal-load redesign.",
    },
}

# Global normalization reference: the most power-hungry calibrated device.
# Both the environment's tdp_normalized channel and the CNN's power channel
# divide by this, so the two stay consistent for any grid layout.
TDP_REFERENCE_WATTS = max(p["tdp_watts"] for p in GPU_HARDWARE_PROFILES.values())

# GPU-agnostic per-job efficiency draws, resolved against a profile at
# placement time by resolve_job_physics().
PHYSICS_FRACTION_COLUMNS = (
    "prefill_watt_frac",
    "decode_watt_frac",
    "prefill_lat_frac",
    "decode_lat_frac",
)


def get_hardware_profile(gpu_model: str) -> dict:
    """Look up a hardware physics profile by GPU model name."""
    key = gpu_model.upper()
    if key not in GPU_HARDWARE_PROFILES:
        raise KeyError(
            f"No hardware profile for GPU '{gpu_model}'. "
            f"Available: {list(GPU_HARDWARE_PROFILES)}"
        )
    return GPU_HARDWARE_PROFILES[key]


def normalize_gpu_models(gpu_models: str | Sequence[str]) -> list[str]:
    """Coerce a scalar-or-sequence GPU spec into a de-duplicated canonical list."""
    if isinstance(gpu_models, str):
        gpu_models = [gpu_models]
    canonical = []
    for model in gpu_models:
        profile = get_hardware_profile(model)      # raises KeyError if uncalibrated
        if profile["gpu_model"] not in canonical:
            canonical.append(profile["gpu_model"])
    if not canonical:
        raise ValueError("At least one GPU model must be specified.")
    return canonical


def lerp(bounds: tuple[float, float], frac):
    """Interpolate within a (low, high) range by a fraction in [0, 1]."""
    low, high = bounds
    return low + frac * (high - low)


def resolve_job_physics(job, profile: dict) -> dict:

    idle = profile["idle_power"]

    def frac(name):
        value = job[name] if name in job else 0.5
        return 0.5 if value is None else float(value)

    prefill_tokens = float(job["prefill_tokens"])
    decode_tokens = float(job["decode_tokens"])

    prefill_wpt = lerp(profile["prefill_watts_per_token"], frac("prefill_watt_frac"))
    decode_wpt  = lerp(profile["decode_watts_per_token"],  frac("decode_watt_frac"))
    prefill_lpt = lerp(profile["prefill_latency_per_token"], frac("prefill_lat_frac"))
    decode_lpt  = lerp(profile["decode_latency_per_token"],  frac("decode_lat_frac"))

    prefill_watts = float(np.clip(
        prefill_tokens * prefill_wpt + idle, *profile["prefill_watts_bounds"]
    ))
    decode_watts = float(np.clip(
        decode_tokens * decode_wpt + idle, *profile["decode_watts_bounds"]
    ))

    return {
        "prefill_watts":    prefill_watts,
        "prefill_duration": prefill_tokens * prefill_lpt,
        "decode_watts":     decode_watts,
        "decode_duration":  decode_tokens * decode_lpt,
    }


def validate_grid_support(fused_df, grid_gpu_models) -> None:

    grid_set = {m.upper() for m in grid_gpu_models}

    uncalibrated = sorted(g for g in grid_set if g not in GPU_HARDWARE_PROFILES)
    if uncalibrated:
        raise GPUMismatchError(
            f"Grid contains uncalibrated GPU(s) {uncalibrated}. "
            f"Calibrated profiles: {sorted(GPU_HARDWARE_PROFILES)}. "
            "Physics cannot be resolved for hardware with no measured constants."
        )

    supported = fused_df.attrs.get("supported_gpus")
    if supported is None:
        raise GPUMismatchError(
            "Fused dataset carries no attrs['supported_gpus'] tag -- cannot "
            "verify it supports the grid's hardware. Rebuild it with "
            "WorkloadFuser.generate_fused_dataset()."
        )

    supported_set = {m.upper() for m in supported}
    missing = sorted(grid_set - supported_set)
    if missing:
        raise GPUMismatchError(
            f"GPU support mismatch: the grid is populated with {missing} but "
            f"the fused dataset only supports {sorted(supported_set)}. "
            "Rebuild the dataset with those GPUs in its supported set."
        )
