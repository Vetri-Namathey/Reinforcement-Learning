"""
Project-wide configuration: filesystem layout anchors and GPU hardware
physics profiles.

Path constants are resolved relative to this file's location (the project
root's `configs/` directory), not the process's current working directory,
so every module behaves the same whether invoked as `python -m src...`
from the repo root, from a test runner, or from an installed console script.
"""

from pathlib import Path

from configs.hardware_profiles import (
    GPU_HARDWARE_PROFILES,
    GPUMismatchError,
    PHYSICS_FRACTION_COLUMNS,
    TDP_REFERENCE_WATTS,
    get_hardware_profile,
    normalize_gpu_models,
    resolve_job_physics,
    validate_grid_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AZURE_TRACE_DIR = DATA_DIR / "AzurePublicDataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ARTIFACTS_DIR / "logs"

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "AZURE_TRACE_DIR",
    "ARTIFACTS_DIR", "MODELS_DIR", "PLOTS_DIR", "LOGS_DIR",
    "GPU_HARDWARE_PROFILES", "GPUMismatchError", "PHYSICS_FRACTION_COLUMNS",
    "TDP_REFERENCE_WATTS", "get_hardware_profile", "normalize_gpu_models",
    "resolve_job_physics", "validate_grid_support",
]
