"""Gymnasium physics environment: the spatial-phase thermal grid."""

from src.environment.thermal_env import (
    ThermalGridEnv,
    make_checkerboard_layout,
    make_split_layout,
)

__all__ = ["ThermalGridEnv", "make_checkerboard_layout", "make_split_layout"]
