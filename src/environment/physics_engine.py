"""
Grid thermal physics: 2D Laplacian heat diffusion and Newton's-law cooling.

Pure functions operating on temperature/power arrays rather than environment
methods, so the physics can be unit-tested and reasoned about independently
of ThermalGridEnv's bookkeeping (queue, phases, reward).
"""

from __future__ import annotations

import numpy as np


def inject_heat(
    temperatures: np.ndarray,
    power_draw: np.ndarray,
    power_to_heat_grid: np.ndarray,
    timestep: float,
) -> np.ndarray:
    """
    Convert power draw into a temperature rise: T += eta * P * dt.

    eta (power_to_heat_grid) is per-cell -- it encodes each device's
    effective thermal resistance, so identical wattage on two different
    GPU models produces two different temperature rises.
    """
    return temperatures + power_to_heat_grid * power_draw * timestep


def apply_laplacian_diffusion(
    temperatures: np.ndarray,
    diffusivity: float,
    timestep: float,
    ambient_temp: float,
) -> np.ndarray:
    """
    Discrete 2D Laplacian heat diffusion, 4-neighbor stencil:
        T_new = T + alpha * grad^2 T * dt
    Neumann (zero-flux) boundary via edge padding.

    Diffusivity stays a scalar even on heterogeneous grids: alpha models
    interface conductance between adjacent packages sharing a chassis, a
    reasonable uniform approximation. Device-specific thermal behaviour
    enters through inject_heat's per-cell eta, not through alpha.
    """
    padded = np.pad(temperatures, pad_width=1, mode="edge")

    laplacian = (
        padded[0:-2, 1:-1] +
        padded[2:,   1:-1] +
        padded[1:-1, 0:-2] +
        padded[1:-1, 2:]   -
        4.0 * temperatures
    )

    result = temperatures + diffusivity * timestep * laplacian
    return np.maximum(result, ambient_temp)


def apply_newtonian_cooling(
    temperatures: np.ndarray,
    cooling_coeff: float,
    timestep: float,
    ambient_temp: float,
) -> np.ndarray:
    """Cooling toward ambient: T -= gamma * dt * (T - T_ambient)."""
    return temperatures - cooling_coeff * timestep * (temperatures - ambient_temp)
