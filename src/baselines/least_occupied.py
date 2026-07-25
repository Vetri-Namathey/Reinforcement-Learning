"""Least-occupied baseline scheduler: load-aware, thermally blind."""

from __future__ import annotations

import numpy as np

from src.environment.thermal_env import ThermalGridEnv

WAIT = ThermalGridEnv.WAIT_ACTION


class LeastOccupiedScheduler:
    """Picks the cell with the lowest occupancy timer (load-aware,
    thermally blind). Waits only if every cell is busy."""

    name = "Least-Occupied"

    def select_action(self, obs: dict) -> int:
        if obs["queue_stats"][0] < 1:
            return WAIT
        occ = obs["occupancy"].ravel()
        target = int(np.argmin(occ))
        return target if occ[target] == 0 else WAIT
