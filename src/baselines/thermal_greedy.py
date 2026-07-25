"""Thermal-greedy baseline scheduler: temperature-aware but 1-step greedy."""

from __future__ import annotations

import numpy as np

from src.environment.thermal_env import ThermalGridEnv

WAIT = ThermalGridEnv.WAIT_ACTION


class ThermalGreedyScheduler:
    """Picks the coldest cell (temperature-aware but 1-step greedy).
    Waits if that cell is busy."""

    name = "Thermal-Greedy"

    def select_action(self, obs: dict) -> int:
        if obs["queue_stats"][0] < 1:
            return WAIT
        target = int(np.argmin(obs["temperatures"].ravel()))
        if obs["occupancy"].ravel()[target] == 0:
            return target
        return WAIT
