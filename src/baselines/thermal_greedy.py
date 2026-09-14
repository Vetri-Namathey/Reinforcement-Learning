"""Thermal-greedy baseline scheduler: temperature-aware but 1-step greedy."""

from __future__ import annotations

import numpy as np


class ThermalGreedyScheduler:
    """Picks the coldest cell (temperature-aware but 1-step greedy).
    Waits if that cell is busy."""

    name = "Thermal-Greedy"

    def select_action(self, obs: dict) -> int:
        valid = np.flatnonzero(obs["action_mask"][:-1])
        num_cells = obs["occupancy"].size 
        if len(valid) == 0:
            return num_cells
        temps = obs["temperatures"].ravel()
        target = valid[np.argmin(temps[valid])]
        return target
