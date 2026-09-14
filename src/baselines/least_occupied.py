"""Least-occupied baseline scheduler: load-aware, thermally blind."""

from __future__ import annotations

import numpy as np


class LeastOccupiedScheduler:
    """Picks the cell with the lowest occupancy timer (load-aware,
    thermally blind). Waits only if every cell is busy."""

    name = "Least-Occupied"

    def select_action(self, obs: dict) -> int:
        valid = np.flatnonzero(obs["action_mask"][:-1])
        num_cells = obs["occupancy"].size 
        if len(valid) == 0:
            return num_cells
        occ = obs["occupancy"].ravel()
        target = valid[np.argmin(occ[valid])]
        return target
