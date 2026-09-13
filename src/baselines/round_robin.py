"""Round-robin baseline scheduler: spatially blind cyclic placement."""

from __future__ import annotations

import numpy as np

from src.environment.thermal_env import ThermalGridEnv

# WAIT = ThermalGridEnv.WAIT_ACTION


class RoundRobinScheduler:
    """Cycles through GPU slots 0..35 sequentially (spatially blind).
    Advances only on successful placement; Waits if the target is busy."""

    name = "Round-Robin"

    def __init__(self):
        self.pointer = 0

    def select_action(self, obs: dict) -> int:
        valid = np.flatnonzero(obs["action_mask"][:-1])
        num_cells = obs["occupancy"].size
        if len(valid) == 0:
            return num_cells
        target = valid[self.pointer % len(valid)]
        self.pointer += 1
        return target
