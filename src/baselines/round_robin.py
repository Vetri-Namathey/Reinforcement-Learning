"""Round-robin baseline scheduler: spatially blind cyclic placement."""

from __future__ import annotations

from src.environment.thermal_env import ThermalGridEnv

# WAIT = ThermalGridEnv.WAIT_ACTION


class RoundRobinScheduler:
    """Cycles through GPU slots 0..35 sequentially (spatially blind).
    Advances only on successful placement; Waits if the target is busy."""

    name = "Round-Robin"

    def __init__(self):
        self.pointer = 0

    def select_action(self, obs: dict) -> int:
        num_cells = obs["occupancy"].size
        if obs["queue_stats"][0] < 1:
            return num_cells
        target = self.pointer % num_cells
        if obs["occupancy"].ravel()[target] == 0:
            self.pointer += 1
            return target
        return num_cells
