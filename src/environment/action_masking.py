"""Action mask computation for the 37-way (36 cells + Wait) action space."""

from __future__ import annotations

import numpy as np


def compute_action_mask(
    occupancy: np.ndarray,
    num_cells: int,
    wait_action: int,
    queue_length: int,
) -> np.ndarray:
    """
    Build the (num_cells + 1)-element binary action mask: free cells are
    valid, Wait is always valid, and an empty queue masks out all
    placements.

    Feasibility is occupancy-only -- job-to-device capability checks (VRAM,
    power envelope) are not modeled here.
    """
    mask = np.zeros(num_cells + 1, dtype=np.int8)
    mask[wait_action] = 1
    if queue_length > 0:
        mask[:num_cells] = (occupancy.ravel() == 0).astype(np.int8)
    return mask
