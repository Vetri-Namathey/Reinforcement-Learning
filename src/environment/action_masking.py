"""Action mask computation for the (num_cells + 1)-way (cells + Wait) action space."""

from __future__ import annotations

import numpy as np


def compute_action_mask(
    slots_used: np.ndarray,
    num_cells: int,
    wait_action: int,
    queue_length: int,
    max_slots: int = 1,
    work_conserving: bool = False,
) -> np.ndarray:
    """
    Build the (num_cells + 1)-element binary action mask: cells with a free
    slot are valid, Wait is always valid, and an empty queue masks out all
    placements.

    Capacity is measured in *slots*, not as a busy/idle bit. With
    max_slots == 1 this is identical to the previous occupancy == 0 test; with
    max_slots > 1 a cell keeps accepting work until it is full, which is what
    keeps the action space from collapsing to one option under load.

    Feasibility is slot-count only -- job-to-device capability checks (VRAM,
    KV-cache footprint, power envelope) are not modeled here.
    """
    mask = np.zeros(num_cells + 1, dtype=np.int8)
    mask[wait_action] = 1
    if queue_length > 0:
        mask[:num_cells] = (slots_used.ravel() < max_slots).astype(np.int8)
        if work_conserving and mask[:num_cells].any():
            # A work-conserving scheduler never idles capacity while work is
            # queued, so Wait is removed whenever a slot exists. Every heuristic
            # baseline already behaves this way.
            #
            # This also makes the problem learnable. The +10 placement reward
            # fires whenever a slot is free, whatever cell is picked, so with
            # Wait available it is a large action-independent term: ~10 per step
            # of variance against ~0.4 per step of actual cell-choice signal.
            # Removing the choice turns +10 into a constant and leaves the
            # thermal and SLA terms -- which DO depend on the cell -- as the
            # whole of the advantage.
            mask[wait_action] = 0
    return mask
