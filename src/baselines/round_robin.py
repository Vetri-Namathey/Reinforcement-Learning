"""Round-robin baseline scheduler: spatially blind cyclic placement."""

from __future__ import annotations


class RoundRobinScheduler:
    """Cycles through GPU slots 0..N-1 sequentially (spatially blind).

    The pointer walks *cell indices*, not positions in the free-cell list.
    Indexing the compacted list of valid cells (`valid[pointer % len(valid)]`)
    is not round-robin: the list shrinks and shifts as cells fill, so the same
    pointer maps to a different cell each step and the traversal order is
    effectively arbitrary. Here the pointer advances over the fixed cell
    numbering and skips any slot that is currently unavailable, which is the
    behaviour the report describes.
    """

    name = "Round-Robin"

    def __init__(self):
        self.pointer = 0

    def select_action(self, obs: dict) -> int:
        num_cells = obs["occupancy"].size
        mask = obs["action_mask"][:num_cells]
        if not mask.any():
            return num_cells                      # WAIT

        # Advance over the fixed 0..N-1 ordering to the next available slot.
        for offset in range(num_cells):
            candidate = (self.pointer + offset) % num_cells
            if mask[candidate]:
                self.pointer = (candidate + 1) % num_cells
                return int(candidate)

        return num_cells                          # unreachable; mask.any() held
