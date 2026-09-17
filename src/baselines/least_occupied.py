"""Least-occupied baseline scheduler: load-aware, thermally blind."""

from __future__ import annotations

import numpy as np


class LeastOccupiedScheduler:
    """Picks the cell holding the fewest jobs (load-aware, thermally blind).

    Ranks on `slot_fill` -- resident job count -- rather than on remaining busy
    seconds. Under continuous batching those differ: a cell can hold one long
    decode job and still be the emptiest place to put work, while a cell about
    to free up is momentarily full. Job count is what "least occupied" means
    when a cell can host several jobs at once. Ties break on remaining time.
    """

    name = "Least-Occupied"

    def select_action(self, obs: dict) -> int:
        num_cells = obs["occupancy"].size
        valid = np.flatnonzero(obs["action_mask"][:num_cells])
        if len(valid) == 0:
            return num_cells
        fill = obs["slot_fill"].ravel()[valid]
        occ = obs["occupancy"].ravel()[valid]
        # lexicographic: fewest jobs first, then soonest to free up
        return int(valid[np.lexsort((occ, fill))[0]])
