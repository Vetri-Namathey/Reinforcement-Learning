"""Best-fit baseline: shortest estimated completion time, job- and heat-aware."""

from __future__ import annotations

import numpy as np


class BestFitScheduler:
    """Places the head job on the cell with the shortest estimated completion time.

    This is deliberately the strongest heuristic in the set. It reads the same
    head-of-queue job features the RL policy sees and combines every signal a
    competent hand-written scheduler would use:

      * **job size**      -- token counts, so big jobs are matched to big devices
      * **capability**    -- per-cell TDP, as a proxy for throughput
      * **contention**    -- resident job count, since batching slows everyone
      * **throttle risk** -- a cell already near its thermal limit will downclock
                             and run the job at THROTTLE_SPEED_FACTOR

    It exists so that any RL win is not a win over job-blind strawmen. Round-robin,
    least-occupied and thermal-greedy each use one signal; this one uses all four.
    What it still cannot do is anticipate: it scores the grid as it stands right
    now, with no model of where heat is heading or what arrives next.

    Estimated time on cell c:

        t(c) ~ work / capability(c) * contention(c) * throttle(c)

    where work is the job's token count, capability scales with TDP, contention
    grows with the number of resident jobs, and throttle applies a fixed penalty
    to cells predicted to cross their thermal limit.
    """

    name = "Best-Fit"

    # Temperature above which a cell is treated as about to throttle. Expressed
    # as a fraction of the hottest per-cell threshold in the fleet (87 C on H100),
    # since the observation does not carry per-cell thresholds.
    THROTTLE_GUARD_C = 87.0
    THROTTLE_PENALTY = 1.0 / 0.6      # jobs run at 60% speed when throttled
    CONTENTION_SLOPE = 0.6            # rough mid-point of prefill/decode betas

    def select_action(self, obs: dict) -> int:
        num_cells = obs["occupancy"].size
        valid = np.flatnonzero(obs["action_mask"][:num_cells])
        if len(valid) == 0:
            return num_cells                      # WAIT

        head = obs["head_job"]
        work = float(head[0]) + float(head[1])    # normalised token counts
        if work <= 0.0:
            work = 1.0                            # empty queue guard

        capability = np.clip(obs["tdp_normalized"].ravel()[valid], 1e-3, None)
        slots = obs["slot_fill"].ravel()[valid]
        temps = obs["temperatures"].ravel()[valid]

        # Contention: every job already resident stretches this one's runtime.
        contention = 1.0 + self.CONTENTION_SLOPE * slots

        # Throttle risk: a cell at or above the guard temperature will downclock.
        # Blend rather than step, so a cell creeping toward the limit is already
        # less attractive than one well clear of it.
        over = np.clip((temps - self.THROTTLE_GUARD_C * 0.92)
                       / (self.THROTTLE_GUARD_C * 0.08), 0.0, 1.0)
        throttle = 1.0 + over * (self.THROTTLE_PENALTY - 1.0)

        est_time = work / capability * contention * throttle
        return int(valid[int(np.argmin(est_time))])
