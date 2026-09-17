"""
Custom Gymnasium environment modeling a 6x6 spatial grid of GPUs with
2D Laplacian heat diffusion physics.

The RL agent schedules LLM inference jobs from a persistent backlog queue
onto GPU cells. Jobs arrive over continuous simulation time per the trace's
real arrival timestamps; unplaceable jobs wait in the queue (never dropped)
and accrue an SLA latency penalty.

Heterogeneous grids
--------------------
Each cell carries its own hardware profile via `gpu_layout_matrix`, so a
6x6 grid may mix H100 and A100 devices. Every per-device physical constant
is therefore a (6, 6) matrix, not a scalar:

    tdp_grid            per-cell TDP ceiling (W)
    idle_power_grid     per-cell idle draw (W)
    THROTTLE_TEMP_GRID  per-cell thermal-slowdown point (degC)
    POWER_TO_HEAT_GRID  per-cell eta, degC per (W*s)

Physics is resolved at *placement* time, not fusion time: a job's watts and
durations depend on which device it lands on, which is unknown until the
agent acts. Jobs carry token counts plus hardware-agnostic efficiency
fractions; `resolve_job_physics()` converts those against the target cell's
profile.

Action space: Discrete(37) - 0..35 place head-of-queue job on cell, 36 = Wait.
Invalid placements are prevented by an action mask in the observation.

Reward:  R = R_place - R_sla - R_thermal
  R_place   = +10 if a job was placed this step
  R_sla     = 0.1 x queue_length            (per-step queue occupancy)
  R_thermal = 3.0 x Sum (max(0, T - T_throttle[i,j]) / T_throttle[i,j])^2

R_sla charges *occupancy*, not accumulated wait. Summing 0.1 x len(queue) over
an episode equals 0.1/dt x (total queueing seconds), so the episode penalty is
linear in total delay -- the standard Little's-law formulation. Charging
Sum(now - arrival) every step instead double-integrates: a job waiting T seconds
costs O(T^2), which grows without bound and drowns the +10 placement term by
several orders of magnitude.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from configs import (
    TDP_REFERENCE_WATTS,
    get_hardware_profile,
    normalize_gpu_models,
    resolve_job_physics,
    validate_grid_support,
)
from src.environment.action_masking import compute_action_mask
from src.environment.physics_engine import (
    apply_laplacian_diffusion,
    apply_newtonian_cooling,
    inject_heat,
)


class ThermalGridEnv(gym.Env):
    """
    Spatial-phase LLM inference scheduling on a 6x6 thermodynamic GPU grid
    with a persistent backlog queue and optional heterogeneous hardware.

    Observation Space (Dict):
        temperatures    : Box(6, 6)  - temperature of each GPU (degC)
        power_draw      : Box(6, 6)  - power draw of each GPU (W)
        occupancy       : Box(6, 6)  - remaining busy time of each GPU (s)
        tdp_normalized  : Box(6, 6)  - per-cell TDP / TDP_REFERENCE_WATTS
        utilization     : Box(6, 6)  - power_draw / per-cell TDP, in [0, 1]
        action_mask     : Box(37,)   - 1 = action currently valid
        queue_stats     : Box(2,)    - [queue_length, head_job_wait_time (s)]

    tdp_normalized and utilization are what make heterogeneity observable:
    without them a cell drawing 400 W is indistinguishable from any other,
    even though that is 57% of an H100's ceiling but 100% of an A100's.

    Physics per step: power injection from workload phases (prefill ->
    decode -> idle), 2D Laplacian heat diffusion (4-neighbor stencil),
    Newton's-law cooling toward ambient.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    GRID_SIZE       = 6
    NUM_CELLS       = GRID_SIZE ** 2
    WAIT_ACTION     = NUM_CELLS
    AMBIENT_TEMP    = 25.0        # degC

    # Physics coefficients (per-second rates, scaled by TIMESTEP)
    THERMAL_DIFFUSIVITY = 0.15    # alpha - heat spreading between cells (1/s)
    COOLING_COEFF       = 0.14    # gamma - cooling toward ambient (1/s)
    TIMESTEP            = 0.1     # dt - simulation timestep (s)

    # Continuous batching. Concurrent jobs on one cell share its SMs and memory
    # bandwidth, so each runs slower as the batch grows:
    #     effective dt for a job = TIMESTEP / (1 + beta * (n_jobs_on_cell - 1))
    # Prefill is compute-bound and batches poorly (beta near 1 = near-serial);
    # decode is memory-bound and batches well (beta near 0 = near-parallel).
    # That is the same prefill/decode asymmetry the project is built on, now
    # expressed in the execution model rather than only in wattage.
    # At n_jobs == 1 both reduce to exactly TIMESTEP, so max_jobs_per_cell=1
    # reproduces the pre-concurrency environment bit for bit.
    PREFILL_CONTENTION = 0.85
    DECODE_CONTENTION  = 0.25

    # Thermal throttling. A cell above its own throttle point downclocks, and
    # every job resident on it executes at this fraction of full speed. This is
    # what makes temperature *consequential* rather than cosmetic: without it a
    # cell at 95 degC runs exactly as fast as one at 60, heat costs nothing in
    # throughput, and the thermal penalty is a shaping term the scheduler can
    # ignore. With it, packing a cell hot delays its jobs, which backs up the
    # queue and raises TTFT -- a delayed cost a myopic policy cannot see coming.
    THROTTLE_SPEED_FACTOR = 0.6

    REWARD_VALID_PLACEMENT = 10.0
    SLA_PENALTY_WEIGHT     = 0.1
    # Weighted so the thermal term is ~24% of the total placement reward over an
    # episode -- present enough to shape behaviour, not so large that refusing to
    # schedule becomes optimal. At 0.5 thermal was 4% and the agent could ignore
    # heat entirely; at 5.0 it reached 39% and dominated throughput. Measured on
    # the 6x6 / 2-slot / 1.4x configuration; recheck if that changes.
    THERMAL_PENALTY_COEFF  = 3.0


    def __init__(
        self,
        fused_dataset: pd.DataFrame,
        gpu_model: str = "H100",
        render_mode: str | None = None,
        gpu_layout_matrix=None,
        throttle_temp_override: float | None = None,
        grid_size: int = 6,
        reward_variant: str = "full",
        enforce_action_mask: bool = True,
        active_cell_mask=None,
        episode_jobs: int | None = None,
        randomize_start: bool = False,
        max_jobs_per_cell: int = 1,
        work_conserving: bool = True,
        candidate_k: int | None = None,
    ):
        """
        Args:
            fused_dataset: DataFrame from WorkloadFuser carrying token counts,
                per-job efficiency fractions, and attrs["supported_gpus"].
            gpu_model: Model used to fill the grid when gpu_layout_matrix is
                None (homogeneous case).
            render_mode: Optional Gymnasium render mode ('human' or 'ansi').
            gpu_layout_matrix: Optional (6, 6) array of GPU model names
                assigning hardware per (i, j). Overrides gpu_model.
            throttle_temp_override: Pin every cell to one throttle threshold
                (e.g. 80.0) instead of per-profile values. Use this to
                reproduce pre-heterogeneity baseline numbers.
            grid_size: Grid dimension (e.g., 6 for 6x6).
            reward_variant: One of ("full", "no_thermal", "no_sla", "linear_thermal").
                Selects which reward terms to include for ablation studies.
            enforce_action_mask: If True, assert all actions are valid (for MaskablePPO).
                If False, allow invalid actions (for vanilla PPO with penalty wrapper).
            episode_jobs: Jobs per episode. None replays the entire dataset every
                episode. Set it to make `fused_dataset` a pool that episodes are
                sampled from.
            randomize_start: With episode_jobs set, draw a fresh random window of
                the pool on every reset (seeded by reset(seed=...)). Without this
                the env is fully deterministic and PPO memorises one trajectory
                instead of learning a policy.
            max_jobs_per_cell: Concurrent jobs one cell may host (continuous
                batching). 1 reproduces the single-job-per-GPU environment
                exactly. Above 1, cell power is the sum of resident jobs'
                marginal draws and each job slows per PREFILL/DECODE_CONTENTION.
                This is what keeps the action space from collapsing: at K=1 under
                load the mask leaves 0-1 legal cells on ~95% of steps, so no
                policy can express a preference.

        Raises:
            GPUMismatchError: if any grid device is uncalibrated or outside
                the dataset's supported set.
        """
        super().__init__()
        self.render_mode = render_mode
        self.enforce_action_mask = enforce_action_mask

        self.GRID_SIZE   = grid_size
        self.NUM_CELLS   = self.GRID_SIZE ** 2
        self.WAIT_ACTION = self.NUM_CELLS

        self.gpu_layout = self._build_layout(gpu_layout_matrix, gpu_model)
        self.gpu_models = normalize_gpu_models(np.unique(self.gpu_layout).tolist())
        self.is_heterogeneous = len(self.gpu_models) > 1

        validate_grid_support(fused_dataset, self.gpu_models)

        # Per-cell profile objects drive placement-time physics resolution
        gs = self.GRID_SIZE
        self.cell_profiles = np.empty((gs, gs), dtype=object)
        for i in range(gs):
            for j in range(gs):
                self.cell_profiles[i, j] = get_hardware_profile(self.gpu_layout[i, j])

        self.tdp_grid           = self._profile_grid("tdp_watts")
        self.idle_power_grid    = self._profile_grid("idle_power")
        self.POWER_TO_HEAT_GRID = self._profile_grid("power_to_heat")
        if throttle_temp_override is not None:
            self.THROTTLE_TEMP_GRID = np.full(
                (gs, gs), float(throttle_temp_override), dtype=np.float32
            )
        else:
            self.THROTTLE_TEMP_GRID = self._profile_grid("throttle_temp")

        # Dominant device, and scalar aliases kept for reporting/back-compat.
        # Prefer the *_grid matrices in any physics or assertion code.
        models, counts = np.unique(self.gpu_layout, return_counts=True)
        self.gpu_model = str(models[int(np.argmax(counts))])
        self.gpu_profile = get_hardware_profile(self.gpu_model)
        self.idle_power = float(self.idle_power_grid.max())
        self.tdp_watts = float(self.tdp_grid.max())

        self.fused_dataset = (
            fused_dataset.sort_values("arrival_time").reset_index(drop=True)
        )
        # The dataset is a *pool*; each episode plays a contiguous window of it.
        # With episode_jobs=None the window is the whole pool (legacy behaviour).
        self._all_records = self.fused_dataset.to_dict("records")
        self.episode_jobs = (
            None if episode_jobs is None
            else min(int(episode_jobs), len(self._all_records))
        )
        self.randomize_start = bool(randomize_start)
        self._job_records = self._all_records
        self.total_jobs = len(self._job_records)
        self.max_jobs_per_cell = max(1, int(max_jobs_per_cell))
        self.work_conserving = bool(work_conserving)
        self.candidate_k = None if candidate_k is None else max(1, int(candidate_k))

        self.queue = []
        self.current_time = 0.0
        self.next_arrival_idx = 0

        self.action_space = spaces.Discrete(self.NUM_CELLS + 1)

        grid_shape = (gs, gs)
        self.observation_space = spaces.Dict({
            "temperatures": spaces.Box(
                low=0.0, high=200.0, shape=grid_shape, dtype=np.float32
            ),
            "power_draw": spaces.Box(
                low=0.0, high=float(self.tdp_grid.max()),
                shape=grid_shape, dtype=np.float32
            ),
            "occupancy": spaces.Box(
                low=0.0, high=1e6, shape=grid_shape, dtype=np.float32
            ),
            "tdp_normalized": spaces.Box(
                low=0.0, high=1.0, shape=grid_shape, dtype=np.float32
            ),
            "utilization": spaces.Box(
                low=0.0, high=1.0, shape=grid_shape, dtype=np.float32
            ),
            # Fraction of this cell's concurrency slots in use. Distinct from
            # `occupancy` (remaining seconds): a cell can be nearly free on time
            # but full on slots, or hold one long job and still accept work.
            "slot_fill": spaces.Box(
                low=0.0, high=1.0, shape=grid_shape, dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(self.NUM_CELLS + 1,), dtype=np.int8
            ),
            "queue_stats": spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            ),
            # The job about to be placed. Without this the policy schedules
            # blind: a 4,107-token prefill and a 2-token one are identical
            # inputs, so it cannot size-match work to hardware. Real serving
            # systems know token counts at admission.
            #   [prefill_tokens/4096, decode_tokens/1024,
            #    prefill_watt_frac, decode_watt_frac]
            "head_job": spaces.Box(
                low=0.0, high=10.0, shape=(4,), dtype=np.float32
            ),
        })

        # Static channel: per-cell capability, constant across the episode
        self._tdp_normalized = (
            self.tdp_grid / TDP_REFERENCE_WATTS
        ).astype(np.float32)

        self.temperatures = None
        self.power_draw   = None
        self.occupancy    = None
        self.slots_used   = None
        self.cell_jobs    = None
        self._last_obs_mask = None

        # Per-cell dominant workload phase: 0=idle, 1=any prefill, 2=decode only
        self.cell_phase = None

        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.jobs_completed = 0
        self.throttle_events = 0
        self.throttled_job_steps = 0   # (job, step) pairs that ran downclocked
        self.job_step_total = 0        # (job, step) pairs executed in total
        self.dropped_jobs = 0        # always 0: backlog queue never drops

        self.successful_jobs = []          # per-placed-job stats for TTFT
        self.throttling_steps = 0          # steps with any cell over its threshold
        self.total_steps = 0
        self.spatial_variance_sum = 0.0
        
        # Reward variant for ablation studies
        if reward_variant not in ("full", "no_thermal", "no_sla", "linear_thermal"):
            raise ValueError(
                f"reward_variant must be one of ('full', 'no_thermal', 'no_sla', "
                f"'linear_thermal'), got '{reward_variant}'"
            )
        self.reward_variant = reward_variant

        self.active_cell_mask = (
            np.asarray(active_cell_mask, dtype=bool)
            if active_cell_mask is not None
            else np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        )

    def _build_layout(self, gpu_layout_matrix, gpu_model: str) -> np.ndarray:
        """Resolve the (6, 6) grid of GPU model names."""
        gs = self.GRID_SIZE
        if gpu_layout_matrix is None:
            canonical = get_hardware_profile(gpu_model)["gpu_model"]
            return np.full((gs, gs), canonical, dtype=object)

        layout = np.asarray(gpu_layout_matrix, dtype=object)
        if layout.shape != (gs, gs):
            raise ValueError(
                f"gpu_layout_matrix must be {(gs, gs)}, got {layout.shape}"
            )
        # Canonicalize names; raises KeyError on an uncalibrated device
        return np.array(
            [[get_hardware_profile(str(layout[i, j]))["gpu_model"]
              for j in range(gs)] for i in range(gs)],
            dtype=object,
        )

    def _profile_grid(self, key: str) -> np.ndarray:
        """Materialize a per-cell constant into a (6, 6) float32 matrix."""
        gs = self.GRID_SIZE
        return np.array(
            [[float(self.cell_profiles[i, j][key]) for j in range(gs)]
             for i in range(gs)],
            dtype=np.float32,
        )

    def _agent_mask(self) -> np.ndarray:
        """The mask the agent was last shown, which is what its action must satisfy.

        step() advances the clock and ingests arrivals before executing the
        action, so the mask it would compute is one tick newer than the one the
        policy saw. Cell validity is unaffected (slots only free up inside
        _advance_workloads, which runs later), but under work_conserving the
        Wait action can be *revoked* by an arrival in that gap -- a policy that
        legitimately chose Wait on an empty queue would then be rejected for it.
        Validate against what was actually shown.
        """
        if self._last_obs_mask is None:
            return self._compute_action_mask()
        return self._last_obs_mask

    def _shortlist(self, mask: np.ndarray) -> np.ndarray:
        """Narrow the legal cells to the candidate_k most promising ones.

        With 36 cells open the agent must find a good placement by blind search
        over 36 near-identical options, and measurably never does. Ranking on a
        cheap, policy-independent score and exposing only the top k turns that
        search into a small comparison, while leaving the *interesting* part of
        the decision -- which of several plausible cells suits THIS job, given
        its size and the device underneath -- entirely to the policy.

        The ranking deliberately uses only load and heat, never job size or
        device capability, so it cannot pre-solve the job-to-device matching
        that the agent is supposed to learn.
        """
        cells = mask[:self.NUM_CELLS].astype(bool)
        n_valid = int(cells.sum())
        if self.candidate_k is None or n_valid <= self.candidate_k:
            return mask

        fill = self.slots_used.ravel() / self.max_jobs_per_cell
        heat = self.temperatures.ravel() / self.THROTTLE_TEMP_GRID.ravel()
        score = np.where(cells, fill + heat, np.inf)   # lower is better

        # Stratify by device type. A flat top-k on load and heat can return six
        # A100s and no H100 at all, which silently removes the job-to-device
        # choice the agent exists to make -- measured as Best-Fit collapsing from
        # 2876 to 1619 because the cells it wanted were never offered. Taking the
        # best few of EACH model guarantees every device type stays on the menu
        # while still cutting 36 options down to k.
        tdp_flat = self.tdp_grid.ravel()
        tiers = np.unique(tdp_flat)
        per_tier = max(1, self.candidate_k // len(tiers))

        keep = []
        for tier in tiers:
            in_tier = np.where((tdp_flat == tier) & cells)[0]
            if in_tier.size == 0:
                continue
            order = in_tier[np.argsort(score[in_tier])]
            keep.extend(order[:per_tier].tolist())

        # Any leftover budget goes to the best remaining cells, whatever tier.
        if len(keep) < self.candidate_k:
            rest = [i for i in np.argsort(score) if cells[i] and i not in set(keep)]
            keep.extend(rest[: self.candidate_k - len(keep)])

        narrowed = np.zeros_like(mask)
        narrowed[self.WAIT_ACTION] = mask[self.WAIT_ACTION]
        narrowed[np.array(keep, dtype=int)] = 1
        return narrowed

    def _compute_action_mask(self) -> np.ndarray:
        # Capacity first, WAIT last. Deciding work-conserving inside
        # compute_action_mask strips WAIT based on slots that active_cell_mask
        # and the shortlist may then remove -- on the campus fleet, where only 6
        # of 36 cells exist, that produced an all-zero mask and crashed the run.
        # Every filter is applied first; WAIT is then set from what actually
        # survived, so it is always available when nothing can be placed.
        mask = compute_action_mask(
            self.slots_used, self.NUM_CELLS, self.WAIT_ACTION, len(self.queue),
            max_slots=self.max_jobs_per_cell,
            work_conserving=False,
        )
        mask[:self.NUM_CELLS] &= self.active_cell_mask.ravel().astype(np.int8)
        mask = self._shortlist(mask)

        placeable = bool(mask[:self.NUM_CELLS].any())
        mask[self.WAIT_ACTION] = 0 if (self.work_conserving and placeable) else 1
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        gs = self.GRID_SIZE

        self._job_records = self._sample_episode_window()
        self.total_jobs = len(self._job_records)

        self.temperatures = np.full((gs, gs), self.AMBIENT_TEMP, dtype=np.float32)

        # Resident jobs per cell. Each entry is a dict carrying that job's own
        # phase, remaining prefill/decode seconds and marginal watts, so a cell
        # can hold several jobs at different phases simultaneously.
        self.cell_jobs = [[[] for _ in range(gs)] for _ in range(gs)]
        self._last_obs_mask = None

        # Derived views, recomputed from cell_jobs by _recompute_cell_state()
        self.power_draw = self.idle_power_grid.copy()
        self.occupancy  = np.zeros((gs, gs), dtype=np.float32)
        self.slots_used = np.zeros((gs, gs), dtype=np.int32)
        self.cell_phase = np.zeros((gs, gs), dtype=np.int32)

        self.queue = []
        self.current_time = 0.0
        self.next_arrival_idx = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.jobs_completed = 0
        self.throttle_events = 0
        self.throttled_job_steps = 0
        self.job_step_total = 0
        self.dropped_jobs = 0

        self.successful_jobs = []
        self.throttling_steps = 0
        self.total_steps = 0
        self.spatial_variance_sum = 0.0

        self._recompute_cell_state()

        # Ingest jobs arriving exactly at t=0 so the first action mask
        # reflects them
        self._ingest_arrivals()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """
        One step: advance time and ingest arrivals, execute the (masked)
        action, run thermodynamics, compute R = R_place - R_sla - R_thermal.
        """
        self.step_count += 1
        self.total_steps += 1

        self.current_time += self.TIMESTEP
        self._ingest_arrivals()

        mask = self._agent_mask()
        if self.enforce_action_mask:
            assert mask[action] == 1, (
                f"Invalid action {action}: masked out "
                f"(queue_length={len(self.queue)}, mask={mask.tolist()})"
            )

        job_placed = False
        if action != self.WAIT_ACTION and len(self.queue) > 0:
            row, col = divmod(action, self.GRID_SIZE)
            # active_cell_mask is a hard capacity constraint, not just a mask
            # hint: an unmasked caller must not be able to schedule onto a cell
            # the deployment does not physically have.
            has_slot = self.slots_used[row, col] < self.max_jobs_per_cell
            if has_slot and self.active_cell_mask[row, col]:
                job = self.queue.pop(0)
                # Deferred physics: the same job draws different power and runs
                # for a different duration depending on the device it lands on.
                phys = resolve_job_physics(job, self.cell_profiles[row, col])

                # Marginal watts: each job's resolved draw already includes the
                # cell's idle floor, so summing raw values would count idle once
                # per job. Store the delta and add idle back once per cell.
                idle = float(self.idle_power_grid[row, col])
                self.cell_jobs[row][col].append({
                    "phase":     1,
                    "pf_remain": phys["prefill_duration"],
                    "dec_remain": phys["decode_duration"],
                    "pf_watts":  max(0.0, phys["prefill_watts"] - idle),
                    "dec_watts": max(0.0, phys["decode_watts"] - idle),
                })
                self._recompute_cell_state()
                self.jobs_placed += 1
                job_placed = True

                # TTFT here = queueing delay until placement (excludes prefill)
                self.successful_jobs.append({
                    "job_id":           job["job_id"],
                    "arrival_time":     job["arrival_time"],
                    "placement_time":   self.current_time,
                    "gpu_model":        self.gpu_layout[row, col],
                    "prefill_duration": phys["prefill_duration"],
                    "ttft":             self.current_time - job["arrival_time"],
                })

        # Integrate the power that actually flowed during [t, t+dt) BEFORE
        # advancing the phase timers. Advancing first meant a job whose prefill
        # was shorter than one timestep flipped to decode before any heat was
        # injected, so its prefill spike -- the whole premise of the study --
        # never entered the physics at all (23% of the Azure conv trace).
        #
        # Element-wise: eta varies per device, so identical watts on an H100
        # and an A100 cell produce different temperature rises.
        self.temperatures = inject_heat(
            self.temperatures, self.power_draw, self.POWER_TO_HEAT_GRID,
            self.TIMESTEP,
        )

        self._advance_workloads()

        self.temperatures = apply_laplacian_diffusion(
            self.temperatures, self.THERMAL_DIFFUSIVITY, self.TIMESTEP,
            self.AMBIENT_TEMP,
        )

        self.temperatures = apply_newtonian_cooling(
            self.temperatures, self.COOLING_COEFF, self.TIMESTEP,
            self.AMBIENT_TEMP,
        )

        if np.any(self.temperatures > self.THROTTLE_TEMP_GRID):
            self.throttling_steps += 1
        self.spatial_variance_sum += float(np.var(self.temperatures))

        # Compute reward components
        r_place = self.REWARD_VALID_PLACEMENT if job_placed else 0.0

        # Queue *occupancy*, not accumulated wait -- see the module docstring.
        r_sla = self.SLA_PENALTY_WEIGHT * len(self.queue)

        # Excess is normalised by each cell's own throttle point before squaring.
        # Two reasons. (1) Scale: raw degrees-squared explodes once concurrency
        # pushes cells far past threshold -- at 110 degC the raw term reaches
        # ~1,600/step against a +10 placement reward, making "schedule nothing"
        # optimal by a factor of 35. Normalised, it lands at ~25% of the
        # placement reward, which is a real trade-off rather than a landslide.
        # (2) Heterogeneity: an A100 5 degC over its 85 degC limit and an H100
        # 5 degC over its 87 degC limit are equally stressed, and a dimensionless
        # ratio says so where absolute degrees does not.
        excess = np.maximum(0.0, self.temperatures - self.THROTTLE_TEMP_GRID)
        excess_norm = excess / self.THROTTLE_TEMP_GRID
        r_thermal_quad = self.THERMAL_PENALTY_COEFF * float(np.sum(excess_norm ** 2))
        r_thermal_linear = 0.5 * float(np.sum(excess_norm))  # linear variant
        self.throttle_events += int(np.sum(excess > 0))

        # Select reward variant
        if self.reward_variant == "full":
            reward = r_place - r_sla - r_thermal_quad
        elif self.reward_variant == "no_thermal":
            reward = r_place - r_sla
        elif self.reward_variant == "no_sla":
            reward = r_place - r_thermal_quad
        elif self.reward_variant == "linear_thermal":
            reward = r_place - r_sla - r_thermal_linear
        else:
            reward = r_place - r_sla - r_thermal_quad  # fallback

        # Terminate once the trace is ingested, the backlog is cleared AND every
        # placed job has finished. Ending at an empty queue cut the episode
        # while cells were still drawing decode power, so the final cohort's
        # thermal tail -- the hottest part of the episode -- was never simulated.
        terminated = (
            self.next_arrival_idx >= self.total_jobs
            and len(self.queue) == 0
            and not np.any(self.slots_used > 0)
        )
        truncated = False

        self.total_reward += reward

        info = self._get_info()
        if terminated:
            ttfts = [j["ttft"] for j in self.successful_jobs]
            info["p99_ttft"] = float(np.percentile(ttfts, 99)) if ttfts else 0.0
            info["final_tttr"] = self.throttling_steps / max(self.total_steps, 1)
            info["mean_spatial_variance"] = (
                self.spatial_variance_sum / max(self.total_steps, 1)
            )
            # Fraction of all job-execution time that ran downclocked -- the
            # throughput cost of heat, as opposed to TTTR which only says
            # whether *some* cell was hot.
            total_job_steps = max(self.job_step_total, 1)
            info["throttled_work_ratio"] = self.throttled_job_steps / total_job_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    def _sample_episode_window(self) -> list:
        """
        Pick this episode's contiguous slice of the job pool and re-base its
        arrival times to t=0, so every episode starts from the same clock
        regardless of where in the trace it was drawn from.
        """
        records = self._all_records
        if self.episode_jobs is not None and self.episode_jobs < len(records):
            span = len(records) - self.episode_jobs
            start = (
                int(self.np_random.integers(0, span + 1))
                if self.randomize_start else 0
            )
            records = records[start : start + self.episode_jobs]

        if not records:
            return []

        t0 = records[0]["arrival_time"]
        if t0 == 0.0:
            return list(records)
        return [{**r, "arrival_time": r["arrival_time"] - t0} for r in records]

    def _ingest_arrivals(self):
        """Move trace jobs with arrival_time <= current_time into the queue."""
        while (
            self.next_arrival_idx < self.total_jobs
            and self._job_records[self.next_arrival_idx]["arrival_time"]
                <= self.current_time
        ):
            self.queue.append(self._job_records[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _recompute_cell_state(self):
        """
        Rebuild the derived per-cell arrays from `cell_jobs`.

        power_draw = idle + sum of resident jobs' marginal draws, clamped to the
        cell's own TDP -- a cell cannot draw more than its package allows no
        matter how many jobs are batched onto it.
        """
        gs = self.GRID_SIZE
        for i in range(gs):
            for j in range(gs):
                jobs = self.cell_jobs[i][j]
                self.slots_used[i, j] = len(jobs)
                if not jobs:
                    self.power_draw[i, j] = self.idle_power_grid[i, j]
                    self.occupancy[i, j] = 0.0
                    self.cell_phase[i, j] = 0
                    continue

                watts = float(self.idle_power_grid[i, j])
                remain = 0.0
                any_prefill = False
                for job in jobs:
                    if job["phase"] == 1:
                        watts += job["pf_watts"]
                        any_prefill = True
                        remain = max(remain, job["pf_remain"] + job["dec_remain"])
                    else:
                        watts += job["dec_watts"]
                        remain = max(remain, job["dec_remain"])

                self.power_draw[i, j] = min(watts, float(self.tdp_grid[i, j]))
                self.occupancy[i, j] = remain
                self.cell_phase[i, j] = 1 if any_prefill else 2

    def _advance_workloads(self):
        """
        Advance every resident job; handle prefill -> decode -> completion.

        Concurrent jobs share the cell, so each one's clock runs slow by
        1 / (1 + beta * (n - 1)). With one job on the cell this is exactly
        TIMESTEP and the behaviour is identical to the single-job environment.
        """
        gs = self.GRID_SIZE
        for i in range(gs):
            for j in range(gs):
                jobs = self.cell_jobs[i][j]
                if not jobs:
                    continue

                n = len(jobs)
                self.job_step_total += n
                # Temperature here is post-injection, pre-diffusion: the heat
                # this cell actually carried during [t, t+dt).
                throttled = self.temperatures[i, j] > self.THROTTLE_TEMP_GRID[i, j]
                if throttled:
                    self.throttled_job_steps += n

                survivors = []
                for job in jobs:
                    beta = (self.PREFILL_CONTENTION if job["phase"] == 1
                            else self.DECODE_CONTENTION)
                    step = self.TIMESTEP / (1.0 + beta * (n - 1))
                    if throttled:
                        step *= self.THROTTLE_SPEED_FACTOR

                    if job["phase"] == 1:
                        job["pf_remain"] -= step
                        if job["pf_remain"] <= 0:
                            # Carry the overshoot into decode so a prefill
                            # shorter than one step does not get a free tick.
                            job["dec_remain"] += job["pf_remain"]
                            job["pf_remain"] = 0.0
                            job["phase"] = 2
                    else:
                        job["dec_remain"] -= step

                    if job["phase"] == 2 and job["dec_remain"] <= 0:
                        self.jobs_completed += 1
                        continue          # job leaves the cell
                    survivors.append(job)

                self.cell_jobs[i][j] = survivors

        self._recompute_cell_state()

    def _get_obs(self) -> dict:
        head_wait = (
            self.current_time - self.queue[0]["arrival_time"]
            if self.queue else 0.0
        )
        # Localized utilization: 400 W is 0.57 on an H100 but 1.00 on an A100.
        # Clipped because prefill_watts_bounds allow a draw at exactly TDP.
        utilization = np.clip(self.power_draw / self.tdp_grid, 0.0, 1.0)

        return {
            "temperatures":   self.temperatures.copy(),
            "power_draw":     self.power_draw.copy(),
            "occupancy":      self.occupancy.copy(),
            "tdp_normalized": self._tdp_normalized.copy(),
            "utilization":    utilization.astype(np.float32),
            "slot_fill":      (self.slots_used / self.max_jobs_per_cell
                               ).astype(np.float32),
            "action_mask":    self._remember_mask(),
            "queue_stats":    np.array(
                [len(self.queue), head_wait], dtype=np.float32
            ),
            "head_job":       self._head_job_features(),
        }

    def _remember_mask(self) -> np.ndarray:
        """Compute the action mask and record it as the one the agent was shown."""
        self._last_obs_mask = self._compute_action_mask()
        return self._last_obs_mask

    def _head_job_features(self) -> np.ndarray:
        """
        Hardware-agnostic descriptor of the head-of-queue job.

        Token counts drive both power and duration, and the two efficiency
        fractions carry the job's per-phase batching efficiency. All four are
        device-independent, so the policy can reason about job size without the
        feature changing meaning depending on which cell it is considering.
        Zeros when the queue is empty.
        """
        if not self.queue:
            return np.zeros(4, dtype=np.float32)
        job = self.queue[0]

        def frac(name):
            value = job[name] if name in job else 0.5
            return 0.5 if value is None else float(value)

        return np.array([
            float(job["prefill_tokens"]) / 4096.0,
            float(job["decode_tokens"]) / 1024.0,
            frac("prefill_watt_frac"),
            frac("decode_watt_frac"),
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "gpu_model":        self.gpu_model,
            "gpu_models":       list(self.gpu_models),
            "heterogeneous":    self.is_heterogeneous,
            "step":             self.step_count,
            "current_time":     round(self.current_time, 4),
            "jobs_placed":      self.jobs_placed,
            "jobs_remaining":   self.total_jobs - self.jobs_placed,
            "queue_length":     len(self.queue),
            "dropped_jobs":     self.dropped_jobs,
            "max_temperature":  float(np.max(self.temperatures)),
            "mean_temperature": float(np.mean(self.temperatures)),
            "throttle_events":  self.throttle_events,
            "total_reward":     self.total_reward,
            "active_cells":     int(np.sum(self.slots_used > 0)),
            "jobs_completed":   self.jobs_completed,
            "throttled_job_steps": self.throttled_job_steps,
            "jobs_resident":    int(np.sum(self.slots_used)),
            "max_slot_fill":    int(np.max(self.slots_used)) if self.slots_used is not None else 0,
            "spatial_variance": float(np.var(self.temperatures)),
            "mean_utilization": float(np.mean(self.power_draw / self.tdp_grid)),
            "thermal_headroom": float(
                np.min(self.THROTTLE_TEMP_GRID - self.temperatures)
            ),
            "tttr":             self.throttling_steps / max(self.total_steps, 1),
        }

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """ASCII grid rendering (works on any console encoding).
        Markers: '!' past this cell's throttle point, '~' warm (>60C), '.' ok."""
        gs = self.GRID_SIZE
        fleet = "/".join(self.gpu_models)
        lines = []
        lines.append(f"\n{'-' * 62}")
        lines.append(f"  THERMAL GRID STATE ({fleet})  |  "
                      f"Step: {self.step_count}  |  t={self.current_time:.1f}s  |  "
                      f"Placed: {self.jobs_placed}/{self.total_jobs}  |  "
                      f"Queue: {len(self.queue)}")
        lines.append(f"{'-' * 62}")
        lines.append(f"  Temperature Map (C)  [threshold is per-cell]:")
        lines.append(f"  {'-' * 43}")

        for i in range(gs):
            row_str = "  |"
            for j in range(gs):
                temp = self.temperatures[i, j]
                if temp > self.THROTTLE_TEMP_GRID[i, j]:
                    marker = "!"
                elif temp > 60.0:
                    marker = "~"
                else:
                    marker = "."
                row_str += f" {temp:5.1f}{marker}|"
            lines.append(row_str)

        lines.append(f"  {'-' * 43}")

        if self.is_heterogeneous:
            lines.append(f"  Hardware Layout:")
            lines.append(f"  {'-' * 43}")
            for i in range(gs):
                row_str = "  |"
                for j in range(gs):
                    row_str += f" {str(self.gpu_layout[i, j]):>5s} |"
                lines.append(row_str)
            lines.append(f"  {'-' * 43}")

        lines.append(f"  Max: {np.max(self.temperatures):.1f}C  "
                      f"Mean: {np.mean(self.temperatures):.1f}C  "
                      f"Util: {np.mean(self.power_draw / self.tdp_grid) * 100:.0f}%  "
                      f"Active: {int(np.sum(self.occupancy > 0))}/{self.NUM_CELLS}")
        lines.append(f"{'-' * 62}")

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


def make_checkerboard_layout(model_a: str = "H100", model_b: str = "A100",
                             grid_size: int = 6) -> np.ndarray:
    """Alternating two-device layout - a useful heterogeneity stress test."""
    return np.array(
        [[model_a if (i + j) % 2 == 0 else model_b for j in range(grid_size)]
         for i in range(grid_size)],
        dtype=object,
    )


def make_split_layout(model_a: str = "H100", model_b: str = "A100",
                      grid_size: int = 6) -> np.ndarray:
    """Top-half / bottom-half layout - models a rack with two device tiers."""
    return np.array(
        [[model_a if i < grid_size // 2 else model_b for _ in range(grid_size)]
         for i in range(grid_size)],
        dtype=object,
    )


if __name__ == "__main__":
    from src.data_pipeline.workload_fuser import WorkloadFuser

    fuser = WorkloadFuser(gpu_models=["H100", "A100"], seed=42)
    dataset = fuser.generate_fused_dataset(num_requests=10, source="azure")

    layout = make_checkerboard_layout("H100", "A100")
    env = ThermalGridEnv(
        fused_dataset=dataset,
        gpu_layout_matrix=layout,
        render_mode="human",
    )
    obs, info = env.reset()
    print(f"Heterogeneous: {info['heterogeneous']} | fleet: {info['gpu_models']}")
    print(f"TDP grid  : {env.tdp_grid.min():.0f}W .. {env.tdp_grid.max():.0f}W")
    print(f"Idle grid : {env.idle_power_grid.min():.0f}W .. "
          f"{env.idle_power_grid.max():.0f}W")
    print(f"Throttle  : {env.THROTTLE_TEMP_GRID.min():.0f}C .. "
          f"{env.THROTTLE_TEMP_GRID.max():.0f}C")

    rng = np.random.default_rng(0)
    for step in range(100):
        valid_actions = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 20 == 0 or terminated:
            env.render()
            label = "WAIT" if action == env.WAIT_ACTION else f"GPU[{action}]"
            print(f"  Action: {label} -> Reward: {reward:.2f} | "
                  f"Queue: {info['queue_length']}")
        if terminated:
            print("Episode terminated: trace ingested and backlog cleared.")
            break
