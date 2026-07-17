"""
thermal_grid_env.py
===================
Custom Gymnasium environment modeling a 6×6 spatial grid of H100 GPUs
with 2D Laplacian heat diffusion physics.

The RL agent schedules LLM inference jobs from a persistent backlog queue
onto GPU cells. Jobs arrive over continuous simulation time according to
the trace's real arrival timestamps; jobs that cannot be placed yet wait
in the queue (never dropped) and accrue an SLA latency penalty.

Action space: Discrete(37)
  - 0..35 : place the head-of-queue job on GPU cell (row, col)
  - 36    : WAIT (advance time without placing)

Invalid placements are prevented by an action mask exposed in the
observation (occupied cells masked out; all placements masked out when
the queue is empty), replacing the old flat -100 overlap penalty.

Reward:  R = R_place − R_sla − R_thermal
  R_place   = +10 if a job was placed this step
  R_sla     = 0.1 × Σ (current_time − arrival_time) over queued jobs
  R_thermal = 0.5 × Σ max(0, T − 80°C)² over all cells

Physics engine per step:
  - Power injection from workload phases (prefill → decode → idle)
  - 2D Laplacian heat diffusion between adjacent cells
  - Ambient cooling pulling temperatures toward 25.0 °C
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from data_fusion_pipeline import GPUMismatchError, get_hardware_profile


class ThermalGridEnv(gym.Env):
    """
    Gymnasium environment for spatial-phase LLM inference scheduling
    on a 6×6 thermodynamic GPU grid with a persistent backlog queue.

    Observation Space (Dict):
        temperatures : Box(6, 6)  — current temperature of each GPU (°C)
        power_draw   : Box(6, 6)  — current power draw of each GPU (W)
        occupancy    : Box(6, 6)  — remaining busy time of each GPU (s)
        action_mask  : Box(37,)   — 1 = action currently valid, 0 = invalid
        queue_stats  : Box(2,)    — [queue_length, head_job_wait_time (s)]

    Action Space:
        Discrete(37) — 0..35 place head-of-queue job on that GPU cell,
                       36 = WAIT (always valid)

    Physics:
        - 2D discrete Laplacian heat diffusion (4-neighbor stencil)
        - Newton's law of cooling toward ambient (25.0 °C)
        - Power-to-heat conversion: ΔT = η × P × Δt
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # ──────────────────────────────────────────────────────────────
    # Grid & Physics Constants
    # ──────────────────────────────────────────────────────────────
    GRID_SIZE       = 6           # 6×6 grid → 36 GPUs
    NUM_CELLS       = GRID_SIZE ** 2
    WAIT_ACTION     = NUM_CELLS   # Action index 36 = "Wait"
    AMBIENT_TEMP    = 25.0        # Ambient / inlet air temperature (°C)
    THROTTLE_TEMP   = 80.0        # Thermal throttling threshold (°C)

    # Physics coefficients (per-second rates, scaled by TIMESTEP)
    THERMAL_DIFFUSIVITY = 0.15    # α — heat spreading rate between cells (1/s)
    COOLING_COEFF       = 0.05    # γ — cooling toward ambient (1/s)
    POWER_TO_HEAT       = 0.02    # η — Watts → °C conversion factor
    TIMESTEP            = 0.1     # Δt — simulation timestep (seconds)

    # Reward constants:  R = R_place − R_sla − R_thermal
    REWARD_VALID_PLACEMENT = 10.0
    SLA_PENALTY_WEIGHT     = 0.1    # × Σ queue wait times (s)
    THERMAL_PENALTY_COEFF  = 0.5    # × Σ max(0, T − 80)²

    def __init__(
        self,
        fused_dataset: pd.DataFrame,
        gpu_model: str = "H100",
        render_mode: str | None = None,
    ):
        """
        Initialize the thermal grid environment.

        Args:
            fused_dataset: DataFrame from WorkloadFuser with columns:
                [job_id, arrival_time, prefill_watts, prefill_duration,
                 decode_watts, decode_duration]
                and attrs["gpu_model"] tagging the GPU it was fused for.
            gpu_model: GPU model populating the grid. Must match the GPU
                the fused dataset was built for (GPUMismatchError otherwise).
            render_mode: Optional Gymnasium render mode ('human' or 'ansi').
        """
        super().__init__()
        self.render_mode = render_mode

        # ── GPU hardware profile for the grid cells ──
        self.gpu_profile = get_hardware_profile(gpu_model)
        self.gpu_model = self.gpu_profile["gpu_model"]
        self.idle_power = self.gpu_profile["idle_power"]
        self.tdp_watts = self.gpu_profile["tdp_watts"]

        # ── GPU-matching guard: dataset physics must be for this GPU ──
        dataset_gpu = fused_dataset.attrs.get("gpu_model")
        if dataset_gpu is not None and dataset_gpu.upper() != self.gpu_model.upper():
            raise GPUMismatchError(
                f"GPU mismatch: fused dataset was built for '{dataset_gpu}' "
                f"but the grid is populated with '{self.gpu_model}' GPUs. "
                "Wattages/durations are only valid on matching hardware."
            )

        # ── Store the fused job trace (arrival-ordered) ──
        self.fused_dataset = (
            fused_dataset.sort_values("arrival_time").reset_index(drop=True)
        )
        self.total_jobs = len(self.fused_dataset)
        self._job_records = self.fused_dataset.to_dict("records")

        # ── Persistent backlog queue & simulation clock ──
        self.queue = []              # jobs arrived but not yet placed
        self.current_time = 0.0      # continuous simulation time (s)
        self.next_arrival_idx = 0    # next trace row to ingest

        # ── Action space: 36 GPU cells + 1 Wait action ──
        self.action_space = spaces.Discrete(self.NUM_CELLS + 1)

        # ── Observation space ──
        grid_shape = (self.GRID_SIZE, self.GRID_SIZE)
        self.observation_space = spaces.Dict({
            "temperatures": spaces.Box(
                low=0.0, high=200.0, shape=grid_shape, dtype=np.float32
            ),
            "power_draw": spaces.Box(
                low=0.0, high=self.tdp_watts, shape=grid_shape, dtype=np.float32
            ),
            "occupancy": spaces.Box(
                low=0.0, high=1e6, shape=grid_shape, dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(self.NUM_CELLS + 1,), dtype=np.int8
            ),
            "queue_stats": spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            ),
        })

        # ── Internal state matrices (initialized in reset()) ──
        self.temperatures = None
        self.power_draw   = None
        self.occupancy    = None

        # ── Per-cell workload tracking ──
        # Each cell tracks: [phase, watts, remaining_time]
        #   phase: 0=idle, 1=prefill, 2=decode
        self.cell_phase          = None
        self.cell_prefill_watts  = None
        self.cell_prefill_remain = None
        self.cell_decode_watts   = None
        self.cell_decode_remain  = None

        # ── Metrics ──
        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.throttle_events = 0
        self.dropped_jobs = 0        # always 0: backlog queue never drops

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API: reset()
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        gs = self.GRID_SIZE

        # Initialize state matrices
        self.temperatures = np.full((gs, gs), self.AMBIENT_TEMP, dtype=np.float32)
        self.power_draw   = np.full((gs, gs), self.idle_power, dtype=np.float32)
        self.occupancy    = np.zeros((gs, gs), dtype=np.float32)

        # Initialize per-cell workload state
        self.cell_phase          = np.zeros((gs, gs), dtype=np.int32)   # 0=idle
        self.cell_prefill_watts  = np.zeros((gs, gs), dtype=np.float32)
        self.cell_prefill_remain = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_watts   = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_remain  = np.zeros((gs, gs), dtype=np.float32)

        # Reset queue, clock, and counters
        self.queue = []
        self.current_time = 0.0
        self.next_arrival_idx = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.throttle_events = 0
        self.dropped_jobs = 0

        # Ingest any jobs arriving exactly at t=0 so the first
        # observation's action mask reflects them
        self._ingest_arrivals()

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API: step()
    # ──────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Execute one environment step:
          A. Advance simulation time and ingest newly-arrived jobs
          B. Execute the (masked) action: place head-of-queue job or Wait
          C. Thermodynamics: timers, heat injection, diffusion, cooling
          D. Reward: R = R_place − R_sla − R_thermal

        Args:
            action: Integer in [0, 36] — GPU cell index, or 36 = Wait.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # ── A. Time Progression & Ingestion ──
        self.current_time += self.TIMESTEP
        self._ingest_arrivals()

        # ── B. Action Execution ──
        mask = self._compute_action_mask()
        assert mask[action] == 1, (
            f"Invalid action {action}: masked out "
            f"(queue_length={len(self.queue)}, mask={mask.tolist()})"
        )

        job_placed = False
        if action == self.WAIT_ACTION or len(self.queue) == 0:
            pass  # Wait — grid untouched, time already advanced
        else:
            job = self.queue.pop(0)
            row, col = divmod(action, self.GRID_SIZE)

            self.cell_phase[row, col]          = 1  # Prefill phase
            self.cell_prefill_watts[row, col]  = job["prefill_watts"]
            self.cell_prefill_remain[row, col] = job["prefill_duration"]
            self.cell_decode_watts[row, col]   = job["decode_watts"]
            self.cell_decode_remain[row, col]  = job["decode_duration"]

            total_duration = job["prefill_duration"] + job["decode_duration"]
            self.occupancy[row, col] = total_duration
            self.power_draw[row, col] = job["prefill_watts"]
            self.jobs_placed += 1
            job_placed = True

        # ── C. Thermodynamics ──
        # C1. Advance workload timers (prefill → decode → idle)
        self._advance_workloads()

        # C2. Heat injection from current power draw
        heat_injection = self.POWER_TO_HEAT * self.power_draw * self.TIMESTEP
        self.temperatures += heat_injection

        # C3. 2D Laplacian heat diffusion
        self._apply_laplacian_diffusion()

        # C4. Ambient cooling (Newton's law) toward 25.0 °C
        self.temperatures -= self.COOLING_COEFF * self.TIMESTEP * (
            self.temperatures - self.AMBIENT_TEMP
        )

        # ── D. Reward: R = R_place − R_sla − R_thermal ──
        r_place = self.REWARD_VALID_PLACEMENT if job_placed else 0.0

        queue_wait_sum = sum(
            self.current_time - j["arrival_time"] for j in self.queue
        )
        r_sla = self.SLA_PENALTY_WEIGHT * queue_wait_sum

        excess = np.maximum(0.0, self.temperatures - self.THROTTLE_TEMP)
        r_thermal = self.THERMAL_PENALTY_COEFF * float(np.sum(excess ** 2))
        self.throttle_events += int(np.sum(excess > 0))

        reward = r_place - r_sla - r_thermal

        # ── Termination: full trace ingested and backlog cleared ──
        terminated = (
            self.next_arrival_idx >= self.total_jobs and len(self.queue) == 0
        )
        truncated = False

        self.total_reward += reward
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────────────────────
    # Queue: Arrival Ingestion
    # ──────────────────────────────────────────────────────────────

    def _ingest_arrivals(self):
        """Move trace jobs with arrival_time <= current_time into the queue."""
        while (
            self.next_arrival_idx < self.total_jobs
            and self._job_records[self.next_arrival_idx]["arrival_time"]
                <= self.current_time
        ):
            self.queue.append(self._job_records[self.next_arrival_idx])
            self.next_arrival_idx += 1

    # ──────────────────────────────────────────────────────────────
    # Action Masking
    # ──────────────────────────────────────────────────────────────

    def _compute_action_mask(self) -> np.ndarray:
        """
        Build the 37-element binary action mask.

        Indices 0-35: 1 if the GPU cell is free (occupancy == 0), else 0.
        Index 36 (Wait): always 1.
        If the queue is empty, all placement actions are masked to 0
        (the agent is forced to Wait).
        """
        mask = np.zeros(self.NUM_CELLS + 1, dtype=np.int8)
        mask[self.WAIT_ACTION] = 1
        if len(self.queue) > 0:
            mask[: self.NUM_CELLS] = (self.occupancy.ravel() == 0).astype(np.int8)
        return mask

    # ──────────────────────────────────────────────────────────────
    # Physics: Workload Timer Advancement
    # ──────────────────────────────────────────────────────────────

    def _advance_workloads(self):
        """
        Advance all cell workload timers by one timestep.
        Handles phase transitions: prefill → decode → idle.
        """
        dt = self.TIMESTEP
        gs = self.GRID_SIZE

        for i in range(gs):
            for j in range(gs):
                phase = self.cell_phase[i, j]

                if phase == 1:  # Prefill phase
                    self.cell_prefill_remain[i, j] -= dt
                    if self.cell_prefill_remain[i, j] <= 0:
                        # Transition to decode phase
                        self.cell_phase[i, j] = 2
                        self.power_draw[i, j] = self.cell_decode_watts[i, j]

                elif phase == 2:  # Decode phase
                    self.cell_decode_remain[i, j] -= dt
                    if self.cell_decode_remain[i, j] <= 0:
                        # Job complete — return to idle
                        self.cell_phase[i, j] = 0
                        self.power_draw[i, j] = self.idle_power
                        self.occupancy[i, j] = 0.0
                        self.cell_prefill_watts[i, j]  = 0.0
                        self.cell_decode_watts[i, j]   = 0.0

                # Decrement occupancy timer
                if self.occupancy[i, j] > 0:
                    self.occupancy[i, j] = max(0.0, self.occupancy[i, j] - dt)

    # ──────────────────────────────────────────────────────────────
    # Physics: 2D Laplacian Heat Diffusion
    # ──────────────────────────────────────────────────────────────

    def _apply_laplacian_diffusion(self):
        """
        Apply discrete 2D Laplacian heat diffusion using a 4-neighbor stencil.

        ∇²T[i,j] = T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4·T[i,j]

        Temperature update:
            T_new = T + α · ∇²T · Δt

        Boundary condition: Neumann (zero-flux) — padded with edge values.
        """
        # Pad with edge values (zero-flux Neumann boundary)
        padded = np.pad(self.temperatures, pad_width=1, mode="edge")

        # 4-neighbor Laplacian stencil
        laplacian = (
            padded[0:-2, 1:-1] +   # Up
            padded[2:,   1:-1] +   # Down
            padded[1:-1, 0:-2] +   # Left
            padded[1:-1, 2:]   -   # Right
            4.0 * self.temperatures
        )

        # Apply diffusion
        self.temperatures += self.THERMAL_DIFFUSIVITY * self.TIMESTEP * laplacian

        # Clamp to physical bounds (cannot go below ambient)
        self.temperatures = np.maximum(self.temperatures, self.AMBIENT_TEMP)

    # ──────────────────────────────────────────────────────────────
    # Observation & Info Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """Return the current observation dictionary (with action mask)."""
        head_wait = (
            self.current_time - self.queue[0]["arrival_time"]
            if self.queue else 0.0
        )
        return {
            "temperatures": self.temperatures.copy(),
            "power_draw":   self.power_draw.copy(),
            "occupancy":    self.occupancy.copy(),
            "action_mask":  self._compute_action_mask(),
            "queue_stats":  np.array(
                [len(self.queue), head_wait], dtype=np.float32
            ),
        }

    def _get_info(self) -> dict:
        """Return auxiliary diagnostic information."""
        return {
            "gpu_model":        self.gpu_model,
            "step":             self.step_count,
            "current_time":     round(self.current_time, 4),
            "jobs_placed":      self.jobs_placed,
            "jobs_remaining":   self.total_jobs - self.jobs_placed,
            "queue_length":     len(self.queue),
            "dropped_jobs":     self.dropped_jobs,   # backlog queue: always 0
            "max_temperature":  float(np.max(self.temperatures)),
            "mean_temperature": float(np.mean(self.temperatures)),
            "throttle_events":  self.throttle_events,
            "total_reward":     self.total_reward,
            "active_cells":     int(np.sum(self.occupancy > 0)),
        }

    # ──────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────

    def render(self):
        """Render the grid state to the console."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Generate an ASCII text representation of the thermal grid.

        ASCII-only so it renders on any console encoding (Windows cp1252
        included). Markers: '!' hot (>throttle), '~' warm (>60C), '.' ok.
        """
        gs = self.GRID_SIZE
        lines = []
        lines.append(f"\n{'-' * 55}")
        lines.append(f"  THERMAL GRID STATE ({self.gpu_model})  |  "
                      f"Step: {self.step_count}  |  t={self.current_time:.1f}s  |  "
                      f"Placed: {self.jobs_placed}/{self.total_jobs}  |  "
                      f"Queue: {len(self.queue)}")
        lines.append(f"{'-' * 55}")
        lines.append(f"  Temperature Map (C):")
        lines.append(f"  {'-' * 43}")

        for i in range(gs):
            row_str = "  |"
            for j in range(gs):
                temp = self.temperatures[i, j]
                if temp > self.THROTTLE_TEMP:
                    marker = "!"
                elif temp > 60.0:
                    marker = "~"
                else:
                    marker = "."
                row_str += f" {temp:5.1f}{marker}|"
            lines.append(row_str)

        lines.append(f"  {'-' * 43}")
        lines.append(f"  Max: {np.max(self.temperatures):.1f}C  "
                      f"Mean: {np.mean(self.temperatures):.1f}C  "
                      f"Active: {int(np.sum(self.occupancy > 0))}/36")
        lines.append(f"{'-' * 55}")

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_fusion_pipeline import WorkloadFuser

    fuser = WorkloadFuser(gpu_model="H100", seed=42)
    dataset = fuser.generate_fused_dataset(num_requests=10, source="azure")

    env = ThermalGridEnv(fused_dataset=dataset, gpu_model="H100", render_mode="human")
    obs, info = env.reset()
    print(f"Initial info: {info}")

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
