"""
Custom Gymnasium environment modeling a 6x6 spatial grid of GPUs with
2D Laplacian heat diffusion physics.

The RL agent schedules LLM inference jobs from a persistent backlog queue
onto GPU cells. Jobs arrive over continuous simulation time per the trace's
real arrival timestamps; unplaceable jobs wait in the queue (never dropped)
and accrue an SLA latency penalty.

Action space: Discrete(37) — 0..35 place head-of-queue job on cell, 36 = Wait.
Invalid placements are prevented by an action mask in the observation.

Reward:  R = R_place − R_sla − R_thermal
  R_place   = +10 if a job was placed this step
  R_sla     = 0.1 × Σ (current_time − arrival_time) over queued jobs
  R_thermal = 0.5 × Σ max(0, T − 80°C)² over all cells
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from data_fusion_pipeline import GPUMismatchError, get_hardware_profile


class ThermalGridEnv(gym.Env):
    """
    Spatial-phase LLM inference scheduling on a 6x6 thermodynamic GPU grid
    with a persistent backlog queue.

    Observation Space (Dict):
        temperatures : Box(6, 6)  — temperature of each GPU (°C)
        power_draw   : Box(6, 6)  — power draw of each GPU (W)
        occupancy    : Box(6, 6)  — remaining busy time of each GPU (s)
        action_mask  : Box(37,)   — 1 = action currently valid
        queue_stats  : Box(2,)    — [queue_length, head_job_wait_time (s)]

    Physics per step: power injection from workload phases (prefill →
    decode → idle), 2D Laplacian heat diffusion (4-neighbor stencil),
    Newton's-law cooling toward ambient.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    GRID_SIZE       = 6
    NUM_CELLS       = GRID_SIZE ** 2
    WAIT_ACTION     = NUM_CELLS
    AMBIENT_TEMP    = 25.0        # °C
    THROTTLE_TEMP   = 80.0        # °C

    # Physics coefficients (per-second rates, scaled by TIMESTEP)
    THERMAL_DIFFUSIVITY = 0.15    # α — heat spreading rate between cells (1/s)
    COOLING_COEFF       = 0.05    # γ — cooling toward ambient (1/s)
    POWER_TO_HEAT       = 0.02    # η — Watts → °C conversion factor
    TIMESTEP            = 0.1     # Δt — simulation timestep (s)

    REWARD_VALID_PLACEMENT = 10.0
    SLA_PENALTY_WEIGHT     = 0.1
    THERMAL_PENALTY_COEFF  = 0.5

    def __init__(
        self,
        fused_dataset: pd.DataFrame,
        gpu_model: str = "H100",
        render_mode: str | None = None,
    ):
        """
        Args:
            fused_dataset: DataFrame from WorkloadFuser with columns
                [job_id, arrival_time, prefill_watts, prefill_duration,
                 decode_watts, decode_duration] and attrs["gpu_model"].
            gpu_model: GPU model populating the grid. Must match the GPU
                the fused dataset was built for (GPUMismatchError otherwise).
            render_mode: Optional Gymnasium render mode ('human' or 'ansi').
        """
        super().__init__()
        self.render_mode = render_mode

        self.gpu_profile = get_hardware_profile(gpu_model)
        self.gpu_model = self.gpu_profile["gpu_model"]
        self.idle_power = self.gpu_profile["idle_power"]
        self.tdp_watts = self.gpu_profile["tdp_watts"]

        dataset_gpu = fused_dataset.attrs.get("gpu_model")
        if dataset_gpu is not None and dataset_gpu.upper() != self.gpu_model.upper():
            raise GPUMismatchError(
                f"GPU mismatch: fused dataset was built for '{dataset_gpu}' "
                f"but the grid is populated with '{self.gpu_model}' GPUs. "
                "Wattages/durations are only valid on matching hardware."
            )

        self.fused_dataset = (
            fused_dataset.sort_values("arrival_time").reset_index(drop=True)
        )
        self.total_jobs = len(self.fused_dataset)
        self._job_records = self.fused_dataset.to_dict("records")

        self.queue = []
        self.current_time = 0.0
        self.next_arrival_idx = 0

        self.action_space = spaces.Discrete(self.NUM_CELLS + 1)

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

        self.temperatures = None
        self.power_draw   = None
        self.occupancy    = None

        # Per-cell workload phase: 0=idle, 1=prefill, 2=decode
        self.cell_phase          = None
        self.cell_prefill_watts  = None
        self.cell_prefill_remain = None
        self.cell_decode_watts   = None
        self.cell_decode_remain  = None

        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.throttle_events = 0
        self.dropped_jobs = 0        # always 0: backlog queue never drops

        self.successful_jobs = []          # per-placed-job stats for TTFT
        self.throttling_steps = 0          # steps with any cell > THROTTLE_TEMP
        self.total_steps = 0
        self.spatial_variance_sum = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        gs = self.GRID_SIZE

        self.temperatures = np.full((gs, gs), self.AMBIENT_TEMP, dtype=np.float32)
        self.power_draw   = np.full((gs, gs), self.idle_power, dtype=np.float32)
        self.occupancy    = np.zeros((gs, gs), dtype=np.float32)

        self.cell_phase          = np.zeros((gs, gs), dtype=np.int32)
        self.cell_prefill_watts  = np.zeros((gs, gs), dtype=np.float32)
        self.cell_prefill_remain = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_watts   = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_remain  = np.zeros((gs, gs), dtype=np.float32)

        self.queue = []
        self.current_time = 0.0
        self.next_arrival_idx = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.throttle_events = 0
        self.dropped_jobs = 0

        self.successful_jobs = []
        self.throttling_steps = 0
        self.total_steps = 0
        self.spatial_variance_sum = 0.0

        # Ingest jobs arriving exactly at t=0 so the first action mask
        # reflects them
        self._ingest_arrivals()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """
        One step: advance time and ingest arrivals, execute the (masked)
        action, run thermodynamics, compute R = R_place − R_sla − R_thermal.
        """
        self.step_count += 1
        self.total_steps += 1

        self.current_time += self.TIMESTEP
        self._ingest_arrivals()

        mask = self._compute_action_mask()
        assert mask[action] == 1, (
            f"Invalid action {action}: masked out "
            f"(queue_length={len(self.queue)}, mask={mask.tolist()})"
        )

        job_placed = False
        if action != self.WAIT_ACTION and len(self.queue) > 0:
            job = self.queue.pop(0)
            row, col = divmod(action, self.GRID_SIZE)

            self.cell_phase[row, col]          = 1
            self.cell_prefill_watts[row, col]  = job["prefill_watts"]
            self.cell_prefill_remain[row, col] = job["prefill_duration"]
            self.cell_decode_watts[row, col]   = job["decode_watts"]
            self.cell_decode_remain[row, col]  = job["decode_duration"]

            total_duration = job["prefill_duration"] + job["decode_duration"]
            self.occupancy[row, col] = total_duration
            self.power_draw[row, col] = job["prefill_watts"]
            self.jobs_placed += 1
            job_placed = True

            # TTFT here = queueing delay until placement (excludes prefill)
            self.successful_jobs.append({
                "job_id":           job["job_id"],
                "arrival_time":     job["arrival_time"],
                "placement_time":   self.current_time,
                "prefill_duration": job["prefill_duration"],
                "ttft":             self.current_time - job["arrival_time"],
            })

        self._advance_workloads()

        heat_injection = self.POWER_TO_HEAT * self.power_draw * self.TIMESTEP
        self.temperatures += heat_injection

        self._apply_laplacian_diffusion()

        self.temperatures -= self.COOLING_COEFF * self.TIMESTEP * (
            self.temperatures - self.AMBIENT_TEMP
        )

        if np.any(self.temperatures > self.THROTTLE_TEMP):
            self.throttling_steps += 1
        self.spatial_variance_sum += float(np.var(self.temperatures))

        r_place = self.REWARD_VALID_PLACEMENT if job_placed else 0.0

        queue_wait_sum = sum(
            self.current_time - j["arrival_time"] for j in self.queue
        )
        r_sla = self.SLA_PENALTY_WEIGHT * queue_wait_sum

        excess = np.maximum(0.0, self.temperatures - self.THROTTLE_TEMP)
        r_thermal = self.THERMAL_PENALTY_COEFF * float(np.sum(excess ** 2))
        self.throttle_events += int(np.sum(excess > 0))

        reward = r_place - r_sla - r_thermal

        # Terminate when the full trace is ingested and the backlog cleared
        terminated = (
            self.next_arrival_idx >= self.total_jobs and len(self.queue) == 0
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

        return self._get_obs(), float(reward), terminated, truncated, info

    def _ingest_arrivals(self):
        """Move trace jobs with arrival_time <= current_time into the queue."""
        while (
            self.next_arrival_idx < self.total_jobs
            and self._job_records[self.next_arrival_idx]["arrival_time"]
                <= self.current_time
        ):
            self.queue.append(self._job_records[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _compute_action_mask(self) -> np.ndarray:
        """
        Build the 37-element binary action mask: free cells are valid,
        Wait is always valid, and an empty queue masks out all placements.
        """
        mask = np.zeros(self.NUM_CELLS + 1, dtype=np.int8)
        mask[self.WAIT_ACTION] = 1
        if len(self.queue) > 0:
            mask[: self.NUM_CELLS] = (self.occupancy.ravel() == 0).astype(np.int8)
        return mask

    def _advance_workloads(self):
        """Advance cell workload timers; handle prefill → decode → idle."""
        dt = self.TIMESTEP
        gs = self.GRID_SIZE

        for i in range(gs):
            for j in range(gs):
                phase = self.cell_phase[i, j]

                if phase == 1:
                    self.cell_prefill_remain[i, j] -= dt
                    if self.cell_prefill_remain[i, j] <= 0:
                        self.cell_phase[i, j] = 2
                        self.power_draw[i, j] = self.cell_decode_watts[i, j]

                elif phase == 2:
                    self.cell_decode_remain[i, j] -= dt
                    if self.cell_decode_remain[i, j] <= 0:
                        self.cell_phase[i, j] = 0
                        self.power_draw[i, j] = self.idle_power
                        self.occupancy[i, j] = 0.0
                        self.cell_prefill_watts[i, j]  = 0.0
                        self.cell_decode_watts[i, j]   = 0.0

                if self.occupancy[i, j] > 0:
                    self.occupancy[i, j] = max(0.0, self.occupancy[i, j] - dt)

    def _apply_laplacian_diffusion(self):
        """
        Discrete 2D Laplacian heat diffusion, 4-neighbor stencil:
            T_new = T + α · ∇²T · Δt
        Neumann (zero-flux) boundary via edge padding.
        """
        padded = np.pad(self.temperatures, pad_width=1, mode="edge")

        laplacian = (
            padded[0:-2, 1:-1] +
            padded[2:,   1:-1] +
            padded[1:-1, 0:-2] +
            padded[1:-1, 2:]   -
            4.0 * self.temperatures
        )

        self.temperatures += self.THERMAL_DIFFUSIVITY * self.TIMESTEP * laplacian
        self.temperatures = np.maximum(self.temperatures, self.AMBIENT_TEMP)

    def _get_obs(self) -> dict:
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
        return {
            "gpu_model":        self.gpu_model,
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
            "active_cells":     int(np.sum(self.occupancy > 0)),
            "spatial_variance": float(np.var(self.temperatures)),
            "tttr":             self.throttling_steps / max(self.total_steps, 1),
        }

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """ASCII grid rendering (works on any console encoding).
        Markers: '!' hot (>throttle), '~' warm (>60C), '.' ok."""
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
