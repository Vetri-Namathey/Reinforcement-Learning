"""
thermal_grid_env.py
===================
Custom Gymnasium environment modeling a 6×6 spatial grid of H100 GPUs
with 2D Laplacian heat diffusion physics.

The RL agent selects which GPU cell (0–35) to place an incoming LLM
inference job on. The physics engine simulates:
  - Power injection from workload placement (prefill → decode → idle)
  - 2D Laplacian heat diffusion between adjacent cells
  - Ambient cooling pulling temperatures toward 25.0 °C
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ThermalGridEnv(gym.Env):
    """
    Gymnasium environment for spatial-phase LLM inference scheduling
    on a 6×6 thermodynamic GPU grid.

    Observation Space (Dict):
        temperatures : Box(6, 6) — current temperature of each GPU (°C)
        power_draw   : Box(6, 6) — current power draw of each GPU (W)
        occupancy    : Box(6, 6) — remaining busy time of each GPU (s)

    Action Space:
        Discrete(36) — index of the GPU cell to place the next job on

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
    AMBIENT_TEMP    = 25.0        # Ambient / inlet air temperature (°C)
    IDLE_POWER      = 50.0        # GPU idle power draw (W)
    THROTTLE_TEMP   = 80.0        # Thermal throttling threshold (°C)

    # Physics coefficients
    THERMAL_DIFFUSIVITY = 0.15    # α — heat spreading rate between cells
    COOLING_COEFF       = 0.05    # γ — cooling toward ambient per step
    POWER_TO_HEAT       = 0.02    # η — Watts → °C conversion factor
    TIMESTEP            = 1.0     # Δt — simulation timestep (seconds)

    # Reward constants
    REWARD_VALID_PLACEMENT  = 10.0
    REWARD_OVERLAP_PENALTY  = -100.0
    THERMAL_PENALTY_COEFF   = 0.5   # α in the quadratic penalty

    def __init__(
        self,
        fused_dataset: pd.DataFrame,
        render_mode: str | None = None,
    ):
        """
        Initialize the thermal grid environment.

        Args:
            fused_dataset: DataFrame from WorkloadFuser with columns:
                [job_id, prefill_watts, prefill_duration, decode_watts, decode_duration]
            render_mode: Optional Gymnasium render mode ('human' or 'ansi').
        """
        super().__init__()
        self.render_mode = render_mode

        # ── Store the fused job queue ──
        self.fused_dataset = fused_dataset.reset_index(drop=True)
        self.total_jobs = len(self.fused_dataset)
        self.current_job_idx = 0

        # ── Action space: pick one of 36 GPU cells ──
        self.action_space = spaces.Discrete(self.GRID_SIZE ** 2)

        # ── Observation space: three 6×6 float matrices ──
        grid_shape = (self.GRID_SIZE, self.GRID_SIZE)
        self.observation_space = spaces.Dict({
            "temperatures": spaces.Box(
                low=0.0, high=200.0, shape=grid_shape, dtype=np.float32
            ),
            "power_draw": spaces.Box(
                low=0.0, high=1000.0, shape=grid_shape, dtype=np.float32
            ),
            "occupancy": spaces.Box(
                low=0.0, high=1e6, shape=grid_shape, dtype=np.float32
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

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API: reset()
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        gs = self.GRID_SIZE

        # Initialize state matrices
        self.temperatures = np.full((gs, gs), self.AMBIENT_TEMP, dtype=np.float32)
        self.power_draw   = np.full((gs, gs), self.IDLE_POWER, dtype=np.float32)
        self.occupancy    = np.zeros((gs, gs), dtype=np.float32)

        # Initialize per-cell workload state
        self.cell_phase          = np.zeros((gs, gs), dtype=np.int32)   # 0=idle
        self.cell_prefill_watts  = np.zeros((gs, gs), dtype=np.float32)
        self.cell_prefill_remain = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_watts   = np.zeros((gs, gs), dtype=np.float32)
        self.cell_decode_remain  = np.zeros((gs, gs), dtype=np.float32)

        # Reset counters
        self.current_job_idx = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.jobs_placed = 0
        self.throttle_events = 0

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API: step()
    # ──────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Execute one environment step:
          1. Attempt to place the current job on the selected GPU cell
          2. Advance workload timers across all cells
          3. Apply 2D Laplacian heat diffusion
          4. Apply ambient cooling
          5. Compute reward

        Args:
            action: Integer in [0, 35] — the GPU cell index.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        row, col = divmod(action, self.GRID_SIZE)
        reward = 0.0

        # ── 1. Job Placement ──
        if self.current_job_idx < self.total_jobs:
            job = self.fused_dataset.iloc[self.current_job_idx]

            if self.occupancy[row, col] > 0:
                # Cell is occupied — overlap penalty
                reward += self.REWARD_OVERLAP_PENALTY
            else:
                # Valid placement — assign workload to this cell
                reward += self.REWARD_VALID_PLACEMENT
                self.cell_phase[row, col]          = 1  # Prefill phase
                self.cell_prefill_watts[row, col]  = job["prefill_watts"]
                self.cell_prefill_remain[row, col] = job["prefill_duration"]
                self.cell_decode_watts[row, col]   = job["decode_watts"]
                self.cell_decode_remain[row, col]  = job["decode_duration"]

                total_duration = job["prefill_duration"] + job["decode_duration"]
                self.occupancy[row, col] = total_duration
                self.power_draw[row, col] = job["prefill_watts"]
                self.jobs_placed += 1

            self.current_job_idx += 1

        # ── 2. Advance Workload Timers ──
        self._advance_workloads()

        # ── 3. Physics: Heat Injection from Power ──
        heat_injection = self.POWER_TO_HEAT * self.power_draw * self.TIMESTEP
        self.temperatures += heat_injection

        # ── 4. Physics: 2D Laplacian Heat Diffusion ──
        self._apply_laplacian_diffusion()

        # ── 5. Physics: Ambient Cooling (Newton's Law) ──
        self.temperatures -= self.COOLING_COEFF * (
            self.temperatures - self.AMBIENT_TEMP
        )

        # ── 6. Thermal Penalty ──
        hot_cells = self.temperatures > self.THROTTLE_TEMP
        if np.any(hot_cells):
            excess = self.temperatures[hot_cells] - self.THROTTLE_TEMP
            thermal_penalty = self.THERMAL_PENALTY_COEFF * np.sum(excess ** 2)
            reward -= thermal_penalty
            self.throttle_events += int(np.sum(hot_cells))

        # ── 7. Termination ──
        terminated = self.current_job_idx >= self.total_jobs
        truncated = False

        self.total_reward += reward
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

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
                        self.power_draw[i, j] = self.IDLE_POWER
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
            T_new = T + α · ∇²T

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
        self.temperatures += self.THERMAL_DIFFUSIVITY * laplacian

        # Clamp to physical bounds (cannot go below ambient)
        self.temperatures = np.maximum(self.temperatures, self.AMBIENT_TEMP)

    # ──────────────────────────────────────────────────────────────
    # Observation & Info Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """Return the current observation dictionary."""
        return {
            "temperatures": self.temperatures.copy(),
            "power_draw":   self.power_draw.copy(),
            "occupancy":    self.occupancy.copy(),
        }

    def _get_info(self) -> dict:
        """Return auxiliary diagnostic information."""
        return {
            "step":             self.step_count,
            "jobs_placed":      self.jobs_placed,
            "jobs_remaining":   self.total_jobs - self.current_job_idx,
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
        """Generate an ANSI text representation of the thermal grid."""
        gs = self.GRID_SIZE
        lines = []
        lines.append(f"\n{'─' * 55}")
        lines.append(f"  THERMAL GRID STATE  │  Step: {self.step_count}  │  "
                      f"Jobs placed: {self.jobs_placed}/{self.total_jobs}")
        lines.append(f"{'─' * 55}")
        lines.append(f"  Temperature Map (°C):")
        lines.append(f"  {'─' * 43}")

        for i in range(gs):
            row_str = "  │"
            for j in range(gs):
                temp = self.temperatures[i, j]
                # Color coding: normal(<60), warm(60-80), HOT(>80)
                if temp > self.THROTTLE_TEMP:
                    marker = "🔥"
                elif temp > 60.0:
                    marker = "🟡"
                else:
                    marker = "🟢"
                row_str += f" {temp:5.1f}{marker}│"
            lines.append(row_str)

        lines.append(f"  {'─' * 43}")
        lines.append(f"  Max: {np.max(self.temperatures):.1f}°C  "
                      f"Mean: {np.mean(self.temperatures):.1f}°C  "
                      f"Active: {int(np.sum(self.occupancy > 0))}/36")
        lines.append(f"{'─' * 55}")

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_fusion_pipeline import WorkloadFuser

    fuser = WorkloadFuser(seed=42)
    dataset = fuser.generate_fused_dataset(num_requests=10)

    env = ThermalGridEnv(fused_dataset=dataset, render_mode="human")
    obs, info = env.reset()
    print(f"Initial info: {info}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"  Action: GPU[{action}] → Reward: {reward:.2f}")
        if terminated:
            break
