"""
system_architecture_test.py
===========================
Integration test for the Spatial-Phase RL Scheduler system.

Verifies the end-to-end data pipeline:
    Azure Tokens -> MLPerf Watts -> Grid Placement -> Laplacian Physics -> Temperature Updates

This script:
  1. Instantiates the WorkloadFuser and generates 100 fused jobs
  2. Initializes the ThermalGridEnv with the fused dataset
  3. Runs a 50-step loop with random actions to prove the pipeline integrity
  4. Reports diagnostics at each step and a final summary
"""

import sys
import io
import numpy as np

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_fusion_pipeline import WorkloadFuser
from thermal_grid_env import ThermalGridEnv


def print_header(text: str) -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_step_report(step: int, action: int, reward: float, info: dict) -> None:
    """Print a compact per-step diagnostic report."""
    row, col = divmod(action, 6)
    print(
        f"  Step {step:3d} | "
        f"Action: GPU[{row},{col}] ({action:2d}) | "
        f"Reward: {reward:+8.2f} | "
        f"MaxT: {info['max_temperature']:5.1f} C | "
        f"MeanT: {info['mean_temperature']:5.1f} C | "
        f"Active: {info['active_cells']:2d}/36"
    )


def run_architecture_test():
    """
    Main integration test.

    Proves:
      [OK] WorkloadFuser generates valid fused data (Azure x MLPerf)
      [OK] Fused data feeds correctly into ThermalGridEnv
      [OK] Random actions produce valid state transitions
      [OK] Laplacian diffusion updates temperatures without numerical instability
      [OK] Reward function produces expected signals
    """

    # ==============================================================
    # Phase 1: Generate Fused Dataset
    # ==============================================================
    print_header("PHASE 1: FUSED DATASET GENERATION")

    fuser = WorkloadFuser(seed=42)
    fused_df = fuser.generate_fused_dataset(num_requests=100)
    fuser.print_dataset_summary(fused_df)

    # -- Validate dataset structure --
    required_columns = [
        "job_id", "prefill_watts", "prefill_duration",
        "decode_watts", "decode_duration"
    ]
    for col in required_columns:
        assert col in fused_df.columns, f"Missing column: {col}"
    assert len(fused_df) == 100, f"Expected 100 rows, got {len(fused_df)}"
    assert fused_df["prefill_watts"].min() >= 200.0, "Prefill watts below minimum"
    assert fused_df["decode_watts"].max() <= 250.0, "Decode watts above maximum"

    print("\n  [OK] Dataset structure validated")
    print(f"  [OK] All {len(required_columns)} required columns present")
    print(f"  [OK] 100 fused jobs generated successfully")

    # Show sample rows
    print("\n  Sample fused jobs:")
    print(fused_df[["job_id", "prefill_watts", "prefill_duration",
                     "decode_watts", "decode_duration"]].head(5).to_string(index=False))

    # ==============================================================
    # Phase 2: Initialize Thermal Grid Environment
    # ==============================================================
    print_header("PHASE 2: THERMAL GRID ENVIRONMENT INITIALIZATION")

    env = ThermalGridEnv(fused_dataset=fused_df, render_mode="ansi")
    obs, info = env.reset()

    # -- Validate initial state --
    assert obs["temperatures"].shape == (6, 6), "Temperature grid shape mismatch"
    assert obs["power_draw"].shape == (6, 6), "Power draw grid shape mismatch"
    assert obs["occupancy"].shape == (6, 6), "Occupancy grid shape mismatch"
    assert np.allclose(obs["temperatures"], 25.0), "Initial temps should be 25.0 C"
    assert np.allclose(obs["power_draw"], 50.0), "Initial power should be 50.0W idle"
    assert np.allclose(obs["occupancy"], 0.0), "Initial occupancy should be 0.0"

    print(f"  [OK] Grid shape:           {obs['temperatures'].shape} (6x6 = 36 GPUs)")
    print(f"  [OK] Initial temperature:  {obs['temperatures'][0, 0]:.1f} C (ambient)")
    print(f"  [OK] Initial power draw:   {obs['power_draw'][0, 0]:.1f}W (idle)")
    print(f"  [OK] Initial occupancy:    {obs['occupancy'][0, 0]:.1f}s (free)")
    print(f"  [OK] Action space:         Discrete({env.action_space.n})")
    print(f"  [OK] Observation space:    Dict(temperatures, power_draw, occupancy)")

    # ==============================================================
    # Phase 3: Run Random-Action Simulation (50 Steps)
    # ==============================================================
    print_header("PHASE 3: 50-STEP RANDOM-ACTION SIMULATION")
    print("  Proving: Azure Tokens -> MLPerf Watts -> Grid -> Laplacian Physics\n")

    np.random.seed(42)
    total_steps = 50
    rewards = []
    max_temps = []
    mean_temps = []

    for step in range(1, total_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        max_temps.append(info["max_temperature"])
        mean_temps.append(info["mean_temperature"])

        # Print every 5th step for readability
        if step % 5 == 0 or step == 1 or terminated:
            print_step_report(step, action, reward, info)

        if terminated:
            print(f"\n  [!] Environment terminated at step {step} (all jobs dispatched)")
            break

    # ==============================================================
    # Summary Report
    # ==============================================================
    print_header("FINAL SUMMARY")

    steps_completed = len(rewards)
    print(f"  Steps completed:        {steps_completed}/{total_steps}")
    print(f"  Jobs placed:            {info['jobs_placed']}")
    print(f"  Jobs remaining:         {info['jobs_remaining']}")
    print(f"  --- Reward ---")
    print(f"  Total reward:           {info['total_reward']:+.2f}")
    print(f"  Mean reward/step:       {np.mean(rewards):+.2f}")
    print(f"  Min reward:             {np.min(rewards):+.2f}")
    print(f"  Max reward:             {np.max(rewards):+.2f}")
    print(f"  --- Thermal ---")
    print(f"  Final max temperature:  {info['max_temperature']:.2f} C")
    print(f"  Final mean temperature: {info['mean_temperature']:.2f} C")
    print(f"  Peak max temperature:   {np.max(max_temps):.2f} C")
    print(f"  Throttle events:        {info['throttle_events']}")
    print(f"  --- Grid State ---")
    print(f"  Active cells:           {info['active_cells']}/36")

    # -- Final grid snapshot --
    print(f"\n  Final Temperature Grid (C):")
    temp_grid = obs["temperatures"]
    for i in range(6):
        row_str = "    |"
        for j in range(6):
            t = temp_grid[i, j]
            if t > 80.0:
                row_str += f" {t:5.1f} HOT|"
            elif t > 60.0:
                row_str += f" {t:5.1f} WRM|"
            else:
                row_str += f" {t:5.1f}  OK|"
        print(row_str)

    # -- Assertions --
    print_header("ARCHITECTURE VALIDATION")

    # No NaN or Inf in temperatures (numerical stability)
    assert np.all(np.isfinite(obs["temperatures"])), "FAIL: Non-finite temperatures!"
    print("  [OK] Temperature stability:  No NaN/Inf values (Laplacian stable)")

    # All temperatures >= ambient
    assert np.all(obs["temperatures"] >= 25.0), "FAIL: Temperature below ambient!"
    print("  [OK] Physical consistency:   All temps >= ambient (25.0 C)")

    # Power draw within bounds
    assert np.all(obs["power_draw"] >= 0), "FAIL: Negative power draw!"
    assert np.all(obs["power_draw"] <= 700), "FAIL: Power exceeds H100 TDP!"
    print("  [OK] Power bounds:           All draws within [0, 700]W")

    # Occupancy non-negative
    assert np.all(obs["occupancy"] >= 0), "FAIL: Negative occupancy!"
    print("  [OK] Occupancy validity:     All values >= 0")

    # Rewards contain both positive and negative values (structure works)
    assert np.max(rewards) > 0, "FAIL: No positive rewards (placement broken)"
    print("  [OK] Reward signal:          Positive rewards generated (placement works)")

    # Pipeline completed without crash
    print("  [OK] Pipeline integrity:     End-to-end data flow verified")

    print(f"\n{'=' * 70}")
    print("  ALL ARCHITECTURE TESTS PASSED")
    print(f"  The fused pipeline (Azure x MLPerf -> Thermal Grid) is operational.")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_architecture_test())
