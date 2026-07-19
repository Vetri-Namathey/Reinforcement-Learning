"""
Integration test for the Spatial-Phase RL Scheduler system.

Verifies the end-to-end pipeline: Azure tokens -> MLPerf watts -> grid
placement -> Laplacian physics -> temperature updates. Generates 100 fused
jobs, initializes ThermalGridEnv, runs a masked random-action simulation,
and asserts on physical/structural invariants.
"""

import sys
import io
import numpy as np

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_fusion_pipeline import (
    AzureTraceLoader,
    GPUMismatchError,
    WorkloadFuser,
    get_hardware_profile,
)
from thermal_grid_env import ThermalGridEnv


def print_header(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_step_report(step: int, action: int, reward: float, info: dict) -> None:
    if action == 36:
        action_label = "WAIT      "
    else:
        row, col = divmod(action, 6)
        action_label = f"GPU[{row},{col}] ({action:2d})"
    print(
        f"  Step {step:3d} | "
        f"Action: {action_label} | "
        f"Reward: {reward:+8.2f} | "
        f"MaxT: {info['max_temperature']:5.1f} C | "
        f"Queue: {info['queue_length']:3d} | "
        f"Active: {info['active_cells']:2d}/36"
    )


def run_architecture_test():
    # Phase 1: fused dataset generation
    print_header("PHASE 1: FUSED DATASET GENERATION (REAL AZURE TRACE x H100)")

    gpu_model = "H100"
    profile = get_hardware_profile(gpu_model)

    fuser = WorkloadFuser(gpu_model=gpu_model, seed=42)
    fused_df = fuser.generate_fused_dataset(
        num_requests=100, source="azure", workload="conv"
    )
    fuser.print_dataset_summary(fused_df)

    required_columns = [
        "job_id", "prefill_watts", "prefill_duration",
        "decode_watts", "decode_duration"
    ]
    for col in required_columns:
        assert col in fused_df.columns, f"Missing column: {col}"
    assert len(fused_df) == 100, f"Expected 100 rows, got {len(fused_df)}"

    pw_lo, pw_hi = profile["prefill_watts_bounds"]
    dw_lo, dw_hi = profile["decode_watts_bounds"]
    assert fused_df["prefill_watts"].min() >= pw_lo, "Prefill watts below envelope"
    assert fused_df["prefill_watts"].max() <= pw_hi, "Prefill watts above envelope"
    assert fused_df["decode_watts"].min() >= dw_lo, "Decode watts below envelope"
    assert fused_df["decode_watts"].max() <= dw_hi, "Decode watts above envelope"

    assert fused_df.attrs.get("gpu_model") == gpu_model, "Fused dataset lost GPU tag"

    print("\n  [OK] Dataset structure validated")
    print(f"  [OK] All {len(required_columns)} required columns present")
    print(f"  [OK] {len(fused_df)} fused jobs generated from "
          f"{fused_df.attrs.get('source')}")
    print(f"  [OK] Watts within {gpu_model} envelopes: "
          f"prefill [{pw_lo:.0f}, {pw_hi:.0f}]W, decode [{dw_lo:.0f}, {dw_hi:.0f}]W")
    print(f"  [OK] GPU tag propagated: attrs['gpu_model'] = "
          f"'{fused_df.attrs['gpu_model']}'")

    # Phase 1b: GPU mismatch must be rejected
    print_header("PHASE 1b: GPU-MATCHING GUARD")

    trace_df = AzureTraceLoader(workload="conv").load(max_requests=5)
    print(f"  Trace fleet GPU:  {trace_df.attrs['gpu_model']}")
    print(f"  Fusing against:   A100 hardware profile (deliberate mismatch)")

    try:
        WorkloadFuser(gpu_model="A100", seed=0).apply_hardware_physics(trace_df)
        raise AssertionError("FAIL: H100 trace x A100 physics was not rejected!")
    except GPUMismatchError as exc:
        print(f"  [OK] Fusion rejected mismatched GPUs: {exc}")

    try:
        ThermalGridEnv(fused_dataset=fused_df, gpu_model="A100")
        raise AssertionError("FAIL: H100 dataset on A100 grid was not rejected!")
    except GPUMismatchError:
        print("  [OK] Environment rejected H100 dataset on an A100 grid")

    print("\n  Sample fused jobs:")
    print(fused_df[["job_id", "prefill_watts", "prefill_duration",
                     "decode_watts", "decode_duration"]].head(5).to_string(index=False))

    # Phase 2: environment initialization
    print_header("PHASE 2: THERMAL GRID ENVIRONMENT INITIALIZATION")

    env = ThermalGridEnv(fused_dataset=fused_df, gpu_model=gpu_model,
                         render_mode="ansi")
    obs, info = env.reset()

    assert obs["temperatures"].shape == (6, 6), "Temperature grid shape mismatch"
    assert obs["power_draw"].shape == (6, 6), "Power draw grid shape mismatch"
    assert obs["occupancy"].shape == (6, 6), "Occupancy grid shape mismatch"
    assert np.allclose(obs["temperatures"], 25.0), "Initial temps should be 25.0 C"
    assert np.allclose(obs["power_draw"], env.idle_power), \
        f"Initial power should be {env.idle_power}W idle"
    assert np.allclose(obs["occupancy"], 0.0), "Initial occupancy should be 0.0"

    assert env.action_space.n == 37, "Action space must be Discrete(37)"
    assert obs["action_mask"].shape == (37,), "Action mask shape mismatch"
    assert obs["action_mask"][36] == 1, "Wait action must always be valid"
    assert obs["queue_stats"].shape == (2,), "Queue stats shape mismatch"
    if info["queue_length"] == 0:
        assert np.all(obs["action_mask"][:36] == 0), \
            "Empty queue must mask all placement actions"
    else:
        assert np.all(obs["action_mask"][:36] == 1), \
            "All-free grid with queued jobs must unmask all placements"
    assert int(obs["queue_stats"][0]) == info["queue_length"], \
        "queue_stats[0] must equal queue length"

    print(f"  [OK] Grid GPU model:       {env.gpu_model} "
          f"(matches dataset '{fused_df.attrs['gpu_model']}')")
    print(f"  [OK] Grid shape:           {obs['temperatures'].shape} (6x6 = 36 GPUs)")
    print(f"  [OK] Initial temperature:  {obs['temperatures'][0, 0]:.1f} C (ambient)")
    print(f"  [OK] Initial power draw:   {obs['power_draw'][0, 0]:.1f}W (idle)")
    print(f"  [OK] Initial occupancy:    {obs['occupancy'][0, 0]:.1f}s (free)")
    print(f"  [OK] Action space:         Discrete({env.action_space.n}) "
          f"(36 GPUs + Wait)")
    print(f"  [OK] Action mask:          shape (37,), Wait always valid, "
          f"queue-aware")
    print(f"  [OK] Observation space:    Dict(temperatures, power_draw, "
          f"occupancy, action_mask, queue_stats)")
    print(f"  [OK] Initial queue:        {info['queue_length']} job(s) "
          f"(arrivals at t=0)")

    # Phase 3: masked random-action simulation
    print_header("PHASE 3: 200-STEP MASKED RANDOM-ACTION SIMULATION")
    print("  Proving: Azure Tokens -> H100 Watts -> Queue -> Grid -> "
          "Laplacian Physics\n")

    rng = np.random.default_rng(42)
    total_steps = 200
    rewards = []
    max_temps = []
    mean_temps = []
    wait_steps = 0

    for step in range(1, total_steps + 1):
        valid_actions = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(valid_actions))
        if action == 36:
            wait_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        max_temps.append(info["max_temperature"])
        mean_temps.append(info["mean_temperature"])

        if step % 20 == 0 or step == 1 or terminated:
            print_step_report(step, action, reward, info)

        if terminated:
            print(f"\n  [!] Environment terminated at step {step} "
                  f"(trace ingested, backlog cleared)")
            break

    print_header("FINAL SUMMARY")

    steps_completed = len(rewards)
    print(f"  Steps completed:        {steps_completed}/{total_steps}")
    print(f"  Simulated time:         {info['current_time']:.1f}s")
    print(f"  Jobs placed:            {info['jobs_placed']}")
    print(f"  Jobs remaining:         {info['jobs_remaining']}")
    print(f"  Queue length (final):   {info['queue_length']}")
    print(f"  Dropped jobs:           {info['dropped_jobs']} (backlog never drops)")
    print(f"  Wait actions taken:     {wait_steps}/{steps_completed}")
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

    print_header("ARCHITECTURE VALIDATION")

    assert np.all(np.isfinite(obs["temperatures"])), "FAIL: Non-finite temperatures!"
    print("  [OK] Temperature stability:  No NaN/Inf values (Laplacian stable)")

    assert np.all(obs["temperatures"] >= 25.0), "FAIL: Temperature below ambient!"
    print("  [OK] Physical consistency:   All temps >= ambient (25.0 C)")

    assert np.all(obs["power_draw"] >= 0), "FAIL: Negative power draw!"
    assert np.all(obs["power_draw"] <= env.tdp_watts), \
        f"FAIL: Power exceeds {env.gpu_model} TDP!"
    print(f"  [OK] Power bounds:           All draws within "
          f"[0, {env.tdp_watts:.0f}]W ({env.gpu_model} TDP)")

    assert np.all(obs["occupancy"] >= 0), "FAIL: Negative occupancy!"
    print("  [OK] Occupancy validity:     All values >= 0")

    assert np.max(rewards) > 0, "FAIL: No positive rewards (placement broken)"
    print("  [OK] Reward signal:          Positive rewards generated (placement works)")

    assert info["dropped_jobs"] == 0, "FAIL: Backlog queue dropped jobs!"
    assert info["jobs_placed"] + info["queue_length"] + \
        (100 - env.next_arrival_idx) == 100, "FAIL: Job conservation violated!"
    print("  [OK] Queue conservation:     placed + queued + not-yet-arrived = total")

    print("  [OK] Pipeline integrity:     End-to-end data flow verified")

    print(f"\n{'=' * 70}")
    print("  ALL ARCHITECTURE TESTS PASSED")
    print(f"  The fused pipeline (Azure x MLPerf -> Thermal Grid) is operational.")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_architecture_test())
