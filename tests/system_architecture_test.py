"""
Integration test for the Spatial-Phase RL Scheduler system.

Verifies the end-to-end pipeline: Azure tokens -> per-cell hardware physics
-> grid placement -> Laplacian physics -> temperature updates, across both
homogeneous and heterogeneous (mixed H100/A100) grids.

Physics-verification focus: every per-device invariant is asserted
element-wise against a (6, 6) matrix. A scalar check against the fleet-wide
maximum TDP would silently pass an A100 cell drawing 690 W, so scalar
safety assertions are treated as defects, not shortcuts.

Runnable both as a script (`python tests/system_architecture_test.py`) and
via pytest (`pytest`, which discovers `test_architecture_suite` below).
"""

import sys
import io
from pathlib import Path

import numpy as np

# Allow `python tests/system_architecture_test.py` (bare script path) to
# resolve absolute `configs`/`src` imports. `python -m tests.system_architecture_test`
# and pytest already have the project root on sys.path without this.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs import (
    PHYSICS_FRACTION_COLUMNS,
    TDP_REFERENCE_WATTS,
    GPUMismatchError,
    get_hardware_profile,
    resolve_job_physics,
    validate_grid_support,
)
from src.data_pipeline.trace_loader import AzureTraceLoader
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import (
    ThermalGridEnv,
    make_checkerboard_layout,
    make_split_layout,
)


def print_header(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def single_cell_equilibrium(gpu_model: str, power_draw: float) -> float:
    """Approximate steady-state temperature for one cell under sustained load."""
    profile = get_hardware_profile(gpu_model)
    ambient = ThermalGridEnv.AMBIENT_TEMP
    return float(
        ambient + profile["power_to_heat"] * power_draw / ThermalGridEnv.COOLING_COEFF
    )


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
    # ------------------------------------------------------------------
    # Phase 1: fused dataset generation (deferred physics)
    # ------------------------------------------------------------------
    print_header("PHASE 1: FUSED DATASET GENERATION (REAL AZURE TRACE)")

    fuser = WorkloadFuser(gpu_models=["H100", "A100"], seed=42)
    fused_df = fuser.generate_fused_dataset(
        num_requests=100, source="azure", workload="conv"
    )
    fuser.print_dataset_summary(fused_df)

    required_columns = ["job_id", "arrival_time", "prefill_tokens", "decode_tokens"]
    for col in required_columns:
        assert col in fused_df.columns, f"Missing primary payload column: {col}"
    for col in PHYSICS_FRACTION_COLUMNS:
        assert col in fused_df.columns, f"Missing efficiency fraction column: {col}"
        values = fused_df[col].values
        assert np.all((values >= 0.0) & (values <= 1.0)), \
            f"{col} outside [0, 1] — cannot interpolate a hardware range"
    assert len(fused_df) == 100, f"Expected 100 rows, got {len(fused_df)}"

    assert fused_df.attrs.get("physics") == "deferred", \
        "Dataset not tagged as deferred-physics"
    supported = fused_df.attrs.get("supported_gpus")
    assert supported == frozenset({"H100", "A100"}), \
        f"Expected both GPUs supported, got {supported}"

    print("\n  [OK] Primary payload:      tokens preserved as job features")
    print(f"  [OK] Efficiency fractions: {len(PHYSICS_FRACTION_COLUMNS)} columns, "
          f"all within [0, 1]")
    print(f"  [OK] Physics mode:         '{fused_df.attrs['physics']}' "
          f"(resolved at placement, not fusion)")
    print(f"  [OK] Supported GPU set:    {sorted(supported)}")

    # ------------------------------------------------------------------
    # Phase 1b: the A100 hardcode regression
    # ------------------------------------------------------------------
    print_header("PHASE 1b: A100 DATASET REGRESSION (SYNTHETIC TRACE HARDCODE)")

    a100_df = WorkloadFuser(gpu_models="A100", seed=7).generate_fused_dataset(
        num_requests=20, source="synthetic"
    )
    assert a100_df.attrs["gpu_model"] == "A100", \
        "Synthetic trace still tagged with the hardcoded fleet GPU"
    assert a100_df.attrs["supported_gpus"] == frozenset({"A100"})
    assert len(a100_df) == 20

    print("  [OK] A100-only synthetic dataset builds without GPUMismatchError")
    print(f"  [OK] Trace tagged '{a100_df.attrs['gpu_model']}' "
          f"(was hardcoded 'H100' pre-fix)")

    # ------------------------------------------------------------------
    # Phase 1c: deferred physics resolves differently per device
    # ------------------------------------------------------------------
    print_header("PHASE 1c: DEFERRED PHYSICS - ONE JOB, TWO DEVICES")

    job = fused_df.iloc[0]
    h100_phys = resolve_job_physics(job, get_hardware_profile("H100"))
    a100_phys = resolve_job_physics(job, get_hardware_profile("A100"))

    print(f"  Job 0: {int(job['prefill_tokens'])} prefill / "
          f"{int(job['decode_tokens'])} decode tokens")
    for model, phys in (("H100", h100_phys), ("A100", a100_phys)):
        print(f"    {model}: prefill {phys['prefill_watts']:6.1f}W / "
              f"{phys['prefill_duration']*1e3:7.2f}ms | "
              f"decode {phys['decode_watts']:6.1f}W / "
              f"{phys['decode_duration']:5.2f}s")

    assert h100_phys["prefill_watts"] != a100_phys["prefill_watts"], \
        "FAIL: identical watts on different hardware — physics not deferred"
    assert h100_phys["prefill_duration"] < a100_phys["prefill_duration"], \
        "FAIL: H100 must be faster than A100 at prefill"
    assert h100_phys["decode_duration"] < a100_phys["decode_duration"], \
        "FAIL: H100 must be faster than A100 at decode"

    print("\n  [OK] Same job resolves to different watts on different devices")
    print("  [OK] H100 completes both phases faster than A100")

    # ------------------------------------------------------------------
    # Phase 1d: set-based GPU support guard
    # ------------------------------------------------------------------
    print_header("PHASE 1d: SET-BASED GPU SUPPORT GUARD")

    # Legal: a mixed grid drawn from the dataset's supported set
    validate_grid_support(fused_df, ["H100", "A100"])
    print("  [OK] Mixed H100/A100 grid accepted against a dual-support dataset")

    # Illegal: grid device outside the dataset's supported set
    try:
        validate_grid_support(a100_df, ["H100"])
        raise AssertionError("FAIL: unsupported grid device was not rejected!")
    except GPUMismatchError as exc:
        print(f"  [OK] Unsupported device rejected: {str(exc)[:68]}...")

    # Illegal: uncalibrated hardware anywhere on the grid
    try:
        validate_grid_support(fused_df, ["H100", "V100"])
        raise AssertionError("FAIL: uncalibrated GPU was not rejected!")
    except GPUMismatchError as exc:
        print(f"  [OK] Uncalibrated device rejected: {str(exc)[:68]}...")

    # Illegal: trace provenance outside the fuser's calibrated set
    trace_df = AzureTraceLoader(workload="conv").load(max_requests=5)
    try:
        WorkloadFuser(gpu_models="A100", seed=0).apply_hardware_physics(trace_df)
        raise AssertionError("FAIL: H100 trace fused into an A100-only set!")
    except GPUMismatchError as exc:
        print(f"  [OK] Trace provenance enforced: {str(exc)[:68]}...")

    # Illegal: env construction against an unsupported grid
    try:
        ThermalGridEnv(fused_dataset=a100_df, gpu_model="H100")
        raise AssertionError("FAIL: env accepted an unsupported grid device!")
    except GPUMismatchError:
        print("  [OK] Environment rejected an H100 grid on an A100-only dataset")

    # ------------------------------------------------------------------
    # Phase 2: environment initialization (homogeneous + heterogeneous)
    # ------------------------------------------------------------------
    print_header("PHASE 2: THERMAL GRID ENVIRONMENT INITIALIZATION")

    homo_env = ThermalGridEnv(fused_dataset=fused_df, gpu_model="H100")
    homo_env.reset()
    assert homo_env.is_heterogeneous is False
    assert np.all(homo_env.tdp_grid == 700.0), "Homogeneous H100 TDP grid wrong"
    assert np.all(homo_env.idle_power_grid == 76.0)
    print("  [OK] Homogeneous H100 grid: uniform 700W TDP / 76W idle "
          "(backward compatible)")

    layout = make_checkerboard_layout("H100", "A100")
    env = ThermalGridEnv(fused_dataset=fused_df, gpu_layout_matrix=layout,
                         render_mode="ansi")
    obs, info = env.reset()

    assert env.is_heterogeneous is True, "Mixed layout not detected"
    assert sorted(info["gpu_models"]) == ["A100", "H100"]

    h100_cells = (env.gpu_layout == "H100")
    a100_cells = (env.gpu_layout == "A100")
    assert h100_cells.sum() == 18 and a100_cells.sum() == 18, \
        "Checkerboard must split the grid evenly"

    # Every per-device constant must be a matrix that actually varies
    for name, grid, h_val, a_val in [
        ("tdp_grid",            env.tdp_grid,            700.0, 400.0),
        ("idle_power_grid",     env.idle_power_grid,      76.0,  55.0),
        ("THROTTLE_TEMP_GRID",  env.THROTTLE_TEMP_GRID,   87.0,  85.0),
        ("POWER_TO_HEAT_GRID",  env.POWER_TO_HEAT_GRID,  0.020, 0.035),
    ]:
        assert grid.shape == (6, 6), f"{name} is not a (6, 6) matrix"
        assert np.allclose(grid[h100_cells], h_val), \
            f"{name} wrong on H100 cells (expected {h_val})"
        assert np.allclose(grid[a100_cells], a_val), \
            f"{name} wrong on A100 cells (expected {a_val})"
        print(f"  [OK] {name:<20} H100={h_val:<6} A100={a_val:<6} "
              f"(per-cell, varies)")

    # ------------------------------------------------------------------
    # Phase 2b: observation tensor exposes heterogeneity
    # ------------------------------------------------------------------
    print_header("PHASE 2b: 6-CHANNEL OBSERVATION TENSOR")

    expected_keys = {"temperatures", "power_draw", "occupancy",
                     "tdp_normalized", "utilization", "slot_fill",
                     "action_mask", "queue_stats", "head_job"}
    assert set(obs) == expected_keys, f"Observation keys mismatch: {set(obs)}"

    for key in ("temperatures", "power_draw", "occupancy",
                "tdp_normalized", "utilization", "slot_fill"):
        assert obs[key].shape == (6, 6), f"{key} grid shape mismatch"

    assert np.allclose(obs["slot_fill"], 0.0), "Grid should start with no jobs resident"

    assert np.allclose(obs["temperatures"], 25.0), "Initial temps should be 25.0 C"
    assert np.allclose(obs["occupancy"], 0.0), "Initial occupancy should be 0.0"

    # Idle power must now be per-cell, not a single scalar
    assert np.allclose(obs["power_draw"][h100_cells], 76.0)
    assert np.allclose(obs["power_draw"][a100_cells], 55.0)

    # Ch 4: per-cell capability, normalized by the fleet-wide reference
    assert np.allclose(obs["tdp_normalized"][h100_cells], 700.0 / TDP_REFERENCE_WATTS)
    assert np.allclose(obs["tdp_normalized"][a100_cells], 400.0 / TDP_REFERENCE_WATTS)

    # Ch 5: localized saturation — must differ from ch 2 for mixed hardware
    assert np.all((obs["utilization"] >= 0.0) & (obs["utilization"] <= 1.0))
    assert np.allclose(obs["utilization"][h100_cells], 76.0 / 700.0)
    assert np.allclose(obs["utilization"][a100_cells], 55.0 / 400.0)

    assert env.action_space.n == 37, "Action space must be Discrete(37)"
    assert obs["action_mask"].shape == (37,), "Action mask shape mismatch"
    # Under the work-conserving mask, Wait is available exactly when no
    # placement is possible -- capacity is never idled while work is queued.
    has_slot = bool((obs["slot_fill"] < 1.0).any())
    if env.queue and has_slot:
        assert obs["action_mask"][36] == 0,             "Work-conserving: Wait must be masked out while a slot is free"
        assert obs["action_mask"][:36].any(), "No placement offered despite free slots"
    else:
        assert obs["action_mask"][36] == 1,             "Wait must be available when nothing can be placed"

    # And with work_conserving disabled, Wait is always legal again.
    permissive = ThermalGridEnv(fused_dataset=fused_df, gpu_layout_matrix=layout,
                                work_conserving=False)
    p_obs, _ = permissive.reset()
    assert p_obs["action_mask"][36] == 1,         "work_conserving=False must leave Wait available"
    assert obs["queue_stats"].shape == (2,), "Queue stats shape mismatch"
    assert obs["head_job"].shape == (4,), "Head-job feature shape mismatch"

    # The head-of-queue job must be visible, and must actually describe it --
    # a policy that cannot see token counts is scheduling blind.
    if env.queue:
        head = env.queue[0]
        assert abs(obs["head_job"][0] - head["prefill_tokens"] / 4096.0) < 1e-5,             "head_job[0] does not encode the head job's prefill tokens"
        assert abs(obs["head_job"][1] - head["decode_tokens"] / 1024.0) < 1e-5,             "head_job[1] does not encode the head job's decode tokens"
        print(f"  [OK] Head job visible: {int(head['prefill_tokens'])} prefill / "
              f"{int(head['decode_tokens'])} decode tokens -> "
              f"{np.round(obs['head_job'], 3).tolist()}")
    assert int(obs["queue_stats"][0]) == info["queue_length"], \
        "queue_stats[0] must equal queue length"

    print(f"  [OK] Observation keys:     {sorted(expected_keys)}")
    print(f"  [OK] Ch 4 tdp_normalized:  H100={700/TDP_REFERENCE_WATTS:.3f}  "
          f"A100={400/TDP_REFERENCE_WATTS:.3f}")
    print(f"  [OK] Ch 5 utilization:     H100={76/700:.3f}  A100={55/400:.3f} "
          f"at idle")
    print(f"       ^ observability proof:  a 400W draw reads "
          f"{400/700:.2f} on H100 but {400/400:.2f} on A100 — "
          f"indistinguishable under global scaling")
    print(f"  [OK] Action space:         Discrete({env.action_space.n}) "
          f"(36 GPUs + Wait)")
    print(f"  [OK] Initial queue:        {info['queue_length']} job(s) "
          f"(arrivals at t=0)")

    # ------------------------------------------------------------------
    # Phase 2c: placement-time physics is cell-dependent
    # ------------------------------------------------------------------
    print_header("PHASE 2c: PLACEMENT-TIME PHYSICS RESOLUTION")

    # Same dataset, same head-of-queue job, two different target devices.
    # Cell (0,0) is H100 (action 0); cell (0,1) is A100 (action 1).
    env_h = ThermalGridEnv(fused_dataset=fused_df, gpu_layout_matrix=layout)
    env_a = ThermalGridEnv(fused_dataset=fused_df, gpu_layout_matrix=layout)
    env_h.reset()
    env_a.reset()
    assert env_h.gpu_layout[0, 0] == "H100" and env_a.gpu_layout[0, 1] == "A100"

    env_h.step(0)   # place on the H100 cell
    env_a.step(1)   # place the same job on the A100 cell

    # Jobs now live in per-cell lists and store *marginal* watts (draw above the
    # cell's idle floor), so add idle back to recover the absolute prefill draw.
    job_h = env_h.cell_jobs[0][0][0]
    job_a = env_a.cell_jobs[0][1][0]
    w_h = float(job_h["pf_watts"]) + float(env_h.idle_power_grid[0, 0])
    w_a = float(job_a["pf_watts"]) + float(env_a.idle_power_grid[0, 1])
    d_h = float(env_h.occupancy[0, 0])
    d_a = float(env_a.occupancy[0, 1])

    print(f"  Head job on H100 cell (0,0): {w_h:6.1f}W prefill, "
          f"{d_h:6.2f}s total occupancy")
    print(f"  Head job on A100 cell (0,1): {w_a:6.1f}W prefill, "
          f"{d_a:6.2f}s total occupancy")

    assert w_h != w_a, \
        "FAIL: identical prefill watts — env is not resolving physics per cell"
    assert d_a > d_h, "FAIL: A100 must take longer than H100 for the same job"
    assert w_h <= 700.0 and w_a <= 400.0, "FAIL: placement exceeded device TDP"

    print("\n  [OK] Watts and duration depend on the target cell's hardware")
    print("  [OK] Neither placement exceeded its own device's TDP")

    # ------------------------------------------------------------------
    # Phase 3: masked random-action simulation
    # ------------------------------------------------------------------
    print_header("PHASE 3: 200-STEP MASKED RANDOM-ACTION SIMULATION")
    print("  Proving: Azure Tokens -> Per-Cell Watts -> Queue -> Grid -> "
          "Laplacian Physics\n")

    obs, info = env.reset()
    rng = np.random.default_rng(42)
    total_steps = 200
    rewards = []
    max_temps = []
    wait_steps = 0
    tdp_violations = 0

    for step in range(1, total_steps + 1):
        valid_actions = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(valid_actions))
        if action == 36:
            wait_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Element-wise TDP check every single step, not just at the end
        tdp_violations += int(np.sum(obs["power_draw"] > env.tdp_grid + 1e-6))

        rewards.append(reward)
        max_temps.append(info["max_temperature"])

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
    print(f"  Min / Max reward:       {np.min(rewards):+.2f} / {np.max(rewards):+.2f}")
    print(f"  --- Thermal ---")
    print(f"  Final max temperature:  {info['max_temperature']:.2f} C")
    print(f"  Peak max temperature:   {np.max(max_temps):.2f} C")
    print(f"  Min thermal headroom:   {info['thermal_headroom']:.2f} C "
          f"(vs per-cell threshold)")
    print(f"  Throttle events:        {info['throttle_events']}")
    print(f"  --- Utilization ---")
    print(f"  Mean utilization:       {info['mean_utilization']*100:.1f}% of per-cell TDP")
    print(f"  Active cells:           {info['active_cells']}/36")

    # ------------------------------------------------------------------
    # Phase 4: element-wise physical invariants
    # ------------------------------------------------------------------
    print_header("ARCHITECTURE VALIDATION (ELEMENT-WISE)")

    assert np.all(np.isfinite(obs["temperatures"])), "FAIL: Non-finite temperatures!"
    print("  [OK] Temperature stability:  No NaN/Inf values (Laplacian stable)")

    assert np.all(obs["temperatures"] >= 25.0), "FAIL: Temperature below ambient!"
    print("  [OK] Physical consistency:   All temps >= ambient (25.0 C)")

    assert np.all(obs["power_draw"] >= 0), "FAIL: Negative power draw!"

    h100_equilibrium = single_cell_equilibrium("H100", 700.0)
    assert h100_equilibrium < 140.0, (
        f"FAIL: single-cell H100 equilibrium {h100_equilibrium:.1f}C is too hot"
    )
    print(f"  [OK] Single-cell equilibrium: H100 @ 700W -> {h100_equilibrium:.1f}C "
        f"(gamma=0.14, calibrated for realistic thermal signal under Azure load)")

    a100_equilibrium = single_cell_equilibrium("A100", 400.0)
    assert a100_equilibrium < 140.0, (
        f"FAIL: single-cell A100 equilibrium {a100_equilibrium:.1f}C is too hot"
    )
    print(f"  [OK] Single-cell equilibrium: A100 @ 400W -> {a100_equilibrium:.1f}C")

    h100_idle_eq = single_cell_equilibrium("H100", 76.0)
    assert h100_idle_eq < 50.0, (
        f"FAIL: H100 idle equilibrium {h100_idle_eq:.1f}C implausibly high for idle"
    )
    print(f"  [OK] Idle equilibrium:        H100 @ 76W (idle) -> {h100_idle_eq:.1f}C")

    # THE element-wise check. A scalar `<= env.tdp_watts` (700W) would pass
    # an A100 cell drawing 690W — 172% of its real ceiling.
    over = obs["power_draw"] > env.tdp_grid + 1e-6
    assert not np.any(over), (
        f"FAIL: {int(over.sum())} cell(s) exceeded their own TDP: "
        f"{obs['power_draw'][over]} vs {env.tdp_grid[over]}"
    )
    assert tdp_violations == 0, \
        f"FAIL: {tdp_violations} per-step TDP violations during the rollout"
    print(f"  [OK] Per-cell TDP bounds:    all draws <= own device TDP, "
          f"every step ({steps_completed} checked)")

    a100_max = float(obs["power_draw"][a100_cells].max())
    assert a100_max <= 400.0 + 1e-6, f"FAIL: A100 cell drew {a100_max}W"
    print(f"  [OK] A100 ceiling binding:   peak A100 draw {a100_max:.1f}W <= 400W "
          f"(a scalar 700W check would not catch this)")

    assert np.all(obs["occupancy"] >= 0), "FAIL: Negative occupancy!"
    print("  [OK] Occupancy validity:     All values >= 0")

    # Utilization must stay a valid fraction on every device
    util = obs["power_draw"] / env.tdp_grid
    assert np.all((util >= 0.0) & (util <= 1.0 + 1e-6)), \
        "FAIL: utilization outside [0, 1] — per-cell normalization broken"
    print("  [OK] Localized utilization:  All cells within [0, 1] of own TDP")

    assert np.max(rewards) > 0, "FAIL: No positive rewards (placement broken)"
    print("  [OK] Reward signal:          Positive rewards generated")

    assert info["dropped_jobs"] == 0, "FAIL: Backlog queue dropped jobs!"
    assert info["jobs_placed"] + info["queue_length"] + \
        (100 - env.next_arrival_idx) == 100, "FAIL: Job conservation violated!"
    print("  [OK] Queue conservation:     placed + queued + not-yet-arrived = total")

    # ------------------------------------------------------------------
    # Phase 4b: continuous batching (concurrency)
    # ------------------------------------------------------------------
    print_header("PHASE 4b: CONTINUOUS BATCHING / CONCURRENCY")

    # K=1 must reproduce the single-job environment bit for bit.
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    env_k1 = ThermalGridEnv(fused_dataset=fused_df, gpu_model="H100")
    env_df = ThermalGridEnv(fused_dataset=fused_df, gpu_model="H100",
                            max_jobs_per_cell=1)
    obs_a, _ = env_k1.reset(seed=3)
    obs_b, _ = env_df.reset(seed=3)
    for _ in range(120):
        a = int(rng_a.choice(np.flatnonzero(obs_a["action_mask"])))
        b = int(rng_b.choice(np.flatnonzero(obs_b["action_mask"])))
        assert a == b, "K=1 diverged from the default env"
        obs_a, r_a, t_a, _, _ = env_k1.step(a)
        obs_b, r_b, t_b, _, _ = env_df.step(b)
        assert abs(r_a - r_b) < 1e-9, "K=1 reward diverged"
        if t_a or t_b:
            break
    assert np.allclose(env_k1.temperatures, env_df.temperatures),         "K=1 temperature field diverged"
    print("  [OK] max_jobs_per_cell=1 reproduces the single-job env exactly")

    # K>1: a cell must accept several jobs, and sum their power.
    # work_conserving=False: this probe deliberately idles to build a backlog,
    # which the work-conserving mask forbids.
    env_k4 = ThermalGridEnv(fused_dataset=fused_df, gpu_model="H100",
                            max_jobs_per_cell=4, work_conserving=False)
    obs4, _ = env_k4.reset(seed=1)
    while len(env_k4.queue) < 3 and env_k4.next_arrival_idx < env_k4.total_jobs:
        obs4, _, _, _, _ = env_k4.step(env_k4.WAIT_ACTION)

    idle0 = float(env_k4.idle_power_grid[0, 0])
    powers = [float(env_k4.power_draw[0, 0])]
    for _ in range(3):
        if obs4["action_mask"][0] != 1:
            break
        obs4, _, _, _, _ = env_k4.step(0)      # stack onto cell (0, 0)
        powers.append(float(env_k4.power_draw[0, 0]))

    assert env_k4.slots_used[0, 0] >= 2,         f"Cell (0,0) should hold multiple jobs, holds {env_k4.slots_used[0, 0]}"
    assert powers[-1] > powers[0], "Stacking jobs did not raise cell power"
    assert powers[-1] <= env_k4.tdp_grid[0, 0] + 1e-6,         f"Batched cell drew {powers[-1]:.0f} W, above its {env_k4.tdp_grid[0, 0]:.0f} W TDP"
    print(f"  [OK] {int(env_k4.slots_used[0, 0])} jobs co-resident on one cell: "
          f"{powers[0]:.0f} W idle -> {powers[-1]:.0f} W "
          f"(TDP {env_k4.tdp_grid[0, 0]:.0f} W, idle {idle0:.0f} W)")

    # Power must be idle + sum of marginals, never idle counted per job
    marg = sum(j["pf_watts"] if j["phase"] == 1 else j["dec_watts"]
               for j in env_k4.cell_jobs[0][0])
    assert abs(powers[-1] - min(idle0 + marg, float(env_k4.tdp_grid[0, 0]))) < 1e-3,         "Cell power is not idle + sum(marginal draws)"
    print("  [OK] Cell power = idle + sum(marginal job draws), idle counted once")

    # A full cell must be masked out
    env_full = ThermalGridEnv(fused_dataset=fused_df, gpu_model="H100",
                              max_jobs_per_cell=2, work_conserving=False)
    obsf, _ = env_full.reset(seed=5)
    while len(env_full.queue) < 3 and env_full.next_arrival_idx < env_full.total_jobs:
        obsf, _, _, _, _ = env_full.step(env_full.WAIT_ACTION)
    for _ in range(2):
        if obsf["action_mask"][0] == 1:
            obsf, _, _, _, _ = env_full.step(0)
    assert env_full.slots_used[0, 0] == 2, "Cell should be at capacity"
    assert obsf["action_mask"][0] == 0, "A full cell must be masked out"
    print("  [OK] Cell at capacity is removed from the action mask")

    # ------------------------------------------------------------------
    # Phase 5: alternate layout smoke test
    # ------------------------------------------------------------------
    print_header("PHASE 5: SPLIT-RACK LAYOUT SMOKE TEST")

    split_env = ThermalGridEnv(
        fused_dataset=fused_df, gpu_layout_matrix=make_split_layout("H100", "A100")
    )
    split_obs, _ = split_env.reset()
    assert np.all(split_env.tdp_grid[:3, :] == 700.0), "Top half must be H100"
    assert np.all(split_env.tdp_grid[3:, :] == 400.0), "Bottom half must be A100"

    rng = np.random.default_rng(1)
    for _ in range(50):
        valid = np.flatnonzero(split_obs["action_mask"])
        split_obs, _, term, _, _ = split_env.step(int(rng.choice(valid)))
        assert np.all(split_obs["power_draw"] <= split_env.tdp_grid + 1e-6)
        if term:
            break
    print("  [OK] Split layout: top 3 rows H100 / bottom 3 rows A100")
    print("  [OK] Per-cell TDP held across 50 steps on a second layout")

    # Uniform-threshold override for baseline reproducibility
    pinned = ThermalGridEnv(
        fused_dataset=fused_df, gpu_layout_matrix=layout, throttle_temp_override=80.0
    )
    assert np.all(pinned.THROTTLE_TEMP_GRID == 80.0)
    print("  [OK] throttle_temp_override pins a uniform 80 C for A/B comparison")

    print(f"\n{'=' * 70}")
    print("  ALL ARCHITECTURE TESTS PASSED")
    print("  Heterogeneous H100/A100 grids resolve physics per cell and hold")
    print("  every physical invariant element-wise.")
    print(f"{'=' * 70}\n")

    return 0


def test_architecture_suite():
    """pytest entry point: the script above is the actual assertion suite."""
    assert run_architecture_test() == 0


if __name__ == "__main__":
    # Fix Windows console encoding for Unicode output. Only applied for a
    # direct script run: pytest replaces sys.stdout with its own capture
    # object, and wrapping that object's .buffer breaks pytest's capture
    # teardown ("I/O operation on closed file").
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.exit(run_architecture_test())
