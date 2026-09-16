"""
validate_physics_against_real_gpu.py
-------------------------------------
Sim-to-real validation of the simulator's thermal model against real campus
GPU telemetry (6 devices, 7 days, 10-minute sampling).

WHAT THIS DATASET CAN AND CANNOT IDENTIFY
------------------------------------------
The simulator's per-cell thermal model is

    dT/dt = eta * P  -  gamma * (T - T_ambient)

whose exact solution over a step of length dt is

    T(t+dt) = T_eq + (T(t) - T_eq) * exp(-gamma*dt),   T_eq = T_ambient + (eta/gamma)*P

The campus telemetry is sampled every **600 s**, while the thermal time constant
is 1/gamma ~ 7 s. So exp(-gamma*dt) = exp(-84) ~ 1e-37: the memory term is zero
to machine precision at every sample. Each reading is an *independent* steady
state, and the model collapses to a straight line:

    T = T_ambient + (eta/gamma) * P

Consequences, and the reason this script was rewritten:

  1. **gamma is NOT identifiable from this data.** Sweeping gamma across four
     orders of magnitude while holding eta/gamma fixed gives a bit-identical
     RMSE. The previous version of this script reported "fitted gamma = 0.117
     ... 0.140" from an L-BFGS-B run started at gamma=0.14 -- that was the
     optimiser sitting still in a flat direction, not a measurement. Any claim
     that this dataset validates the simulator's gamma is unsupported.
     Identifying gamma needs sub-minute (ideally ~1 s) sampling.

  2. **eta alone is NOT identifiable either** -- only the ratio eta/gamma, in
     degC per watt. Quote eta only as "eta implied at the simulator's gamma".

  3. **T_ambient must be fitted, not assumed.** The previous version hardcoded
     ambient=25.0 C. The real intercepts run 23.6-35.9 C (the RTX6000 pair sit
     in a noticeably warmer part of the room). Forcing a low intercept tilts the
     line steeper and inflates the slope -- which is how the RTX6000 profile
     ended up with a power_to_heat roughly 2x its true value.

Run: python -X utf8 Clg_Data/validate_physics_against_real_gpu.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLG_DIR = Path(__file__).parent
DATASET = CLG_DIR / "gpu_telemetry_identified.csv"
OUT_CSV = CLG_DIR / "physics_validation_results.csv"

GPUS = ["A100_1", "A100_2", "A100_3", "A100_4", "RTX6000_1", "RTX6000_2"]

# The simulator's constants, for comparison only -- never fitted here.
SIM_GAMMA = 0.14
SIM_AMBIENT = 25.0
PROFILE_ETA = {"A100": 0.035, "RTX6000": 0.037}

# A fit is flagged low-confidence below these; a near-idle GPU barely varies its
# power, so its slope is poorly conditioned however good the R^2 looks.
MIN_R2 = 0.70
MIN_MEAN_UTIL = 5.0


def steady_state_fit(power: np.ndarray, temp: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares fit of T = ambient + (eta/gamma) * P. Both free."""
    design = np.column_stack([power, np.ones_like(power)])
    (slope, intercept), *_ = np.linalg.lstsq(design, temp, rcond=None)
    pred = design @ np.array([slope, intercept])
    rmse = float(np.sqrt(np.mean((pred - temp) ** 2)))
    ss_res = float(np.sum((temp - pred) ** 2))
    ss_tot = float(np.sum((temp - temp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2, rmse


def fixed_ambient_rmse(power: np.ndarray, temp: np.ndarray, ratio: float,
                       ambient: float = SIM_AMBIENT) -> float:
    """RMSE of the old formulation: ambient pinned, only the slope free."""
    return float(np.sqrt(np.mean((ambient + ratio * power - temp) ** 2)))


def main() -> int:
    if not DATASET.exists():
        print(f"[!!] {DATASET.name} not found. Build it first:")
        print("     python -X utf8 Clg_Data/finalize_clg_dataset.py")
        return 1

    df = pd.read_csv(DATASET, index_col=0, parse_dates=True)
    dt = df.index.to_series().diff().dt.total_seconds().median()

    print("=" * 78)
    print("  IDENTIFIABILITY CHECK")
    print("=" * 78)
    print(f"  sample interval dt              : {dt:.0f} s")
    print(f"  simulator thermal time constant : {1 / SIM_GAMMA:.1f} s  (gamma = {SIM_GAMMA})")
    print(f"  memory term exp(-gamma*dt)      : {np.exp(-SIM_GAMMA * dt):.3e}")
    print("  -> the transient is dead at every sample: each reading is an")
    print("     independent steady state. gamma is unidentifiable here; only")
    print("     the ratio eta/gamma (degC per watt) can be measured.\n")

    print("=" * 78)
    print("  STEADY-STATE FIT   T = ambient + (eta/gamma) * P   (both parameters free)")
    print("=" * 78)
    header = (f"  {'GPU':10s} {'eta/gamma':>10s} {'ambient':>9s} {'R^2':>6s} "
              f"{'RMSE':>7s} {'RMSE@amb=25':>12s} {'util%':>7s}  flag")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for gpu in GPUS:
        joined = pd.concat(
            [df[f"{gpu}_power"], df[f"{gpu}_temp"], df[f"{gpu}_util"]], axis=1
        ).dropna()
        power = joined.iloc[:, 0].to_numpy(float)
        temp = joined.iloc[:, 1].to_numpy(float)
        util = joined.iloc[:, 2].to_numpy(float)

        ratio, ambient, r2, rmse = steady_state_fit(power, temp)
        rmse_pinned = fixed_ambient_rmse(power, temp, ratio)

        low_conf = r2 < MIN_R2 or util.mean() < MIN_MEAN_UTIL
        flag = "LOW-CONF" if low_conf else ""
        rows.append({
            "gpu": gpu,
            "device": gpu.rsplit("_", 1)[0],
            "eta_over_gamma": ratio,
            "ambient_fit": ambient,
            "r2": r2,
            "rmse": rmse,
            "rmse_fixed_ambient_25": rmse_pinned,
            "mean_util_pct": util.mean(),
            "n_samples": len(joined),
            "low_confidence": low_conf,
        })
        print(f"  {gpu:10s} {ratio:10.4f} {ambient:9.1f} {r2:6.3f} "
              f"{rmse:7.2f} {rmse_pinned:12.2f} {util.mean():7.1f}  {flag}")

    results = pd.DataFrame(rows)
    print("\n  LOW-CONF = R^2 < %.2f or mean utilization < %.0f%%: a near-idle GPU's"
          % (MIN_R2, MIN_MEAN_UTIL))
    print("  power barely varies, so its slope is poorly conditioned.\n")

    print("=" * 78)
    print(f"  IMPLIED eta AT THE SIMULATOR'S gamma = {SIM_GAMMA}")
    print("=" * 78)
    print("  eta_implied = (eta/gamma)_measured * gamma_sim. This is the only")
    print("  legitimate way to turn this dataset into a power_to_heat constant.\n")
    print(f"  {'device':10s} {'n':>3s} {'eta/gamma':>10s} {'eta_implied':>12s} "
          f"{'profile eta':>12s} {'ratio':>7s}")
    print("  " + "-" * 60)

    summary = []
    for device, group in results[~results.low_confidence].groupby("device"):
        measured = group.eta_over_gamma.mean()
        implied = measured * SIM_GAMMA
        profile = PROFILE_ETA.get(device, float("nan"))
        print(f"  {device:10s} {len(group):3d} {measured:10.4f} {implied:12.4f} "
              f"{profile:12.4f} {profile / implied:7.2f}x")
        summary.append((device, measured, implied, profile))

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    for device, measured, implied, profile in summary:
        off = profile / implied
        verdict = "consistent" if 0.8 <= off <= 1.25 else f"OFF BY {off:.2f}x"
        print(f"  {device:10s} profile power_to_heat={profile:.4f} vs "
              f"measured-implied={implied:.4f}  -> {verdict}")
    print("\n  gamma: NOT VALIDATED by this dataset (see identifiability check).")
    print("  Any statement that campus telemetry confirms the simulator's gamma")
    print("  is unsupported; 10-minute sampling cannot see a 7-second transient.")

    results.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved -> {OUT_CSV.relative_to(CLG_DIR.parent)}  ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
