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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLG_DIR = Path(__file__).parent
DATASET = CLG_DIR / "gpu_telemetry_identified.csv"
OUT_CSV = CLG_DIR / "physics_validation_results.csv"
PARITY_PNG = CLG_DIR / "physics_validation_parity.png"
SERIES_PNG = CLG_DIR / "physics_validation_timeseries.png"

# Fraction of the (chronological) record used to fit; the remainder is held out.
# Fitting and scoring on the same samples describes the data, it does not
# validate a model -- the split is what turns this into a prediction test.
TRAIN_FRACTION = 0.60

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


def holdout_fit(power: np.ndarray, temp: np.ndarray, train_fraction: float):
    """
    Fit on the first `train_fraction` of the record, score on the rest.

    The split is chronological, not random: shuffling would leak, because
    consecutive 15-minute samples are highly correlated and a random test point
    almost always sits between two training points. Splitting by time asks the
    honest question -- calibrate on the first few days, then predict days the
    fit has never seen.
    """
    n = len(power)
    cut = int(n * train_fraction)
    if cut < 10 or n - cut < 10:
        return None

    p_tr, t_tr = power[:cut], temp[:cut]
    p_te, t_te = power[cut:], temp[cut:]

    slope, intercept, r2_tr, rmse_tr = steady_state_fit(p_tr, t_tr)

    pred_te = intercept + slope * p_te
    resid = pred_te - t_te
    rmse_te = float(np.sqrt(np.mean(resid ** 2)))
    mae_te = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t_te - t_te.mean()) ** 2))
    r2_te = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"cut": cut, "slope": slope, "intercept": intercept,
            "rmse_train": rmse_tr, "r2_train": r2_tr,
            "rmse_test": rmse_te, "r2_test": r2_te, "mae_test": mae_te,
            "t_std_train": float(t_tr.std()), "t_std_test": float(t_te.std())}


def make_figures(panels, train_fraction: float):
    """Parity plot and measured-vs-predicted time series, one panel per GPU."""
    # ---- parity: predicted against measured, held-out samples only ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Simulator thermal model vs real campus GPUs -- held-out samples",
                 fontsize=14, fontweight="bold")
    for ax, d in zip(axes.flat, panels):
        ax.scatter(d["t_te"], d["pred_te"], s=9, alpha=0.45,
                   color="#B8562F", edgecolors="none")
        lo = min(d["t_te"].min(), d["pred_te"].min()) - 2
        hi = max(d["t_te"].max(), d["pred_te"].max()) + 2
        ax.plot([lo, hi], [lo, hi], color="#444", lw=1.2, ls="--", label="perfect")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{d['gpu']}   R^2={d['r2_test']:.3f}  RMSE={d['rmse_test']:.2f} C",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("measured temperature (C)")
        ax.set_ylabel("predicted temperature (C)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PARITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- time series: the whole week, with the split marked ----
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.1 * len(panels)),
                             sharex=True)
    fig.suptitle(f"Measured vs predicted temperature "
                 f"(fit on first {train_fraction:.0%}, predicted thereafter)",
                 fontsize=14, fontweight="bold")
    for ax, d in zip(np.atleast_1d(axes), panels):
        ax.plot(d["index"], d["temp"], color="#2C3E5A", lw=0.9,
                label="measured", alpha=0.85)
        ax.plot(d["index"], d["pred_all"], color="#B8562F", lw=1.1,
                label="predicted", alpha=0.9)
        ax.axvline(d["index"][d["cut"]], color="#666", ls="--", lw=1.2)
        ax.set_ylabel(d["gpu"] + " (C)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    np.atleast_1d(axes)[0].text(
        0.01, 0.95, "<- fit    |    held out ->", transform=np.atleast_1d(axes)[0].transAxes,
        fontsize=9, va="top", color="#666")
    fig.tight_layout()
    fig.savefig(SERIES_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    rows, panels = [], []
    for gpu in GPUS:
        joined = pd.concat(
            [df[f"{gpu}_power"], df[f"{gpu}_temp"], df[f"{gpu}_util"]], axis=1
        ).dropna()
        power = joined.iloc[:, 0].to_numpy(float)
        temp = joined.iloc[:, 1].to_numpy(float)
        util = joined.iloc[:, 2].to_numpy(float)

        ratio, ambient, r2, rmse = steady_state_fit(power, temp)
        rmse_pinned = fixed_ambient_rmse(power, temp, ratio)
        ho = holdout_fit(power, temp, TRAIN_FRACTION)
        if ho is not None:
            panels.append({
                "gpu": gpu, "index": joined.index,
                "temp": temp, "cut": ho["cut"],
                "t_te": temp[ho["cut"]:],
                "pred_te": ho["intercept"] + ho["slope"] * power[ho["cut"]:],
                "pred_all": ho["intercept"] + ho["slope"] * power,
                "r2_test": ho["r2_test"], "rmse_test": ho["rmse_test"],
                "rmse_train": ho["rmse_train"], "mae_test": ho["mae_test"],
                "low_conf": None,
            })

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
            "holdout_rmse_train": ho["rmse_train"] if ho else float("nan"),
            "holdout_rmse_test": ho["rmse_test"] if ho else float("nan"),
            "holdout_r2_test": ho["r2_test"] if ho else float("nan"),
            "holdout_mae_test": ho["mae_test"] if ho else float("nan"),
            "holdout_t_std_test": ho["t_std_test"] if ho else float("nan"),
        })
        if ho is not None:
            panels[-1]["low_conf"] = low_conf
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

    # ── hold-out: fit on the first days, predict the rest ──
    print("\n" + "=" * 78)
    print(f"  HOLD-OUT VALIDATION   fit on first {TRAIN_FRACTION:.0%} of the record, "
          f"predict the rest")
    print("=" * 78)
    print("  Fitting and scoring on the same samples describes the data; this")
    print("  predicts days the fit never saw.\n")
    print("  RMSE is the primary metric. R^2 is reported but is unreliable when the")
    print("  held-out window is near-isothermal: R^2 = 1 - SSres/SStot, so a small")
    print("  SStot drives it sharply negative even where RMSE is small and stable.")
    print("  The 'T std test' column shows when that applies.")
    print("")
    print(f"  {'GPU':10s} {'RMSE fit':>9s} {'RMSE test':>10s} {'delta':>8s} "
          f"{'MAE test':>9s} {'T std test':>11s} {'R^2 test':>10s}  flag")
    print("  " + "-" * 82)
    for row in rows:
        if np.isnan(row["holdout_rmse_test"]):
            continue
        flags = []
        if row["low_confidence"]:
            flags.append("LOW-CONF")
        if row["holdout_t_std_test"] < 5.0:
            flags.append("R^2 n/a (flat window)")
        delta = row["holdout_rmse_test"] - row["holdout_rmse_train"]
        print(f"  {row['gpu']:10s} {row['holdout_rmse_train']:9.2f} "
              f"{row['holdout_rmse_test']:10.2f} {delta:+8.2f} "
              f"{row['holdout_mae_test']:9.2f} {row['holdout_t_std_test']:11.2f} "
              f"{row['holdout_r2_test']:10.3f}  {', '.join(flags)}")

    trust = results[~results.low_confidence]
    if len(trust):
        print(f"\n  High-confidence devices: mean held-out RMSE "
              f"{trust.holdout_rmse_test.mean():.2f} C, "
              f"best {trust.holdout_rmse_test.min():.2f} C")
        degrade = trust.holdout_rmse_test.mean() - trust.holdout_rmse_train.mean()
        print(f"  Degradation from fit to held-out: {degrade:+.2f} C "
              f"-- a small number here means the model generalises, not memorises.")

    if panels:
        make_figures(panels, TRAIN_FRACTION)
        print(f"\n  Figures -> {PARITY_PNG.name}, {SERIES_PNG.name}")

    results.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved -> {OUT_CSV.relative_to(CLG_DIR.parent)}  ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
