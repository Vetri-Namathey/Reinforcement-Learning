"""
finalize_clg_dataset.py
------------------------
Builds the ONE final, correctly-labeled campus GPU dataset:
  - pairs power + clock + utilization per physical GPU (via correlation)
  - matches the 8 temperature files down to the 6 real GPUs (2 are re-exports)
  - labels each GPU as A100_n or RTX6000_n using the SM-clock-ceiling test
  - outputs a single wide CSV, one column set per physical GPU

Usage (from repo root):
    python -X utf8 Clg_Data/finalize_clg_dataset.py
"""
from __future__ import annotations
import re
from pathlib import Path
from itertools import product
import pandas as pd

CLG_DIR = Path(__file__).parent
ENCODINGS = ["utf-16", "utf-8", "latin-1"]
SEPS = ["\t", ","]

def _series_id(fname: str) -> str:
    m = re.search(r"(\d{2}_\d{2}_\d{2})", fname)
    return m.group(1) if m else fname

def _parse_value(series: pd.Series) -> pd.Series:
    raw = series.astype(str)
    ghz_mask = raw.str.contains("GHz", na=False)
    nums = raw.str.extract(r"([\d.]+)")[0].astype(float)
    nums[ghz_mask] = nums[ghz_mask] * 1000
    return nums

def robust_load(fpath: Path) -> pd.Series | None:
    for enc, sep in product(ENCODINGS, SEPS):
        try:
            df = pd.read_csv(fpath, encoding=enc, sep=sep)
            df.columns = [c.strip().strip('"') for c in df.columns]
            if len(df.columns) < 2:
                continue
            time_col, val_col = df.columns[0], df.columns[1]
            if "time" not in time_col.lower():
                continue
            df[time_col] = pd.to_datetime(df[time_col], errors="raise")
            df = df.set_index(time_col).sort_index()
            vals = _parse_value(df[val_col])
            if vals.notna().sum() < 5:
                continue
            return vals
        except Exception:
            continue
    return None

def load_category(prefix: str) -> dict[str, pd.Series]:
    out = {}
    for fpath in sorted(CLG_DIR.glob(f"{prefix}*.csv")):
        vals = robust_load(fpath)
        if vals is not None:
            out[_series_id(fpath.name)] = vals
    return out

def best_match(target: pd.Series, candidates: dict[str, pd.Series], used: set):
    best_id, best_corr = None, -2
    for cid, cser in candidates.items():
        if cid in used:
            continue
        joined = pd.concat([target, cser], axis=1, sort=True).dropna()
        if len(joined) < 20:
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if corr is not None and corr > best_corr:
            best_id, best_corr = cid, corr
    return best_id, best_corr

if __name__ == "__main__":
    power  = load_category("GPU Power Usage")
    clocks = load_category("GPU SM Clocks")
    util   = load_category("GPU Utilization")
    temp   = load_category("GPU Temperature")
    print(f"Loaded: power={len(power)} clocks={len(clocks)} util={len(util)} temp={len(temp)}")

    device_type = {cid: ("A100" if s.max() < 1500 else "RTX6000") for cid, s in clocks.items()}

    gpus = {}
    used_power, used_util, used_temp = set(), set(), set()
    a100_n = rtx_n = 0
    for cid, cser in clocks.items():
        pid, pcorr = best_match(cser, power, used_power); used_power.add(pid)
        uid, ucorr = best_match(cser, util, used_util);   used_util.add(uid)
        if device_type[cid] == "A100":
            a100_n += 1; label = f"A100_{a100_n}"
        else:
            rtx_n += 1; label = f"RTX6000_{rtx_n}"
        gpus[label] = {"clock_id": cid, "power_id": pid, "util_id": uid}
        print(f"{label}: clock={cid} power={pid}(r={pcorr:.3f}) util={uid}(r={ucorr:.3f})")

    print("\nMatching 8 temperature files to 6 GPUs via correlation with power...")
    for label, info in gpus.items():
        tid, tcorr = best_match(power[info["power_id"]], temp, used_temp)
        used_temp.add(tid)
        info["temp_id"] = tid
        print(f"{label}: temp={tid} (r={tcorr:.3f})")
    print(f"Unmatched (likely duplicate re-export) temperature files: {sorted(set(temp) - used_temp)}")

    pieces = []
    for label, info in gpus.items():
        for metric, series_dict, key in [("power", power, "power_id"), ("clock", clocks, "clock_id"),
                                          ("util", util, "util_id"), ("temp", temp, "temp_id")]:
            s = series_dict[info[key]].resample("15min").mean()
            s.name = f"{label}_{metric}"
            pieces.append(s)

    final = pd.concat(pieces, axis=1, sort=True)
    out_path = CLG_DIR / "gpu_telemetry_identified.csv"
    final.to_csv(out_path)
    print(f"\nSaved final dataset -> {out_path}  shape={final.shape}")