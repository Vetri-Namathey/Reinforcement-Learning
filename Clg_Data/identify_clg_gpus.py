"""
identify_clg_gpus.py
---------------------
1. Recovers ALL files per category by trying every encoding/separator
   combo per-file, instead of trusting the category-name prefix.
2. Uses SM Clock ceiling (1410 MHz = A100, ~2500+ MHz = RTX6000) as the
   ground-truth device-type label — proven cleaner than peak wattage.
3. Cross-correlates Power/Utilization series against each Clock series
   (matched on shared timestamps) to figure out which files belong to
   the SAME physical GPU, since file export order does not guarantee this.

Usage (from repo root):
    python -X utf8 Clg_Data/identify_clg_gpus.py
"""
from __future__ import annotations
import re
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np

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
    """Try every encoding/sep combo; return the first that parses cleanly."""
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
            return vals  # success
        except Exception:
            continue
    return None

def load_category(prefix: str) -> dict[str, pd.Series]:
    out = {}
    for fpath in sorted(CLG_DIR.glob(f"{prefix}*.csv")):
        vals = robust_load(fpath)
        sid = _series_id(fpath.name)
        if vals is None:
            print(f"  [FAILED to parse] {fpath.name}")
            continue
        out[sid] = vals
        print(f"  loaded {fpath.name:55s} -> id '{sid}'  "
              f"range {vals.min():.1f}-{vals.max():.1f}  n={len(vals)}")
    return out

if __name__ == "__main__":
    print("=" * 78 + "\n  RECOVERING ALL FILES (auto-detecting encoding per file)\n" + "=" * 78)
    print("\n--- Power ---");       power = load_category("GPU Power Usage")
    print("\n--- SM Clocks ---");   clocks = load_category("GPU SM Clocks")
    print("\n--- Utilization ---"); util = load_category("GPU Utilization")
    print("\n--- Temperature ---"); temp = load_category("GPU Temperature")
    print(f"\nRecovered counts: power={len(power)} clocks={len(clocks)} "
          f"util={len(util)} temp={len(temp)}  (expect 6, 6, 6, 8)")

    print("\n" + "=" * 78 + "\n  DEVICE TYPE FROM CLOCK CEILING (ground truth)\n" + "=" * 78)
    device_type = {}
    for sid, s in clocks.items():
        peak = s.max()
        dtype = "A100" if peak < 1500 else "RTX6000"
        device_type[sid] = dtype
        print(f"  clock series {sid}: peak={peak:.0f} MHz -> {dtype}")

    print("\n" + "=" * 78 + "\n  MATCHING power/utilization SERIES TO EACH CLOCK SERIES\n"
          "  (by correlation on shared timestamps — same physical GPU should\n"
          "   rise and fall together across all three metrics)\n" + "=" * 78)
    for csid, cser in clocks.items():
        best_p, best_p_corr = None, -2
        for psid, pser in power.items():
            joined = pd.concat([cser, pser], axis=1).dropna()
            if len(joined) < 20:
                continue
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            if corr > best_p_corr:
                best_p, best_p_corr = psid, corr
        best_u, best_u_corr = None, -2
        for usid, user in util.items():
            joined = pd.concat([cser, user], axis=1).dropna()
            if len(joined) < 20:
                continue
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            if corr > best_u_corr:
                best_u, best_u_corr = usid, corr
        print(f"\n  Clock {csid} ({device_type[csid]}):")
        print(f"    best-matching power series: {best_p}  (corr={best_p_corr:.3f})")
        print(f"    best-matching util  series: {best_u}  (corr={best_u_corr:.3f})")