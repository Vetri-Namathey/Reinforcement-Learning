"""
Extract training diagnostics from TensorBoard event logs.

Reads TensorFlow event files and exports key PPO metrics to CSV for analysis.
Metrics extracted:
  - rollout/ep_rew_mean: episode reward (learning curve)
  - train/explained_variance: fraction of variance explained by value function
  - train/clip_fraction: fraction of policy updates clipped
  - train/approx_kl: approximate KL divergence
  - train/entropy_loss: policy entropy (exploration)
"""

import argparse
from pathlib import Path
from collections import defaultdict
import csv
import json
import re

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def extract_from_event_file(event_dir: Path) -> dict:
    """
    Extract training metrics from a TensorBoard event directory.
    
    Args:
        event_dir: Path to directory containing tfevents.* files
        
    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    if not HAS_TENSORBOARD:
        raise ImportError(
            "tensorboard not installed. Install with: pip install tensorboard"
        )
    
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    
    metrics = {}
    
    # Extract scalar metrics of interest
    metric_names = [
        "rollout/ep_rew_mean",
        "train/explained_variance",
        "train/clip_fraction",
        "train/approx_kl",
        "train/entropy_loss",
        "rollout/ep_len_mean",
    ]
    
    for metric in metric_names:
        try:
            scalar_events = ea.Scalars(metric)
            metrics[metric] = [(event.step, event.value) for event in scalar_events]
        except KeyError:
            # Metric not found in this run
            pass
    
    return metrics


def aggregate_results(logdir: Path) -> dict:
    """
    Aggregate metrics from all subdirectories in logdir.
    
    Expected structure:
        logdir/
          MaskablePPO_1/          -> "full" (default)
          MaskablePPO_2/          -> unused
          MaskablePPO_no_thermal/
          MaskablePPO_no_sla/
          MaskablePPO_linear_thermal/
          PPO_1/                  -> "no_masking"
    """
    results = defaultdict(dict)
    best_points: dict[str, int] = {}
    best_source: dict[str, str] = {}

    def run_number(p: Path) -> int:
        m = re.search(r"_(\d+)$", p.name)
        return int(m.group(1)) if m else 0
    
    for subdir in sorted((d for d in logdir.iterdir() if d.is_dir()), key=run_number):
        name = subdir.name

        if "no_thermal" in name:
            key = "MaskablePPO_no_thermal"
        elif "no_sla" in name:
            key = "MaskablePPO_no_sla"
        elif "linear_thermal" in name:
            key = "MaskablePPO_linear_thermal"
        elif name.startswith("MaskablePPO"):
            key = "MaskablePPO"
        elif name.startswith("PPO"):
            key = "PPO"
        else:
            key = name
        
        try:
            metrics = extract_from_event_file(subdir)
        except Exception as e:
            print(f"[!!] Failed to extract {name}: {e}")
            continue

        # Several runs map to one key (MaskablePPO_1, _2, ... -> "MaskablePPO").
        # Keep the *longest* run, not the last one: assigning unconditionally
        # let a 2048-step smoke test silently overwrite a finished 100k-step
        # training run, which is how the committed learning curves ended up
        # being single points.
        # Longest run wins; ties go to the newest. Subdirs are iterated in
        # ascending run number, so >= lets a later run of equal length take
        # over -- which is what you want when an old run predates a physics
        # or reward change and is no longer comparable.
        points = len(metrics.get("rollout/ep_rew_mean", []))
        if points >= best_points.get(key, -1):
            if key in best_points:
                print(f"  {name} -> '{key}' ({points} pts) replaces "
                      f"{best_source[key]} ({best_points[key]} pts)")
            else:
                print(f"[OK] Extracted from {name} -> '{key}' ({points} pts)")
            results[key] = metrics
            best_points[key] = points
            best_source[key] = name
        else:
            print(f"  skipped {name} -> '{key}' ({points} pts; keeping "
                  f"{best_source[key]} at {best_points[key]} pts)")

    return results


def export_csv(results: dict, output_dir: Path) -> None:
    """
    Export metrics to CSV files for plotting.
    
    One CSV per experiment, columns: step, ep_rew_mean, explained_variance, 
    clip_fraction, approx_kl, entropy_loss
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for experiment_name, metrics in results.items():
        if not metrics:
            continue
        
        # Merge all metrics into a unified table
        # Use ep_rew_mean steps as baseline
        step_data = defaultdict(dict)
        
        for metric_name, values in metrics.items():
            short_name = metric_name.split("/")[-1]
            for step, value in values:
                step_data[step][short_name] = value
        
        # Write CSV
        csv_path = output_dir / f"{experiment_name.replace(' ', '_').replace('[', '').replace(']', '')}.csv"
        
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["step", "ep_rew_mean", "ep_len_mean", "explained_variance", "clip_fraction", "approx_kl", "entropy_loss"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for step in sorted(step_data.keys()):
                row = {"step": step}
                row.update(step_data[step])
                writer.writerow(row)
        
        print(f"  -> Saved {csv_path.name} ({len(step_data)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training diagnostics from TensorBoard event logs"
    )
    parser.add_argument(
        "--logdir", type=Path, default=Path("artifacts/logs"),
        help="Directory containing TensorBoard event files (default: artifacts/logs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/diagnostics"),
        help="Output directory for CSV files (default: artifacts/diagnostics)",
    )
    args = parser.parse_args()
    
    if not args.logdir.exists():
        print(f"[!!] Log directory not found: {args.logdir}")
        return 1
    
    print(f"[Diagnostics Extraction]")
    print(f"  Input:  {args.logdir}")
    print(f"  Output: {args.output}\n")
    
    results = aggregate_results(args.logdir)
    
    if not results:
        print("[!!] No results extracted")
        return 1
    
    print(f"\n[Exporting CSV files]")
    export_csv(results, args.output)
    
    print(f"\n[OK] Extraction complete")
    print(f"  Use these CSVs to plot learning curves and training diagnostics")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
