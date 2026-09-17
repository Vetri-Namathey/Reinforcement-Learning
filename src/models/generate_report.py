"""
Results Report Generator for the Spatial-Phase RL Scheduler.

Generates a self-contained results report from existing artifacts:
  - Training diagnostics CSVs  (artifacts/diagnostics/)
  - Baseline comparison metrics (run live via run_baselines harness)

Outputs:
  artifacts/report/
    01_baseline_comparison_bar.png       4 core metrics across schedulers
    02_placement_heatmaps.png            Per-scheduler 6x6 cell placement density
    03_temperature_timeline.png          Max/mean temp over simulation steps
    04_queue_depth_timeline.png          Queue depth over simulation steps
    05_learning_curves.png               Episode reward vs training steps (all variants)
    06_ablation_summary_table.png        Summary table rendered as image
    07_training_diagnostics.png          EV, clip fraction, KL, entropy
    results.json                         All metrics in machine-readable format

Usage:
    python -m src.models.generate_report
    python -m src.models.generate_report --jobs 500
    python -m src.models.generate_report --jobs 100 --workload code
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from configs import ARTIFACTS_DIR, MODELS_DIR
from src.baselines.run_baselines import (
    all_schedulers,
    build_layout,
    resolve_candidate_k,
)
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import ThermalGridEnv
from src.models.train_agent import (
    DEFAULT_ARRIVAL_ACCEL,
    DEFAULT_CANDIDATE_K,
    DEFAULT_LAYOUT,
    DEFAULT_SLOTS,
    EVAL_OFFSET,
    TRAIN_POOL_JOBS,
)

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

REPORT_DIR = ARTIFACTS_DIR / "report"
DIAGNOSTICS_DIR = ARTIFACTS_DIR / "diagnostics"

# Color palette
COLORS = {
    "Round-Robin":     "#5B8DEF",
    "Least-Occupied":  "#F2994A",
    "Thermal-Greedy":  "#27AE60",
    "Best-Fit":        "#8E44AD",
    "RL Agent":        "#EB5757",
}
ABLATION_COLORS = {
    "MaskablePPO":                "#2E86AB",
    "MaskablePPO_no_thermal":     "#F18F01",
    "MaskablePPO_no_sla":         "#C73E1D",
    "MaskablePPO_linear_thermal": "#6A994E",
    "PPO":                        "#A23B72",
}


# ── Instrumented Episode Runner ──────────────────────────────────────────

def run_instrumented_episode(scheduler, fused_df, gpu_model="H100", grid_size=6,
                             slots=DEFAULT_SLOTS, layout_matrix=None,
                             candidate_k=DEFAULT_CANDIDATE_K):
    """Run one episode collecting per-step telemetry for timeline plots.

    ``candidate_k`` must match the checkpoint's training value -- see the note on
    ``run_baselines.run_episode``. Every scheduler in a report shares it, so the
    comparison table is a like-for-like one.
    """
    env = ThermalGridEnv(fused_dataset=fused_df, gpu_model=gpu_model, grid_size=grid_size,
                         gpu_layout_matrix=layout_matrix, max_jobs_per_cell=slots,
                         candidate_k=candidate_k)
    obs, info = env.reset()

    step_data = []
    actions_per_cell = np.zeros(env.NUM_CELLS, dtype=int)
    terminated = False
    steps = 0

    while not terminated and steps < 500_000:
        action = scheduler.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if action < env.NUM_CELLS:
            actions_per_cell[action] += 1

        step_data.append({
            "step": steps,
            "max_temp": float(np.max(obs["temperatures"])),
            "mean_temp": float(np.mean(obs["temperatures"])),
            "queue_length": info["queue_length"],
            "power_draw": float(np.mean(obs["power_draw"])),
            "active_cells": info["active_cells"],
            "reward": float(reward),
        })

    # Terminal metrics
    ttfts = np.array([j["ttft"] for j in env.successful_jobs])
    sim_time = max(env.current_time, 1e-9)
    sla_attained = int(np.sum(ttfts <= 5.0)) if len(ttfts) else 0

    metrics = {
        "p99_ttft":           float(np.percentile(ttfts, 99)) if len(ttfts) else 0.0,
        "sla_goodput":        sla_attained / sim_time,
        "sla_attainment_pct": 100.0 * sla_attained / max(len(ttfts), 1),
        "tttr":               env.throttling_steps / max(env.total_steps, 1),
        "throttled_work_ratio": env.throttled_job_steps / max(env.job_step_total, 1),
        "mean_spatial_var":   env.spatial_variance_sum / max(env.total_steps, 1),
        "jobs_placed":        len(ttfts),
        "peak_temp":          float(np.max([s["max_temp"] for s in step_data])),
        "sim_time":           sim_time,
        "steps":              steps,
        "placement_spread":   float(np.std(actions_per_cell)),
        # The RL objective itself, plus its per-term split. Reported so the
        # table shows the quantity being optimised next to the operational
        # metrics, rather than only the latter.
        "total_reward":       float(env.total_reward),
        "reward_place":       float(env.reward_totals["place"]),
        "reward_sla":         float(env.reward_totals["sla"]),
        "reward_thermal":     float(env.reward_totals["thermal"]),
    }

    return metrics, step_data, actions_per_cell


# ── Plot Generators ──────────────────────────────────────────────────────

def plot_baseline_comparison_bar(all_metrics: dict, output_dir: Path):
    """Plot 1: 4 core metrics as grouped bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline Comparison -- 4 Core Evaluation Metrics",
                 fontsize=16, fontweight="bold", y=0.98)

    metric_configs = [
        ("p99_ttft", "P99 TTFT (s)", "lower is better"),
        ("sla_attainment_pct", "SLA Attainment (%)", "higher is better"),
        ("tttr", "Thermal Throttle Ratio", "lower is better"),
        ("mean_spatial_var", "Mean Spatial Variance (°C²)", "lower is better"),
    ]

    names = list(all_metrics.keys())
    colors = [COLORS.get(n, "#999999") for n in names]
    x = np.arange(len(names))

    for ax, (key, label, note) in zip(axes.flat, metric_configs):
        values = [all_metrics[n][key] for n in names]
        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label}\n({note})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}" if val < 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "01_baseline_comparison_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_placement_heatmaps(all_placements: dict, grid_size: int, output_dir: Path):
    """Plot 2: Per-scheduler 6x6 cell placement density heatmaps."""
    n = len(all_placements)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 4.5))
    fig.suptitle("Job Placement Distribution per Cell",
                 fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (name, counts) in zip(axes, all_placements.items()):
        grid = counts.reshape(grid_size, grid_size)
        im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", aspect="equal")
        ax.set_title(name, fontsize=11, fontweight="bold")
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j, i, f"{grid[i, j]}", ha="center", va="center",
                        fontsize=8, color="black" if grid[i, j] < grid.max() * 0.7 else "white")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jobs placed")

    fig.tight_layout()
    path = output_dir / "02_placement_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_temperature_timeline(all_step_data: dict, output_dir: Path):
    """Plot 3: Max and mean temperature over simulation steps."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Temperature Timeline", fontsize=14, fontweight="bold")

    for name, step_data in all_step_data.items():
        steps = [s["step"] for s in step_data]
        color = COLORS.get(name, "#999999")
        ax1.plot(steps, [s["max_temp"] for s in step_data],
                 label=name, color=color, linewidth=1.5, alpha=0.8)
        ax2.plot(steps, [s["mean_temp"] for s in step_data],
                 label=name, color=color, linewidth=1.5, alpha=0.8)

    ax1.set_ylabel("Max Temperature (°C)", fontsize=11)
    ax1.set_title("Peak Cell Temperature", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Simulation Step", fontsize=11)
    ax2.set_ylabel("Mean Temperature (°C)", fontsize=11)
    ax2.set_title("Grid-Average Temperature", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "03_temperature_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_queue_depth_timeline(all_step_data: dict, output_dir: Path):
    """Plot 4: Queue depth over simulation steps."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Queue Depth Over Time", fontsize=14, fontweight="bold")

    for name, step_data in all_step_data.items():
        steps = [s["step"] for s in step_data]
        color = COLORS.get(name, "#999999")
        ax.plot(steps, [s["queue_length"] for s in step_data],
                label=name, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Queue Depth", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "04_queue_depth_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_learning_curves(diagnostics_dir: Path, output_dir: Path):
    """Plot 5: Episode reward vs training steps from diagnostics CSVs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Learning Curves -- Episode Reward vs Training Steps",
                 fontsize=14, fontweight="bold")

    found = False
    for csv_file in sorted(diagnostics_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        name = csv_file.stem
        if "ep_rew_mean" not in df.columns or df["ep_rew_mean"].dropna().empty:
            continue
        color = ABLATION_COLORS.get(name, "#999999")
        label = name.replace("MaskablePPO_", "").replace("MaskablePPO", "full (MaskablePPO)")
        ax.plot(df["step"], df["ep_rew_mean"], linewidth=2,
                label=label, color=color, alpha=0.85, marker="o", markersize=4)
        found = True

    if not found:
        ax.text(0.5, 0.5, "No training data available yet.\nRun train_agent.py first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Episode Reward (mean)", fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "05_learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_ablation_summary_table(diagnostics_dir: Path, output_dir: Path):
    """Plot 6: Summary metrics table rendered as image."""
    rows = []
    for csv_file in sorted(diagnostics_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        name = csv_file.stem
        if df.empty:
            continue
        row = {"Experiment": name.replace("MaskablePPO_", "").replace("MaskablePPO", "full")}
        if "ep_rew_mean" in df.columns and not df["ep_rew_mean"].dropna().empty:
            best_idx = df["ep_rew_mean"].idxmax()
            row["Best Reward"] = f"{df.loc[best_idx, 'ep_rew_mean']:.1f}"
            row["@ Step"] = f"{df.loc[best_idx, 'step']:.0f}"
        if "explained_variance" in df.columns:
            last_valid = df["explained_variance"].dropna()
            row["Final EV"] = f"{last_valid.iloc[-1]:.3f}" if not last_valid.empty else "—"
        if "clip_fraction" in df.columns:
            last_valid = df["clip_fraction"].dropna()
            row["Final Clip"] = f"{last_valid.iloc[-1]:.3f}" if not last_valid.empty else "—"
        if "approx_kl" in df.columns:
            last_valid = df["approx_kl"].dropna()
            row["Final KL"] = f"{last_valid.iloc[-1]:.4f}" if not last_valid.empty else "—"
        rows.append(row)

    if not rows:
        print("  [!!] No diagnostics data for summary table")
        return

    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * len(rows)))
    ax.axis("off")
    ax.set_title("Ablation Study -- Training Summary Metrics",
                 fontsize=14, fontweight="bold", pad=20)

    table_data = [[row.get(col, "—") for col in rows[0].keys()] for row in rows]
    table = ax.table(cellText=table_data, colLabels=list(rows[0].keys()),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j, key in enumerate(rows[0].keys()):
        table[0, j].set_facecolor("#2E86AB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(rows[0])):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    path = output_dir / "06_ablation_summary_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_training_diagnostics(diagnostics_dir: Path, output_dir: Path):
    """Plot 7: PPO training diagnostics (EV, clip, KL, entropy)."""
    diag_metrics = [
        ("explained_variance", "Explained Variance", 0.9, "Target"),
        ("clip_fraction", "Clip Fraction", None, None),
        ("approx_kl", "Approx KL Divergence", 0.02, "Stability limit"),
        ("entropy_loss", "Entropy Loss", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Diagnostics", fontsize=16, fontweight="bold")

    for ax, (metric, title, threshold, threshold_label) in zip(axes.flat, diag_metrics):
        found = False
        for csv_file in sorted(diagnostics_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            name = csv_file.stem
            if metric not in df.columns or df[metric].dropna().empty:
                continue
            color = ABLATION_COLORS.get(name, "#999999")
            label = name.replace("MaskablePPO_", "").replace("MaskablePPO", "full")
            valid = df.dropna(subset=[metric])
            ax.plot(valid["step"], valid[metric], linewidth=2,
                    label=label, color=color, alpha=0.85, marker="o", markersize=3)
            found = True

        if threshold is not None:
            ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1,
                       alpha=0.5, label=threshold_label)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(fontsize=9, loc="best")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")

    fig.tight_layout()
    path = output_dir / "07_training_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ── JSON Results Dump ────────────────────────────────────────────────────

def save_results_json(all_metrics: dict, config: dict, output_dir: Path):
    """Save all metrics as a structured JSON for reproducibility."""

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": {name: metrics for name, metrics in all_metrics.items()},
    }

    path = output_dir / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"  [OK] {path.name}")


def save_markdown_report(config: dict, output_dir: Path):
    """Generate a Markdown file embedding the generated plots."""
    md_content = f"""# Reinforcement Learning Scheduler: Evaluation & Diagnostics

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Configuration:** {config['jobs']} jobs | {config['workload']} workload | GPU: {config['gpu_model']}
**Trace slice:** rows {config['trace_offset']}–{config['trace_offset'] + config['jobs']} ({config['trace_split']}) | cooling γ = {config['cooling_coeff']}

## 1. Baseline Performance Comparison

### Core Evaluation Metrics
![Baseline Comparison - Core Metrics](./01_baseline_comparison_bar.png)

### Load Distribution (Placement Heatmaps)
![Placement Heatmaps](./02_placement_heatmaps.png)

### Environmental Impact: Temperature Timeline
![Temperature Timeline](./03_temperature_timeline.png)

### System Backlog: Queue Depth
![Queue Depth Timeline](./04_queue_depth_timeline.png)

---

## 2. RL Agent Training Diagnostics

### Learning Curves (Episode Rewards)
![Learning Curves](./05_learning_curves.png)

### PPO Stability Diagnostics
![Training Diagnostics](./07_training_diagnostics.png)

### Ablation Study Summary
![Ablation Summary Table](./06_ablation_summary_table.png)
"""
    path = output_dir / "REPORT.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [OK] {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate results report")
    parser.add_argument("--jobs", type=int, default=500,
                        help="Number of jobs for baseline comparison (default: 500)")
    parser.add_argument("--workload", choices=["conv", "code"], default="conv",
                        help="Azure trace variant (default: conv)")
    parser.add_argument("--gpu-model", default="H100",
                        help="GPU model for baseline grid (default: H100)")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--layout", choices=["homogeneous", "checkerboard", "split"],
                        default=DEFAULT_LAYOUT,
                        help="Grid hardware layout. Must match how the agent was trained.")
    parser.add_argument("--accel", type=float, default=DEFAULT_ARRIVAL_ACCEL,
                        help="Arrival-rate compression; must match training.")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                        help=f"Concurrent jobs per cell (default: {DEFAULT_SLOTS}). "
                             "Must match what the agent was trained with.")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                        help=f"Shortlist width, i.e. legal cells per decision "
                             f"(default: {DEFAULT_CANDIDATE_K}, matching the shipped "
                             "checkpoint). Pass 0 for the unrestricted 36-cell space; "
                             "the RL row is then off-policy and not comparable.")
    parser.add_argument("--offset", type=int, default=EVAL_OFFSET,
                        help=f"Trace row offset for the evaluation slice "
                             f"(default: {EVAL_OFFSET}). train_agent.py trains on "
                             f"rows 0..{TRAIN_POOL_JOBS}, so the default here is "
                             f"held-out traffic. Anything below {TRAIN_POOL_JOBS} "
                             f"overlaps the training pool and is reported as such.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_k = resolve_candidate_k(args.candidate_k)
    overlaps_training = args.offset < TRAIN_POOL_JOBS
    split = "TRAINING OVERLAP" if overlaps_training else "held-out"
    print(f"\n{'=' * 60}")
    print(f"  REPORT GENERATOR -- Spatial-Phase RL Scheduler")
    print(f"  Jobs: {args.jobs} | Workload: {args.workload} | GPU: {args.gpu_model}")
    print(f"  Trace slice: rows {args.offset}..{args.offset + args.jobs} ({split})")
    print(f"  Concurrency: {args.slots} job(s)/cell | arrival accel: {args.accel}x "
          f"| layout: {args.layout}")
    print(f"  Action space: candidate_k="
          f"{candidate_k if candidate_k else 'unrestricted (36 cells)'}"
          f" (trained: {DEFAULT_CANDIDATE_K})")
    print(f"{'=' * 60}")
    if overlaps_training:
        print(f"  [!!] offset {args.offset} < {TRAIN_POOL_JOBS}: this slice overlaps")
        print( "       train_agent.py's training pool. The RL agent is being scored")
        print( "       on traffic it trained on — not a generalization result.")
    if candidate_k != DEFAULT_CANDIDATE_K:
        print(f"  [!!] candidate_k {candidate_k} != {DEFAULT_CANDIDATE_K} used for training.")
        print( "       The agent is being scored on a different action space from the")
        print( "       one its advantages were computed over. The RL row is off-policy;")
        print( "       results.json records action_space_match=false.")

    # ── Step 1: Run baselines with full instrumentation ──────────────
    print(f"\n[Step 1/4] Running baseline schedulers...")

    layout_matrix = build_layout(args.layout, args.grid_size, args.gpu_model)
    gpu_models = ([args.gpu_model] if layout_matrix is None
                  else [args.gpu_model, "A100"])
    fuser = WorkloadFuser(gpu_models=gpu_models, seed=42)
    fused_df = fuser.generate_fused_dataset(
        num_requests=args.jobs, source="azure", workload=args.workload,
        offset=args.offset, arrival_accel=args.accel,
    )
    print(f"  Loaded {len(fused_df)} jobs "
          f"({fused_df.attrs['source']}, GPU={fused_df.attrs['gpu_model']})")

    model_path = MODELS_DIR / "spatial_thermal_ppo.zip"
    model = None
    if model_path.exists():
        print(f"  Found trained RL policy: {model_path.name}")
        model = MaskablePPO.load(str(model_path))
    else:
        print(f"  [!] No trained RL policy found at {model_path}. RL agent will be omitted from baseline comparison.")
    schedulers = all_schedulers(model)

    all_metrics = {}
    all_step_data = {}
    all_placements = {}

    for sched in schedulers:
        print(f"  Running {sched.name}...")
        metrics, step_data, placements = run_instrumented_episode(
            sched, fused_df, gpu_model=args.gpu_model, grid_size=args.grid_size,
            slots=args.slots, layout_matrix=layout_matrix, candidate_k=candidate_k,
        )
        all_metrics[sched.name] = metrics
        all_step_data[sched.name] = step_data
        all_placements[sched.name] = placements
        print(f"    {metrics['steps']} steps, {metrics['jobs_placed']} jobs placed")

    # ── Step 2: Generate baseline plots ──────────────────────────────
    print(f"\n[Step 2/4] Generating baseline comparison plots...")
    plot_baseline_comparison_bar(all_metrics, REPORT_DIR)
    plot_placement_heatmaps(all_placements, args.grid_size, REPORT_DIR)
    plot_temperature_timeline(all_step_data, REPORT_DIR)
    plot_queue_depth_timeline(all_step_data, REPORT_DIR)

    # ── Step 3: Generate training diagnostics plots ──────────────────
    print(f"\n[Step 3/4] Generating training diagnostics plots...")
    if DIAGNOSTICS_DIR.exists() and any(DIAGNOSTICS_DIR.glob("*.csv")):
        plot_learning_curves(DIAGNOSTICS_DIR, REPORT_DIR)
        plot_ablation_summary_table(DIAGNOSTICS_DIR, REPORT_DIR)
        plot_training_diagnostics(DIAGNOSTICS_DIR, REPORT_DIR)
    else:
        print("  [!!] No diagnostics CSVs found -- skipping training plots")
        print("  Run: python -m src.models.extract_diagnostics")

    # ── Step 4: Save JSON & Markdown ─────────────────────────────────
    print(f"\n[Step 4/4] Saving structured results & markdown...")
    config = {
        "jobs": args.jobs,
        "workload": args.workload,
        "gpu_model": args.gpu_model,
        "grid_size": args.grid_size,
        "trace_offset": args.offset,
        "trace_split": "training-overlap" if overlaps_training else "held-out",
        "cooling_coeff": ThermalGridEnv.COOLING_COEFF,
        "slots_per_cell": args.slots,
        "candidate_k": candidate_k,
        "candidate_k_trained": DEFAULT_CANDIDATE_K,
        "action_space_match": candidate_k == DEFAULT_CANDIDATE_K,
        "layout": args.layout,
        "arrival_accel": args.accel,
    }
    save_results_json(all_metrics, config, REPORT_DIR)
    save_markdown_report(config, REPORT_DIR)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  REPORT COMPLETE -- {len(list(REPORT_DIR.glob('*')))} files generated")
    print(f"  Output: {REPORT_DIR}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
