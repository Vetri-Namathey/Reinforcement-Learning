"""
Generate comparison plots for RL ablation study.

Plots learning curves and training diagnostics from extracted CSV files.
Generates:
  1. Learning curves (ep_rew_mean) for masking ablation
  2. Learning curves for reward variant ablation
  3. Diagnostic dashboards (explained_variance, clip_fraction, approx_kl)
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv_files(diagnostics_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files from diagnostics directory."""
    data = {}
    for csv_file in diagnostics_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        data[csv_file.stem] = df
    return data


def plot_masking_ablation(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Compare learning curves: MaskablePPO vs vanilla PPO."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        "MaskablePPO": "#2E86AB",
        "PPO": "#A23B72",
    }
    
    for name, df in data.items():
        if name in colors:
            if "ep_rew_mean" in df.columns:
                ax.plot(
                    df["step"], df["ep_rew_mean"],
                    linewidth=2, label=name, color=colors[name], alpha=0.8
                )
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Masking Ablation: MaskablePPO vs Vanilla PPO\n(Action masking vs penalty-based constraints)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path = output_dir / "01_masking_ablation_learning_curve.png"
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_reward_ablation(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Compare learning curves: 4 reward variants."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        "MaskablePPO": "#2E86AB",
        "MaskablePPO_no_thermal": "#F18F01",
        "MaskablePPO_no_sla": "#C73E1D",
        "MaskablePPO_linear_thermal": "#6A994E",
    }
    
    for name, df in data.items():
        if name in colors and "ep_rew_mean" in df.columns:
            ax.plot(
                df["step"], df["ep_rew_mean"],
                linewidth=2, label=name.replace("MaskablePPO_", "").replace("MaskablePPO", "full"),
                color=colors[name], alpha=0.8
            )
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Reward Ablation Study: 4 Reward Function Variants\n(full, no_thermal, no_sla, linear_thermal)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path = output_dir / "02_reward_ablation_learning_curves.png"
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_training_diagnostics(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot PPO diagnostic metrics (explained_variance, clip_fraction, approx_kl)."""
    
    # Filter to get best representative run(s)
    key_experiments = [
        ("MaskablePPO", "MaskablePPO (with masking)"),
        ("PPO", "vanilla PPO (penalty-based)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Diagnostics: MaskablePPO vs Vanilla PPO", fontsize=16, fontweight="bold")
    
    for (name, label), ax in zip(key_experiments, axes.flat[:len(key_experiments)]):
        if name not in data:
            continue
        
        df = data[name]
        
        # Plot 1: Explained variance
        if "explained_variance" in df.columns:
            ax.plot(df["step"], df["explained_variance"], linewidth=2, color="#2E86AB", label="Explained Variance")
            ax.axhline(y=0.9, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Target (0.9)")
            ax.set_ylabel("Explained Variance", fontsize=11)
        
        # Add secondary axis for clip_fraction
        ax2 = ax.twinx()
        if "clip_fraction" in df.columns:
            ax2.plot(df["step"], df["clip_fraction"], linewidth=2, color="#A23B72", label="Clip Fraction", linestyle="--")
            ax2.set_ylabel("Clip Fraction", fontsize=11, color="#A23B72")
            ax2.tick_params(axis="y", labelcolor="#A23B72")
        
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
    
    # Plot 3: Approx KL divergence
    ax = axes.flat[2]
    for name, label in key_experiments:
        if name in data and "approx_kl" in data[name].columns:
            df = data[name]
            ax.plot(df["step"], df["approx_kl"], linewidth=2, label=label, alpha=0.8)
    ax.axhline(y=0.02, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Stability threshold (0.02)")
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Approx KL Divergence", fontsize=11)
    ax.set_title("Policy Divergence (Lower is more stable)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Entropy loss
    ax = axes.flat[3]
    for name, label in key_experiments:
        if name in data and "entropy_loss" in data[name].columns:
            df = data[name]
            ax.plot(df["step"], df["entropy_loss"], linewidth=2, label=label, alpha=0.8)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Entropy Loss", fontsize=11)
    ax.set_title("Policy Entropy (Exploration vs Exploitation)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path = output_dir / "03_ppo_diagnostics_dashboard.png"
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_all_metrics_table(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Generate summary table of best metrics for each variant."""
    summary = []
    
    for name, df in data.items():
        if df.empty:
            continue
        
        row = {"Experiment": name.replace("MaskablePPO_", "").replace("MaskablePPO", "full")}
        
        # Best (highest) episode reward
        if "ep_rew_mean" in df.columns:
            best_reward_idx = df["ep_rew_mean"].idxmax()
            row["Best ep_rew_mean"] = f"{df.loc[best_reward_idx, 'ep_rew_mean']:.1f}"
            row["Best at step"] = f"{df.loc[best_reward_idx, 'step']:.0f}"
        
        # Final metrics
        if not df.empty:
            last_row = df.iloc[-1]
            if "explained_variance" in df.columns:
                row["Final Explained Variance"] = f"{last_row['explained_variance']:.3f}"
            if "clip_fraction" in df.columns:
                row["Final Clip Fraction"] = f"{last_row['clip_fraction']:.3f}"
            if "approx_kl" in df.columns:
                row["Final Approx KL"] = f"{last_row['approx_kl']:.4f}"
        
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    
    # Save as CSV
    csv_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Also print as table
    print("\n[Summary Metrics Table]")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for RL ablation study results"
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("artifacts/diagnostics"),
        help="Directory containing extracted CSV diagnostics (default: artifacts/diagnostics)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/plots"),
        help="Output directory for plots (default: artifacts/plots)",
    )
    args = parser.parse_args()
    
    if not args.diagnostics_dir.exists():
        print(f"✗ Diagnostics directory not found: {args.diagnostics_dir}")
        print("  First run: python -m src.models.extract_diagnostics --logdir artifacts/logs")
        return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Plotting Ablation Results]")
    print(f"  Input:  {args.diagnostics_dir}")
    print(f"  Output: {args.output_dir}\n")
    
    # Load data
    data = load_csv_files(args.diagnostics_dir)
    
    if not data:
        print("✗ No CSV files found in diagnostics directory")
        return 1
    
    print(f"✓ Loaded {len(data)} experiment results\n[Generating plots]")
    
    # Generate plots
    plot_masking_ablation(data, args.output_dir)
    plot_reward_ablation(data, args.output_dir)
    plot_training_diagnostics(data, args.output_dir)
    plot_all_metrics_table(data, args.output_dir)
    
    print(f"\n✓ Plotting complete")
    print(f"  Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
