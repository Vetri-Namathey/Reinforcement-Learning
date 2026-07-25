"""
Training pipeline for the Spatial-Phase RL Scheduler.

Trains a MaskablePPO agent (sb3-contrib) on ThermalGridEnv using the custom
SpatialCNN feature extractor. Action masking is supplied to MaskablePPO
through the ActionMasker wrapper.

Usage (run from the project root):
    python -m src.models.train_agent                     # full 100k-timestep run
    python -m src.models.train_agent --timesteps 5000    # short smoke run
    python -m src.models.train_agent --layout checkerboard   # mixed H100/A100 grid
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs import LOGS_DIR, MODELS_DIR
from src.data_pipeline.workload_fuser import WorkloadFuser
from src.environment.thermal_env import (
    ThermalGridEnv,
    make_checkerboard_layout,
    make_split_layout,
)
from src.models.spatial_cnn import SpatialCNN

MODEL_PATH = MODELS_DIR / "spatial_thermal_ppo"
VECNORM_PATH = MODELS_DIR / "vec_normalize.pkl"

LAYOUTS = {
    "homogeneous": None,
    "checkerboard": make_checkerboard_layout,
    "split":        make_split_layout,
}


def mask_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker hook: pull the current 37-element mask from the env."""
    return env.unwrapped._compute_action_mask().astype(bool)


def make_env(
    num_jobs: int = 500,
    workload: str = "conv",
    gpu_model: str = "H100",
    seed: int = 42,
    max_episode_steps: int = 5_000,
    layout: str = "homogeneous",
    secondary_gpu: str = "A100",
) -> gym.Env:
    """
    Build the wrapped training environment:
        ThermalGridEnv -> TimeLimit -> ActionMasker -> Monitor

    TimeLimit truncates episodes where an untrained policy Waits forever.

    Args:
        layout: "homogeneous", "checkerboard", or "split". Non-homogeneous
            layouts mix gpu_model with secondary_gpu across the grid, and
            the fused dataset is built to support both.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Choose from {list(LAYOUTS)}")

    layout_fn = LAYOUTS[layout]
    if layout_fn is None:
        gpu_models = [gpu_model]
        layout_matrix = None
    else:
        gpu_models = [gpu_model, secondary_gpu]
        layout_matrix = layout_fn(gpu_model, secondary_gpu)

    fuser = WorkloadFuser(gpu_models=gpu_models, seed=seed)
    dataset = fuser.generate_fused_dataset(
        num_requests=num_jobs, source="azure", workload=workload
    )
    env = ThermalGridEnv(
        fused_dataset=dataset,
        gpu_model=gpu_model,
        gpu_layout_matrix=layout_matrix,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    layout: str = "homogeneous",
) -> MaskablePPO:
    # Raw thermal penalties reach ~1e4/step under burst traffic, so rewards
    # are normalized and clipped. norm_obs=False: SpatialCNN expects the
    # exact physical matrices and does its own fixed scaling.
    env = DummyVecEnv([lambda: make_env(seed=seed, layout=layout)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    try:
        import tensorboard  # noqa: F401
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        tb_log = str(LOGS_DIR)
    except ImportError:
        tb_log = None
        print("[!] tensorboard not installed — training without TB logging.")

    policy_kwargs = dict(
        features_extractor_class=SpatialCNN,
        features_extractor_kwargs=dict(
            cnn_features_dim=256,
            queue_features_dim=32,
        ),
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log,
    )

    print(f"\nDevice: {model.device}")
    print(f"Feature extractor:\n{model.policy.features_extractor}\n")
    print(f"Training MaskablePPO for {total_timesteps:,} timesteps...\n")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    # VecNormalize statistics are required to reproduce the same reward
    # scaling at eval time (VecNormalize.load(VECNORM_PATH, venv))
    env.save(str(VECNORM_PATH))
    print(f"\nModel saved to {MODEL_PATH}.zip")
    print(f"VecNormalize statistics saved to {VECNORM_PATH}")
    return model


def evaluate(model: MaskablePPO, episodes: int = 3,
             layout: str = "homogeneous") -> None:
    """Run greedy masked rollouts and report scheduling/thermal metrics."""
    env = make_env(seed=123, layout=layout)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        peak_temp = 0.0

        while not done:
            action_masks = mask_fn(env)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
            peak_temp = max(peak_temp, info["max_temperature"])

        print(
            f"  Eval episode {ep}: reward={ep_reward:+.1f} | "
            f"placed={info['jobs_placed']} | queue={info['queue_length']} | "
            f"peakT={peak_temp:.1f}C | util={info['mean_utilization']*100:.0f}% | "
            f"throttle={info['throttle_events']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on ThermalGridEnv"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--layout", choices=list(LAYOUTS), default="homogeneous",
        help="Grid hardware layout (default: homogeneous H100)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run 3 greedy evaluation episodes after training",
    )
    args = parser.parse_args()

    trained_model = train(
        total_timesteps=args.timesteps, seed=args.seed, layout=args.layout
    )
    if args.eval:
        print("\nEvaluating trained policy...")
        evaluate(trained_model, layout=args.layout)


if __name__ == "__main__":
    main()
