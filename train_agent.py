"""
train_agent.py
==============
Training pipeline for the Spatial-Phase RL Scheduler.

Trains a MaskablePPO agent (sb3-contrib) on the ThermalGridEnv using a
custom CNN feature extractor:

  - The three 6x6 grid observations (temperatures, power_draw, occupancy)
    are stacked into a (batch, 3, 6, 6) tensor and processed spatially
    by a small CNN, preserving the 2D adjacency the Laplacian physics
    creates (a hot neighbor matters -- an MLP would destroy that).
  - queue_stats [queue_length, head_job_wait_time] is processed by a
    small linear layer.
  - Both streams are concatenated into the final feature vector.

Action masking (occupied cells / forced Wait on empty queue) is supplied
to MaskablePPO through the ActionMasker wrapper, reading the same mask
the environment already exposes in its observation.

Usage:
    python train_agent.py                     # full 100k-timestep run
    python train_agent.py --timesteps 5000    # short smoke run
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from data_fusion_pipeline import WorkloadFuser
from thermal_grid_env import ThermalGridEnv

MODEL_PATH = "spatial_thermal_ppo"   # SB3 appends .zip


# ══════════════════════════════════════════════════════════════════════
# 1. Custom Feature Extractor
# ══════════════════════════════════════════════════════════════════════

class SpatialCNN(BaseFeaturesExtractor):
    """
    Dict-observation feature extractor.

    Grid stream:   temperatures/power_draw/occupancy (each 6x6) are
                   stacked channel-wise -> (B, 3, 6, 6) -> two Conv2d+ReLU
                   layers -> flatten -> linear head.
    Queue stream:  queue_stats (2,) -> linear layer + ReLU.
    Output:        concat(grid_features, queue_features).

    The action_mask key is intentionally NOT fed to the network — masking
    is enforced in MaskablePPO's action distribution, not learned.
    """

    GRID_KEYS = ("temperatures", "power_draw", "occupancy")

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_features_dim: int = 256,
        queue_features_dim: int = 32,
    ):
        super().__init__(
            observation_space,
            features_dim=cnn_features_dim + queue_features_dim,
        )

        grid_shape = observation_space["temperatures"].shape   # (6, 6)
        n_channels = len(self.GRID_KEYS)                        # 3

        # Per-channel normalization scales (bring inputs to ~[0, 1])
        self.temp_scale = float(observation_space["temperatures"].high.max())
        self.power_scale = float(observation_space["power_draw"].high.max())
        self.occupancy_scale = 60.0    # seconds; typical job < 1 min
        self.queue_scale = torch.tensor([100.0, 60.0])  # [length, wait s]

        # ── Grid stream: (B, 3, 6, 6) spatial CNN ──
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer the flattened CNN size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, *grid_shape)
            n_flatten = self.cnn(dummy).shape[1]                # 64*6*6 = 2304

        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, cnn_features_dim),
            nn.ReLU(),
        )

        # ── Queue stream: (B, 2) linear ──
        self.queue_net = nn.Sequential(
            nn.Linear(observation_space["queue_stats"].shape[0],
                      queue_features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Stack the three 6x6 grids along the channel dimension
        grid = torch.stack(
            [
                observations["temperatures"] / self.temp_scale,
                observations["power_draw"] / self.power_scale,
                observations["occupancy"] / self.occupancy_scale,
            ],
            dim=1,
        )                                                       # (B, 3, 6, 6)

        grid_features = self.cnn_head(self.cnn(grid))           # (B, 256)

        queue = observations["queue_stats"] / self.queue_scale.to(
            observations["queue_stats"].device
        )
        queue_features = self.queue_net(queue)                  # (B, 32)

        return torch.cat([grid_features, queue_features], dim=1)


# ══════════════════════════════════════════════════════════════════════
# 2. Environment Factory + Action Mask Hook
# ══════════════════════════════════════════════════════════════════════

def mask_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker hook: pull the current 37-element mask from the env."""
    return env.unwrapped._compute_action_mask().astype(bool)


def make_env(
    num_jobs: int = 500,
    workload: str = "conv",
    gpu_model: str = "H100",
    seed: int = 42,
    max_episode_steps: int = 5_000,
) -> gym.Env:
    """
    Build the wrapped training environment:
        ThermalGridEnv -> TimeLimit -> ActionMasker -> Monitor

    TimeLimit truncates episodes where an untrained policy Waits forever
    (the backlog termination condition would otherwise never trigger).
    """
    fuser = WorkloadFuser(gpu_model=gpu_model, seed=seed)
    dataset = fuser.generate_fused_dataset(
        num_requests=num_jobs, source="azure", workload=workload
    )
    env = ThermalGridEnv(fused_dataset=dataset, gpu_model=gpu_model)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


# ══════════════════════════════════════════════════════════════════════
# 3. Training Loop
# ══════════════════════════════════════════════════════════════════════

def train(total_timesteps: int = 100_000, seed: int = 42) -> MaskablePPO:
    env = make_env(seed=seed)

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
        tensorboard_log="./tb_logs",
    )

    print(f"\nDevice: {model.device}")
    print(f"Feature extractor:\n{model.policy.features_extractor}\n")
    print(f"Training MaskablePPO for {total_timesteps:,} timesteps...\n")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}.zip")
    return model


# ══════════════════════════════════════════════════════════════════════
# 4. Quick Post-Training Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate(model: MaskablePPO, episodes: int = 3) -> None:
    """Run greedy masked rollouts and report scheduling/thermal metrics."""
    env = make_env(seed=123)

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
            f"peakT={peak_temp:.1f}C | throttle={info['throttle_events']}"
        )


if __name__ == "__main__":
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
        "--eval", action="store_true",
        help="Run 3 greedy evaluation episodes after training",
    )
    args = parser.parse_args()

    trained_model = train(total_timesteps=args.timesteps, seed=args.seed)
    if args.eval:
        print("\nEvaluating trained policy...")
        evaluate(trained_model)
