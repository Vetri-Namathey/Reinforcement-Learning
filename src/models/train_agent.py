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
from stable_baselines3 import PPO
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

LAYOUTS = {
    "homogeneous": None,
    "checkerboard": make_checkerboard_layout,
    "split":        make_split_layout,
}

# ── Train / eval trace split ─────────────────────────────────────────────
# Training draws a fresh random EPISODE_JOBS-long window out of trace rows
# 0..TRAIN_POOL_JOBS on every reset; evaluation uses a fixed, disjoint slice
# starting at EVAL_OFFSET. Training on a single fixed window (the previous
# behaviour) gives PPO one deterministic trajectory to memorise -- the reward
# curve rises but nothing generalises, and held-out performance collapses.
TRAIN_POOL_JOBS = 4000     # trace rows 0..4000 form the training pool
EPISODE_JOBS    = 500      # jobs per episode
EVAL_OFFSET     = 4000     # eval slice starts where the pool ends
EVAL_JOBS       = 500


class InvalidActionPenaltyWrapper(gym.Wrapper):
    """Wrapper that penalizes invalid actions instead of masking them.
    
    Used for vanilla PPO ablation to compare masking vs. penalty-based constraint.
    """
    
    INVALID_ACTION_PENALTY = -100.0
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        env = self.env.unwrapped
        placed_before = env.jobs_placed

        obs, reward, terminated, truncated, info = self.env.step(action)

        # An action is invalid iff it named a cell but placed nothing: the cell
        # was busy/inactive, or the queue was empty.
        #
        # Testing obs["occupancy"][action] > 0 *after* the step (the previous
        # implementation) inverted this: a successful placement is precisely
        # what sets occupancy > 0, so every legal placement was fined -100 and
        # the agent learned never to schedule. That made the masking ablation a
        # measurement of this bug rather than of masking.
        if action != env.WAIT_ACTION and env.jobs_placed == placed_before:
            reward += self.INVALID_ACTION_PENALTY

        return obs, reward, terminated, truncated, info


def mask_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker hook: pull the current (num_cells+1)-element mask from the env."""
    return env.unwrapped._compute_action_mask().astype(bool)


def make_env(
    num_jobs: int = 500,
    workload: str = "conv",
    gpu_model: str = "H100",
    seed: int = 42,
    max_episode_steps: int = 5_000,
    layout: str = "homogeneous",
    secondary_gpu: str = "A100",
    grid_size: int = 6,
    reward_variant: str = "full",
    use_masking: bool = True,
    throttle_override: float | None = None,
    offset: int = 0,
    episode_jobs: int | None = None,
    randomize_start: bool = False,
) -> gym.Env:
    """
    Build the wrapped training environment.
    
    Without masking: ThermalGridEnv -> TimeLimit -> InvalidActionPenalty -> Monitor
    With masking:    ThermalGridEnv -> TimeLimit -> ActionMasker -> Monitor

    Args:
        reward_variant: One of ("full", "no_thermal", "no_sla", "linear_thermal")
        use_masking: If True, use ActionMasker for structurally-enforced constraints.
                     If False, penalize invalid actions in the reward signal.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Choose from {list(LAYOUTS)}")

    layout_fn = LAYOUTS[layout]
    if layout_fn is None:
        gpu_models = [gpu_model]
        layout_matrix = None
    else:
        gpu_models = [gpu_model, secondary_gpu]
        layout_matrix = layout_fn(gpu_model, secondary_gpu, grid_size=grid_size)

    fuser = WorkloadFuser(gpu_models=gpu_models, seed=seed)
    dataset = fuser.generate_fused_dataset(
        num_requests=num_jobs, source="azure", workload=workload, offset=offset
    )
    env = ThermalGridEnv(
        fused_dataset=dataset,
        gpu_model=gpu_model,
        gpu_layout_matrix=layout_matrix,
        grid_size=grid_size,
        reward_variant=reward_variant,
        enforce_action_mask=use_masking,  # Allow invalid actions when not using masking
        throttle_temp_override=throttle_override,
        episode_jobs=episode_jobs,
        randomize_start=randomize_start,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    if use_masking:
        env = ActionMasker(env, mask_fn)
    else:
        env = InvalidActionPenaltyWrapper(env)
    
    env = Monitor(env)
    return env


def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    layout: str = "homogeneous",
    grid_size: int = 6,
    reward_variant: str = "full",
    use_masking: bool = True,
    throttle_override: float | None = None,
    offset: int = 0,
) -> MaskablePPO | PPO:
    """Train an RL agent on ThermalGridEnv.
    
    Args:
        reward_variant: One of ("full", "no_thermal", "no_sla", "linear_thermal")
        use_masking: If True, use MaskablePPO + ActionMasker; if False, use vanilla PPO + penalty.
    
    Returns:
        Trained model (MaskablePPO or PPO depending on use_masking).
    """
    # Generate unique model path based on variant
    model_suffix = f"_{reward_variant}" if reward_variant != "full" else ""
    masking_suffix = "" if use_masking else "_no_masking"
    model_path = MODELS_DIR / f"spatial_thermal_ppo{model_suffix}{masking_suffix}"
    vecnorm_path = MODELS_DIR / f"vec_normalize{model_suffix}{masking_suffix}.pkl"
    
    env = DummyVecEnv([
        lambda: make_env(
            num_jobs=TRAIN_POOL_JOBS, seed=seed, layout=layout,
            grid_size=grid_size, reward_variant=reward_variant,
            use_masking=use_masking, throttle_override=throttle_override,
            offset=offset, episode_jobs=EPISODE_JOBS, randomize_start=True,
        )
    ])
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

    # Choose between MaskablePPO and vanilla PPO
    if use_masking:
        model_class = MaskablePPO
        policy_name = "MultiInputPolicy"
    else:
        model_class = PPO
        policy_name = "MultiInputPolicy"

    model = model_class(
        policy=policy_name,
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

    agent_type = "MaskablePPO" if use_masking else "vanilla PPO"
    reward_label = f" [{reward_variant}]" if reward_variant != "full" else ""
    print(f"\nDevice: {model.device}")
    print(f"Agent: {agent_type}{reward_label}")
    print(f"Feature extractor:\n{model.policy.features_extractor}\n")
    print(f"Training {agent_type} for {total_timesteps:,} timesteps...\n")

    agent_prefix = "MaskablePPO" if use_masking else "PPO"
    tb_log_name = agent_prefix if reward_variant == "full" else f"{agent_prefix}_{reward_variant}"
    model.learn(total_timesteps=total_timesteps, progress_bar=False, tb_log_name=tb_log_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    
    print(f"\nModel saved to {model_path}.zip")
    print(f"VecNormalize statistics saved to {vecnorm_path}")
    
    return model


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model: MaskablePPO | PPO,
    episodes: int = 3,
    layout: str = "homogeneous",
    grid_size: int = 6,
    reward_variant: str = "full",
    use_masking: bool = True,
    throttle_override: float | None = None,
    offset: int = EVAL_OFFSET,
) -> None:
    """Run greedy masked rollouts and report scheduling/thermal metrics on held-out data."""
    env = make_env(
        num_jobs=EVAL_JOBS, seed=123, layout=layout, grid_size=grid_size,
        reward_variant=reward_variant, use_masking=use_masking,
        throttle_override=throttle_override, offset=offset,
        episode_jobs=None, randomize_start=False,
    )
    print(f"  Eval slice: trace rows {offset}..{offset + EVAL_JOBS} "
          f"(disjoint from the 0..{TRAIN_POOL_JOBS} training pool)")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        peak_temp = 0.0

        while not done:
            if use_masking:
                action_masks = mask_fn(env)
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            
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
        description="Train RL agent on ThermalGridEnv with ablation support"
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
        "--grid-size", type=int, default=6,
        help="Grid dimension (default: 6 for 6x6)",
    )
    parser.add_argument(
        "--reward-variant", choices=["full", "no_thermal", "no_sla", "linear_thermal"],
        default="full",
        help="Reward function variant for ablation study (default: full)",
    )
    parser.add_argument(
        "--use-masking", action="store_true", default=True,
        help="Use action masking (MaskablePPO) instead of penalty-based constraints",
    )
    parser.add_argument(
        "--no-masking", dest="use_masking", action="store_false",
        help="Disable action masking; use penalty-based invalid action handling (vanilla PPO ablation)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run 3 greedy evaluation episodes after training",
    )
    parser.add_argument(
        "--throttle-override", type=float, default=None,
        help="Override throttle threshold (e.g. 72.0) to force real thermal crossings for ablation",
    )
    args = parser.parse_args()

    trained_model = train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        layout=args.layout,
        grid_size=args.grid_size,
        reward_variant=args.reward_variant,
        use_masking=args.use_masking,
        # Must reach training too: passing it only to evaluate() trained the
        # agent at the real 87/85 C thresholds and then scored it at the
        # overridden one, so the ablation compared two different environments.
        throttle_override=args.throttle_override,
    )
    
    if args.eval:
        print("\nEvaluating trained policy...")
        evaluate(
            trained_model,
            layout=args.layout,
            grid_size=args.grid_size,
            reward_variant=args.reward_variant,
            use_masking=args.use_masking,
            throttle_override=args.throttle_override,
        )


if __name__ == "__main__":
    main()
