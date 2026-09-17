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

# Concurrent jobs per cell (continuous batching). The environment itself still
# defaults to 1 for backward compatibility, but the *experiment* runs at 3:
# at K=1 the action mask leaves 0-1 legal cells on ~95% of loaded steps, so the
# MDP has no decision to make and every scheduler scores identically. K=3 keeps
# ~36 options open and puts peak temperature above the throttle point, which is
# what gives the thermal objective a gradient. Verify with:
#     python -m src.baselines.separation_gate --slots 1
#     python -m src.baselines.separation_gate --slots 3
DEFAULT_SLOTS = 3

# Offered-load scaling, kept at 1.0 = the trace's own recorded arrival rate.
# Raising it does create wider baseline separation, but it buys that by filling
# every cell: at slots=2/accel=1.4 the mask left <=1 legal cell on 53.5% of
# decisions, which is the same collapsed action space that made the original
# environment unlearnable. slots=3 at the recorded rate gives 35.6 legal cells
# per decision AND 88.7% separation, so no scaling is needed. Verify both with
# `python -m src.baselines.separation_gate`.
DEFAULT_ARRIVAL_ACCEL = 1.0

# ── PPO hyperparameters ──────────────────────────────────────────────────
# gamma must cover the episode, not a fraction of it. Episodes run ~1,150 steps
# and heat persists for 70+ (thermal time constant 1/0.14 = 7 s = 70 steps), but
# gamma=0.99 gives an effective horizon of only ~100 steps: 0.99^500 = 0.0066, so
# the cost of a hotspot 500 steps out is discounted to nothing. The agent then
# front-loads placements for the immediate +10 and lets the grid cook -- measured
# as episode reward FALLING 2310 -> 1060 across 100k steps while explained
# variance sat at 0.98. The critic was right; the objective was myopic.
#   0.999 -> horizon ~1000 steps, which matches the episode.
DEFAULT_GAMMA = 0.999
# Without an entropy bonus the policy collapsed onto 24 of 36 cells (placement
# std 10.8 vs 2.1 for least-occupied), concentrating heat into permanent hotspots.
DEFAULT_ENT_COEF = 0.01
# Longer rollouts give GAE a better view of a long episode.
DEFAULT_N_STEPS = 1024
# Parallel environments. One env means every gradient comes from a single
# trajectory, and with per-window reward varying by +/-1280 the policy gradient
# is mostly window noise. Eight independent windows per update average that out.
# n_steps is per-env, so 8 x 1024 = 8192 transitions per update.
DEFAULT_N_ENVS = 8

# The grid is heterogeneous by default. On a uniform H100 grid every cell is
# interchangeable, so the gap between a random policy and the best heuristic is
# only 453 reward (10%) -- below what PPO can resolve. Mixing H100 and A100 makes
# device choice matter: the same gap is 6,767 (15x larger), because putting a
# 4,000-token prefill on a 400 W A100 instead of a 700 W H100 is a real mistake
# with a real cost. Best-Fit, the one baseline that matches job size to device,
# beats every other heuristic by >2,500 reward here.
DEFAULT_LAYOUT = "checkerboard"

# Candidate shortlist size. The policy chooses among the k least-loaded, coolest
# legal cells rather than all 36. Ranking uses only load and heat -- never job
# size or device type -- so the job-to-device matching the agent must learn is
# left entirely to it.
DEFAULT_CANDIDATE_K = 6
# VecNormalize divides each reward by the running std of the *discounted return*
# and clips the result. Early in training that std is tiny, so rewards blow up and
# clip at +/-10; as it grows the effective signal shrinks. The result is a moving,
# clipped objective. Raw rewards here already sit in a sane ~[-15, +10] band, so
# normalising buys nothing and costs stationarity.
# Returns here run to ~2000, so an unnormalised value loss reaches ~5e3 against a
# policy-gradient loss of ~4e-3 -- a ratio of 1e6. Normalising returns to unit
# scale is what keeps the two terms in the same universe.
DEFAULT_NORM_REWARD = True


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
    slots: int = DEFAULT_SLOTS,
    arrival_accel: float = DEFAULT_ARRIVAL_ACCEL,
    env_seed: int | None = None,
    candidate_k: int | None = DEFAULT_CANDIDATE_K,
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
        num_requests=num_jobs, source="azure", workload=workload, offset=offset,
        arrival_accel=arrival_accel,
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
        max_jobs_per_cell=slots,
        candidate_k=candidate_k,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    if use_masking:
        env = ActionMasker(env, mask_fn)
    else:
        env = InvalidActionPenaltyWrapper(env)
    
    env = Monitor(env)
    if env_seed is not None:
        env.reset(seed=env_seed)
    return env


def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    layout: str = DEFAULT_LAYOUT,
    grid_size: int = 6,
    reward_variant: str = "full",
    use_masking: bool = True,
    throttle_override: float | None = None,
    offset: int = 0,
    slots: int = DEFAULT_SLOTS,
    arrival_accel: float = DEFAULT_ARRIVAL_ACCEL,
    gamma: float = DEFAULT_GAMMA,
    ent_coef: float = DEFAULT_ENT_COEF,
    n_steps: int = DEFAULT_N_STEPS,
    norm_reward: bool = DEFAULT_NORM_REWARD,
    episode_jobs: int = EPISODE_JOBS,
    randomize: bool = True,
    n_envs: int = DEFAULT_N_ENVS,
    candidate_k: int | None = DEFAULT_CANDIDATE_K,
    init_from: str | None = None,
    learning_rate: float = 3e-4,
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
    
    def _make(rank: int):
        # Distinct seeds so the eight workers draw different episode windows.
        def _init():
            return make_env(
                num_jobs=TRAIN_POOL_JOBS, seed=seed, layout=layout,
                grid_size=grid_size, reward_variant=reward_variant,
                use_masking=use_masking, throttle_override=throttle_override,
                offset=offset, episode_jobs=episode_jobs, randomize_start=randomize,
                slots=slots, arrival_accel=arrival_accel, env_seed=seed + 1000 * rank,
                candidate_k=candidate_k,
            )
        return _init

    env = DummyVecEnv([_make(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=norm_reward,
                       clip_reward=10.0, gamma=gamma)

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
        # pi=[] puts the action head directly on the extractor output, so the
        # per-cell score at index k feeds action k through a single weight.
        # Any hidden layer here re-scrambles the spatial correspondence that
        # SpatialCNN's 1x1 scoring head exists to preserve. The critic keeps
        # its hidden layers -- it needs to mix the whole grid into one number.
        net_arch=dict(pi=[], vf=[64, 64]),
        # Give the actor its own CNN. Shared (SB3's default) means one feature
        # extractor receives gradients from both heads -- and with value_loss at
        # ~5e3 against policy_gradient_loss at ~4e-3, essentially 100% of that
        # gradient is value regression. The measured effect was the policy
        # decaying smoothly from 1820 to 261 reward over 300k steps on a fixed
        # deterministic window while explained_variance held at 0.90: the critic
        # was learning perfectly and dragging the actor's features down with it.
        share_features_extractor=False,
    )

    # Choose between MaskablePPO and vanilla PPO
    if use_masking:
        model_class = MaskablePPO
        policy_name = "MultiInputPolicy"
    else:
        model_class = PPO
        policy_name = "MultiInputPolicy"

    if init_from:
        # Warm start from a behaviour-cloned policy. PPO cannot find a good
        # policy in this 37-action space from scratch (see pretrain_bc.py), so
        # the RL stage refines a strong starting point instead of searching for
        # one. Lower the learning rate: the default 3e-4 walks away from the
        # warm start before the freshly-initialised critic is accurate enough to
        # give useful advantages.
        init_path = MODELS_DIR / init_from
        print(f"Warm start from {init_path}.zip")
        model = model_class.load(str(init_path), env=env, device="auto")
        model.verbose = 1          # the BC checkpoint is saved silent
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
        model._setup_lr_schedule()
    else:
        model = model_class(
        policy=policy_name,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=256,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log,
        )

    agent_type = "MaskablePPO" if use_masking else "vanilla PPO"
    print(f"Concurrency: {slots} job(s) per cell | arrival accel: {arrival_accel}x")
    print(f"candidate_k={candidate_k}")
    print(f"layout={layout} | gamma={gamma} ent_coef={ent_coef} "
          f"n_steps={n_steps}x{n_envs}envs norm_reward={norm_reward}")
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
    slots: int = DEFAULT_SLOTS,
    arrival_accel: float = DEFAULT_ARRIVAL_ACCEL,
) -> None:
    """Run greedy masked rollouts and report scheduling/thermal metrics on held-out data."""
    env = make_env(
        num_jobs=EVAL_JOBS, seed=123, layout=layout, grid_size=grid_size,
        reward_variant=reward_variant, use_masking=use_masking,
        throttle_override=throttle_override, offset=offset,
        episode_jobs=None, randomize_start=False, slots=slots,
        arrival_accel=arrival_accel,
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
            f"peakT={peak_temp:.1f}C | TTTR={info['tttr']:.4f} | "
            f"throttled_work={info.get('throttled_work_ratio', 0.0)*100:.1f}%"
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
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                        help="Shortlist size the policy chooses among (0 = all cells).")
    parser.add_argument("--init-from", default=None,
                        help="Warm start from a checkpoint in artifacts/models "
                             "(e.g. bc_pretrained), instead of random init.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--layout", choices=list(LAYOUTS), default=DEFAULT_LAYOUT,
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
        "--slots", type=int, default=DEFAULT_SLOTS,
        help=f"Concurrent jobs per cell (default: {DEFAULT_SLOTS}). "
             "1 disables continuous batching and collapses the action space.",
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--episode-jobs", type=int, default=EPISODE_JOBS)
    parser.add_argument("--fixed-window", action="store_true",
                        help="Disable window randomisation (diagnostic).")
    parser.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument(
        "--arrival-accel", type=float, default=DEFAULT_ARRIVAL_ACCEL,
        help=f"Compress trace inter-arrival gaps by this factor "
             f"(default: {DEFAULT_ARRIVAL_ACCEL}). 1.0 = recorded rate.",
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
        slots=args.slots,
        arrival_accel=args.arrival_accel,
        gamma=args.gamma, ent_coef=args.ent_coef, n_steps=args.n_steps,
        n_envs=args.n_envs,
        candidate_k=(args.candidate_k or None),
        init_from=args.init_from, learning_rate=args.lr,
        episode_jobs=args.episode_jobs, randomize=not args.fixed_window,
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
            slots=args.slots, arrival_accel=args.arrival_accel,
        )


if __name__ == "__main__":
    main()
