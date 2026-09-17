"""
Return-filtered offline RL for the Spatial-Phase scheduler.

Why not PPO
------------
On-policy PPO does not learn this task. It fails from random initialisation, it
fails with the action space cut to six candidates, and it degrades a policy that
already imitates the best heuristic at 95% accuracy. The cause is exploration and
credit assignment, not representation: the network can express a good policy, PPO
just cannot find or keep one when the largest reward term (+10 per placement) is
nearly independent of which cell was chosen and the thermal consequence lands 70+
steps later, mixed across 36 cells.

What this does instead
-----------------------
1. Roll out *perturbed* copies of Best-Fit -- the teacher, but with a probability
   eps of taking a different candidate. Each perturbation is a small, random
   deviation from a known-good strategy.
2. Score every episode by its actual return.
3. Keep only the episodes that beat the deterministic teacher, and clone those.

Step 3 is what makes this reinforcement learning rather than imitation: the
training signal is the *outcome*, not the teacher's choice. The teacher supplies
a sensible region of policy space to search in; the returns decide what is kept.
Deviations that happened to work are reinforced, deviations that did not are
discarded. That is policy improvement by selection, and it can exceed the teacher
because it learns from trajectories the teacher would never have produced.

This is filtered behaviour cloning / reward-weighted regression, the standard
offline-RL recipe when on-policy exploration is infeasible.

Usage (from the project root):
    python -m src.models.offline_rl
    python -m src.models.offline_rl --episodes 120 --keep-quantile 0.25
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from configs import MODELS_DIR
from src.baselines.best_fit import BestFitScheduler
from src.models.spatial_cnn import SpatialCNN
from src.models.train_agent import (DEFAULT_ARRIVAL_ACCEL, DEFAULT_CANDIDATE_K, DEFAULT_LAYOUT,
                                    DEFAULT_SLOTS, EPISODE_JOBS, TRAIN_POOL_JOBS, make_env)

OUT_PATH = MODELS_DIR / "offline_rl"


class PerturbedBestFit:
    """Best-Fit that takes a random legal cell with probability eps."""

    name = "Perturbed Best-Fit"

    def __init__(self, eps: float, seed: int):
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.inner = BestFitScheduler()

    def select_action(self, obs):
        if self.rng.random() < self.eps:
            valid = np.flatnonzero(obs["action_mask"])
            return int(self.rng.choice(valid))
        return self.inner.select_action(obs)


def rollout(env, policy, seed):
    """Play one episode, returning its transitions and total return."""
    obs, _ = env.reset(seed=seed)
    obs_buf, act_buf, mask_buf = [], [], []
    total, terminated, truncated, steps = 0.0, False, False, 0
    while not (terminated or truncated) and steps < 300_000:
        action = policy.select_action(obs)
        obs_buf.append({k: v.copy() for k, v in obs.items()})
        act_buf.append(action)
        mask_buf.append(obs["action_mask"].copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
    return obs_buf, act_buf, mask_buf, total


def main() -> int:
    ap = argparse.ArgumentParser(description="Return-filtered offline RL")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--keep-quantile", type=float, default=0.30,
                    help="Fraction of episodes to keep, best-return first.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env(num_jobs=TRAIN_POOL_JOBS, layout=DEFAULT_LAYOUT, episode_jobs=EPISODE_JOBS,
                   randomize_start=True, slots=DEFAULT_SLOTS,
                   arrival_accel=DEFAULT_ARRIVAL_ACCEL, candidate_k=DEFAULT_CANDIDATE_K,
                   env_seed=args.seed)

    # ── 1. teacher reference on each window, then perturbed attempts ──
    print(f"[1/3] Rolling out {args.episodes} episodes "
          f"(teacher + perturbations) ...")
    episodes, teacher_returns, improved = [], [], 0
    rng = np.random.default_rng(args.seed)
    for i in range(args.episodes):
        window_seed = int(rng.integers(0, 10_000))
        # deterministic teacher on this exact window, as the bar to beat
        _, _, _, base = rollout(env, BestFitScheduler(), window_seed)
        teacher_returns.append(base)
        eps = float(rng.choice([0.05, 0.10, 0.20]))
        o, a, m, total = rollout(env, PerturbedBestFit(eps, window_seed), window_seed)
        episodes.append({"obs": o, "act": a, "mask": m, "ret": total, "base": base})
        if total > base:
            improved += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.episodes} episodes  "
                  f"({improved} beat the teacher so far)")

    gains = np.array([e["ret"] - e["base"] for e in episodes])
    print(f"\n  perturbations that beat the teacher: {improved}/{args.episodes} "
          f"({100 * improved / args.episodes:.0f}%)")
    print(f"  mean gain over teacher: {gains.mean():+.0f}   best: {gains.max():+.0f}")

    # ── 2. keep only the episodes that actually beat the teacher ──
    winners = [e for e in episodes if e["ret"] > e["base"]]
    if not winners:
        print("\n  No perturbation beat the teacher. Nothing to learn from -- "
              "the teacher is locally optimal under this reward.")
        return 1
    winners.sort(key=lambda e: e["ret"] - e["base"], reverse=True)
    n_keep = max(1, int(len(episodes) * args.keep_quantile))
    winners = winners[:n_keep]
    print(f"[2/3] Training on the top {len(winners)} episodes "
          f"(mean gain {np.mean([w['ret'] - w['base'] for w in winners]):+.0f})")

    obs_buf, act_buf, mask_buf = [], [], []
    for w in winners:
        obs_buf += w["obs"]
        act_buf += w["act"]
        mask_buf += w["mask"]

    # ── 3. clone the winning behaviour ──
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=SpatialCNN,
            features_extractor_kwargs=dict(cnn_features_dim=256, queue_features_dim=32),
            net_arch=dict(pi=[], vf=[64, 64]),
            share_features_extractor=False,
        ),
        verbose=0, seed=args.seed,
    )
    device = model.device
    keys = list(obs_buf[0].keys())
    observations = {k: torch.as_tensor(np.stack([o[k] for o in obs_buf])).float().to(device)
                    for k in keys}
    actions = torch.as_tensor(np.array(act_buf)).long().to(device)
    masks = torch.as_tensor(np.stack(mask_buf)).bool().to(device)

    policy = model.policy
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n = len(act_buf)
    order = np.arange(n)
    print(f"[3/3] Cloning {n} winning transitions for {args.epochs} epochs ...")
    print(f"  {'epoch':>6s}{'loss':>10s}{'agreement':>12s}")
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        total_loss, correct = 0.0, 0
        for start in range(0, n, args.batch):
            batch = order[start:start + args.batch]
            batch_obs = {k: observations[k][batch] for k in keys}
            dist = policy.get_distribution(batch_obs, action_masks=masks[batch])
            loss = -dist.log_prob(actions[batch]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(batch)
            correct += int((dist.distribution.probs.argmax(1) == actions[batch]).sum())
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  {epoch:6d}{total_loss / n:10.4f}{100 * correct / n:11.1f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_PATH))
    print(f"\nSaved -> {OUT_PATH}.zip")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
