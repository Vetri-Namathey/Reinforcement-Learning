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
Each round:
  1. Roll out *perturbed* copies of the current reference policy -- the same
     policy, but with probability eps of taking a different legal candidate.
  2. Score every episode by its actual return, against that reference policy's
     own return on the identical window.
  3. Keep only the episodes that beat the reference, and clone those.

Step 3 is what makes this reinforcement learning rather than imitation: the
training signal is the *outcome*, not the reference's choice. The reference
supplies a sensible region of policy space to search in; the returns decide what
is kept. Deviations that happened to work are reinforced, deviations that did not
are discarded. That is policy improvement by selection, and it can exceed the
reference because it learns from trajectories the reference would never produce.

This is filtered behaviour cloning / reward-weighted regression, the standard
offline-RL recipe when on-policy exploration is infeasible.

Why it iterates
---------------
One round of this is rejection sampling around a *fixed* teacher, and its ceiling
is the best trajectory inside a fixed exploration radius of that teacher: with
perturbation rate eps, the achievable return is bounded by sup over the
neighbourhood N_eps(teacher), no matter how many samples are drawn. If Best-Fit
is locally near-optimal -- plausible, since it reads the same features the agent
does -- one round cannot escape it, which is exactly what a first round measured.

Rounds 2+ therefore re-sample around the policy that round 1 produced, so the
neighbourhood moves instead of being redrawn around the same centre. That is
generalised policy iteration, and it is also the only adaptation path in this
repo: re-running it on freshly collected deployment episodes is how the policy
would be refreshed as real hardware drifts.

Each round is accepted only if it beats the incumbent on a held-out set of
validation windows that no round trains on. A round that fails validation is
discarded and the incumbent is kept, so iterating cannot silently degrade the
shipped policy -- the failure mode that makes naive self-imitation collapse.

Usage (from the project root):
    python -m src.models.offline_rl
    python -m src.models.offline_rl --rounds 1            # the original single round
    python -m src.models.offline_rl --episodes 120 --keep-quantile 0.25
    python -m src.models.offline_rl --init-from bc_pretrained
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from configs import MODELS_DIR
from src.baselines.best_fit import BestFitScheduler
from src.models.spatial_cnn import SpatialCNN
from src.models.train_agent import (DEFAULT_ARRIVAL_ACCEL, DEFAULT_CANDIDATE_K, DEFAULT_LAYOUT,
                                    DEFAULT_SLOTS, EPISODE_JOBS, TRAIN_POOL_JOBS, make_env)

OUT_PATH = MODELS_DIR / "offline_rl"


class PerturbedScheduler:
    """Any scheduler, but taking a random legal action with probability eps.

    The exploration operator for the filter step. Perturbations are uniform over
    the *legal* set, so every sampled trajectory is still a valid schedule --
    the search explores which cell to prefer, never whether a placement is
    admissible.
    """

    name = "Perturbed"

    def __init__(self, inner, eps: float, seed: int):
        self.inner = inner
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.name = f"Perturbed {getattr(inner, 'name', type(inner).__name__)}"

    def select_action(self, obs):
        if self.rng.random() < self.eps:
            valid = np.flatnonzero(obs["action_mask"])
            return int(self.rng.choice(valid))
        return self.inner.select_action(obs)


class PolicyScheduler:
    """Wraps a MaskablePPO policy in the scheduler interface used above.

    Lets a trained round become the reference policy the next round perturbs
    around, which is what turns one round of rejection sampling into policy
    iteration.
    """

    name = "Learned policy"

    def __init__(self, model: MaskablePPO):
        self.model = model

    def select_action(self, obs):
        action, _ = self.model.predict(
            obs, action_masks=obs["action_mask"].astype(bool), deterministic=True)
        return int(action)


def rollout(env, policy, seed, collect=True):
    """Play one episode, returning its transitions and total return."""
    obs, _ = env.reset(seed=seed)
    obs_buf, act_buf, mask_buf = [], [], []
    total, terminated, truncated, steps = 0.0, False, False, 0
    while not (terminated or truncated) and steps < 300_000:
        action = policy.select_action(obs)
        if collect:
            obs_buf.append({k: v.copy() for k, v in obs.items()})
            act_buf.append(action)
            mask_buf.append(obs["action_mask"].copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
    return obs_buf, act_buf, mask_buf, total


def build_model(env, seed: int, init_from: str | None):
    """Fresh MaskablePPO, optionally warm-started from an existing checkpoint."""
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=SpatialCNN,
            features_extractor_kwargs=dict(cnn_features_dim=256, queue_features_dim=32),
            net_arch=dict(pi=[], vf=[64, 64]),
            share_features_extractor=False,
        ),
        verbose=0, seed=seed,
    )
    if init_from:
        # The README used to describe a BC warm-start feeding this stage while
        # the code built a fresh network and ignored bc_pretrained.zip entirely.
        # The chain is now real and opt-in, and the default stays fresh so the
        # filter step is measured on its own.
        src = MODELS_DIR / init_from
        donor = MaskablePPO.load(str(src))
        model.policy.load_state_dict(donor.policy.state_dict())
        print(f"  warm-started policy weights from {src.name}.zip")
    return model


def clone(model, obs_buf, act_buf, mask_buf, epochs, lr, batch, quiet=False):
    """Supervised-clone the given transitions into the model's policy."""
    device = model.device
    keys = list(obs_buf[0].keys())
    observations = {k: torch.as_tensor(np.stack([o[k] for o in obs_buf])).float().to(device)
                    for k in keys}
    actions = torch.as_tensor(np.array(act_buf)).long().to(device)
    masks = torch.as_tensor(np.stack(mask_buf)).bool().to(device)

    policy = model.policy
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(act_buf)
    order = np.arange(n)
    if not quiet:
        print(f"  cloning {n} winning transitions for {epochs} epochs ...")
        print(f"  {'epoch':>6s}{'loss':>10s}{'agreement':>12s}")
    for epoch in range(epochs):
        np.random.shuffle(order)
        total_loss, correct = 0.0, 0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            batch_obs = {k: observations[k][idx] for k in keys}
            dist = policy.get_distribution(batch_obs, action_masks=masks[idx])
            loss = -dist.log_prob(actions[idx]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(idx)
            correct += int((dist.distribution.probs.argmax(1) == actions[idx]).sum())
        if not quiet and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  {epoch:6d}{total_loss / n:10.4f}{100 * correct / n:11.1f}%")
    return model


def validate(env, policy, windows) -> float:
    """Mean return over a fixed set of windows that no round trains on."""
    return float(np.mean([rollout(env, policy, w, collect=False)[3] for w in windows]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Return-filtered offline RL")
    ap.add_argument("--rounds", type=int, default=3,
                    help="Policy-iteration rounds. 1 reproduces the original "
                         "single-round rejection sampling around Best-Fit.")
    ap.add_argument("--episodes", type=int, default=120,
                    help="Perturbed episodes sampled per round.")
    ap.add_argument("--keep-quantile", type=float, default=0.30,
                    help="Fraction of episodes to keep, best-return first.")
    ap.add_argument("--val-windows", type=int, default=6,
                    help="Held-out windows used to accept or reject each round.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="Checkpoint in artifacts/models to warm-start round 1 from, "
                         "e.g. bc_pretrained. Default: fresh network.")
    args = ap.parse_args()

    env = make_env(num_jobs=TRAIN_POOL_JOBS, layout=DEFAULT_LAYOUT, episode_jobs=EPISODE_JOBS,
                   randomize_start=True, slots=DEFAULT_SLOTS,
                   arrival_accel=DEFAULT_ARRIVAL_ACCEL, candidate_k=DEFAULT_CANDIDATE_K,
                   env_seed=args.seed)

    rng = np.random.default_rng(args.seed)
    # Validation windows are drawn once, from a disjoint seed range, and no
    # round ever trains on them. Accepting a round on its own training windows
    # would accept overfitting to the sampled perturbations.
    val_windows = [int(w) for w in rng.integers(50_000, 60_000, size=args.val_windows)]

    teacher = BestFitScheduler()
    incumbent_score = validate(env, teacher, val_windows)
    print(f"Validation bar: Best-Fit scores {incumbent_score:.0f} "
          f"over {len(val_windows)} held-out windows\n")

    best_model = None
    best_score = incumbent_score
    reference = teacher
    history = []

    for rnd in range(1, args.rounds + 1):
        print("=" * 70)
        print(f"  ROUND {rnd}/{args.rounds} -- perturbing around "
              f"{getattr(reference, 'name', 'policy')}")
        print("=" * 70)

        # ── 1. reference on each window, then perturbed attempts ──
        episodes, improved = [], 0
        for i in range(args.episodes):
            window_seed = int(rng.integers(0, 10_000))
            _, _, _, base = rollout(env, reference, window_seed, collect=False)
            eps = float(rng.choice([0.05, 0.10, 0.20]))
            o, a, m, total = rollout(env, PerturbedScheduler(reference, eps, window_seed),
                                     window_seed)
            episodes.append({"obs": o, "act": a, "mask": m, "ret": total, "base": base})
            if total > base:
                improved += 1
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{args.episodes} episodes "
                      f"({improved} beat the reference so far)")

        gains = np.array([e["ret"] - e["base"] for e in episodes])
        print(f"  beat the reference: {improved}/{args.episodes} "
              f"({100 * improved / args.episodes:.0f}%)   "
              f"mean gain {gains.mean():+.0f}   best {gains.max():+.0f}")

        # ── 2. keep only the episodes that actually beat the reference ──
        winners = [e for e in episodes if e["ret"] > e["base"]]
        if not winners:
            print("  No perturbation beat the reference -- it is locally optimal "
                  "under this reward at this eps. Stopping.")
            break
        winners.sort(key=lambda e: e["ret"] - e["base"], reverse=True)
        winners = winners[:max(1, int(len(episodes) * args.keep_quantile))]
        print(f"  training on the top {len(winners)} episodes (mean gain "
              f"{np.mean([w['ret'] - w['base'] for w in winners]):+.0f})")

        obs_buf, act_buf, mask_buf = [], [], []
        for w in winners:
            obs_buf += w["obs"]
            act_buf += w["act"]
            mask_buf += w["mask"]

        # ── 3. clone the winning behaviour ──
        # Rounds after the first continue from the incumbent's weights, so each
        # round refines a policy rather than restarting the search.
        if best_model is None:
            model = build_model(env, args.seed, args.init_from)
        else:
            model = copy.deepcopy(best_model)
        clone(model, obs_buf, act_buf, mask_buf, args.epochs, args.lr, args.batch)

        # ── 4. accept only on held-out improvement ──
        score = validate(env, PolicyScheduler(model), val_windows)
        delta = score - best_score
        accepted = score > best_score
        print(f"\n  validation: {score:.0f} vs incumbent {best_score:.0f} "
              f"({delta:+.0f})  ->  {'ACCEPTED' if accepted else 'REJECTED, keeping incumbent'}")
        history.append({"round": rnd, "val_score": score, "incumbent": best_score,
                        "accepted": accepted, "improved_frac": improved / args.episodes})

        if accepted:
            best_model, best_score = model, score
            reference = PolicyScheduler(best_model)
        elif best_model is None:
            # Round 1 failed to beat Best-Fit. Keep it anyway as the shipped
            # policy -- it is the only network there is -- but say so plainly.
            best_model = model
            print("  (round 1 is kept as the only candidate policy despite not "
                  "clearing the Best-Fit bar; later rounds may still improve it)")
        print()

    if best_model is None:
        print("No policy was produced.")
        return 1

    print("=" * 70)
    print("  SUMMARY")
    print(f"  {'round':>6}{'val score':>12}{'incumbent':>12}{'verdict':>12}")
    for h in history:
        print(f"  {h['round']:6d}{h['val_score']:12.0f}{h['incumbent']:12.0f}"
              f"{'accept' if h['accepted'] else 'reject':>12}")
    print(f"\n  Best-Fit bar: {incumbent_score:.0f}   final policy: {best_score:.0f} "
          f"({best_score - incumbent_score:+.0f})")
    print("  This is a training-time validation figure on 6 windows, not the")
    print("  reported result. Score the saved checkpoint with:")
    print("      python -m src.models.held_out_eval")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save(str(OUT_PATH))
    print(f"\nSaved -> {OUT_PATH}.zip")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
