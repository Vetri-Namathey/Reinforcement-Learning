"""
Behaviour-cloning warm start for the Spatial-Phase RL scheduler.

Why this stage exists
----------------------
PPO from a random initialisation does not learn this task. Measured across ten
configurations (discount 0.99-0.999, entropy bonus, spatial policy head, separate
feature extractors, work-conserving mask, immediate placement credit, fixed
single-window episodes, 300k steps), the policy converged to or below a uniform
random policy every time.

The cause is not representational. Cloning Best-Fit reaches ~90% action agreement,
so the network can express a good policy comfortably. The cause is exploration:
37 discrete actions, a reward whose largest term (+10 per placement) is almost
independent of which cell is chosen, and thermal consequences that land 70+ steps
later. Random exploration almost never stumbles onto a coherent placement strategy,
so there is no gradient toward one.

Supervised pretraining removes that barrier. The agent starts where the strongest
heuristic already is, and PPO then only has to improve locally -- the same
supervised-then-RL staging used in AlphaGo and standard imitation-learning
pipelines. The RL stage is what must earn the final result: if fine-tuning does
not beat the teacher, this is a clone and should be reported as one.

Usage (from the project root):
    python -m src.models.pretrain_bc
    python -m src.models.pretrain_bc --demos 20000 --epochs 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from configs import MODELS_DIR
from src.baselines.best_fit import BestFitScheduler
from src.models.spatial_cnn import SpatialCNN
from src.models.train_agent import (DEFAULT_ARRIVAL_ACCEL, DEFAULT_LAYOUT, DEFAULT_SLOTS,
                                    EPISODE_JOBS, TRAIN_POOL_JOBS, make_env)

BC_PATH = MODELS_DIR / "bc_pretrained"


def collect(env, teacher, n_demos: int, seed: int = 0):
    """Roll the teacher out and record (observation, action, mask) triples."""
    obs_buf, act_buf, mask_buf = [], [], []
    obs, _ = env.reset(seed=seed)
    for i in range(n_demos):
        action = teacher.select_action(obs)
        obs_buf.append({k: v.copy() for k, v in obs.items()})
        act_buf.append(action)
        mask_buf.append(obs["action_mask"].copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + i + 1)
    return obs_buf, act_buf, mask_buf


def main() -> int:
    ap = argparse.ArgumentParser(description="Behaviour-clone Best-Fit as a PPO warm start")
    ap.add_argument("--demos", type=int, default=16000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    args = ap.parse_args()

    env = make_env(num_jobs=TRAIN_POOL_JOBS, layout=args.layout, episode_jobs=EPISODE_JOBS,
                   randomize_start=True, slots=args.slots,
                   arrival_accel=DEFAULT_ARRIVAL_ACCEL, env_seed=args.seed)

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

    print(f"[1/2] Collecting {args.demos} Best-Fit demonstrations "
          f"({args.layout}, slots={args.slots}) ...")
    obs_buf, act_buf, mask_buf = collect(env, BestFitScheduler(), args.demos, args.seed)

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

    print(f"[2/2] Cloning for {args.epochs} epochs on {n} samples ...")
    print(f"  {'epoch':>6s}{'loss':>10s}{'agreement':>12s}")
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        total, correct = 0.0, 0
        for start in range(0, n, args.batch):
            batch = order[start:start + args.batch]
            batch_obs = {k: observations[k][batch] for k in keys}
            dist = policy.get_distribution(batch_obs, action_masks=masks[batch])
            loss = -dist.log_prob(actions[batch]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
            correct += int((dist.distribution.probs.argmax(1) == actions[batch]).sum())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  {epoch:6d}{total / n:10.4f}{100 * correct / n:11.1f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(BC_PATH))
    print(f"\nSaved warm start -> {BC_PATH}.zip")
    print("Fine-tune it with:  python -m src.models.train_agent --init-from bc_pretrained")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
