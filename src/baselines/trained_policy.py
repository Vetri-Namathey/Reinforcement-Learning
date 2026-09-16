"""
Wrapper for the trained RL agent to evaluate it using the same benchmark
harness as the heuristic baselines.
"""
from sb3_contrib import MaskablePPO


class TrainedPolicyScheduler:
    """Wraps a trained MaskablePPO model so it can run through the same
    run_episode() harness as the heuristic baselines — identical metric
    tracking (power, queue depth, spread, timing), zero duplicated code."""
    name = "Spatial-Phase RL"

    def __init__(self, model: MaskablePPO, name: str = "Spatial-Phase RL"):
        self.model = model
        self.name = name

    def select_action(self, obs: dict) -> int:
        action_mask = obs["action_mask"].astype(bool)   # full 37 elements — do NOT slice off WAIT
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)
