"""Custom CNN feature extractor for ThermalGridEnv's Dict observation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from configs import TDP_REFERENCE_WATTS


class SpatialCNN(BaseFeaturesExtractor):
    """
    Dict-observation feature extractor.

    Grid stream: six 6x6 planes stacked channel-wise -> (B, 6, 6, 6) ->
    CNN -> linear head. Queue stream: queue_stats (2,) concatenated with
    head_job (4,) -> linear layer. Output: concat of both streams.

    The head_job features are what let the policy size-match work to hardware:
    without them it cannot tell a 4,000-token prefill from a 50-token one, and
    no amount of grid information recovers that.

    Channels:
        0 temperatures    T / 200                    thermal state
        1 power_draw      P / TDP_REFERENCE_WATTS    absolute watts
        2 occupancy       busy_seconds / 60          availability
        3 tdp_normalized  TDP_cell / TDP_REFERENCE   per-cell capability
        4 utilization     P / TDP_cell               localized saturation
        5 slot_fill       jobs_resident / K          concurrency headroom

    Channels 1 and 4 are deliberately different normalizations of power,
    not duplicates. Absolute watts (ch 1) drive heat generation and
    conduction to neighbours; utilization (ch 4) drives this device's own
    throttle risk. Normalizing ch 1 per-cell as well would make it
    bit-identical to ch 4 and discard the absolute magnitude the Laplacian
    actually depends on.

    The action_mask key is intentionally NOT fed to the network -- masking
    is enforced in MaskablePPO's action distribution, not learned.
    """

    GRID_KEYS = (
        "temperatures",
        "power_draw",
        "occupancy",
        "tdp_normalized",
        "utilization",
        "slot_fill",
    )

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_features_dim: int = 256,
        queue_features_dim: int = 32,
    ):
        n_cells = int(np.prod(observation_space["temperatures"].shape))
        super().__init__(
            observation_space,
            features_dim=cnn_features_dim + queue_features_dim + n_cells,
        )

        grid_shape = observation_space["temperatures"].shape
        n_channels = len(self.GRID_KEYS)

        missing = [k for k in self.GRID_KEYS if k not in observation_space.spaces]
        if missing:
            raise KeyError(
                f"Observation space is missing grid channel(s) {missing}. "
                "The environment predates the heterogeneous refactor."
            )

        # Fixed scales bringing each input to ~[0, 1].
        # power_scale is the fleet-wide TDP reference rather than this grid's
        # max, so a pure-A100 grid and a mixed grid share one absolute scale
        # and transfer between layouts stays meaningful.
        self.temp_scale = float(observation_space["temperatures"].high.max())
        self.power_scale = float(TDP_REFERENCE_WATTS)
        self.occupancy_scale = 60.0
        # queue_stats is unbounded so it needs scaling; head_job arrives
        # pre-normalised from the env, hence the trailing ones.
        self.queue_scale = torch.tensor([100.0, 60.0, 1.0, 1.0, 1.0, 1.0])

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, *grid_shape)
            n_flatten = self.flatten(self.conv(dummy)).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, cnn_features_dim),
            nn.ReLU(),
        )

        # Per-cell scoring head: a 1x1 convolution collapses the 64-channel
        # feature map to one scalar per cell, giving GRID_SIZE^2 values whose
        # index k corresponds *positionally* to action k.
        #
        # Without this the only path from a cell's state to its action logit ran
        # through Flatten -> Linear(2304, 256) -> Linear(256, 64) -> Linear(64, 37),
        # so the network had to rediscover which of 36 inputs drives which of 36
        # outputs through two bottlenecks. A policy as simple as "pick the least
        # loaded cell" was therefore very hard to represent, and PPO degraded
        # below a uniform-random policy instead of learning it. Paired with
        # net_arch=dict(pi=[]) in train_agent, each cell score now has a direct
        # weight to its own logit.
        self.cell_scores = nn.Conv2d(64, 1, kernel_size=1)

        queue_input_dim = (observation_space["queue_stats"].shape[0]
                           + observation_space["head_job"].shape[0])
        self.queue_net = nn.Sequential(
            nn.Linear(queue_input_dim, queue_features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        grid = torch.stack(
            [
                observations["temperatures"] / self.temp_scale,
                observations["power_draw"] / self.power_scale,
                observations["occupancy"] / self.occupancy_scale,
                # Already in [0, 1] from the env; no rescaling needed
                observations["tdp_normalized"],
                observations["utilization"],
                # Already in [0, 1] from the env; no rescaling needed
                observations["slot_fill"],
            ],
            dim=1,
        )

        conv_map = self.conv(grid)                       # (B, 64, H, W)
        grid_features = self.cnn_head(self.flatten(conv_map))
        per_cell = self.cell_scores(conv_map).flatten(1)  # (B, H*W)

        queue = torch.cat(
            [observations["queue_stats"], observations["head_job"]], dim=1
        ) / self.queue_scale.to(observations["queue_stats"].device)
        queue_features = self.queue_net(queue)

        return torch.cat([grid_features, queue_features, per_cell], dim=1)
