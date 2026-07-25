"""Custom CNN feature extractor for ThermalGridEnv's Dict observation."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from configs import TDP_REFERENCE_WATTS


class SpatialCNN(BaseFeaturesExtractor):
    """
    Dict-observation feature extractor.

    Grid stream: five 6x6 planes stacked channel-wise -> (B, 5, 6, 6) ->
    CNN -> linear head. Queue stream: queue_stats (2,) -> linear layer.
    Output: concat of both streams.

    Channels:
        0 temperatures    T / 200                    thermal state
        1 power_draw      P / TDP_REFERENCE_WATTS    absolute watts
        2 occupancy       busy_seconds / 60          availability
        3 tdp_normalized  TDP_cell / TDP_REFERENCE   per-cell capability
        4 utilization     P / TDP_cell               localized saturation

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
    )

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
        self.queue_scale = torch.tensor([100.0, 60.0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, *grid_shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, cnn_features_dim),
            nn.ReLU(),
        )

        self.queue_net = nn.Sequential(
            nn.Linear(observation_space["queue_stats"].shape[0],
                      queue_features_dim),
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
            ],
            dim=1,
        )

        grid_features = self.cnn_head(self.cnn(grid))

        queue = observations["queue_stats"] / self.queue_scale.to(
            observations["queue_stats"].device
        )
        queue_features = self.queue_net(queue)

        return torch.cat([grid_features, queue_features], dim=1)
