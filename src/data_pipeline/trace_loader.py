"""
Loads the real Microsoft Azure LLM Inference Trace 2023
(TIMESTAMP, ContextTokens, GeneratedTokens) from AzurePublicDataset.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from configs import AZURE_TRACE_DIR

# GPU model of the fleet the Azure 2023 trace was served on (Splitwise, ISCA 2024)
AZURE_TRACE_GPU_MODEL = "H100"

AZURE_TRACE_BASE_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
)
AZURE_TRACE_FILES = {
    "code": "AzureLLMInferenceTrace_code.csv",
    "conv": "AzureLLMInferenceTrace_conv.csv",
}


class AzureTraceLoader:
    """
    Loads the real Microsoft Azure LLM Inference Trace 2023
    (TIMESTAMP, ContextTokens, GeneratedTokens). Normalizes arrival times
    to seconds-since-trace-start and tags the DataFrame with the serving
    fleet's GPU model so downstream fusion can verify provenance.
    """

    def __init__(self, workload: str = "conv", data_dir: str | Path | None = None):
        """
        Args:
            workload: "conv" (chat) or "code" (coding copilot) trace.
            data_dir: Directory holding the trace CSVs. Defaults to
                      <project_root>/data/AzurePublicDataset.
        """
        if workload not in AZURE_TRACE_FILES:
            raise ValueError(
                f"Unknown workload '{workload}'. Choose from {list(AZURE_TRACE_FILES)}"
            )
        self.workload = workload
        self.data_dir = Path(data_dir) if data_dir is not None else AZURE_TRACE_DIR
        self.csv_path = self.data_dir / AZURE_TRACE_FILES[workload]
        self.gpu_model = AZURE_TRACE_GPU_MODEL

    def ensure_downloaded(self) -> Path:
        """Download the trace CSV from the AzurePublicDataset repo if missing."""
        if not self.csv_path.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            url = f"{AZURE_TRACE_BASE_URL}/{AZURE_TRACE_FILES[self.workload]}"
            print(f"  Downloading Azure trace: {url}")
            urllib.request.urlretrieve(url, self.csv_path)
        return self.csv_path

    def load(self, max_requests: int | None = None) -> pd.DataFrame:
        """
        Load and normalize the trace, optionally truncated to the first
        max_requests rows. Returns a DataFrame with columns
        [arrival_time, prefill_tokens, decode_tokens], tagged with
        attrs["gpu_model"] and attrs["source"].
        """
        self.ensure_downloaded()
        raw = pd.read_csv(self.csv_path)

        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Azure trace missing columns: {missing}")

        timestamps = pd.to_datetime(raw["TIMESTAMP"])
        arrival_time = (timestamps - timestamps.min()).dt.total_seconds()

        trace_df = pd.DataFrame({
            "arrival_time":   np.round(arrival_time.values, 4),
            "prefill_tokens": raw["ContextTokens"].astype(np.int64).values,
            "decode_tokens":  raw["GeneratedTokens"].astype(np.int64).values,
        }).sort_values("arrival_time", ignore_index=True)

        if max_requests is not None:
            trace_df = trace_df.head(max_requests).copy()

        trace_df.attrs["gpu_model"] = self.gpu_model
        trace_df.attrs["source"] = f"AzureLLMInferenceTrace2023/{self.workload}"
        return trace_df


if __name__ == "__main__":
    loader = AzureTraceLoader(workload="conv")
    df = loader.load(max_requests=5)
    print(f"Loaded {len(df)} rows from {loader.csv_path}")
    print(df.to_string(index=False))
