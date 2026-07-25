"""
RL architectures and training: SpatialCNN feature extractor + MaskablePPO pipeline.

train_agent's train/evaluate/make_env are intentionally not re-exported here:
train_agent.py doubles as the `python -m src.models.train_agent` CLI entry
point, and eagerly importing it from this __init__ causes Python to import
it twice (once via the package, once as __main__), which triggers a
RuntimeWarning. Import them directly: `from src.models.train_agent import train`.
"""

from src.models.spatial_cnn import SpatialCNN

__all__ = ["SpatialCNN"]
