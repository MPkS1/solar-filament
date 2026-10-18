"""Disk cache for full-resolution probability maps.

Tiled inference over a 2048x2048 frame dominates the cost of every evaluation,
so maps are cached under ``outputs/cache/probs`` (git-ignored) and reused across
runs. The post-processing sweep then costs only the cheap connected-components
step per parameter combination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import read_image
from .predict import predict_logits_tiled

DEFAULT_CACHE_DIR = Path("outputs/cache/probs")


def prob_map(model, gray: np.ndarray, device, tile: int = 512) -> np.ndarray:
    """Sigmoid of the tiled logits, float32 in [0, 1]."""
    logits = predict_logits_tiled(model, gray, device=device, tile=tile)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def cached_prob_map(
    image_path: Path,
    model,
    device,
    tile: int = 512,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """Probability map for one frame, computing and caching it on first use.

    ``image_path`` is the frame (train or test); the cache key is the file stem.
    Pass ``gray`` to avoid re-reading the image when the caller already has it.
    """
    path = Path(cache_dir) / f"{Path(image_path).stem}.npy"
    if path.exists() and not refresh:
        return np.load(path)

    if gray is None:
        gray = read_image(Path(image_path))
    prob = prob_map(model, gray, device=device, tile=tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prob)
    return prob
