"""Disk cache for full-resolution probability maps.

Tiled inference over a 2048x2048 frame dominates the cost of every evaluation,
so maps are cached under ``outputs/cache/probs`` and reused across runs. The
post-processing sweep then costs only the cheap connected-components step per
parameter combination.

Maps are namespaced by *how they were produced* (raw vs preprocessed input, with
or without TTA). Sharing one flat directory across checkpoints and input modes
silently mixed incompatible maps in an earlier run, so the variant is part of the
path now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import read_image
from .predict import predict_logits_tiled

DEFAULT_CACHE_DIR = Path("outputs/cache/probs")


def variant_tag(preprocess: bool = False, tta: bool = False) -> str:
    """Directory-safe name for an inference configuration."""
    return f"{'pre' if preprocess else 'raw'}{'+tta' if tta else ''}"


def prob_map(model, gray: np.ndarray, device, tile: int = 512,
             preprocess: bool = False, tta: bool = False) -> np.ndarray:
    """Sigmoid of the tiled logits, float32 in [0, 1]."""
    logits = predict_logits_tiled(model, gray, device=device, tile=tile,
                                  preprocess=preprocess, tta=tta)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def prepare_input(gray: np.ndarray, preprocess: bool) -> np.ndarray:
    """Frame -> model input, matching ``predict_logits_tiled``."""
    if not preprocess:
        return gray
    from .preprocess import preprocess_frame
    x, _ = preprocess_frame(gray)
    return x


def cached_prob_map(
    image_path: Path,
    model,
    device,
    tile: int = 512,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    gray: np.ndarray | None = None,
    preprocess: bool = False,
    tta: bool = False,
) -> np.ndarray:
    """Probability map for one frame, computing and caching it on first use.

    ``image_path`` is the frame (train or test); the cache key is the file stem
    plus the inference variant. Pass ``gray`` to avoid re-reading the image when
    the caller already has it.
    """
    path = Path(cache_dir) / variant_tag(preprocess, tta) / f"{Path(image_path).stem}.npy"
    if path.exists() and not refresh:
        return np.load(path)

    if gray is None:
        gray = read_image(Path(image_path))
    prob = prob_map(model, prepare_input(gray, preprocess), device=device,
                    tile=tile, preprocess=preprocess, tta=tta)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prob)
    return prob
