"""Rasterisation and COCO-RLE helpers.

Two directions we need:
  * polygons -> binary mask       (training targets / visualisation / EDA)
  * mask     -> RLE counts string (competition submission format)

The submission wants *RLE counts only* (a compact string), 2048x2048, one row
per predicted filament. ``pycocotools.mask.encode`` produces exactly that after
transposing to Fortran order.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_util


def polygons_to_mask(
    polygons: Iterable[np.ndarray],
    height: int,
    width: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Rasterise polygons (x, y points) into a single uint8 mask {0,1}.

    ``scale`` multiplies coordinates (for rasterising at a reduced resolution).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if scale != 1.0:
        polys = [
            np.round(np.asarray(p, dtype=np.float64) * scale).astype(np.int32)
            for p in polygons
        ]
    else:
        polys = [np.asarray(p, dtype=np.float64).astype(np.int32) for p in polygons]
    polys = [p for p in polys if p.shape[0] >= 3]
    if polys:
        cv2.fillPoly(mask, polys, 1)
    return mask


def mask_to_rle_counts(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> COCO RLE *counts* string (no size)."""
    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(fortran)
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def rle_counts_to_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    if isinstance(rle_counts, str):
        rle_counts = rle_counts.encode("ascii")
    rle = {"size": [height, width], "counts": rle_counts}
    return mask_util.decode(rle).astype(np.uint8)


def semantic_union(view, height: int, width: int, scale: float = 1.0) -> np.ndarray:
    """Union of all filaments in a view (used for EDA / agreement)."""
    h = int(round(height * scale))
    w = int(round(width * scale))
    return polygons_to_mask(view.polygons, h, w, scale=scale)


def rle_counts_to_size(rle_counts: str) -> tuple[int, int]:
    """Return (h, w) of an RLE payload that still carries its size key.

    Only used if someone hands us a full RLE dict-serialised string; the plain
    counts we emit do not include size.
    """
    raise NotImplementedError("plain counts carry no size; pass h/w explicitly")
