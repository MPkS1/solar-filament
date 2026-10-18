"""Plotting helpers for MAGFiLO EDA and (later) prediction figures."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# BGR-ish distinct colours (rendered via matplotlib RGB below, so pick RGB).
CATEGORY_COLORS = {
    1: (0.20, 0.70, 1.00),   # Left       - blue
    2: (1.00, 0.45, 0.20),   # Right      - orange
    3: (0.55, 0.85, 0.25),   # Unident.   - green
    4: (0.90, 0.40, 0.85),   # Ambiguous  - magenta
}
DEFAULT_COLOR = (1.0, 0.85, 0.20)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Grayscale uint8 -> float RGB in [0, 1] with light contrast stretch."""
    if image.ndim == 3:
        img = image[..., ::-1] if image.shape[2] == 3 else image
        return np.clip(img.astype(np.float32) / 255.0, 0, 1)
    x = image.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat(x[..., None], 3, axis=2)


def overlay_polygons(
    image: np.ndarray,
    polygons: Sequence[np.ndarray],
    categories: Sequence[int] | None = None,
    scale: float = 1.0,
    fill_alpha: float = 0.35,
    line_thickness: int = 2,
) -> np.ndarray:
    """Return an RGB float image with polygons filled + outlined."""
    rgb = _to_rgb(image).copy()
    if not len(polygons):
        return rgb
    cats = list(categories) if categories is not None else [0] * len(polygons)
    for poly, cat in zip(polygons, cats):
        pts = np.round(np.asarray(poly, dtype=np.float64) * scale).astype(np.int32)
        if pts.shape[0] < 3:
            continue
        colour = np.array(CATEGORY_COLORS.get(int(cat), DEFAULT_COLOR))
        layer = np.zeros_like(rgb)
        cv2.fillPoly(layer, [pts], colour.tolist())
        sel = layer.any(axis=2)
        rgb[sel] = (1 - fill_alpha) * rgb[sel] + fill_alpha * layer[sel]
        cv2.polylines(
            rgb, [pts], isClosed=True,
            color=tuple(float(c) for c in colour), thickness=line_thickness,
        )
    return np.clip(rgb, 0, 1)


def panel(
    items: Sequence[tuple[np.ndarray, str]],
    path: str,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> None:
    """Save a grid of (RGB image, title) tiles."""
    n = len(items)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, title) in zip(axes.ravel(), items):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_fig(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
