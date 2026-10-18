"""Solar-disc geometry: the disc is nearly a fixed circle in every frame."""

from __future__ import annotations

import cv2
import numpy as np

from .paths import IMAGE_SIZE


def disk_circle(image: np.ndarray, relative_radius: float = 1.0) -> tuple[float, float, float]:
    """Estimate (cx, cy, r) of the solar disc for a grayscale image."""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = image.shape[:2]
        return w / 2, h / 2, 0.46 * min(h, w) * relative_radius
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return float(cx), float(cy), float(r) * relative_radius


def disk_mask_at(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    return m


def disk_mask(image: np.ndarray, relative_radius: float = 1.0) -> np.ndarray:
    cx, cy, r = disk_circle(image, relative_radius)
    return disk_mask_at(*image.shape[:2], cx, cy, r)


# Empirically measured on the train set at native resolution (EDA).
NATIVE_DISK_RADIUS = 937.0


def approximate_disk_mask(h: int = IMAGE_SIZE, w: int = IMAGE_SIZE) -> np.ndarray:
    """Fixed-circle disc mask; use when per-frame estimation is undesirable."""
    return disk_mask_at(h, w, w / 2, h / 2, NATIVE_DISK_RADIUS * (min(h, w) / IMAGE_SIZE))
