"""Inference: probability maps -> instances -> submission CSV.

Path:  grayscale frame -> tiled U-Net forward -> blended logits -> threshold
-> connected components -> (optional) disc mask / area filter / hole fill
-> per-instance RLE counts -> CSV row ``<stem>_<n>,<counts>``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import IMAGENET_LIKE, list_test_files, read_image
from .disk import approximate_disk_mask, disk_mask
from .model import UNet
from .paths import TEST_IMAGES
from .rle import mask_to_rle_counts


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(checkpoint: str | Path, base: int = 32, depth: int = 4, device: str = "cpu") -> UNet:
    model = UNet(in_ch=1, base=base, depth=depth)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_logits_tiled(
    model: UNet,
    gray: np.ndarray,
    device: str = "cpu",
    tile: int = 512,
    stride: int | None = None,
    batch_tiles: int = 8,
) -> np.ndarray:
    """Blend tiled logits over a full-resolution grayscale image."""
    if stride is None:
        stride = tile // 2
    h, w = gray.shape
    acc = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != h - tile:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] != w - tile:
        xs.append(max(w - tile, 0))

    # cosine window avoids seams
    win1d = np.hanning(tile).astype(np.float32)
    win1d = np.clip(win1d, 1e-3, None)
    win = np.outer(win1d, win1d)

    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch_tiles):
        chunk = coords[i:i + batch_tiles]
        batch = []
        for (y, x) in chunk:
            patch = gray[y:y + tile, x:x + tile].astype(np.float32) / 255.0
            patch = (patch - IMAGENET_LIKE[0]) / IMAGENET_LIKE[1]
            batch.append(patch[None])
        t = torch.from_numpy(np.stack(batch)).to(device)
        logits = model(t).cpu().numpy()[:, 0]
        for (y, x), logit in zip(chunk, logits):
            acc[y:y + tile, x:x + tile] += logit * win
            weight[y:y + tile, x:x + tile] += win
    return acc / np.maximum(weight, 1e-6)


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def prob_to_instances(
    prob: np.ndarray,
    threshold: float = 0.8,
    min_area: int = 600,
    fill: bool = True,
    disk: np.ndarray | None = None,
    max_area_frac: float | None = 0.05,
) -> np.ndarray:
    """Probability map -> int32 label map (0 background, 1..K instances).

    ``disk`` is an optional {0,1} mask restricting predictions to the solar disc.
    """
    binary = (prob >= threshold).astype(np.uint8)
    if disk is not None:
        binary &= (disk > 0).astype(np.uint8)
    if fill:
        binary = _fill_holes(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(labels)
    next_id = 0
    frame = float(prob.size)
    min_area = max(1, int(round(min_area)))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area_frac is not None and area > max_area_frac * frame:
            continue
        next_id += 1
        out[labels == i] = next_id
    return out


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    from .metrics import fill_holes
    return fill_holes(binary).astype(np.uint8)


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
def labels_to_rows(stem: str, labels: np.ndarray, h: int = 2048, w: int = 2048) -> list[tuple[str, str]]:
    if labels.shape != (h, w):
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    rows = []
    for i in range(1, int(labels.max()) + 1):
        mask = (labels == i).astype(np.uint8)
        rows.append((f"{stem}_{i}", mask_to_rle_counts(mask)))
    return rows


def write_submission(rows: list[tuple[str, str]], path: str | Path = "submission.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    return path


def run_inference(
    checkpoint: str | Path,
    out_csv: str | Path = "submission.csv",
    base: int = 32,
    depth: int = 4,
    device: str = "cpu",
    tile: int = 512,
    threshold: float = 0.8,
    min_area: int = 600,
    max_area_frac: float | None = 0.05,
    relative_disk_radius: float | None = 0.98,
    limit: int | None = None,
) -> Path:
    model = load_model(checkpoint, base=base, depth=depth, device=device)
    files = list_test_files()
    if limit:
        files = files[:limit]
    rows: list[tuple[str, str]] = []
    for k, name in enumerate(files, 1):
        gray = read_image(TEST_IMAGES / name)
        prob = predict_logits_tiled(model, gray, device=device, tile=tile)
        prob = 1.0 / (1.0 + np.exp(-prob))
        disk = disk_mask(gray, relative_radius=relative_disk_radius) if relative_disk_radius else None
        labels = prob_to_instances(
            prob, threshold=threshold, min_area=min_area,
            max_area_frac=max_area_frac, disk=disk,
        )
        rows.extend(labels_to_rows(Path(name).stem, labels))
        if k % 20 == 0:
            print(f"  {k}/{len(files)} images, {len(rows)} instances so far")
    return write_submission(rows, out_csv)


def run_inference_classical(
    out_csv: str | Path = "submission_classical.csv",
    min_area: int = 400,
    sigma: float = 15.0,
    ratio: float = 0.75,
    relative_disk_radius: float = 0.88,
    max_area_frac: float = 0.008,
    limit: int | None = None,
) -> Path:
    """No-ML fallback: Gaussian background subtraction then dark-structure threshold."""
    files = list_test_files()
    if limit:
        files = files[:limit]
    rows: list[tuple[str, str]] = []
    for k, name in enumerate(files, 1):
        gray = read_image(TEST_IMAGES / name)
        labels = classical_instances(
            gray, min_area=min_area, sigma=sigma, ratio=ratio,
            relative_disk_radius=relative_disk_radius, max_area_frac=max_area_frac,
        )
        rows.extend(labels_to_rows(Path(name).stem, labels))
        if k % 20 == 0:
            print(f"  {k}/{len(files)} images, {len(rows)} instances so far")
    return write_submission(rows, out_csv)


def classical_instances(
    gray: np.ndarray,
    min_area: int = 400,
    sigma: float = 15.0,
    ratio: float = 0.75,
    relative_disk_radius: float = 0.88,
    max_area_frac: float = 0.008,
) -> np.ndarray:
    """Dark thin structures on the disc via local background subtraction.

    A Gaussian blur is the background estimate (filaments are *darker* than
    their surroundings ~99.8 % of the time -- see EDA), so ``bg - gray`` is
    positive on filaments. The disc mask is deliberately tight (0.88 of the
    Otsu radius) because the bright limb ring otherwise dominates detections.
    """
    disk = disk_mask(gray, relative_radius=relative_disk_radius)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark = bg.astype(np.float32) - gray.astype(np.float32)
    region = dark[disk > 0]
    if region.size == 0:
        return np.zeros_like(gray, np.int32)
    thr = ratio * float(np.percentile(region, 99))
    binary = ((dark > thr) & (disk > 0)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return prob_to_instances(
        binary.astype(np.float32), threshold=0.5, min_area=min_area,
        fill=True, disk=None, max_area_frac=max_area_frac,
    )
