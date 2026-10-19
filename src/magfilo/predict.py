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


def _d4_transforms() -> list[tuple]:
    """The 8 dihedral transforms, each as a (forward, inverse) pair.

    Solar images carry no canonical orientation for a CNN: the disc is a circle
    and filaments are elongated structures, so mirroring/rotating a tile and
    un-mirroring the logits is a free source of extra evidence.
    """
    out = []
    for k in range(4):
        for flip in (False, True):
            def fwd(a, k=k, flip=flip):
                return np.rot90(np.fliplr(a) if flip else a, k)

            def inv(b, k=k, flip=flip):
                a = np.rot90(b, -k)
                return np.fliplr(a) if flip else a

            out.append((fwd, inv))
    return out


def _tile_input(patch: np.ndarray, preprocess: bool) -> np.ndarray:
    """Tiny helper so the training-free path stays byte-identical to before."""
    if preprocess:
        return patch.astype(np.float32)
    p = patch.astype(np.float32) / 255.0
    return (p - IMAGENET_LIKE[0]) / IMAGENET_LIKE[1]


@torch.no_grad()
def predict_logits_tiled(
    model: UNet,
    gray: np.ndarray,
    device: str = "cpu",
    tile: int = 512,
    stride: int | None = None,
    batch_tiles: int = 8,
    preprocess: bool = False,
    tta: bool = False,
) -> np.ndarray:
    """Blend tiled logits over a full-resolution frame.

    ``gray`` is raw uint8 when ``preprocess=False`` (the model then applies the
    legacy ``IMAGENET_LIKE`` normalisation) and an already-standardised float32
    frame when ``preprocess=True``.
    """
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
    variants = _d4_transforms() if tta else None
    step = max(1, batch_tiles // len(variants)) if variants else batch_tiles

    for i in range(0, len(coords), step):
        chunk = coords[i:i + step]
        batch, meta = [], []
        for ci, (y, x) in enumerate(chunk):
            xin = _tile_input(gray[y:y + tile, x:x + tile], preprocess)
            if variants is None:
                batch.append(xin[None])
                meta.append((ci, None))
            else:
                for fwd, inv in variants:
                    batch.append(np.ascontiguousarray(fwd(xin))[None])
                    meta.append((ci, inv))
        t = torch.from_numpy(np.stack(batch)).to(device)
        logits = model(t).cpu().numpy()[:, 0]

        gathered: dict[int, list[np.ndarray]] = {}
        for (ci, inv), lg in zip(meta, logits):
            gathered.setdefault(ci, []).append(lg if inv is None else inv(lg))
        for ci, (y, x) in enumerate(chunk):
            blended = gathered[ci][0] if len(gathered[ci]) == 1 \
                else np.mean(gathered[ci], axis=0)
            acc[y:y + tile, x:x + tile] += blended.astype(np.float32) * win
            weight[y:y + tile, x:x + tile] += win
    return acc / np.maximum(weight, 1e-6)


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def prob_to_instances(
    prob: np.ndarray,
    threshold: float = 0.88,
    min_area: int = 550,
    fill: bool = True,
    disk: np.ndarray | None = None,
    max_area_frac: float | None = 0.05,
) -> np.ndarray:
    """Probability map -> int32 label map (0 background, 1..K instances).

    ``disk`` is an optional {0,1} mask restricting predictions to the solar disc.

    A flat ``min_area`` floor is used deliberately. A confidence-gated variant
    (relax the floor for components with a high peak probability) was implemented
    and measured: it rescues a few small genuine filaments but admits more false
    positives, and PQ prices both at 0.5, so it scored worse at every setting.
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
    threshold: float = 0.88,
    min_area: int = 550,
    max_area_frac: float | None = 0.05,
    relative_disk_radius: float | None = 0.98,
    limit: int | None = None,
    preprocess: bool = False,
    tta: bool = False,
) -> Path:
    model = load_model(checkpoint, base=base, depth=depth, device=device)
    files = list_test_files()
    if limit:
        files = files[:limit]
    rows: list[tuple[str, str]] = []
    for k, name in enumerate(files, 1):
        gray = read_image(TEST_IMAGES / name)
        if preprocess:
            from .preprocess import preprocess_frame
            x, _ = preprocess_frame(gray)
        else:
            x = gray
        prob = predict_logits_tiled(model, x, device=device, tile=tile,
                                    preprocess=preprocess, tta=tta)
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
