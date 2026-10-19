"""PyTorch data pipeline for MAGFiLO (semantic masks + CC instance separation).

Design notes:
  * Splits are **by physical file stem**, never by annotation view -- otherwise
    the two annotator passes of the same frame leak across train/val.
  * Targets are binary semantic masks (class labels are ignored: they are not
    scored and are noisy across annotators).
  * Training crops are biased toward filament bounding boxes to fight the
    ~0.36 % foreground imbalance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco import CocoData, View, load_coco
from .paths import IMAGE_SIZE, TEST_IMAGES, TRAIN_IMAGES

IMAGENET_LIKE = (0.5, 0.25)  # (mean, std) after x/255 -- neutral per-frame choice


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def split_stems(coco: CocoData, val_frac: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    stems = sorted(coco.by_stem)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    return sorted(stems[n_val:]), sorted(stems[:n_val])


def views_for_stems(coco: CocoData, stems: list[str]) -> list[View]:
    out: list[View] = []
    for s in stems:
        out.extend(coco.by_stem[s])
    return out


# --------------------------------------------------------------------------- #
# rasterisation
# --------------------------------------------------------------------------- #
def rasterize_view(view: View, size: int = IMAGE_SIZE, scale: float = 1.0) -> np.ndarray:
    """Binary union mask for a view at ``size * scale`` resolution."""
    h = w = int(round(size * scale))
    mask = np.zeros((h, w), np.uint8)
    if scale != 1.0:
        polys = [np.round(p * scale).astype(np.int32) for p in view.polygons]
    else:
        polys = [np.asarray(p, dtype=np.int32) for p in view.polygons]
    polys = [p for p in polys if p.shape[0] >= 3]
    if polys:
        cv2.fillPoly(mask, polys, 1)
    return mask


def rasterize_view_instances(view: View, size: int = IMAGE_SIZE, scale: float = 1.0) -> np.ndarray:
    """Per-polygon instance label map (0 = background, i = polygon i)."""
    h = w = int(round(size * scale))
    labels = np.zeros((h, w), np.int32)
    for i, p in enumerate(view.polygons, start=1):
        pts = np.round(np.asarray(p, dtype=np.float64) * scale).astype(np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(labels, [pts], i)
    return labels


def read_image(path: Path, scale: float = 1.0) -> np.ndarray | None:
    if scale == 1.0:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if scale == 0.5:
        return cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if scale == 0.25:
        return cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    image: np.ndarray  # (C, h, w) float32
    mask: np.ndarray   # (1, h, w) float32
    name: str


class FilamentDataset(Dataset):
    def __init__(
        self,
        views: list[View],
        img_dir: Path = TRAIN_IMAGES,
        crop: int = 512,
        train: bool = True,
        scale: float = 1.0,
        pos_crop_prob: float = 0.6,
        limit: int | None = None,
        preprocess: bool = False,
        sigma: float = 25.0,
    ):
        self.views = views[:limit] if limit else views
        self.img_dir = img_dir
        self.crop = crop
        self.train = train
        self.scale = scale
        self.pos_crop_prob = pos_crop_prob
        self.limit = limit
        self.preprocess = preprocess
        self.sigma = sigma
        self._size = int(round(IMAGE_SIZE * scale))

    def __len__(self) -> int:
        return len(self.views)

    # -- helpers ---------------------------------------------------------
    def _positive_center(self, view: View, idx: int = 0) -> tuple[int, int] | None:
        if not view.polygons:
            return None
        poly = (random.choice(view.polygons) if self.train
                else view.polygons[idx % len(view.polygons)]) * self.scale
        if poly.shape[0] < 1:
            return None
        x, y = poly[:, 0].mean(), poly[:, 1].mean()
        return int(y), int(x)

    def _crop_origin(self, cy: int, cx: int) -> tuple[int, int]:
        size = self._size
        half = self.crop // 2
        jitter = self.crop // 4 if self.train else 0
        cy += random.randint(-jitter, jitter) if jitter else 0
        cx += random.randint(-jitter, jitter) if jitter else 0
        y0 = min(max(cy - half, 0), size - self.crop)
        x0 = min(max(cx - half, 0), size - self.crop)
        return y0, x0

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        return img, mask

    # -- main ------------------------------------------------------------
    def __getitem__(self, idx: int) -> Crop:
        view = self.views[idx]
        if self.preprocess:
            if self.scale != 1.0:
                raise ValueError("preprocess=True requires scale=1.0 (the disc geometry is "
                                 "estimated at native resolution)")
            from .preprocess import preprocess_frame
            gray = read_image(self.img_dir / view.file_name, 1.0)
            if gray is None:
                gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
            img, _ = preprocess_frame(gray, sigma=self.sigma)
        else:
            img = read_image(self.img_dir / view.file_name, self.scale)
            if img is None:
                img = np.zeros((self._size, self._size), np.uint8)
        mask = rasterize_view(view, IMAGE_SIZE, self.scale)

        if self.train and random.random() < self.pos_crop_prob:
            c = self._positive_center(view)
        elif not self.train and self.pos_crop_prob > 0:
            c = self._positive_center(view, idx)  # deterministic, filament-centred
        else:
            c = None
        if c is None:
            cy, cx = random.randint(0, self._size - 1), random.randint(0, self._size - 1)
        else:
            cy, cx = c
        y0, x0 = self._crop_origin(cy, cx)
        img = img[y0:y0 + self.crop, x0:x0 + self.crop]
        mask = mask[y0:y0 + self.crop, x0:x0 + self.crop]

        if self.train:
            img, mask = self._augment(img, mask)

        img = np.ascontiguousarray(img)
        if self.preprocess:
            x = img.astype(np.float32)          # already standardised over the disc
        else:
            x = img.astype(np.float32) / 255.0
            x = (x - IMAGENET_LIKE[0]) / IMAGENET_LIKE[1]
        return Crop(
            image=x[None].copy(),
            mask=mask.astype(np.float32)[None].copy(),
            name=view.file_name,
        )


def collate(batch: list[Crop]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.from_numpy(np.stack([b.image for b in batch])),
        "mask": torch.from_numpy(np.stack([b.mask for b in batch])),
        "name": [b.name for b in batch],
    }


class TestDataset(Dataset):
    """Whole test images at a reduced scale (for tiled / resized inference)."""

    def __init__(self, file_names: list[str], img_dir: Path = TEST_IMAGES, scale: float = 1.0,
                 preprocess: bool = False, sigma: float = 25.0):
        self.file_names = file_names
        self.img_dir = img_dir
        self.scale = scale
        self.preprocess = preprocess
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        name = self.file_names[idx]
        img = read_image(self.img_dir / name, self.scale)
        if img is None:
            img = np.zeros((int(IMAGE_SIZE * self.scale),) * 2, np.uint8)
        if self.preprocess:
            from .preprocess import preprocess_frame
            x, _ = preprocess_frame(img, sigma=self.sigma)
        else:
            x = img.astype(np.float32) / 255.0
            x = (x - IMAGENET_LIKE[0]) / IMAGENET_LIKE[1]
        return {"image": torch.from_numpy(np.ascontiguousarray(x)[None]), "file_name": name,
                "shape": img.shape}


def list_test_files() -> list[str]:
    return sorted(p.name for p in TEST_IMAGES.glob("*.jpeg"))


def build_train_val(crop: int = 512, val_frac: float = 0.2, seed: int = 42, limit: int | None = None,
                    preprocess: bool = False):
    coco = load_coco()
    train_stems, val_stems = split_stems(coco, val_frac, seed)
    return (
        FilamentDataset(views_for_stems(coco, train_stems), crop=crop, train=True, limit=limit,
                        preprocess=preprocess),
        FilamentDataset(views_for_stems(coco, val_stems), crop=crop, train=False, limit=limit,
                        preprocess=preprocess),
        coco,
    )
