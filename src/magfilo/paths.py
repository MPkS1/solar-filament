"""Canonical locations for the MAGFiLO 1.0 Kaggle 2026 dataset.

Everything is derived from ``REPO_ROOT`` so the code keeps working if the
project is moved or run on Kaggle (set the ``MAGFILO_ROOT`` env var to
override).
"""

from __future__ import annotations

import os
from pathlib import Path

# E:\kaggle competetions\solar  (i.e. parents[2] of this file)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Override with MAGFILO_ROOT when the layout differs (e.g. Kaggle):
#   /kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
DATA_ROOT = Path(os.environ.get("MAGFILO_ROOT", REPO_ROOT / "MAGFiLO_1.0_Kaggle_2026"))

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
TRAIN_IMAGES = TRAIN_DIR / "train_images"
TEST_IMAGES = TEST_DIR / "test_images"
TRAIN_JSON = TRAIN_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

OUTPUTS = REPO_ROOT / "outputs"
EDA_OUT = OUTPUTS / "eda"

IMAGE_SIZE = 2048  # all images are square, fixed size


def ensure_dirs() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:  # tiny self-check
    for p in (DATA_ROOT, TRAIN_IMAGES, TEST_IMAGES, TRAIN_JSON):
        print(f"{'ok ' if p.exists() else 'MISSING'} {p}")


if __name__ == "__main__":
    main()
