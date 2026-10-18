"""Generate kaggle/solar_filament_unet.ipynb (keeps the JSON valid)."""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().splitlines(keepends=True)})


md("""
# Solar Filament Segmentation 2026 - U-Net baseline

Instance segmentation of solar filaments in 2048x2048 GONG H-alpha frames
(MAGFiLO 1.0). This notebook trains the pure-PyTorch U-Net from the
`solar` toolkit and writes a submission CSV (pycocotools RLE counts).

**Before running:** add the competition data via *Add Input -> Competitions*,
and either attach the `src/` toolkit as a dataset or git-clone it (cell below).
Enable a GPU accelerator.
""")

code("""
!pip install -q pycocotools
""")

code("""
import os, sys, glob, subprocess
from pathlib import Path

# --- locate the competition data -------------------------------------------
cands = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)
if cands:
    os.environ["MAGFILO_ROOT"] = cands[0]
    print("data root:", cands[0])
else:
    print("WARNING: competition data not found - attach it as an input")

# --- make the toolkit importable -------------------------------------------
# Option A: attached as a dataset (look for the src/magfilo package)
toolkit = glob.glob("/kaggle/input/**/src/magfilo", recursive=True)
if toolkit:
    sys.path.insert(0, str(Path(toolkit[0]).parents[1]))
    print("toolkit:", toolkit[0])
elif not Path("src/magfilo").exists():
    # Option B: clone; replace with your own repo URL
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/<you>/solar-filament.git", "solar"], check=True)
    sys.path.insert(0, str(Path("solar/src").resolve()))
print("sys.path[0:2]:", sys.path[:2])

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
""")

code("""
# quick sanity check on the data / toolkit
from magfilo.coco import load_coco
coco = load_coco()
print(coco.n_physical(), "physical images /", coco.n_views(), "views /",
      coco.n_annotations(), "filaments")
""")

md("## Train\n\nFilenames and splits are by *physical frame*, so the multi-rater\nviews of one image never leak across train/val. Targets are binary masks\n(class labels are noisy and are not scored).")

code("""
!python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 \
    --lr 3e-4 --val-frac 0.2 --workers 2 --out checkpoints/unet.pt
""")

md("## Offline score on the held-out split\n\nThe local replica of the scoring\n(`instance_wise_dice`, semantic Dice, Panoptic Quality, one-to-many /\nmany-to-one relations). Remember the inter-annotator ceiling is ~0.64 Dice.")

code("""
!python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 12
""")

md("## Predict the test set and validate the submission")

code("""
!python scripts/infer.py --checkpoint checkpoints/unet.pt --tile 512 --threshold 0.5 \
    --min-area 40 --out outputs/submissions/submission.csv
""")

code("""
!python scripts/validate_submission.py outputs/submissions/submission.csv --visualize 6
""")

md("""
## Notes / next steps

- The submission must be `filament_id,segmentation_rle` (2048x2048 RLE counts).
- `instance_wise_dice` merges touching masks and fills holes before scoring;
  emitting a few extra instances is penalised (density hurts).
- Prefer coverage over abstention: the final rubric scores mean Dice per image,
  where an empty prediction scores 0.
- Improvements: Mask R-CNN / YOLO-seg detector instead of semantic+CC, native
  resolution or tiled training, D4 TTA, and multi-view consensus targets.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "solar_filament_unet.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
