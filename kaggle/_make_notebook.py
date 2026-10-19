"""Generate kaggle/solar_filament_unet.ipynb (keeps the JSON valid)."""

import json
from pathlib import Path

REPO_URL = "https://github.com/MPkS1/solar-filament.git"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().splitlines(keepends=True)})


md("""
# Solar Filament Segmentation 2026 - U-Net baseline

Instance segmentation of solar filaments in 2048x2048 GONG H-alpha frames
(MAGFiLO 1.0). Trains the pure-PyTorch U-Net from the
[solar-filament](https://github.com/MPkS1/solar-filament) toolkit and writes a
submission CSV (pycocotools RLE counts).

**Before running, set these in the right-hand panel:**

1. **Add Input -> Competitions -> Solar Filament Segmentation Challenge 2026**
   (mounts read-only at `/kaggle/input/competitions/filament-segmentation-2026`)
2. **Accelerator: GPU** (T4 / P100)
3. **Internet: On** (needed to `git clone` the toolkit)

Note: `/kaggle/input` is read-only and holds the *data*. The code is cloned into
`/kaggle/working/`, which is the only writable location (and is what gets saved
as notebook output).
""")

code("""
!pip install -q pycocotools
""")

md("## 1. Get the toolkit into `/kaggle/working`")

code(f"""
import subprocess
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_DIR = Path("/kaggle/working/solar-filament") if Path("/kaggle/working").exists() \\
    else Path("solar-filament")

if not (REPO_DIR / "src" / "magfilo").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

print("repo:", REPO_DIR.resolve(), "| exists:", REPO_DIR.exists())
print("module:", sorted(p.name for p in (REPO_DIR / "src" / "magfilo").glob("*.py")))
""")

md("""
## 2. Point at the competition data and make the package importable

`MAGFILO_ROOT` must be the directory that *contains* `train/` and `test/`, i.e.
`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`.
We locate it with a glob rather than hard-coding the path (the mount prefix can
vary between `/kaggle/input/<slug>` and `/kaggle/input/competitions/<slug>`).
""")

code("""
import os, sys, glob
from pathlib import Path

REPO_DIR = Path("/kaggle/working/solar-filament") if Path("/kaggle/working").exists() \\
    else Path("solar-filament")

# --- competition data (read-only mount) -----------------------------------
cands = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)
assert cands, ("Competition data not found. Add Input -> Competitions -> "
               "Solar Filament Segmentation Challenge 2026")
os.environ["MAGFILO_ROOT"] = cands[0]

# --- make `import magfilo` work: the *src* dir goes on sys.path ------------
sys.path.insert(0, str(REPO_DIR / "src"))

import magfilo
from magfilo.coco import load_coco

print("data root :", os.environ["MAGFILO_ROOT"])
print("magfilo   :", magfilo.__file__)
coco = load_coco()
print(f"{coco.n_physical()} physical images / {coco.n_views()} views / "
      f"{coco.n_annotations()} filaments")

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
""")

md("""
## 3. Activate the repo root

The scripts call `python scripts/...` and write into `outputs/`, so the working
directory must be the repo root (cells below inherit this via `%cd`).
""")

code("""
%cd /kaggle/working/solar-filament
!python -c "import sys; sys.path.insert(0,'src'); import magfilo; print('import OK from', magfilo.__file__)"
""")

md("""
## 4. Train

Splits are by *physical frame*, so the multi-rater views of one image never leak
across train/val. Targets are binary masks (class labels are noisy and are not
scored).

Two things changed from the first baseline:

- `--preprocess` feeds the network limb-flattened, locally contrast-normalised
  frames instead of raw grayscale. Measured on annotated frames this roughly
  doubles filament contrast and makes it uniform across the disc, which is what
  the published MAGFiLO models do.
- `--val-pq-frames 8` scores eight *whole* held-out frames with the organizers'
  own PQ every epoch and keeps the best checkpoint. The previous recipe selected
  on Dice over filament-centred crops, which never sees background and therefore
  cannot see the false-positive rate -- the reason the held-out estimate and the
  leaderboard disagreed.

~40 epochs at crop 512 is a reasonable first run on a T4.
""")

code("""
!python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 \\
    --lr 3e-4 --val-frac 0.2 --workers 4 --preprocess --val-pq-frames 8 \\
    --out checkpoints/unet.pt
""")

md("""
## 5. Offline score on the held-out split

The organizers' overview says the leaderboard **shows the mean Dice** for ~50 % of
the test images, while the rubric ranks on **Panoptic Quality**. `evaluate_val.py`
runs the organizers' own scoring rule: every annotator view is a separate entry
and TP/FP/FN are pooled before PQ is formed. The organizers describe any PQ > 0.35
as "of great value".

`tune_postprocess.py` sweeps the inference threshold and minimum filament area and
ranks the combinations by that same PQ. Note the sweep now runs in the same
`--preprocess` space the model was trained in -- previously the offline evaluation
and the submission pipeline were not scored the same way.
""")

code("""
!python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 16 --preprocess
""")

code("""
!python scripts/tune_postprocess.py --checkpoint checkpoints/unet.pt --limit 16 --preprocess
""")

md("""
## 6. Predict the test set and validate the submission

The post-processing parameters are taken from the sweep above: the highest-PQ
combination found on the held-out split, not a hard-coded guess.
""")

code("""
import json
from pathlib import Path

cfg = json.loads(Path("outputs/tune_postprocess.json").read_text())
print("tuned post-processing: threshold", cfg["threshold"], "| min_area", cfg["min_area"],
      "| held-out PQ", round(cfg["pq"], 4))
""")

code("""
!python scripts/infer.py --checkpoint checkpoints/unet.pt --tile 512 \\
    --preprocess --tta \\
    --threshold {cfg['threshold']} --min-area {cfg['min_area']} \\
    --out outputs/submissions/submission.csv
""")

code("""
!python scripts/validate_submission.py outputs/submissions/submission.csv --visualize 6
""")

md("""
## 7. Download

The submission is at `/kaggle/working/solar-filament/outputs/submissions/submission.csv`
(format: `filament_id,segmentation_rle`, 2048x2048 RLE counts).

## Notes / next steps

- **The leaderboard metric is Panoptic Quality**, pooled over annotator-image
  entries with `PQ = sum(IoU of TP) / (|TP| + 0.5|FP| + 0.5|FN)`. Predicting
  nothing scores 0 and flooding the frame with masks scores ~0, so the metric
  rewards a calibrated density rather than over- or under-detection.
- **Instance separation is not the bottleneck.** Measured on 16 held-out frames:
  only 5 one-to-many and 1 many-to-one relation between predictions and GT, so
  plain connected components are adequate. The error is *detection*: ~40 % of
  predictions are spurious and ~35 % of ground-truth filaments are missed, and
  the missed ones are the small ones (median 707 px vs 1248 px for detected).
- Ideas to improve: Mask R-CNN / YOLO-seg detector instead of semantic+CC,
  native-resolution or tiled training, multi-view consensus targets, longer
  schedules (40 epochs is a first run), and a small classifier on candidate masks
  to suppress the false positives that currently force an aggressive area floor.
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
