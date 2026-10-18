# Kaggle packaging

Training runs on Kaggle (this dev machine is CPU-only). The notebook
`kaggle/solar_filament_unet.ipynb` does everything below automatically.

## Paths on Kaggle

`/kaggle/input` is **read-only** and holds the *data*:

| Path | What it is |
|---|---|
| `/kaggle/input/competitions/filament-segmentation-2026` | competition mount (read-only) |
| `/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026` | dataset root (`train/`, `test/`) -> `MAGFILO_ROOT` |

The mount prefix varies (`/kaggle/input/<slug>` vs
`/kaggle/input/competitions/<slug>`), so locate the data by glob:

```python
import glob, os
os.environ["MAGFILO_ROOT"] = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)[0]
```

Your **code** goes in `/kaggle/working/` (the only writable location, and what
becomes the notebook's saved output).

## Getting the toolkit in

**Option A - git clone (recommended, needs Internet: On)**

```python
import subprocess
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/MPkS1/solar-filament.git", "/kaggle/working/solar-filament"], check=True)
%cd /kaggle/working/solar-filament
```

Then the scripts run directly (`python scripts/train.py ...`) and the package is
importable via:

```python
import sys; sys.path.insert(0, "/kaggle/working/solar-filament/src")
import magfilo
```

Note the path added is the **`src`** directory itself (the one that contains the
`magfilo/` package), not the repo root.

**Option B - Kaggle Dataset**

Upload `src/` as a dataset; it mounts read-only at `/kaggle/input/<dataset-slug>/src/magfilo`.
Add the `src` directory to `sys.path` the same way. Lightweight, but you must
re-upload on every code change, and `scripts/` (used by the notebook's
`!python scripts/...` cells) is not in `src/`, so prefer Option A.

## Checklist

1. **Add Input -> Competitions -> Solar Filament Segmentation Challenge 2026**
2. **Accelerator: GPU** (T4 / P100)
3. **Internet: On** (for the clone)

## Run

```bash
python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 --workers 2
python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 12
python scripts/infer.py --checkpoint checkpoints/unet.pt --tile 512 \
    --out outputs/submissions/submission.csv
python scripts/validate_submission.py outputs/submissions/submission.csv --visualize 6
```
