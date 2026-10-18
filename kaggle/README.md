# Kaggle packaging

Training is intended to run on Kaggle (this dev machine is CPU-only). Two ways
to get the code onto Kaggle:

## Option A - Kaggle Dataset (recommended)
1. Zip this repo's `src/` (and `scripts/`) and upload it as a private Kaggle
   dataset, e.g. `solar-filament-toolkit`.
2. Attach it to your notebook (path appears under `/kaggle/input/...`).
3. Add the code to `sys.path` and run.

## Option B - Git clone
Push this repo and `git clone` it inside the notebook.

## Data
Add the competition data (`Add Input -> Competitions -> Solar Filament
Segmentation Challenge 2026`). It mounts at
`/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`.

The code picks it up automatically; otherwise set the env var:

```python
import os
os.environ["MAGFILO_ROOT"] = "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
```

## Run
```bash
python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 --workers 2
python scripts/infer.py --checkpoint checkpoints/unet.pt --tile 512 \
    --out outputs/submissions/submission.csv
python scripts/validate_submission.py outputs/submissions/submission.csv --visualize 6
```

`kaggle/solar_filament_unet.ipynb` wires these steps together in notebook form.
