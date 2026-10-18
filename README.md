# Solar Filament Segmentation Challenge 2026 — local workspace

Kaggle competition: **`filament-segmentation-2026`** (IEEE BigData Cup 2026) — instance
segmentation of solar filaments in 2048×2048 GONG H-alpha observations (MAGFiLO 1.0).

This folder holds the toolkit, EDA, training/inference code and the Kaggle packaging. Training
runs on **Kaggle notebooks** (this machine is CPU-only and is used for development).

## Dataset layout (`MAGFiLO_1.0_Kaggle_2026/`)

```
train/train_images/   707 physical images (.jpeg, 2048x2048, grayscale)
train/MAGFiLO_1.0_Annotations_kaggle2026_train.json   (46 MB COCO)
test/test_images/     180 images, no labels
```

## Submission format (from the competition's Submission File page)

```
filament_id,segmentation_rle
20150125172714Mh_1,f8uSDds...
20150125172714Mh_2,...
```

* `filament_id` = `<file stem>_<n>`, unique per row.
* `segmentation_rle` = pycocotools RLE **counts only** (no size, no quotes).
* Masks are always 2048×2048.

## Evaluation

* **Leaderboard ranking = Panoptic Quality (PQ).** The organizers replaced the original
  per-instance Dice on 2026-08-07 — a per-instance Dice forces an arbitrary decision about
  unmatched predictions — and re-scored the leaderboard on 2026-08-12. The number *displayed*
  on the scoreboard is the mean Dice (`torchmetrics.segmentation.DiceScore`); the *ranking*
  is PQ. Do not tune against `instance_wise_dice`.
* PQ is pooled over annotator-image entries, a match requires **IoU > 0.5**, and
  `PQ = sum(IoU of TP) / (|TP| + 0.5|FP| + 0.5|FN)`. Both false positives and false negatives
  cost 0.5 each, so the metric is not gameable: predicting nothing scores 0, and flooding the
  frame with masks scores ≈0. The organizers state that **PQ > 0.35 is "of great value"**.
* Public LB scores only ~50% of the 180 test images → differences below ~0.01 are noise.
* Final rubric = **70% quantitative / 30% qualitative**. Quantitative = mean Dice
  (`torchmetrics.segmentation.DiceScore`), the PQ metric itself, Dice/IoU distributions, and
  the distribution of one-to-many / many-to-one GT↔prediction relations. Qualitative =
  pipeline writeup, morphology figures, code quality. A public GitHub repo, a pinned
  `requirements.txt` and a notebook covering the whole pipeline are **required** for final
  evaluation.

## EDA highlights (see `outputs/eda/`)

| Quantity | Value |
|---|---|
| Physical train images / annotator views / filaments | 707 / 1154 / 8199 |
| Views per image | 1: 411, 2: 145, 3: 151 (296 multi-rater) |
| Filaments per view | mean 7.1, median 7, max 26 |
| Classes | Left 2535, Right 2590, Unidentifiable 3074, **Ambiguous 0** |
| Filament area (px) | p50 1228, p90 4685, p99 13703 |
| Filament pixels as fraction of frame | ≈ 0.36 % |
| BBox elongation | p50 1.66, p90 3.23 |
| Spine length (px) | p50 111, p90 285 (present for 100 % of annotations) |
| Solar disk radius | ≈ 937 px at 2048 (disk nearly fills the frame) |
| **Local contrast** (filament vs immediate ring) | median **+10.5**, filament darker in **99.8 %** of cases |
| **Inter-annotator agreement** (semantic Dice) | mean **0.644**, p10 0.474, p50 0.675, p90 0.785; 12 % of pairs < 0.5 |

### Implications

1. **Label noise bounds the score.** Two expert annotators agree at only ~0.64 Dice on the
   same frame — no model can exceed the ground truth by more than the ground truths agree
   with each other. It is not worth chasing tiny LB decimals; prefer robust, well-documented
   methods and coverage.
2. **Class labels are unreliable.** The `Ambiguous` class is absent and the same filament is
   labelled differently by different annotators (see `fig_multi_view.png`). The competition
   scores *segmentation masks only* → predict binary instance masks, ignore category.
3. **Filaments are hard for global thresholding.** They are darker than their *local*
   surroundings ~always, but only ~28 % of views show the effect against the whole disc
   (limb darkening, plage and other dark structures confound a global threshold). Local /
   learned features are needed.
4. **Extreme foreground imbalance.** Filaments are ~0.36 % of pixels → use Dice/Tversky
   style losses, not plain BCE; tile/batch sampling matters.
5. **Copy the framing.** Disk radius is nearly constant (~937 px) → mask to the disc and
   normalise per-frame; the four-character instrument suffix in the filename may be a useful
   domain signal.

## Layout

```
src/magfilo/            reusable toolkit
  paths.py              dataset locations (MAGFILO_ROOT override)
  coco.py               COCO loader -> View objects, multi-view indexing
  rle.py                polygons <-> mask, mask -> RLE counts
  disk.py               solar-disc geometry
  metrics.py            official PQ (`pq_pooled`) + legacy instance-wise Dice diagnostics
  data.py               torch Dataset, by-frame splits, rasterisation, augmentation
  model.py              pure-torch U-Net + losses
  predict.py            tiled inference -> instances -> submission CSV
  cache.py              disk cache for full-resolution probability maps
  runtime.py            device selection
scripts/
  eda.py                EDA: stats.json + figures
  test_metrics.py       analytic checks for the metric implementation
  train.py              CLI training loop
  infer.py              CLI inference -> submission.csv
  classical_baseline.py no-ML fallback submission
  evaluate_val.py       score a checkpoint on the held-out split (official PQ)
  tune_postprocess.py   sweep threshold x min_area, ranked by official PQ
  inspect_checkpoint.py print a checkpoint's args + recorded validation score
  validate_submission.py format + integrity checks, optional preview
  debug_intensity.py    sanity check for the intensity measurement
  debug_classical.py    parameter exploration for the classical baseline
kaggle/                 notebook + packaging notes
outputs/                generated (git-ignored)
```

## Pipeline

```powershell
# 1. metrics (fast, no GPU)
python scripts/test_metrics.py

# 2. smoke-test the training plumbing on CPU
python scripts/train.py --smoke

# 3. real training (Kaggle GPU)
python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 --workers 2

# 4. score on the held-out split with the OFFICIAL PQ rule
python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 16

# 4b. tune post-processing (probability maps are cached; re-runs are cheap)
python scripts/tune_postprocess.py --checkpoint checkpoints/unet.pt --limit 16

# 5. predict the test set and validate
python scripts/infer.py --checkpoint checkpoints/unet.pt --out outputs/submissions/submission.csv
python scripts/validate_submission.py outputs/submissions/submission.csv --visualize 6

# no-ML fallback (valid submission, runs on CPU in minutes)
python scripts/classical_baseline.py
```

## Metric implementation

`src/magfilo/metrics.py` contains two layers.

**`pq_pooled` — the leaderboard metric.** A faithful re-implementation of the organizers' own
`Self_Evaluation_Notebook` (`get_pq_score`), verified against it by reading the published
source. Three properties are easy to get wrong:

* **One entry per annotator-image.** GT is keyed `<annotator_id>-<image_id>`, so a frame with
  three annotators is scored three times against the same prediction — it contributes 3× its
  true positives *and* 3× its false positives. Scoring against a single "best" annotator view
  understates the penalty.
* **Pooled globally, not averaged per image.** TP/FP/FN accumulate across every entry and one
  global `sum(IoU_TP) / (|TP| + .5|FP| + .5|FN)` is formed. A per-image mean gives a
  systematically different (usually higher) number.
* **Raw masks.** The official scorer decodes RLEs and computes IoU directly — it does *not*
  merge touching masks or fill holes.

**`score_dataset` — the older per-instance Dice machinery** (all-pairs overlap matching with
optional CC re-instancing, per-pair Dice, one-to-many / many-to-one bookkeeping). Kept as a
diagnostic only; note that its `pairs` mean is *free* for unmatched GT and unmatched
predictions, which is exactly the ambiguity the organizers removed.

`python scripts/test_metrics.py` runs **26 analytic checks** — including that PQ is pooled
rather than a per-image mean, that IoU exactly 0.5 is *not* a true positive, and that a
prediction with no GT scores 0 — and reproduces the 0.644 inter-annotator agreement from the EDA.

## Results

All numbers below use the official PQ rule (pooled over annotator-image entries on a
by-frame held-out split, seed 42, so the model has never seen these frames).

### The first run

| | Value |
|---|---|
| Checkpoint | 40 epochs, crop 512, base 32, 7.77 M params |
| Training-time `val_dice` | 0.6822 — **optimistic**, measured on filament-centred crops |
| Held-out PQ, default post-processing (0.5 / 40) | 0.277 |
| Held-out PQ, tuned post-processing (0.8 / 600) | **0.427** |
| Held-out semantic Dice, tuned | 0.669 |
| Submission produced by the run | 3176 rows, 180/180 frames, 17.6 instances/frame |

### Post-processing is worth more than extra training

16 held-out frames, 22 annotator-image entries, mean GT density 6.7 filaments per view:

| threshold | min_area | instances/view | **PQ** | semantic Dice | TP/FP/FN |
|---|---|---|---|---|---|
| 0.50 | 40 (default) | 15.1 | 0.277 | 0.613 | 95 / 210 / 39 |
| 0.50 | 600 | 8.8 | 0.373 | 0.637 | 88 / 81 / 46 |
| 0.60 | 300 | 11.0 | 0.350 | 0.632 | 95 / 124 / 39 |
| 0.70 | 600 | 8.2 | 0.391 | 0.656 | 87 / 74 / 47 |
| **0.80** | **600** | 7.8 | **0.427** | 0.669 | 90 / 57 / 44 |
| 0.85 | 600 | 7.4 | 0.412 | 0.672 | 84 / 58 / 50 |
| 0.80 | 900 | 5.8 | 0.405 | 0.654 | 74 / 36 / 60 |
| 0.80 | 300 | 10.1 | 0.398 | 0.659 | 101 / 101 / 33 |

The default threshold over-detects 2.3x and that alone costs ~0.15 PQ — roughly as
much as the gap between a working baseline and a good entry. The optimum is broad and
flat (threshold 0.74–0.84, min_area 450–750 all land within ~0.01 of each other), so
the exact argmax is not meaningful on this many frames; pick from the middle of the
plateau.

### The retired metric is not usable as a guide

For the same 16 frames, the old per-instance Dice is *maximised* by predicting almost
nothing:

| threshold | min_area | instances/view | old `instance_wise_dice` | **PQ** |
|---|---|---|---|---|
| 0.90 | 2400 | 1.2 | **0.796** | 0.244 |
| 0.85 | 1800 | 2.1 | 0.764 | 0.413 |
| 0.75 | 600 | 6.1 | 0.730 | 0.632 |

The configuration that wins the old metric loses three quarters of its PQ. Tuning
against `instance_wise_dice` is actively harmful.

### Ceiling

Inter-annotator semantic Dice is **0.644** (598 pairs). The tuned model reaches 0.669
semantic Dice against individual annotator views, i.e. it already agrees with a given
annotator slightly *better* than two annotators agree with each other on the same
frame. The remaining PQ loss is dominated by label disagreement, not model capacity:
TP 90 vs FP 57 vs FN 44 means a third of ground-truth filaments are still missed.

## Baselines

| Approach | Instances/img | median area px | Notes |
|---|---|---|---|
| Ground truth | ~7 | 1228 | reference |
| Classical CV (`scripts/classical_baseline.py`) | 19.2 | 560 | 3077 rows over 160/180 frames, valid CSV, no training |
| U-Net (40 epochs) + tuned post-processing | 7.8 | 836 | held-out PQ 0.427, semantic Dice 0.669 |

The classical detector over-fires (~2.7× the GT density) and leaves an occasional
limb-arc false positive; it is a pipeline check and fallback, not a competitive entry.

## References

* Competition: <https://www.kaggle.com/competitions/filament-segmentation-2026>
* MAGFiLO dataset paper: <https://doi.org/10.1038/s41597-024-03876-y>
* PQ paper: <https://doi.org/10.1109/CVPR.2019.00963>
* Organizers' scorer (authoritative): <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>

