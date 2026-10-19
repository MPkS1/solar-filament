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
  preprocess.py         limb-darkening flattening + local-contrast normalisation
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
  probe_metric.py       which scalar the leaderboard shows, from the same masks
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
python scripts/train.py --smoke --preprocess

# 3. real training (Kaggle GPU). --val-pq-frames selects the checkpoint on the
#    organizers' full-frame PQ instead of Dice over filament-centred crops.
python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32 --workers 4 \
    --preprocess --val-pq-frames 8

# 4. score on the held-out split with the OFFICIAL PQ rule
python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 16 --preprocess

# 4b. tune post-processing (probability maps are cached; re-runs are cheap)
python scripts/tune_postprocess.py --checkpoint checkpoints/unet.pt --limit 16 --preprocess

# 5. predict the test set and validate (8-way dihedral TTA)
python scripts/infer.py --checkpoint checkpoints/unet.pt --preprocess --tta \
    --out outputs/submissions/submission.csv
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

## Version 2: what the leaderboard gap actually was

The tuned submission scored **0.28** on the leaderboard while the held-out estimate said **0.424**.
That gap was investigated before any tuning, and three hypotheses were eliminated with evidence:

| hypothesis | evidence | verdict |
|---|---|---|
| submission encoding / orientation bug | decoded RLE is **bit-identical** to fresh local inference (IoU 1.0000); round trip is not transposed; the organizers' own `mask_util.decode` agrees; masks sit on genuinely dark pixels (mean background−image **6.23** under the masks vs **−0.46** across the disc) | eliminated |
| train/test domain shift | test is statistically identical to train: mean intensity 83.62 vs 83.58, p99 154.8 vs 155.2, same years, same six stations, all 2048² | eliminated |
| the held-out estimate was simply too good | it was measured on the same 16 frames used to tune the threshold; an independent disjoint 16 gives **0.363**, not 0.424 | **partly responsible** |

Two real defects remained, and both are fixed in v2.

### 1. The input representation threw away the physics

Every raw frame is dominated by **limb darkening** (the disc is ~2× brighter at the centre
than at the limb) and by slowly varying background. A network trained on raw grayscale has to
learn to undo both, and the published MAGFiLO models (EdgeAttNet, MORDEN) do it up front
instead. `src/magfilo/preprocess.py` now divides out the radial intensity profile, subtracts a
local background, and standardises over the disc.

Measured on 24 annotated frames:

| | raw | preprocessed |
|---|---|---|
| filament contrast (mean over radii) | −0.241 | **−0.62** |
| contrast spread across the disc (relative) | 41 % | **26 %** |
| best single global-threshold Dice | 0.0225 | **0.0730** |
| best single global-threshold PQ | 0.0107 | **0.0182** |

Contrast roughly doubles and becomes more uniform with radius, and a plain global threshold
improves 3×. The chain costs 69 ms/frame (was 942 ms before vectorising the radial profile and
estimating the smooth background at ¼ resolution).

### 2. Model selection was measuring the wrong thing

`val_dice` was computed on **filament-centred crops only**. Every crop was guaranteed to contain
a filament, so the metric could never see background and could not see the false-positive rate —
which is precisely what the leaderboard punished. `train.py --val-pq-frames N` now runs the
organizers' pooled PQ on N *whole* held-out frames each epoch and keeps the best checkpoint.

### 3. The remaining error is detection, not instance splitting

Scored against the densest annotator view on 16 held-out frames:

| | count |
|---|---|
| true positives | 71 |
| false positives | **49** (≈40 % of predictions) |
| false negatives | **41** (≈35 % of ground-truth filaments) |
| one-to-many relations | 5 |
| many-to-one relations | 1 |

Only six split/merge relations in total, so connected components are *not* the bottleneck and a
watershed/boundary head would gain little. The missed filaments are the small ones: median
**707 px** for a miss vs **1248 px** for a detection, and **36 %** of misses fall below the
`min_area = 550` floor (vs 9 % of detections). The area floor is therefore doing real damage,
but only because the small false positives outnumber the small true ones — which is a model
problem, not a post-processing one.

### Also in v2

- **8-way dihedral TTA** at inference (`--tta`), averaged in logit space per tile.
- **Cache namespacing.** Probability maps now live under `raw/`, `pre/`, `raw+tta/`, `pre+tta/`.
  An earlier run silently reused maps from a *different checkpoint* because the cache was flat;
  that is now structurally impossible.
- **`--preprocess` must match** between training and inference, and the dataset raises if the
  combination is inconsistent (preprocessing needs native resolution for the disc geometry).

## Results

All numbers below use the official PQ rule (pooled over annotator-image entries on a
by-frame held-out split, seed 42, so the model has never seen these frames).

### The submission

| | earlier run | current run |
|---|---|---|
| Post-processing | 0.5 / 40 (default) | **0.88 / 550** (swept) |
| Rows | 3176 | **1641** |
| Test frames covered | 180 / 180 | 179 / 180 |
| Instances per frame | 17.6 | **9.2** |
| Median instance area | 836 px | **1246 px** (GT 1228) |
| Held-out PQ | 0.270 | **0.424** |
| Held-out semantic Dice | 0.605 | **0.682** |

Model: 40 epochs, crop 512, base 32, 7.77 M params. The training-time `val_dice` (0.6847) is
semantic Dice on *filament-centred* crops, so it is optimistic and not comparable to these
numbers: every crop is guaranteed to contain a filament, so the model is never asked how
confident it should be on background.

One test frame (`20200310113230Ch`, March 2020 — solar minimum) receives no predictions at
all above the tuned threshold. It is not a corrupt frame; it simply has no dark region
confident enough to survive the filter.

### What the leaderboard actually measures

Two submissions were probed against the leaderboard, which settles a question that had been
open since the first run:

| submission | our held-out PQ | leaderboard | ratio |
|---|---|---|---|
| 3176 rows, 0.5 / 40 | 0.270 | **0.16** | 0.59 |
| 1641 rows, 0.88 / 550 | 0.424 | **0.28** | 0.66 |

The ordering is preserved, so the held-out PQ is a valid *ranking* signal even though its
absolute value is inflated. Five candidate definitions were then computed from the **same
masks**; only one survives:

| candidate | 0.88/550 | 0.5/40 | ratio spread |
|---|---|---|---|
| **pooled PQ (IoU > 0.5)** | 0.4235 | 0.2697 | **0.068** |
| pooled PQ at IoU > 0.65 | 0.2864 | 0.1450 | 0.126 |
| per-image mean PQ | 0.4252 | 0.2759 | 0.079 |
| semantic (union) Dice | 0.6817 | 0.6047 | 0.146 |
| instance-wise Dice (matched pairs) | 0.7316 | 0.6483 | 0.136 |

(`python scripts/probe_metric.py --checkpoint <ckpt>` reproduces this table.)

- **Not semantic Dice.** 0.68 held-out cannot map to 0.16, despite the organizers' wording
  ("the scoreboard will show ... the mean Dice score"). That wording appears to be inaccurate.
- **Not instance-wise Dice** either: matched-pair Dice is 0.73, further from the board than PQ.
- **Not a per-image mean.** Mean per-frame PQ (0.4252) equals pooled PQ (0.4235) to 0.4%, and
  no held-out frame scores zero, so no catastrophic-frame effect exists.
- **Not a stricter IoU threshold.** PQ at IoU > 0.65 reproduces both leaderboard values
  absolutely (0.286 and 0.145), but its two ratios disagree by twice as much as at IoU > 0.5,
  and 0.65 is a most unusual choice. It cannot be excluded, only disfavoured.

What remains is a uniform test-set difficulty factor of about 0.6, i.e. the test annotation
protocol is stricter than the train protocol (or the public half is a harder subset). Fitting
the two points affinely gives `LB ~= 0.78 * PQ_heldout - 0.05`, so a leaderboard 0.64 would
require a held-out PQ near **0.89**. Against the measured ceiling (matched true positives have
mean IoU 0.677, capping a *perfect* detector near PQ 0.68), that is not reachable by improving
this architecture's post-processing. It needs tighter masks *and* better detection together.

### Post-processing is worth more than extra training

From the notebook's own sweep — 16 held-out frames, 22 annotator-image entries, mean GT
density 6.7 filaments per view:

| threshold | min_area | instances/view | **PQ** | semantic Dice | TP/FP/FN |
|---|---|---|---|---|---|
| 0.50 | 40 (old default) | 15.5 | 0.270 | 0.605 | 93 / 217 / 41 |
| 0.70 | 40 | 14.6 | 0.298 | 0.629 | 97 / 197 / 37 |
| 0.84 | 40 | 13.1 | 0.328 | 0.656 | 97 / 166 / 37 |
| 0.50 | 550 | 9.2 | 0.351 | 0.629 | 86 / 96 / 48 |
| 0.80 | 900 | 6.1 | 0.388 | 0.660 | 72 / 45 / 62 |
| 0.84 | 550 | 8.0 | 0.412 | 0.676 | 88 / 67 / 46 |
| **0.88** | **550** | 7.5 | **0.424** | 0.682 | 87 / 57 / 47 |
| 0.90 | 450 | 8.0 | 0.433 | 0.685 | 92 / 63 / 42 |

Two separable effects, the second easy to miss:

* **threshold 0.5 → 0.88** at a fixed `min_area` of 550: 0.351 → 0.424;
* **`min_area` 40 → 550** at a fixed threshold of 0.84: 0.328 → 0.412. The area floor alone is
  worth ~0.08 PQ, because the smallest real filament is ~209 px (p1) and the median is 1228 px,
  so anything far below that is near-certainly a false positive.

Together they recover **+0.154 PQ with no retraining**, and the predicted instance-area
distribution lands on the ground truth (median 1246 px vs 1228 px) instead of running small.

The plateau is broad — threshold 0.84–0.92 × min_area 450–550 all land within ~0.02 — so the
argmax is not meaningful on 16 frames. An independent check on a disjoint 16 frames (frames
16–31, previous checkpoint) confirmed the direction of the effect: 0.272 for the default
against 0.354–0.373 across the tuned region.

### The retired metric is not usable as a guide

On 10 held-out frames, the old per-instance Dice (still computed by `score_dataset`) is
*maximised* by predicting almost nothing — because unmatched ground truth and unmatched
predictions are both free under it:

| threshold | min_area | instances/view | old `instance_wise_dice` | PQ (per-image, superseded script) |
|---|---|---|---|---|
| 0.90 | 2400 | 1.2 | **0.796** | 0.244 |
| 0.85 | 1800 | 2.1 | 0.764 | 0.413 |
| 0.75 | 600 | 6.1 | 0.730 | 0.632 |

The configuration that wins the old metric throws away three quarters of its PQ, which is
exactly why the organizers retired it. Tuning against `instance_wise_dice` is actively harmful.

### Ceiling

Inter-annotator semantic Dice is **0.644** (598 pairs). The tuned model reaches **0.682**
semantic Dice against individual annotator views — it agrees with a given annotator slightly
*better* than two annotators agree with each other on the same frame. The residual PQ loss is
therefore dominated by label disagreement rather than model capacity: at 0.88 / 550 the counts
are TP 87 / FP 57 / **FN 47**, i.e. about a third of ground-truth filaments are missed while
about four predictions in ten land on filaments no annotator marked. Further gains need a
stronger model or multi-view consensus targets, not more threshold tuning.

### One idea that did not work

`20200310113230Ch` (March 2020, solar minimum) receives no predictions at all, yet the model
reaches 0.99 confidence there on a ~444 px dark region — a plausible small filament silently
discarded by the 550 px floor. A two-tier rule was therefore implemented: relax the area floor
for components whose *peak* probability is very high. It scored **worse at every setting**
(0.418 versus 0.424 at 0.88 / 550, and down to 0.384 as the gate widened): it rescued a few
genuine small filaments but admitted more false positives, and PQ prices a miss and a spurious
mask identically. The flat floor was kept and the idea reverted rather than shipped as dead
code.

## Baselines

| Approach | Instances/img | median area px | Notes |
|---|---|---|---|
| Ground truth | ~7 | 1228 | reference |
| Classical CV (`scripts/classical_baseline.py`) | 19.2 | 560 | 3077 rows over 160/180 frames, valid CSV, no training |
| U-Net (40 epochs) + swept post-processing | 9.2 | 1246 | held-out PQ 0.424, semantic Dice 0.682 |

The classical detector over-fires (~2.7× the GT density) and leaves an occasional
limb-arc false positive; it is a pipeline check and fallback, not a competitive entry.

## References

* Competition: <https://www.kaggle.com/competitions/filament-segmentation-2026>
* MAGFiLO dataset paper: <https://doi.org/10.1038/s41597-024-03876-y>
* PQ paper: <https://doi.org/10.1109/CVPR.2019.00963>
* Organizers' scorer (authoritative): <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>

