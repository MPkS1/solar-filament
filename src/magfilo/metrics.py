"""Local replica of the competition's scoring.

The leaderboard metric is `instance_wise_dice`; the published rubric scores
mean Dice (`torchmetrics DiceScore`, foreground-only), Dice/IoU distributions,
and the distribution of one-to-many / many-to-one GT<->prediction relations.
We also provide Panoptic Quality, since the overview page names PQ as the
primary criterion.

Key semantics recovered from the organizers' public sibling evaluator
(`dataresearchlab/unet_filament_detection: eval_instancewise.py`):

* **CC re-instancing** -- instance masks that touch are merged and interior
  holes are filled (`findContours(RETR_EXTERNAL)` + `drawContours(FILLED)`)
  *before* scoring.
* **All-pairs matching** -- a GT instance matches every predicted instance it
  overlaps by at least one pixel (not Hungarian, not IoU>0.5).
* **Per-pair Dice** -- ``2|I| / (|GT| + |Pred|)``.
* An image with **no predicted masks emits no rows** -> it is *free* under the
  per-pair mean, but scores 0 under a per-image semantic mean.

Two aggregation families are exposed (both are decodable from the source):
``pairs`` (mean over matched pairs) and ``image`` (mean over per-image means).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# mask utilities
# --------------------------------------------------------------------------- #
def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask (uint8/bool, {0,1})."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    # flood background from every border pixel so enclosed holes stay 0
    border = np.zeros_like(ff)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    cv2.floodFill(ff, pad, (0, 0), 2)
    if ff.max() < 2:  # (0,0) was foreground; retry from a known background px
        ys, xs = np.where(m == 0)
        if len(ys):
            ff = m.copy()
            pad = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, pad, (int(xs[0]), int(ys[0])), 2)
    holes = (ff == 0)
    return (m.astype(bool) | holes)


def reinstantiate_cc(masks: np.ndarray) -> np.ndarray:
    """Merge touching masks into connected components, fill holes.

    ``masks`` is (N, H, W) bool. Returns an int32 label map (0 = background,
    1..K = instances) -- exactly the re-instancing step of the scorer.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("expected (N, H, W)")
    if masks.shape[0] == 0:
        return np.zeros(masks.shape[1:], dtype=np.int32)
    union = fill_holes(masks.any(axis=0))
    n, labels = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def label_mask_to_bool_list(labels: np.ndarray) -> np.ndarray:
    k = int(labels.max())
    return np.stack([labels == i for i in range(1, k + 1)]) if k else np.zeros((0, *labels.shape), bool)


# --------------------------------------------------------------------------- #
# pair matching
# --------------------------------------------------------------------------- #
@dataclass
class ImageMatch:
    image: str
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (gt_id, pred_id, dice)
    gt_only: list[int] = field(default_factory=list)
    pred_only: list[int] = field(default_factory=list)
    gt_touch: dict[int, int] = field(default_factory=dict)    # gt -> #preds overlapped
    pred_touch: dict[int, int] = field(default_factory=dict)  # pred -> #gts overlapped

    @property
    def dice_values(self) -> list[float]:
        return [d for _, _, d in self.pairs]

    @property
    def one_to_many(self) -> int:
        return sum(1 for v in self.gt_touch.values() if v > 1)

    @property
    def many_to_one(self) -> int:
        return sum(1 for v in self.pred_touch.values() if v > 1)


def match_image(gt_labels: np.ndarray, pred_labels: np.ndarray, name: str = "") -> ImageMatch:
    """All-pairs overlap matching between two label maps."""
    im = ImageMatch(image=name)
    gk, pk = int(gt_labels.max()), int(pred_labels.max())
    if gk == 0 or pk == 0:
        im.gt_only = list(range(1, gk + 1))
        im.pred_only = list(range(1, pk + 1))
        return im

    g_flat = gt_labels.ravel().astype(np.int64)
    p_flat = pred_labels.ravel().astype(np.int64)
    both = (g_flat > 0) & (p_flat > 0)
    # co-occurrence counts of (gt, pred) pairs over overlapping pixels
    combo = np.bincount(g_flat[both] * (pk + 1) + p_flat[both],
                        minlength=(gk + 1) * (pk + 1)).reshape(gk + 1, pk + 1)
    g_area = np.bincount(g_flat, minlength=gk + 1)
    p_area = np.bincount(p_flat, minlength=pk + 1)

    gt_hit = np.zeros(gk + 1, bool)
    pred_hit = np.zeros(pk + 1, bool)
    for gi in range(1, gk + 1):
        for pi in range(1, pk + 1):
            inter = int(combo[gi, pi])
            if inter == 0:
                continue
            dice = 2.0 * inter / (int(g_area[gi]) + int(p_area[pi]))
            im.pairs.append((gi, pi, dice))
            gt_hit[gi] = True
            pred_hit[pi] = True
    im.gt_touch = {gi: int((combo[gi, 1:] > 0).sum()) for gi in range(1, gk + 1)}
    im.pred_touch = {pi: int((combo[1:, pi] > 0).sum()) for pi in range(1, pk + 1)}
    im.gt_only = [gi for gi in range(1, gk + 1) if not gt_hit[gi]]
    im.pred_only = [pi for pi in range(1, pk + 1) if not pred_hit[pi]]
    return im


def panoptic_quality(gt_labels: np.ndarray, pred_labels: np.ndarray, iou_thr: float = 0.5) -> tuple[float, int, int, int]:
    """Standard PQ: match with IoU > thr, then |TP| / (|TP| + .5|FP| + .5|FN|)."""
    gk, pk = int(gt_labels.max()), int(pred_labels.max())
    if gk == 0 and pk == 0:
        return 1.0, 0, 0, 0
    g_flat, p_flat = gt_labels.ravel().astype(np.int64), pred_labels.ravel().astype(np.int64)
    g_area = np.bincount(g_flat, minlength=gk + 1)
    p_area = np.bincount(p_flat, minlength=pk + 1)
    both = (g_flat > 0) & (p_flat > 0)
    combo = np.bincount(g_flat[both] * (pk + 1) + p_flat[both],
                        minlength=(gk + 1) * (pk + 1)).reshape(gk + 1, pk + 1)
    tp = matched_g = matched_p = 0
    for gi in range(1, gk + 1):
        for pi in range(1, pk + 1):
            inter = int(combo[gi, pi])
            if not inter:
                continue
            iou = inter / (int(g_area[gi]) + int(p_area[pi]) - inter)
            if iou > iou_thr:
                tp += 1
                matched_g += 1
                matched_p += 1
    fp = pk - matched_p
    fn = gk - matched_g
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = 1.0 if denom == 0 else tp / denom
    return pq, tp, fp, fn


# --------------------------------------------------------------------------- #
# dataset-level aggregation
# --------------------------------------------------------------------------- #
def semantic_dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / (a.sum() + b.sum() + eps))


@dataclass
class ScoreReport:
    metrics: dict[str, float]
    per_image: list[dict] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "\n".join(f"  {k:<22} {v:.4f}" for k, v in self.metrics.items())


def score_dataset(
    gt: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    reinstantiate: bool = True,
) -> ScoreReport:
    """Score label maps keyed by image name.

    ``gt`` / ``pred`` map image name -> int32 label map (0 = background).
    Set ``reinstantiate=False`` to score the raw masks without CC merging.
    """
    names = sorted(set(gt) | set(pred))
    pair_dices: list[float] = []
    image_pair_means: list[float] = []
    sem_dices: list[float] = []
    pqs: list[float] = []
    tp = fp = fn = 0
    o2m = m2o = 0
    per_image = []

    for n in names:
        g = gt.get(n, np.zeros((1, 1), np.int32))
        p = pred.get(n, np.zeros_like(g))
        if reinstantiate:
            g = reinstantiate_cc(label_mask_to_bool_list(g))
            p = reinstantiate_cc(label_mask_to_bool_list(p))
        im = match_image(g, p, n)
        pq, t, f_p, f_n = panoptic_quality(g, p)
        tp += t; fp += f_p; fn += f_n
        o2m += im.one_to_many; m2o += im.many_to_one
        pqs.append(pq)
        sem_dices.append(semantic_dice(g > 0, p > 0))
        d = im.dice_values
        if d:
            pair_dices.extend(d)
            image_pair_means.append(float(np.mean(d)))
        per_image.append({
            "image": n, "n_gt": int(g.max()), "n_pred": int(p.max()),
            "pq": pq, "n_pairs": len(d), "mean_pair_dice": float(np.mean(d)) if d else None,
            "sem_dice": sem_dices[-1], "one_to_many": im.one_to_many,
            "many_to_one": im.many_to_one,
        })

    metrics = {
        "instance_wise_dice/pairs": float(np.mean(pair_dices)) if pair_dices else float("nan"),
        "instance_wise_dice/image": float(np.mean(image_pair_means)) if image_pair_means else float("nan"),
        "semantic_dice_mean": float(np.mean(sem_dices)) if sem_dices else float("nan"),
        "panoptic_quality": float(np.mean(pqs)) if pqs else float("nan"),
        "pq_tp": tp, "pq_fp": fp, "pq_fn": fn,
        "one_to_many": o2m, "many_to_one": m2o,
        "n_images": len(names), "n_images_with_pairs": len(image_pair_means),
    }
    return ScoreReport(metrics=metrics, per_image=per_image)
