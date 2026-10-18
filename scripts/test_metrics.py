"""Analytic checks for magfilo.metrics.

Runnable directly (`python scripts/test_metrics.py`) or via pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import metrics as M  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  <- ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def labelize(spec: dict[int, tuple[slice, slice]], shape=(64, 64)) -> np.ndarray:
    lab = np.zeros(shape, np.int32)
    for k, (ys, xs) in spec.items():
        lab[ys, xs] = k
    return lab


# --------------------------------------------------------------------------- #
def test_semantic_dice() -> None:
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    check("semantic dice identical = 1", abs(M.semantic_dice(a, a) - 1) < 1e-6)
    b = np.zeros((10, 10), bool); b[6:9, 6:9] = True
    check("semantic dice disjoint = 0", M.semantic_dice(a, b) < 1e-6)
    c = np.zeros((10, 10), bool); c[2:6, 2:4] = True  # half of a
    check("semantic dice half = 2/3", abs(M.semantic_dice(a, c) - 2 / 3) < 1e-6,
          f"got {M.semantic_dice(a, c)}")


def test_reinstantiate() -> None:
    # two squares sharing an edge -> merge into one CC; diagonal touch -> merge (8-conn)
    m = np.zeros((3, 20, 20), bool)
    m[0, 2:8, 2:8] = True
    m[1, 2:8, 8:12] = True   # touches m0 along column 7/8
    m[2, 12:16, 12:16] = True
    lab = M.reinstantiate_cc(m)
    check("touching masks merge to one CC", int(lab.max()) == 2, f"got {lab.max()}")

    # hole filling
    ring = np.zeros((1, 20, 20), bool)
    ring[0, 2:16, 2:16] = True
    ring[0, 6:12, 6:12] = False
    lab2 = M.reinstantiate_cc(ring)
    filled = (lab2 > 0).sum()
    check("interior hole filled", filled == 14 * 14, f"got {filled}")

    # empty input
    check("empty masks -> empty labels", M.reinstantiate_cc(np.zeros((0, 5, 5), bool)).max() == 0)


def test_match_and_pq() -> None:
    gt = labelize({1: (slice(10, 20), slice(10, 20))})
    pred = labelize({1: (slice(10, 20), slice(10, 20))})
    im = M.match_image(gt, pred, "x")
    check("identical instance -> 1 pair dice 1.0", len(im.pairs) == 1 and abs(im.pairs[0][2] - 1) < 1e-6)
    pq, tp, fp, fn = M.panoptic_quality(gt, pred)
    check("PQ identical = 1.0", abs(pq - 1) < 1e-6 and (tp, fp, fn) == (1, 0, 0), f"{pq} {tp} {fp} {fn}")

    # one GT wide rect, two preds inside -> one-to-many and many-to-one bookkeeping
    gt2 = labelize({1: (slice(10, 20), slice(10, 30))})
    pred2 = labelize({1: (slice(10, 20), slice(10, 18)), 2: (slice(10, 20), slice(22, 30))})
    im2 = M.match_image(gt2, pred2, "y")
    check("one GT matched by two preds", im2.one_to_many == 1 and len(im2.pairs) == 2)
    check("preds each touch one GT", im2.many_to_one == 0)

    # two GTs, one merged pred -> many-to-one
    gt3 = labelize({1: (slice(10, 20), slice(10, 18)), 2: (slice(10, 20), slice(22, 30))})
    pred3 = labelize({1: (slice(10, 20), slice(10, 30))})
    im3 = M.match_image(gt3, pred3, "z")
    check("one pred matched by two GTs", im3.many_to_one == 1 and len(im3.pairs) == 2)

    # IoU<=0.5 not a PQ true positive
    gt4 = labelize({1: (slice(0, 10), slice(0, 10))})
    pred4 = labelize({1: (slice(0, 10), slice(8, 18))})  # 20px overlap / 180 union ~0.11
    pq4, tp4, fp4, fn4 = M.panoptic_quality(gt4, pred4)
    check("low-IoU pair is FP+FN", (tp4, fp4, fn4) == (0, 1, 1), f"{tp4} {fp4} {fn4}")
    check("PQ low-IoU = 0", abs(pq4) < 1e-6)


def test_dataset_aggregation() -> None:
    # image A: perfect; image B: empty prediction (free under 'pairs', zero semantic)
    gt = {
        "A": labelize({1: (slice(10, 20), slice(10, 20)), 2: (slice(30, 40), slice(30, 40))}),
        "B": labelize({1: (slice(5, 15), slice(5, 15))}),
    }
    pred = {
        "A": labelize({1: (slice(10, 20), slice(10, 20)), 2: (slice(30, 40), slice(30, 40))}),
        "B": np.zeros((64, 64), np.int32),
    }
    rep = M.score_dataset(gt, pred)
    m = rep.metrics
    check("pairs-mean unaffected by empty image", abs(m["instance_wise_dice/pairs"] - 1.0) < 1e-6,
          f"got {m['instance_wise_dice/pairs']}")
    check("image-mean unaffected by empty image", abs(m["instance_wise_dice/image"] - 1.0) < 1e-6)
    check("semantic mean = 0.5 (empty image scores 0)", abs(m["semantic_dice_mean"] - 0.5) < 1e-6,
          f"got {m['semantic_dice_mean']}")
    check("n_images_with_pairs counts only A", m["n_images_with_pairs"] == 1)


def test_inter_annotator_reproduction() -> None:
    """The metric's semantic Dice should reproduce the EDA agreement figure."""
    from magfilo.coco import load_coco
    from magfilo.rle import semantic_union
    coco = load_coco()
    dices = []
    for stem, views in coco.by_stem.items():
        if len(views) < 2:
            continue
        unions = [semantic_union(v, v.height, v.width, scale=0.25).astype(bool) for v in views]
        for i in range(len(unions)):
            for j in range(i + 1, len(unions)):
                dices.append(M.semantic_dice(unions[i], unions[j]))
    mean = float(np.mean(dices))
    check(f"inter-annotator semantic Dice ~0.644 (got {mean:.3f}, n={len(dices)})",
          abs(mean - 0.644) < 0.02)


if __name__ == "__main__":
    test_semantic_dice()
    test_reinstantiate()
    test_match_and_pq()
    test_dataset_aggregation()
    test_inter_annotator_reproduction()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all metric checks passed")
