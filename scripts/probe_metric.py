"""Probe what the leaderboard actually scores, from one checkpoint.

The organizers' overview says the scoreboard shows "the mean Dice score" while the
rubric ranks on Panoptic Quality. On this competition that wording is wrong: two
submissions with held-out semantic Dice 0.605 and 0.682 scored 0.16 and 0.28.
This script computes every plausible definition from the *same* masks and prints
the ratio between each and the leaderboard score, so the definition can be
identified rather than assumed.

Two leaderboard observations are baked in (see `--lb` to supply your own):

    0.5/40   3176 rows  -> 0.16
    0.88/550 1641 rows  -> 0.28

A candidate is right when its ratio is the *same* for both submissions: a real
metric differs from the board only by test-set difficulty, which is common to
both. Result on this checkpoint: pooled PQ has a ratio spread of 0.068, versus
0.106-0.146 for every Dice variant and 0.125 for PQ at IoU > 0.65.

Example:
    python scripts/probe_metric.py --checkpoint checkpoints/unet.pt --limit 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.cache import cached_prob_map  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402
from magfilo.data import rasterize_view_instances, read_image, split_stems  # noqa: E402
from magfilo.disk import disk_mask  # noqa: E402
from magfilo.metrics import pq_pooled, score_dataset, semantic_dice  # noqa: E402
from magfilo.predict import load_model, prob_to_instances  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402


def instances(prob, gray, threshold, min_area):
    return prob_to_instances(prob, threshold=threshold, min_area=min_area,
                             max_area_frac=0.05,
                             disk=disk_mask(gray, relative_radius=0.98))


def candidates(entries, frames):
    """All candidate scalars from one set of (gt, pred) annotator-image entries."""
    pooled = pq_pooled(entries)
    out = {
        "pq_iou50": pooled.pq,
        "pq_iou65": pq_pooled(entries, iou_thr=0.65).pq,
        "mean_frame": float(np.mean(
            [pq_pooled([(g, p) for g, p in fr]).pq for fr in frames])),
        "semantic_dice": float(np.mean([semantic_dice(g > 0, p > 0) for g, p in entries])),
    }
    gt_map = {f"v{i}-{i}": g for i, (g, _) in enumerate(entries)}
    pred_map = {f"v{i}-{i}": p for i, (_, p) in enumerate(entries)}
    m = score_dataset(gt_map, pred_map).metrics
    out["instance_dice"] = m.get("instance_wise_dice/pairs", float("nan"))
    return out, pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    # (threshold, min_area, leaderboard score) for submissions already probed
    configs = [((0.88, 550), 0.28), ((0.50, 40), 0.16)]

    device = pick_device(args.device)
    model = load_model(args.checkpoint, base=args.base, depth=args.depth, device=device)
    coco = load_coco()
    _, val_stems = split_stems(coco, args.val_frac, args.seed)
    stems = val_stems[:args.limit]

    per_config = []
    for (th, ma), lb in configs:
        entries, frames = [], []
        for stem in stems:
            img = paths.TRAIN_IMAGES / f"{stem}.jpeg"
            gray = read_image(img)
            prob = cached_prob_map(img, model, device, tile=args.tile, gray=gray,
                                   refresh=args.refresh_cache,
                                   preprocess=args.preprocess, tta=args.tta)
            pred = instances(prob, gray, th, ma)
            gts = [rasterize_view_instances(v, 2048, 1.0).astype(np.int32)
                   for v in coco.by_stem[stem]]
            frames.append([(g, pred) for g in gts])
            entries.extend((g, pred) for g in gts)
        vals, pooled = candidates(entries, frames)
        per_config.append((f"{th}/{ma}", lb, vals))
        print(f"{th}/{ma}: PQ {pooled.pq:.4f}  TP/FP/FN {pooled.tp}/{pooled.fp}/{pooled.fn}")

    names = list(per_config[0][2])
    print()
    print(f"{'candidate':<16} {per_config[0][0]:>9} {'LB':>6} {'ratio':>7} "
          f"{per_config[1][0]:>9} {'LB':>6} {'ratio':>7} {'spread':>7}")
    print("-" * 74)
    ranked = []
    for k in names:
        a, b = per_config[0][2][k], per_config[1][2][k]
        ra = per_config[0][1] / a if a else float("nan")
        rb = per_config[1][1] / b if b else float("nan")
        ranked.append((abs(ra - rb), k, a, b, ra, rb))
    for spread, k, a, b, ra, rb in sorted(ranked):
        print(f"{k:<16} {a:>9.4f} {per_config[0][1]:>6.2f} {ra:>7.3f} "
              f"{b:>9.4f} {per_config[1][1]:>6.2f} {rb:>7.3f} {spread:>7.3f}")
    print()
    print("Smallest spread wins: that candidate differs from the board only by")
    print("test-set difficulty, which is common to both submissions.")


if __name__ == "__main__":
    main()
