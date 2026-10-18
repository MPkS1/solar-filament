"""Offline evaluation on the held-out split, using the official scoring rule.

The leaderboard ranks submissions by **Panoptic Quality**. `magfilo.metrics.
pq_pooled` is a faithful re-implementation of the organizers'
`Self_Evaluation_Notebook`, so the number printed here is the number the
leaderboard would show -- with the caveat that it is measured on our own
held-out split, against MAGFiLO's multiple annotator views.

Every annotator view is scored separately against the same prediction (that is
what the official scorer does: GT is keyed ``"<annotator>-<image>"``), and the
TP/FP/FN counts are pooled before forming PQ.

Example:
    python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.cache import cached_prob_map  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402
from magfilo.data import rasterize_view_instances, read_image, split_stems  # noqa: E402
from magfilo.disk import disk_mask  # noqa: E402
from magfilo.metrics import pq_pooled, score_dataset  # noqa: E402
from magfilo.predict import load_model, prob_to_instances  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402
from magfilo.viz import overlay_polygons, panel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--limit", type=int, default=8, help="number of val frames to score")
    ap.add_argument("--offset", type=int, default=0, help="skip this many val frames first")
    ap.add_argument("--refresh-cache", action="store_true", help="ignore cached probability maps")
    ap.add_argument("--save-fig", default=None, help="render GT vs prediction panels here")
    ap.add_argument("--fig-frames", type=int, default=6)
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, base=args.base, depth=args.depth, device=device)

    coco = load_coco()
    _, val_stems = split_stems(coco, args.val_frac, args.seed)
    stems = val_stems[args.offset: args.offset + args.limit]
    print(f"scoring {len(stems)} held-out frames on {device} "
          f"(val split index {args.offset}..{args.offset + len(stems) - 1})")
    print(f"post-processing: threshold={args.threshold} min_area={args.min_area}")

    pred_maps: dict[str, np.ndarray] = {}
    for stem in stems:
        gray = read_image(paths.TRAIN_IMAGES / f"{stem}.jpeg")
        prob = cached_prob_map(paths.TRAIN_IMAGES / f"{stem}.jpeg", model, device,
                               tile=args.tile, refresh=args.refresh_cache, gray=gray)
        labels = prob_to_instances(prob, threshold=args.threshold, min_area=args.min_area,
                                   disk=disk_mask(gray, relative_radius=0.98))
        pred_maps[stem] = labels

    views_by_stem = {s: [rasterize_view_instances(v, 2048, 1.0).astype(np.int32)
                         for v in coco.by_stem[s]] for s in stems}

    # --- official metric: pooled PQ over every annotator-image entry ---------
    rep = pq_pooled((gt, pred_maps[stem])
                    for stem, views in views_by_stem.items() for gt in views)
    print("\n=== official scoring (organizers' Self_Evaluation_Notebook) ===")
    print(rep)

    print("\nper-frame (PQ computed over that frame's annotator views):")
    for stem in stems:
        sub = pq_pooled((gt, pred_maps[stem]) for gt in views_by_stem[stem])
        n_gt = int(np.mean([int(g.max()) for g in views_by_stem[stem]]))
        print(f"  {stem:<22} views={len(views_by_stem[stem])} n_gt={n_gt:<3} "
              f"n_pred={int(pred_maps[stem].max()):<3} PQ={sub.pq:.3f} "
              f"sem={sub.semantic_dice_mean:.3f} "
              f"TP/FP/FN={sub.tp}/{sub.fp}/{sub.fn}")

    # --- secondary diagnostics (older evaluator) ----------------------------
    densest = {s: max(views_by_stem[s], key=lambda g: int(g.max())) for s in stems}
    diag = score_dataset(densest, pred_maps, reinstantiate=False)
    print("\n=== diagnostic: instance_wise_dice vs densest view (not the leaderboard) ===")
    print("  pairs", f"{diag.metrics['instance_wise_dice/pairs']:.4f}",
          " image", f"{diag.metrics['instance_wise_dice/image']:.4f}",
          " one_to_many", int(diag.metrics['one_to_many']),
          " many_to_one", int(diag.metrics['many_to_one']))

    if args.save_fig:
        _save_comparison(stems[: args.fig_frames], densest, pred_maps, args.save_fig)


def _labels_to_polys(labels: np.ndarray) -> list[np.ndarray]:
    polys = []
    for i in range(1, int(labels.max()) + 1):
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if c.shape[0] >= 3:
                polys.append(c.reshape(-1, 2).astype(np.float32))
    return polys


def _save_comparison(stems, gt_by_view, pred_maps, out_path: str) -> None:
    items = []
    for stem in stems:
        img = read_image(paths.TRAIN_IMAGES / f"{stem}.jpeg", scale=0.25)
        gt_polys = _labels_to_polys(gt_by_view[stem])
        pr_polys = _labels_to_polys(pred_maps[stem])
        polys = gt_polys + pr_polys
        cats = [4] * len(gt_polys) + [1] * len(pr_polys)  # magenta = GT, blue = pred
        rgb = overlay_polygons(img, polys, cats, scale=0.25, fill_alpha=0.28)
        items.append((rgb, f"{stem}\nGT {len(gt_polys)} (magenta) vs pred {len(pr_polys)} (blue)"))
    panel(items, out_path, ncols=3, suptitle="Held-out frames: ground truth vs U-Net prediction")
    print(f"\nwrote comparison -> {out_path}")


if __name__ == "__main__":
    main()
