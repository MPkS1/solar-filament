"""Sweep inference post-processing (threshold x min_area) on the held-out split.

The probability maps are computed **once per frame** and then reused for every
parameter combination, so the sweep is cheap. Ranks combinations by the local
replica of instance-wise Dice and by Panoptic Quality.

Example:
    python scripts/tune_postprocess.py --checkpoint solar-filament/checkpoints/unet.pt --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402
from magfilo.data import rasterize_view_instances, read_image, split_stems  # noqa: E402
from magfilo.disk import disk_mask  # noqa: E402
from magfilo.metrics import score_dataset  # noqa: E402
from magfilo.predict import load_model, predict_logits_tiled, prob_to_instances  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="solar-filament/checkpoints/unet.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=10, help="held-out frames to use")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--min-areas", type=int, nargs="+", default=[40, 120, 300, 600])
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, base=args.base, depth=args.depth, device=device)

    coco = load_coco()
    _, val_stems = split_stems(coco, args.val_frac, args.seed)
    stems = val_stems[: args.limit]
    print(f"precomputing probability maps for {len(stems)} held-out frames on {device} ...")

    probs: dict[str, np.ndarray] = {}
    disks: dict[str, np.ndarray] = {}
    gts: dict[str, np.ndarray] = {}
    for stem in stems:
        gray = read_image(paths.TRAIN_IMAGES / f"{stem}.jpeg")
        prob = 1.0 / (1.0 + np.exp(-predict_logits_tiled(model, gray, device=device, tile=args.tile)))
        probs[stem] = prob
        disks[stem] = disk_mask(gray, relative_radius=0.98)
        densest = max(coco.by_stem[stem], key=lambda v: v.n_filaments)
        gts[stem] = rasterize_view_instances(densest, 2048, 1.0).astype(np.int32)
    print("done.\n")

    n_gt = float(np.mean([int(g.max()) for g in gts.values()]))
    print(f"GT instances/frame (after nothing): {n_gt:.1f}\n")

    rows = []
    for thr in args.thresholds:
        for ma in args.min_areas:
            pred = {s: prob_to_instances(probs[s], threshold=thr, min_area=ma,
                                         disk=disks[s], max_area_frac=0.05) for s in stems}
            rep = score_dataset(gts, pred)
            m = rep.metrics
            n_pred = float(np.mean([p.max() for p in pred.values()]))
            rows.append({
                "thr": thr, "min_area": ma, "n_pred": n_pred,
                "iw_dice_pairs": m["instance_wise_dice/pairs"],
                "iw_dice_image": m["instance_wise_dice/image"],
                "pq": m["panoptic_quality"],
                "sem_dice": m["semantic_dice_mean"],
                "o2m": m["one_to_many"], "m2o": m["many_to_one"],
            })

    rows.sort(key=lambda r: (-(r["iw_dice_pairs"] if np.isfinite(r["iw_dice_pairs"]) else -1)))
    print(f"{'thr':>5} {'min_area':>9} {'n_pred':>7} {'iw_dice/pairs':>14} "
          f"{'iw_dice/img':>12} {'PQ':>7} {'sem_dice':>9} {'1->N':>5} {'N->1':>5}")
    for r in rows:
        print(f"{r['thr']:>5} {r['min_area']:>9} {r['n_pred']:>7.1f} "
              f"{r['iw_dice_pairs']:>14.4f} {r['iw_dice_image']:>12.4f} "
              f"{r['pq']:>7.4f} {r['sem_dice']:>9.4f} {r['o2m']:>5} {r['m2o']:>5}")
    best = rows[0]
    print(f"\nbest: threshold={best['thr']} min_area={best['min_area']} "
          f"(instance_wise_dice/pairs {best['iw_dice_pairs']:.4f})")


if __name__ == "__main__":
    main()
