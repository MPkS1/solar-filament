"""Offline evaluation on the held-out split using the local metric replica.

For each val physical frame we predict instances with the U-Net, rasterise each
annotator's ground truth, and score with `magfilo.metrics` (instance-wise Dice,
semantic Dice, Panoptic Quality, one-to-many / many-to-one relations).

Example:
    python scripts/evaluate_val.py --checkpoint checkpoints/unet.pt --limit 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402
from magfilo.data import read_image, split_stems, views_for_stems  # noqa: E402
from magfilo.metrics import score_dataset  # noqa: E402
from magfilo.predict import load_model, predict_logits_tiled, prob_to_instances  # noqa: E402
from magfilo.data import rasterize_view_instances  # noqa: E402
from magfilo.disk import disk_mask  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--limit", type=int, default=8, help="number of val frames to score")
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, base=args.base, depth=args.depth, device=device)

    coco = load_coco()
    train_stems, val_stems = split_stems(coco, args.val_frac, args.seed)
    stems = val_stems[: args.limit]
    print(f"scoring {len(stems)} held-out frames on {device}")

    pred_maps: dict[str, np.ndarray] = {}
    for stem in stems:
        gray = read_image(paths.TRAIN_IMAGES / f"{stem}.jpeg")
        logits = predict_logits_tiled(model, gray, device=device, tile=args.tile)
        prob = 1.0 / (1.0 + np.exp(-logits))
        disk = disk_mask(gray, relative_radius=0.98)
        labels = prob_to_instances(prob, threshold=args.threshold, min_area=args.min_area, disk=disk)
        pred_maps[stem] = labels

    # score against the densest annotator view for each frame
    gt_by_view: dict[str, np.ndarray] = {}
    for stem in stems:
        views = coco.by_stem[stem]
        densest = max(views, key=lambda v: v.n_filaments)
        gt_by_view[stem] = _view_labels(densest)

    rep = score_dataset(gt_by_view, pred_maps)
    print("\n=== instance_wise_dice (CC-reinstanced) ===")
    print(rep)
    print("\nper-frame sample:")
    for row in rep.per_image[:8]:
        print(f"  {row['image']:<22} n_gt={row['n_gt']:<3} n_pred={row['n_pred']:<3} "
              f"pq={row['pq']:.2f} sem={row['sem_dice']:.3f} pairs={row['n_pairs']}")

    # also without CC reinstancing for comparison
    rep_raw = score_dataset(gt_by_view, pred_maps, reinstantiate=False)
    print("\n=== instance_wise_dice (raw masks, no CC) ===")
    print("  pairs", f"{rep_raw.metrics['instance_wise_dice/pairs']:.4f}",
          " image", f"{rep_raw.metrics['instance_wise_dice/image']:.4f}")


def _view_labels(view) -> np.ndarray:
    return rasterize_view_instances(view, 2048, 1.0).astype(np.int32)


if __name__ == "__main__":
    main()
