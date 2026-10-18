"""Sweep inference post-processing on the held-out split, ranked by official PQ.

The leaderboard metric is Panoptic Quality, so that is what we optimise. The
sweep mirrors the organizers' scorer exactly: every annotator view is a separate
entry, and TP/FP/FN are pooled across all entries before PQ is formed.

Probability maps are computed once per frame and cached on disk
(``outputs/cache/probs``), so re-running the sweep costs only the connected-
components step per combination. The first run pays for the tiled inference.

Example:
    python scripts/tune_postprocess.py --checkpoint checkpoints/unet.pt --limit 16
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
from magfilo.metrics import pq_pooled  # noqa: E402
from magfilo.predict import load_model, prob_to_instances  # noqa: E402
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
    ap.add_argument("--limit", type=int, default=16, help="held-out frames to use")
    ap.add_argument("--offset", type=int, default=0, help="skip this many val frames first")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.7, 0.8, 0.84, 0.88])
    ap.add_argument("--min-areas", type=int, nargs="+",
                    default=[40, 300, 550, 900])
    ap.add_argument("--json", default="outputs/tune_postprocess.json",
                    help="write the best post-processing config here")
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, base=args.base, depth=args.depth, device=device)

    coco = load_coco()
    _, val_stems = split_stems(coco, args.val_frac, args.seed)
    stems = val_stems[args.offset: args.offset + args.limit]
    print(f"using {len(stems)} held-out frames "
          f"(val split index {args.offset}..{args.offset + len(stems) - 1})")

    probs: dict[str, np.ndarray] = {}
    disks: dict[str, np.ndarray] = {}
    gts: dict[str, list[np.ndarray]] = {}
    for i, stem in enumerate(stems, 1):
        img_path = paths.TRAIN_IMAGES / f"{stem}.jpeg"
        gray = read_image(img_path)
        probs[stem] = cached_prob_map(img_path, model, device, tile=args.tile,
                                      refresh=args.refresh_cache, gray=gray)
        disks[stem] = disk_mask(gray, relative_radius=0.98)
        # int16 label maps: plenty for instance counts, half the memory of int32
        gts[stem] = [rasterize_view_instances(v, 2048, 1.0).astype(np.int16)
                     for v in coco.by_stem[stem]]
        print(f"  [{i}/{len(stems)}] {stem}: {len(gts[stem])} annotator view(s)", end="\r")
    print()

    n_entries = sum(len(v) for v in gts.values())
    n_gt = float(np.mean([np.mean([int(g.max()) for g in views]) for views in gts.values()]))
    print(f"{n_entries} annotator-image entries; mean GT filaments per view: {n_gt:.1f}\n")

    rows = []
    for thr in args.thresholds:
        for ma in args.min_areas:
            pred = {s: prob_to_instances(probs[s], threshold=thr, min_area=ma,
                                         disk=disks[s], max_area_frac=0.05) for s in stems}
            rep = pq_pooled((gt, pred[stem])
                            for stem, views in gts.items() for gt in views)
            rows.append({
                "thr": thr, "min_area": ma,
                "n_pred": float(np.mean([p.max() for p in pred.values()])),
                "pq": rep.pq, "tp": rep.tp, "fp": rep.fp, "fn": rep.fn,
                "sem_dice": rep.semantic_dice_mean,
            })

    rows.sort(key=lambda r: -r["pq"])
    print(f"{'thr':>5} {'min_area':>9} {'n_pred':>7} {'PQ':>7} {'sem_dice':>9} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    for r in rows:
        print(f"{r['thr']:>5} {r['min_area']:>9} {r['n_pred']:>7.1f} {r['pq']:>7.4f} "
              f"{r['sem_dice']:>9.4f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")

    best = rows[0]
    print(f"\nbest PQ {best['pq']:.4f} at threshold={best['thr']} min_area={best['min_area']} "
          f"(n_pred {best['n_pred']:.1f} vs GT {n_gt:.1f})")
    print("the organizers describe any PQ > 0.35 as 'of great value'")

    if args.json:
        import json
        payload = {
            "threshold": best["thr"],
            "min_area": best["min_area"],
            "pq": best["pq"],
            "semantic_dice": best["sem_dice"],
            "checkpoint": args.checkpoint,
            "n_frames": len(stems),
            "n_annotator_image_entries": n_entries,
            "mean_gt_instances_per_view": n_gt,
            "all": rows,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
