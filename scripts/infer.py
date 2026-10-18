"""Run the trained U-Net over the test set and write a submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo.predict import run_inference  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet.pt")
    ap.add_argument("--out", default="outputs/submissions/submission.csv")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--max-area-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    path = run_inference(
        checkpoint=args.checkpoint, out_csv=args.out, base=args.base, depth=args.depth,
        device=pick_device(args.device), tile=args.tile, threshold=args.threshold,
        min_area=args.min_area, max_area_frac=args.max_area_frac, limit=args.limit,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
