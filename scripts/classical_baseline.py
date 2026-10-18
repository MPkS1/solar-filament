"""Classical-CV baseline: produces a valid submission with no training.

Useful as a sanity check of the submission pipeline and as a fallback. Matches
the ~0.46 ballpark of the public no-ML notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo.predict import run_inference_classical  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/submissions/submission_classical.csv")
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=15.0, help="Gaussian background sigma")
    ap.add_argument("--ratio", type=float, default=0.75, help="fraction of p99 darkness to threshold")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    path = run_inference_classical(
        out_csv=args.out, min_area=args.min_area, sigma=args.sigma, ratio=args.ratio, limit=args.limit,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
