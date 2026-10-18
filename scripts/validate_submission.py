"""Validate a submission CSV against the competition format.

Checks: header/columns, unique ids, id stem matches a test image, RLE counts
decode losslessly to a 2048x2048 mask, and reports instance/area statistics.
Optionally renders predicted outlines on the test frames.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.data import list_test_files, read_image  # noqa: E402
from magfilo.rle import rle_counts_to_mask  # noqa: E402
from magfilo.viz import overlay_polygons, panel  # noqa: E402

H = W = 2048


def validate(csv_path: Path, visualize: int = 0, out_png: str | None = None) -> dict:
    errors: list[str] = []
    rows: list[tuple[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["filament_id", "segmentation_rle"]:
            errors.append(f"unexpected header: {header}")
        for r in reader:
            if len(r) != 2:
                errors.append(f"bad row width {len(r)}: {r[:2]}")
                continue
            rows.append((r[0], r[1]))

    test_stems = {Path(n).stem for n in list_test_files()}
    ids = [r[0] for r in rows]
    dup = len(ids) - len(set(ids))
    if dup:
        errors.append(f"{dup} duplicate filament_id values")

    per_image: dict[str, int] = defaultdict(int)
    areas: list[int] = []
    bad_stem = 0
    decode_fail = 0
    for fid, counts in rows:
        stem = fid.rsplit("_", 1)[0]
        if stem not in test_stems:
            bad_stem += 1
        per_image[stem] += 1
        try:
            m = rle_counts_to_mask(counts, H, W)
            areas.append(int(m.sum()))
        except Exception as exc:  # pragma: no cover
            decode_fail += 1
            if decode_fail <= 3:
                errors.append(f"decode failed for {fid}: {exc}")
    if bad_stem:
        errors.append(f"{bad_stem} rows reference an unknown test stem")
    if decode_fail:
        errors.append(f"{decode_fail} rows failed RLE decode")
    if areas and max(areas) == 0:
        errors.append("all decoded masks are empty")

    report = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "images_covered": len(per_image),
        "test_images_total": len(test_stems),
        "instances_per_image_mean": float(np.mean(list(per_image.values()))) if per_image else 0.0,
        "instances_per_image_max": max(per_image.values(), default=0),
        "area_px_p50": float(np.percentile(areas, 50)) if areas else 0.0,
        "area_px_p90": float(np.percentile(areas, 90)) if areas else 0.0,
        "errors": errors,
    }

    print(f"file: {csv_path}")
    for k, v in report.items():
        if k != "errors":
            print(f"  {k:<26} {v}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
    else:
        print("  format: OK")

    if visualize:
        _visualize(rows, visualize, out_png or str(paths.EDA_OUT / "fig_submission_preview.png"))
    return report


def _visualize(rows: list[tuple[str, str]], n: int, out_png: str) -> None:
    by_image: dict[str, list[np.ndarray]] = defaultdict(list)
    for fid, counts in rows:
        stem = fid.rsplit("_", 1)[0]
        if len(by_image) >= n and stem not in by_image:
            continue
        m = rle_counts_to_mask(counts, H, W)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if c.shape[0] >= 3:
                by_image[stem].append(c.reshape(-1, 2).astype(np.float32))
    items = []
    for stem, polys in list(by_image.items())[:n]:
        img = read_image(paths.TEST_IMAGES / f"{stem}.jpeg", scale=0.25)
        rgb = overlay_polygons(img, polys, scale=0.25)
        items.append((rgb, f"{stem} ({len(polys)} filaments)"))
    panel(items, out_png, ncols=3, suptitle="Classical baseline predictions on test frames")
    print(f"  wrote preview -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="outputs/submissions/submission_classical.csv")
    ap.add_argument("--visualize", type=int, default=0, help="render this many frames")
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()
    report = validate(Path(args.csv), args.visualize, args.out_png)
    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
