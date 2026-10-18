"""Exploratory data analysis for the MAGFiLO solar filament dataset.

Produces ``outputs/eda/stats.json`` plus a set of PNG figures and prints a
readable report. Run:

    python scripts/eda.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo import paths  # noqa: E402
from magfilo.coco import CATEGORIES, CocoData, load_coco  # noqa: E402
from magfilo.disk import disk_circle, disk_mask  # noqa: E402
from magfilo.rle import polygons_to_mask, semantic_union  # noqa: E402
from magfilo.viz import panel, save_fig  # noqa: E402

RNG = random.Random(20261018)
EDA_SCALE = 0.25  # rasterise/read at 512x512 for speed
N_INTENSITY_SAMPLES = 140


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def read_gray(file_name: str, test: bool = False, scale: float = EDA_SCALE) -> np.ndarray | None:
    d = paths.TEST_IMAGES if test else paths.TRAIN_IMAGES
    p = d / file_name
    if not p.exists():
        return None
    if scale == 0.25:
        return cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if scale == 0.5:
        return cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


# --------------------------------------------------------------------------- #
# Analysis blocks
# --------------------------------------------------------------------------- #
def structural_stats(coco: CocoData) -> dict:
    views_per_stem = Counter(len(vs) for vs in coco.by_stem.values())
    annots_per_view = Counter(v.n_filaments for v in coco.views)
    cat_counts = Counter(c for v in coco.views for c in v.category_ids)

    areas = np.array([a for v in coco.views for a in v.areas], dtype=np.float64)
    frame = float(paths.IMAGE_SIZE ** 2)

    elong = []
    spine_len = []
    for v in coco.views:
        for bb in v.bboxes:
            w, h = max(bb[2], 1e-6), max(bb[3], 1e-6)
            elong.append(max(w, h) / min(w, h))
        for sp in v.spines:
            if sp.shape[0] >= 2:
                spine_len.append(float(np.sum(np.linalg.norm(np.diff(sp, axis=0), axis=1))))

    annotators = Counter(v.annotator for v in coco.views)

    return {
        "n_physical_images": coco.n_physical(),
        "n_views": coco.n_views(),
        "n_annotations": coco.n_annotations(),
        "n_annotators": len(annotators),
        "views_per_image": {str(k): v for k, v in sorted(views_per_stem.items())},
        "multi_view_images": sum(v for k, v in views_per_stem.items() if k >= 2),
        "filaments_per_view": {str(k): v for k, v in sorted(annots_per_view.items())},
        "filaments_per_view_mean": coco.n_annotations() / coco.n_views(),
        "filaments_per_view_p50": pct([v.n_filaments for v in coco.views], 50),
        "filaments_per_view_max": max(v.n_filaments for v in coco.views),
        "category_counts": {CATEGORIES.get(k, str(k)): v for k, v in sorted(cat_counts.items())},
        "area_px": {q: pct(areas, q) for q in (1, 10, 25, 50, 75, 90, 99)},
        "area_mean_px": float(areas.mean()),
        "area_fraction_of_frame_p50": pct(areas, 50) / frame,
        "total_filament_frac_of_frame": float(areas.sum()) / (frame * coco.n_views()),
        "bbox_elongation": {q: pct(elong, q) for q in (10, 50, 90)},
        "spine_length_px": {q: pct(spine_len, q) for q in (10, 50, 90)},
        "spine_coverage": sum(1 for v in coco.views for _ in v.spines) / max(1, coco.n_annotations()),
    }


def disk_and_intensity(coco: CocoData) -> dict:
    """Are filaments dark *relative to the chromosphere on the disc*?

    The raw frame includes black sky outside the solar disc, which dominates a
    naive "outside filaments" statistic. We therefore compare filament pixels
    against disc (chromosphere) pixels only, and additionally measure each
    filament against a thin ring just outside its own boundary.
    """
    rng = random.Random(1)
    sample = rng.sample(coco.views, min(N_INTENSITY_SAMPLES, len(coco.views)))
    radii, inside_means, disc_bg_means, contrasts = [], [], [], []
    ring_contrasts = []
    inside_px: list[np.ndarray] = []
    bg_px: list[np.ndarray] = []
    for v in sample:
        img = read_gray(v.file_name)
        if img is None:
            continue
        _, _, r = disk_circle(img)
        if np.isfinite(r):
            radii.append(r)
        disk = disk_mask(img).astype(bool)
        m = semantic_union(v, v.height, v.width, scale=EDA_SCALE).astype(bool)
        if m.sum() == 0 or disk.sum() == 0:
            continue
        imgf = img.astype(np.float32)
        inside_means.append(float(imgf[m].mean()))
        disc_bg = disk & ~m
        disc_bg_means.append(float(imgf[disc_bg].mean()))
        contrasts.append(float(imgf[disc_bg].mean() - imgf[m].mean()))
        inside_px.append(imgf[m][::7].copy())
        bg_px.append(imgf[disc_bg][::53].copy())

        # per-filament: mean inside vs mean in a ring just outside the polygon
        for poly in v.polygons[:4]:
            pts = np.round(poly * EDA_SCALE).astype(np.int32)
            if pts.shape[0] < 3:
                continue
            fm = np.zeros(img.shape, np.uint8)
            cv2.fillPoly(fm, [pts], 1)
            fmb = fm.astype(bool)
            ring = (cv2.dilate(fm, np.ones((9, 9), np.uint8)) > 0) & ~fmb & disk
            if fmb.sum() and ring.sum():
                ring_contrasts.append(float(imgf[ring].mean() - imgf[fmb].mean()))

    ins = np.concatenate(inside_px) if inside_px else np.array([])
    bg = np.concatenate(bg_px) if bg_px else np.array([])
    return {
        "disk_radius_px_512_median": float(np.median(radii)) if radii else float("nan"),
        "disk_radius_px_2048_est": float(np.median(radii) * 4) if radii else float("nan"),
        "mean_inside_filament": float(np.mean(inside_means)) if inside_means else float("nan"),
        "mean_disc_background": float(np.mean(disc_bg_means)) if disc_bg_means else float("nan"),
        "median_contrast_discbg_minus_filament": float(np.median(contrasts)) if contrasts else float("nan"),
        "contrast_positive_fraction": float(np.mean([c > 0 for c in contrasts])) if contrasts else float("nan"),
        "per_filament_ring_contrast_p50": float(np.median(ring_contrasts)) if ring_contrasts else float("nan"),
        "per_filament_ring_contrast_positive_fraction": (
            float(np.mean([c > 0 for c in ring_contrasts])) if ring_contrasts else float("nan")
        ),
        "_inside_px": ins,
        "_bg_px": bg,
    }


def inter_annotator_agreement(coco: CocoData, scale: float = EDA_SCALE) -> dict:
    """Semantic union Dice between annotators on the same physical frame."""
    dices = []
    per_image = []
    for stem, views in coco.by_stem.items():
        if len(views) < 2:
            continue
        unions = [semantic_union(v, v.height, v.width, scale=scale).astype(bool) for v in views]
        d_local = []
        for i in range(len(unions)):
            for j in range(i + 1, len(unions)):
                a, b = unions[i], unions[j]
                inter = np.logical_and(a, b).sum()
                denom = a.sum() + b.sum()
                if denom == 0:
                    continue
                d_local.append(2.0 * inter / denom)
        if d_local:
            dices.extend(d_local)
            per_image.append((stem, float(np.mean(d_local)), len(views)))
    per_image.sort(key=lambda t: -t[1])
    return {
        "n_pairs": len(dices),
        "dice_mean": float(np.mean(dices)) if dices else float("nan"),
        "dice_p10": pct(dices, 10), "dice_p50": pct(dices, 50), "dice_p90": pct(dices, 90),
        "frac_pairs_below_0.5": float(np.mean([d < 0.5 for d in dices])) if dices else float("nan"),
        "worst_images": per_image[-5:],
        "best_images": per_image[:5],
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_samples(coco: CocoData) -> None:
    order = sorted(coco.views, key=lambda v: v.n_filaments)
    picks = [order[0], order[len(order) // 8], order[len(order) // 4],
             order[len(order) // 2], order[int(len(order) * 0.75)],
             order[-1]]
    items = []
    for v in picks:
        img = read_gray(v.file_name)
        if img is None:
            continue
        from magfilo.viz import overlay_polygons
        rgb = overlay_polygons(img, v.polygons, v.category_ids, scale=EDA_SCALE)
        items.append((rgb, f"{v.stem}  ({v.n_filaments} filaments)\nid={v.image_id}"))
    panel(items, str(paths.EDA_OUT / "fig_samples_train.png"), ncols=3,
          suptitle="MAGFiLO train: expert filaments overlaid (colour = Left/Right/Unident./Ambiguous)")


def fig_multi_view(coco: CocoData) -> None:
    from magfilo.viz import overlay_polygons
    stems = [s for s, vs in coco.by_stem.items() if len(vs) >= 3]
    if not stems:
        return
    stem = max(stems, key=lambda s: len(coco.by_stem[s]))
    views = coco.by_stem[stem]
    items = []
    for v in views[:6]:
        img = read_gray(v.file_name)
        if img is None:
            continue
        rgb = overlay_polygons(img, v.polygons, v.category_ids, scale=EDA_SCALE)
        items.append((rgb, f"annotator {v.annotator}\n{v.n_filaments} filaments"))
    panel(items, str(paths.EDA_OUT / "fig_multi_view.png"), ncols=3,
          suptitle=f"Same physical frame, independent annotators: {stem}")


def fig_distributions(coco: CocoData, stats: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    counts = [v.n_filaments for v in coco.views]
    axes[0, 0].hist(counts, bins=range(0, max(counts) + 2), color="#3b7dd8")
    axes[0, 0].set_title(f"Filaments per annotated view (mean {stats['filaments_per_view_mean']:.1f})")
    axes[0, 0].set_xlabel("filaments"); axes[0, 0].set_ylabel("views")

    vpc = Counter(len(vs) for vs in coco.by_stem.values())
    ks = sorted(vpc)
    axes[0, 1].bar([str(k) for k in ks], [vpc[k] for k in ks], color="#e08a3c")
    axes[0, 1].set_title(f"Annotator views per physical image "
                         f"({stats['n_physical_images']} images, {stats['n_views']} views)")
    axes[0, 1].set_xlabel("views"); axes[0, 1].set_ylabel("images")

    areas = np.array([a for v in coco.views for a in v.areas])
    axes[1, 0].hist(np.log10(np.clip(areas, 1, None)), bins=60, color="#4aa35a")
    axes[1, 0].set_title(f"Filament area (log10 px), median {stats['area_px'][50]:.0f} px")
    axes[1, 0].set_xlabel("log10 area (px)")

    cat = stats["category_counts"]
    axes[1, 1].bar(list(cat.keys()), list(cat.values()),
                   color=["#3498db", "#e67e22", "#8bc34a", "#e91e63"][:len(cat)])
    axes[1, 1].set_title("Class label distribution")
    axes[1, 1].set_ylabel("filaments")
    save_fig(fig, str(paths.EDA_OUT / "fig_distributions.png"))


def fig_intensity(inten: dict, coco: CocoData) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ins, bg = inten["_inside_px"], inten["_bg_px"]
    bins = np.linspace(0, 255, 80)
    axes[0].hist(bg, bins=bins, density=True, alpha=0.55, label="disc background", color="#888")
    axes[0].hist(ins, bins=bins, density=True, alpha=0.55, label="inside filaments", color="#c0392b")
    axes[0].set_title(f"Pixel intensity on the disc\n"
                      f"median ring contrast {inten['per_filament_ring_contrast_p50']:.1f} "
                      f"(filament darker in {100 * inten['per_filament_ring_contrast_positive_fraction']:.0f}% of cases)")
    axes[0].set_xlabel("8-bit intensity"); axes[0].legend()

    order = sorted(coco.views, key=lambda v: v.n_filaments)
    items = []
    from magfilo.viz import overlay_polygons
    for v in [order[len(order) // 3], order[-2]]:
        img = read_gray(v.file_name)
        if img is None:
            continue
        rgb = overlay_polygons(img, v.polygons, v.category_ids, scale=EDA_SCALE)
        items.append((rgb, f"{v.stem} ({v.n_filaments} filaments)"))
    if items:
        ax = axes[1]
        ax.axis("off")
        # stack the two example tiles horizontally into the second axis
        tiles = []
        for rgb, title in items:
            t = cv2.resize(rgb, (256, 256))
            tiles.append(t)
        axes[1].imshow(np.concatenate(tiles, axis=1))
        axes[1].set_title("Example: dark filament channels")
    save_fig(fig, str(paths.EDA_OUT / "fig_intensity.png"))


def fig_agreement(agree: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title(f"Inter-annotator agreement on the same frame\n"
                 f"mean Dice {agree['dice_mean']:.3f} over {agree['n_pairs']} pairs")
    # recompute a quick histogram by re-running? we stored only summary; embed
    # a bar summary instead to avoid recomputation cost
    labels = ["p10", "p50", "p90", "mean"]
    vals = [agree["dice_p10"], agree["dice_p50"], agree["dice_p90"], agree["dice_mean"]]
    ax.bar(labels, vals, color="#5d6dc0")
    ax.axhline(agree["dice_mean"], color="k", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("semantic Dice")
    save_fig(fig, str(paths.EDA_OUT / "fig_agreement.png"))


def main() -> None:
    paths.ensure_dirs()
    print(f"Data root: {paths.DATA_ROOT}")
    coco = load_coco()
    print(f"Loaded {coco.n_physical()} physical images / {coco.n_views()} views / "
          f"{coco.n_annotations()} filaments\n")

    stats = structural_stats(coco)
    inten = disk_and_intensity(coco)
    agree = inter_annotator_agreement(coco)

    # strip heavy arrays before serialising
    inten_serial = {k: v for k, v in inten.items() if not k.startswith("_")}
    stats["intensity"] = inten_serial
    stats["inter_annotator_agreement"] = agree

    (paths.EDA_OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps({**stats, "intensity": inten_serial}, indent=2, default=str))

    fig_samples(coco)
    fig_multi_view(coco)
    fig_distributions(coco, stats)
    fig_intensity(inten, coco)
    fig_agreement(agree)
    print(f"\nFigures + stats written to {paths.EDA_OUT}")


if __name__ == "__main__":
    main()
