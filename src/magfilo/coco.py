"""COCO annotation loading and indexing for MAGFiLO.

The competition JSON is COCO-style with two twists:
  * ``image.id`` is a *string* like ``"040301-20140609195854Bh"`` = annotator
    batch + ``-`` + physical file name (no extension). The same physical frame
    can appear many times (multi-rater).
  * ``annotation.segmentation`` is a single polygon (list of one flat
    [x0,y0,x1,y1,...] path) -- NOT RLE, despite the page saying "polygons in
    RLE format".
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import TRAIN_JSON

CATEGORIES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


@dataclass
class View:
    """One annotator's pass over one physical image."""

    image_id: str          # e.g. "040301-20140609195854Bh"
    file_name: str         # e.g. "20140609195854Bh.jpeg"
    width: int
    height: int
    date_captured: str
    polygons: list[np.ndarray] = field(default_factory=list)   # each (-1, 2) float
    category_ids: list[int] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    spines: list[np.ndarray] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def annotator(self) -> str:
        return self.image_id.split("-", 1)[0]

    @property
    def n_filaments(self) -> int:
        return len(self.polygons)


class CocoData:
    def __init__(self, d: dict[str, Any]):
        self.raw = d
        self.info = d.get("info", {})
        self.categories = d.get("categories", [])
        self.licenses = d.get("licenses", [])

        self.views: list[View] = []
        by_image_id: dict[str, View] = {}
        for im in d["images"]:
            v = View(
                image_id=im["id"],
                file_name=im["file_name"],
                width=im["width"],
                height=im["height"],
                date_captured=im.get("date_captured", ""),
            )
            by_image_id[im["id"]] = v
            self.views.append(v)

        for a in d["annotations"]:
            v = by_image_id.get(a["image_id"])
            if v is None:
                continue
            seg = a["segmentation"]
            # Normalise: could be [poly] or [[poly]] or an RLE dict.
            if isinstance(seg, list):
                poly = seg[0] if seg and isinstance(seg[0], list) else seg
                v.polygons.append(np.asarray(poly, dtype=np.float32).reshape(-1, 2))
            elif isinstance(seg, dict):
                # not expected for this dataset, keep as-is
                v.polygons.append(np.empty((0, 2), dtype=np.float32))
            v.category_ids.append(int(a.get("category_id", 0)))
            v.areas.append(float(a.get("area", 0.0)))
            v.bboxes.append(list(a.get("bbox", [0, 0, 0, 0])))
            sp = a.get("spine")
            if sp:
                v.spines.append(np.asarray(sp, dtype=np.float32).reshape(-1, 2))

        # physical file name -> list of views
        self.by_stem: dict[str, list[View]] = defaultdict(list)
        for v in self.views:
            self.by_stem[v.stem].append(v)

    # -- convenience -----------------------------------------------------
    def n_views(self) -> int:
        return len(self.views)

    def n_physical(self) -> int:
        return len(self.by_stem)

    def n_annotations(self) -> int:
        return sum(v.n_filaments for v in self.views)

    def multi_view_stems(self, min_views: int = 2) -> list[str]:
        return [s for s, vs in self.by_stem.items() if len(vs) >= min_views]


def load_coco(path: Path | str = TRAIN_JSON) -> CocoData:
    with open(path, "r", encoding="utf-8") as f:
        return CocoData(json.load(f))
