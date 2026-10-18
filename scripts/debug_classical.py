import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from magfilo import paths  # noqa: E402
from magfilo.data import list_test_files, read_image  # noqa: E402
from magfilo.disk import disk_mask  # noqa: E402

files = list_test_files()[:5]
print("GT reference: median filament area 1228 px, ~7 filaments/image\n")

for rel in (0.88, 0.92):
    for sigma in (15, 25):
        for ratio in (0.6, 0.75):
            for min_area in (400, 1200):
                rows = []
                for name in files:
                    gray = read_image(paths.TEST_IMAGES / name)
                    disk = disk_mask(gray, relative_radius=rel)
                    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    dark = bg.astype(np.float32) - gray.astype(np.float32)
                    r = dark[disk > 0]
                    thr = ratio * float(np.percentile(r, 99))
                    b = ((dark > thr) & (disk > 0)).astype(np.uint8)
                    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
                    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                                    if int(stats[i, cv2.CC_STAT_AREA]) >= min_area), reverse=True)
                    rows.append((len(areas), areas[0] if areas else 0))
                n_inst = np.mean([r[0] for r in rows])
                top = max(r[1] for r in rows)
                flag = "  <-- limb blob?" if top > 60000 else ""
                print(f"rel={rel} sig={sigma:>2} ratio={ratio} min_area={min_area:>5}: "
                      f"inst/img={n_inst:5.1f}  max_area={top:>7}{flag}")
