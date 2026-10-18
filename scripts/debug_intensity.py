import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from magfilo import paths  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402

coco = load_coco()
v = sorted(coco.views, key=lambda v: v.n_filaments)[len(coco.views) // 2]
print("view", v.stem, v.image_id, "n=", v.n_filaments)

img = cv2.imread(str(paths.TRAIN_IMAGES / v.file_name), cv2.IMREAD_REDUCED_GRAYSCALE_4)
print("img", img.shape, img.dtype, "min/max/mean", img.min(), img.max(), round(float(img.mean()), 1))
print("percentiles 1/50/99", np.percentile(img, [1, 50, 99]).round(1))

m = np.zeros(img.shape, np.uint8)
for poly in v.polygons:
    cv2.fillPoly(m, [np.round(poly * 0.25).astype(np.int32)], 1)
mb = m.astype(bool)
print("union px", int(mb.sum()), "mean inside", round(float(img[mb].mean()), 1),
      "mean outside", round(float(img[~mb].mean()), 1))

for k, poly in enumerate(v.polygons[:3]):
    fm = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(fm, [np.round(poly * 0.25).astype(np.int32)], 1)
    fmb = fm.astype(bool)
    ring = (cv2.dilate(fm, np.ones((9, 9), np.uint8)) > 0) & ~fmb
    print(f"  fil{k}: inside({int(fmb.sum())}px)={img[fmb].mean():.1f}  "
          f"ring({int(ring.sum())}px)={img[ring].mean():.1f}  "
          f"contrast(ring-inside)={float(img[ring].mean() - img[fmb].mean()):.1f}")
    p0 = poly * 0.25
    cx, cy = int(p0[:, 0].mean()), int(p0[:, 1].mean())
    bb = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
    print("    centroid(512)=({},{}) bbox=({:.0f},{:.0f},{:.0f},{:.0f})".format(cx, cy, *bb))

# crop around first filament and save side-by-side with mask
poly0 = v.polygons[0] * 0.25
cx, cy = int(poly0[:, 0].mean()), int(poly0[:, 1].mean())
y0, y1 = max(cy - 70, 0), min(cy + 70, img.shape[0])
x0, x1 = max(cx - 70, 0), min(cx + 70, img.shape[1])
crop = img[y0:y1, x0:x1]
cm = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
both = np.concatenate([crop, cm], axis=1)
out = paths.EDA_OUT / "_debug_crop.png"
cv2.imwrite(str(out), both)
print("saved", out, "crop mean", round(float(crop.mean()), 1))
print("crop intensity: filament px mean", round(float(crop[cm > 0].mean()), 1),
      "non-filament px mean", round(float(crop[cm == 0].mean()), 1))

# --- replicate the EDA expression exactly to locate the bug ---
from magfilo.viz import CATEGORY_COLORS  # noqa: F401
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)
(cx2, cy2), r2 = cv2.minEnclosingCircle(c)
disk = np.zeros(img.shape, np.uint8)
cv2.circle(disk, (img.shape[1] // 2, img.shape[0] // 2), int(round(r2)), 1, -1)
disk = disk.astype(bool)
print("otsu disk r(512)=", round(r2, 1), "disk px frac=", round(float(disk.mean()), 3))

imgf = img.astype(np.float32)
poly = v.polygons[0]
pts = np.round(poly * 0.25).astype(np.int32)
fm = np.zeros(img.shape, np.uint8)
cv2.fillPoly(fm, [pts], 1)
eda_ring = cv2.dilate(fm, np.ones((9, 9), np.uint8), iterations=1) & ~fm.astype(bool) & disk
dbg_ring = (cv2.dilate(fm, np.ones((9, 9), np.uint8)) > 0) & ~fm.astype(bool)
print("EDA ring dtype", eda_ring.dtype, "sum", int(eda_ring.sum()),
      "mean_inside", round(float(imgf[fm.astype(bool)].mean()), 1),
      "mean_ring", round(float(imgf[eda_ring].mean()), 1))
print("DBG ring sum", int(dbg_ring.sum()), "mean", round(float(imgf[dbg_ring].mean()), 1))
print("inside disc-background mean", round(float(imgf[disk & ~fm.astype(bool)].mean()), 1))

