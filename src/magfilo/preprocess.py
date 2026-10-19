"""Full-disc preprocessing for GONG H-alpha frames.

Measured on MAGFiLO (see ``outputs/eda/stats.json``), two effects dominate a raw
frame and both hurt a plain U-Net trained with a fixed input normalisation:

1. **Limb darkening.** The disc is roughly twice as bright at the centre as at
   the limb, so one fixed learned filter means very different things at
   different radii. Dividing by the radial intensity profile flattens the disc,
   which is what the published MAGFiLO models do (EdgeAttNet, MORDEN).

2. **Large-scale background.** Filaments are dark *relative to their local
   surroundings* -- true for 99.8 % of annotated filaments -- but the surrounding
   brightness varies slowly across the disc (network, plage, supergranulation).
   Subtracting a smooth local background isolates the filaments from those slow
   variations.

Measured effect on real annotated frames: filament contrast roughly doubles
(-0.24 -> -0.62 sigma) and becomes more uniform with radius (relative spread
41 % -> 26 %), which makes even a single global threshold 3x better.

The chain is deterministic, so the exact same operating point is used at
training and at inference time.

Usage::

    from magfilo.preprocess import load_frame
    x = load_frame(path)          # float32, zero-mean/unit-std over the disc
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .disk import disk_circle

EPS = 1e-6
DEFAULT_SIGMA = 25.0      # local-background scale, px at 2048 (filaments are ~40 px wide)
DEFAULT_NBINS = 128
BG_DOWNSCALE = 4          # background is smooth; estimating it at 1/4 res is equivalent


def _radius_map(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    yy = np.arange(h, dtype=np.float32)[:, None] - np.float32(cy)
    xx = np.arange(w, dtype=np.float32)[None, :] - np.float32(cx)
    rad = yy * yy + xx * xx          # broadcasts to (h, w) without building both grids
    np.sqrt(rad, out=rad)
    return rad


def _profile_from_radius(gray: np.ndarray, rad: np.ndarray, r: float,
                         nbins: int = DEFAULT_NBINS, step: int = 4) -> np.ndarray:
    """Mean intensity per radial bin, smoothed.

    Estimated on a 1/``step`` grid: a 128-bin profile over a 2048 frame still
    averages thousands of pixels per bin, so subsampling changes the profile by
    far less than the frame-to-frame variation and costs 1/16 of the time.
    """
    r_s = rad[::step, ::step]
    g_s = gray[::step, ::step]
    inside = r_s <= r
    idx = np.clip((r_s[inside] * np.float32(nbins / max(r, EPS))).astype(np.int32),
                  0, nbins - 1)
    vals = g_s[inside].astype(np.float64)
    sums = np.bincount(idx, weights=vals, minlength=nbins)
    cnts = np.bincount(idx, minlength=nbins).astype(np.float64)
    prof = sums / np.maximum(cnts, 1.0)

    empty = cnts == 0
    if empty.any():
        good = np.flatnonzero(~empty)
        prof[empty] = (np.interp(np.flatnonzero(empty), good, prof[good])
                       if good.size else 1.0)

    k = max(1, nbins // 32)
    return np.convolve(prof, np.ones(k, np.float64) / k, mode="same")


def radial_profile(gray: np.ndarray, cx: float, cy: float, r: float,
                   nbins: int = DEFAULT_NBINS) -> np.ndarray:
    rad = _radius_map(*gray.shape[:2], cx, cy)
    return _profile_from_radius(gray, rad, r, nbins)


def _flatten_with_radius(gray: np.ndarray, rad: np.ndarray, r: float,
                         nbins: int = DEFAULT_NBINS) -> np.ndarray:
    prof = _profile_from_radius(gray, rad, r, nbins).astype(np.float32)
    idx = (rad * np.float32(nbins / max(r, EPS))).astype(np.int32)
    np.clip(idx, 0, nbins - 1, out=idx)
    gain = np.take(prof, idx)
    out = gray.astype(np.float32)
    np.divide(out, gain, out=out)
    return out


def limb_flatten(gray: np.ndarray, cx: float, cy: float, r: float,
                 nbins: int = DEFAULT_NBINS) -> tuple[np.ndarray, np.ndarray]:
    """Divide out the radial intensity profile. Returns (flattened, disc mask)."""
    rad = _radius_map(*gray.shape[:2], cx, cy)
    return _flatten_with_radius(gray, rad, r, nbins), rad <= r


def local_background(img: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth background, estimated at reduced resolution then upsampled.

    A Gaussian of sigma=25 px is band-limited far below the 2048 grid, so doing
    it on a 512 grid costs 1/16 of the work for no visible difference.
    """
    h, w = img.shape[:2]
    if sigma < BG_DOWNSCALE:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sh, sw = max(2, h // BG_DOWNSCALE), max(2, w // BG_DOWNSCALE)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / BG_DOWNSCALE,
                             sigmaY=sigma / BG_DOWNSCALE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(gray: np.ndarray, sigma: float = DEFAULT_SIGMA,
                     nbins: int = DEFAULT_NBINS,
                     standardize: bool = True) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Raw frame -> limb-flattened, locally contrast-normalised float32 image.

    Dark filaments stay dark (negative after standardisation).
    """
    cx, cy, r = disk_circle(gray, 1.0)
    rad = _radius_map(*gray.shape[:2], cx, cy)
    flat = _flatten_with_radius(gray, rad, r, nbins)
    lc = flat - local_background(flat, sigma)
    if standardize:
        inside = rad <= r
        n_in = float(inside.sum())
        if n_in > 0:
            s1 = float(np.sum(lc, where=inside, dtype=np.float64))
            s2 = float(np.sum(lc * lc, where=inside, dtype=np.float64))
            mu = s1 / n_in
            sd = float(np.sqrt(max(s2 / n_in - mu * mu, 0.0)))
            np.subtract(lc, np.float32(mu), out=lc)
            np.divide(lc, np.float32(max(sd, EPS)), out=lc)
    return np.ascontiguousarray(lc, dtype=np.float32), (cx, cy, r)


def load_frame(path: Path | str, preprocess: bool = True,
               sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """Read a frame as the model expects it: (H, W) float32.

    ``preprocess=False`` returns the raw uint8 grayscale, so legacy checkpoints
    (trained on raw input) stay loadable.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read {path}")
    if not preprocess:
        return gray
    out, _ = preprocess_frame(gray, sigma=sigma)
    return out
