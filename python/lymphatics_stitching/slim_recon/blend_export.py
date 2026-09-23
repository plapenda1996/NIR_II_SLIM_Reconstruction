# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
"""Mosaic blending and depth-coded colorization for SLIM stitching.

Two problems this module solves:

1. Tile seams in the grayscale mosaic. The earlier stitch averaged tiles with
   uniform weights inside overlap regions, which produces step discontinuities
   at tile boundaries (constant inside, jumping at the edge). It also let
   zero-valued background pixels of a tile contribute to the average, biasing
   the overlap toward zero. Replaced here by:
     - per-tile illumination normalization (background subtraction + median
       gain match across tiles),
     - linear feather weights (distance-to-edge ramp), with a *valid-mask*
       (tile > 0) gating contributions,
     - optional Burt-Adelson multi-band (Laplacian-pyramid) blending for
       residual low-frequency steps.

2. Speckle in the depth-coded color image. Per-pixel argmax-over-z is fragile
   in low-SNR regions because a noisy z-profile's peak slice is essentially
   random. Replaced by:
     - intensity-weighted-mean depth ``z_bar = sum(z*I^p) / sum(I^p)``,
     - axial smoothing of the volume before computing the weighted mean,
     - confidence mask from MIP percentiles, hard-zeroing background pixels,
     - guided-filter spatial regularization using MIP as guide,
     - HSV compositing with V = MIP_normalized * confidence so hue (depth) is
       only visible where there is real signal and brightness scales with MIP.

All functions are pure / numpy-only. ``tiles`` arguments are lists of dicts
with at least keys ``'mip'`` (H, W) float32, ``'volume'`` (H, W, D) float32,
and integer canvas-pixel positions ``'x'``, ``'y'``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter, grey_opening


# ── Feather weights ─────────────────────────────────────────────────────────

def feather_weight(h: int, w: int, margin: int) -> np.ndarray:
    """Separable distance-to-edge ramp: 1 in interior, linearly to 0 over ``margin``.

    Returns a (h, w) float32 array. ``margin`` is clamped to ``min(h, w) // 2``
    so the ramp never collapses to zero everywhere.
    """
    margin = max(1, min(int(margin), min(h, w) // 2))
    yy = np.arange(h, dtype=np.float32)
    xx = np.arange(w, dtype=np.float32)
    dy = np.minimum(yy, h - 1 - yy)
    dx = np.minimum(xx, w - 1 - xx)
    dy = np.clip(dy / margin, 0.0, 1.0)
    dx = np.clip(dx / margin, 0.0, 1.0)
    return (dy[:, None] * dx[None, :]).astype(np.float32)


# ── Per-tile illumination normalization ─────────────────────────────────────

def estimate_background(mip: np.ndarray, opening_radius: int = 64) -> np.ndarray:
    """Estimate slowly-varying background by morphological opening + smoothing."""
    r = max(3, int(opening_radius))
    bg = grey_opening(mip, size=r)
    bg = gaussian_filter(bg, sigma=r / 2.0)
    return bg.astype(np.float32)


def normalize_tile_illumination(mip: np.ndarray, opening_radius: int = 64) -> np.ndarray:
    """Subtract a low-percentile background to remove vignetting / exposure drift."""
    bg = estimate_background(mip, opening_radius)
    return np.clip(mip.astype(np.float32) - bg, 0.0, None)


def equalize_tile_gains(tile_mips):
    """Scale each tile so its non-zero median matches the global median.

    Cheap stand-in for full pairwise-overlap gain matching: removes the
    largest residual inter-tile mean differences without doing a graph
    least-squares solve.
    """
    medians = []
    for m in tile_mips:
        nz = m[m > 0]
        medians.append(float(np.median(nz)) if nz.size else 0.0)
    valid_meds = [v for v in medians if v > 0]
    if not valid_meds:
        return [m.astype(np.float32) for m in tile_mips]
    target = float(np.median(valid_meds))
    out = []
    for m, med in zip(tile_mips, medians):
        if med > 0:
            out.append((m * (target / med)).astype(np.float32))
        else:
            out.append(m.astype(np.float32))
    return out


# ── Linear (feather) blending ───────────────────────────────────────────────

def blend_linear(tiles, tile_imgs, canvas_size, margin: int | None = None,
                 valid_threshold: float = 0.0):
    """Linear feather blending.

    Composite: ``out = sum(img_i * w_i * v_i) / sum(w_i * v_i)`` where ``w_i``
    is the feather and ``v_i = (img_i > valid_threshold)`` excludes background
    pixels from the average. ``tiles`` and ``tile_imgs`` must align by index.
    """
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    acc = np.zeros((ch, cw), dtype=np.float64)
    wsum = np.zeros((ch, cw), dtype=np.float64)
    for tile, img in zip(tiles, tile_imgs):
        th, tw = img.shape[:2]
        tx, ty = int(tile['x']), int(tile['y'])
        x0 = max(0, tx); y0 = max(0, ty)
        x1 = min(cw, tx + tw); y1 = min(ch, ty + th)
        if x1 <= x0 or y1 <= y0:
            continue
        sx0 = x0 - tx; sy0 = y0 - ty
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
        m = margin if margin is not None else max(8, min(th, tw) // 8)
        w = feather_weight(th, tw, m)[sy0:sy1, sx0:sx1]
        sub = img[sy0:sy1, sx0:sx1].astype(np.float32)
        v = (sub > valid_threshold).astype(np.float32)
        eff = w * v
        acc[y0:y1, x0:x1] += sub * eff
        wsum[y0:y1, x0:x1] += eff
    out = np.zeros((ch, cw), dtype=np.float32)
    valid = wsum > 1e-9
    out[valid] = (acc[valid] / wsum[valid]).astype(np.float32)
    return out


def blend_linear_weighted(tiles, tile_imgs, tile_weights, canvas_size,
                          margin: int | None = None):
    """Like ``blend_linear`` but per-tile ``tile_weights[i]`` (e.g. MIP) is the
    intensity weight stacked on top of the feather. Use for blending depth maps
    where bright pixels should dominate.
    """
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    acc = np.zeros((ch, cw), dtype=np.float64)
    wsum = np.zeros((ch, cw), dtype=np.float64)
    for tile, img, weight in zip(tiles, tile_imgs, tile_weights):
        th, tw = img.shape[:2]
        tx, ty = int(tile['x']), int(tile['y'])
        x0 = max(0, tx); y0 = max(0, ty)
        x1 = min(cw, tx + tw); y1 = min(ch, ty + th)
        if x1 <= x0 or y1 <= y0:
            continue
        sx0 = x0 - tx; sy0 = y0 - ty
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
        m = margin if margin is not None else max(8, min(th, tw) // 8)
        feather = feather_weight(th, tw, m)[sy0:sy1, sx0:sx1]
        sub = img[sy0:sy1, sx0:sx1].astype(np.float32)
        wt = weight[sy0:sy1, sx0:sx1].astype(np.float32)
        v = (wt > 0).astype(np.float32)
        eff = feather * wt * v
        acc[y0:y1, x0:x1] += sub * eff
        wsum[y0:y1, x0:x1] += eff
    out = np.zeros((ch, cw), dtype=np.float32)
    valid = wsum > 1e-9
    out[valid] = (acc[valid] / wsum[valid]).astype(np.float32)
    return out


# ── Multi-band (Burt & Adelson) blending ────────────────────────────────────

def _down2(x: np.ndarray) -> np.ndarray:
    """Gaussian-smooth and decimate by 2 (REDUCE in Burt-Adelson notation)."""
    smoothed = gaussian_filter(x.astype(np.float32), sigma=1.0)
    return smoothed[::2, ::2]


def _up2(x: np.ndarray, target_shape) -> np.ndarray:
    """Upsample by 2 (zero-fill + Gaussian smooth, scaled by 4)."""
    h, w = target_shape
    h_src, w_src = x.shape
    out = np.zeros((h, w), dtype=np.float32)
    out[:h_src * 2:2, :w_src * 2:2] = x
    return gaussian_filter(out, sigma=1.0) * 4.0


def gaussian_pyramid(img: np.ndarray, n_levels: int):
    pyr = [img.astype(np.float32)]
    for _ in range(n_levels - 1):
        pyr.append(_down2(pyr[-1]))
    return pyr


def laplacian_pyramid(img: np.ndarray, n_levels: int):
    g = gaussian_pyramid(img, n_levels)
    lap = []
    for i in range(n_levels - 1):
        lap.append(g[i] - _up2(g[i + 1], g[i].shape))
    lap.append(g[-1])  # Gaussian residual at the top
    return lap


def collapse_pyramid(pyr):
    out = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        out = _up2(out, pyr[i].shape) + pyr[i]
    return out


def blend_multiband(tiles, tile_imgs, canvas_size, n_levels: int = 5,
                    margin: int | None = None):
    """Streaming Burt-Adelson multi-band blending across ``len(tiles)`` images.

    Memory: pyramidal accumulators at canvas resolution at each level (≤ ~1.3×
    canvas float32 in total). Each tile's pyramid is built on tile-local
    buffers and accumulated into the canvas pyramid at the level-scaled
    position. Avoids materializing N canvas-sized images in RAM.

    The blend mask used at each level is the per-tile feather weight × validity
    mask, smoothed by the same Gaussian decimation as the rest of the pyramid.
    """
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    n_levels = max(1, int(n_levels))
    sizes = []
    h_l, w_l = ch, cw
    for _ in range(n_levels):
        sizes.append((max(1, h_l), max(1, w_l)))
        h_l = max(1, h_l // 2)
        w_l = max(1, w_l // 2)

    accs = [np.zeros(s, dtype=np.float64) for s in sizes]
    wgts = [np.zeros(s, dtype=np.float64) for s in sizes]

    for tile, img in zip(tiles, tile_imgs):
        th, tw = img.shape[:2]
        if th < 2 or tw < 2:
            continue
        tx, ty = int(tile['x']), int(tile['y'])
        m = margin if margin is not None else max(8, min(th, tw) // 8)
        feather = feather_weight(th, tw, m)
        valid = (img > 0).astype(np.float32)
        mask = feather * valid
        # Limit pyramid depth so each level has at least 2 px on the smaller dim.
        max_lvl_tile = max(1, int(np.log2(min(th, tw))))
        n_l = min(n_levels, max_lvl_tile + 1)
        lap_pyr = laplacian_pyramid(img.astype(np.float32), n_l)
        msk_pyr = gaussian_pyramid(mask, n_l)
        for L in range(n_l):
            tlh, tlw = lap_pyr[L].shape
            txL, tyL = tx >> L, ty >> L
            CH_L, CW_L = sizes[L]
            x0L = max(0, txL); y0L = max(0, tyL)
            x1L = min(CW_L, txL + tlw); y1L = min(CH_L, tyL + tlh)
            if x1L <= x0L or y1L <= y0L:
                continue
            sxL = x0L - txL; syL = y0L - tyL
            wL = msk_pyr[L][syL:syL + (y1L - y0L), sxL:sxL + (x1L - x0L)]
            lL = lap_pyr[L][syL:syL + (y1L - y0L), sxL:sxL + (x1L - x0L)]
            accs[L][y0L:y1L, x0L:x1L] += lL * wL
            wgts[L][y0L:y1L, x0L:x1L] += wL

    blended = []
    for L in range(n_levels):
        out = np.zeros(sizes[L], dtype=np.float32)
        valid = wgts[L] > 1e-9
        out[valid] = (accs[L][valid] / wgts[L][valid]).astype(np.float32)
        blended.append(out)
    result = collapse_pyramid(blended)
    return np.clip(result[:ch, :cw], 0.0, None).astype(np.float32)


# ── Depth-coded color: top-K weighted depth + soft conf + vessel-aware ─────

def attenuation_correct_volume(volume: np.ndarray,
                                fg_pct: float = 80.0) -> np.ndarray:
    """Per-slice gain so deep slices match shallow slices for depth weighting.

    Each slice is divided by a robust foreground statistic (default 80th
    percentile across pixels in that slice). Without this step, depth-weighted
    means systematically bias shallow because tissue attenuation makes deep
    slices dimmer by orders of magnitude.
    """
    D = volume.shape[2]
    flat = volume.reshape(-1, D).astype(np.float32)
    per_slice = np.percentile(flat, float(fg_pct), axis=0)
    per_slice = np.maximum(per_slice, 1e-9)
    return (volume.astype(np.float32) / per_slice[None, None, :]).astype(np.float32)


def top_k_weighted_depth(volume: np.ndarray, k: int = 3, p: float = 1.0,
                          sigma_z: float = 1.5,
                          attenuation_correct: bool = True,
                          eps: float = 1e-9):
    """Return ``(depth, mip)`` using top-K intensity-weighted mean depth.

    Sort each (x, y) z-profile, keep the brightest ``k`` slices (after optional
    per-slice attenuation correction), and compute
        ``depth = sum_k (z_k * I^p) / sum_k I^p``.
    With p=1 (the new default — was p=2) the depth distribution stays diverse
    instead of collapsing to a narrow band; top-K avoids contamination from
    noise-only slices. ``mip`` is the original (un-attenuation-corrected) max.
    """
    H, W, D = volume.shape
    k = max(1, min(int(k), D))
    v_orig = volume.astype(np.float32)
    if sigma_z and sigma_z > 0:
        v_smooth = gaussian_filter1d(v_orig, sigma=float(sigma_z), axis=2)
    else:
        v_smooth = v_orig
    if attenuation_correct:
        v_for_depth = attenuation_correct_volume(v_smooth)
    else:
        v_for_depth = v_smooth
    # argpartition gives the indices of the top-k entries along axis=2.
    idx = np.argpartition(v_for_depth, -k, axis=2)[..., -k:]  # (H, W, k)
    vals = np.take_along_axis(v_for_depth, idx, axis=2)
    weights = np.power(np.clip(vals, 0.0, None), float(p))
    z = idx.astype(np.float32)
    num = np.sum(weights * z, axis=2)
    den = np.sum(weights, axis=2) + eps
    depth = (num / den).astype(np.float32)
    mip = np.max(v_orig, axis=2).astype(np.float32)
    return depth, mip


def argmax_depth(volume: np.ndarray, sigma_z: float = 1.5) -> np.ndarray:
    """Pure argmax-over-z depth (used as a "sharp" reference estimator)."""
    v = volume.astype(np.float32)
    if sigma_z and sigma_z > 0:
        v = gaussian_filter1d(v, sigma=float(sigma_z), axis=2)
    return np.argmax(v, axis=2).astype(np.float32)


def weighted_mean_depth(volume: np.ndarray, p: float = 1.0,
                        sigma_z: float = 1.5,
                        attenuation_correct: bool = True,
                        eps: float = 1e-9):
    """Backward-compat wrapper that uses the new top-K=full pipeline.

    Kept so older callers / saved scripts still work; new code should call
    :func:`top_k_weighted_depth` directly.
    """
    D = volume.shape[2]
    return top_k_weighted_depth(volume, k=D, p=p, sigma_z=sigma_z,
                                 attenuation_correct=attenuation_correct,
                                 eps=eps)


# ── Confidence: soft sigmoid keyed off background statistics ───────────────

def confidence_soft(mip: np.ndarray, k: float = 2.0,
                    bg_pct: float = 10.0,
                    bg_corner_frac: float = 0.0) -> np.ndarray:
    """Soft confidence in [0, 1].

    ``conf = sigmoid((MIP - mu_bg) / (k * sigma_bg))`` with mu_bg, sigma_bg
    estimated from the lowest ``bg_pct`` of MIP pixels (or, if
    ``bg_corner_frac > 0``, from the four ``bg_corner_frac``-sized canvas
    corners — usually empty for microscopy mosaics).

    Soft, *not* hard: dim true signal is preserved as partially-saturated
    color rather than killed outright.
    """
    if bg_corner_frac > 0:
        H, W = mip.shape
        ch_h = max(1, int(H * bg_corner_frac))
        ch_w = max(1, int(W * bg_corner_frac))
        corners = np.concatenate([
            mip[:ch_h, :ch_w].ravel(),
            mip[:ch_h, -ch_w:].ravel(),
            mip[-ch_h:, :ch_w].ravel(),
            mip[-ch_h:, -ch_w:].ravel(),
        ])
        bg = corners[corners > 0] if (corners > 0).any() else corners
    else:
        flat = mip.flatten()
        nz = flat[flat > 0]
        if nz.size == 0:
            return np.zeros_like(mip, dtype=np.float32)
        bg = nz[nz <= np.percentile(nz, float(bg_pct))]
        if bg.size == 0:
            bg = nz[: max(1, nz.size // 10)]
    mu_bg = float(np.mean(bg))
    sigma_bg = float(np.std(bg)) + 1e-9
    z = (mip - mu_bg) / (float(k) * sigma_bg)
    conf = 1.0 / (1.0 + np.exp(-z))
    return conf.astype(np.float32)


# Backward-compat alias for old callers.
confidence_from_mip = confidence_soft


def confidence_from_argmax_agreement(weighted_depth: np.ndarray,
                                      argmax_depth_arr: np.ndarray,
                                      threshold: float = 2.0) -> np.ndarray:
    """Returns ~1 where the two depth estimators agree, falling off Gaussian.

    Multiply this into the main confidence to *reduce* (not eliminate)
    saturation on pixels where the two estimators disagree by more than a
    couple slices — those are the genuinely ambiguous depth pixels.
    """
    diff = np.abs(weighted_depth - argmax_depth_arr)
    return np.exp(-(diff / max(float(threshold), 1e-9)) ** 2).astype(np.float32)


# ── Guided filter (kept; used as anisotropic-smoothing fallback) ───────────

def guided_filter(p: np.ndarray, guide: np.ndarray, radius: int = 4,
                  eps: float = 1e-3) -> np.ndarray:
    """He et al. 2010 box-filter guided filter (numpy, no cv2)."""
    I = guide.astype(np.float32)
    P = p.astype(np.float32)
    if I.ndim == 3:
        I = I.mean(axis=2)
    size = 2 * int(radius) + 1
    mean_I = uniform_filter(I, size=size)
    mean_P = uniform_filter(P, size=size)
    mean_IP = uniform_filter(I * P, size=size)
    cov_IP = mean_IP - mean_I * mean_P
    mean_II = uniform_filter(I * I, size=size)
    var_I = mean_II - mean_I * mean_I
    a = cov_IP / (var_I + float(eps))
    b = mean_P - a * mean_I
    mean_a = uniform_filter(a, size=size)
    mean_b = uniform_filter(b, size=size)
    return (mean_a * I + mean_b).astype(np.float32)


# ── Vessel-aware depth regularization ──────────────────────────────────────

def vesselness_map(mip: np.ndarray, sigmas=(1.0, 2.0, 4.0),
                   use_frangi: bool = True) -> np.ndarray:
    """Multi-scale Frangi (or Sato) tubular-structure response in [0, 1]."""
    from skimage.filters import frangi, sato
    img = mip.astype(np.float32)
    mx = float(img.max())
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float32)
    img = img / mx
    if use_frangi:
        v = frangi(img, sigmas=sigmas, black_ridges=False)
    else:
        v = sato(img, sigmas=sigmas, black_ridges=False)
    vmx = float(v.max())
    if vmx > 0:
        v = v / vmx
    return v.astype(np.float32)


def median_filter_within_vessel(depth: np.ndarray, vessel_mask: np.ndarray,
                                 size: int = 7) -> np.ndarray:
    """Apply a 2D median filter, but only update pixels inside ``vessel_mask``.

    Kills isolated wrong-depth pixels along a vessel without bleeding across
    vessel boundaries (since pixels outside the mask are left untouched).
    """
    from scipy.ndimage import median_filter
    smoothed = median_filter(depth.astype(np.float32), size=int(size))
    return np.where(vessel_mask, smoothed, depth).astype(np.float32)


def skeleton_propagated_depth(depth: np.ndarray, vessel_mask: np.ndarray,
                               sigma: float = 8.0,
                               min_component_size: int = 12) -> np.ndarray:
    """Skeleton-aware depth smoothing.

    1. Skeletonize ``vessel_mask`` to get 1-pixel centerlines.
    2. For each connected centerline component (within its bounding box), run a
       Gaussian-weighted average of depth with the centerline as the support
       mask. Because the centerline is 1-pixel-wide, this acts as a 1-D
       smoothing along the curve.
    3. Propagate centerline depth outward using the Euclidean distance
       transform's nearest-skeleton-pixel index, so every vessel pixel
       inherits the smoothed depth of its nearest centerline.
    Pixels outside the vessel mask keep the original ``depth``.
    """
    from skimage.morphology import skeletonize
    from scipy.ndimage import (gaussian_filter, label, find_objects,
                                distance_transform_edt)

    skel = skeletonize(vessel_mask).astype(bool)
    if not skel.any():
        return depth.astype(np.float32)

    smoothed_skel = depth.astype(np.float32).copy()
    labels, n_comp = label(skel)
    slices = find_objects(labels)
    margin = max(3, int(3 * sigma))
    H, W = depth.shape
    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        y0e = max(0, y0 - margin); y1e = min(H, y1 + margin)
        x0e = max(0, x0 - margin); x1e = min(W, x1 + margin)
        comp_local = (labels[y0e:y1e, x0e:x1e] == cid)
        if int(comp_local.sum()) < int(min_component_size):
            continue
        depth_local = depth[y0e:y1e, x0e:x1e].astype(np.float32)
        cd = np.where(comp_local, depth_local, 0.0)
        cm = comp_local.astype(np.float32)
        num = gaussian_filter(cd, float(sigma))
        den = gaussian_filter(cm, float(sigma)) + 1e-9
        sm_local = num / den
        out_local = smoothed_skel[y0e:y1e, x0e:x1e]
        out_local[comp_local] = sm_local[comp_local]
        smoothed_skel[y0e:y1e, x0e:x1e] = out_local

    inv_skel = ~skel
    _, idxs = distance_transform_edt(inv_skel, return_indices=True)
    propagated = smoothed_skel[idxs[0], idxs[1]].astype(np.float32)
    return np.where(vessel_mask, propagated, depth).astype(np.float32)


# ── V-channel processing pipeline ──────────────────────────────────────────

def percentile_stretch(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5,
                       fg_only: bool = True) -> np.ndarray:
    """Clip to ``[p_lo, p_hi]`` percentiles and rescale to [0, 1].

    With ``fg_only=True`` the percentiles are computed over nonzero pixels
    only, so a mostly-empty canvas doesn't compress the dynamic range.
    """
    arr = img.astype(np.float32)
    if fg_only:
        nz = arr[arr > 0]
        sample = nz if nz.size else arr.flatten()
    else:
        sample = arr.flatten()
    if sample.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(sample, float(p_lo)))
    hi = float(np.percentile(sample, float(p_hi)))
    span = max(hi - lo, 1e-9)
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def white_tophat(img: np.ndarray, radius: int = 12) -> np.ndarray:
    """Morphological white tophat to suppress diffuse out-of-focus background."""
    from skimage.morphology import disk, white_tophat as wth
    r = max(1, int(radius))
    se = disk(r)
    return wth(img.astype(np.float32), footprint=se).astype(np.float32)


def clahe_v(img: np.ndarray, clip_limit: float = 0.02,
            kernel_grid=(16, 16)) -> np.ndarray:
    """CLAHE on a [0, 1] V channel.

    skimage's ``equalize_adapthist`` uses ``clip_limit`` in [0, 1] (not the
    OpenCV-style 1–4 range). 0.02 ≈ OpenCV clipLimit=2.0.
    """
    from skimage.exposure import equalize_adapthist
    H, W = img.shape
    gy = max(2, int(kernel_grid[0]))
    gx = max(2, int(kernel_grid[1]))
    kh = max(8, H // gy)
    kw = max(8, W // gx)
    out = equalize_adapthist(np.clip(img, 0.0, 1.0),
                              clip_limit=float(clip_limit),
                              kernel_size=(kh, kw))
    return out.astype(np.float32)


def unsharp_v(img: np.ndarray, radius: float = 1.5,
              amount: float = 0.4) -> np.ndarray:
    """Unsharp mask for crisper vessel edges (applied after CLAHE)."""
    from skimage.filters import unsharp_mask
    out = unsharp_mask(np.clip(img, 0.0, 1.0),
                        radius=float(radius), amount=float(amount),
                        preserve_range=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def process_v_channel(mip: np.ndarray,
                      use_tophat: bool = True, tophat_radius: int = 12,
                      v_pct_lo: float = 1.0, v_pct_hi: float = 99.5,
                      use_clahe: bool = True, clahe_clip: float = 0.02,
                      clahe_grid=(16, 16),
                      use_unsharp: bool = True, unsharp_radius: float = 1.5,
                      unsharp_amount: float = 0.4) -> np.ndarray:
    """Produce the V (brightness) channel that the user wants to see.

    Order: tophat → percentile stretch → CLAHE → unsharp. This gives a [0, 1]
    array that should look essentially identical to the gray MIP target image
    described as ``stitched_gray_3.tif``.
    """
    v = mip.astype(np.float32)
    if use_tophat:
        v = white_tophat(v, radius=tophat_radius)
    v = percentile_stretch(v, p_lo=v_pct_lo, p_hi=v_pct_hi)
    if use_clahe:
        v = clahe_v(v, clip_limit=clahe_clip, kernel_grid=clahe_grid)
    if use_unsharp:
        v = unsharp_v(v, radius=unsharp_radius, amount=unsharp_amount)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


# ── Final HSV compositing: V tracks MIP, S tracks confidence ───────────────

def colorize_depth_hsv(depth: np.ndarray, v_channel: np.ndarray,
                       conf: np.ndarray, cmap_name: str = 'turbo',
                       depth_lo: float | None = None,
                       depth_hi: float | None = None,
                       saturation_boost: float = 1.2,
                       depth_gain_alpha: float = 0.0) -> np.ndarray:
    """Compose ``H = colormap(depth), S = conf, V = v_channel``.

    This is the central architectural fix: V is **decoupled** from confidence,
    so low-SNR pixels become *gray* (S=0) at full MIP brightness, instead of
    *black* (V=0). Acceptance criterion #1 (color brightness must equal MIP
    brightness) is structurally guaranteed by this design.

    ``depth_lo`` / ``depth_hi`` are the percentiles used to map depth → hue.
    When omitted they default to confidence-weighted percentiles so background
    depth values do not bias the colormap range.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if depth_lo is None or depth_hi is None:
        w = np.clip(conf.flatten(), 0.0, 1.0)
        d = depth.flatten()
        if w.sum() > 1e-6:
            order = np.argsort(d)
            sd = d[order]; sw = w[order]
            cw = np.cumsum(sw) / w.sum()
            if depth_lo is None:
                depth_lo = float(sd[max(0, np.searchsorted(cw, 0.01))])
            if depth_hi is None:
                idx = min(sd.size - 1, np.searchsorted(cw, 0.99))
                depth_hi = float(sd[idx])
        else:
            depth_lo = float(depth.min())
            depth_hi = float(depth.max())
    span = max(float(depth_hi - depth_lo), 1e-9)
    depth_norm = np.clip((depth - depth_lo) / span, 0.0, 1.0)

    cmap = plt.colormaps.get_cmap(cmap_name)
    rgb_from_depth = cmap(depth_norm)[:, :, :3].astype(np.float32)
    hsv = mcolors.rgb_to_hsv(rgb_from_depth)

    v = np.clip(v_channel.astype(np.float32), 0.0, 1.0)
    if depth_gain_alpha > 0:
        v = np.clip(v * (1.0 + float(depth_gain_alpha) * depth_norm), 0.0, 1.0)

    hsv[:, :, 1] = np.clip(conf.astype(np.float32) * float(saturation_boost),
                            0.0, 1.0)
    hsv[:, :, 2] = v
    rgb = mcolors.hsv_to_rgb(hsv)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


# ── Diagnostics: seam strength per row / column ─────────────────────────────

def seam_diagnostic(image: np.ndarray):
    """Return (row_diff, col_diff): mean |neighbour difference| per row / col.

    For an image of shape (H, W), ``row_diff`` has length H-1 (vertical seams
    show as peaks) and ``col_diff`` has length W-1 (horizontal seams show as
    peaks). Convert to grayscale by averaging channels if ``image.ndim == 3``.
    """
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    row_diff = np.mean(np.abs(np.diff(img, axis=0)), axis=1)
    col_diff = np.mean(np.abs(np.diff(img, axis=1)), axis=0)
    return row_diff, col_diff


def find_seam_peaks(diff: np.ndarray, threshold_factor: float = 4.0):
    """Return indices of ``diff`` more than ``threshold_factor`` MADs above median."""
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) or 1e-9
    threshold = med + threshold_factor * mad
    peaks = np.where(diff > threshold)[0]
    return peaks, threshold


def print_seam_report(label: str, image: np.ndarray, top_k: int = 8):
    """Pretty-print the strongest row and column seam locations."""
    row_d, col_d = seam_diagnostic(image)
    row_peaks, row_thr = find_seam_peaks(row_d)
    col_peaks, col_thr = find_seam_peaks(col_d)
    row_top = row_peaks[np.argsort(row_d[row_peaks])[::-1][:top_k]] \
        if row_peaks.size else np.array([], dtype=int)
    col_top = col_peaks[np.argsort(col_d[col_peaks])[::-1][:top_k]] \
        if col_peaks.size else np.array([], dtype=int)
    print(f"[seam] {label}:")
    print(f"  row_diff  median={float(np.median(row_d)):.4f}  "
          f"mean={float(np.mean(row_d)):.4f}  max={float(np.max(row_d)):.4f}  "
          f"thr={row_thr:.4f}  peaks(y)={list(map(int, row_top))}")
    print(f"  col_diff  median={float(np.median(col_d)):.4f}  "
          f"mean={float(np.mean(col_d)):.4f}  max={float(np.max(col_d)):.4f}  "
          f"thr={col_thr:.4f}  peaks(x)={list(map(int, col_top))}")
    return {'row_diff': row_d, 'col_diff': col_d,
            'row_peaks': row_top, 'col_peaks': col_top,
            'row_thr': row_thr, 'col_thr': col_thr}


# ── High-level pipeline ─────────────────────────────────────────────────────

def stitch_gray(tiles, canvas_size, blend: str = 'linear',
                normalize: bool = True, margin: int | None = None,
                n_levels: int = 5):
    """One-shot grayscale mosaic: per-tile illumination → tile gain match → blend.

    ``blend`` is ``'linear'`` (linear feather) or ``'multiband'`` (5-level
    Laplacian). Returns a (ch, cw) float32 mosaic, normalized to its own
    99.5-percentile so caller can scale to uint8.
    """
    tile_mips = []
    for t in tiles:
        m = t['mip'].astype(np.float32)
        if normalize:
            m = normalize_tile_illumination(m)
        tile_mips.append(m)
    if normalize:
        tile_mips = equalize_tile_gains(tile_mips)

    if blend == 'multiband':
        out = blend_multiband(tiles, tile_mips, canvas_size, n_levels=n_levels,
                              margin=margin)
    else:
        out = blend_linear(tiles, tile_mips, canvas_size, margin=margin)
    return out


def diagnose_color_vs_mip(rgb: np.ndarray, mip_v: np.ndarray) -> dict:
    """Quantify how closely V(color) tracks the gray MIP and report S, depth.

    Acceptance criterion #1 ("brightness of the color image must match the
    gray MIP exactly") translates to ``mean(V_color - MIP_v) ≈ 0`` and small
    stddev. The HSV→RGB→HSV round-trip is essentially exact, so any non-zero
    mean here means the V-channel pipeline is doing something it shouldn't.
    """
    import matplotlib.colors as mcolors
    arr = rgb.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(arr)
    diff = hsv[:, :, 2] - np.clip(mip_v.astype(np.float32), 0.0, 1.0)
    return {
        'mean_V': float(np.mean(hsv[:, :, 2])),
        'mean_S': float(np.mean(hsv[:, :, 1])),
        'mean_V_minus_MIP': float(np.mean(diff)),
        'std_V_minus_MIP': float(np.std(diff)),
        'visible_pixels': int(np.sum(hsv[:, :, 2] > 0.05)),
    }


def stitch_color(tiles, canvas_size,
                 # depth estimator
                 weight_power: float = 1.0,
                 top_k: int = 3,
                 sigma_z: float = 1.5,
                 attenuation_correct: bool = True,
                 use_argmax_agreement: bool = True,
                 argmax_threshold: float = 2.0,
                 # confidence
                 conf_k: float = 2.0,
                 conf_bg_pct: float = 10.0,
                 conf_corner_frac: float = 0.0,
                 # vessel-aware smoothing
                 use_vesselness: bool = True,
                 vessel_sigmas=(1.0, 2.0, 4.0),
                 vessel_threshold: float = 0.05,
                 use_median_within_vessel: bool = True,
                 median_size: int = 7,
                 use_skeleton_smooth: bool = True,
                 skel_sigma: float = 8.0,
                 use_guided_filter: bool = True,
                 gf_radius: int = 4,
                 gf_eps: float = 1e-3,
                 # V-channel processing
                 use_tophat: bool = True,
                 tophat_radius: int = 12,
                 v_pct_lo: float = 1.0,
                 v_pct_hi: float = 99.5,
                 use_clahe: bool = True,
                 clahe_clip: float = 0.02,
                 clahe_grid=(16, 16),
                 use_unsharp: bool = True,
                 unsharp_radius: float = 1.5,
                 unsharp_amount: float = 0.4,
                 depth_gain_alpha: float = 0.0,
                 saturation_boost: float = 1.2,
                 # blending of tile mosaics
                 cmap: str = 'turbo',
                 blend: str = 'linear',
                 normalize: bool = True,
                 margin: int | None = None,
                 n_levels: int = 5,
                 verbose: bool = True,
                 return_layers: bool = False):
    """Depth-coded color mosaic. ``V = MIP, S = conf, H = cmap(depth)``.

    Pipeline summary:
      1. Per tile: top-K weighted-mean depth (with per-slice attenuation
         correction so deep slices contribute on equal footing) + raw MIP.
         An argmax-depth is also computed; agreement with the weighted depth
         multiplies into the confidence later.
      2. Per tile (optional): illumination-normalize the MIP.
      3. Across tiles: linear or multi-band blend of MIP; MIP-weighted blend
         of depth (bright pixels dominate the depth average).
      4. On the canvas: confidence = soft sigmoid of MIP, optionally
         multiplied by argmax-agreement.
      5. Vessel-aware regularization of depth: Frangi vesselness → median
         filter inside vessel mask → skeletonize + per-component Gaussian →
         distance-transform propagation outward. Guided filter as fallback
         on the rest.
      6. V channel = white tophat → percentile stretch → CLAHE → unsharp,
         applied to the **blended MIP**, NOT to confidence-modulated MIP.
      7. Compose H = cmap(depth), S = conf, V = processed MIP.
    """
    if verbose:
        print(f"[stitch_color] {len(tiles)} tiles, canvas={canvas_size[0]}x{canvas_size[1]}")

    tile_mips, tile_depths, tile_argmax = [], [], []
    for t in tiles:
        depth, mip = top_k_weighted_depth(
            t['volume'], k=top_k, p=weight_power, sigma_z=sigma_z,
            attenuation_correct=attenuation_correct)
        tile_depths.append(depth)
        tile_argmax.append(argmax_depth(t['volume'], sigma_z=sigma_z))
        if normalize:
            mip = normalize_tile_illumination(mip)
        tile_mips.append(mip)

    if normalize:
        meds = [float(np.median(m[m > 0])) if (m > 0).any() else 0.0
                for m in tile_mips]
        valid = [v for v in meds if v > 0]
        target = float(np.median(valid)) if valid else 0.0
        gains = [(target / med) if (med > 0 and target > 0) else 1.0
                  for med in meds]
        tile_mips = [m * g for m, g in zip(tile_mips, gains)]

    # ── Cross-tile blending ──
    if blend == 'multiband':
        canvas_mip = blend_multiband(tiles, tile_mips, canvas_size,
                                      n_levels=n_levels, margin=margin)
    else:
        canvas_mip = blend_linear(tiles, tile_mips, canvas_size, margin=margin)
    canvas_depth = blend_linear_weighted(
        tiles, tile_depths, tile_mips, canvas_size, margin=margin)
    canvas_argmax = blend_linear_weighted(
        tiles, tile_argmax, tile_mips, canvas_size, margin=margin)

    # ── Confidence ──
    conf = confidence_soft(canvas_mip, k=conf_k, bg_pct=conf_bg_pct,
                            bg_corner_frac=conf_corner_frac)
    if use_argmax_agreement:
        agreement = confidence_from_argmax_agreement(
            canvas_depth, canvas_argmax, threshold=argmax_threshold)
        conf = conf * agreement

    # ── Vessel-aware depth smoothing ──
    depth_smooth = canvas_depth.copy()
    vmask = None
    if use_vesselness:
        vness = vesselness_map(canvas_mip, sigmas=vessel_sigmas)
        vmask = vness > float(vessel_threshold)
        if verbose:
            print(f"[stitch_color] vesselness mask: "
                  f"{int(vmask.sum())} px ({100.0 * vmask.mean():.1f}%)")
        if use_median_within_vessel and vmask.any():
            depth_smooth = median_filter_within_vessel(
                depth_smooth, vmask, size=median_size)
        if use_skeleton_smooth and vmask.any():
            depth_smooth = skeleton_propagated_depth(
                depth_smooth, vmask, sigma=skel_sigma)
    if use_guided_filter:
        depth_smooth = guided_filter(
            depth_smooth, canvas_mip, radius=gf_radius, eps=gf_eps)

    # ── V-channel pipeline (independent of confidence) ──
    v_channel = process_v_channel(
        canvas_mip,
        use_tophat=use_tophat, tophat_radius=tophat_radius,
        v_pct_lo=v_pct_lo, v_pct_hi=v_pct_hi,
        use_clahe=use_clahe, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
        use_unsharp=use_unsharp, unsharp_radius=unsharp_radius,
        unsharp_amount=unsharp_amount,
    )

    # ── Compose ──
    rgb = colorize_depth_hsv(
        depth_smooth, v_channel, conf,
        cmap_name=cmap,
        saturation_boost=saturation_boost,
        depth_gain_alpha=depth_gain_alpha,
    )

    if verbose:
        diag = diagnose_color_vs_mip(rgb, v_channel)
        print(f"[stitch_color]   mean V             = {diag['mean_V']:.4f}")
        print(f"[stitch_color]   mean S             = {diag['mean_S']:.4f}  "
              f"(target 0.15-0.30)")
        print(f"[stitch_color]   mean(V - MIP)      = {diag['mean_V_minus_MIP']:+.5f}  "
              f"(target ~0)")
        print(f"[stitch_color]   std(V - MIP)       = {diag['std_V_minus_MIP']:.5f}  "
              f"(target ~0)")
        print(f"[stitch_color]   visible (V > 0.05) = {diag['visible_pixels']:,} px")
        # Depth histogram (10 bins, weighted by confidence so bg doesn't dominate)
        wd = depth_smooth.flatten()
        ww = conf.flatten()
        if ww.sum() > 1e-6:
            hist, edges = np.histogram(wd, bins=10, weights=ww)
            hist_pct = 100.0 * hist / hist.sum()
            print("[stitch_color]   depth histogram (conf-weighted):")
            for h, e0, e1 in zip(hist_pct, edges[:-1], edges[1:]):
                bar = '#' * int(round(h / 4))
                print(f"             {e0:6.2f}..{e1:6.2f}  {h:5.1f}%  {bar}")

    if return_layers:
        return rgb, {
            'canvas_mip': canvas_mip,
            'v_channel': v_channel,
            'conf': conf,
            'depth': depth_smooth,
            'depth_raw': canvas_depth,
            'argmax_depth': canvas_argmax,
            'vesselness_mask': vmask,
        }
    return rgb
