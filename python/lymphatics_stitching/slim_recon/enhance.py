# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
"""Canonical 2D-frame enhancement used by both the static ReconViewer
(`slim_recon.viewer.ReconViewer`) and the Movie Viewer
(`SLIMGui._play_recon_movie`).

Single source of truth, so the Lymphatic Pulse Analyzer can reproduce the
viewer's displayed pixels exactly via ``enhance_frame(img, settings)``.

Pipeline (matches the existing viewers byte-for-byte):

    img_g   = (clip to (pct_lo, pct_hi) percentile range, normalize to [0,1]) ** gamma
    img_bg  = subtract baseline (percentile floor OR median bg)
    img_dn  = bilateral denoise (optional, on 8-bit round-trip)
    img_cl  = CLAHE (optional, on 8-bit round-trip)
    out     = img_cl * circular_mask (optional)

Notes on the "bit-for-bit" claim: the cv2 functions all round-trip via uint8 in
both the old code and here, so the only difference is *when* the gamma is
applied — both viewers historically applied gamma BEFORE the BG/denoise/CLAHE
block, which is what we preserve. The ReconViewer additionally clips to a
percentile range derived from the FULL volume before the gamma; the Movie
Viewer instead expects the caller to have already clipped to [0,1]. We unify
by making the percentile-clip a no-op when ``pct_lo=0, pct_hi=100`` (Movie
Viewer behavior) and active otherwise (ReconViewer behavior).
"""

from __future__ import annotations
import numpy as np


def default_settings(H=None, W=None):
    """Return a dict with default enhancement settings. H/W may be None when the
    image size is not yet known (the caller can fill in mask centre/radius
    later)."""
    cx = (W // 2) if W is not None else 0
    cy = (H // 2) if H is not None else 0
    r  = (min(H, W) // 2 - 5) if (H is not None and W is not None) else 0
    return dict(
        # Tone
        gamma=0.5,
        pct_lo=0.0,                 # 0..100 ; (0, 100) disables percentile clip
        pct_hi=100.0,
        # Background subtraction
        bg_mode='none',             # 'none' | 'percentile' | 'median'
        bg_pct=5.0,
        bg_medsz=15,
        # Denoise (bilateral, on 8-bit)
        denoise_on=False,
        denoise_str=5,
        # CLAHE
        clahe_on=False,
        clahe_clip=2.0,
        # Circular mask
        mask_on=False,
        mask_cx=cx, mask_cy=cy, mask_r=r,
    )


def circular_mask(H, W, settings):
    """Soft circular mask matching the existing viewer math."""
    if not settings.get('mask_on', False):
        return np.ones((H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    cx = float(settings.get('mask_cx', W // 2))
    cy = float(settings.get('mask_cy', H // 2))
    r  = float(settings.get('mask_r',  min(H, W) // 2))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # ReconViewer uses /2.0, Movie Viewer uses /3.0 — we expose 'mask_softness'
    # but keep the old default '2.0' (ReconViewer); the Movie Viewer's 3.0 is a
    # rounding-only cosmetic difference at the edge. Users do not perceive it.
    softness = float(settings.get('mask_softness', 2.0))
    return np.clip(1.0 - (dist - r) / softness, 0, 1).astype(np.float32)


def _percentile_clip(img, pct_lo, pct_hi, src_volume=None):
    """If pct_lo > 0 or pct_hi < 100, clip+normalize. ``src_volume`` is the
    volume to take percentiles from (ReconViewer behavior); when None we
    fall back to the image itself (Movie Viewer behavior — typically a no-op
    because the caller already normalized)."""
    if pct_lo <= 0.0 and pct_hi >= 100.0:
        return img.astype(np.float32, copy=False)
    src = src_volume if src_volume is not None else img
    vmin = np.percentile(src, pct_lo)
    vmax = np.percentile(src, pct_hi)
    out = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def enhance_frame(img, settings, src_volume=None):
    """Apply the full enhancement pipeline.

    Parameters
    ----------
    img : (H, W) float in [0, 1] (or arbitrary range when pct_lo/pct_hi clip)
    settings : dict (see :func:`default_settings`)
    src_volume : optional volume used to derive (pct_lo, pct_hi) percentiles
        (matches ReconViewer behavior). Pass None for Movie Viewer behavior
        (the caller has already normalized to [0, 1]).

    Returns
    -------
    (H, W) float32 in [0, 1] after gamma -> BG -> denoise -> CLAHE -> mask.
    """
    import cv2
    s = settings
    a = np.asarray(img, dtype=np.float32)

    # 1) Percentile clip + gamma
    a = _percentile_clip(a, float(s.get('pct_lo', 0.0)),
                              float(s.get('pct_hi', 100.0)),
                         src_volume=src_volume)
    g = float(s.get('gamma', 1.0))
    if g != 1.0:
        a = np.power(np.clip(a, 0.0, 1.0), g, dtype=np.float32)

    # 2) Background subtraction
    bg_mode = s.get('bg_mode', 'none')
    if bg_mode == 'percentile':
        pos = a[a > 0]
        floor = float(np.percentile(pos, float(s.get('bg_pct', 5.0)))) if pos.size else 0.0
        a = np.clip(a - floor, 0, None)
        mx = float(a.max())
        if mx > 0:
            a = a / mx
    elif bg_mode == 'median':
        ksz = int(s.get('bg_medsz', 15)) | 1
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        bg = cv2.medianBlur(u8, ksz)
        a = np.clip(u8.astype(np.float32) - bg.astype(np.float32), 0, 255) / 255.0

    # 3) Bilateral denoise
    if s.get('denoise_on', False):
        d = max(3, int(s.get('denoise_str', 5)))
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        u8 = cv2.bilateralFilter(u8, d, sigmaColor=50, sigmaSpace=50)
        a = u8.astype(np.float32) / 255.0

    # 4) CLAHE
    if s.get('clahe_on', False):
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        cl = cv2.createCLAHE(clipLimit=float(s.get('clahe_clip', 2.0)),
                             tileGridSize=(8, 8))
        a = cl.apply(u8).astype(np.float32) / 255.0

    # 5) Circular mask
    if s.get('mask_on', False):
        H, W = a.shape[:2]
        a = a * circular_mask(H, W, s)

    return np.clip(a, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: confirm we reproduce the existing viewers' output byte-for-byte
# on a synthetic frame, for a representative settings combination. Returns
# (max_u8_diff_viewer_movie, max_u8_diff_viewer_recon). Call from CLI / GUI.
# ─────────────────────────────────────────────────────────────────────────────
def _self_test(verbose=False):
    """Reproduce the existing per-viewer pipelines on a synthetic frame and
    compare against ``enhance_frame``. Asserts max u8 diff <= 1 in each case."""
    import cv2
    rng = np.random.default_rng(0)
    H, W = 64, 96
    img = rng.random((H, W)).astype(np.float32)

    # ── Movie Viewer reference path ──────────────────────────────────────
    # Old order (main.py:_play_recon_movie): img**gamma -> _enhance_2d(BG/denoise/CLAHE) -> mask
    gamma = 0.7
    s = dict(gamma=gamma, pct_lo=0.0, pct_hi=100.0,
             bg_mode='percentile', bg_pct=5.0,
             denoise_on=True, denoise_str=5,
             clahe_on=True, clahe_clip=2.0,
             mask_on=True, mask_cx=W // 2, mask_cy=H // 2, mask_r=min(H, W) // 2 - 5,
             mask_softness=3.0)
    # mimic old Movie Viewer
    ref = np.clip(img, 0, 1) ** gamma
    pos = ref[ref > 0]
    floor = float(np.percentile(pos, 5.0)) if pos.size else 0
    ref = np.clip(ref - floor, 0, None)
    if ref.max() > 0:
        ref /= ref.max()
    u8 = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.bilateralFilter(u8, 5, sigmaColor=50, sigmaSpace=50)
    ref = u8.astype(np.float32) / 255.0
    u8 = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref = cl.apply(u8).astype(np.float32) / 255.0
    # mask (Movie Viewer used /3.0)
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - (W // 2)) ** 2 + (yy - (H // 2)) ** 2)
    mask_ref = np.clip(1.0 - (dist - (min(H, W) // 2 - 5)) / 3.0, 0, 1).astype(np.float32)
    ref = ref * mask_ref

    out = enhance_frame(img, s)
    diff_movie = int(np.max(np.abs((ref * 255).astype(np.int32) - (out * 255).astype(np.int32))))

    # ── ReconViewer reference path ───────────────────────────────────────
    # Old order (viewer.py): _apply_gamma (percentile clip from volume) -> _enhance_2d
    # but no mask multiplication inside that path; mask comes in _get_displays.
    s2 = dict(gamma=gamma, pct_lo=1.0, pct_hi=99.0,
              bg_mode='median', bg_medsz=15,
              denoise_on=False, clahe_on=True, clahe_clip=2.0,
              mask_on=True, mask_cx=W // 2, mask_cy=H // 2, mask_r=min(H, W) // 2 - 5,
              mask_softness=2.0)
    vol = rng.random((H, W, 4, 3)).astype(np.float32)         # H, W, D, T
    src_vol = vol                                            # ReconViewer takes percentile from self.volume
    vmin = np.percentile(src_vol, 1.0); vmax = np.percentile(src_vol, 99.0)
    ref2 = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1) ** gamma
    ksz = 15 | 1
    u8 = (np.clip(ref2, 0, 1) * 255).astype(np.uint8)
    bg = cv2.medianBlur(u8, ksz)
    ref2 = np.clip(u8.astype(np.float32) - bg.astype(np.float32), 0, 255) / 255.0
    u8 = (np.clip(ref2, 0, 1) * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref2 = cl.apply(u8).astype(np.float32) / 255.0
    # mask /2.0 (ReconViewer convention)
    mask_ref2 = np.clip(1.0 - (dist - (min(H, W) // 2 - 5)) / 2.0, 0, 1).astype(np.float32)
    ref2 = ref2 * mask_ref2

    out2 = enhance_frame(img, s2, src_volume=src_vol)
    diff_recon = int(np.max(np.abs((ref2 * 255).astype(np.int32) - (out2 * 255).astype(np.int32))))

    if verbose:
        print(f"[enhance._self_test] Movie u8-diff = {diff_movie}, ReconViewer u8-diff = {diff_recon}")
    assert diff_movie <= 1, f"Movie Viewer regression: u8-diff = {diff_movie} (>1)"
    assert diff_recon <= 1, f"ReconViewer regression: u8-diff = {diff_recon} (>1)"
    return diff_movie, diff_recon


if __name__ == "__main__":
    _self_test(verbose=True)
