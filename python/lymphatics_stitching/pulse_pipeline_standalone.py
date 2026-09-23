#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - lymphatic pulse detection - produces Fig. 3k-m
# environment: pulse_py314
"""
pulse_pipeline_standalone.py
============================
Standalone reproduction of the SLIMGui "Pulse Flow Analyzer" lymphatic
pulse-analysis pipeline (main.py), applied directly to a TIFF stack.

WHAT IT DOES
------------
1. Rebuilds the LOCAL->ROOT frame map for a doubly-concatenated video:

       original 28,978-frame recording (965.93 s @ 30 fps)
         --(concate_set_ver1)-->  intermediate 2,945-frame video
         --(concate_set_ver2)-->  final ~941-frame video   (the TIFF you have)

   so every frame of the final TIFF carries its REAL time in the original
   recording (root_frame / fps).  This is exactly the frame-provenance contract
   in main.py (make_root_map / compose_map): the map always points at the ROOT,
   which is what lets repeated extractions be un-scrambled back to real time.

2. For each segment in concate_set_ver2, runs the SAME analysis the recon Pulse
   Flow Analyzer runs (main.py `_recon_block` -> `pulse_arclength_core`):
       drift/photometry-corrected crop
         -> consecutive-frame difference (rectified, leading edge)
         -> arclength profile b(s,t) along the FG flow line (band ±band px)
         -> bolus position s*(t) = arclength at the peak of b
         -> RANSAC regression of s*(t) vs REAL time -> propagation velocity
       plus: diameter kymograph -> EDD/ESD/EF, intensity -> dF/F0 & frequency,
       cross-correlation & kymograph velocities, spatial bolus FWHM & mass,
       contraction / relaxation velocity.

3. Aggregates the per-segment results (pooled velocity ± SD, CV, 95% CI,
   inverse-variance weighted mean, Q/I²; pooled EF/EDD/ESD/transport) and
   computes ROOT-time physiology (contraction peaks in real time, inter-
   contraction interval, honest frequency + recording coverage).

The numerical kernels between the two banner comments below are copied VERBATIM
from main.py so the numbers reproduce the GUI.  Everything else (frame-map
rebuild, the standalone per-segment driver, aggregation, outputs) is authored
around them.  See the NOTE in --help for drift/photometry caveats.

USAGE
-----
    # edit the CONFIG dict at the bottom, then:
    python pulse_pipeline_standalone.py
    # ...or pass everything on the command line (CLI overrides CONFIG):
    python pulse_pipeline_standalone.py --tiff final.tif --ver2 concate_set_ver2.json \
        --ver1 concate_set_ver1.json --outdir pulse_out
    python pulse_pipeline_standalone.py -h        # full options + notes

Requires: numpy, scipy, tifffile, matplotlib (scikit-learn optional, for RANSAC;
falls back to a numpy RANSAC if absent — identical to main.py's behavior).
"""

import numpy as np

# ============================================================================
#  VERBATIM NUMERICAL KERNELS FROM main.py  (do not edit — keeps numbers exact)
# ============================================================================

# ---- _sample_along_polyline  (main.py L65-89, verbatim) ----
def _sample_along_polyline(pts_xy, n_samples):
    """Resample an ordered polyline at N equally arc-spaced points.
    pts_xy: (M,2) array of (x,y). Returns (centers Nx2, tangents Nx2 unit, arclen N in px)."""
    P = np.asarray(pts_xy, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return None, None, None
    seg = np.diff(P, axis=0)                         # (M-1, 2)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])         # (M-1,)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-9:
        return None, None, None
    targets = np.linspace(0.0, total, int(max(2, n_samples)))
    centers = np.empty((len(targets), 2), float)
    tangents = np.empty((len(targets), 2), float)
    for i, td in enumerate(targets):
        # find segment containing td
        k = int(np.searchsorted(cum, td, side='right') - 1)
        k = max(0, min(k, len(seg) - 1))
        t = (td - cum[k]) / (seg_len[k] + 1e-12)
        centers[i] = P[k] + t * seg[k]
        # tangent = local segment direction (forward); unit-normalize
        tv = seg[k] / (seg_len[k] + 1e-12)
        tangents[i] = tv
    return centers, tangents, targets

# ---- _perp_profile  (main.py L92-108, verbatim) ----
def _perp_profile(img2d, center_xy, tangent_xy, length_px, n_samples=None):
    """Bilinear-interp 1D intensity profile perpendicular to ``tangent_xy`` through
    ``center_xy``, length ``length_px`` (centered). Uses scipy.ndimage.map_coordinates.
    Returns the 1D profile (length n_samples, default = round(length_px) + 1)."""
    from scipy.ndimage import map_coordinates
    cx, cy = float(center_xy[0]), float(center_xy[1])
    tx, ty = float(tangent_xy[0]), float(tangent_xy[1])
    # perpendicular unit vector (rotate tangent 90 CCW): (-ty, tx)
    px, py = -ty, tx
    L = float(length_px)
    n = int(round(L) + 1) if n_samples is None else int(n_samples)
    s = np.linspace(-L / 2.0, L / 2.0, n)
    xs = cx + s * px
    ys = cy + s * py
    # map_coordinates uses (row, col) = (y, x)
    prof = map_coordinates(img2d, np.vstack([ys, xs]), order=1, mode='nearest')
    return prof.astype(np.float32)

# ---- _fwhm_from_profile  (main.py L111-147, verbatim) ----
def _fwhm_from_profile(profile, frac=0.5):
    """Width (px) at ``frac × (max − base)`` of a 1D profile. Base = min over the
    profile; peak = max. Linear-interpolate left/right crossings; NaN if invalid."""
    p = np.asarray(profile, float)
    if p.size < 3:
        return float('nan')
    base = float(p.min()); peak = float(p.max())
    if peak - base < 1e-9:
        return float('nan')
    thr = base + frac * (peak - base)
    above = p >= thr
    if not above.any():
        return float('nan')
    # locate the contiguous run that contains the global argmax (the main peak)
    pk = int(np.argmax(p))
    # left crossing: rightmost i < pk where above[i] is False  ->  interp i..i+1
    li = pk
    while li > 0 and above[li - 1]:
        li -= 1
    # ri: leftmost i > pk where above[i] is False
    ri = pk
    while ri < len(p) - 1 and above[ri + 1]:
        ri += 1
    # left interp
    if li == 0:
        left = 0.0
    else:
        y0, y1 = p[li - 1], p[li]
        left = (li - 1) + (thr - y0) / (y1 - y0 + 1e-12)
    # right interp
    if ri == len(p) - 1:
        right = float(len(p) - 1)
    else:
        y0, y1 = p[ri], p[ri + 1]
        right = ri + (thr - y0) / (y1 - y0 + 1e-12)
    w = float(right - left)
    return w if w > 0 else float('nan')

# ---- _diameter_kymograph  (main.py L150-162, verbatim) ----
def _diameter_kymograph(proj_THW, centers, tangents, profile_len_px, fwhm_frac, px_um):
    """For each centerline sample point and frame, FWHM lumen width in µm.
    Returns diam_um[N, T]."""
    T = proj_THW.shape[0]; N = centers.shape[0]
    diam_um = np.full((N, T), np.nan, np.float32)
    for t in range(T):
        img = proj_THW[t]
        for i in range(N):
            prof = _perp_profile(img, centers[i], tangents[i], profile_len_px)
            w_px = _fwhm_from_profile(prof, fwhm_frac)
            if w_px == w_px:                          # not NaN
                diam_um[i, t] = w_px * px_um
    return diam_um

# ---- _intensity_kymograph_band  (main.py L165-179, verbatim) ----
def _intensity_kymograph_band(proj_THW, centers, band_half_px, px_um):
    """Mean intensity in a small (2r+1)x(2r+1) box around each centerline point,
    per frame. Returns intens[N, T]. (Simple, robust; the full polyline-band
    version is fine but this matches each sampled point directly.)"""
    T, H, W = proj_THW.shape
    N = centers.shape[0]; r = max(0, int(band_half_px))
    intens = np.zeros((N, T), np.float32)
    cx = centers[:, 0].astype(int); cy = centers[:, 1].astype(int)
    for i in range(N):
        x = cx[i]; y = cy[i]
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        if x1 > x0 and y1 > y0:
            intens[i] = proj_THW[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return intens

# ---- _pulse_metrics_from_trace  (main.py L182-226, verbatim) ----
def _pulse_metrics_from_trace(trace, fps, hp_window_s=5.0, prom_mult=0.5,
                              dist_s=0.3, savgol_w=7):
    """Generic pulse-detection from a 1D trace. savgol smooth -> high-pass detrend
    (uniform_filter1d of size hp_window_s*fps) -> find_peaks(prom=prom_mult*std,
    distance=dist_s*fps) -> hilbert envelope. Returns a dict with all arrays
    + scalar metrics. ``hp_window_s`` is the SECONDS scale of the slow drift to
    remove (5 s default suits lymphatic pulse; 2 s suits cardiac)."""
    from scipy.signal import savgol_filter, find_peaks, hilbert
    from scipy import ndimage as _ndi
    x = np.asarray(trace, float)
    n = len(x); fps = float(fps)
    if n < 3:
        return dict(n=n, fps=fps, signal_hp=x, envelope=x, peaks=np.array([], int),
                    intervals=np.array([]), rates_bpm=np.array([]),
                    amps=np.array([]), peak_times_s=np.array([]),
                    freq_cpm=float('nan'), interval_s_mean=float('nan'))
    w = int(max(3, savgol_w)); w |= 1                # odd
    w = min(w, n if n % 2 else n - 1)
    try:
        x_sm = savgol_filter(x, w, min(3, w - 1))
    except Exception:
        x_sm = _ndi.uniform_filter1d(x, max(3, w))
    hp = max(3, int(round(hp_window_s * fps)))
    x_hp = x_sm - _ndi.uniform_filter1d(x_sm, size=hp)
    pk, _ = find_peaks(x_hp, distance=max(1, int(round(dist_s * fps))),
                       prominence=np.std(x_hp) * prom_mult + 1e-12)
    env = np.abs(hilbert(x_hp - x_hp.mean()))
    t_ax = np.arange(n) / fps
    if len(pk) >= 2:
        intervals = np.diff(pk) / fps
        rates_bpm = 60.0 / intervals                   # generic 'bpm' = events/min
        amps = x_hp[pk[:-1]]
        peak_times_s = t_ax[pk[:-1]]
        freq_cpm = float(60.0 / np.mean(intervals)) if intervals.size else float('nan')
        interval_s_mean = float(np.mean(intervals))
    else:
        intervals = np.array([]); rates_bpm = np.array([])
        amps = np.array([]); peak_times_s = np.array([])
        freq_cpm = float('nan'); interval_s_mean = float('nan')
    return dict(n=n, fps=fps, t_s=t_ax, signal_hp=x_hp, envelope=env,
                peaks=pk, intervals=intervals, rates_bpm=rates_bpm,
                amps=amps, peak_times_s=peak_times_s,
                freq_cpm=freq_cpm, interval_s_mean=interval_s_mean,
                hp_window_s=float(hp_window_s), prom_mult=float(prom_mult),
                dist_s=float(dist_s), savgol_w=int(w))

# ---- _velocity_from_kymograph  (main.py L229-265, verbatim) ----
def _velocity_from_kymograph(kymo, fps, px_um):
    """Per-point peak times -> linear fit position(µm) vs time(s).
    Returns (velocity_um_s, slope_residual_um, direction_sign). The kymograph is
    (N_pts, T) with arc-length increasing along axis 0. Velocity sign: +
    distal-direction propagation (peak appears later at larger arc-length)."""
    from scipy.signal import find_peaks
    N, T = kymo.shape
    if N < 3 or T < 5:
        return float('nan'), float('nan'), 0
    # For each point row, take the strongest peak time (in seconds)
    peak_t_s = np.full(N, np.nan)
    for i in range(N):
        row = kymo[i]
        if not np.isfinite(row).any():
            continue
        rrow = row - np.nanmean(row)
        rrow = np.nan_to_num(rrow, nan=0.0)
        pk, _ = find_peaks(rrow, prominence=np.nanstd(rrow) * 0.4 + 1e-9)
        if len(pk):
            peak_t_s[i] = pk[int(np.argmax(rrow[pk]))] / fps
    valid = np.isfinite(peak_t_s)
    if valid.sum() < 3:
        return float('nan'), float('nan'), 0
    pos_um = np.arange(N) * (kymo.shape[0] and (1.0))   # placeholder; caller sets via px_um
    pos_um = np.linspace(0.0, (N - 1), N)               # in "sample index"; caller passes arclen
    # We don't know arclen here directly; use point index * mean_spacing_um instead via px_um.
    # Caller should pass kymo computed at fixed arc spacing → here we assume arclen is uniform
    # and pass px_um as the spacing PER SAMPLE step (i.e., total_len_um / (N-1)).
    pos_um = np.arange(N) * float(px_um)
    coeffs, residuals, *_ = np.polyfit(pos_um[valid], peak_t_s[valid], 1, full=True)
    slope_s_per_um = float(coeffs[0])                   # s/µm
    if abs(slope_s_per_um) < 1e-12:
        return float('nan'), float('nan'), 0
    v_um_s = 1.0 / slope_s_per_um
    direction = int(np.sign(v_um_s))
    resid = float(residuals[0]) if len(residuals) else float('nan')
    return float(v_um_s), resid, direction

# ---- _velocity_from_xcorr  (main.py L268-283, verbatim) ----
def _velocity_from_xcorr(trace_prox, trace_dist, sep_um, fps):
    """Lag (s) of distal vs proximal trace via normalized cross-correlation.
    velocity_um_s = sep_um / lag_s. Direction: sign(lag) = anterograde if lag > 0
    (distal peaks after proximal). Returns (velocity_um_s, lag_s, direction)."""
    a = np.asarray(trace_prox, float); b = np.asarray(trace_dist, float)
    if a.size < 5 or b.size < 5 or a.size != b.size:
        return float('nan'), float('nan'), 0
    a = a - a.mean(); b = b - b.mean()
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    corr = np.correlate(b, a, mode='full') / norm
    lag = int(np.argmax(corr)) - (len(a) - 1)
    lag_s = lag / float(fps)
    if abs(lag_s) < 1.0 / fps:                          # sub-sample / undefined
        return float('inf'), lag_s, 0
    v = float(sep_um) / lag_s
    return float(v), float(lag_s), int(np.sign(lag_s))

# ---- _ransac_line  (main.py L458-495, verbatim) ----
def _ransac_line(t, s, thr=0.0, min_samples=3, max_trials=200, seed=0):
    """Numpy-only RANSAC for y = m*x + b. thr<=0 -> auto (MAD of OLS residuals).
    Returns (slope, intercept, inlier_mask)."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    n = int(t.size)
    if n < 2:
        return float('nan'), float('nan'), np.zeros(n, dtype=bool)
    if thr is None or thr <= 0:
        try:
            m0, b0 = np.polyfit(t, s, 1); r0 = s - (m0 * t + b0)
            mad = float(np.median(np.abs(r0 - np.median(r0))))
            thr = max(1e-9, 2.5 * 1.4826 * (mad if mad > 0 else float(np.std(r0)) + 1e-9))
        except Exception:
            thr = 1.0
    rng = np.random.default_rng(seed)
    best_inliers = None; best_slope = 0.0; best_int = 0.0
    k_sample = max(2, min(int(min_samples), n))
    for _ in range(int(max_trials)):
        idx = rng.choice(n, size=k_sample, replace=False)
        try:
            slope_i, int_i = np.polyfit(t[idx], s[idx], 1)
        except Exception:
            continue
        resid = np.abs(s - (slope_i * t + int_i))
        inlier = resid <= thr
        if best_inliers is None or int(inlier.sum()) > int(best_inliers.sum()):
            best_inliers = inlier
            best_slope, best_int = float(slope_i), float(int_i)
    if best_inliers is None or int(best_inliers.sum()) < 2:
        slope, intercept = np.polyfit(t, s, 1)
        resid = np.abs(s - (slope * t + intercept))
        return float(slope), float(intercept), (resid <= thr)
    try:
        slope, intercept = np.polyfit(t[best_inliers], s[best_inliers], 1)
        best_slope, best_int = float(slope), float(intercept)
    except Exception:
        pass
    return best_slope, best_int, best_inliers

# ---- _robust_line_fit  (main.py L498-564, verbatim) ----
def _robust_line_fit(t, s, method='ransac', thr=0.0, min_samples=3, max_trials=200):
    """Robust slope of s(t). method: ransac|theilsen|huber|ols.
    Returns dict(slope, intercept, inliers, method, thr, r2, se_slope, n).
    r2 / se_slope are computed on the INLIERS. sklearn is optional."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    n = int(t.size)
    if n < 2:
        return dict(slope=float('nan'), intercept=float('nan'),
                    inliers=np.zeros(n, bool), method='none', thr=0.0,
                    r2=float('nan'), se_slope=float('nan'), n=n)
    slope_ols, int_ols = np.polyfit(t, s, 1)
    resid_ols = s - (slope_ols * t + int_ols)
    mad = float(np.median(np.abs(resid_ols)))
    if thr is None or thr <= 0:
        thr = max(1.0, 2.5 * 1.4826 * (mad if mad > 0 else float(np.std(resid_ols))))
    slope = intercept = None; mask = None; method_used = 'ols'
    if method == 'ransac':
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            r = RANSACRegressor(estimator=LinearRegression(), residual_threshold=thr,
                                min_samples=max(2, int(min_samples)),
                                max_trials=int(max_trials), random_state=0)
            r.fit(t.reshape(-1, 1), s)
            slope = float(r.estimator_.coef_[0]); intercept = float(r.estimator_.intercept_)
            mask = r.inlier_mask_.astype(bool); method_used = 'ransac(sklearn)'
        except Exception:
            slope, intercept, mask = _ransac_line(t, s, thr, min_samples=int(min_samples),
                                                  max_trials=int(max_trials))
            method_used = 'ransac(pure)'
    elif method == 'theilsen':
        try:
            from sklearn.linear_model import TheilSenRegressor
            r = TheilSenRegressor(random_state=0); r.fit(t.reshape(-1, 1), s)
            slope = float(r.coef_[0]); intercept = float(r.intercept_)
            resid = s - (slope * t + intercept); mad2 = float(np.median(np.abs(resid)))
            mask = np.abs(resid) <= max(thr, 2.5 * 1.4826 * mad2); method_used = 'theilsen'
        except Exception:
            slope = None
    elif method == 'huber':
        try:
            from sklearn.linear_model import HuberRegressor
            r = HuberRegressor(); r.fit(t.reshape(-1, 1), s)
            slope = float(r.coef_[0]); intercept = float(r.intercept_)
            if hasattr(r, 'outliers_'):
                mask = ~r.outliers_.astype(bool)
            else:
                resid = s - (slope * t + intercept); mad2 = float(np.median(np.abs(resid)))
                mask = np.abs(resid) <= max(thr, 2.5 * 1.4826 * mad2)
            method_used = 'huber'
        except Exception:
            slope = None
    if slope is None:
        slope, intercept = float(slope_ols), float(int_ols)
        mask = np.abs(resid_ols) <= max(thr, 2.5 * 1.4826 * mad); method_used = 'ols'
    mask = np.asarray(mask, bool)
    ni = int(mask.sum())
    if ni >= 2:
        ti = t[mask]; si = s[mask]; pred = slope * ti + intercept
        ss_res = float(np.sum((si - pred) ** 2))
        ss_tot = float(np.sum((si - si.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        sxx = float(np.sum((ti - ti.mean()) ** 2)) + 1e-12
        se_slope = float(np.sqrt(ss_res / max(ni - 2, 1)) / np.sqrt(sxx))
    else:
        r2 = float('nan'); se_slope = float('nan')
    return dict(slope=float(slope), intercept=float(intercept), inliers=mask,
                method=method_used, thr=float(thr), r2=r2, se_slope=se_slope, n=n)

# ---- _fwhm_profile  (main.py L567-596, verbatim) ----
def _fwhm_profile(b, arclen_um, pk_i):
    """Linear-interpolated FWHM (µm) of an arclength profile around index pk_i.
    NaN if the profile is too flat."""
    b = np.asarray(b, float); s = np.asarray(arclen_um, float)
    if b.size < 3 or pk_i < 0 or pk_i >= b.size:
        return float('nan')
    base = float(np.nanmin(b)); peak = float(b[pk_i])
    if peak - base < 1e-12:
        return float('nan')
    half = base + 0.5 * (peak - base)
    li = pk_i
    while li > 0 and b[li - 1] >= half:
        li -= 1
    if li == 0:
        left_s = s[0]
    else:
        y0, y1 = b[li - 1], b[li]
        tt = (half - y0) / (y1 - y0 + 1e-12)
        left_s = s[li - 1] + tt * (s[li] - s[li - 1])
    ri = pk_i
    while ri < b.size - 1 and b[ri + 1] >= half:
        ri += 1
    if ri == b.size - 1:
        right_s = s[-1]
    else:
        y0, y1 = b[ri], b[ri + 1]
        tt = (half - y0) / (y1 - y0 + 1e-12)
        right_s = s[ri] + tt * (s[ri + 1] - s[ri])
    w = float(right_s - left_s)
    return w if w > 0 else float('nan')

# ---- file_index_same_file  (main.py L599-608, verbatim) ----
def file_index(boundaries, t):
    """Index of the file containing frame t. `boundaries` = [0, n0, n0+n1, ..., T]."""
    import bisect
    return max(0, bisect.bisect_right(list(boundaries), int(t)) - 1)


def same_file(boundaries, *ts):
    """True iff every frame index in `ts` lies inside the same file."""
    f = file_index(boundaries, ts[0])
    return all(file_index(boundaries, t) == f for t in ts[1:])

# ---- frame_provenance  (main.py L614-642, verbatim) ----
def make_root_map(T):
    """Identity map for a root stack: local i -> root i."""
    return np.arange(int(T), dtype=np.int64)


def compose_map(parent_map, local_idx):
    """New frame_map after extracting `local_idx` from a stack whose map is `parent_map`.
    Composition is what makes repeated extractions work: the map always points at the ROOT."""
    return np.asarray(parent_map, np.int64)[np.asarray(local_idx, np.int64)]


def invert_map(frame_map, root_T):
    """root index -> local index, or -1 where the root frame is absent. int32[root_T]."""
    fm = np.asarray(frame_map, np.int64)
    inv = np.full(int(root_T), -1, np.int32)
    ok = (fm >= 0) & (fm < int(root_T))
    inv[fm[ok]] = np.arange(fm.size, dtype=np.int32)[ok]
    return inv


def runs_from_root(root_idx):
    """Contiguous runs of a sorted root-index array -> [(r0, r1)] half-open (r1 exclusive)."""
    r = np.asarray(root_idx, np.int64)
    if r.size == 0:
        return []
    brk = np.where(np.diff(r) != 1)[0]
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [r.size - 1]])
    return [(int(r[s]), int(r[e]) + 1) for s, e in zip(starts, ends)]

# ---- _huber_affine  (main.py L645-663, verbatim) ----
def _huber_affine(x, y, iters=5, c=1.345):
    """Robust y ≈ a*x + b by IRLS (Huber). Returns (a, b, r2, n)."""
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 8:
        return (1.0, 0.0, 0.0, int(x.size))
    w = np.ones_like(x)
    a = b = 0.0
    for _ in range(iters):
        W = w.sum(); mx = (w * x).sum() / W; my = (w * y).sum() / W
        sxx = (w * (x - mx) ** 2).sum(); sxy = (w * (x - mx) * (y - my)).sum()
        a = sxy / (sxx + 1e-12); b = my - a * mx
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / (c * s); w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    r = y - (a * x + b)
    ss = np.sum((y - y.mean()) ** 2) + 1e-12
    return (float(a), float(b), float(1.0 - np.sum(r * r) / ss), int(x.size))

# ---- _qq_affine  (main.py L666-673, verbatim) ----
def _qq_affine(x_samp, y_samp, qlo=2.0, qhi=98.0, nq=25):
    """Distribution-based affine match (drift-insensitive): fit the quantiles of y against the
    quantiles of x. Returns (a, b)."""
    qs = np.linspace(qlo, qhi, nq)
    qx = np.percentile(np.asarray(x_samp, float), qs)
    qy = np.percentile(np.asarray(y_samp, float), qs)
    a, b, _, _ = _huber_affine(qx, qy)
    return float(a), float(b)

# ---- _monotone_lut  (main.py L676-687, verbatim) ----
def _monotone_lut(x_src, x_ref, K=12, qlo=0.5, qhi=99.9):
    """Quantile map taking SOURCE intensities onto REFERENCE intensities.
    Returns (xk, yk) knots with xk strictly increasing. K=2 -> affine-like; K>=64 -> histogram
    matching. Apply with np.interp(frame, xk, yk)."""
    q = np.linspace(qlo, qhi, int(max(2, K)))
    xk = np.percentile(np.asarray(x_src, float), q)
    yk = np.percentile(np.asarray(x_ref, float), q)
    # enforce strict monotonicity (dedupe ties, then cumulative max)
    for i in range(1, xk.size):
        if xk[i] <= xk[i - 1]: xk[i] = xk[i - 1] + 1e-6
    yk = np.maximum.accumulate(yk)
    return xk.astype(np.float64), yk.astype(np.float64)

# ---- _apply_lut  (main.py L690-693, verbatim) ----
def _apply_lut(a, xk, yk):
    """Clamped monotone LUT. Values outside the knot range are held at the end knots."""
    return np.interp(np.asarray(a, np.float32), xk, yk,
                     left=float(yk[0]), right=float(yk[-1])).astype(np.float32)

# ---- selected_frame_difference  (main.py L2184-2225, verbatim) ----
def selected_frame_difference(proj_THW, frame_indices, mode='consecutive',
                              ref_idx=None, rolling_window=11, dt=1,
                              sigma_xy=1.0, polarity='leading'):
    """Per-pair difference at each selected frame. Returns dict[t_k] -> (H,W).

    mode:
        'consecutive'      : D_k = I[t_k] - I[t_{k-1}]  (skip first)
        'fixed_ref'        : D_k = I[t_k] - I[ref_idx]
        'rolling_baseline' : D_k = I[t_k] - mean(I[t_k-w..t_k+w])
    polarity: 'signed' | 'absolute' | 'leading' (max(D,0)) | 'trailing' (max(-D,0))
    """
    from scipy.ndimage import gaussian_filter, uniform_filter1d
    proj = np.asarray(proj_THW, np.float32)
    T, H, W = proj.shape
    frames = sorted({int(f) for f in frame_indices if 0 <= int(f) < T})
    if not frames:
        return {}
    def _smooth(img):
        return gaussian_filter(img, sigma=float(sigma_xy), mode='nearest') if sigma_xy > 0 else img
    out = {}
    if mode == 'fixed_ref':
        if ref_idx is None: ref_idx = frames[0]
        ref = _smooth(proj[max(0, min(int(ref_idx), T - 1))])
        for t in frames:
            d = _smooth(proj[t]) - ref
            out[t] = _polar(d, polarity)
    elif mode == 'rolling_baseline':
        win = int(max(2, rolling_window))
        baseline = uniform_filter1d(proj, size=win, axis=0, mode='nearest')
        for t in frames:
            d = _smooth(proj[t]) - _smooth(baseline[t])
            out[t] = _polar(d, polarity)
    else:                                              # 'consecutive'
        dtv = int(max(1, dt))
        prev = None
        for t in frames:
            if prev is None:
                prev = t; continue
            d = _smooth(proj[t]) - _smooth(proj[max(0, prev - 0)])  # prev frame in selection
            out[t] = _polar(d, polarity)
            prev = t
    return out

# ---- _polar  (main.py L2228-2232, verbatim) ----
def _polar(d, polarity):
    if polarity == 'absolute':  return np.abs(d).astype(np.float32)
    if polarity == 'leading':   return np.maximum(d, 0).astype(np.float32)
    if polarity == 'trailing':  return np.maximum(-d, 0).astype(np.float32)
    return d.astype(np.float32)                        # 'signed'

# ---- arclength_profile  (main.py L2286-2305, verbatim) ----
def arclength_profile(D_pos_HW, centers, tangents, px_um, cross_half_px=6):
    """For each centerline sample, integrate ``D_pos_HW`` across the local
    perpendicular cross-section (full width = 2*cross_half_px + 1 px), giving
    a 1D arclength profile b(s) in (N,) units of D × px (raw integral)."""
    from scipy.ndimage import map_coordinates
    D = np.asarray(D_pos_HW, np.float32)
    H, W = D.shape
    N = centers.shape[0]
    L = 2 * int(cross_half_px) + 1
    s = np.linspace(-float(cross_half_px), float(cross_half_px), L)
    b = np.zeros(N, np.float32)
    for i in range(N):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        tx, ty = float(tangents[i, 0]), float(tangents[i, 1])
        px, py = -ty, tx                                # perpendicular unit
        xs = cx + s * px
        ys = cy + s * py
        prof = map_coordinates(D, np.vstack([ys, xs]), order=1, mode='nearest')
        b[i] = float(np.nansum(prof))
    return b

# ---- pulse_arclength_core  (main.py L2308-2362, verbatim) ----
def pulse_arclength_core(proj, frames, cl, n_pts, band_px, pxu, mode='consecutive',
                         ref_idx=None, sigma_xy=1.0, tau=1, rolling_window=11,
                         sat=None, polarity='leading'):
    """CANONICAL recon arclength pipeline — the single source shared by the recon window
    (`_compute_pulse_arclength`) and the manual batch (`_recon_block`).

    difference (`selected_frame_difference`) -> optional saturation NaN-mask ->
    b(s,t) = `arclength_profile(max(D,0), centers, tangents, pxu, band)` ->
    per-column s*(t)=arclen[argmax(b)], amp_at_star, fwhm.  NO prominence gate here (that is a
    fit-time choice, applied by `recon_velocity_from_arclength`).  Does not change the math."""
    proj = np.asarray(proj, np.float32)
    T = proj.shape[0]
    pm = proj
    if tau and int(tau) > 1:
        from scipy.ndimage import uniform_filter1d
        pm = uniform_filter1d(proj, size=int(tau), axis=0, mode='nearest')
    frames = sorted({int(f) for f in frames if 0 <= int(f) < T})
    Dmap = selected_frame_difference(pm, frames, mode=mode, ref_idx=ref_idx,
                                     rolling_window=rolling_window, dt=1,
                                     sigma_xy=float(sigma_xy), polarity=polarity)

    def _satmask(d, t):
        if sat is None or t >= np.asarray(sat).shape[0]:
            return d
        m = np.asarray(sat)[min(t, np.asarray(sat).shape[0] - 1)]
        if not m.any():
            return d
        out = d.astype(np.float32, copy=True); out[m] = np.nan
        return out

    present = [t for t in frames if Dmap.get(t) is not None]
    cl = np.asarray(cl, float) if cl is not None else None
    have_cl = cl is not None and len(cl) >= 2
    b_profiles = []; arclen_um = None; centers = tangents = None
    if have_cl:
        centers, tangents, arclen_px = _sample_along_polyline(cl, int(n_pts))
        arclen_um = arclen_px * float(pxu)
        for t in present:
            d = _satmask(Dmap[t], t)
            b = arclength_profile(np.maximum(np.nan_to_num(d, nan=0.0), 0),
                                  centers, tangents, float(pxu), cross_half_px=int(band_px))
            b_profiles.append(b)
    s_star_um = np.zeros(len(present), np.float32)
    amp_at_star = np.zeros(len(present), np.float32)
    fwhm_um = np.full(len(present), np.nan, np.float32)
    if have_cl and b_profiles:
        for k, b in enumerate(b_profiles):
            pk = int(np.argmax(b))
            s_star_um[k] = float(arclen_um[pk]); amp_at_star[k] = float(b[pk])
            fwhm_um[k] = _fwhm_profile(b, arclen_um, pk)
    B = np.stack(b_profiles, axis=1) if b_profiles else None
    return dict(present=present, mode=mode, Dmap=Dmap, sat_mask=_satmask,
                centers=centers, tangents=tangents, arclen_um=arclen_um,
                have_cl=have_cl, b_profiles=b_profiles, B=B, kymo=B,
                s_star_um=s_star_um, amp_at_star=amp_at_star, fwhm_um=fwhm_um)

# ---- recon_velocity_from_arclength  (main.py L2365-2398, verbatim) ----
def recon_velocity_from_arclength(res, fps, method='ransac', prom=0.0,
                                  thr=0.0, min_samples=3, max_trials=200):
    """LITERAL recon velocity for one interval, from `pulse_arclength_core` output.
    t = present/fps, s = s_star_um; keep columns with amp_at_star >= prom*max(amp) (prom=0 keeps
    all — the recon fit applies no prominence gate); fit s(t) with `_robust_line_fit`. FWHM is
    reported, not required. Returns slope/se/r2/inliers/keep/transport/direction and the arrays."""
    present = np.asarray(res.get('present', []), float)
    s = np.asarray(res.get('s_star_um', []), float)
    amp = np.asarray(res.get('amp_at_star', []), float)
    fps = float(fps)
    keep = np.zeros(present.size, bool)
    if present.size >= 2 and s.size == present.size:
        fin = np.isfinite(s)
        if amp.size == present.size and np.isfinite(amp).any() and prom and prom > 0:
            keep = fin & (amp >= float(prom) * float(np.nanmax(amp)))
        else:
            keep = fin
    t = present / max(fps, 1e-9)
    tk = t[keep]; sk = s[keep]; n_kept = int(keep.sum())
    slope = se = r2 = float('nan'); inl = np.zeros(present.size, bool); n_used = 0
    if n_kept >= 2:
        fit = _robust_line_fit(tk, sk, method=method, thr=thr,
                               min_samples=min_samples, max_trials=max_trials)
        slope = float(fit['slope']); r2 = float(fit['r2']); se = float(fit['se_slope'])
        im = np.asarray(fit['inliers'], bool); kidx = np.where(keep)[0]
        if kidx.size == im.size: inl[kidx[im]] = True
        n_used = int(im.sum())
    transport = float(sk[-1] - sk[0]) if n_kept >= 2 else float('nan')
    direction = ('antegrade' if (np.isfinite(slope) and slope > 0)
                 else ('retrograde' if np.isfinite(slope) else '-'))
    return dict(slope_um_s=slope, se=se, r2=r2, n_used=n_used, n_kept=n_kept,
                inliers=inl, keep=keep, t_keep=tk, s_keep=sk, direction=direction,
                transport_um=transport, s_star_um=s, amp=amp, fwhm=res.get('fwhm_um'),
                arclen_um=res.get('arclen_um'))

# ---- bolus_profile_metrics  (main.py L2401-2437, verbatim) ----
def bolus_profile_metrics(b_s, arclen_um):
    """Centroid (µm), FWHM width (µm), peak position (µm), total mass (sum)."""
    b = np.asarray(b_s, float); s = np.asarray(arclen_um, float)
    if b.size == 0 or not np.isfinite(b).any() or float(b.sum()) <= 0:
        return dict(centroid_s_um=float('nan'), width_um=float('nan'),
                    peak_s_um=float('nan'), total_mass=0.0, valid=False)
    centroid = float(np.sum(b * s) / np.sum(b))
    peak_i   = int(np.argmax(b))
    peak_s   = float(s[peak_i])
    # FWHM: linear-interp left/right crossings at half-max
    half = 0.5 * float(b[peak_i])
    above = b >= half
    if not above.any():
        width = float('nan')
    else:
        li = peak_i
        while li > 0 and above[li - 1]:
            li -= 1
        ri = peak_i
        while ri < b.size - 1 and above[ri + 1]:
            ri += 1
        # interp
        if li == 0:
            left_s = s[0]
        else:
            y0, y1 = b[li - 1], b[li]
            t = (half - y0) / (y1 - y0 + 1e-12)
            left_s = s[li - 1] + t * (s[li] - s[li - 1])
        if ri == b.size - 1:
            right_s = s[-1]
        else:
            y0, y1 = b[ri], b[ri + 1]
            t = (half - y0) / (y1 - y0 + 1e-12)
            right_s = s[ri] + t * (s[ri + 1] - s[ri])
        width = float(right_s - left_s)
    return dict(centroid_s_um=centroid, width_um=width, peak_s_um=peak_s,
                total_mass=float(np.sum(b)), valid=True)


# ============================================================================
#  WRAPPER  (authored; drives the verbatim kernels above)
# ============================================================================
import os, sys, json, math, argparse
import numpy as np

# ----------------------------------------------------------------------------
#  Optional per-frame BACKGROUND photometry (affine_bg) — self-contained
#  approximation of main.py's _frame_photometry('affine_bg').  DEFAULT: OFF.
#  See the big NOTE in main() for why the GUI's exact drift+photometry state
#  cannot be recovered from the JSON, and what turning this on does.
# ----------------------------------------------------------------------------
def _line_dilation_mask(H, W, fg_xy, half_px):
    """Boolean mask of pixels within `half_px` of the FG polyline (segment-wise
    point-to-segment distance).  Used to EXCLUDE the vessel from the background."""
    P = np.asarray(fg_xy, float)
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = np.full((H, W), np.inf, float)
    for i in range(len(P) - 1):
        a = P[i]; b = P[i + 1]; ab = b - a
        L2 = float(ab @ ab) + 1e-12
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2
        t = np.clip(t, 0.0, 1.0)
        px = a[0] + t * ab[0]; py = a[1] + t * ab[1]
        d2 = np.minimum(d2, (xx - px) ** 2 + (yy - py) ** 2)
    return d2 <= float(half_px) ** 2


def frame_photometry_affine_bg(base, t0, t1, fg_xy, fps, drift_d=None,
                               bg_exclude_px=18.0, border_px=6,
                               smooth_s=1.0, maxcorr=0.6, log=print):
    """Per-frame affine background correction for ONE segment, estimated on
    BACKGROUND pixels only (frame minus a dilated band around the FG flow line,
    minus a border) and low-passed in time; REFUSED (identity) if a_t/b_t track
    the vessel pulse g(t) beyond `maxcorr`.  Mirrors _frame_photometry('affine_bg').
    Returns dict(a[n], b[n], mode, rejected, t0, t1) — apply with (fr-b)/a."""
    from scipy.signal import savgol_filter
    T, H, W = base.shape
    t0 = int(t0); t1 = int(t1); n = t1 - t0
    ident = dict(a=None, b=None, mode='off', rejected=False, t0=t0, t1=t1, r_a=0.0, r_b=0.0)
    if n < 3:
        return ident
    bm = np.zeros((H, W), bool)
    bm[border_px:H - border_px, border_px:W - border_px] = True
    bm &= ~_line_dilation_mask(H, W, fg_xy, bg_exclude_px)
    ys, xs = np.nonzero(bm)
    if ys.size < 50:
        log("[fphoto] background mask < 50 px — identity."); return ident
    if ys.size > 4000:
        sel = np.linspace(0, ys.size - 1, 4000).astype(int); ys = ys[sel]; xs = xs[sel]

    def _bgpx(t):
        fr = np.asarray(base[t], np.float32)
        if drift_d is not None and t < len(drift_d):
            from scipy.ndimage import shift as nds
            fr = nds(fr, (-float(drift_d[t, 0]), -float(drift_d[t, 1])), order=1, mode='nearest')
        return fr[ys, xs]

    # FG (vessel) signal g(t) over the segment, for the anti-leak guard
    fgm = _line_dilation_mask(H, W, fg_xy, max(2.0, bg_exclude_px * 0.4))
    fys, fxs = np.nonzero(fgm)
    g = np.array([float(np.asarray(base[t], np.float32)[fys, fxs].mean()) for t in range(t0, t1)]) \
        if fys.size else np.zeros(n)
    # reference = quiet frames (band-mean FG below segment median); fallback first<=10
    quiet = np.where(g <= np.median(g))[0] + t0
    if quiet.size < 3:
        quiet = np.arange(t0, min(t1, t0 + 10))
    ref_px = np.median(np.stack([_bgpx(t) for t in quiet], 0), 0)

    a = np.ones(n); b = np.zeros(n)
    for k, t in enumerate(range(t0, t1)):
        a[k], b[k] = _qq_affine(ref_px, _bgpx(t))       # distribution-based affine
    a_raw = a.copy(); b_raw = b.copy(); ref_med = float(np.median(ref_px))
    win = max(3, int(round(float(smooth_s) * fps))) | 1
    if n > win >= 5:
        try:
            a = savgol_filter(a, win, 2); b = savgol_filter(b, win, 2)
        except Exception:
            pass

    def _detr(x):
        x = np.asarray(x, float); tt = np.arange(x.size)
        return x - np.polyval(np.polyfit(tt, x, 2), tt) if x.size >= 5 else (x - x.mean())
    gb = _detr(g); gbn = gb / (np.linalg.norm(gb) + 1e-9)

    def _leak(v_raw, scale):
        dv = _detr(v_raw); frac = abs(float(dv @ gbn)) / (abs(scale) + 1e-9)
        if frac < 0.03 or np.std(gb) < 1e-9 or np.std(dv) < 1e-9:
            return 0.0
        return abs(float(np.corrcoef(dv, gb)[0, 1]))
    r_a = _leak(a_raw, float(np.mean(a_raw))); r_b = _leak(b_raw, ref_med)
    if r_a > maxcorr or r_b > maxcorr:
        log(f"[fphoto] REFUSED seg [{t0},{t1}): |r_a|={r_a:.2f} |r_b|={r_b:.2f} > {maxcorr:.2f} "
            f"— identity, segment marked rejected.")
        return dict(ident, r_a=r_a, r_b=r_b, rejected=True)
    return dict(a=a, b=b, mode='affine_bg', rejected=False, t0=t0, t1=t1, r_a=r_a, r_b=r_b)


def frame_photo_apply(fr, t, fp):
    """Apply per-frame affine correction (identity if off/rejected). Ports _frame_photo_apply."""
    fr = np.asarray(fr, np.float32)
    if not fp or fp.get('mode', 'off') == 'off' or fp.get('rejected') or fp.get('a') is None:
        return fr
    k = int(t) - int(fp.get('t0', 0))
    if k < 0 or k >= len(fp['a']):
        return fr
    a = float(fp['a'][k]); b = float(fp['b'][k])
    return (fr - b) / max(a, 1e-6)


# ----------------------------------------------------------------------------
#  Frame-map reconstruction  (LOCAL final-video frame -> ROOT recording frame)
# ----------------------------------------------------------------------------
def build_frame_map(ver2, ver1=None, log=print):
    """Rebuild the final concatenated video's LOCAL->ROOT frame map from the
    segment records.  The final video is the ver2 segments concatenated IN LIST
    ORDER (idx 0,1,...); each contributes root frames [root_t0, root_t1).  This
    is exactly compose_map(ver1_map, ver2_local) — but ver2 already stores the
    composed root range, so we use that as the source of truth and (if ver1 is
    given) INDEPENDENTLY cross-check it against the ver1 concatenation.

    Returns:
        frame_map : int64[M]              local final-frame -> root frame
        seg_ranges: list of dicts with local (V2) range + root range + peak
    """
    s2 = ver2['segments']

    # --- optional cross-check: rebuild intermediate video V1 from ver1 ---
    if ver1 is not None:
        s1 = ver1['segments']; off = 0; v1map = []
        for s in s1:
            L = int(s['t1']) - int(s['t0'])
            v1map.append((off, off + L, int(s['t0']))); off += L
        V1_len = off

        def v1_to_root(x):
            for lo, hi, r0 in v1map:
                if lo <= x < hi:
                    return r0 + (x - lo)
            return None
        bad = 0
        for s in s2:
            r0m = v1_to_root(int(s['t0']))
            r1e = v1_to_root(int(s['t1']) - 1)
            r1m = (r1e + 1) if r1e is not None else None
            if not (r0m == int(s['root_t0']) and r1m == int(s['root_t1'])):
                bad += 1
                log(f"[frame_map] WARNING idx{s['idx']}: ver1-composition root "
                    f"[{r0m},{r1m}) != stored root [{s['root_t0']},{s['root_t1']})")
        log(f"[frame_map] intermediate (ver1) video = {V1_len} frames; "
            f"ver2<->ver1 cross-check: {'ALL CONSISTENT' if bad == 0 else f'{bad} MISMATCH(ES)'}")

    # --- authoritative map from ver2 root ranges, concatenated in list order ---
    off = 0; fmap = []; seg_ranges = []
    for s in s2:
        r0 = int(s['root_t0']); r1 = int(s['root_t1']); L = r1 - r0
        Lloc = int(s['t1']) - int(s['t0'])
        if L != Lloc:
            log(f"[frame_map] WARNING idx{s['idx']}: root length {L} != local length {Lloc}")
        seg_ranges.append(dict(idx=int(s['idx']), v2_t0=off, v2_t1=off + L,
                               root_t0=r0, root_t1=r1,
                               t_peak_root=int(s.get('t_peak_root', (r0 + r1) // 2)),
                               include=bool(s.get('include', True)),
                               note=str(s.get('note', ''))))
        fmap.extend(range(r0, r1)); off += L
    return np.asarray(fmap, np.int64), seg_ranges


# ----------------------------------------------------------------------------
#  Standalone per-segment analyzer  (mirrors main.py _recon_block, L23722-23874)
# ----------------------------------------------------------------------------
def _measure_peak_local(seg_t0, seg_t1, B, t_idx, I, frame_map):
    """Ported _measure_peak: peak = frame with most rectified bolus mass m(t)=sum_s max(B,0);
    fallback = argmax band-mean intensity over the window. Returns (tp_local, tp_root)."""
    t0 = int(seg_t0); t1 = int(seg_t1)
    def _to_root(i): return int(frame_map[int(i)]) if frame_map is not None else int(i)
    if B is not None and getattr(B, 'ndim', 0) == 2 and B.shape[1] >= 1 \
       and t_idx is not None and len(t_idx) == B.shape[1]:
        m = np.nansum(np.maximum(np.asarray(B, float), 0.0), axis=0)
        if np.isfinite(m).any():
            mv = m[np.isfinite(m)]
            med = float(np.median(mv)); spread = float(np.std(mv))
            prominence = (float(np.nanmax(m)) - med) / (spread + 1e-12)
            if spread > 1e-9 and prominence >= 1.0:
                col = int(np.nanargmax(m)); tp = int(np.asarray(t_idx)[col])
                tp = int(np.clip(tp, t0, t1 - 1)); return tp, _to_root(tp)
    if I is not None and getattr(I, 'ndim', 0) == 2 and I.shape[1] >= 1:
        prof_t = np.nanmean(np.asarray(I, float), axis=0)
        if np.isfinite(prof_t).any():
            tp = t0 + int(np.nanargmax(prof_t)); tp = int(np.clip(tp, t0, t1 - 1))
            return tp, _to_root(tp)
    tp = int((t0 + t1) // 2); return tp, _to_root(tp)


def recon_block(base, seg, prof, frame_map, drift_d=None, fp=None, log=print):
    """Apply the recon Lymphatic Pulse Analysis to ONE interval — the SAME kernels
    the GUI's _recon_block runs, so the numbers reproduce.  `seg` = dict(idx,t0,t1)
    with LOCAL (final-video) frames; `prof` = the profile dict from the JSON;
    `frame_map` maps local->root for real-time reporting; `drift_d` (T,2)=(dy,dx)
    shifts or None; `fp` = per-frame photometry dict or None (identity)."""
    fps = float(prof['fps']); px_um = float(prof['px_um'])
    t0 = int(seg['t0']); t1 = int(seg['t1'])
    N = int(prof['N']); band = int(prof['band'])
    diam_len_px = int(prof['diam_len_px']); diam_frac = float(prof['diam_frac'])
    fg_roi = np.asarray(prof['fg_roi'], float)

    if str(prof.get('step_mode', 'fixed_step')) == 'fixed_count':
        step = max(1, int(round((t1 - t0) / max(1, int(prof.get('n_target', 40))))))
    else:
        step = int(prof.get('step', 1))

    centers, tangents, arclen_px = _sample_along_polyline(fg_roi, N)
    arclen_um = arclen_px * px_um
    H, W = base.shape[1], base.shape[2]
    padc = diam_len_px + band + 4
    x0 = int(np.clip(np.floor(centers[:, 0].min()) - padc, 0, W - 1))
    x1 = int(np.clip(np.ceil(centers[:, 0].max()) + padc, 1, W))
    y0 = int(np.clip(np.floor(centers[:, 1].min()) - padc, 0, H - 1))
    y1 = int(np.clip(np.ceil(centers[:, 1].max()) + padc, 1, H))

    # drift-compensated, (optionally) photometry-corrected crop
    proj = np.empty((t1 - t0, y1 - y0, x1 - x0), np.float32)
    for k, t in enumerate(range(t0, t1)):
        fr = np.asarray(base[t], np.float32)
        if drift_d is not None and t < len(drift_d):
            from scipy.ndimage import shift as nds
            fr = nds(fr, (-float(drift_d[t, 0]), -float(drift_d[t, 1])), order=1, mode='nearest')
        fr = frame_photo_apply(fr, t, fp)               # identity if fp is None/off
        proj[k] = fr[y0:y1, x0:x1]
    cc = centers.copy(); cc[:, 0] -= x0; cc[:, 1] -= y0

    # intensity + diameter kymographs (N x T)
    I = _intensity_kymograph_band(proj, cc, band, px_um)
    D = _diameter_kymograph(proj, cc, tangents, diam_len_px, diam_frac, px_um)

    # VELOCITY: literal recon pulse pipeline (single source pulse_arclength_core)
    nfr = t1 - t0
    if str(prof.get('step_mode', 'fixed_step')) == 'fixed_count':
        _nt = max(2, int(prof.get('n_target', 40)))
        sel = np.unique(np.linspace(0, nfr - 1, min(_nt, nfr)).astype(int)).tolist()
    else:
        sel = list(range(0, nfr, max(1, step)))
    cl_crop = fg_roi.copy(); cl_crop[:, 0] -= x0; cl_crop[:, 1] -= y0
    _res = pulse_arclength_core(proj, sel, cl_crop, n_pts=N, band_px=band, pxu=px_um,
                                mode=str(prof['diff']), sigma_xy=0.0, sat=None)
    _vel = recon_velocity_from_arclength(_res, fps, method=str(prof['fit']),
                                         prom=float(prof['min_prom']))
    present = np.asarray(_res['present'], int)
    t_idx = present + t0
    B = _res['B']; s_star = _vel['s_star_um']; amp_star = _res['amp_at_star']
    gate = _vel['keep']; inliers = _vel['inliers']
    slope = _vel['slope_um_s']; se = _vel['se']; r2 = _vel['r2']
    n_gated = int(_vel['n_kept']); n_used = int(_vel['n_used'])
    direction = _vel['direction']; transport_um = _vel['transport_um']
    ejection_s = float((present[-1] - present[0]) / fps) if present.size >= 2 else float('nan')

    spacing_um = float(arclen_um[-1]) / max(N - 1, 1)
    vkymo, _kr, _kd = _velocity_from_kymograph(I, fps, spacing_um)
    vxcorr, lag_s, _xd = _velocity_from_xcorr(I[0], I[-1], float(arclen_um[-1]), fps)
    pm = _pulse_metrics_from_trace(np.nanmean(I, axis=0), fps)

    fwhm_um = float('nan'); mass = float('nan')
    if B is not None and B.shape[1] >= 1:
        col = int(np.argmax(np.nansum(np.maximum(B, 0.0), axis=0)))
        bpm = bolus_profile_metrics(np.maximum(B[:, col], 0.0), arclen_um)
        fwhm_um = float(bpm['width_um']); mass = float(bpm['total_mass'])

    Dmed = np.nanmedian(D, axis=0)
    EDD = float(np.nanmax(Dmed)) if np.isfinite(Dmed).any() else float('nan')
    ESD = float(np.nanmin(Dmed)) if np.isfinite(Dmed).any() else float('nan')
    EF = ((EDD ** 2 - ESD ** 2) / EDD ** 2) if (np.isfinite(EDD) and EDD > 0) else float('nan')
    dD = (EDD - ESD) if (np.isfinite(EDD) and np.isfinite(ESD)) else float('nan')
    pct_dD = (100.0 * dD / EDD) if (np.isfinite(dD) and EDD > 0) else float('nan')
    contr_v = relax_v = float('nan')
    if np.isfinite(Dmed).sum() >= 3:
        Df = np.nan_to_num(Dmed, nan=float(np.nanmean(Dmed)))
        dDdt = np.gradient(Df) * fps
        imin = int(np.nanargmin(Dmed))
        if imin >= 1: contr_v = float(np.nanmax(np.abs(dDdt[:imin + 1])))
        if imin <= len(Dmed) - 2: relax_v = float(np.nanmax(np.abs(dDdt[imin:])))

    n_diff_frames = int(len(t_idx)) if t_idx is not None else 0
    enough_diff = n_diff_frames >= int(prof['min_gated'])
    passed = bool(enough_diff and n_gated >= int(prof['min_gated'])
                  and np.isfinite(r2) and r2 >= float(prof['min_r2']))
    if not enough_diff:
        reason = f"n_diff_frames={n_diff_frames}<{prof['min_gated']}"
    else:
        reason = '' if passed else f"n_gated={n_gated}/{prof['min_gated']} r2={r2:.2f}/{prof['min_r2']}"
    quality = 'PASS' if passed else 'LOW'

    t_mid = int((t0 + t1) // 2)
    tp_local, tp_root_meas = _measure_peak_local(t0, t1, B, t_idx, I, frame_map)

    if frame_map is not None:
        root_t_idx = frame_map[np.asarray(t_idx, np.int64)] if t_idx is not None else None
        t0_root = int(frame_map[t0]); t1_root = int(frame_map[t1 - 1]) + 1
        t_peak_root = int(frame_map[tp_local])
    else:
        root_t_idx = np.asarray(t_idx) if t_idx is not None else None
        t0_root = t0; t1_root = t1; t_peak_root = int(tp_root_meas)

    log(f"[recon] seg#{int(seg['idx'])} local[{t0},{t1}) root[{t0_root},{t1_root}) "
        f"peak_root={t_peak_root} ({t_peak_root/fps:.2f}s) | v={slope:+.1f}±{se:.1f} um/s "
        f"r2={r2:.3f} inl={n_used}/{n_gated} | EF={EF:.3f} EDD={EDD:.1f} ESD={ESD:.1f} -> {quality}")

    return dict(idx=int(seg['idx']), t0=t0, t1=t1, t_peak=tp_local, t_mid=t_mid,
                t0_root=t0_root, t1_root=t1_root, t_peak_root=t_peak_root,
                t_peak_root_s=float(t_peak_root / fps),
                root_t_idx=root_t_idx, step=int(step), n_diff_frames=n_diff_frames,
                quality=quality, passed=passed, reason=reason, include=True,
                v_bolus=float(slope), slope=float(slope), se=float(se), r2=float(r2),
                n_gated=n_gated, n_used=n_used, inlier_frac=float(n_used / max(n_gated, 1)),
                direction=direction, transport_um=float(transport_um),
                ejection_s=float(ejection_s), spatial_fwhm_um=fwhm_um, bolus_mass=mass,
                vkymo=float(vkymo), vxcorr=float(vxcorr), lag_s=float(lag_s),
                EDD=EDD, ESD=ESD, EF=EF, dD=dD, pct_dD=pct_dD, contr_v=contr_v, relax_v=relax_v,
                freq_cpm_local=float(pm.get('freq_cpm', float('nan'))),
                fphoto_rejected=bool(fp.get('rejected')) if fp else False,
                # arrays for figures
                B=B, I=I, D=D, Dmed=Dmed, s_star=s_star, amp_star=amp_star,
                gate=gate, inliers=inliers, arclen_um=arclen_um, t_idx=t_idx,
                tt=(t_idx.astype(float) / fps if t_idx is not None else None))


# ----------------------------------------------------------------------------
#  Aggregation  (ports _aggregate_slopes / _agg core, without the Tk vars)
# ----------------------------------------------------------------------------
def agg_scalar(vals):
    """mean, SD(ddof=1), t-95% CI half-width, CV%, n over finite values (ports _agg)."""
    a = np.asarray(vals, float); a = a[np.isfinite(a)]; n = int(a.size)
    if n == 0:
        return dict(mean=float('nan'), sd=float('nan'), ci=float('nan'), cv=float('nan'), n=0)
    mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    ci = float('nan')
    if n > 1:
        try:
            from scipy.stats import t as _tdist
            ci = float(_tdist.ppf(0.975, n - 1) * sd / np.sqrt(n))
        except Exception:
            ci = float(1.96 * sd / np.sqrt(n))
    cv = float(100.0 * sd / abs(mean)) if abs(mean) > 1e-12 else float('nan')
    return dict(mean=mean, sd=sd, ci=ci, cv=cv, n=n)


def aggregate_velocity(rows):
    """Pool per-segment bolus velocity: unweighted mean±SD, 95% CI, CV, plus
    inverse-variance weighted mean and Cochran Q/I² (ports _aggregate_slopes core)."""
    recs = [r for r in rows if r.get('include', True) and np.isfinite(r.get('slope', np.nan))]
    b = np.array([r['slope'] for r in recs], float)
    se = np.array([r.get('se', np.nan) for r in recs], float)
    n = int(b.size)
    if n == 0:
        return dict(n=0)
    mean = float(b.mean()); sd = float(b.std(ddof=1)) if n > 1 else 0.0
    med = float(np.median(b))
    cv = float(100.0 * sd / abs(mean)) if abs(mean) > 1e-12 else float('nan')
    try:
        from scipy.stats import t as _t
        hw = float(_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float('nan')
    except Exception:
        hw = float(1.96 * sd / np.sqrt(n)) if n > 1 else float('nan')
    lo, hi = mean - hw, mean + hw
    wmean = wlo = whi = Q = I2 = float('nan')
    good = np.isfinite(se) & (se > 0)
    if good.sum() >= 2:
        w = np.where(good, 1.0 / np.maximum(se, 1e-12) ** 2, 0.0)
        wm = float(np.sum(w * b) / np.sum(w)); wmean = wm
        wse = float(np.sqrt(1.0 / np.sum(w))); wlo, whi = wm - 1.96 * wse, wm + 1.96 * wse
        Q = float(np.sum(w * (b - wm) ** 2)); df = n - 1
        I2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
    return dict(n=n, mean=mean, median=med, sd=sd, ci=(lo, hi), cv=cv,
                wmean=wmean, wci=(wlo, whi), Q=Q, I2=I2, slopes=b, ses=se)


# ----------------------------------------------------------------------------
#  Root-time physiology  (ports _root_timeline_stats core)
# ----------------------------------------------------------------------------
def root_timeline_stats(rows, frame_map, root_T, fps, scope='windows_only', doublet_s=1.0):
    """ROOT-time contraction physiology with an honest frequency denominator.
    peaks/ISI from SORTED root-time peaks; coverage = searched/root_T; frequency
    is reported as biased_cpm (events/searched-time), an exhaustive estimate
    (events/whole-recording), and the ISI-derived rate (60/ISI_mean)."""
    rec_s = (root_T / fps) if (root_T and fps > 0) else float('nan')
    rows_s = sorted(rows, key=lambda r: float(r.get('t_peak_root', 0)))
    peaks_s = np.array([float(r['t_peak_root']) / fps for r in rows_s], float)
    n_events = int(peaks_s.size)
    isi = np.diff(peaks_s) if peaks_s.size >= 2 else np.array([])
    isi_mean = float(np.mean(isi)) if isi.size else float('nan')
    isi_sd = float(np.std(isi, ddof=1)) if isi.size >= 2 else float('nan')
    isi_cv = float(100.0 * isi_sd / isi_mean) if (isi.size >= 2 and isi_mean > 0) else float('nan')
    freq_from_isi = float(60.0 / isi_mean) if (np.isfinite(isi_mean) and isi_mean > 0) else float('nan')
    searched = int(np.unique(np.asarray(frame_map, np.int64)).size) if frame_map is not None else 0
    coverage = (searched / root_T) if root_T else float('nan')
    biased_cpm = (60.0 * n_events / (searched / fps)) if searched else float('nan')
    exhaustive_cpm = (60.0 * n_events / rec_s) if (np.isfinite(rec_s) and rec_s > 0) else float('nan')
    doublets = [dict(a=rows_s[i]['idx'], b=rows_s[i + 1]['idx'], isi_s=float(isi[i]))
                for i in range(isi.size) if isi[i] < doublet_s]
    return dict(rec_s=rec_s, root_T=root_T, fps=fps, peaks_s=peaks_s, n_events=n_events,
                isi=isi, isi_mean=isi_mean, isi_sd=isi_sd, isi_cv=isi_cv,
                freq_from_isi=freq_from_isi, searched_frames=searched, coverage=coverage,
                biased_cpm=biased_cpm, exhaustive_cpm=exhaustive_cpm,
                doublets=doublets, doublet_s=doublet_s, rows_sorted=rows_s, scope=scope)


# ----------------------------------------------------------------------------
#  Output: CSV + summary + figures
# ----------------------------------------------------------------------------
_CSV_COLS = [
    'idx', 'quality', 'passed', 'reason',
    'v2_t0', 'v2_t1', 'n_frames',
    'root_t0', 'root_t1', 't_peak_root', 't_peak_root_s',
    'v_bolus_um_s', 'se_um_s', 'r2', 'n_gated', 'n_used', 'inlier_frac',
    'direction', 'transport_um', 'ejection_s',
    'vxcorr_um_s', 'vkymo_um_s', 'lag_s',
    'EDD_um', 'ESD_um', 'EF', 'dD_um', 'pct_dD', 'contr_v_um_s', 'relax_v_um_s',
    'spatial_fwhm_um', 'bolus_mass', 'freq_cpm_local', 'fphoto_rejected', 'step',
]


def write_csv(rows, seg_ranges, path):
    import csv
    rr = {r['idx']: r for r in seg_ranges}
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(_CSV_COLS)
        for r in sorted(rows, key=lambda x: x['idx']):
            sr = rr.get(r['idx'], {})
            w.writerow([
                r['idx'], r['quality'], int(r['passed']), r['reason'],
                sr.get('v2_t0', ''), sr.get('v2_t1', ''), r['t1'] - r['t0'],
                r['t0_root'], r['t1_root'], r['t_peak_root'], f"{r['t_peak_root_s']:.3f}",
                f"{r['v_bolus']:.4g}", f"{r['se']:.4g}", f"{r['r2']:.4g}",
                r['n_gated'], r['n_used'], f"{r['inlier_frac']:.3g}",
                r['direction'], f"{r['transport_um']:.4g}", f"{r['ejection_s']:.4g}",
                f"{r['vxcorr']:.4g}", f"{r['vkymo']:.4g}", f"{r['lag_s']:.4g}",
                f"{r['EDD']:.4g}", f"{r['ESD']:.4g}", f"{r['EF']:.4g}", f"{r['dD']:.4g}",
                f"{r['pct_dD']:.4g}", f"{r['contr_v']:.4g}", f"{r['relax_v']:.4g}",
                f"{r['spatial_fwhm_um']:.4g}", f"{r['bolus_mass']:.4g}",
                f"{r['freq_cpm_local']:.4g}", int(r['fphoto_rejected']), r['step'],
            ])


def make_figures(rows, agg_v, agg_ef, rt, prof, out_png):
    """Compact multi-panel QA figure: (a) per-segment velocity forest, (b) s*(t)
    regressions, (c) D(phase) ensemble with EDD/ESD, (d) root-time raster."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fps = float(prof['fps'])
    recs = [r for r in rows if np.isfinite(r.get('slope', np.nan))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=140)
    axA, axB, axC, axD = axes.ravel()

    # (a) forest
    if recs:
        yy = np.arange(len(recs))[::-1]
        for i, r in enumerate(recs):
            ci = 1.96 * (r['se'] if np.isfinite(r['se']) else 0.0)
            axA.errorbar([r['slope']], [yy[i]], xerr=[ci], fmt='o', ms=4, lw=1.0,
                         color=('#333' if r['passed'] else '#d62728'), capsize=2)
            axA.text(-0.02, yy[i], f"#{r['idx']}", transform=axA.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7)
        if np.isfinite(agg_v.get('mean', np.nan)):
            axA.axvline(agg_v['mean'], color='#0072B2', ls='--', lw=1.0)
            axA.axvspan(agg_v['ci'][0], agg_v['ci'][1], color='#0072B2', alpha=0.12)
    axA.set_yticks([]); axA.set_xlabel('per-segment bolus velocity (µm/s)')
    axA.set_title(f"(a) velocity forest — pooled {agg_v.get('mean', float('nan')):+.1f} "
                  f"± {agg_v.get('sd', float('nan')):.1f} µm/s (CV {agg_v.get('cv', float('nan')):.0f}%, n={agg_v.get('n', 0)})",
                  fontsize=8)
    axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)

    # (b) s*(t) regressions (recentered per segment)
    for r in recs:
        tt = r.get('tt'); s = r.get('s_star'); g = r.get('gate')
        if tt is None or s is None or g is None or not np.any(g):
            continue
        t0 = tt[g].min()
        axB.plot(tt[g] - t0, s[g], '.', ms=3, alpha=0.5)
        # regression line over the kept span
        if np.isfinite(r['slope']):
            xs = np.array([0, tt[g].max() - t0])
            b0 = np.median(s[g] - r['slope'] * (tt[g] - t0))
            axB.plot(xs, r['slope'] * xs + b0, '-', lw=1.0, alpha=0.7)
    axB.set_xlabel('time within segment (s)'); axB.set_ylabel('bolus position s*(µm)')
    axB.set_title('(b) s*(t) regressions (slope = velocity)', fontsize=8)
    axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

    # (c) D(phase) ensemble
    ph = np.linspace(0, 1, 100); segs = []
    for r in rows:
        Dm = r.get('Dmed')
        if Dm is None or np.isfinite(Dm).sum() < 3:
            continue
        segs.append(np.interp(ph, np.linspace(0, 1, len(Dm)),
                              np.nan_to_num(Dm, nan=float(np.nanmean(Dm)))))
    if segs:
        M = np.array(segs); mu = M.mean(0); sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1])
        axC.plot(ph, mu, color='#0072B2', lw=1.6)
        axC.fill_between(ph, mu - sd, mu + sd, color='#0072B2', alpha=0.2, lw=0)
        if np.isfinite(agg_ef.get('mean', np.nan)):
            axC.set_title(f"(c) D(phase) ensemble — EF={agg_ef['mean']:.3f}±{agg_ef['sd']:.3f} "
                          f"(2D-projected estimate)", fontsize=8)
    axC.set_xlabel('phase'); axC.set_ylabel('diameter (µm)')
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    # (d) root-time raster of contractions
    pk = rt['peaks_s']
    axD.vlines(pk, 0, 1, color='#D55E00', lw=1.4)
    for r in rt['rows_sorted']:
        axD.text(float(r['t_peak_root']) / fps, 1.03, f"#{r['idx']}", fontsize=6,
                 ha='center', va='bottom', rotation=90)
    axD.set_xlim(0, rt['rec_s']); axD.set_ylim(0, 1.25); axD.set_yticks([])
    axD.set_xlabel('real time in original recording (s)')
    axD.set_title(f"(d) contractions in root time — n={rt['n_events']}, "
                  f"ISI={rt['isi_mean']:.0f}±{rt['isi_sd']:.0f}s, "
                  f"coverage={100*rt['coverage']:.1f}%", fontsize=8)
    axD.spines['top'].set_visible(False); axD.spines['right'].set_visible(False)

    fig.suptitle("Lymphatic Pulse Flow Analyzer — standalone reproduction "
                 f"(fps={fps:g}, {prof['px_um']:g} µm/px, N={prof['N']}, band±{prof['band']}px, "
                 f"diff={prof['diff']}, fit={prof['fit']})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)


# ============================================================================
#  ENTRYPOINT
# ============================================================================
#  You can either (a) edit the CONFIG dict below and run `python
#  pulse_pipeline_standalone.py`, or (b) pass the same values on the command
#  line (CLI overrides CONFIG).  Run with -h to see all options.
# ----------------------------------------------------------------------------
CONFIG = dict(
    tiff="final_940.tif",                 # the doubly-concatenated video you have
    ver2="concate_set_ver2.json",         # segments + profile for the FINAL video
    ver1="concate_set_ver1.json",         # (optional) first extraction, for cross-check
    outdir="pulse_out",
    # --- original recording (for real-time / frequency).  30 fps * 28978 = 965.93 s ---
    root_frames=28978,
    root_fps=30.0,
    # --- corrections (see NOTE below) ---
    drift_npy=None,                       # optional path to (T,2)=(dy,dx) shift array; None = no drift
    photometry="off",                     # 'off' (exact, default) | 'affine_bg' (approx of GUI)
    # --- frequency reporting honesty ---
    scope="windows_only",                 # 'windows_only' | 'exhaustive'
    doublet_s=10.0,                        # peaks closer than this (root s) flagged as a doublet
)

_NOTE = """\
NOTE ON DRIFT & PHOTOMETRY
--------------------------
The profile stored in the JSON records that the GUI run used drift=True and
per-frame background photometry fphoto='affine_bg' (LUT photometry 'off').  The
NUMERICAL PIPELINE is reproduced here exactly, but the drift displacement array
and the photometry state are computed interactively inside SLIMGui and are NOT
saved in the JSON, so they cannot be recovered from these files alone.

  * drift_npy=None (default): no drift correction.  For a stable preparation the
    difference-based velocity is nearly unchanged; if you exported the GUI's
    drift array, pass it via --drift-npy (shape (T,2) = per-frame (dy,dx)).
  * photometry='off' (default): fully reproducible baseline; matches the profile
    for the file-level LUT (which was 'off').  photometry='affine_bg' turns on a
    SELF-CONTAINED approximation of the GUI's background-restricted affine
    correction (it re-estimates a quiet-frame reference and an anti-leak guard).
    It is an approximation because the exact FG-line width / registration ROI are
    not in the JSON; the anti-leak guard usually reduces it to identity on these
    short single-contraction segments anyway.

Everything downstream of the corrected crop — difference -> arclength b(s,t) ->
s*(t) -> RANSAC velocity, diameter -> EDD/ESD/EF, intensity -> dF/F0 & frequency
— is the SAME code path (pulse_arclength_core / recon_velocity_from_arclength /
_diameter_kymograph / _pulse_metrics_from_trace) the GUI's _recon_block runs.
"""


def _load_tiff(path):
    import tifffile
    arr = tifffile.memmap(path) if os.path.getsize(path) > 200 * 1024 * 1024 else tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise SystemExit(f"Expected a (T,H,W) stack, got shape {arr.shape}")
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Standalone lymphatic Pulse Flow Analyzer pipeline for a doubly-concatenated TIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_NOTE)
    ap.add_argument("--tiff", default=CONFIG["tiff"], help="final concatenated video (TIFF stack)")
    ap.add_argument("--ver2", default=CONFIG["ver2"], help="concate_set_ver2.json (segments + profile)")
    ap.add_argument("--ver1", default=CONFIG["ver1"], help="concate_set_ver1.json (optional cross-check)")
    ap.add_argument("--outdir", default=CONFIG["outdir"])
    ap.add_argument("--root-frames", type=int, default=CONFIG["root_frames"])
    ap.add_argument("--root-fps", type=float, default=CONFIG["root_fps"])
    ap.add_argument("--drift-npy", default=CONFIG["drift_npy"], help="(T,2) (dy,dx) shift array .npy")
    ap.add_argument("--photometry", choices=["off", "affine_bg"], default=CONFIG["photometry"])
    ap.add_argument("--scope", choices=["windows_only", "exhaustive"], default=CONFIG["scope"])
    ap.add_argument("--doublet-s", type=float, default=CONFIG["doublet_s"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "pulse_log.txt")
    _logf = open(log_path, "w")

    def log(msg):
        print(msg); _logf.write(str(msg) + "\n"); _logf.flush()

    log("=" * 78)
    log("Lymphatic Pulse Flow Analyzer — standalone reproduction")
    log("=" * 78)

    # --- load JSONs + profile ---
    ver2 = json.load(open(args.ver2))
    ver1 = json.load(open(args.ver1)) if (args.ver1 and os.path.exists(args.ver1)) else None
    prof = dict(ver2["profile"])
    prof["fps"] = float(prof.get("fps", args.root_fps))
    log(f"[profile] fps={prof['fps']} px_um={prof['px_um']} N={prof['N']} band=±{prof['band']}px "
        f"diff={prof['diff']} pos={prof.get('pos')} fit={prof['fit']} step_mode={prof['step_mode']} "
        f"step={prof['step']} n_target={prof['n_target']} min_prom={prof['min_prom']} "
        f"min_r2={prof['min_r2']} min_gated={prof['min_gated']} diam_len_px={prof['diam_len_px']} "
        f"diam_frac={prof['diam_frac']} photo={prof.get('photo')} fphoto={prof.get('fphoto')}")
    fg = np.asarray(prof["fg_roi"], float)
    log(f"[profile] FG flow line: {len(fg)} vertices, x[{fg[:,0].min():.0f},{fg[:,0].max():.0f}] "
        f"y[{fg[:,1].min():.0f},{fg[:,1].max():.0f}]")

    # --- frame map (local final-video frame -> root recording frame) ---
    frame_map, seg_ranges = build_frame_map(ver2, ver1, log=log)
    M = int(frame_map.size)
    log(f"[frame_map] final video = {M} frames; root span [{int(frame_map.min())},"
        f"{int(frame_map.max())}] ; original recording = {args.root_frames} frames / "
        f"{args.root_fps} fps = {args.root_frames/args.root_fps:.2f} s")

    # --- load the TIFF and reconcile length ---
    base = _load_tiff(args.tiff)
    T_tiff, H, W = base.shape
    log(f"[tiff] {args.tiff}: shape (T={T_tiff}, H={H}, W={W}), dtype={base.dtype}")
    if T_tiff != M:
        log(f"[tiff] WARNING: TIFF has {T_tiff} frames but the JSON implies {M}. "
            f"Segment ranges will be CLIPPED to the TIFF; check the extraction if |Δ|>a few frames.")
    if fg[:, 0].max() >= W or fg[:, 1].max() >= H:
        log(f"[tiff] WARNING: FG flow line extends outside the image ({W}x{H}). "
            f"Wrong TIFF / mismatched field of view?")

    # --- optional drift array ---
    drift_d = None
    if args.drift_npy and os.path.exists(args.drift_npy):
        drift_d = np.load(args.drift_npy)
        log(f"[drift] loaded {args.drift_npy} shape {drift_d.shape} (expects (T,2)=(dy,dx))")
    else:
        log("[drift] none (no drift correction) — see NOTE in --help")
    log(f"[photometry] {args.photometry}" + (" (self-contained approximation)"
        if args.photometry == "affine_bg" else " (file-level LUT off; exact)"))

    # --- run every included segment ---
    rows = []
    for sr in seg_ranges:
        if not sr["include"]:
            log(f"[recon] seg#{sr['idx']} excluded in JSON — skipped."); continue
        a = int(sr["v2_t0"]); b = min(int(sr["v2_t1"]), T_tiff)
        if b - a < 3 or a >= T_tiff:
            log(f"[recon] seg#{sr['idx']} range [{a},{b}) too short after clipping — skipped."); continue
        seg = dict(idx=sr["idx"], t0=a, t1=b)
        fp = None
        if args.photometry == "affine_bg":
            fp = frame_photometry_affine_bg(base, a, b, fg, prof["fps"], drift_d=drift_d, log=log)
        rows.append(recon_block(base, seg, prof, frame_map, drift_d=drift_d, fp=fp, log=log))

    if not rows:
        raise SystemExit("No segments produced a result — check the TIFF / JSON / flow line.")

    # --- aggregate + root-time physiology ---
    agg_v = aggregate_velocity(rows)
    agg_ef = agg_scalar([r["EF"] for r in rows])
    agg_edd = agg_scalar([r["EDD"] for r in rows]); agg_esd = agg_scalar([r["ESD"] for r in rows])
    agg_tr = agg_scalar([r["transport_um"] for r in rows])
    agg_fw = agg_scalar([r["spatial_fwhm_um"] for r in rows])
    rt = root_timeline_stats(rows, frame_map, args.root_frames, args.root_fps,
                             scope=args.scope, doublet_s=args.doublet_s)

    # --- write outputs ---
    csv_path = os.path.join(args.outdir, "segment_metrics.csv")
    write_csv(rows, seg_ranges, csv_path)
    fmap_path = os.path.join(args.outdir, "frame_map_local_to_root.csv")
    with open(fmap_path, "w") as f:
        f.write("local_frame,root_frame,root_time_s\n")
        for i, rf in enumerate(frame_map):
            f.write(f"{i},{int(rf)},{int(rf)/args.root_fps:.4f}\n")

    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        def out(s=""):
            f.write(s + "\n")
        out("Lymphatic Pulse Flow Analyzer — standalone reproduction")
        out("=" * 70)
        out(f"TIFF                 : {args.tiff}  (T={T_tiff}, {H}x{W})")
        out(f"Final video length   : {M} frames (JSON) ; original {args.root_frames} frames "
            f"/ {args.root_fps} fps = {args.root_frames/args.root_fps:.2f} s")
        out(f"Segments analyzed    : {len(rows)} (of {len(seg_ranges)} in ver2)")
        out(f"Drift / photometry   : drift={'array' if drift_d is not None else 'none'} ; "
            f"photometry={args.photometry}")
        out("")
        out("PER-SEGMENT (velocity from RANSAC of s*(t); EF from projected diameter kymograph)")
        out("-" * 70)
        hdr = (f"{'idx':>3} {'quality':>7} {'root peak(s)':>12} {'v(µm/s)':>10} {'±se':>7} "
               f"{'r2':>5} {'dir':>10} {'EF':>6} {'EDD':>6} {'ESD':>6} {'transp':>7}")
        out(hdr)
        for r in sorted(rows, key=lambda x: x["idx"]):
            out(f"{r['idx']:>3} {r['quality']:>7} {r['t_peak_root_s']:>12.2f} "
                f"{r['v_bolus']:>+10.1f} {r['se']:>7.1f} {r['r2']:>5.2f} {r['direction']:>10} "
                f"{r['EF']:>6.3f} {r['EDD']:>6.1f} {r['ESD']:>6.1f} {r['transport_um']:>7.1f}")
        out("")
        out("POOLED (biological replicate = segment)")
        out("-" * 70)
        if agg_v.get("n", 0):
            out(f"bolus velocity   : {agg_v['mean']:+.1f} ± {agg_v['sd']:.1f} µm/s  "
                f"(95% CI [{agg_v['ci'][0]:.1f}, {agg_v['ci'][1]:.1f}], CV {agg_v['cv']:.0f}%, n={agg_v['n']})")
            if np.isfinite(agg_v.get("wmean", np.nan)):
                out(f"  inv-var weighted: {agg_v['wmean']:+.1f} µm/s "
                    f"[{agg_v['wci'][0]:.1f}, {agg_v['wci'][1]:.1f}]  (Q={agg_v['Q']:.1f}, I²={agg_v['I2']:.0f}%)")
        out(f"EF (2D est.)     : {agg_ef['mean']:.3f} ± {agg_ef['sd']:.3f}  (n={agg_ef['n']})")
        out(f"EDD / ESD (µm)   : {agg_edd['mean']:.1f} ± {agg_edd['sd']:.1f}  /  "
            f"{agg_esd['mean']:.1f} ± {agg_esd['sd']:.1f}")
        out(f"transport (µm)   : {agg_tr['mean']:.1f} ± {agg_tr['sd']:.1f}")
        out(f"bolus FWHM (µm)  : {agg_fw['mean']:.1f} ± {agg_fw['sd']:.1f}")
        out("")
        out("ROOT-TIME CONTRACTION PHYSIOLOGY (real time in the original recording)")
        out("-" * 70)
        out(f"contractions (n)      : {rt['n_events']}")
        out(f"peaks (root s)        : " + ", ".join(f"{p:.1f}" for p in rt['peaks_s']))
        out(f"inter-contraction ISI : {rt['isi_mean']:.1f} ± {rt['isi_sd']:.1f} s "
            f"(CV {rt['isi_cv']:.0f}%, n={max(rt['n_events']-1,0)})")
        out(f"rate from ISI         : {rt['freq_from_isi']:.2f} contractions/min  "
            f"(= 60 / mean ISI)")
        out(f"coverage              : {rt['searched_frames']}/{rt['root_T']} = "
            f"{100*rt['coverage']:.2f}% of the recording was extracted")
        out(f"biased rate           : {rt['biased_cpm']:.2f} cpm  (events / searched time — "
            f"selection-biased upward, NOT a frequency)")
        out(f"if exhaustive         : {rt['exhaustive_cpm']:.2f} cpm  (events / whole {rt['rec_s']:.0f}s "
            f"recording — only valid if you searched the WHOLE video)")
        if rt['doublets']:
            out(f"doublets (<{rt['doublet_s']:.0f}s): " +
                ", ".join(f"#{d['a']}-#{d['b']} ({d['isi_s']:.1f}s)" for d in rt['doublets']))
        out("")
        out("Outputs: segment_metrics.csv, frame_map_local_to_root.csv, "
            "pulse_qa.png, pulse_log.txt")
        out("")
        out("Caveat: EF here is a 2D-projected intensity-FWHM ESTIMATE, not a volumetric")
        out("measurement — state this in any figure caption (as the GUI does).")

    log("")
    log(open(summary_path).read())

    if not args.no_figures:
        try:
            png = os.path.join(args.outdir, "pulse_qa.png")
            make_figures(rows, agg_v, agg_ef, rt, prof, png)
            log(f"[fig] wrote {png}")
        except Exception as e:
            log(f"[fig] figure generation failed: {e!r}")

    log(f"\nDone. All outputs in: {os.path.abspath(args.outdir)}")
    _logf.close()
    return rows


if __name__ == "__main__":
    main()
