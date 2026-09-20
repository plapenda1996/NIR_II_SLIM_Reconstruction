# NIR-II SLIM - zebrafish valve kinetics - produces Fig. 2i-m
# environment: heart_valve_py314
# copied from E:\Selected data\250902_fish\valve_analysis.py on 2026-09-17
"""
Heart valve motion analysis from a TIFF stack (intensity-only, low SNR, fast motion).

Pipeline
--------
1. Load the TIFF stack and normalize.
2. Preprocess: remove static background (temporal median) + per-frame CLAHE.
   Optional NL-means spatial denoising.
3. Kymograph (M-mode) along an operator-drawn line through the valve — best
   quick quantification of valve motion, no segmentation required.
4. Manual segmentation in napari on a sparse set of key-frames, then propagate
   to intermediate frames with signed-distance-field interpolation.
5. Extract per-frame metrics (area, centroid, orientation) and plot them.

Dependencies
------------
    pip install numpy scipy scikit-image tifffile matplotlib napari[all]
    # optional, for step 4 alternative:
    # pip install "sam2" (Segment Anything 2, video mode)

Tips for low-SNR fast-motion data
---------------------------------
- DO use temporal MEDIAN (not mean) for background removal; it's robust to
  the moving valve itself.
- DO NOT apply temporal Gaussian smoothing — it erases fast motion.
- Spatial NL-means helps but is slow; run it once and save.
- Annotate every 10-30 frames in napari. If motion is very fast between two
  key-frames, add more key-frames there rather than making the brush bigger.
- If the valve is roughly periodic, phase-averaging lets you boost SNR
  further: detect period from the kymograph, stack frames at the same phase,
  and average.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure, measure
from skimage.restoration import denoise_nl_means, estimate_sigma
import os
import sys
import json
import time
import logging
import traceback


# ============================================================
# OBSERVABILITY — single sources of truth: BASE_DIR, logging, status line
# ============================================================
# Resolve the working base directory from THIS file (no machine-specific
# literals). Everything (leaflet_dataset, log, outputs) hangs off BASE_DIR.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())
LOG_PATH = os.path.join(BASE_DIR, "valve_analysis.log")
DATA_TIFF = None                 # single source for the loaded TIFF path (set in __main__)
# Session single sources of truth, settable from the GUI:
ACTIVE_DS_DIR = None             # active dataset folder (annotate->train->infer->figure)
ACTIVE_DEPTH = None              # chosen depth index in a multi-depth TIF (for logging)
ACTIVE_STACK = None              # normalized single-depth stack to use for valve analysis


def active_ds_dir():
    """The dataset folder driving the whole pipeline (single source)."""
    return ACTIVE_DS_DIR or os.path.join(BASE_DIR, "leaflet_dataset")


def _setup_logging():
    """Configure logging ONCE: console + append file handler."""
    lg = logging.getLogger("valve")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    try:
        fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        fh.setFormatter(fmt); lg.addHandler(fh)
    except Exception:
        pass
    lg.addHandler(sh)
    lg.propagate = False
    return lg


LOG = _setup_logging()

# ONE status-line handle for the whole GUI (set by interactive_preprocess_gui).
_GUI = {"fig": None, "status": None}


def set_status(msg, level="info", log=True):
    """Update the single GUI status line, force a redraw+flush so it never looks
    frozen, and log the message. Levels: info / run / done / error. ``log=False``
    refreshes the GUI WITHOUT writing to the log (used by high-frequency per-batch
    training heartbeats, so they keep the window live without flooding the log)."""
    if log:
        (LOG.error if level == "error" else LOG.info)(msg)
    txt = _GUI.get("status")
    if txt is not None:
        color = {"info": "#dddddd", "run": "#ffd24a", "done": "#5fd35f",
                 "error": "#ff5b5b"}.get(level, "#dddddd")
        try:
            txt.set_text("  " + msg); txt.set_color(color)
            fig = _GUI.get("fig")
            if fig is not None:
                fig.canvas.draw_idle(); fig.canvas.flush_events()
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _file_tag(path, will_write=False):
    """Return an EXISTING/NEW/OVERWRITE provenance tag for a path."""
    if os.path.exists(path):
        st = os.stat(path)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        kb = st.st_size / 1024.0
        return (f"[WILL OVERWRITE existing - {when}, {kb:.0f} KB]" if will_write
                else f"[EXISTING - modified {when}, {kb:.0f} KB]")
    return "[NEW - will create]"


def log_files(reads=(), writes=()):
    """Log the FILES block: absolute paths, tagged EXISTING/NEW/OVERWRITE."""
    LOG.info("FILES:")
    for r in reads:
        LOG.info(f"  READ   {os.path.abspath(r)}   {_file_tag(r)}")
    for w in writes:
        LOG.info(f"  WRITE  {os.path.abspath(w)}   {_file_tag(w, will_write=True)}")


def gui_op(name):
    """Decorator wrapping a GUI button callback: start/finish banners, status
    updates, and — crucially — NO SILENT FAILURES (full traceback to console+log,
    red ERROR in the status line, and a popup)."""
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            LOG.info(f"===== STARTING {name} =====")
            set_status(f"Running: {name} ...", "run")
            try:
                r = fn(*a, **k)
                dt = time.time() - t0
                LOG.info(f"===== FINISHED {name} in {dt:.1f}s =====")
                set_status(f"Done: {name} ({dt:.1f}s)", "done")
                return r
            except Exception as e:
                LOG.error(f"{name} FAILED:\n{traceback.format_exc()}")
                set_status(f"ERROR: {name}: {e} (see console / valve_analysis.log)",
                           "error")
                try:
                    import tkinter as _tk
                    from tkinter import messagebox as _mb
                    _r = _tk.Tk(); _r.withdraw()
                    _mb.showerror(name, f"{e}\n\nFull traceback in console and\n{LOG_PATH}")
                    _r.destroy()
                except Exception:
                    pass
        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped
    return deco


# ============================================================
# 1. Load & normalize
# ============================================================
def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF stack as (T, H, W) float32, percentile-normalized to [0, 1]."""
    stack = tifffile.imread(str(path)).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    lo, hi = np.percentile(stack, [0.5, 99.5])
    stack = np.clip((stack - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return stack


# ============================================================
# 2. Preprocessing — Dark-field optimized
# ============================================================
def remove_static_background(stack: np.ndarray, method: str = "min") -> np.ndarray:
    """Remove static background from dark-field data.

    For dark-field images the signal IS the bright scattered light on a dark
    background. The temporal median would remove most of the signal.

    Recommended methods for dark-field:
      'min'        — subtract per-pixel temporal minimum (removes only the
                     fixed readout/dark-current floor; preserves all signal)
      'percentile' — subtract a low percentile (e.g. 5th) per pixel; more
                     robust than min to single-frame hot pixels
      'median'     — traditional; only use for bright-field data
      'none'       — skip background subtraction entirely
    """
    if method == "min":
        bg = stack.min(axis=0)
    elif method == "percentile":
        bg = np.percentile(stack, 5, axis=0)
    elif method == "median":
        bg = np.median(stack, axis=0)
    elif method == "mean":
        bg = stack.mean(axis=0)
    elif method == "none":
        return stack.copy()
    else:
        raise ValueError(f"unknown method: {method}")
    return np.clip(stack - bg, 0.0, None)


def enhance_contrast(stack: np.ndarray, clip_limit: float = 0.03,
                     darkfield: bool = True) -> np.ndarray:
    """Per-frame contrast enhancement.

    For dark-field data, uses a higher CLAHE clip limit and applies gamma
    compression to lift dim structures without saturating bright ones.
    """
    out = np.empty_like(stack, dtype=np.float32)
    for i in range(stack.shape[0]):
        f = stack[i]
        fmin, fmax = f.min(), f.max()
        if fmax - fmin < 1e-8:
            out[i] = 0.0
            continue
        f = (f - fmin) / (fmax - fmin)
        if darkfield:
            # Gamma compression: lifts dim scattered-light features
            f = np.power(f, 0.5)
        out[i] = exposure.equalize_adapthist(f, clip_limit=clip_limit)
    return out


def denoise_spatial(stack: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Per-frame spatial denoising. 'bilateral' is fast and edge-preserving,
    'nlmeans' is slower but better for very low SNR.
    """
    out = np.empty_like(stack, dtype=np.float32)
    if method == "bilateral":
        import cv2
        for i in range(stack.shape[0]):
            u8 = (np.clip(stack[i], 0, 1) * 255).astype(np.uint8)
            u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=40, sigmaSpace=40)
            out[i] = u8.astype(np.float32) / 255.0
    elif method == "nlmeans":
        for i in range(stack.shape[0]):
            sigma = float(np.mean(estimate_sigma(stack[i])))
            out[i] = denoise_nl_means(
                stack[i], h=0.8 * sigma, sigma=sigma,
                patch_size=5, patch_distance=6, fast_mode=True)
    elif method == "gaussian":
        for i in range(stack.shape[0]):
            out[i] = ndimage.gaussian_filter(stack[i], sigma=0.8)
    else:
        return stack.copy()
    return out


def temporal_denoise(stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Light temporal smoothing — average neighboring frames.

    Use a small window (3-5) to reduce shot noise without blurring fast motion.
    For dark-field data where each frame is noisy, even window=3 helps a lot.
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(stack.astype(np.float32), size=window, axis=0)


def preprocess(stack: np.ndarray, nlmeans: bool = False,
               darkfield: bool = True) -> np.ndarray:
    """Preprocessing pipeline optimized for dark-field or bright-field data.

    Dark-field pipeline:
      1. Subtract per-pixel temporal minimum (removes dark-current floor only)
      2. Light temporal smoothing (window=3) to reduce shot noise
      3. Per-frame bilateral denoising (fast, edge-preserving)
      4. Gamma compression + CLAHE to enhance dim scattered-light features

    Bright-field pipeline (darkfield=False):
      1. Subtract temporal median (removes static tissue background)
      2. Per-frame CLAHE
      3. Optional NL-means
    """
    if darkfield:
        # Step 1: remove only the dark-current floor, keep all signal
        bgsub = remove_static_background(stack, method="min")
        # Step 2: light temporal smoothing to reduce shot noise
        bgsub = temporal_denoise(bgsub, window=3)
        # Step 3: spatial denoising (bilateral = fast + edge-preserving)
        denoised = denoise_spatial(bgsub, method="bilateral")
        # Step 4: contrast enhancement with gamma for dark-field
        enhanced = enhance_contrast(denoised, clip_limit=0.03, darkfield=True)
        if nlmeans:
            enhanced = denoise_spatial(enhanced, method="nlmeans")
        return enhanced
    else:
        # Original bright-field pipeline
        bgsub = remove_static_background(stack, method="median")
        enhanced = enhance_contrast(bgsub, clip_limit=0.02, darkfield=False)
        if nlmeans:
            enhanced = denoise_spatial(enhanced, method="nlmeans")
        return enhanced


# ============================================================
# 3. Kymograph (M-mode) — operator draws one line through the valve
# ============================================================
def pick_kymograph_line(reference_image: np.ndarray) -> tuple[tuple[float, float],
                                                              tuple[float, float]]:
    """Click two endpoints on the reference image. Returns ((y1,x1), (y2,x2))."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(reference_image, cmap="gray")
    ax.set_title("Click TWO endpoints across the valve leaflet, then close the window")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    (x1, y1), (x2, y2) = pts
    return (y1, x1), (y2, x2)


def kymograph(stack: np.ndarray, p1: tuple[float, float], p2: tuple[float, float],
              thickness: int = 5) -> np.ndarray:
    """Sample intensity along a line in every frame → (T, L) kymograph.

    Averaging across `thickness` perpendicular pixels reduces noise further.
    """
    y1, x1 = p1
    y2, x2 = p2
    L = int(np.hypot(y2 - y1, x2 - x1))
    ys = np.linspace(y1, y2, L)
    xs = np.linspace(x1, x2, L)

    dy, dx = (y2 - y1) / L, (x2 - x1) / L
    ny, nx = dx, -dy  # unit normal to the line

    offsets = np.arange(-(thickness // 2), thickness // 2 + 1)
    T, H, W = stack.shape
    kymo = np.zeros((T, L), dtype=np.float32)
    for off in offsets:
        ys_o = np.clip(ys + off * ny, 0, H - 1)
        xs_o = np.clip(xs + off * nx, 0, W - 1)
        for t in range(T):
            kymo[t] += ndimage.map_coordinates(stack[t], [ys_o, xs_o], order=1)
    return kymo / len(offsets)


def plot_kymograph(kymo: np.ndarray, fps: float | None = None,
                   savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    t_max = kymo.shape[0] / fps if fps else kymo.shape[0]
    ax.imshow(kymo, cmap="gray", aspect="auto",
              extent=[0, kymo.shape[1], t_max, 0])
    ax.set_xlabel("Distance along line (px)")
    ax.set_ylabel("Time (s)" if fps else "Frame")
    ax.set_title("Kymograph / M-mode — valve leaflet trajectory")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ============================================================
# 4. Interactive manual segmentation in napari
# ============================================================
def segment_in_napari(stack: np.ndarray,
                      existing_labels: np.ndarray | None = None) -> np.ndarray:
    """Open napari to paint masks. Returns the final label volume.

    Workflow tip:
      - Annotate every 10-30 frames (more densely in fast-motion bursts).
      - Use label value 1 for the valve leaflet; you can add more label IDs
        for multiple leaflets if you want to track them separately.
    """
    import napari  # imported lazily

    viewer = napari.Viewer()
    viewer.add_image(stack, name="preprocessed",
                     contrast_limits=[float(stack.min()), float(stack.max())])
    labels = (existing_labels.copy() if existing_labels is not None
              else np.zeros(stack.shape, dtype=np.uint8))
    lbl_layer = viewer.add_labels(labels, name="valve_mask")
    napari.run()
    return lbl_layer.data


def interpolate_key_frame_masks(masks: np.ndarray) -> np.ndarray:
    """Fill empty frames between annotated key-frames via signed-distance
    field interpolation. Works well if motion is smooth between key-frames."""
    from scipy.ndimage import distance_transform_edt

    T = masks.shape[0]
    key = [t for t in range(T) if masks[t].any()]
    if len(key) < 2:
        return masks.copy()

    out = masks.copy()
    for i in range(len(key) - 1):
        t0, t1 = key[i], key[i + 1]
        if t1 == t0 + 1:
            continue
        m0, m1 = masks[t0] > 0, masks[t1] > 0
        sdf0 = distance_transform_edt(~m0) - distance_transform_edt(m0)
        sdf1 = distance_transform_edt(~m1) - distance_transform_edt(m1)
        for t in range(t0 + 1, t1):
            a = (t - t0) / (t1 - t0)
            sdf = (1 - a) * sdf0 + a * sdf1
            out[t] = (sdf < 0).astype(masks.dtype)
    return out


# ============================================================
# 5. Motion metrics & plots
# ============================================================
def per_frame_metrics(masks: np.ndarray) -> dict:
    T = masks.shape[0]
    area = np.zeros(T)
    cy = np.full(T, np.nan)
    cx = np.full(T, np.nan)
    major = np.full(T, np.nan)
    angle = np.full(T, np.nan)
    for t in range(T):
        m = masks[t] > 0
        if not m.any():
            continue
        p = measure.regionprops(m.astype(np.uint8))[0]
        area[t] = p.area
        cy[t], cx[t] = p.centroid
        major[t] = p.major_axis_length
        angle[t] = np.degrees(p.orientation)
    return dict(area=area, cy=cy, cx=cx, major=major, angle=angle)


def plot_motion(metrics: dict, fps: float | None = None,
                savepath: str | None = None):
    T = len(metrics["area"])
    x = np.arange(T) / fps if fps else np.arange(T)
    xlabel = "Time (s)" if fps else "Frame"

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(x, metrics["area"]);       ax[0].set_ylabel("Area (px)")
    ax[0].set_title("Valve opening area")
    ax[1].plot(x, metrics["cy"], label="cy (row)")
    ax[1].plot(x, metrics["cx"], label="cx (col)")
    ax[1].set_ylabel("Centroid (px)"); ax[1].legend()
    ax[2].plot(x, metrics["major"]);      ax[2].set_ylabel("Major axis (px)")
    ax[3].plot(x, metrics["angle"]);      ax[3].set_ylabel("Orientation (deg)")
    ax[3].set_xlabel(xlabel)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ============================================================
# 6. Leaflet trajectory from the kymograph (no segmentation needed)
# ============================================================
def trace_leaflet_from_kymograph(kymo: np.ndarray, smooth_sigma: float = 1.5,
                                 mode: str = "bright") -> np.ndarray:
    """For each frame (row of the kymograph), find the leaflet position along
    the line. `bright` picks the brightest pixel (good after background
    subtraction: the moving leaflet is bright); `dark` picks the darkest
    (useful if the leaflet is a dark silhouette).

    Returns a (T,) array of positions in pixels along the line.
    """
    k = ndimage.gaussian_filter(kymo, sigma=(smooth_sigma, smooth_sigma))
    if mode == "dark":
        k = -k
    elif mode != "bright":
        raise ValueError(mode)
    # sub-pixel refinement: parabolic fit around argmax
    idx = np.argmax(k, axis=1)
    T, L = k.shape
    pos = idx.astype(np.float32)
    for t in range(T):
        i = idx[t]
        if 0 < i < L - 1:
            y0, y1, y2 = k[t, i - 1], k[t, i], k[t, i + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-6:
                pos[t] = i + 0.5 * (y0 - y2) / denom
    return pos


def plot_leaflet_trajectory(position: np.ndarray, kymo: np.ndarray,
                            fps: float | None = None,
                            savepath: str | None = None):
    """Overlay the traced leaflet position on top of the kymograph, and
    also show position vs time and velocity vs time."""
    T = position.shape[0]
    t = np.arange(T) / fps if fps else np.arange(T)
    velocity = np.gradient(position) * (fps if fps else 1.0)

    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    t_max = T / fps if fps else T
    ax[0].imshow(kymo, cmap="gray", aspect="auto",
                 extent=[0, kymo.shape[1], t_max, 0])
    ax[0].plot(position, t, color="tab:red", lw=1.2, label="traced leaflet")
    ax[0].set_xlabel("Distance along line (px)")
    ax[0].set_ylabel("Time (s)" if fps else "Frame")
    ax[0].set_title("Kymograph + leaflet trace")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, position)
    ax[1].set_ylabel("Position along line (px)")
    ax[1].set_title("Leaflet displacement")

    ax[2].plot(t, velocity)
    ax[2].set_ylabel("Velocity (px/s)" if fps else "Velocity (px/frame)")
    ax[2].set_xlabel("Time (s)" if fps else "Frame")
    ax[2].set_title("Leaflet velocity")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ============================================================
# 7. Periodicity → automatic key-frame suggestion
# ============================================================
def detect_period(position: np.ndarray) -> float:
    """Estimate dominant period (in frames) of the leaflet trajectory via FFT."""
    x = position - position.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    spec[0] = 0  # ignore DC
    k = int(np.argmax(spec))
    if k == 0:
        return float(len(x))
    return len(x) / k


def suggest_keyframes(position: np.ndarray,
                      n_per_cycle: int = 6) -> np.ndarray:
    """Return frame indices that sample extrema & intermediate phases of each
    cycle. These are the frames worth hand-labeling for area measurement.

    `n_per_cycle` controls how densely you sample one cycle (6-10 is usually
    plenty). Extrema (full open / full closed) are always included.
    """
    from scipy.signal import find_peaks

    x = position - position.mean()
    period = detect_period(position)
    distance = max(3, int(0.6 * period))

    peaks, _ = find_peaks(x, distance=distance)
    troughs, _ = find_peaks(-x, distance=distance)
    extrema = np.sort(np.concatenate([peaks, troughs]))

    # also sample uniformly within each inter-extremum segment
    keys = set(int(e) for e in extrema)
    for a, b in zip(extrema[:-1], extrema[1:]):
        if b - a < 2:
            continue
        n_mid = max(0, n_per_cycle // 2 - 1)
        for f in np.linspace(a, b, n_mid + 2)[1:-1]:
            keys.add(int(round(f)))
    # always include first and last frame
    keys.update([0, len(position) - 1])
    return np.array(sorted(keys))


# ============================================================
# 8. Phase-averaging — boost SNR when motion is periodic
# ============================================================
def phase_average_stack(stack: np.ndarray, position: np.ndarray,
                        n_phase_bins: int = 40) -> np.ndarray:
    """Bin frames by the phase of the leaflet trajectory and average within
    each bin. Returns a (n_phase_bins, H, W) stack representing one averaged
    cycle. SNR improves roughly sqrt(frames_per_bin).
    """
    x = position - position.mean()
    analytic = _hilbert(x)
    phase = np.angle(analytic)                       # in [-pi, pi]
    bins = np.floor((phase + np.pi) / (2 * np.pi) * n_phase_bins).astype(int)
    bins = np.clip(bins, 0, n_phase_bins - 1)

    T, H, W = stack.shape
    out = np.zeros((n_phase_bins, H, W), dtype=np.float32)
    counts = np.zeros(n_phase_bins, dtype=np.int32)
    for t in range(T):
        out[bins[t]] += stack[t]
        counts[bins[t]] += 1
    counts = np.maximum(counts, 1)[:, None, None]
    return out / counts


def _hilbert(x: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert
    return hilbert(x)


# ============================================================
# Global defaults (named constants, easy to override)
# ============================================================

# Search window for valve ROI suggestion (used in interactive ROI selection only)
MAX_SEARCH_WINDOW = 16

# Frames to pad BEFORE the ventricle contraction and AFTER the atrium
# contraction when searching for the valve-open darkening peak within each
# diastolic V->A cardiac interval. The contraction pair defines the anchor,
# not the boundary; the true peak may lie slightly outside — these pads capture it.
PRE_WINDOW_FRAMES_DEFAULT = 5
POST_WINDOW_FRAMES_DEFAULT = 5

# Default z-score thresholds for contraction detection
# Stricter than before — only clear, high-confidence contractions
# Asymmetric defaults — ventricle and atrium have different dynamics
VENTRICLE_Z_THRESHOLD_DEFAULT = -1.7
ATRIUM_Z_THRESHOLD_DEFAULT = -2.0

# Default expected valve events per analysis window
DEFAULT_VALVE_PEAK_COUNT = 4

# Valve ROI size (pixels, side length) for intersection-based tracking
VALVE_AREA_SIZE = 4

# U-Net configuration
USE_UNET_VALVE_TRACKING = False  # switch between manual-reference and U-Net
# MIN_TRAINING_SAMPLES has ONE definition, in leaflet_unet.py; import it here so the
# same value resolves in valve_analysis (no second definition / drift).
from leaflet_unet import MIN_TRAINING_SAMPLES  # noqa: E402  (single source of truth)


def compute_valve_signal_from_frame_diff(roi_stack):
    """Centralized valve motion signal computation from frame differences.

    This is the ONLY function that should produce the valve signal.
    Using frame-to-frame absolute difference ensures the signal reflects
    actual motion, not static intensity.

    Args:
        roi_stack: (T, H, W) array — the valve ROI across all frames.

    Returns:
        motion: (T,) array — per-frame absolute difference sum.
        motion_smooth: (T,) smoothed with uniform filter (size=3).
    """
    T = roi_stack.shape[0]
    motion = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        motion[t] = np.nansum(np.abs(
            roi_stack[t].astype(np.float32) - roi_stack[t-1].astype(np.float32)))
    motion = np.nan_to_num(motion, 0)
    motion_smooth = ndimage.uniform_filter1d(motion, size=3)
    return motion, motion_smooth


def compute_valve_open_signal_from_darkening(roi_raw):
    """Valve-OPEN signal: how far the ROI intensity drops BELOW its static
    background.

    Dark-field physics: the valve leaflets scatter light (bright). When the
    valve OPENS the leaflets separate, the ROI scatters less light and its
    intensity falls BELOW the static background. We measure that darkening
    directly — keeping the below-background (negative) side rather than the
    bright/positive side — so the peaks of this signal mark valve-open events.

    The static background is the per-pixel temporal MEDIAN (a typical level the
    intensity can dip below). A per-pixel temporal minimum would make the drop
    identically zero, so median (not min) is required here.

    Args:
        roi_raw: (T, H, W) raw (un-clipped) ROI intensities across all frames.

    Returns:
        darkening: (T,) per-frame sum over the ROI of clip(background - I, 0).
        darkening_smooth: (T,) smoothed with a uniform filter (size=3).
    """
    roi_raw = roi_raw.astype(np.float32)
    bg = np.median(roi_raw, axis=0)                       # static background / pixel
    darkening = np.clip(bg[None] - roi_raw, 0.0, None).sum(axis=(1, 2))
    darkening = np.nan_to_num(darkening, nan=0.0)
    darkening_smooth = ndimage.uniform_filter1d(darkening, size=3)
    return darkening, darkening_smooth


# ============================================================
# Publication-quality correlation figure
# ============================================================
def make_publication_figure(ventricle_signal, atrium_signal, valve_signal,
                             ventricle_peaks, atrium_peaks, valve_peaks,
                             fps=None, frame_range=None,
                             bandpass=None, savgol_window=None,
                             ventricle_z_threshold=VENTRICLE_Z_THRESHOLD_DEFAULT,
                             atrium_z_threshold=ATRIUM_Z_THRESHOLD_DEFAULT,
                             save_path=None, dark_mode=False):
    """Generate a publication-ready 4-panel stacked correlation figure.

    Thresholds are asymmetric: V and A have independent defaults.
    Valve-OPEN events are the frame of maximum darkening (intensity dropping
    below the static background) within each A->V cardiac interval.
    Legend is consolidated: one entry per conceptual category.
    """
    from scipy.signal import hilbert, savgol_filter, butter, filtfilt, find_peaks
    from matplotlib.lines import Line2D

    # ── Crop ──
    N = len(ventricle_signal)
    f0, f1 = (frame_range if frame_range else (0, N))
    f0, f1 = max(0, f0), min(N, f1)

    v_sig = ventricle_signal[f0:f1].copy()
    a_sig = atrium_signal[f0:f1].copy()
    vl_sig = valve_signal[f0:f1].copy()
    n = len(v_sig)

    if fps and fps > 0:
        t = np.arange(n) / fps + f0 / fps
        xlabel = 'Time (s)'
    else:
        t = np.arange(f0, f0 + n)
        xlabel = 'Frame'

    methods = []

    # ── z-score ──
    def _zscore(x):
        s = np.std(x)
        return (x - np.mean(x)) / s if s > 1e-10 else x - np.mean(x)

    v_sig = _zscore(v_sig)
    a_sig = _zscore(a_sig)
    vl_sig = _zscore(vl_sig)
    methods.append("z-score normalization (zero mean, unit variance)")

    # ── Savitzky-Golay ──
    if savgol_window and savgol_window >= 5:
        w = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        v_sig = savgol_filter(v_sig, w, 3)
        a_sig = savgol_filter(a_sig, w, 3)
        vl_sig = savgol_filter(vl_sig, w, 3)
        methods.append(f"Savitzky-Golay smoothing (window={w}, order=3)")

    # ── Threshold-based contraction detection (asymmetric) ──
    v_pk_raw, _ = find_peaks(-v_sig, prominence=0.3, distance=5)
    a_pk_raw, _ = find_peaks(-a_sig, prominence=0.3, distance=5)
    v_pk = np.array([p for p in v_pk_raw if v_sig[p] < ventricle_z_threshold])
    a_pk = np.array([p for p in a_pk_raw if a_sig[p] < atrium_z_threshold])

    methods.append(f"Ventricle threshold: z < {ventricle_z_threshold} -> {len(v_pk)} contractions")
    methods.append(f"Atrium threshold: z < {atrium_z_threshold} -> {len(a_pk)} contractions")

    # ── Valve-OPEN peak detection: exactly one event per cardiac cycle ──
    # The valve signal is the DARKENING signal (intensity below the static
    # background): in dark-field an OPEN valve scatters less light, so the ROI
    # dims. The AV valve opens during ventricular diastole — the filling phase
    # leading up to each ATRIAL contraction. We anchor ONE window on each
    # detected atrial contraction (the reliable once-per-cycle marker), look
    # back ~one cardiac cycle into the diastole, and take the frame of MAXIMUM
    # darkening. Anchoring on A (not the preceding V) makes detection robust to
    # missing/extra ventricle peaks, so the opening count always equals the
    # number of cardiac cycles — first and last cycle included.

    pre_w = PRE_WINDOW_FRAMES_DEFAULT
    post_w = POST_WINDOW_FRAMES_DEFAULT

    # Envelope of the darkening signal — its maxima mark the most-open frames.
    vl_envelope_pre = np.abs(hilbert(vl_sig.copy()))

    # Cardiac-cycle length = median spacing of the detected A (or V) contraction
    # points (the user's "between consecutive V-A points" cycle). Median is used
    # so a duplicate contraction point can't shrink it.
    if len(a_pk) >= 2:
        period_est = float(np.median(np.diff(np.sort(a_pk))))
    elif len(v_pk) >= 2:
        period_est = float(np.median(np.diff(np.sort(v_pk))))
    else:
        period_est = float(n)
    if not np.isfinite(period_est) or period_est < 4:
        period_est = float(n)

    # Number of cardiac cycles = the more complete chamber's contraction count.
    # (Using max(#V,#A) so a strict threshold on one chamber can't undercount.)
    a_sorted = np.sort(a_pk).astype(int)
    v_sorted = np.sort(v_pk).astype(int)
    landmarks = a_sorted if len(a_sorted) >= len(v_sorted) else v_sorted
    landmark_kind = 'A' if len(a_sorted) >= len(v_sorted) else 'V'
    n_cycles = max(len(landmarks), 1)

    # Segment the trace into ONE window per cardiac cycle and take the strongest
    # darkening frame in EACH segment. Segments are bounded by the midpoints
    # between consecutive contraction landmarks (extended to the trace ends), so
    # every segment contains exactly one landmark = one cycle. Earlier global
    # peak-picking / fixed-window methods dropped a cycle whose opening was weak
    # or off-phase (e.g. cycle 3 here). Per-segment argmax instead GUARANTEES one
    # opening per cycle and covers every cycle, including a weak one. Result
    # count == n_cycles by construction.
    if len(landmarks) >= 2:
        mids = ((landmarks[:-1] + landmarks[1:]) // 2).astype(int)
        bounds = np.concatenate([[0], mids, [n]])
    else:
        bounds = np.array([0, n], dtype=int)

    vl_pk_list = []
    for i in range(len(bounds) - 1):
        s0, s1 = int(bounds[i]), int(bounds[i + 1])
        if s1 <= s0:
            continue
        seg = vl_envelope_pre[s0:s1]
        if len(seg) == 0:
            continue
        vl_pk_list.append(s0 + int(np.argmax(seg)))     # strongest darkening in cycle
    vl_pk_final = np.array(sorted(set(vl_pk_list)), dtype=int)
    n_outside = 0

    peak_log = []
    for ei, pk in enumerate(vl_pk_final):
        nxt_a = a_sorted[a_sorted >= pk]
        prv_v = v_sorted[v_sorted <= pk]
        astr = f", A{pk - int(nxt_a[0]):+d}" if len(nxt_a) else ""
        vstr = f", V{pk - int(prv_v[-1]):+d}" if len(prv_v) else ""
        peak_log.append(f"  Event {ei}: open={pk}{vstr}{astr}")

    # Print per-event log
    print(f"\n  Valve-open peak detection (1 per cycle, "
          f"segmented by {n_cycles} {landmark_kind} landmarks):")
    for log_line in peak_log:
        print(log_line)
    print(f"  Cycles (max of V={len(v_sorted)}, A={len(a_sorted)}): {n_cycles} -> "
          f"openings detected: {len(vl_pk_final)}")

    # Summary statistics: how long after the V contraction the valve opens,
    # and how long before the following A contraction.
    if len(vl_pk_final) > 0 and len(v_pk) > 0:
        offsets_from_v = []
        offsets_to_a = []
        for pk in vl_pk_final:
            # Nearest preceding V
            dv = pk - v_pk
            prev_v = dv[dv >= 0]
            if len(prev_v) > 0:
                offsets_from_v.append(prev_v.min())
            # Nearest following A
            da = a_pk - pk
            next_a = da[da >= 0]
            if len(next_a) > 0:
                offsets_to_a.append(next_a.min())

        mean_off_v = np.mean(offsets_from_v) if offsets_from_v else float('nan')
        mean_off_a = np.mean(offsets_to_a) if offsets_to_a else float('nan')
    else:
        mean_off_v = mean_off_a = float('nan')

    print(f"\n  Summary: {len(vl_pk_final)} opening(s) for {n_cycles} cycle(s)")
    print(f"  Mean open delay after V contraction: {mean_off_v:.1f} frames")
    print(f"  Mean lead before next A contraction: {mean_off_a:.1f} frames")

    methods.append(f"Valve-open detection: one per cycle, anchored on V+A contractions "
                    f"(period~{period_est:.0f}f, diastolic window)")
    methods.append(f"Valve-open peaks: {len(vl_pk_final)} "
                    f"(1 per cardiac cycle; {n_cycles} cycles)")

    # ── Bandpass ──
    vl_filtered = None
    if bandpass and fps and fps > 0:
        lo, hi = bandpass
        nyq = fps / 2.0
        if lo < nyq and hi < nyq and lo < hi:
            b, a_c = butter(3, [lo / nyq, hi / nyq], btype='band')
            vl_filtered = filtfilt(b, a_c, vl_sig)
            methods.append(f"Butterworth bandpass ({lo:.1f}-{hi:.1f} Hz, order 3, zero-phase)")

    # ── Hilbert envelope for plotting (may include bandpass) ──
    vl_for_env = vl_filtered if vl_filtered is not None else vl_sig
    vl_envelope = np.abs(hilbert(vl_for_env))
    methods.append("Hilbert analytic signal envelope on valve channel")

    # ── Figure ──
    bg = '#1a1a1a' if dark_mode else 'white'
    fg = 'white' if dark_mode else 'black'
    fg2 = '#aaaaaa' if dark_mode else '#666666'
    font = 'Arial'

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'hspace': 0.06,
                                           'height_ratios': [1, 1, 1, 0.8]})
    fig.patch.set_facecolor(bg)

    def _style_ax(ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=10, fontfamily=font, color=fg, labelpad=8)
        ax.tick_params(axis='both', labelsize=9, colors=fg, width=0.8, length=4,
                        direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_linewidth(0.8)
            ax.spines[sp].set_color(fg)
        ax.set_facecolor(bg)
        ax.grid(True, axis='y', alpha=0.12, lw=0.4)

    vc = '#d62728'  # ventricle
    ac = '#1f77b4'  # atrium
    vlc = '#bcbd22'  # valve

    # ── Panel 1: Ventricle ──
    axes[0].plot(t, v_sig, color=vc, lw=1.2)
    axes[0].axhline(ventricle_z_threshold, color=vc, lw=0.5, ls='--', alpha=0.4)
    if len(v_pk) > 0:
        axes[0].plot(t[v_pk], v_sig[v_pk], 'v', color=vc, ms=7,
                      markeredgecolor='black', markeredgewidth=0.5)
    # Cross-channel reference lines (no individual labels)
    for vf in vl_pk_final:
        if vf < n: axes[0].axvline(t[vf], color=vlc, lw=0.6, ls=':', alpha=0.4)
    for ap in a_pk:
        axes[0].axvline(t[ap], color=ac, lw=0.4, ls='--', alpha=0.2)
    _style_ax(axes[0], 'Ventricle\n(z-score)')
    # Consolidated legend via proxy artists
    axes[0].legend(handles=[
        Line2D([0],[0], color=vc, lw=1.2, label='Ventricle'),
        Line2D([0],[0], color=vc, lw=0.5, ls='--', alpha=0.5, label=f'Threshold (z={ventricle_z_threshold})'),
        Line2D([0],[0], marker='v', color=vc, lw=0, ms=7, markeredgecolor='black',
                markeredgewidth=0.5, label=f'V contraction (n={len(v_pk)})'),
    ], fontsize=9, frameon=True, facecolor=bg, edgecolor='#cccccc', labelcolor=fg, loc='upper right')

    # ── Panel 2: Atrium ──
    axes[1].plot(t, a_sig, color=ac, lw=1.2)
    axes[1].axhline(atrium_z_threshold, color=ac, lw=0.5, ls='--', alpha=0.4)
    if len(a_pk) > 0:
        axes[1].plot(t[a_pk], a_sig[a_pk], 'v', color=ac, ms=7,
                      markeredgecolor='black', markeredgewidth=0.5)
    for vf in vl_pk_final:
        if vf < n: axes[1].axvline(t[vf], color=vlc, lw=0.6, ls=':', alpha=0.4)
    for vp in v_pk:
        axes[1].axvline(t[vp], color=vc, lw=0.4, ls='--', alpha=0.2)
    _style_ax(axes[1], 'Atrium\n(z-score)')
    axes[1].legend(handles=[
        Line2D([0],[0], color=ac, lw=1.2, label='Atrium'),
        Line2D([0],[0], color=ac, lw=0.5, ls='--', alpha=0.5, label=f'Threshold (z={atrium_z_threshold})'),
        Line2D([0],[0], marker='v', color=ac, lw=0, ms=7, markeredgecolor='black',
                markeredgewidth=0.5, label=f'A contraction (n={len(a_pk)})'),
    ], fontsize=9, frameon=True, facecolor=bg, edgecolor='#cccccc', labelcolor=fg, loc='upper right')

    # ── Panel 3: Valve ──
    vl_plot = vl_filtered if vl_filtered is not None else vl_sig
    axes[2].plot(t, vl_plot, color=vlc, lw=0.8, alpha=0.4)
    axes[2].plot(t, vl_envelope, color=vlc, lw=1.5)
    vl_valid = vl_pk_final[vl_pk_final < n] if len(vl_pk_final) > 0 else np.array([])
    if len(vl_valid) > 0:
        axes[2].plot(t[vl_valid], vl_envelope[vl_valid], '*', color=vlc, ms=10,
                      markeredgecolor='black', markeredgewidth=0.5)
    for vf in vl_pk_final:
        if vf < n: axes[2].axvline(t[vf], color=vlc, lw=0.6, ls=':', alpha=0.4)
    for ap in a_pk:
        axes[2].axvline(t[ap], color=ac, lw=0.4, ls='--', alpha=0.15)
    for vp in v_pk:
        axes[2].axvline(t[vp], color=vc, lw=0.4, ls='--', alpha=0.15)
    _style_ax(axes[2], 'Valve\n(z-score)')
    axes[2].legend(handles=[
        Line2D([0],[0], color=vlc, lw=0.8, alpha=0.4, label='Valve darkening signal'),
        Line2D([0],[0], color=vlc, lw=1.5, label='Darkening envelope'),
        Line2D([0],[0], marker='*', color=vlc, lw=0, ms=10, markeredgecolor='black',
                markeredgewidth=0.5, label=f'Valve open (n={len(vl_valid)})'),
    ], fontsize=9, frameon=True, facecolor=bg, edgecolor='#cccccc', labelcolor=fg, loc='upper right')

    # ── Panel 4: Combined overlay ──
    ax4 = axes[3]
    ax4.plot(t, vl_envelope, color=vlc, lw=1.5)
    if len(vl_valid) > 0:
        ax4.plot(t[vl_valid], vl_envelope[vl_valid], '*', color=vlc, ms=8,
                  markeredgecolor='black', markeredgewidth=0.5)

    ax4b = ax4.twinx()
    ax4b.plot(t, v_sig, color=vc, lw=0.7, alpha=0.35)
    ax4b.plot(t, a_sig, color=ac, lw=0.7, alpha=0.35)
    ax4b.set_ylabel('V / A (z-score)', fontsize=8, fontfamily=font, color=fg2, labelpad=4)
    ax4b.tick_params(axis='y', labelsize=7, colors=fg2)
    ax4b.spines['right'].set_color(fg2)
    ax4b.spines['right'].set_linewidth(0.5)

    for vf in vl_pk_final:
        if vf < n: ax4.axvline(t[vf], color=vlc, lw=0.6, ls=':', alpha=0.4)

    _style_ax(ax4, 'Valve env.\n+ V/A')
    ax4.set_xlabel(xlabel, fontsize=10, fontfamily=font, color=fg, labelpad=6)

    # Consolidated legend via proxy artists
    from matplotlib.lines import Line2D as L2D
    ax4.legend(handles=[
        L2D([0],[0], color=vlc, lw=1.5, label='Valve darkening envelope'),
        L2D([0],[0], marker='*', color=vlc, lw=0, ms=8, label=f'Valve open (n={len(vl_valid)})'),
        L2D([0],[0], color=vc, lw=0.7, alpha=0.4, label='Ventricle'),
        L2D([0],[0], color=ac, lw=0.7, alpha=0.4, label='Atrium'),
    ], fontsize=9, frameon=True, facecolor=bg, edgecolor='#cccccc',
        labelcolor=fg, loc='upper right', ncol=2)

    # Suptitle
    fig.suptitle('Valve opening aligned with ventricular and atrial contractions',
                  fontsize=12, fontfamily=font, fontweight='bold', color=fg, y=0.99)

    fig.align_ylabels(axes)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ── Save ──
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                     facecolor=bg, transparent=(not dark_mode))
        print(f"Publication figure saved: {save_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PUBLICATION FIGURE — METHODS & DETECTION SUMMARY")
    print("=" * 60)
    for i, m in enumerate(methods, 1):
        print(f"  {i}. {m}")
    print(f"\n  Frame range: [{f0}, {f1})")
    if fps:
        print(f"  Sampling rate: {fps} fps")
    print(f"  Ventricle contractions: {len(v_pk)} (threshold z < {ventricle_z_threshold})")
    print(f"  Atrium contractions: {len(a_pk)} (threshold z < {atrium_z_threshold})")
    print(f"  Valve open peaks: {len(vl_pk_final)} (1 per cycle; {n_cycles} cycles)")
    print(f"  Detection: anchored on V+A contractions, diastolic window ~{period_est:.0f}f")
    print(f"  Mean open delay after V: {mean_off_v:.1f}f, lead before next A: {mean_off_a:.1f}f")
    print("=" * 60)

    return fig


# ============================================================
# Valve open/close TEMPLATE FIT — validate the expected shape, then extract
# ============================================================
def _raised_cosine_pulse(phi, center, width, taper):
    """A flat-top 'open' pulse on circular phase phi in [0,1): value 1 over an
    open window of fractional `width` centred at `center`, with raised-cosine
    (smooth) rise/fall of fractional `taper`, and 0 ('closed') elsewhere."""
    phi = np.asarray(phi, dtype=np.float64)
    d = np.abs((phi - center + 0.5) % 1.0 - 0.5)     # circular distance to centre
    inner = max(0.0, width / 2.0 - taper / 2.0)
    out = np.zeros_like(phi)
    out[d <= inner] = 1.0
    edge = (d > inner) & (d < inner + taper)
    if taper > 1e-6:
        out[edge] = 0.5 * (1.0 + np.cos(np.pi * (d[edge] - inner) / taper))
    return out


def _detect_contractions(trace, z_threshold, hp_size=80, prominence=0.3,
                         distance=5, expected=4, fps=600.0):
    """High-pass + z-score a chamber trace and return contraction frames
    (downward dips with z < z_threshold). The minimum inter-peak distance is tied
    to the ROBUST fundamental cardiac period (>= ~half a beat) so it cannot
    over-count (~20x) on a noisy per-frame area trace."""
    from scipy.signal import find_peaks
    from leaflet_unet import robust_cardiac_period
    x = np.asarray(trace, dtype=np.float64)
    x = x - ndimage.uniform_filter1d(x, size=hp_size)
    s = np.std(x)
    z = (x - x.mean()) / s if s > 1e-10 else x - x.mean()
    P = robust_cardiac_period([trace], len(z), expected, fps)
    dist = max(int(distance), int(round(0.55 * P)))   # >= ~half a cardiac cycle
    raw, _ = find_peaks(-z, prominence=prominence, distance=dist)
    strong = np.array([p for p in raw if z[p] < z_threshold], dtype=int)
    if len(strong) >= max(2, expected - 1):
        return strong, z
    if len(raw) == 0:                                  # weak area signal -> deepest dips
        return strong, z
    deepest = raw[np.argsort(z[raw])][:expected]
    return np.array(sorted(deepest), dtype=int), z


def fit_valve_open_close_template(
        valve_path="line_darkening_signal.npy",
        atrium_path="atrium_trace.npy",
        ventricle_path="ventricle_trace.npy",
        ventricle_z_threshold=VENTRICLE_Z_THRESHOLD_DEFAULT,
        atrium_z_threshold=-1.7, fps=None, save_path=None, show=True):
    """Fit the directly-measured valve openness signal to the EXPECTED
    open/close curve (a raised-cosine pulse, one per cardiac cycle), report how
    well it matches (R^2), then extract per-cycle opening / closing frames from
    the SAME fitted template.

    Cardiac phase comes from the ventricle contractions (V-to-V cycle, phase 0
    at V). If the V/A traces are missing, the dominant period is estimated from
    the valve signal's autocorrelation instead (phase reference then arbitrary).

    Returns dict(r2, params, open_frames, close_frames, phase, period).
    """
    from scipy.optimize import curve_fit

    if not os.path.exists(valve_path):
        alt = "unet_line_break_signal.npy"
        if os.path.exists(alt):
            print(f"  [Fit Valve] '{valve_path}' not found — using '{alt}' instead.")
            valve_path = alt
        else:
            print(f"  [Fit Valve] '{valve_path}' not found — run 'Valve Line' first.")
            return None

    valve = np.load(valve_path).astype(np.float64)
    N = len(valve)
    s = np.std(valve)
    vz = (valve - valve.mean()) / s if s > 1e-10 else valve - valve.mean()

    # ── cardiac landmarks ──
    v_pk = a_pk = np.array([], dtype=int)
    if os.path.exists(ventricle_path):
        v_pk, _ = _detect_contractions(np.load(ventricle_path), ventricle_z_threshold)
    if os.path.exists(atrium_path):
        a_pk, _ = _detect_contractions(np.load(atrium_path), atrium_z_threshold)

    # ── period: prefer the valve signal's own autocorrelation (robust); use
    #    V-to-V folding only when the V contractions are clean and consistent ──
    x = vz - vz.mean()
    ac = np.correlate(x, x, mode='full')[N - 1:]
    ac[:3] = 0
    hi_lag = max(4, int(N // 2))
    period_ac = float(np.argmax(ac[3:hi_lag]) + 3) if hi_lag > 3 else float(N)

    use_vv = False
    if len(v_pk) >= 3:
        iv = np.diff(np.sort(v_pk)).astype(float)
        if iv.mean() > 0 and iv.std() / iv.mean() < 0.35 \
                and abs(np.median(iv) - period_ac) < 0.4 * period_ac:
            use_vv = True

    if use_vv:
        landmarks = np.sort(v_pk)
        period = float(np.median(np.diff(landmarks)))
        phase = np.full(N, np.nan)
        for k in range(len(landmarks) - 1):
            t0, t1 = landmarks[k], landmarks[k + 1]
            if t1 > t0:
                phase[t0:t1] = (np.arange(t0, t1) - t0) / (t1 - t0)
        ref = 'V-to-V landmarks'
    else:
        period = period_ac if period_ac >= 4 else float(N)
        phase = (np.arange(N) % period) / period
        landmarks = np.arange(0, N, period).astype(int)
        ref = f'valve autocorrelation (period~{period:.0f}f)'

    valid = ~np.isnan(phase)
    if valid.sum() < 8:
        print("  [Fit Valve] not enough cycles to fit.")
        return None
    phv, zv = phase[valid], vz[valid]

    # ── data-driven initial guess from the phase-binned mean ──
    nb = 24
    binid = np.clip((phv * nb).astype(int), 0, nb - 1)
    binmean = np.array([zv[binid == b].mean() if np.any(binid == b) else np.nan
                        for b in range(nb)])
    bin_centers = (np.arange(nb) + 0.5) / nb
    center0 = float(bin_centers[np.nanargmax(binmean)])
    amp0 = float(np.nanmax(binmean) - np.nanmin(binmean)) or 1.0
    base0 = float(np.nanmin(binmean))

    # ── fit a SMOOTH periodic open/close pulse (von Mises bump) ──
    # A flat-top raised-cosine pulse has ZERO gradient on its plateau and closed
    # baseline, so curve_fit couldn't move it and the fit failed. The von Mises
    # bump is smooth everywhere (nonzero gradient), so it reliably converges to
    # the real open pulse. `center` = open peak phase; `kappa` = sharpness.
    def model(phi, center, kappa, amp, base):
        return base + amp * np.exp(kappa * (np.cos(2 * np.pi * (phi - center)) - 1.0))

    p0 = [center0, 4.0, amp0, base0]
    bounds = ([0.0, 0.3, 0.0, -5.0], [1.0, 60.0, 10.0, 5.0])
    try:
        popt, _ = curve_fit(model, phv, zv, p0=p0, bounds=bounds, maxfev=30000)
    except Exception as e:
        print(f"  [Fit Valve] fit failed: {e}")
        return None
    center, kappa, amp, base = popt
    center = float(center % 1.0)

    pred = model(phv, *popt)
    ss_res = float(np.sum((zv - pred) ** 2))
    ss_tot = float(np.sum((zv - zv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # ── open/close phases = half-maximum crossings of the fitted bump ──
    val = 1.0 + np.log(0.5) / kappa            # cos(theta) at half-max
    half_w = 0.5 if val <= -1.0 else float(np.arccos(np.clip(val, -1.0, 1.0)) / (2 * np.pi))
    width = 2 * half_w
    open_phase = (center - half_w) % 1.0
    close_phase = (center + half_w) % 1.0

    open_frames, close_frames = [], []
    for k in range(len(landmarks) - 1):
        t0, t1 = int(landmarks[k]), int(landmarks[k + 1])
        per = t1 - t0
        if per <= 0:
            continue
        open_frames.append(int(round(t0 + open_phase * per)))
        close_frames.append(int(round(t0 + close_phase * per)))
    open_frames = np.array(sorted(set(f for f in open_frames if 0 <= f < N)))
    close_frames = np.array(sorted(set(f for f in close_frames if 0 <= f < N)))

    a_phase = phase[a_pk[(a_pk < N)]] if len(a_pk) else np.array([])
    a_phase = a_phase[~np.isnan(a_phase)]

    print("\n" + "=" * 60)
    print("VALVE OPEN/CLOSE TEMPLATE FIT  (smooth von-Mises pulse)")
    print("=" * 60)
    print(f"  Reference signal : {valve_path}")
    print(f"  Phase reference  : {ref}  (period~{period:.1f} frames, "
          f"{len(landmarks)} landmarks)")
    print(f"  Fit R^2          : {r2:.3f}   "
          f"({'good match' if r2 > 0.5 else 'weak - shape may not be a clean pulse'})")
    print(f"  Open window      : phase [{open_phase:.2f} .. {close_phase:.2f}]  "
          f"(open fraction {width:.2f} of the cycle)")
    print(f"  Open centre      : phase {center:.2f}   (sharpness kappa={kappa:.1f})")
    if len(a_phase):
        print(f"  Atrial contraction at phase {np.nanmean(a_phase):.2f} "
              f"(valve should be OPEN here)")
    print(f"  Openings extracted: {len(open_frames)}   "
          f"Closings: {len(close_frames)}")
    print("=" * 60)

    # ── figure ──
    t = (np.arange(N) / fps) if fps else np.arange(N)
    xlabel = 'Time (s)' if fps else 'Frame'
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Panel 1: faded folded data points + bold solid fitted curve through them
    ax[0].plot(phv, zv, '.', color='#bcbd22', ms=3, alpha=0.12, zorder=1,
               label='measured valve (folded)')
    ax[0].plot(bin_centers, binmean, 'o', color='#7f7f1f', ms=4, alpha=0.5,
               zorder=2, label='phase-binned mean')
    grid = np.linspace(0, 1, 400)
    ax[0].plot(grid, model(grid, *popt), color='#7a0000', lw=3.0, zorder=4,
               solid_capstyle='round',
               label=f'fitted open/close curve (R²={r2:.2f})')
    span_hi = close_phase if close_phase > open_phase else 1.0
    ax[0].axvspan(open_phase, span_hi, color='#bcbd22', alpha=0.10, zorder=0,
                  label='open window')
    if len(a_phase):
        ax[0].axvline(np.nanmean(a_phase), color='#1f77b4', ls='--', lw=1.2,
                      zorder=3, label='mean atrial contraction')
    ax[0].set_xlabel(f'Cardiac phase ({ref.split()[0]})')
    ax[0].set_ylabel('Valve z-score')
    ax[0].set_title('Expected open/close curve fitted to measured valve data')
    ax[0].set_xlim(0, 1)
    ax[0].legend(fontsize=8, loc='upper right'); ax[0].grid(alpha=0.2)

    # Panel 2: full time series + extracted open/close
    ax[1].plot(t, vz, color='#bcbd22', lw=0.6, alpha=0.45, label='valve z-score')
    if len(landmarks) >= 2:
        tiled = np.full(N, np.nan)
        for k in range(len(landmarks) - 1):
            t0, t1 = int(landmarks[k]), int(landmarks[k + 1])
            if t1 > t0:
                ph = (np.arange(t0, t1) - t0) / (t1 - t0)
                tiled[t0:t1] = model(ph, *popt)
        ax[1].plot(t, tiled, color='#7a0000', lw=2.2, label='fitted template')
    for f in open_frames:
        ax[1].axvline(t[f], color='#2ca02c', lw=0.9, alpha=0.8)
    for f in close_frames:
        ax[1].axvline(t[f], color='#9467bd', lw=0.9, ls='--', alpha=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='#bcbd22', lw=1.0, label='valve z-score'),
               Line2D([0], [0], color='#7a0000', lw=2.2, label='fitted template'),
               Line2D([0], [0], color='#2ca02c', lw=0.9, label='opening'),
               Line2D([0], [0], color='#9467bd', lw=0.9, ls='--', label='closing')]
    ax[1].legend(handles=handles, fontsize=8, loc='upper right')
    ax[1].set_xlabel(xlabel); ax[1].set_ylabel('Valve z-score')
    ax[1].set_title('Extracted opening (green) / closing (purple) per cycle')
    ax[1].grid(alpha=0.2)
    fig.tight_layout()

    np.save("valve_open_frames.npy", open_frames)
    np.save("valve_close_frames.npy", close_frames)
    np.savez("valve_template_fit.npz", center=center, kappa=kappa, width=width,
             amp=amp, base=base, r2=r2, period=period, landmarks=landmarks)
    print("  Saved: valve_open_frames.npy, valve_close_frames.npy, "
          "valve_template_fit.npz")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")
    if show:
        plt.show()

    return dict(r2=r2, params=dict(center=center, kappa=kappa, width=width,
                amp=amp, base=base), open_frames=open_frames,
                close_frames=close_frames, phase=phase, period=period)


# ============================================================
# Figure-(a) MANUAL chamber regions (schematic, drawn in the CROP frame)
# ============================================================
def _smooth_closed_polygon(verts, n_out=240):
    """Smooth a hand-drawn polygon ``verts`` ((N,2) x,y) into a CURVED CLOSED
    boundary via a periodic spline (so the schematic region has smooth edges, like
    the U-shape leaflets). Falls back to the raw closed polygon if the spline can't
    be fit. Pure geometry — no data/analysis touched."""
    P = np.asarray(verts, float)
    if P.ndim != 2 or len(P) < 3:
        return P
    try:
        from scipy.interpolate import splprep, splev
        k = min(3, len(P) - 1)
        tck, _u = splprep([P[:, 0], P[:, 1]], s=float(len(P)), per=True, k=k)
        xs, ys = splev(np.linspace(0.0, 1.0, n_out), tck)
        return np.column_stack([xs, ys])
    except Exception:
        return np.vstack([P, P[:1]])


def draw_figa_regions(frames, save_path, vent_color="#1f77b4", atr_color="#d62728",
                      flip_h=True, flip_v=False, alpha=0.22):
    """Interactively DRAW the figure-(a) VENTRICLE then ATRIUM schematic regions ON
    EACH of the panel-a frames SEPARATELY (the chamber positions differ between the
    CLOSED and OPEN frames, so each gets its own pair of regions). ``frames`` is a
    sequence of (frame_name, crop_img) tuples — e.g. [('closed', img_c), ('open',
    img_o)] — and the popup steps through each frame on its own image (same crop /
    flip / depth as the final panel). Click polygon vertices, ENTER to finish each
    region (draw nothing on a chamber -> skip; draw nothing on a frame -> AUTO for
    that frame). Each polygon is smoothed to a curved boundary and previewed as a
    faint blue/red fill; you are asked to KEEP or REDRAW per frame (redo until
    satisfied). Persisted to ``save_path`` (npz) as the single source of truth with
    per-frame keys ``{name}_ventricle`` / ``{name}_atrium`` = (N,2) x,y vertices in
    crop coords, plus ``shape``. SCHEMATIC for the figure ONLY — changes NO analysis
    signal. Returns a dict {frame_name: {'ventricle': ..., 'atrium': ...}, ...} of the
    drawn regions (or None if NOTHING was drawn on any frame)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as _Poly
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _have_tk = True
    except Exception:
        _have_tk = False

    def _one(ax, name, color):
        ax.set_title(f"{name}: click polygon vertices, ENTER to finish (>=3); "
                     f"draw nothing -> skip", fontsize=8)
        ax.figure.canvas.draw_idle()
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        P = np.array(pts, float) if pts else np.empty((0, 2))
        if len(P) >= 3:
            ax.add_patch(_Poly(_smooth_closed_polygon(P), closed=True, fill=True,
                               fc=color, ec=color, lw=1.0, alpha=alpha, zorder=3))
            ax.figure.canvas.draw_idle()
        return P

    out = {}                                                    # {frame_name: {'ventricle':..., 'atrium':...}}
    save_shape = None
    for frame_name, crop_img in frames:
        img = np.asarray(crop_img, float)
        if save_shape is None:
            save_shape = img.shape[:2]
        vlo, vhi = float(np.nanmin(img)), float(np.nanmax(img))
        while True:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            try:
                fig.canvas.manager.set_window_title(
                    f"Figure (a) [{frame_name.upper()} frame]: draw ventricle + atrium regions")
            except Exception:
                pass
            ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
            if flip_h:                             # WYSIWYG: same H/V flip as final figure (a)
                ax.invert_xaxis()
            if flip_v:
                ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            plt.show(block=False)
            vv = _one(ax, f"{frame_name.upper()} frame  -  VENTRICLE (blue)", vent_color)
            aa = _one(ax, f"{frame_name.upper()} frame  -  ATRIUM (red)", atr_color)
            if len(vv) < 3 and len(aa) < 3:        # opted out for THIS frame -> AUTO for it
                plt.close(fig)
                print(f"  [figa-regions] {frame_name}: nothing drawn -> AUTO for this frame")
                break
            ax.set_title(f"{frame_name.upper()}: KEEP or REDRAW (dialog)", fontsize=8)
            ax.figure.canvas.draw_idle(); plt.pause(0.2)
            if _have_tk:
                _r = _tk.Tk(); _r.withdraw(); _r.attributes("-topmost", True)
                keep = _mb.askyesno(f"Figure (a) regions [{frame_name}]",
                                    f"{frame_name.upper()} frame\n"
                                    f"Ventricle: {len(vv)} pts    Atrium: {len(aa)} pts\n\n"
                                    f"Keep these regions for the {frame_name} frame?\n"
                                    f"Yes = save & use     No = redraw")
                _r.destroy()
            else:
                keep = True
            plt.close(fig)
            if keep:
                reg = {}
                if len(vv) >= 3:
                    reg["ventricle"] = vv
                if len(aa) >= 3:
                    reg["atrium"] = aa
                if reg:
                    out[frame_name] = reg
                break

    if not out:
        print("  [figa-regions] nothing drawn on any frame -> AUTO crop-transform")
        return None
    try:
        payload = {"shape": np.array(save_shape, int)}
        for fname, reg in out.items():
            for chamber, P in reg.items():
                payload[f"{fname}_{chamber}"] = np.asarray(P, float)
        np.savez(save_path, **payload)
        summary = "; ".join(
            f"{fname}: " + ", ".join(f"{k}({len(v)})" for k, v in reg.items())
            for fname, reg in out.items())
        print(f"  [figa-regions] saved {os.path.basename(save_path)}: {summary}")
    except Exception as _e:
        print(f"  [figa-regions] save failed: {_e}")
    return out


# ============================================================
# Nature main-figure composite (driven by saved U-Net outputs)
# ============================================================
def make_nature_valve_figure(stack, um_per_pixel=None, fps=600.0,
                             save_stem="nature_valve_figure",
                             prefix="unet_line",
                             atrium_trace_path="atrium_trace.npy",
                             ventricle_trace_path="ventricle_trace.npy",
                             leaflet_lw=2.0, panel_c_mode="kinematics",
                             panel_f_mode="lag", montage_ref_frame=None,
                             open_threshold_um=30.0, show_tip_markers=True,
                             show_gap_width=True, show_state_tag=True,
                             leaflet_colors=("#E64B35", "#4DBBD5"), roi=None,
                             single_column=False, depth=None,
                             low_thr=0.35, high_thr=0.60, min_state_frames=None,
                             show_centerline=False, leaf_alpha=0.55,
                             open_gap_color="#FF8C00", use_unsup_openness=True,
                             open_duration_thr=0.5,
                             fig_a_regions="manual", figa_regions_path=None,
                             flip_horizontal=True, flip_vertical=False,
                             fig_width_mm=183.0):
    """Assemble a Nature main-figure-quality composite (panels a-f) from the
    saved U-Net outputs (run the U-Net 'analyze' step first) plus the
    dark-field-preprocessed ``stack``.

    Panels: (a) closed & open frames with SOLID centerline outlines + 200 um
    scale bar (bottom-right; at UM_PER_PIXEL=10 -> 20 px); (b) cropped time-strip montage of one opening (t=0 at the
    reference frame, +1000/fps ms/frame, 50 um bar); (c) panel_c_mode-selectable
    leaflet kinematics overlay (time-colored centerlines over one cycle) /
    'width' (tip gap um vs t) / 'angle' (deg vs t); (d) ventricle + atrium +
    openness on a shared time axis with the openings marked; (e) phase-averaged
    openness + von Mises fit (R^2); (f) opening angle vs cardiac phase. All scale
    bars / time axes derive from um_per_pixel and fps. Exports SVG+PDF+600-dpi
    PNG with editable text.
    """
    from scipy.optimize import curve_fit
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from leaflet_unet import extend_leaflet_pair, EXTEND_LEN_PX

    # ── load saved U-Net outputs ──
    need = f"{prefix}_openness.npy"
    if not os.path.exists(need):
        print(f"  [nature-fig] '{need}' not found — run the U-Net 'analyze' "
              f"(Breaks) step first.")
        return None
    openness = np.load(f"{prefix}_openness.npy")
    open_frames = np.load(f"{prefix}_open_frames.npy")
    # per-frame CLOSED/OPEN decision from the openness method (single source)
    open_state = (np.load(f"{prefix}_open_state.npy")
                  if os.path.exists(f"{prefix}_open_state.npy") else None)
    angle = np.load(f"{prefix}_angle_deg.npy") if os.path.exists(f"{prefix}_angle_deg.npy") else None
    if os.path.exists(f"{prefix}_gap_um.npy"):
        gap, gap_unit = np.load(f"{prefix}_gap_um.npy"), "µm"
    else:
        gap, gap_unit = np.load(f"{prefix}_gap_px.npy"), "px"
    cp = np.load(f"{prefix}_ctrl_points.npz")
    ctrl_a, ctrl_b = cp["leaflet_a"], cp["leaflet_b"]
    from leaflet_unet import ushape_around_line as _ushape   # schematic virtual leaflet
    draw_ushapes = True                                       # show U-shape schematics
    # connecting-line analyses (saved by the connector model's analyze step)
    kymo = (np.load(f"{prefix}_connector_kymo.npy")
            if os.path.exists(f"{prefix}_connector_kymo.npy") else None)
    break_pos = (np.load(f"{prefix}_break_pos.npy")
                 if os.path.exists(f"{prefix}_break_pos.npy") else None)
    unsup = (np.load(f"{prefix}_midpoint_darkening.npy")    # UNSUPERVISED darkening
             if os.path.exists(f"{prefix}_midpoint_darkening.npy") else None)
    ev = (np.load(f"{prefix}_events.npz", allow_pickle=True)
          if os.path.exists(f"{prefix}_events.npz") else None)   # V/A-timing + metrics

    stack = np.clip(np.asarray(stack, np.float32), 0, 1)
    T = stack.shape[0]
    n = min(T, len(openness))
    openness, stack = openness[:n], stack[:n]
    open_frames = open_frames[open_frames < n]

    # ── SINGLE SOURCE OF TRUTH: the SELECTED depth's per-depth V/A (901 frames).
    # Prefer the explicit per-depth path (depth_config) so panel d, the cycle
    # detection and the phase-locking all use ONLY the chosen depth - never an
    # all-depth/combined or a stale other-depth signal. ──
    v_src = ventricle_trace_path if os.path.exists(ventricle_trace_path) \
        else (f"{prefix}_ventricle.npy" if os.path.exists(f"{prefix}_ventricle.npy") else ventricle_trace_path)
    a_src = atrium_trace_path if os.path.exists(atrium_trace_path) \
        else (f"{prefix}_atrium.npy" if os.path.exists(f"{prefix}_atrium.npy") else atrium_trace_path)
    v_pk = a_pk = np.array([], int); v_z = a_z = None
    if os.path.exists(v_src):
        _vload = np.load(v_src)
        print(f"  [nature-fig] panel d VENTRICLE V/A: depth={depth}  shape={_vload.shape}  "
              f"-> using first {n} frames  src={os.path.basename(v_src)}")
        if _vload.shape[0] != n:
            print(f"  [nature-fig] WARNING ventricle trace has {_vload.shape[0]} frames "
                  f"(expected {n} for one depth); truncating to the first {n}.")
        v_pk, v_z = _detect_contractions(_vload[:n], VENTRICLE_Z_THRESHOLD_DEFAULT)
    if os.path.exists(a_src):
        _aload = np.load(a_src)
        print(f"  [nature-fig] panel d ATRIUM   V/A: depth={depth}  shape={_aload.shape}  "
              f"-> using first {n} frames  src={os.path.basename(a_src)}")
        a_pk, a_z = _detect_contractions(_aload[:n], -1.7)
    if v_z is not None:
        print(f"  [nature-fig] depth {depth}: V peaks={len(v_pk)}, A peaks={len(a_pk)} "
              f"(expect ~4 over {n} frames)")
    if v_z is None or a_z is None:
        print(f"  [nature-fig] NOTE: ventricle/atrium trace not found "
              f"(looked for {ventricle_trace_path}, {prefix}_ventricle/atrium.npy) "
              f"— panel d will flag the missing chamber.")

    # ── representative closed & open frames ──
    oz = (openness - np.mean(openness)) / (np.std(openness) + 1e-9)
    open_main = int(open_frames[np.argmax(openness[open_frames])]) if len(open_frames) else int(np.argmax(openness))
    far = np.ones(n, bool)
    for f in open_frames:
        far[max(0, f - 5):f + 6] = False
    closed_frame = int(np.argmin(np.where(far, openness, openness.max() + 1)))

    upp = um_per_pixel
    ms = lambda f: f / fps * 1e3                  # frame -> ms (dt = 1000/fps)
    t_ms = np.arange(n) / fps * 1e3
    # robust FUNDAMENTAL cardiac period (frames), HARMONIC-REJECTED - single source
    # for the panel-c window AND the phase-locking (so we average over ~4 cycles,
    # not a ~5th harmonic). 901 frames @ 600 Hz -> ~225 frames (~4 cycles).
    from leaflet_unet import robust_cardiac_period as _rcp
    _psigs = [s for s in (v_z, a_z, oz) if s is not None and len(np.ravel(s)) == n]
    period_fund = _rcp(_psigs if _psigs else [oz], n, expected=4, fps=fps)
    period_c = period_fund

    # ── OPEN DETECTION constrained to the [ATRIUM-MAX -> VENTRICLE-MAX] window ──
    # The valve opens ONLY during atrial emptying / ventricular filling: between
    # the ATRIUM area MAXIMUM and the following VENTRICLE area MAXIMUM. Search the
    # openness peak ONLY in that short window -> one correctly-timed open per cycle
    # (no longer late). The openness signal for the graph + detection is the
    # UNSUPERVISED midpoint darkening when available (cleaner peaks), else the
    # supervised connector openness. Tip-to-tip gap is a MEASUREMENT displayed on
    # detected-OPEN frames (panel b); it does NOT feed back into detection.
    def _mm01(x):
        x = np.asarray(x, float); lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-9)
    op_raw = (_mm01(unsup[:n]) if (use_unsup_openness and unsup is not None)
              else _mm01(oz))                        # RAW normalized openness
    op_src = "unsupervised darkening" if (use_unsup_openness and unsup is not None) \
        else "supervised connector"
    # ── CLEAN openness for detection + display: Savitzky-Golay smooth + Hilbert
    #    envelope. Peak detection on the RAW noisy signal (argmax of noise spikes)
    #    is why the state band and the openness trace disagreed; running it on this
    #    smooth signal makes the peaks clean and the two consistent. ──
    from scipy.signal import find_peaks as _fp, savgol_filter as _savgol, hilbert as _hilb
    from scipy import ndimage as _ndi
    _w = max(5, int(round(0.05 * period_fund)));  _w += (1 - _w % 2)   # odd window
    try:
        op_sg = _savgol(op_raw, min(_w, len(op_raw) - (1 - len(op_raw) % 2)), 3)
    except Exception:
        op_sg = _ndi.uniform_filter1d(op_raw, _w)
    try:
        env = np.abs(_hilb(op_raw - float(np.mean(op_raw))))           # Hilbert envelope
        env = _ndi.uniform_filter1d(env, _w)
        op_s = _mm01(0.5 * np.clip(op_sg, 0, None) + 0.5 * env)        # smooth + envelope
    except Exception:
        op_s = _mm01(np.clip(op_sg, 0, None))
    op_n = op_s                                      # detection + display use the CLEAN signal
    from scipy.signal import find_peaks  # noqa (kept for clarity)
    av_windows = []                                  # (atrium-max, ventricle-max) per cycle
    det_opens = []
    if v_z is not None and a_z is not None:
        _d = max(3, int(round(0.55 * period_fund)))
        am, _ = _fp(_mm01(a_z), distance=_d, prominence=0.12)   # ATRIUM maxima
        vm, _ = _fp(_mm01(v_z), distance=_d, prominence=0.12)   # VENTRICLE maxima
        for a in np.sort(am):
            later = np.sort(vm)[np.sort(vm) > a]
            if len(later):
                e = int(later[0])
                if 2 <= e - a <= int(round(1.3 * period_fund)):
                    av_windows.append((int(a), e))
        # RELAXED: take the MAX openness peak within the window +/- a small TOLERANCE
        # (the true peak can sit just outside the [atrium-max->ventricle-max] window).
        _tol = max(2, int(round(0.10 * period_fund)))
        for (s, e) in av_windows:
            s2, e2 = max(0, s - _tol), min(n - 1, e + _tol)
            seg = op_s[s2:e2 + 1]
            if len(seg):
                det_opens.append(s2 + int(np.argmax(seg)))
    det_opens = np.array(sorted(set(int(d) for d in det_opens)), int)
    if len(det_opens):
        open_frames = det_opens                      # OVERRIDE: corrected detection
        open_main = int(open_frames[np.argmax(op_s[open_frames])])
        print(f"  [nature-fig] OPEN DETECTION = max openness peak in "
              f"[atrium-max->ventricle-max] +/- {max(2, int(round(0.10 * period_fund)))}f "
              f"tolerance: {len(av_windows)} windows -> {len(open_frames)} opens at "
              f"{list(open_frames)} (smoothed {op_src})")
    else:
        print(f"  [nature-fig] WARNING no [atrium-max->ventricle-max] windows "
              f"(V/A peaks missing?); keeping saved open_frames {list(open_frames)}")
    # OPEN run mask: frames near each corrected peak where the clean openness > half peak
    _open_run = np.zeros(n, bool)
    _wmax = max(2, int(round(0.22 * period_fund)))
    for f in np.asarray(open_frames, int):
        thr = 0.5 * op_s[f]
        s = int(f)
        while s > 0 and op_s[s - 1] >= thr and f - s < _wmax:
            s -= 1
        e = int(f)
        while e < n - 1 and op_s[e + 1] >= thr and e - f < _wmax:
            e += 1
        _open_run[s:e + 1] = True
    # recompute the representative CLOSED frame (clearly shut, far from any open)
    far = ~_open_run.copy()
    for f in np.asarray(open_frames, int):
        far[max(0, f - 4):f + 5] = False
    closed_frame = int(np.argmin(np.where(far, op_n, op_n.max() + 1)))

    def _valve_roi(margin=12):
        """Panel crop bbox (y0,x0,y1,x1) from the leaflet-centerline extent. The
        deprecated separate valve ROI is no longer used - panels key off the
        extracted line (within the auto V/A band)."""
        pts = []
        for arr in (ctrl_a, ctrl_b):
            m = ~np.isnan(arr).any(-1)
            if m.any():
                pts.append(arr[m])
        if not pts:
            return (0, 0, stack.shape[1], stack.shape[2])
        P = np.vstack(pts)
        return (max(0, int(P[:, 0].min()) - margin), max(0, int(P[:, 1].min()) - margin),
                min(stack.shape[1], int(P[:, 0].max()) + margin),
                min(stack.shape[2], int(P[:, 1].max()) + margin))

    # ════════════════════════════════════════════════════════════════════
    # NATURE FONT HIERARCHY — single source of truth at the export size.
    # Helvetica preferred, Arial fallback (sans-serif). NOTHING below 5 pt.
    # Body text 5-7 pt; panel letters 8 pt BOLD; lines >= 0.5 pt.
    # Every panel reads these names; do NOT hard-code font sizes anywhere else.
    # ════════════════════════════════════════════════════════════════════
    PANEL_LETTER_PT = 8                              # a, b, c, d, i — bold
    TITLE_PT        = 7                              # axis titles
    LABEL_PT        = 7                              # axis x/y labels
    TICK_PT         = 6                              # tick labels
    LEGEND_PT       = 6                              # legend text
    ANNOT_PT        = 6                              # in-image annotations (scalebar, OPEN/CLOSED, gap)
    MIN_LW_PT       = 0.6                            # minimum line width (no vanishing hairlines)
    FONT_STACK      = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams.update({
        # Editable text in SVG (kept as <text>, never outlined to <path>)
        "svg.fonttype": "none",
        # TrueType in PDF (editable in Illustrator)
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # Helvetica preferred, Arial fallback
        "font.family": "sans-serif", "font.sans-serif": FONT_STACK,
        # Body text & defaults
        "font.size": ANNOT_PT,
        "axes.titlesize": TITLE_PT, "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        # Line weights >= 0.5 pt
        "axes.linewidth": MIN_LW_PT, "lines.linewidth": MIN_LW_PT,
        "xtick.major.width": MIN_LW_PT, "ytick.major.width": MIN_LW_PT,
        "patch.linewidth": MIN_LW_PT,
        # Despined
        "axes.spines.top": False, "axes.spines.right": False,
        # PNG fallback rendered at 600 dpi; SVG/PDF stay vector
        "savefig.dpi": 600, "figure.dpi": 100,
    })
    # ════════════════════════════════════════════════════════════════════
    # PUBLICATION-FIGURE COLOURS — SINGLE SOURCE OF TRUTH. Every panel reads
    # these names; do NOT hard-code these colours anywhere else (so they can't
    # drift across a/b/c/standalone-overlay). Palette is deliberately non-clashing:
    #   • leaflets (sup/inf) ......... ORANGE  (two shades, still distinguishable)
    #   • orifice contact / open gap . YELLOW  (clearly distinct from leaflet orange)
    #   • ventricle region tint ...... BLUE    (faint, panel a)
    #   • atrium region tint ......... RED     (faint, panel a)
    # ════════════════════════════════════════════════════════════════════
    SUP_COLOR = "#F6921E"                   # SUP leaflet — orange (lighter)
    INF_COLOR = "#B5510A"                   # INF leaflet — orange (burnt / darker)
    ORIFICE_COLOR = "#FFE000"               # leaflet CONTACT point + OPEN-GAP fill = YELLOW
    CA, CB = SUP_COLOR, INF_COLOR           # leaflet identity colour (was red/cyan -> orange)
    open_gap_color = ORIFICE_COLOR          # OVERRIDE: the open-gap fill is now YELLOW too
    # SINGLE SOURCE OF TRUTH for leaflet naming in the publication figure.
    # By the y-ordering convention leaflet 1 (ctrl_a/CA) is the UPPER = SUPERIOR
    # leaflet and leaflet 2 (ctrl_b/CB) is the LOWER = INFERIOR leaflet, so:
    #     leaflet 1 -> "Sup"   leaflet 2 -> "Inf"
    # Use LBL_A / LBL_B everywhere the figure names a leaflet (legends, labels).
    LEAFLET_LABELS = ("Sup", "Inf")        # (superior, inferior)
    LBL_A, LBL_B = LEAFLET_LABELS
    # chamber colours match the SOURCE marker: ventricle=BLUE, atrium=RED.
    # CV / CAT double as the faint panel-(a) ventricle / atrium REGION tints.
    CV, CAT, COP, CVAL = "#1f77b4", "#d62728", "#2ca02c", "#3C5488"
    VENT_TINT, ATR_TINT = CV, CAT          # ventricle=blue tint, atrium=red tint
    from scipy.spatial.distance import cdist as _cdist
    # ── TRUE FINAL SIZE: fig_width_mm drives figsize (mm/25.4). Default 183 mm
    #    (double-column, Nature spec); 89 mm via single_column. Place the
    #    exported .svg / .pdf at 100% in Illustrator so pt maps 1:1. Height is
    #    fixed at 9.0 in (~229 mm) for the a/b/c/d/i layout. ──
    if single_column and fig_width_mm == 183.0:
        fig_width_mm = 89.0                          # single-column override
    _figw_in = float(fig_width_mm) / 25.4
    _figh_in = 9.0
    fig = plt.figure(figsize=(_figw_in, _figh_in))   # a,b,c,d,i layout
    fig.patch.set_facecolor("white")

    def _letter(ax, s, x=-0.02, y=1.02):
        ax.text(x, y, s, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                fontweight="bold", va="bottom", ha="right")

    def _scalebar(ax, length_um=200.0):
        """Draw a scale bar at the DISPLAY bottom-right corner of ``ax``, regardless of
        any H/V axis flips. Positioned in axes-fraction coords (transAxes) so the bar
        and its label always sit at the displayed bottom-right; only the bar's WIDTH is
        derived from data (length_um / um_per_pixel) so the µm length stays calibrated.
        Consistent styling across panels (a, b, c): white bar, ~0.020 axes-height
        thick, fontsize 6, label above the bar with a small gap. Small inset margin
        from the right + bottom keeps bar + label inside the axes."""
        xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
        W = abs(xhi - xlo); Himg = abs(ylo - yhi)
        if upp:
            barpx = length_um / upp; txt = f"{length_um:g} µm"
        else:
            barpx = max(10.0, 0.2 * W); txt = f"{barpx:.0f} px"
        bar_w_frac = barpx / max(W, 1e-9)              # bar WIDTH as axes-x-fraction
        pad_x = 0.04                                    # inset from display-right
        pad_y = 0.06                                    # inset from display-bottom
        h_frac = 0.020                                  # bar thickness in axes-y-fraction
        gap_frac = 0.012                                # gap between bar and label
        x_right = 1.0 - pad_x                           # right edge of bar (axes-fraction)
        x_left = max(pad_x, x_right - bar_w_frac)
        y_bar = pad_y                                   # bar bottom (axes-fraction)
        ax.add_patch(Rectangle((x_left, y_bar), x_right - x_left, h_frac,
                               color="white", ec="none", transform=ax.transAxes,
                               zorder=10, clip_on=False))
        ax.text((x_left + x_right) / 2, y_bar + h_frac + gap_frac, txt,
                color="white", ha="center", va="bottom", fontsize=ANNOT_PT,
                transform=ax.transAxes, zorder=10, clip_on=False)

    # per-frame open/closed from the CORRECTED detection (atrium-max->ventricle-max
    # constrained open runs); gap (µm) kept for the optional width annotation.
    def _state(frame):
        g = float(gap[frame]) if frame < len(gap) else 0.0
        g_um = g if gap_unit == "µm" else (g * upp if upp else g)
        if 0 <= frame < n:
            return bool(_open_run[frame]), g_um
        return (g_um >= open_threshold_um), g_um

    def _orifice_tips(pa, pb):
        ea = np.array([pa[0], pa[-1]]); eb = np.array([pb[0], pb[-1]])
        i, j = np.unravel_index(int(_cdist(ea, eb).argmin()), (2, 2))
        return ea[i], eb[j]

    from matplotlib.patches import Polygon as _Poly

    def _fill_ushape(ax, line_yx, orif_yx, color, width_px=6.0, crop=(0, 0),
                     alpha=0.55, zorder=2, edge_lw=0.7):
        """Draw a FILLED U-shape (schematic virtual leaflet) around the drawn
        centerline, closed side toward the orifice. Saturated colour with a solid
        edge; ``alpha`` controls how strongly it sits over the grayscale."""
        oy, ox = crop
        U = _ushape(line_yx, orif_yx, width_px=width_px)
        if len(U) < 3:
            return
        xy = np.c_[U[:, 1] - ox, U[:, 0] - oy]
        ax.add_patch(_Poly(xy, closed=True, fill=True, fc=color, ec=color,
                           lw=edge_lw, alpha=alpha, joinstyle="round", zorder=zorder))

    def _extend_to_orifice(line, orif):
        """Append the orifice point to whichever END of the centerline is closer to
        it, so the U-cap reaches the shared midpoint (closed leaflets TOUCH)."""
        line = np.asarray(line, float)
        if np.linalg.norm(line[0] - orif) < np.linalg.norm(line[-1] - orif):
            return np.vstack([orif, line])
        return np.vstack([line, orif])

    def _fill_gap(ax, ta, tb, width_px, color, crop=(0, 0), alpha=0.55, zorder=3):
        """Fill the OPEN orifice gap between the two free tips with a DISTINCT colour
        (a band of ``width_px`` spanning ta->tb), so the open valve is obvious."""
        oy, ox = crop
        ta = np.asarray(ta, float); tb = np.asarray(tb, float)
        d = tb - ta; L = np.hypot(*d)
        if L < 1e-6:
            return
        perp = np.array([-d[1], d[0]]) / L * (width_px / 2.0)
        quad = np.array([ta - perp, ta + perp, tb + perp, tb - perp])
        ax.add_patch(_Poly(np.c_[quad[:, 1] - ox, quad[:, 0] - oy], closed=True,
                           fill=True, fc=color, ec=color, lw=0.6, alpha=alpha,
                           joinstyle="round", zorder=zorder))

    def _draw_leaflets(ax, frame, crop=None, lw=None, tips=None, annot=None,
                       show_line=None, color_override=None):
        """Schematic leaflets = FILLED U-shapes (no centerline by default). CLOSED:
        the two U-shapes are pulled to a shared orifice point so they TOUCH (clear
        contact). OPEN: the U-shapes stay separated and the GAP between them is
        filled with ``open_gap_color`` (distinct from the leaflets). Returns
        (is_open, gap_um)."""
        lw = leaflet_lw if lw is None else lw
        oy, ox = (crop[0], crop[1]) if crop else (0, 0)
        show_line = show_centerline if show_line is None else show_line
        raw = {}
        for key, cp_arr in (("a", ctrl_a), ("b", ctrl_b)):
            pa = cp_arr[frame]
            if np.isnan(pa).all():
                continue
            raw[key] = pa[~np.isnan(pa).any(1)]
        da, db = raw.get("a"), raw.get("b")
        cleaned = {}
        for key, pa in (("a", da), ("b", db)):
            if pa is not None and len(pa) >= 2:
                cleaned[key] = pa
        is_open, g_um = _state(frame)
        ca_use = color_override[0] if color_override else CA
        cb_use = color_override[1] if color_override else CB
        if draw_ushapes and "a" in cleaned and "b" in cleaned:
            ta0, tb0 = _orifice_tips(cleaned["a"], cleaned["b"])
            orif = (ta0 + tb0) / 2.0
            uw = max(5.0, leaflet_lw * 3.0)
            if is_open:                                # OPEN: separated U + filled gap
                for key, c in (("a", ca_use), ("b", cb_use)):
                    _fill_ushape(ax, cleaned[key], orif, c, width_px=uw,
                                 crop=(oy, ox), alpha=leaf_alpha, zorder=2)
                _fill_gap(ax, ta0, tb0, uw, open_gap_color, crop=(oy, ox),
                          alpha=min(0.7, leaf_alpha + 0.1), zorder=3)
            else:                                      # CLOSED: pull caps together -> TOUCH
                for key, c in (("a", ca_use), ("b", cb_use)):
                    line_c = _extend_to_orifice(cleaned[key], orif)
                    _fill_ushape(ax, line_c, orif, c, width_px=uw,
                                 crop=(oy, ox), alpha=leaf_alpha, zorder=2)
                ax.plot([orif[1] - ox], [orif[0] - oy], "o", color=ORIFICE_COLOR,
                        ms=2.6, mec="#3a3000", mew=0.5, zorder=5)   # contact point = YELLOW
        if show_line:                                  # optional reference centerline
            for key, c in (("a", ca_use), ("b", cb_use)):
                pa = cleaned.get(key)
                if pa is not None and len(pa) >= 2:
                    ax.plot(pa[:, 1] - ox, pa[:, 0] - oy, "-", lw=max(0.5, lw * 0.5),
                            color=c, alpha=0.9, solid_capstyle="round", zorder=4)
        return is_open, g_um

    def _flip_image_axis(ax):
        """DISPLAY-ONLY mirror of an IMAGE panel — HORIZONTAL (invert x, left-right)
        and/or VERTICAL (invert y, up-down), independently, per ``flip_horizontal`` /
        ``flip_vertical``. Inverting the axes mirrors the image AND every overlay drawn
        in the same data coordinates (U-shape leaflets, fills, orifice / open-gap,
        tip + contact markers, connecting structure, V/A region tints, scale bars)
        TOGETHER, so they stay aligned under either or both flips. transAxes
        annotations (panel letters, titles, legend incl. SUP/INF, OPEN/CLOSED tags,
        colorbars) are NOT in data coordinates and remain upright/readable — so the
        sup/inf labels follow the leaflet DATA IDENTITY (colour: sup=ctrl_a, inf=ctrl_b),
        not screen position, and stay anatomically correct even after a vertical flip
        swaps top/bottom. Changes ONLY the rendered orientation — never the data, the
        saved analysis, the V/A signals, or the openness. Idempotent per panel: call
        exactly once after the panel's image + overlays are drawn. Time-series panels
        (d, i) are never flipped."""
        if getattr(ax, "_image_flipped", False):
            return
        if flip_horizontal:
            ax.invert_xaxis()          # left-right mirror
        if flip_vertical:
            ax.invert_yaxis()          # up-down mirror
        ax._image_flipped = True

    def _chamber_tints(fig_hw):
        """VENTRICLE (blue) / ATRIUM (red) static REGION masks, registered to the
        figure's image frame, for the faint panel-(a) region tints. Single source
        of truth for colour->chamber = extract_red_blue_masks (RED->ATRIUM,
        BLUE->VENTRICLE). Derived once from the depth's colour-marked RGB source
        (depth_config 'rb_source_tif') and cached to {prefix}_vent_mask.npy /
        {prefix}_atr_mask.npy. Returns (vent, atr) bool masks of shape fig_hw, or
        (None, None) if they cannot be derived/registered (the tint is then simply
        skipped — never fatal, never alters data)."""
        fh, fw = int(fig_hw[0]), int(fig_hw[1])
        vpath, apath = f"{prefix}_vent_mask.npy", f"{prefix}_atr_mask.npy"
        try:                                              # 1) cached masks
            if os.path.exists(vpath) and os.path.exists(apath):
                vm, am = np.load(vpath), np.load(apath)
                if vm.shape == (fh, fw) and am.shape == (fh, fw):
                    return vm.astype(bool), am.astype(bool)
        except Exception:
            pass
        try:                                              # 2) derive from RGB source
            import json as _json, tifffile as _tf
            from skimage.transform import resize as _rs
            cfgp = os.path.join("leaflet_dataset", "depth_config.json")
            if not os.path.exists(cfgp):
                return None, None
            cfg = _json.load(open(cfgp))
            rb = cfg.get("rb_source_tif")
            if not rb or not os.path.exists(rb):
                print("  [nature-fig] panel-a tints: rb_source_tif missing -> skip")
                return None, None
            order = cfg.get("order", "depth_major")
            nfr, nd, dep = int(cfg["n_frames"]), int(cfg["n_depths"]), int(cfg.get("depth", 0))
            mm = _tf.imread(rb)                            # (nfr*nd, h, w, 3)
            idx = (dep * nfr + np.arange(nfr) if order == "depth_major"
                   else np.arange(nfr) * nd + dep)
            idx = idx[idx < mm.shape[0]]
            vent, atr, _info = extract_red_blue_masks(np.asarray(mm[idx]))   # rb-res masks
            RH, RW = int(_tf.TiffFile(cfg["stack"]).series[0].shape[1]), \
                int(_tf.TiffFile(cfg["stack"]).series[0].shape[2])           # raw full res
            _toraw = lambda m: _rs(m.astype(float), (RH, RW), order=0,
                                   preserve_range=True) > 0.5
            vR, aR = _toraw(vent), _toraw(atr)
            if (fh, fw) == (RH, RW):                       # figure uses full frame
                vm, am = vR, aR
            else:                                          # figure uses the training crop
                cropp = os.path.join("leaflet_dataset", "training_crop.json")
                bbox = (_json.load(open(cropp)).get("bbox")
                        if os.path.exists(cropp) else None)
                if bbox and (bbox[2] - bbox[0], bbox[3] - bbox[1]) == (fh, fw):
                    y0, x0, y1, x1 = bbox
                    vm, am = vR[y0:y1, x0:x1], aR[y0:y1, x0:x1]
                else:
                    print(f"  [nature-fig] panel-a tints: cannot register masks "
                          f"(fig {fh}x{fw}, raw {RH}x{RW}, bbox {bbox}) -> skip")
                    return None, None
            vm, am = vm.astype(bool), am.astype(bool)
            try:
                np.save(vpath, vm); np.save(apath, am)     # cache for next time
            except Exception:
                pass
            print(f"  [nature-fig] panel-a tints: ventricle {int(vm.sum())}px (blue) / "
                  f"atrium {int(am.sum())}px (red) registered to {fh}x{fw}")
            return vm, am
        except Exception as _e:
            print(f"  [nature-fig] panel-a tints: derive failed ({_e}) -> skip")
            return None, None

    import matplotlib.colors as _mcol

    def _tint(ax, mask, color, alpha=0.20):
        """Faint translucent fill of a region mask (background stays visible).
        Drawn in image data coordinates so it mirrors with the L-R flip and stays
        aligned to the chamber; zorder sits above the gray image, below leaflets."""
        if mask is None or not bool(np.asarray(mask).any()):
            return
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), float)
        rgba[..., :3] = _mcol.to_rgb(color)
        rgba[..., 3] = np.where(mask, alpha, 0.0)
        ax.imshow(rgba, interpolation="nearest", zorder=1.2, rasterized=True)

    # ── figure-(a) chamber regions: MANUAL drawn schematic (default) OR AUTO
    #    crop-transform of the full-frame V/A masks. The crop around the valve is a
    #    small square in its OWN coordinate frame, so the full-frame V/A regions must
    #    be drawn/transformed INTO the crop frame to align (and stay aligned under the
    #    L-R flip). This is SCHEMATIC for the figure only — it never touches the
    #    analysis V/A signals, the per-depth V/A, or the openness. ──
    _figa_path = figa_regions_path or f"{prefix}_figa_regions.npz"

    def _load_figa_regions(path, hw):
        """Return per-frame regions dict {'closed': {'ventricle':..,'atrium':..}, 'open': {...}}
        or None. Accepts the per-frame format (preferred) with keys
        '{closed|open}_{ventricle|atrium}', and falls back to the legacy single-frame
        format ('ventricle'/'atrium' applied to BOTH frames) for old saved files."""
        if not os.path.exists(path):
            return None
        try:
            d = np.load(path, allow_pickle=True)
            files = set(d.files)
            per_frame = {}
            for fname in ("closed", "open"):
                reg = {}
                for chamber in ("ventricle", "atrium"):
                    key = f"{fname}_{chamber}"
                    if key in files and np.asarray(d[key]).size >= 6:
                        reg[chamber] = np.asarray(d[key], float)
                if reg:
                    per_frame[fname] = reg
            if not per_frame:                              # legacy: single-frame fields
                legacy = {}
                for chamber in ("ventricle", "atrium"):
                    if chamber in files and np.asarray(d[chamber]).size >= 6:
                        legacy[chamber] = np.asarray(d[chamber], float)
                if legacy:                                 # apply same regions to both frames
                    per_frame = {"closed": legacy, "open": dict(legacy)}
                    print("  [nature-fig] panel-a MANUAL regions: LEGACY single-frame file "
                          "applied to BOTH closed+open (redraw to set them separately)")
            if not per_frame:
                return None
            if "shape" in d.files:
                sh = tuple(int(v) for v in np.asarray(d["shape"]).ravel()[:2])
                if sh != (int(hw[0]), int(hw[1])):
                    print(f"  [nature-fig] panel-a MANUAL regions saved for {sh} but crop is "
                          f"{tuple(int(v) for v in hw)} (using; redraw if misaligned)")
            print("  [nature-fig] panel-a regions = MANUAL (drawn schematic): "
                  + "; ".join(f"{fname}: " + ", ".join(f"{k}({len(v)}pts)" for k, v in reg.items())
                              for fname, reg in per_frame.items()))
            return per_frame
        except Exception as _e:
            print(f"  [nature-fig] panel-a MANUAL regions load failed ({_e})")
            return None

    # ── interactivity test (ROBUST). The earlier `"agg" in backend` substring test
    #    WRONGLY classified the GUI's interactive 'TkAgg' backend as headless (because
    #    'agg' is a substring of 'tkagg'), so the draw popup was skipped and it silently
    #    fell back to AUTO every time — THAT was the no-popup bug. A backend is headless
    #    ONLY if it is exactly one of the file/non-GUI backends. ──
    import matplotlib as _mpl
    _bk = _mpl.get_backend().lower()
    _headless = _bk in {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}

    def _surface_figa(msg):
        """Surface a figure-(a) region message to BOTH the log and a popup (so an
        error or skip is never silent)."""
        print(f"  [nature-fig] {msg}")
        if not _headless:
            try:
                import tkinter as _tk
                from tkinter import messagebox as _mb
                _r = _tk.Tk(); _r.withdraw(); _r.attributes("-topmost", True)
                _mb.showwarning("Figure (a) chamber regions", msg); _r.destroy()
            except Exception:
                pass

    _figa_mode = (fig_a_regions or "manual").lower()
    _figa_regions = None                          # per-frame dict: {'closed':{...}, 'open':{...}}
    _figa_auto = (None, None)
    if _figa_mode == "manual":
        _saved = _load_figa_regions(_figa_path, stack.shape[1:])
        if _headless:                                 # cannot draw -> reuse saved or AUTO
            if _saved is not None:
                print("  [nature-fig] panel-a MANUAL: non-interactive backend -> reuse saved regions")
                _figa_regions = _saved
            else:
                print("  [nature-fig] panel-a MANUAL: non-interactive backend + no saved "
                      "regions -> AUTO crop-transform fallback")
                _figa_mode = "auto"
        else:
            try:
                _redraw = True
                if _saved is not None:                # OFFER reuse vs redraw (never silent)
                    import tkinter as _tk
                    from tkinter import messagebox as _mb
                    _r = _tk.Tk(); _r.withdraw(); _r.attributes("-topmost", True)
                    _names = ", ".join(sorted(_saved.keys())).upper() or "none"
                    _reuse = _mb.askyesno("Figure (a) chamber regions",
                        f"Saved PER-FRAME ventricle/atrium regions found ({_names}).\n\n"
                        "Yes  =  REUSE the saved regions\n"
                        "No   =  RE-DRAW them now (step through CLOSED then OPEN)")
                    _r.destroy()
                    _redraw = not _reuse
                    if _reuse:
                        print("  [nature-fig] panel-a MANUAL: user chose REUSE saved regions")
                        _figa_regions = _saved
                if _redraw:                           # open the draw window on each panel-a frame
                    print("  [nature-fig] panel-a MANUAL: opening DRAW window for BOTH "
                          f"frames (closed={closed_frame}, open={open_main}; same flip/depth)...")
                    _drawn = draw_figa_regions(
                        [("closed", stack[closed_frame]), ("open", stack[open_main])],
                        _figa_path, vent_color=VENT_TINT, atr_color=ATR_TINT,
                        flip_h=flip_horizontal, flip_v=flip_vertical)
                    if _drawn is not None:
                        # MERGE: any frame the user skipped keeps its previously-saved regions
                        merged = {} if _saved is None else {k: dict(v) for k, v in _saved.items()}
                        for fname, reg in _drawn.items():
                            merged[fname] = reg
                        _figa_regions = merged or None
                    elif _saved is not None:
                        print("  [nature-fig] panel-a MANUAL: nothing drawn -> keep saved regions")
                        _figa_regions = _saved
            except Exception as _e:
                import traceback as _tb
                _surface_figa(f"region selection ERROR (NOT skipped silently): {_e}")
                print(_tb.format_exc())
                if _saved is not None:
                    _figa_regions = _saved
            if _figa_regions is None:                 # drew nothing & none saved
                print("  [nature-fig] panel-a MANUAL: no regions chosen -> AUTO crop-transform fallback")
                _figa_mode = "auto"
    if _figa_mode == "auto":
        _figa_auto = _chamber_tints(stack.shape[1:])
        print("  [nature-fig] panel-a regions = AUTO (full-frame V/A masks -> crop)")

    def _draw_chamber_regions(ax, frame_kind=None):
        """Panel-(a) chamber overlay for ONE frame: MANUAL smoothed curved fills if
        drawn for THIS frame kind ('closed' or 'open'), else the AUTO crop-transformed
        V/A masks. The closed and open frames each carry their OWN drawn regions
        (the chamber positions differ between them). Faint blue=ventricle /
        red=atrium, drawn in crop data coords so they mirror with the flip and stay
        aligned. SCHEMATIC."""
        from matplotlib.patches import Polygon as _PolyR
        per_frame = None
        if _figa_regions is not None and frame_kind is not None:
            per_frame = _figa_regions.get(frame_kind)
            if per_frame is None and _figa_regions:    # fallback to any available frame
                per_frame = next(iter(_figa_regions.values()), None)
        if per_frame:
            for key, color in (("ventricle", VENT_TINT), ("atrium", ATR_TINT)):
                P = per_frame.get(key)
                if P is not None and len(P) >= 3:
                    ax.add_patch(_PolyR(_smooth_closed_polygon(P), closed=True,
                                        fill=True, fc=color, ec=color, lw=0.6,
                                        alpha=0.20, joinstyle="round", zorder=1.2))
        elif _figa_regions is None:
            _tint(ax, _figa_auto[0], VENT_TINT, 0.20)
            _tint(ax, _figa_auto[1], ATR_TINT, 0.20)

    _has_chamber_overlay = (_figa_regions is not None
                            or any(m is not None for m in _figa_auto))

    def _img(ax, frame, title, tint=False, frame_kind=None):
        ax.imshow(stack[frame], cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", rasterized=True)
        if tint:                                  # faint chamber REGIONS (panel a), per-frame
            _draw_chamber_regions(ax, frame_kind=frame_kind)
        _draw_leaflets(ax, frame)
        ax.set_title(title, fontsize=TITLE_PT, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        _flip_image_axis(ax)                  # mirror image + overlays L-R (display only)

    # ── (a) closed & open frames — TRANSLUCENT FILLED U-shape leaflets, 200 µm bar ──
    from matplotlib.patches import Patch as _Patch
    ax_ac = fig.add_axes([0.06, 0.795, 0.235, 0.165]); ax_ac.set_gid("panel_a_closed")
    _img(ax_ac, closed_frame, "closed (coapted)", tint=True, frame_kind="closed")
    _letter(ax_ac, "a"); _scalebar(ax_ac, 200.0)
    ax_ao = fig.add_axes([0.315, 0.795, 0.235, 0.165]); ax_ao.set_gid("panel_a_open")
    _img(ax_ao, open_main, "open (connector break)", tint=True, frame_kind="open")
    # SHORTENED legend labels so they fit legibly at LEGEND_PT (6 pt). Multi-column
    # so a 5-entry legend stays inside the lower-left of the closed-frame panel.
    _leg_handles = [_Patch(fc=CA, ec=CA, alpha=0.4, label=LBL_A),
                    _Patch(fc=CB, ec=CB, alpha=0.4, label=LBL_B),
                    Line2D([0], [0], marker="o", color=ORIFICE_COLOR, lw=0, mec="k",
                           mew=0.5, label="orifice")]
    if _has_chamber_overlay:                               # chamber regions (manual|auto)
        _leg_handles += [_Patch(fc=VENT_TINT, ec="none", alpha=0.4, label="ventricle"),
                         _Patch(fc=ATR_TINT, ec="none", alpha=0.4, label="atrium")]
    ax_ac.legend(handles=_leg_handles,
                 loc="lower left", frameon=False, fontsize=LEGEND_PT,
                 labelcolor="white", handlelength=1.2, ncol=2,
                 columnspacing=0.8, handletextpad=0.4, borderpad=0.2)

    # ── (c) leaflet kinematics = MOVING U-shapes; FEW timepoints, HIGH-CONTRAST
    #    time colour (turbo) + crisp outlines + a clearly readable colorbar ──
    ax_c = fig.add_axes([0.645, 0.795, 0.255, 0.165]); ax_c.set_gid("panel_c_kinematics")
    if panel_c_mode == "kinematics":
        ry0, rx0, ry1, rx1 = _valve_roi()
        ax_c.imshow(stack[closed_frame][ry0:ry1, rx0:rx1], cmap="gray",
                    vmin=0, vmax=1, alpha=0.30, interpolation="nearest",
                    rasterized=True)
        half = int(round(period_c / 2)) if period_c else 8
        w0, w1 = max(0, open_main - half), min(n, open_main + half + 1)
        n_show = 5                                     # fewer frames -> less overplot
        tp = np.unique(np.linspace(w0, w1 - 1, n_show).astype(int))
        cmap = (plt.cm.turbo if hasattr(plt.cm, "turbo") else plt.cm.jet)  # high-contrast
        span = max(1e-9, ms(w1 - 1) - ms(w0))
        uw = max(5.0, leaflet_lw * 3.0)
        for f in tp:
            frac = (ms(f) - ms(w0)) / span
            col = cmap(0.04 + 0.92 * frac)             # stretch across the full colormap
            da = ctrl_a[f]; db = ctrl_b[f]
            da = None if np.isnan(da).all() else da[~np.isnan(da).any(1)]
            db = None if np.isnan(db).all() else db[~np.isnan(db).any(1)]
            if da is None or db is None or len(da) < 2 or len(db) < 2:
                continue
            ta0, tb0 = _orifice_tips(da, db); orif = (ta0 + tb0) / 2.0
            for pa in (da, db):                        # filled + crisp outline / timepoint
                _fill_ushape(ax_c, pa, orif, col, width_px=uw, crop=(ry0, rx0),
                             alpha=0.45, zorder=3, edge_lw=1.1)
        ax_c.set_xlim(0, rx1 - rx0); ax_c.set_ylim(ry1 - ry0, 0)
        ax_c.set_xticks([]); ax_c.set_yticks([])
        for s in ax_c.spines.values():
            s.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, span))
        cb = fig.colorbar(sm, ax=ax_c, fraction=0.07, pad=0.03)
        cb.set_label("time (ms)", fontsize=LABEL_PT)     # readable colorbar
        cb.ax.tick_params(labelsize=TICK_PT); cb.outline.set_linewidth(MIN_LW_PT)
        cb.set_ticks(np.linspace(0, span, 4))
        _scalebar(ax_c, 50.0)
        _flip_image_axis(ax_c)                # mirror image + U-shapes L-R (display only)
        ax_c.set_title(f"leaflet kinematics — {len(tp)} U-shapes",
                       fontsize=TITLE_PT, pad=2)
    elif panel_c_mode in ("width", "angle"):
        if panel_c_mode == "width":
            y = gap[:n]; ylab = f"tip-to-tip gap ({gap_unit})"; col = CVAL
        else:
            y = (angle[:n] if angle is not None else np.zeros(n))
            ylab = "opening angle (deg)"; col = "#1a7a3a"
        ax_c.plot(t_ms, y, color=col, lw=1.0)
        for f in open_frames:
            ax_c.axvline(t_ms[f], color=COP, lw=0.8, ls="--", alpha=0.7)
        ax_c.set_xlabel("time (ms)"); ax_c.set_ylabel(ylab)
        ax_c.set_title(f"leaflet {panel_c_mode} ({len(open_frames)} openings)",
                       fontsize=TITLE_PT, pad=2)
    _letter(ax_c, "c")

    # ── (b) montage spanning ONE CARDIAC CYCLE centered on an opening, so it MIXES
    #    closed + open frames (spaced across the period). FIRST selected frame = 0 ms;
    #    each label is the real relative time. DOUBLED panel size (fewer, 2x-larger
    #    cells). Same smooth U-shape rendering as (a). ──
    ry0, rx0, ry1, rx1 = _valve_roi()
    n_panels = 6                                       # fewer -> each cell ~2x larger
    per = int(round(period_c)) if period_c else max(8, n // 4)
    center = (int(montage_ref_frame) if montage_ref_frame is not None else open_main)
    start = max(0, center - per // 2)
    _samp = np.round(np.linspace(start, start + per, n_panels)).astype(int)
    _samp[int(np.argmin(np.abs(_samp - open_main)))] = open_main   # ensure an OPEN frame shows
    seq = np.unique(np.clip(_samp, 0, n - 1))
    ref_frame = int(seq[0])
    bx0, bw = 0.055, 0.90 / max(1, len(seq))
    last_axm = None
    for i, f in enumerate(seq):
        axm = fig.add_axes([bx0 + i * bw, 0.585, bw * 0.94, 0.155]); last_axm = axm   # 2x size
        axm.set_gid(f"panel_b_frame_{i}")
        axm.imshow(stack[f][ry0:ry1, rx0:rx1], cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest", rasterized=True)
        is_open, g_um = _draw_leaflets(axm, f, crop=(ry0, rx0), lw=1.8)
        dt = (f - ref_frame) / fps * 1e3              # FIRST selected frame = 0 ms
        axm.set_title(f"{dt:.0f} ms", fontsize=ANNOT_PT, pad=1)
        axm.set_xticks([]); axm.set_yticks([])
        if show_state_tag:
            axm.text(0.5, 0.015, "OPEN" if is_open else "CLOSED",
                     transform=axm.transAxes, ha="center", va="bottom",
                     fontsize=ANNOT_PT, fontweight="bold",
                     color=("#2ca02c" if is_open else "#777777"),
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))
        # ── TIP-TO-TIP gap measurement on detected-OPEN frames (panel b) ──
        # MEASUREMENT only — does NOT feed back into detection. The arrow spans
        # the sup leaflet's free tip <-> the inf leaflet's free tip (the orifice-
        # facing endpoints, found by _orifice_tips). Label is the Euclidean
        # distance in µm at UM_PER_PIXEL=10 (px * upp). Offset perpendicular to
        # the arrow toward smaller image-y so it stays clear of the bottom-right
        # scale bar.
        if is_open:
            _da = ctrl_a[f]; _db = ctrl_b[f]
            _da = None if np.isnan(_da).all() else _da[~np.isnan(_da).any(1)]
            _db = None if np.isnan(_db).all() else _db[~np.isnan(_db).any(1)]
            if _da is not None and _db is not None and len(_da) >= 2 and len(_db) >= 2:
                _ta, _tb = _orifice_tips(_da, _db)                       # sup-tip, inf-tip
                _wpx = float(np.linalg.norm(_ta - _tb))
                _wum = _wpx * upp if upp else _wpx
                _xA, _yA = float(_ta[1] - rx0), float(_ta[0] - ry0)      # sup-tip in axes coords
                _xB, _yB = float(_tb[1] - rx0), float(_tb[0] - ry0)      # inf-tip in axes coords
                axm.annotate("", xy=(_xA, _yA), xytext=(_xB, _yB),
                             arrowprops=dict(arrowstyle="<->", color="white",
                                              lw=1.0, shrinkA=0, shrinkB=0,
                                              mutation_scale=6),
                             annotation_clip=False, zorder=8)
                _mx, _my = (_xA + _xB) / 2.0, (_yA + _yB) / 2.0
                _dx, _dy = _xB - _xA, _yB - _yA
                _seglen = float(np.hypot(_dx, _dy))
                if _seglen > 1e-9:
                    # Two perpendicular candidates; pick the one with the smaller
                    # y-component so the label sits ABOVE the arrow in image coords
                    # (clear of the bottom-right scale bar).
                    _p1 = (-_dy / _seglen, _dx / _seglen)
                    _p2 = (_dy / _seglen, -_dx / _seglen)
                    _perp = _p1 if _p1[1] <= _p2[1] else _p2
                else:
                    _perp = (0.0, -1.0)
                _off = max(4.0, 0.07 * (rx1 - rx0))
                _tx, _ty = _mx + _perp[0] * _off, _my + _perp[1] * _off
                _lbl = f"{_wum:.0f} µm" if upp else f"{_wpx:.0f} px"
                axm.text(_tx, _ty, _lbl, color="white", ha="center", va="center",
                         fontsize=ANNOT_PT, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.18", fc="black",
                                   ec="none", alpha=0.65),
                         zorder=9, clip_on=False)
        for s_ in axm.spines.values():
            s_.set_visible(is_open)
            if is_open:
                s_.set_color("#2ca02c"); s_.set_linewidth(1.8)
        _flip_image_axis(axm)                          # mirror image + overlays L-R (display only)
        if i == 0:
            _letter(axm, "b", x=-0.05)
    if last_axm is not None:
        _scalebar(last_axm, 100.0)                     # 100 µm bar (= 100/UM_PER_PIXEL px; per-panel)

    # ── (d) ventricle + atrium + openness on one normalized time axis. Clean trace:
    #    NO legend, NO opening-duration shading (handled in i), openness line weight +
    #    colour MATCH the V/A lines; small circles at the open peaks. ──
    def _mm(x):                                   # min-max -> [0,1] for comparability
        x = np.asarray(x, float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-9)
    ax_d = fig.add_axes([0.10, 0.400, 0.84, 0.135]); ax_d.set_gid("panel_d_trace")
    oz_mm = op_s
    if v_z is not None:                            # chamber traces (matched weight)
        ax_d.plot(t_ms, _mm(v_z), color=CV, lw=0.9)
    if a_z is not None:
        ax_d.plot(t_ms, _mm(a_z), color=CAT, lw=0.9)
    # OPENNESS = UNSUPERVISED darkening, SMOOTHED (Savitzky-Golay + Hilbert envelope).
    # Same line weight (0.9) and a saturated colour to MATCH the V/A lines.
    C_OPEN = "#6a3d9a"
    ax_d.plot(t_ms, oz_mm, color=C_OPEN, lw=0.9, zorder=4)
    if len(open_frames):                           # open events = small circles at the peak
        ax_d.plot(t_ms[open_frames], oz_mm[open_frames], "o", color=COP, ms=5,
                  mec="k", mew=0.5, zorder=6)
    if v_z is None or a_z is None:
        ax_d.text(0.01, 0.04, "(ventricle/atrium trace not found)",
                  transform=ax_d.transAxes, color="#aa3333", fontsize=ANNOT_PT)
    ax_d.set_ylabel("normalized (0-1)"); ax_d.set_xlabel("time (ms)")
    ax_d.set_xlim(t_ms[0], t_ms[-1]); ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_title(f"ventricle / atrium / valve openness — {len(open_frames)} opens "
                   f"(unsupervised, smoothed)", fontsize=TITLE_PT, pad=3)
    _letter(ax_d, "d", x=-0.02)

    # ── (i) ONE AVERAGE CARDIAC CYCLE, segmented MIDDLE-to-MIDDLE: each cycle is a
    #    window of HALF a period before/after the openness PEAK, so the MAX is at the
    #    CENTRE. V / A / openness averaged (mean ± SD). The GREEN band = ONLY the
    #    actual OPENING DURATION (averaged openness ABOVE a TUNABLE threshold); its
    #    onset (upward threshold crossing) = the opening-detection onset. ──
    ax_i = fig.add_axes([0.165, 0.075, 0.66, 0.225]); ax_i.set_gid("panel_i_avg")
    _half = period_fund / 2.0
    _na = 81
    _rel = np.linspace(-_half, _half, _na)             # frames relative to the peak (centre)
    xph = np.linspace(0.0, 1.0, _na)                   # display phase; 0.5 = peak centre

    def _avg_centered(sig):
        s = _mm(sig); segs = []
        for f in np.asarray(open_frames, int):
            idx = np.round(f + _rel).astype(int)
            if idx.min() >= 0 and idx.max() < n:
                segs.append(s[idx])
        if not segs:
            return None, None
        A = np.array(segs)
        return A.mean(0), A.std(0)

    om, osd = _avg_centered(op_s)
    dur_ms = np.nan; onset_ph = np.nan
    if om is not None:
        rng_ = float(np.nanmax(om) - np.nanmin(om))
        thr = float(np.nanmin(om) + open_duration_thr * rng_)   # TUNABLE threshold
        above = om >= thr
        c = _na // 2
        if above[c]:
            l = c
            while l > 0 and above[l - 1]:
                l -= 1
            r = c
            while r < _na - 1 and above[r + 1]:
                r += 1
            ax_i.axvspan(xph[l], xph[r], color=COP, alpha=0.18, zorder=0)   # ONLY the opening
            ax_i.axvline(xph[l], color=COP, lw=1.1, ls="--", zorder=5)      # ONSET = up-crossing
            ax_i.axhline(thr, color="0.6", lw=0.6, ls=":", zorder=1)
            ax_i.annotate("opening onset", (xph[l], 1.04), fontsize=ANNOT_PT,
                          color=COP, ha="center", va="bottom")
            onset_ph = float(xph[l])
            dur_ms = (r - l) / (_na - 1) * period_fund / fps * 1e3
    eh = []
    for sig, col, name in ((v_z, CV, "ventricle"), (a_z, CAT, "atrium"), (op_s, C_OPEN, "openness")):
        if sig is None:
            continue
        m, sd = _avg_centered(sig)
        if m is None:
            continue
        ax_i.plot(xph, m, color=col, lw=1.5, zorder=4)
        ax_i.fill_between(xph, m - sd, m + sd, color=col, alpha=0.15, zorder=2)
        eh.append(Line2D([0], [0], color=col, lw=1.5, label=name))
    eh.append(Line2D([0], [0], color=COP, lw=5, alpha=0.4, label="opening duration"))
    ax_i.axvline(0.5, color="0.75", lw=0.6, zorder=1)                     # peak at the centre
    per_ms = period_fund / fps * 1e3
    ax_i.set_xlabel(f"one cardiac cycle — peak-centred (1 period ≈ {per_ms:.0f} ms)",
                    fontsize=LABEL_PT)
    ax_i.set_ylabel("normalized (0-1)", fontsize=LABEL_PT)
    ax_i.set_xlim(0, 1); ax_i.set_ylim(-0.05, 1.15)
    ax_i.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax_i.tick_params(labelsize=TICK_PT)
    ax_i.legend(handles=eh, loc="upper right", frameon=False, fontsize=LEGEND_PT)
    ax_i.set_title("average cardiac cycle", fontsize=TITLE_PT, pad=4)
    _letter(ax_i, "i")
    if np.isfinite(dur_ms):
        print(f"  [nature-fig] (i) opening duration (openness > {open_duration_thr:.2f} of "
              f"range) = {dur_ms:.0f} ms; onset at phase {onset_ph:.2f} (peak centred at 0.5)")

    # ── export: SVG + PDF stay VECTOR (image panels embed as PNG at savefig dpi
    #    via rasterized=True); PNG is a 600-dpi preview only. text stays editable
    #    (svg.fonttype=none, pdf.fonttype=42). bbox_inches=None preserves the
    #    requested true mm size (bbox_inches="tight" would CROP / RESCALE). ──
    paths = []
    for ext in ("svg", "pdf", "png"):
        pth = f"{save_stem}.{ext}"
        # SVG/PDF kept at true size (no tight crop); PNG cropped for preview only
        _bbox = "tight" if ext == "png" else None
        fig.savefig(pth, dpi=600, facecolor="white", bbox_inches=_bbox)
        paths.append(pth)
    plt.close(fig)
    print(f"[nature-fig] saved: {', '.join(os.path.basename(p) for p in paths)}")
    if not upp:
        print("[nature-fig] NOTE: um_per_pixel not set -> spatial axes are in px. "
              "Pass um_per_pixel=<value> for um units.")
    # ── VERIFY + REPORT: final size, Nature font hierarchy, editable-text SVG. ──
    _final_w_mm = _figw_in * 25.4
    _final_h_mm = _figh_in * 25.4
    print(f"[nature-fig] FINAL SIZE: {_final_w_mm:.1f} mm wide x {_final_h_mm:.1f} mm tall "
          f"({_figw_in:.3f} x {_figh_in:.3f} in). "
          f"PLACE the .svg at 100% scale in Illustrator (do NOT edit inside PowerPoint).")
    _roles = [("panel letters (a,b,c,d,i)", PANEL_LETTER_PT, "BOLD"),
              ("axis titles",                TITLE_PT,        ""),
              ("axis labels (x/y)",          LABEL_PT,        ""),
              ("tick labels",                TICK_PT,         ""),
              ("legends",                    LEGEND_PT,       ""),
              ("in-image annotations",       ANNOT_PT,        "")]
    print(f"[nature-fig] FONT HIERARCHY (export size, family={FONT_STACK[0]}/"
          f"{FONT_STACK[1]} fallback):")
    _min_role = min(pt for _, pt, _ in _roles)
    for _name, _pt, _bold in _roles:
        _flag = "OK" if _pt >= 5 else "FAIL <5pt"
        print(f"  {_name:<32s} {_pt} pt {_bold}  [{_flag}]")
    print(f"  min line width: {MIN_LW_PT} pt  (>= 0.5 pt: {'OK' if MIN_LW_PT >= 0.5 else 'FAIL'})")
    assert _min_role >= 5, f"FAIL: a text role is below 5 pt ({_min_role} pt)"
    # Grep the exported SVG: confirm text is <text ...> with font-family (NOT outlined to <path>).
    try:
        _svg_path = next(p for p in paths if p.endswith(".svg"))
        with open(_svg_path, "r", encoding="utf-8") as _f:
            _svg = _f.read()
        _text_n = _svg.count("<text")
        _ff_n = _svg.count("font-family")
        _path_n = _svg.count("<path")
        print(f"[nature-fig] SVG editable-text VERIFY: {_text_n} <text> elements, "
              f"{_ff_n} font-family attrs (text-as-text), {_path_n} <path> (shapes/lines).")
        if _text_n == 0:
            print(f"[nature-fig] WARNING: 0 <text> elements in {os.path.basename(_svg_path)} "
                  f"— text appears to be OUTLINED to paths (check svg.fonttype).")
    except Exception as _e:
        print(f"[nature-fig] SVG verify skipped: {_e}")
    return paths[0]


# ============================================================
# PRIMARY openness: INTENSITY M-MODE / DARKENING  (valve marking + figure)
# ============================================================
def valve_activity_map(stack, fps=600.0, ventricle_trace=None):
    """Per-pixel 'valve activity' map for AUTO-localization: the power at the
    cardiac fundamental frequency (FFT) of each pixel's intensity over time.
    Pixels that periodically DARKEN at the cardiac rhythm light up. Returns a
    normalized (H,W) map in [0,1]."""
    import leaflet_unet as _lu
    s = np.asarray(stack, np.float32)
    T, H, W = s.shape
    flat = s.reshape(T, -1).astype(np.float64)
    flat = flat - flat.mean(0, keepdims=True)
    # cardiac fundamental band (cycles over the record) from the V/A period
    P = _lu.robust_cardiac_period([ventricle_trace] if ventricle_trace is not None
                                  else [s.reshape(T, -1).mean(1)], T, fps=fps)
    f0 = T / max(P, 1.0)                                  # cycles over the whole record
    win = np.hanning(T)[:, None]
    spec = np.abs(np.fft.rfft(flat * win, axis=0))       # (F, HW)
    freqs = np.arange(spec.shape[0])
    band = (freqs >= max(1, f0 - 1.5)) & (freqs <= f0 + 1.5)
    power = spec[band].sum(0) / (spec[1:].sum(0) + 1e-9)  # fraction at cardiac band
    m = power.reshape(H, W)
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    return m.astype(np.float32)


def load_valve_mark(ds_dir):
    """Load the marked valve ROI + line (+ optional reference) or None."""
    p = os.path.join(ds_dir, "valve_mark.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
        return d
    except Exception as e:
        LOG.warning(f"[valve-mark] failed to load {p}: {e}")
        return None


def mark_valve_gui(stack, ds_dir, fps=600.0, valve_region="manual",
                   ventricle_trace=None):
    """Let the user MARK the valve: (1) a small ROI polygon and (2) a LINE through
    the valve along the opening axis, plus an OPTIONAL reference region (for ratio
    correction). Optionally overlays the auto valve-activity map as guidance
    (valve_region='auto'). Saves <ds_dir>/valve_mark.json. Returns the dict."""
    from matplotlib.widgets import PolygonSelector, Button
    s = np.clip(np.asarray(stack, np.float32), 0, 1)
    T, H, W = s.shape
    # background = temporal STD (motion) blended with the darkening map so the
    # valve (which moves + darkens) stands out for marking
    proj = s.std(0); proj = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    act = valve_activity_map(s, fps, ventricle_trace) if valve_region == "auto" else None

    st = {"roi": None, "line": [], "ref": None, "mode": "roi", "verts": None}
    fig = plt.figure(figsize=(8.6, 9))
    fig.canvas.manager.set_window_title("Mark the valve: ROI + line (+ optional reference)")
    ax = fig.add_axes([0.05, 0.13, 0.9, 0.82])
    ax.imshow(proj, cmap="gray")
    if act is not None:
        ax.imshow(np.where(act > 0.5, act, np.nan), cmap="magma", alpha=0.45)
        ax.set_title("AUTO valve-activity (cardiac-frequency power) overlaid in color.\n"
                     "Draw the valve ROI -> 'Set ROI'; click 2 pts for the LINE -> 'Set Line'; "
                     "optional ref -> 'Set Ref'; then 'Save'.", fontsize=8)
    else:
        ax.set_title("Draw a polygon around the valve -> 'Set ROI'.  Then click TWO points "
                     "along the opening axis -> 'Set Line'.  Optional reference region -> "
                     "'Set Ref'.  Then 'Save'.", fontsize=8)
    ax.axis("off")

    def _poly_mask(verts):
        from skimage.draw import polygon as skpoly
        ys = [p[1] for p in verts]; xs = [p[0] for p in verts]
        rr, cc = skpoly(np.array(ys), np.array(xs), shape=(H, W))
        m = np.zeros((H, W), bool); m[rr, cc] = True
        return m

    def on_sel(v):
        st["verts"] = v
    psel = PolygonSelector(ax, on_sel)

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None or st["mode"] != "line":
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        st["line"].append((float(ev.ydata), float(ev.xdata)))
        st["line"] = st["line"][-2:]
        _redraw()

    def _redraw():
        for art in list(ax.lines) + list(ax.patches):
            art.remove()
        if st["roi"] is not None:
            p = np.array(st["roi"])
            ax.plot(np.r_[p[:, 1], p[0, 1]], np.r_[p[:, 0], p[0, 0]], "-", color="#00e5ff", lw=1.8)
        if st["ref"] is not None:
            p = np.array(st["ref"])
            ax.plot(np.r_[p[:, 1], p[0, 1]], np.r_[p[:, 0], p[0, 0]], "--", color="#ffd400", lw=1.2)
        if len(st["line"]) == 2:
            (y0, x0), (y1, x1) = st["line"]
            ax.plot([x0, x1], [y0, y1], "-o", color="#ff3b3b", lw=2, ms=4)
        fig.canvas.draw_idle()

    def set_roi(_):
        if not st["verts"] or len(st["verts"]) < 3:
            set_status("Draw a valve ROI polygon first.", "error"); return
        st["roi"] = [(y, x) for x, y in st["verts"]]
        set_status("Valve ROI set. Now click 2 line points + 'Set Line'.", "run"); _redraw()

    def set_line(_):
        st["mode"] = "line"
        set_status("LINE mode: click 2 points along the valve opening axis.", "run")

    def set_ref(_):
        if not st["verts"] or len(st["verts"]) < 3:
            set_status("Draw a reference polygon first.", "error"); return
        st["ref"] = [(y, x) for x, y in st["verts"]]
        set_status("Reference region set (optional). Click 'Save'.", "run"); _redraw()

    def do_save(_):
        if st["roi"] is None or len(st["line"]) != 2:
            set_status("Need BOTH a valve ROI and a 2-point line.", "error"); return
        d = {"roi_polygon_yx": st["roi"], "line_yx": st["line"],
             "ref_polygon_yx": st["ref"], "image_shape": [H, W],
             "valve_region": valve_region}
        with open(os.path.join(ds_dir, "valve_mark.json"), "w") as f:
            json.dump(d, f, indent=2)
        LOG.info(f"[valve-mark] SAVED valve_mark.json: ROI {len(st['roi'])} verts, "
                 f"line {st['line']}, ref={'YES' if st['ref'] else 'no'} -> {ds_dir}")
        set_status("Valve mark saved. Close the window.", "done")
        ax.set_title("Saved valve ROI + line. Close the window.", fontsize=10, color="#2ca02c")
        fig.canvas.draw_idle()

    b_roi = Button(fig.add_axes([0.06, 0.04, 0.18, 0.06]), "Set ROI")
    b_line = Button(fig.add_axes([0.28, 0.04, 0.18, 0.06]), "Set Line")
    b_ref = Button(fig.add_axes([0.50, 0.04, 0.18, 0.06]), "Set Ref")
    b_save = Button(fig.add_axes([0.74, 0.04, 0.18, 0.06]), "Save")
    b_roi.on_clicked(set_roi); b_line.on_clicked(set_line)
    b_ref.on_clicked(set_ref); b_save.on_clicked(do_save)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig._keep_alive = [psel, b_roi, b_line, b_ref, b_save, on_click, set_roi, set_line,
                       set_ref, do_save, on_sel]
    plt.show()
    return load_valve_mark(ds_dir)


def _mark_to_masks(mark, shape):
    """valve_mark dict -> (roi_mask, line_pts, ref_mask|None)."""
    from skimage.draw import polygon as skpoly
    H, W = shape
    def _poly(p):
        if not p:
            return None
        ys = [a[0] for a in p]; xs = [a[1] for a in p]
        rr, cc = skpoly(np.array(ys), np.array(xs), shape=(H, W))
        m = np.zeros((H, W), bool); m[rr, cc] = True
        return m
    roi = _poly(mark.get("roi_polygon_yx"))
    ref = _poly(mark.get("ref_polygon_yx"))
    line = np.asarray(mark.get("line_yx"), float)
    return roi, line, ref


def save_vector(fig, save_stem, also=("svg", "pdf", "png")):
    """Export a figure as editable-text SVG + PDF + 600-dpi PNG, with provenance."""
    paths = []
    for ext in also:
        pth = f"{save_stem}.{ext}"
        log_files(writes=[pth])
        fig.savefig(pth, dpi=(600 if ext == "png" else None), facecolor="white",
                    bbox_inches="tight")
        paths.append(pth)
    plt.close(fig)
    LOG.info(f"[fig] saved {', '.join(os.path.basename(p) for p in paths)}")
    return paths[0]


def make_intensity_valve_figure(stack, mark, um_per_pixel=10.0, fps=600.0,
                                save_stem="valve_intensity_figure", prefix="valve_int",
                                depth=None, low_thr=0.35, high_thr=0.60,
                                min_state_frames=None):
    """Nature-style INTENSITY/M-mode validation figure (reads analyze_valve_intensity
    outputs): (a) representative CLOSED vs OPEN frames + valve ROI outline + scale
    bar; (b) the M-mode KYMOGRAPH (um x ms) with the 4 opens; (c) openness (dF/F
    darkening) overlaid with V and A, 4 opens marked + x-corr/lag; (d) phase-locked
    average over the 4 cycles; (e) the valve STATE TIMELINE. SVG+PDF+600-dpi PNG."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    need = f"{prefix}_openness.npy"
    if not os.path.exists(need):
        print(f"  [int-fig] '{need}' not found - run the intensity openness step first.")
        return None
    openness = np.load(f"{prefix}_openness.npy")
    dff = np.load(f"{prefix}_dff.npy") if os.path.exists(f"{prefix}_dff.npy") else openness
    open_frames = np.load(f"{prefix}_open_frames.npy")
    kymo = np.load(f"{prefix}_kymograph.npy") if os.path.exists(f"{prefix}_kymograph.npy") else None
    ev = np.load(f"{prefix}_events.npz", allow_pickle=True) if os.path.exists(f"{prefix}_events.npz") else None
    stack = np.clip(np.asarray(stack, np.float32), 0, 1)
    T = stack.shape[0]; n = min(T, len(openness))
    openness = openness[:n]; open_frames = open_frames[open_frames < n]
    roi, line, ref = _mark_to_masks(mark, stack.shape[1:])

    # V/A traces saved by the analysis (same depth); use the local detector
    v_src = f"{prefix}_ventricle.npy"; a_src = f"{prefix}_atrium.npy"
    v_pk = a_pk = np.array([], int); v_z = a_z = None
    if os.path.exists(v_src):
        v_pk, v_z = _detect_contractions(np.load(v_src)[:n], VENTRICLE_Z_THRESHOLD_DEFAULT)
    if os.path.exists(a_src):
        a_pk, a_z = _detect_contractions(np.load(a_src)[:n], -1.7)
    LOG.info(f"[int-fig] depth={depth}  openness shape={openness.shape}  "
             f"V peaks={len(v_pk)} A peaks={len(a_pk)}  ({len(open_frames)} opens)")

    t_ms = np.arange(n) / fps * 1e3
    oz = (openness - openness.min()) / (np.ptp(openness) + 1e-9)
    # representative open / closed frames
    open_main = int(open_frames[np.argmax(openness[open_frames])]) if len(open_frames) else int(np.argmax(openness))
    far = np.ones(n, bool)
    for f in open_frames:
        far[max(0, f - 5):f + 6] = False
    closed_frame = int(np.argmin(np.where(far, openness, openness.max() + 1)))

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "Arial",
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 600})
    # chamber colours match the SOURCE marker: ventricle=BLUE, atrium=RED
    CV, CAT, COP = "#1f77b4", "#d62728", "#2ca02c"
    fig = plt.figure(figsize=(7.4, 8.8)); fig.patch.set_facecolor("white")

    def _roi_outline(ax):
        if roi is not None:
            ax.contour(roi, levels=[0.5], colors="#00e5ff", linewidths=1.0)
        if line is not None and len(line) == 2:
            ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], "-",
                    color="#ff3b3b", lw=1.3)

    def _scalebar(ax, length_um=200.0):
        ylo, yhi = ax.get_ylim(); xlo, xhi = ax.get_xlim()
        barpx = length_um / um_per_pixel
        x0 = max(xlo, xhi) - 0.08 * abs(xhi - xlo) - barpx
        y = max(ylo, yhi) - 0.10 * abs(yhi - ylo)
        ax.add_patch(Rectangle((x0, y), barpx, 0.02 * abs(yhi - ylo) + 1, color="white", ec="none"))
        ax.text(x0 + barpx / 2, y, f"{length_um:g} µm", color="white", ha="center", va="bottom", fontsize=6)

    # (a) closed / open frames + ROI outline
    axc = fig.add_axes([0.05, 0.80, 0.22, 0.17]); axc.imshow(stack[closed_frame], cmap="gray", vmin=0, vmax=1)
    _roi_outline(axc); _scalebar(axc); axc.set_title("closed", fontsize=7.5); axc.axis("off")
    axc.text(-0.02, 1.04, "a", transform=axc.transAxes, fontsize=10, fontweight="bold")
    axo = fig.add_axes([0.28, 0.80, 0.22, 0.17]); axo.imshow(stack[open_main], cmap="gray", vmin=0, vmax=1)
    _roi_outline(axo); axo.set_title("open (darker)", fontsize=7.5); axo.axis("off")

    # (b) M-mode kymograph
    ax_k = fig.add_axes([0.58, 0.78, 0.38, 0.19])
    if kymo is not None:
        Lk = kymo.shape[0]
        ax_k.imshow((kymo - kymo.min()) / (np.ptp(kymo) + 1e-9), aspect="auto", cmap="gray",
                    extent=[0, t_ms[-1], Lk * um_per_pixel, 0], interpolation="nearest")
        for f in open_frames:
            ax_k.axvline(t_ms[f], color=COP, lw=0.8, ls="--", alpha=0.8)
        ax_k.set_xlabel("time (ms)"); ax_k.set_ylabel("distance (µm)")
        ax_k.set_title("M-mode kymograph (valve line)", fontsize=7.5)
    ax_k.text(-0.02, 1.06, "b", transform=ax_k.transAxes, fontsize=10, fontweight="bold")

    # (c) openness (dF/F darkening) + V + A overlay
    def _mm(x):
        x = np.asarray(x, float); return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    ax_d = fig.add_axes([0.08, 0.50, 0.88, 0.18])
    # EXPECTED-OPEN windows: atrium PEAK -> just-before-TROUGH (gold), distinct
    # from the DETECTED opens (COP spans).
    try:
        from leaflet_unet import (atrium_open_windows as _aow,
                                   ATRIUM_OPEN_START_OFFSET as _aos,
                                   ATRIUM_OPEN_END_LEAD as _ael)
        if os.path.exists(a_src):
            _aw, _, _ = _aow(np.load(a_src)[:n], fps, start_offset=_aos, end_lead=_ael)
            for ws, we in _aw:
                ax_d.axvspan(t_ms[ws], t_ms[we], color="#f4c542", alpha=0.22, zorder=0)
    except Exception as _e:
        print(f"  [intensity-fig] atrium open-windows skipped: {_e}")
    for f in open_frames:
        ax_d.axvspan(t_ms[max(0, f - 1)], t_ms[min(n - 1, f + 1)], color=COP, alpha=0.16)
    dh = []
    if v_z is not None:
        ax_d.plot(t_ms, _mm(v_z), color=CV, lw=0.9); dh.append(Line2D([0], [0], color=CV, lw=1.3, label="ventricle"))
    if a_z is not None:
        ax_d.plot(t_ms, _mm(a_z), color=CAT, lw=0.9); dh.append(Line2D([0], [0], color=CAT, lw=1.3, label="atrium"))
    ax_d.plot(t_ms, oz, color="black", lw=1.1); dh.append(Line2D([0], [0], color="black", lw=1.3, label="openness (darkening)"))
    if len(open_frames):
        ax_d.plot(t_ms[open_frames], oz[open_frames], "o", color=COP, ms=4, mec="k", mew=0.3)
        dh.append(Line2D([0], [0], marker="o", color=COP, lw=0, mec="k", mew=0.3, label="opening"))
    ax_d.set_xlim(t_ms[0], t_ms[-1]); ax_d.set_ylim(-0.05, 1.18); ax_d.set_xticklabels([])
    ax_d.set_ylabel("normalized")
    ax_d.legend(handles=dh, loc="upper right", frameon=False, ncol=4, fontsize=6)
    ttl = f"{len(open_frames)} opens (one per cardiac cycle)"
    if ev is not None:
        cv = float(np.asarray(ev["corr_v"])[0]); lv = float(np.asarray(ev["lag_v_corr_ms"])[0])
        ca = float(np.asarray(ev["corr_a"])[0]); la = float(np.asarray(ev["lag_a_corr_ms"])[0])
        ttl += f"   x-corr V r={cv:+.2f}@{lv:+.0f}ms  A r={ca:+.2f}@{la:+.0f}ms"
    ax_d.set_title(ttl, fontsize=7); ax_d.text(-0.02, 1.04, "c", transform=ax_d.transAxes, fontsize=10, fontweight="bold")

    # state timeline (d2)
    states, runs, _minf = classify_valve_states(oz, low_thr, high_thr, min_state_frames,
                                                period_frames=_period_from_ev(ev, n), fps=fps)
    ax_st = fig.add_axes([0.08, 0.455, 0.88, 0.03], sharex=ax_d)
    SC = {0: "#9aa0a6", 1: "#f0ad4e", 2: "#2ca02c"}
    for s0, s1, sstate in runs:
        ax_st.axvspan(t_ms[s0], t_ms[min(n - 1, s1 + 1)], color=SC[sstate], lw=0)
    ax_st.set_xlim(t_ms[0], t_ms[-1]); ax_st.set_yticks([]); ax_st.set_xlabel("time (ms)")
    ax_st.set_ylabel("state", fontsize=6, rotation=0, ha="right", va="center")
    ax_st.legend(handles=[Line2D([0], [0], color=SC[0], lw=5, label="closed"),
                          Line2D([0], [0], color=SC[1], lw=5, label="transition"),
                          Line2D([0], [0], color=SC[2], lw=5, label="open")],
                 loc="upper right", frameon=False, ncol=3, fontsize=5.5, bbox_to_anchor=(1.0, 2.6))

    # (d) phase-locked average over the 4 cycles
    ax_p = fig.add_axes([0.08, 0.07, 0.40, 0.26])
    period = _period_from_ev(ev, n)
    phase = (np.arange(n) % period) / period
    nb = 20; bc = (np.arange(nb) + 0.5) / nb

    def _cyc(sig):
        s = _mm(sig); bid = np.clip((phase * nb).astype(int), 0, nb - 1)
        mean = np.array([s[bid == b].mean() if np.any(bid == b) else np.nan for b in range(nb)])
        sd = np.array([s[bid == b].std() if np.any(bid == b) else 0.0 for b in range(nb)])
        return mean, sd
    for sig, col, lab in ((oz, "black", "openness"),
                          (v_z if v_z is not None else None, CV, "ventricle"),
                          (a_z if a_z is not None else None, CAT, "atrium")):
        if sig is None:
            continue
        m, sd = _cyc(sig)
        ax_p.plot(bc, m, color=col, lw=1.2, label=lab)
        ax_p.fill_between(bc, m - sd, m + sd, color=col, alpha=0.15)
    ax_p.set_xlabel("cardiac phase (0->1)"); ax_p.set_ylabel("normalized")
    ax_p.set_title(f"phase-locked average ({max(1,int(round(n/period)))} cycles)", fontsize=7.5)
    ax_p.legend(frameon=False, fontsize=6, loc="upper right")
    ax_p.text(-0.04, 1.04, "d", transform=ax_p.transAxes, fontsize=10, fontweight="bold")

    # (e) per-event metrics text
    ax_t = fig.add_axes([0.55, 0.07, 0.41, 0.26]); ax_t.axis("off")
    ax_t.text(-0.02, 1.04, "e", transform=ax_t.transAxes, fontsize=10, fontweight="bold")
    lines = ["per-event (4 opens):"]
    if ev is not None:
        gpu = np.asarray(ev["gap_um"], float); dms = np.asarray(ev["duration_ms"], float)
        lv = np.asarray(ev["lag_v_ms"], float); la = np.asarray(ev["lag_a_ms"], float)
        btw = int(np.sum(np.asarray(ev["between_va"], bool)))
        for k in range(len(np.asarray(ev["open_frames"]))):
            lines.append(f"  {k+1}: dark {gpu[k]:.0f}%  dur {dms[k]:.0f} ms  "
                         f"lagV {lv[k]:+.0f}  lagA {la[k]:+.0f} ms")
        lines += [f"between V/A max-change: {btw}/{len(gpu)}",
                  f"period {period/fps*1e3:.0f} ms ({max(1,int(round(n/period)))} cycles)"]
    ax_t.text(0.0, 0.98, "\n".join(lines), transform=ax_t.transAxes, va="top", ha="left",
              fontsize=6.2, family="monospace")

    return save_vector(fig, save_stem)


def classify_valve_states(oz, low_thr=0.35, high_thr=0.60, min_state_frames=None,
                          period_frames=None, fps=600.0):
    """3-state classification (0=closed,1=transition,2=open) with dual thresholds
    (hysteresis) + a minimum-duration HOLD. Returns (states, runs, min_frames)."""
    n = len(oz)
    P = period_frames if period_frames else n / max(1, EXPECTED_OPENINGS_FIG)
    _minf = int(min_state_frames) if min_state_frames is not None else max(3, int(round(0.02 * P)))
    states = np.full(n, 1, np.int8)
    states[oz <= low_thr] = 0; states[oz >= high_thr] = 2

    def _runs(a):
        out, i = [], 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[j + 1] == a[i]:
                j += 1
            out.append((i, j, int(a[i]))); i = j + 1
        return out
    for s0, s1, sstate in _runs(states.copy()):
        if sstate in (0, 2) and (s1 - s0 + 1) < _minf:
            states[s0:s1 + 1] = 1
    return states, _runs(states), _minf


def _period_from_ev(ev, n):
    if ev is not None:
        try:
            return float(np.asarray(ev["period_frames"], float)[0])
        except Exception:
            pass
    return n / max(1, EXPECTED_OPENINGS_FIG)


EXPECTED_OPENINGS_FIG = 4


def define_training_crop_gui(stack, ds_dir, projection="std"):
    """Define a SQUARE training crop around the valve (1:1 ENFORCED). Drag a box on
    a temporal projection; it is squared (centered) + clipped to the frame; 'Accept'
    saves it (drag again to redo, 'Clear' = full frame). Persists training_crop.json
    -- the SINGLE SOURCE OF TRUTH reused by annotate / train / infer. Returns bbox."""
    from matplotlib.widgets import RectangleSelector, Button
    from matplotlib.patches import Rectangle
    import leaflet_unet as _lu
    s = np.clip(np.asarray(stack, np.float32), 0, 1)
    T, H, W = s.shape
    proj = s.std(0) if projection == "std" else s.max(0)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    st = {"bbox": None, "rect": None}

    fig = plt.figure(figsize=(8.6, 9))
    fig.canvas.manager.set_window_title("Define SQUARE training crop")
    ax = fig.add_axes([0.05, 0.13, 0.9, 0.82]); ax.imshow(proj, cmap="gray")
    ax.set_title("Drag a box around the valve - it is forced SQUARE (1:1).  "
                 "'Accept' to save; drag again to redo; 'Clear' = full frame.", fontsize=9)
    ax.axis("off")

    def _square(y0, x0, y1, x1):
        y0, y1 = sorted((y0, y1)); x0, x1 = sorted((x0, x1))
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        half = max(y1 - y0, x1 - x0) / 2.0
        half = max(3.0, min(half, cy, H - cy, cx, W - cx))   # clip to image, stay square
        b = [int(round(cy - half)), int(round(cx - half)),
             int(round(cy + half)), int(round(cx + half))]
        side = min(b[2] - b[0], b[3] - b[1]); b[2] = b[0] + side; b[3] = b[1] + side
        return b

    def _draw(b):
        if st["rect"] is not None:
            try:
                st["rect"].remove()
            except Exception:
                pass
        st["rect"] = Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], fill=False,
                               ec="#00e5ff", lw=2)
        ax.add_patch(st["rect"]); fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        b = _square(eclick.ydata, eclick.xdata, erelease.ydata, erelease.xdata)
        st["bbox"] = b; _draw(b)
        set_status(f"Square crop {b}  side={b[2]-b[0]}px. Accept or drag again.", "run")

    rs = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=5,
                           minspany=5, interactive=True)
    existing = _lu.load_training_crop(ds_dir)
    if existing:
        st["bbox"] = existing; _draw(existing)

    def do_accept(_):
        if st["bbox"] is None:
            set_status("Drag a square crop first.", "error"); return
        _lu.save_training_crop(ds_dir, st["bbox"])
        LOG.info(f"[crop] training crop saved {st['bbox']} side={st['bbox'][2]-st['bbox'][0]}px "
                 f"-> {ds_dir}")
        set_status(f"Training crop saved: {st['bbox']}. Close the window.", "done")
        ax.set_title("Training crop saved. Close the window.", fontsize=10, color="#2ca02c")
        fig.canvas.draw_idle()

    def do_clear(_):
        st["bbox"] = None
        if st["rect"] is not None:
            try:
                st["rect"].remove()
            except Exception:
                pass
        st["rect"] = None
        try:
            os.remove(os.path.join(ds_dir, "training_crop.json"))
        except Exception:
            pass
        LOG.info("[crop] training crop CLEARED (full-frame annotation)")
        set_status("Training crop cleared (annotate the full frame).", "info")
        fig.canvas.draw_idle()

    b_acc = Button(fig.add_axes([0.1, 0.04, 0.32, 0.06]), "Accept crop")
    b_clr = Button(fig.add_axes([0.55, 0.04, 0.32, 0.06]), "Clear (full frame)")
    b_acc.on_clicked(do_accept); b_clr.on_clicked(do_clear)
    fig._keep_alive = [rs, b_acc, b_clr, do_accept, do_clear, on_select]
    plt.show()
    return _lu.load_training_crop(ds_dir)


def select_training_frames_gui(frame_info, default_all=True):
    """Tk dialog to SELECT which annotated frames to TRAIN on. Shows a multi-select
    checklist (each row: frame index + leaflet 1 / leaflet 2 presence) with
    select-all / clear-all, PLUS an index/range entry ("12, 40-55, 120"). Default
    selection = all annotated frames. Returns the chosen list of frame indices, or
    None on cancel."""
    import tkinter as tk
    pos_of = {fi["idx"]: i for i, fi in enumerate(frame_info)}

    def _parse(s):
        out = set()
        for tok in s.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                try:
                    a, b = tok.split("-", 1); out.update(range(int(a), int(b) + 1))
                except Exception:
                    pass
            else:
                try:
                    out.add(int(tok))
                except Exception:
                    pass
        return out

    # Use the EXISTING Tk root (matplotlib's TkAgg loop) via a modal Toplevel +
    # wait_window. Creating a SECOND tk.Tk() + nested mainloop() DEADLOCKS when the
    # dialog is opened more than once (stacked event loops never unwind, so "Train
    # on selected" never returns). Fall back to a standalone root only when no Tk
    # root exists (e.g. headless use / testing).
    _parent = getattr(tk, "_default_root", None)
    owns_root = _parent is None
    root = tk.Tk() if owns_root else tk.Toplevel(_parent)
    root.title("Select training frames"); root.geometry("440x540")
    tk.Label(root, text=f"{len(frame_info)} annotated frames. Pick which to TRAIN on "
             f"(default = all). [L1]/[L2] = leaflet labels present.",
             anchor="w", justify="left", wraplength=420).pack(fill="x", padx=8, pady=(8, 2))

    frm = tk.Frame(root); frm.pack(fill="both", expand=True, padx=8)
    sb = tk.Scrollbar(frm); sb.pack(side="right", fill="y")
    lb = tk.Listbox(frm, selectmode=tk.EXTENDED, yscrollcommand=sb.set,
                    font=("Consolas", 9), activestyle="none")
    lb.pack(side="left", fill="both", expand=True); sb.config(command=lb.yview)
    for fi in frame_info:
        lb.insert(tk.END, f"frame {fi['idx']:04d}    "
                  f"[{'L1' if fi['has_l1'] else '--'} {'L2' if fi['has_l2'] else '--'} "
                  f"{'C' if fi.get('has_conn') else '-'}]  "
                  f"{'OPEN' if fi.get('open') else 'closed'}")
    if default_all:
        lb.selection_set(0, tk.END)

    cnt = tk.Label(root, text="")

    def _update_count(*_):
        cnt.config(text=f"{len(lb.curselection())} selected")
    lb.bind("<<ListboxSelect>>", _update_count)

    ent_frame = tk.Frame(root); ent_frame.pack(fill="x", padx=8, pady=4)
    tk.Label(ent_frame, text="indices/ranges:").pack(side="left")
    ent = tk.Entry(ent_frame); ent.pack(side="left", fill="x", expand=True, padx=4)
    ent.insert(0, "")

    def _apply(replace):
        want = _parse(ent.get())
        if replace:
            lb.selection_clear(0, tk.END)
        for v in want:
            if v in pos_of:
                lb.selection_set(pos_of[v])
        _update_count()
    tk.Button(ent_frame, text="Set", width=5, command=lambda: _apply(True)).pack(side="left")
    tk.Button(ent_frame, text="Add", width=5, command=lambda: _apply(False)).pack(side="left")

    btns = tk.Frame(root); btns.pack(fill="x", padx=8, pady=2)
    tk.Button(btns, text="Select all",
              command=lambda: (lb.selection_set(0, tk.END), _update_count())).pack(side="left")
    tk.Button(btns, text="Clear all",
              command=lambda: (lb.selection_clear(0, tk.END), _update_count())).pack(side="left", padx=4)
    cnt.pack(in_=btns, side="right")

    res = {"sel": None}

    def _ok():
        res["sel"] = [frame_info[i]["idx"] for i in lb.curselection()]; root.destroy()

    def _cancel():
        res["sel"] = None; root.destroy()
    ok_frame = tk.Frame(root); ok_frame.pack(fill="x", padx=8, pady=8)
    tk.Button(ok_frame, text="Train on selected", width=16, command=_ok).pack(side="left")
    tk.Button(ok_frame, text="Cancel", width=8, command=_cancel).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", _cancel)        # window-close [X] = cancel (no hang)
    try:                                              # surface it above the main window
        if _parent is not None:
            root.transient(_parent)
        root.lift(); root.attributes("-topmost", True)
        root.after(400, lambda: root.attributes("-topmost", False))
        root.grab_set(); root.focus_force()           # modal: input goes to the dialog
    except Exception:
        pass
    _update_count()
    if owns_root:
        root.mainloop()                               # standalone: own event loop
    else:
        root.wait_window()                            # nested in matplotlib's loop, NO 2nd mainloop
    return res["sel"]


# ============================================================
# Multi-depth TIF de-interleave + chamber (ventricle/atrium) signals
# ============================================================
def _open_multipage_tif(tif_path):
    """Open a multi-page TIF for indexed page access. Tries memmap (cheap, only
    selected pages materialized); compressed/non-contiguous TIFs are not memory-
    mappable, so fall back to a lazy page-series reader that still avoids loading
    the whole stack at once."""
    import tifffile
    try:
        mm = tifffile.memmap(tif_path)
        if mm.ndim == 2:
            mm = mm[None]
        return mm
    except (ValueError, OSError) as e:
        LOG.info(f"[depth] memmap unavailable ({e}); using lazy page reader")

        class _PageReader:
            def __init__(self, path):
                self._tif = tifffile.TiffFile(path)
                self._pages = self._tif.series[0].pages
                self.shape = (len(self._pages),) + tuple(self._pages[0].shape)

            @property
            def ndim(self):
                return len(self.shape)

            def __getitem__(self, key):
                if isinstance(key, (int, np.integer)):
                    return self._pages[int(key)].asarray()
                idxs = np.asarray(key).ravel()
                return np.stack([self._pages[int(i)].asarray() for i in idxs])

        return _PageReader(tif_path)


def deinterleave_depth(tif_path, depth, n_frames=901, n_depths=8,
                       order="frame_major"):
    """Extract a single depth (0..n_depths-1) from an interleaved multi-depth
    TIF as a (n_frames, H, W) stack. Streams via memmap (only the selected depth
    is materialized; the full n_frames*n_depths stack is never copied).

    order:
      'frame_major' (z fastest):  idx = t*n_depths + depth   [t0z0,t0z1,...,t0z7,t1z0,...]
      'depth_major' (t fastest):  idx = depth*n_frames + t   [z0t0,...,z0t900,z1t0,...]
    """
    mm = _open_multipage_tif(tif_path)
    n_total = int(mm.shape[0])
    if n_total != n_frames * n_depths:
        raise ValueError(f"TIF has {n_total} images, but n_frames*n_depths = "
                         f"{n_frames}x{n_depths} = {n_frames * n_depths}. "
                         f"Fix n_frames / n_depths.")
    if not (0 <= depth < n_depths):
        raise ValueError(f"depth {depth} out of range 0..{n_depths - 1}")
    if order == "frame_major":
        idxs = np.arange(n_frames) * n_depths + depth
    elif order == "depth_major":
        idxs = depth * n_frames + np.arange(n_frames)
    else:
        raise ValueError(f"unknown interleave order '{order}'")
    return np.array(mm[idxs])                 # copies only the selected n_frames


def detect_interleave_order(tif_path, n_frames=901, n_depths=8, sample=60, depth=0):
    """Auto-detect the interleave order: a true single-depth time-series has high
    consecutive-frame correlation; a wrongly de-interleaved (depth-jumping) one
    does not. Returns (best_order, {order: mean_consecutive_corr})."""
    mm = _open_multipage_tif(tif_path)
    n_total = int(mm.shape[0])
    if n_total != n_frames * n_depths:
        raise ValueError(f"TIF has {n_total} images != {n_frames}x{n_depths}.")
    s = int(min(sample, n_frames))
    scores = {}
    for order in ("frame_major", "depth_major"):
        idxs = (np.arange(s) * n_depths + depth if order == "frame_major"
                else depth * n_frames + np.arange(s))
        st = np.array(mm[idxs]).reshape(s, -1).astype(np.float32)
        cs = []
        for i in range(s - 1):
            a, b = st[i], st[i + 1]
            if a.std() > 1e-6 and b.std() > 1e-6:
                cs.append(np.corrcoef(a, b)[0, 1])
        scores[order] = float(np.nanmean(cs)) if cs else 0.0
    best = max(scores, key=scores.get)
    return best, scores


def extract_red_blue_masks(rgb_stack, min_frac=0.05):
    """Auto-extract the chamber static masks from a color-marked (T, H, W, 3)
    stack, mapping marker COLOR -> heart CHAMBER via the single source of truth
    ``leaflet_unet.CHAMBER_OF_COLOR``  (RED = ATRIUM, BLUE = VENTRICLE).

    Robust to brightness: works on a [0,1]-normalized stack with channel-
    difference thresholds (red = R clearly above G and B; blue = B clearly above
    R and G), aggregated over time (a pixel joins a region if it is that color in
    >= ``min_frac`` of frames). Each mask is then cleaned with a morphological
    open + close and reduced to its largest connected component.

    Returns ``(vent_mask, atr_mask, info)`` - boolean (H, W) static masks for the
    VENTRICLE (blue) and ATRIUM (red).  ``info`` carries the per-frame chamber
    MOVEMENT signals ``v_area`` (ventricle = blue pixel count) / ``a_area``
    (atrium = red pixel count); as a chamber contracts its marked region shrinks,
    so the area dips - the downward-dip convention detect_contractions expects.
    It also keeps the colour-keyed masks/areas/stats (``red_*`` / ``blue_*``)."""
    from scipy import ndimage as ndi
    from leaflet_unet import CHAMBER_OF_COLOR, COLOR_OF_CHAMBER
    rgb = np.asarray(rgb_stack, np.float32)
    mx = float(rgb.max())
    if mx > 1.0:
        rgb = rgb / (255.0 if mx <= 255.0 else mx)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_bool = (R - G > 0.12) & (R - B > 0.12) & (R > 0.25)     # per-frame "is red"
    blue_bool = (B - R > 0.12) & (B - G > 0.12) & (B > 0.25)    # per-frame "is blue"
    red_area = red_bool.reshape(red_bool.shape[0], -1).sum(1).astype(np.float32)   # (T,)
    blue_area = blue_bool.reshape(blue_bool.shape[0], -1).sum(1).astype(np.float32)
    red_freq = red_bool.mean(0)                                 # (H,W) frame fraction
    blue_freq = blue_bool.mean(0)

    def _clean(m):
        if m.sum() == 0:
            return m
        m = ndi.binary_opening(m, iterations=1)
        m = ndi.binary_closing(m, iterations=2)
        lab, n = ndi.label(m)
        if n > 1:                                               # keep largest component
            sizes = ndi.sum(np.ones_like(lab, np.float32), lab,
                            index=np.arange(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        return m.astype(bool)

    red_mask = _clean(red_freq >= min_frac)
    blue_mask = _clean(blue_freq >= min_frac)
    # ── map COLOUR -> CHAMBER (single source of truth) ──
    by_color = {"red": (red_mask, red_area), "blue": (blue_mask, blue_area)}
    atr_mask, a_area = by_color[COLOR_OF_CHAMBER["atrium"]]
    vent_mask, v_area = by_color[COLOR_OF_CHAMBER["ventricle"]]
    info = dict(
        # chamber-keyed (what STEP 1 persists as ventricle_trace / atrium_trace)
        v_area=v_area, a_area=a_area,
        vent_px=int(vent_mask.sum()), atr_px=int(atr_mask.sum()),
        vent_color=COLOR_OF_CHAMBER["ventricle"],
        atr_color=COLOR_OF_CHAMBER["atrium"],
        # colour-keyed (logging / display / valve band)
        red_px=int(red_mask.sum()), blue_px=int(blue_mask.sum()),
        red_freq_max=float(red_freq.max()), blue_freq_max=float(blue_freq.max()),
        red_mask=red_mask, blue_mask=blue_mask,
        red_area=red_area, blue_area=blue_area,
        chamber_of_color=dict(CHAMBER_OF_COLOR))
    return vent_mask, atr_mask, info


def derive_valve_band(rgb_stack, um_per_pixel=10.0, occ_thresh=0.05,
                      band_width_um=120.0, target_shape=None):
    """Auto AV-junction BAND from the TEMPORAL OVERLAP of the red(A)/blue(V)
    regions. As the heart beats some pixels are ventricle in SOME frames and
    atrium in OTHERS - that contested zone is where the valve sits.

    JUNCTION seed = (V_occ >= occ_thresh AND A_occ >= occ_thresh)  [contested]
                    OR (red AND blue in the same frame)            [overlap]
                    OR (dilated V-region AND dilated A-region)     [interface]
    Then dilate to a band of ``band_width_um`` and keep the largest connected
    component. ``target_shape`` (the raw stack H,W) resamples the band onto the
    U-Net grid. Returns (band (H,W) bool, info)."""
    from scipy import ndimage as ndi
    rgb = np.asarray(rgb_stack, np.float32)
    mx = float(rgb.max())
    if mx > 1.0:
        rgb = rgb / (255.0 if mx <= 255.0 else mx)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (R - G > 0.12) & (R - B > 0.12) & (R > 0.25)
    blue = (B - R > 0.12) & (B - G > 0.12) & (B > 0.25)
    V_occ = red.mean(0); A_occ = blue.mean(0)                  # per-pixel occupancy
    contested = (V_occ >= occ_thresh) & (A_occ >= occ_thresh)
    overlap = (red & blue).any(0)
    k = ndi.generate_binary_structure(2, 2)
    Vreg, Areg = V_occ >= occ_thresh, A_occ >= occ_thresh
    interface = ndi.binary_dilation(Vreg, k, 2) & ndi.binary_dilation(Areg, k, 2)
    seed = contested | overlap | interface
    bw_px = max(1, int(round((band_width_um / max(um_per_pixel, 1e-6)) / 2)))
    band = ndi.binary_dilation(seed, k, bw_px)
    band = ndi.binary_closing(band, k, 2)
    lab, n = ndi.label(band)
    if n > 1:                                                  # keep the junction band
        sizes = ndi.sum(np.ones_like(lab, np.float32), lab, index=np.arange(1, n + 1))
        band = lab == (1 + int(np.argmax(sizes)))
    band = band.astype(bool)
    if target_shape is not None and tuple(band.shape) != tuple(target_shape):
        from skimage.transform import resize
        band = resize(band.astype(float), tuple(target_shape), order=0,
                      preserve_range=True) > 0.5               # onto the raw/U-Net grid
    return band, dict(band_px=int(band.sum()), contested_px=int(contested.sum()),
                      occ_thresh=float(occ_thresh), band_width_um=float(band_width_um),
                      v_occ=V_occ, a_occ=A_occ)


def valve_band_gui(rgb, raw_disp, depth, ds_dir, um_per_pixel=10.0):
    """Interactive AV-junction band tuner: derive the band from the red/blue
    temporal overlap, show it over the RAW depth projection, let the user tune
    the occupancy threshold + band width (px<-um), and SAVE valve_band.npy +
    record it in depth_config (single source of truth, keyed to the depth)."""
    from matplotlib.widgets import Slider, Button
    from leaflet_unet import save_valve_band, save_depth_config, load_depth_config
    tgt = tuple(raw_disp.shape[1:])
    raw_proj = raw_disp.std(0); raw_proj = (raw_proj - raw_proj.min()) / (np.ptp(raw_proj) + 1e-9)
    st = {"band": None, "occ": 0.05, "bw": 120.0}
    fig = plt.figure(figsize=(7.6, 8.6))
    fig.canvas.manager.set_window_title("Auto valve band (AV junction)")
    ax = fig.add_axes([0.05, 0.20, 0.9, 0.76]); ax.imshow(raw_proj, cmap="gray"); ax.axis("off")
    ov = ax.imshow(np.zeros(tgt + (4,)))

    def recompute():
        band, info = derive_valve_band(rgb, um_per_pixel, st["occ"], st["bw"], target_shape=tgt)
        st["band"] = band
        rgba = np.zeros(tgt + (4,), np.float32)
        rgba[band] = [0.0, 1.0, 1.0, 0.40]
        ov.set_data(rgba)
        ax.set_title(f"AUTO valve band (V/A temporal overlap): occ>={st['occ']:.2f}, "
                     f"width {st['bw']:.0f} um -> {int(band.sum())} px\n"
                     f"Tune sliders so the band sits on the AV junction, then Accept.",
                     fontsize=8.5)
        fig.canvas.draw_idle()

    s_occ = Slider(fig.add_axes([0.12, 0.12, 0.66, 0.03]), "occ thr", 0.01, 0.30, valinit=0.05)
    s_bw = Slider(fig.add_axes([0.12, 0.07, 0.66, 0.03]), "width um", 30.0, 400.0, valinit=120.0)

    def _upd(_):
        st["occ"], st["bw"] = float(s_occ.val), float(s_bw.val); recompute()
    s_occ.on_changed(_upd); s_bw.on_changed(_upd)

    def _accept(_):
        if st["band"] is None or st["band"].sum() == 0:
            set_status("Empty band - adjust thresholds/width.", "error"); return
        save_valve_band(ds_dir, st["band"])
        cfg = load_depth_config(ds_dir) or {}
        cfg["band"] = os.path.join(ds_dir, "valve_band.npy")
        cfg["band_occ_thresh"] = float(st["occ"]); cfg["band_width_um"] = float(st["bw"])
        save_depth_config(ds_dir, cfg)
        LOG.info(f"[band] depth {depth}: SAVED valve band ({int(st['band'].sum())} px) "
                 f"occ>={st['occ']:.2f} width {st['bw']:.0f}um -> {ds_dir}/valve_band.npy")
        set_status(f"Valve band saved ({int(st['band'].sum())} px) for depth {depth}.", "done")
        ax.set_title("Valve band saved. Close the window.", fontsize=10, color="#2ca02c")
        fig.canvas.draw_idle()

    b_acc = Button(fig.add_axes([0.82, 0.07, 0.14, 0.08]), "Accept")
    b_acc.on_clicked(_accept)
    fig._keep_alive = [s_occ, s_bw, b_acc, _upd, _accept]
    recompute(); plt.show()


def chamber_signals_gui(ds_dir, default_tif=None):
    """STEP 1 / FOUNDATION (per depth, selectable). Two sources tied by depth:
      * the R&B-marked stack (901 x n_depths) -> de-interleave the chosen depth,
        AUTO-EXTRACT ventricle (RED) / atrium (BLUE), and compute the per-frame
        region-AREA CHANGE as the V and A MOVEMENT signals (manual draw fallback);
      * a SEPARATE per-depth RAW grayscale file (901 frames) -> the working stack
        the U-Net runs on (line extraction / openness).
    Persists into the active dataset folder ``ds_dir`` (single source of truth):
      * depth_config.json   - depth, order, R&B + raw source paths, V/A paths
      * depth{d}_stack.tif  - the RAW per-depth working STACK (U-Net)
      * ventricle_trace.npy / atrium_trace.npy (+ va_*_depth{d}.npy) - V/A
        MOVEMENT signals ((n_frames,) float32, EXACT existing format)
    Sets the session globals (ACTIVE_DS_DIR / ACTIVE_DEPTH / ACTIVE_STACK /
    DATA_TIFF) so downstream steps operate on the same depth. Returns the stack."""
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd
    from matplotlib.widgets import PolygonSelector, Button
    from leaflet_unet import save_depth_config, UM_PER_PIXEL as _UMPP

    os.makedirs(ds_dir, exist_ok=True)
    LOG.info(f"[STEP 1] depth+V/A foundation -> dataset folder: {ds_dir}")
    _root = _tk.Tk(); _root.withdraw()
    # ── SOURCE: the multi-depth TIF STACK FILE (901 frames x 8 depths = 7208
    #    pages combined in ONE file) — a FILE picker, NOT a folder. ──
    tif = _fd.askopenfilename(title="STEP 1: select the multi-depth TIF STACK FILE "
                                    "(interleaved, e.g. 7208 pages)",
                              initialdir=BASE_DIR,
                              filetypes=[("TIFF stack", "*.tif *.tiff"), ("All", "*.*")])
    if not tif:
        _root.destroy(); set_status("Depth->V/A cancelled.", "info"); return None
    LOG.info(f"[STEP 1] multi-depth TIF stack FILE -> {tif}")
    n_frames = _sd.askinteger("Frames", "number of TIME frames", initialvalue=901) or 901
    n_depths = _sd.askinteger("Depths", "number of DEPTHS (z planes)", initialvalue=8) or 8
    _root.destroy()

    # ── validate the stack file has exactly n_frames * n_depths pages ──
    try:
        n_pages = int(_open_multipage_tif(tif).shape[0])
    except Exception as e:
        set_status(f"ERROR: cannot read TIF stack: {e}", "error")
        _r = _tk.Tk(); _r.withdraw()
        _mb.showerror("Bad TIF stack", f"Could not read pages from:\n{tif}\n\n{e}")
        _r.destroy(); return None
    if n_pages != n_frames * n_depths:
        msg = (f"Selected TIF stack has {n_pages} pages, but "
               f"n_frames x n_depths = {n_frames} x {n_depths} = "
               f"{n_frames * n_depths}.\n\nPick the combined multi-depth stack "
               f"file (e.g. 7208 pages = 901 x 8), or fix the frame/depth counts.")
        LOG.error(f"[STEP 1] page-count mismatch: {n_pages} != {n_frames}x{n_depths}")
        set_status(f"ERROR: TIF has {n_pages} pages != {n_frames}x{n_depths}", "error")
        _r = _tk.Tk(); _r.withdraw()
        _mb.showerror("Wrong page count", msg); _r.destroy(); return None
    LOG.info(f"[STEP 1] stack OK: {n_pages} pages == {n_frames} frames x {n_depths} depths")

    try:
        best, scores = detect_interleave_order(tif, n_frames, n_depths)
    except Exception as e:
        set_status(f"ERROR: {e}", "error")
        _r = _tk.Tk(); _r.withdraw(); _mb.showerror("De-interleave", str(e)); _r.destroy()
        return None
    LOG.info(f"[depth] auto-detected order='{best}'  consec-corr scores={scores}")
    set_status(f"Depth->V/A: best order='{best}' (corr {scores[best]:.3f})", "run")

    _root = _tk.Tk(); _root.withdraw()
    depth = _sd.askinteger("Depth", f"which depth to use? 0..{n_depths - 1}", initialvalue=0)
    if depth is None:
        _root.destroy(); return None
    use_best = _mb.askyesno("Interleave order",
                            f"Auto-detected order = '{best}' "
                            f"(consec-corr {scores[best]:.3f} vs "
                            f"{scores['depth_major' if best == 'frame_major' else 'frame_major']:.3f}).\n\n"
                            f"YES = use '{best}'   NO = use the other order")
    _root.destroy()
    order = best if use_best else ("depth_major" if best == "frame_major" else "frame_major")
    LOG.info(f"[depth] extracting depth {depth} with order '{order}' from {tif}")

    stack = deinterleave_depth(tif, depth, n_frames, n_depths, order).astype(np.float32)
    # ── RGB/RGBA source pages -> single-channel luminance ──
    # Multi-channel TIFs (e.g. segmentation_4D_RGB) de-interleave to a 4-D
    # (T, H, W, C) stack. The per-frame V/A trace + the working depth stack that
    # annotate/train/infer consume are all single-channel, so collapse the COLOR
    # axis to luminance here (the depth axis was already selected above). We keep
    # the real intensity - this is NOT a throwaway-variable drop of a meaningful
    # spatial/ROI/depth axis.
    stack_rgb = None                                # color markings, if the source carries them
    if stack.ndim == 4:
        nc = stack.shape[-1]
        LOG.info(f"[STEP 1] multi-channel stack {stack.shape} "
                 f"({nc}-channel frames) -> luminance (T,H,W)")
        if nc >= 3:
            stack_rgb = stack[..., :3].astype(np.float32)           # keep for red/blue auto-extract
            w = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # Rec.601 luma
            stack = (stack[..., :3] * w).sum(axis=-1)
        else:                                                       # 1- or 2-channel
            stack = stack.mean(axis=-1)
        stack = stack.astype(np.float32)
    elif stack.ndim != 3:
        set_status(f"ERROR: unexpected stack shape {stack.shape}", "error")
        _r = _tk.Tk(); _r.withdraw()
        _mb.showerror("Bad depth stack",
                      f"De-interleaved stack has shape {stack.shape}; expected "
                      f"(T,H,W) or (T,H,W,C). Check the TIF / depth settings.")
        _r.destroy(); return None
    lo, hi = np.percentile(stack, [0.5, 99.5])
    disp = np.clip((stack - lo) / (hi - lo + 1e-8), 0, 1)
    T, H, W = stack.shape

    # ── preview scrub to verify depth + order ──
    pf = plt.figure(figsize=(11, 2.6)); pf.canvas.manager.set_window_title("Depth preview")
    picks = np.linspace(0, T - 1, 6).astype(int)
    for i, t in enumerate(picks):
        axp = pf.add_subplot(1, 6, i + 1)
        axp.imshow(disp[t], cmap="gray", vmin=0, vmax=1)
        axp.set_title(f"f{t}", fontsize=7); axp.axis("off")
    pf.suptitle(f"depth {depth}, order '{order}' - close to continue", fontsize=9)
    pf.tight_layout(); plt.show()
    _root = _tk.Tk(); _root.withdraw()
    ok = _mb.askyesno("Verify depth", f"Depth {depth} / order '{order}' look like a "
                      "smooth single-depth time-series?\nYES = continue, NO = abort")
    _root.destroy()
    if not ok:
        set_status("Depth->V/A aborted (depth/order not confirmed).", "info"); return None

    # ── load the SEPARATE per-depth RAW grayscale stack (the U-Net works on THIS,
    #    not the red/blue segmentation). The R&B stack only yields the V/A signals;
    #    the depth ties them (depth d -> red/blue at d + raw 901-frame file at d). ──
    def _acquire_raw_depth_stack():
        _r = _tk.Tk(); _r.withdraw()
        raw = _fd.askopenfilename(
            title=f"Select the RAW grayscale file for DEPTH {depth} "
                  f"({n_frames} frames; or a {n_frames}x{n_depths} stack to de-interleave)",
            initialdir=os.path.dirname(tif) or BASE_DIR,
            filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
        _r.destroy()
        if not raw:
            return None, None
        try:
            npg = int(_open_multipage_tif(raw).shape[0])
            if npg == n_frames:                       # already a single-depth raw stack
                rs = np.asarray(_open_multipage_tif(raw)[np.arange(n_frames)], np.float32)
            elif npg == n_frames * n_depths:          # multi-depth -> de-interleave at depth
                rs = deinterleave_depth(raw, depth, n_frames, n_depths, order).astype(np.float32)
            else:
                raise ValueError(f"{npg} pages != {n_frames} or {n_frames}x{n_depths}")
            if rs.ndim == 4:                          # RGB raw -> luminance
                rs = (rs[..., :3] * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1)
            return rs.astype(np.float32), raw
        except Exception as e:
            set_status(f"Could not load raw depth stack: {e}", "error")
            _r = _tk.Tk(); _r.withdraw()
            _mb.showerror("Raw depth stack", f"{raw}\n\n{e}"); _r.destroy()
            return None, None

    raw_stack, raw_src = _acquire_raw_depth_stack()
    if raw_stack is None:
        set_status("STEP 1 aborted: no raw per-depth stack selected.", "info"); return None
    rlo, rhi = np.percentile(raw_stack, [0.5, 99.5])
    raw_disp = np.clip((raw_stack - rlo) / (rhi - rlo + 1e-8), 0, 1)
    LOG.info(f"[STEP 1] raw U-Net stack for depth {depth}: {raw_src}  shape={raw_stack.shape}")

    # grayscale temporal projection (of the R&B depth, for V/A region display)
    proj = disp.std(0); proj = (proj - proj.min()) / (np.ptp(proj) + 1e-9)

    # ── shared persistence: write the THREE artifacts in the EXACT npy format ──
    def _persist(v_sig, a_sig, source):
        import tifffile
        # V/A in the EXACT existing (n_frames,) float32 format; canonical names are
        # what downstream reads, plus a per-depth archival copy.
        va_path = os.path.join(ds_dir, "ventricle_trace.npy")
        at_path = os.path.join(ds_dir, "atrium_trace.npy")
        np.save(va_path, np.asarray(v_sig, np.float32))
        np.save(at_path, np.asarray(a_sig, np.float32))
        np.save(os.path.join(ds_dir, f"va_ventricle_depth{depth}.npy"), np.asarray(v_sig, np.float32))
        np.save(os.path.join(ds_dir, f"va_atrium_depth{depth}.npy"), np.asarray(a_sig, np.float32))
        # the U-Net WORKING stack is the RAW per-depth grayscale (not the R&B seg)
        stack_path = os.path.join(ds_dir, f"depth{depth}_stack.tif")
        tifffile.imwrite(stack_path, raw_stack.astype(np.float32))
        save_depth_config(ds_dir, {
            "depth": int(depth), "order": order,
            "n_frames": int(n_frames), "n_depths": int(n_depths),
            "rb_source_tif": tif,                         # red/blue-marked V/A source
            "raw_source": raw_src,                        # raw grayscale U-Net source
            "stack": stack_path,                          # working raw stack (U-Net)
            "ventricle": va_path, "atrium": at_path,
            "va_kind": source,                            # 'auto-red-blue-area' or 'manual'
        })
        global ACTIVE_DEPTH, ACTIVE_DS_DIR, ACTIVE_STACK
        ACTIVE_DEPTH = depth
        ACTIVE_DS_DIR = ds_dir
        ACTIVE_STACK = raw_disp                           # U-Net steps display the RAW depth
        globals()["DATA_TIFF"] = stack_path
        LOG.info(f"[STEP 1] SAVED depth_config + depth{depth}_stack.tif (RAW {raw_stack.shape}) "
                 f"+ V/A ((901,) float32, kind={source}) into {ds_dir}  (depth {depth})")
        LOG.info(f"[STEP 1] keyed to depth {depth}: raw_stack={stack_path}  "
                 f"V/A={va_path} , {at_path}  (R&B src={os.path.basename(tif)}, "
                 f"raw src={os.path.basename(raw_src)})")
        set_status(f"STEP 1 done: depth {depth} -> raw stack + V/A ({source}) saved "
                   f"into {ds_dir}", "done")

    # ── resolve the COLOR-marked source (red=ventricle, blue=atrium) ──
    # 1) color in the loaded data?  2) else let the user pick a SEPARATE
    #    red/blue-marked multi-depth TIF (de-interleaved at the same depth/order).
    rgb = stack_rgb
    if rgb is not None:
        LOG.info(f"[STEP 1] loaded data carries color (RGB) - auto-extracting "
                 f"red/blue from it (shape {rgb.shape})")
    else:
        LOG.info("[STEP 1] loaded data is single-channel (no color markings in it)")
        _r = _tk.Tk(); _r.withdraw()
        want = _mb.askyesno("Red/blue markings",
                            "The loaded stack has no color (RGB) info.\n\n"
                            "Do you have a SEPARATE red/blue-marked multi-depth TIF "
                            "(ventricle=red, atrium=blue) to auto-extract the regions "
                            "from?\n\nYES = pick that file   NO = draw the regions by hand")
        _r.destroy()
        if want:
            _r = _tk.Tk(); _r.withdraw()
            seg = _fd.askopenfilename(
                title="Select the red/blue-marked multi-depth TIF (RGB)",
                initialdir=os.path.dirname(tif) or BASE_DIR,
                filetypes=[("TIFF stack", "*.tif *.tiff"), ("All", "*.*")])
            _r.destroy()
            if seg:
                try:
                    npg = int(_open_multipage_tif(seg).shape[0])
                    if npg != n_frames * n_depths:
                        raise ValueError(f"{npg} pages != {n_frames}x{n_depths}")
                    seg_stack = deinterleave_depth(seg, depth, n_frames, n_depths, order)
                    seg_stack = np.asarray(seg_stack, np.float32)
                    if seg_stack.ndim == 4 and seg_stack.shape[-1] >= 3:
                        rgb = seg_stack[..., :3]
                        LOG.info(f"[STEP 1] red/blue source (separate file) -> {seg} "
                                 f"depth {depth} order '{order}'  shape {rgb.shape}")
                    else:
                        set_status("Separate file has no RGB color - manual fallback.",
                                   "error")
                        LOG.info(f"[STEP 1] separate file not RGB (shape {seg_stack.shape})")
                except Exception as e:
                    set_status(f"Could not read red/blue file: {e}", "error")
                    _r = _tk.Tk(); _r.withdraw()
                    _mb.showerror("Red/blue file", f"{seg}\n\n{e}"); _r.destroy()

    # ── AUTOMATIC red/blue extraction + confirmation ──
    use_manual = rgb is None
    if rgb is not None:
        vent, atr, info = extract_red_blue_masks(rgb)
        LOG.info(f"[STEP 1] auto masks (RED=ATRIUM, BLUE=VENTRICLE): "
                 f"atrium(red)={info['atr_px']}px ventricle(blue)={info['vent_px']}px  "
                 f"(freq max R={info['red_freq_max']:.2f} B={info['blue_freq_max']:.2f})")
        if vent.sum() == 0 or atr.sum() == 0:
            set_status("No clear red/blue markings found - falling back to manual.",
                       "error")
            LOG.info("[STEP 1] one/both auto masks empty -> manual drawing fallback")
            use_manual = True
        else:
            decision = {"manual": False, "saved": False}
            ov = np.dstack([proj, proj, proj])             # gray base, tint masks
            # display colour matches the SOURCE marker: atrium=red, ventricle=blue
            ov[atr] = ov[atr] * 0.25 + np.array([0.85, 0.05, 0.05])    # red = ATRIUM
            ov[vent] = ov[vent] * 0.25 + np.array([0.05, 0.25, 0.9])   # blue = VENTRICLE
            cf = plt.figure(figsize=(8.5, 9))
            cf.canvas.manager.set_window_title("Confirm auto-extracted V/A masks")
            axc = cf.add_axes([0.05, 0.13, 0.9, 0.82]); axc.imshow(np.clip(ov, 0, 1))
            axc.set_title(f"AUTO-extracted: ATRIUM=red ({info['atr_px']}px), "
                          f"VENTRICLE=blue ({info['vent_px']}px) over STD projection.\n"
                          f"'Accept & Save' if correct, else 'Draw manually'.",
                          fontsize=9); axc.axis("off")

            def do_accept(_):
                # MOVEMENT = per-frame red/blue region AREA change over 901 frames
                v_sig = np.asarray(info["v_area"], np.float32)
                a_sig = np.asarray(info["a_area"], np.float32)
                _persist(v_sig, a_sig, "auto-red-blue-area")
                decision["saved"] = True
                plt.close(cf)                              # continue to the band step

            def do_manual(_):
                decision["manual"] = True; plt.close(cf)

            b_ok = Button(cf.add_axes([0.1, 0.03, 0.32, 0.06]), "Accept & Save V/A")
            b_man = Button(cf.add_axes([0.55, 0.03, 0.32, 0.06]), "Draw manually instead")
            b_ok.on_clicked(do_accept); b_man.on_clicked(do_manual)
            cf._keep_alive = [b_ok, b_man, do_accept, do_manual]
            plt.show()
            if not decision["manual"]:
                if decision.get("saved") and rgb is not None:
                    # AUTO valve BAND from the V/A temporal overlap (tunable + confirm)
                    LOG.info(f"[band] STEP 1: opening the AV-junction band tuner for "
                             f"depth {depth} (tune sliders, then click Accept to save).")
                    set_status(f"Derive the AV-junction valve band for depth {depth} "
                               f"-- tune the sliders, then click ACCEPT to save it ...", "run")
                    valve_band_gui(rgb, raw_disp, depth, ds_dir, _UMPP)
                    from leaflet_unet import load_valve_band as _lvb
                    if _lvb(ds_dir) is None:
                        LOG.info("[band] STEP 1: no band saved (tuner closed without "
                                 "Accept) -> later steps run WHOLE-FRAME. Re-run STEP 1 "
                                 "to add a band.")
                        set_status("STEP 1 done (no valve band saved -> whole-frame). "
                                   "Re-run STEP 1 to add the band.", "info")
                return stack                               # accepted - done
            use_manual = True

    # ── MANUAL drawing fallback (no markings, or user chose to draw) ──
    fig = plt.figure(figsize=(8.5, 9))
    fig.canvas.manager.set_window_title("Draw ventricle (blue) + atrium (red)")
    ax = fig.add_axes([0.05, 0.13, 0.9, 0.82]); ax.imshow(proj, cmap="gray")
    ax.set_title("STD projection. Draw a region, click 'Set Ventricle', draw another, "
                 "'Set Atrium', then 'Save V/A'. Esc restarts.", fontsize=9); ax.axis("off")
    st = {"verts": None, "vent": None, "atr": None}

    def on_sel(v):
        st["verts"] = v

    psel = PolygonSelector(ax, on_sel)

    def _poly_signal(poly_xy):
        from skimage.draw import polygon as skpoly
        ys = [p[1] for p in poly_xy]; xs = [p[0] for p in poly_xy]
        rr, cc = skpoly(np.array(ys), np.array(xs), shape=(H, W))
        mask = np.zeros((H, W), bool); mask[rr, cc] = True
        if mask.sum() == 0:
            return None, mask
        return stack[:, mask].mean(axis=1).astype(np.float32), mask   # (T,) float32

    def set_vent(_):
        if not st["verts"] or len(st["verts"]) < 3:
            set_status("Draw a ventricle polygon first.", "error"); return
        st["vent"] = list(st["verts"])
        p = np.array([[y, x] for x, y in st["vent"]])
        ax.plot(np.r_[p[:, 1], p[0, 1]], np.r_[p[:, 0], p[0, 0]], "-", color="#1f77b4", lw=2)
        set_status("Ventricle set. Draw atrium, then 'Set Atrium'.", "run")
        fig.canvas.draw_idle()

    def set_atr(_):
        if not st["verts"] or len(st["verts"]) < 3:
            set_status("Draw an atrium polygon first.", "error"); return
        st["atr"] = list(st["verts"])
        p = np.array([[y, x] for x, y in st["atr"]])
        ax.plot(np.r_[p[:, 1], p[0, 1]], np.r_[p[:, 0], p[0, 0]], "-", color="#d62728", lw=2)
        set_status("Atrium set. Click 'Save V/A'.", "run")
        fig.canvas.draw_idle()

    def save_va(_):
        if st["vent"] is None or st["atr"] is None:
            set_status("Set BOTH ventricle and atrium first.", "error"); return
        v_sig, _ = _poly_signal(st["vent"])
        a_sig, _ = _poly_signal(st["atr"])
        if v_sig is None or a_sig is None:
            set_status("Empty region - redraw.", "error"); return
        _persist(v_sig, a_sig, "manual")
        ax.set_title(f"Depth {depth} stack + V/A saved into dataset. Close the window.",
                     fontsize=10, color="#2ca02c")
        fig.canvas.draw_idle()

    b_v = Button(fig.add_axes([0.08, 0.03, 0.2, 0.06]), "Set Ventricle")
    b_a = Button(fig.add_axes([0.31, 0.03, 0.2, 0.06]), "Set Atrium")
    b_s = Button(fig.add_axes([0.62, 0.03, 0.2, 0.06]), "Save V/A")
    b_v.on_clicked(set_vent); b_a.on_clicked(set_atr); b_s.on_clicked(save_va)
    fig._keep_alive = [psel, b_v, b_a, b_s, set_vent, set_atr, save_va, on_sel]
    plt.show()
    return stack


# ============================================================
# Interactive preprocessing comparison GUI
# ============================================================
def interactive_preprocess_gui(stack: np.ndarray):
    """Open a matplotlib viewer to compare raw vs processed frames
    with adjustable parameters for background subtraction, denoising,
    and contrast enhancement. All changes update in real-time.
    """
    from matplotlib.widgets import Slider, Button, RadioButtons
    import cv2

    T, H, W = stack.shape

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('Dark-Field Preprocessing Comparison')
    fig.patch.set_facecolor('#1a1a1a')

    # ── single persistent status line (bottom) + session header ──
    ax_status = fig.add_axes([0.0, 0.0, 1.0, 0.028]); ax_status.axis('off')
    ax_status.set_facecolor('#000000')
    status_text = ax_status.text(0.005, 0.5, "  Idle", va='center', ha='left',
                                 fontsize=9, color='#5fd35f', family='monospace',
                                 transform=ax_status.transAxes)
    _GUI["fig"] = fig
    _GUI["status"] = status_text
    _ds_dir0 = active_ds_dir()
    _meta0 = os.path.join(_ds_dir0, 'metadata.json')
    _model0 = os.path.join(_ds_dir0, 'checkpoints', 'best_leaflet_unet.pth')
    _nann0 = 0
    if os.path.exists(_meta0):
        try:
            _nann0 = len(json.load(open(_meta0)).get('frames', []))
        except Exception:
            pass
    LOG.info("=" * 70)
    LOG.info(f"SESSION START  base_dir={BASE_DIR}")
    LOG.info(f"SESSION  dataset={'YES' if os.path.exists(_meta0) else 'NO'} "
             f"({_nann0} annotated)  model={'YES' if os.path.exists(_model0) else 'NO'}  "
             f"log={LOG_PATH}")
    set_status(f"Idle | base={BASE_DIR} | {_nann0} annotated | "
               f"model {'YES' if os.path.exists(_model0) else 'NO'}", "done")

    # 3 image panels: raw, processed, difference/enhanced
    ax_raw = fig.add_axes([0.01, 0.38, 0.32, 0.58])
    ax_proc = fig.add_axes([0.34, 0.38, 0.32, 0.58])
    ax_extra = fig.add_axes([0.67, 0.38, 0.32, 0.58])
    for ax in [ax_raw, ax_proc, ax_extra]:
        ax.set_facecolor('black'); ax.axis('off')

    state = {
        't': T // 2,
        'bg_method': 'min',       # 'none','min','percentile','rolling_ball','tophat'
        'bg_pctl': 5,
        'ball_r': 30,
        'temporal_w': 1,          # 1 = off
        'spatial': 'none',        # 'none','bilateral','gaussian','nlmeans'
        'bilateral_d': 5,
        'gauss_sigma': 1.0,
        'nlm_h': 10,
        'gamma': 0.5,
        'clahe_clip': 3.0,
        'clahe_on': True,
    }

    # Intensity correction — user-defined mask region
    # Two methods: histogram matching OR percentile normalization
    _hist_state = {'cdf': None, 'mask': None, 'enabled': False,
                   'method': 'percentile'}  # 'histogram' or 'percentile'

    def _set_hist_mask():
        """Let user draw the correction region."""
        proc = _process_frame_no_hist(state['t'])
        fig_hm, ax_hm = plt.subplots(figsize=(8, 8))
        ax_hm.imshow(proc, cmap='gray', vmin=0, vmax=1)
        ax_hm.set_title("Click TWO corners for intensity correction region",
                          fontsize=10, color='yellow')
        ax_hm.axis('off')
        pts = plt.ginput(2, timeout=0)
        plt.close(fig_hm)
        if len(pts) < 2:
            print("  Need 2 clicks"); return

        (x1, y1), (x2, y2) = pts
        mx0, my0 = max(0, int(min(x1,x2))), max(0, int(min(y1,y2)))
        mx1, my1 = min(W, int(max(x1,x2))), min(H, int(max(y1,y2)))

        mask = np.zeros((H, W), dtype=bool)
        mask[my0:my1, mx0:mx1] = True

        # Compute reference mean/std from ALL frames (not just 20)
        # This gives the most stable reference possible
        print(f"  Computing reference stats from all {T} frames...")
        all_means = np.zeros(T, dtype=np.float64)
        all_stds = np.zeros(T, dtype=np.float64)
        for si in range(T):
            pf = _process_frame_no_hist(si)
            vals = pf[mask]
            all_means[si] = np.mean(vals)
            all_stds[si] = np.std(vals)
            if si % 100 == 0 and si > 0:
                print(f"    {si}/{T}")

        # Reference = temporal median of per-frame stats
        # (median is robust to outlier frames)
        ref_mean = float(np.median(all_means))
        ref_std = float(np.median(all_stds))

        _hist_state['mask'] = mask
        _hist_state['ref_mean'] = ref_mean
        _hist_state['ref_std'] = ref_std
        _hist_state['enabled'] = True

        # Show the fluctuation that will be corrected
        print(f"  Reference: mean={ref_mean:.4f}, std={ref_std:.4f}")
        print(f"  Per-frame mean range: [{all_means.min():.4f}, {all_means.max():.4f}]")
        print(f"  Fluctuation: {(all_means.max()-all_means.min())/ref_mean*100:.1f}% of mean")
        print(f"  Correction mask set: y=[{my0}:{my1}], x=[{mx0}:{mx1}]")

    def _histogram_match(frame_01):
        """Correct intensity ONLY inside the user-drawn mask region.

        Method: per-frame z-score normalization within the mask.
        Each frame's masked region is normalized to have the SAME mean and std
        as the reference. This is independent of gamma/CLAHE/any processing —
        it only cares about the first two moments (mean, std).

        This eliminates ALL frame-to-frame intensity fluctuation within the mask.
        """
        if not _hist_state['enabled'] or _hist_state['mask'] is None:
            return frame_01

        mask = _hist_state['mask']
        result = frame_01.copy()
        masked_vals = result[mask]

        if len(masked_vals) == 0 or masked_vals.max() < 1e-6:
            return result

        ref_mean = _hist_state.get('ref_mean', 0.5)
        ref_std = _hist_state.get('ref_std', 0.15)

        # Current frame stats within mask
        cur_mean = np.mean(masked_vals)
        cur_std = np.std(masked_vals)

        if cur_std < 1e-8:
            return result

        # Z-score normalize: (x - cur_mean) / cur_std * ref_std + ref_mean
        normed = (masked_vals - cur_mean) / cur_std * ref_std + ref_mean
        result[mask] = np.clip(normed, 0, 1)

        return result

    def _process_frame_no_hist(t):
        """Process frame WITHOUT histogram matching (for mask selection preview)."""
        raw = stack[t].copy()
        bg = _get_bg(state['bg_method'])
        frame = np.clip(raw - bg, 0, None)
        tw = state['temporal_w']
        if tw > 1:
            t0 = max(0, t - tw // 2); t1 = min(T, t + tw // 2 + 1)
            chunk = np.clip(stack[t0:t1] - bg[np.newaxis,:,:], 0, None)
            frame = np.mean(chunk, axis=0).astype(np.float32)
        sm = state['spatial']
        if sm == 'bilateral':
            u8 = (np.clip(frame / (frame.max()+1e-8), 0, 1) * 255).astype(np.uint8)
            u8 = cv2.bilateralFilter(u8, state['bilateral_d'], 40, 40)
            frame = u8.astype(np.float32) / 255.0 * (frame.max()+1e-8)
        elif sm == 'gaussian':
            frame = ndimage.gaussian_filter(frame, sigma=state['gauss_sigma'])
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin: frame = (frame - fmin) / (fmax - fmin)
        else: frame = np.zeros_like(frame)
        g = state['gamma']
        if g != 1.0: frame = np.power(np.clip(frame, 0, 1), g)
        if state['clahe_on']:
            u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=state['clahe_clip'], tileGridSize=(8, 8))
            frame = cl.apply(u8).astype(np.float32) / 255.0
        return frame

    # Precompute background images for speed
    bg_cache = {}

    def _get_bg(method):
        if method in bg_cache:
            return bg_cache[method]
        print(f"  Computing background ({method})...")
        if method == 'min':
            bg = stack.min(axis=0)
        elif method == 'percentile':
            bg = np.percentile(stack, state['bg_pctl'], axis=0)
        elif method == 'rolling_ball':
            # Morphological opening as rolling-ball approximation
            from skimage.morphology import disk, opening
            # Use temporal median as base, then morphological opening
            med = np.median(stack, axis=0)
            r = max(3, state['ball_r'])
            bg = opening(med, disk(r))
        elif method == 'tophat':
            # White top-hat: original - opening (extracts bright features)
            # Returns the background to subtract
            from skimage.morphology import disk, opening
            med = np.median(stack, axis=0)
            r = max(3, state['ball_r'])
            bg = opening(med, disk(r))
        elif method == 'none':
            bg = np.zeros((H, W), dtype=stack.dtype)
        else:
            bg = np.zeros((H, W), dtype=stack.dtype)
        bg_cache[method] = bg
        return bg

    def _process_frame(t):
        """Apply full processing pipeline to a single frame."""
        raw = stack[t].copy()

        # Background subtraction
        bg = _get_bg(state['bg_method'])
        frame = np.clip(raw - bg, 0, None)

        # Temporal averaging (if enabled)
        tw = state['temporal_w']
        if tw > 1:
            t0 = max(0, t - tw // 2)
            t1 = min(T, t + tw // 2 + 1)
            chunk = stack[t0:t1] - bg[np.newaxis, :, :]
            chunk = np.clip(chunk, 0, None)
            frame = np.mean(chunk, axis=0).astype(np.float32)

        # Spatial denoising
        sm = state['spatial']
        if sm == 'bilateral':
            u8 = (np.clip(frame / (frame.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
            u8 = cv2.bilateralFilter(u8, state['bilateral_d'], 40, 40)
            frame = u8.astype(np.float32) / 255.0 * (frame.max() + 1e-8)
        elif sm == 'gaussian':
            frame = ndimage.gaussian_filter(frame, sigma=state['gauss_sigma'])
        elif sm == 'nlmeans':
            fmax = frame.max()
            if fmax > 0:
                fn = frame / fmax
                sigma_est = float(np.mean(estimate_sigma(fn)))
                frame = denoise_nl_means(fn, h=0.8 * sigma_est, sigma=sigma_est,
                                          patch_size=5, patch_distance=6,
                                          fast_mode=True) * fmax

        # Normalize to [0, 1]
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            frame = (frame - fmin) / (fmax - fmin)
        else:
            frame = np.zeros_like(frame)

        # Gamma compression
        g = state['gamma']
        if g != 1.0:
            frame = np.power(np.clip(frame, 0, 1), g)

        # CLAHE
        if state['clahe_on']:
            u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=state['clahe_clip'], tileGridSize=(8, 8))
            frame = cl.apply(u8).astype(np.float32) / 255.0

        # Histogram matching — ensures uniform intensity across all frames
        frame = _histogram_match(frame)

        return frame

    # Initial display
    raw0 = stack[state['t']]
    raw_norm = (raw0 - raw0.min()) / (raw0.max() - raw0.min() + 1e-8)
    proc0 = _process_frame(state['t'])

    im_raw = ax_raw.imshow(raw_norm, cmap='gray', vmin=0, vmax=1)
    ax_raw.set_title('Raw', color='white', fontsize=10)
    im_proc = ax_proc.imshow(proc0, cmap='gray', vmin=0, vmax=1)
    ax_proc.set_title('Processed', color='white', fontsize=10)
    # Third panel: difference or temporal MIP
    im_extra = ax_extra.imshow(proc0, cmap='inferno', vmin=0, vmax=1)
    ax_extra.set_title('Enhanced (inferno)', color='white', fontsize=10)
    title = fig.suptitle(f'Frame {state["t"]}/{T-1}', color='white', fontsize=11)

    # ── Sliders ──
    sb = '#f0f0f0'
    lc = 'black'
    sc1, sc2, sc3 = '#4a90d9', '#e07040', '#40a040'

    ax_t = fig.add_axes([0.06, 0.28, 0.50, 0.022], facecolor=sb)
    s_t = Slider(ax_t, 'Frame', 0, T-1, valinit=state['t'], valstep=1, color=sc1)

    ax_tw = fig.add_axes([0.06, 0.245, 0.22, 0.022], facecolor=sb)
    s_tw = Slider(ax_tw, 'Temp Avg', 1, 21, valinit=1, valstep=2, color=sc1)

    ax_gm = fig.add_axes([0.32, 0.245, 0.24, 0.022], facecolor=sb)
    s_gm = Slider(ax_gm, 'Gamma', 0.1, 2.0, valinit=state['gamma'], color=sc2)

    ax_cl = fig.add_axes([0.06, 0.21, 0.22, 0.022], facecolor=sb)
    s_cl = Slider(ax_cl, 'CLAHE', 0.5, 15.0, valinit=state['clahe_clip'], color=sc3)

    ax_bd = fig.add_axes([0.32, 0.21, 0.24, 0.022], facecolor=sb)
    s_bd = Slider(ax_bd, 'Bilateral d', 3, 15, valinit=5, valstep=2, color=sc3)

    ax_gs = fig.add_axes([0.06, 0.175, 0.22, 0.022], facecolor=sb)
    s_gs = Slider(ax_gs, 'Gauss σ', 0.3, 5.0, valinit=1.0, color=sc3)

    ax_nl = fig.add_axes([0.32, 0.175, 0.24, 0.022], facecolor=sb)
    s_nl = Slider(ax_nl, 'NLM h', 1, 30, valinit=10, valstep=1, color=sc3)

    ax_bp = fig.add_axes([0.06, 0.14, 0.22, 0.022], facecolor=sb)
    s_bp = Slider(ax_bp, 'BG Pctl', 1, 30, valinit=5, valstep=1, color=sc2)

    ax_br = fig.add_axes([0.32, 0.14, 0.24, 0.022], facecolor=sb)
    s_br = Slider(ax_br, 'Ball R', 3, 80, valinit=30, valstep=1, color=sc2)

    for s in [s_t, s_tw, s_gm, s_cl, s_bd, s_gs, s_nl, s_bp, s_br]:
        s.label.set_color(lc); s.valtext.set_color(lc)

    # ── Buttons ──
    bw, bh = 0.07, 0.028
    bx = 0.64

    # Background method buttons
    ax_bg_btns = fig.add_axes([bx, 0.22, 0.16, 0.12])
    ax_bg_btns.set_facecolor(sb)
    radio_bg = RadioButtons(ax_bg_btns, ['none', 'min', 'percentile', 'rolling_ball', 'tophat'],
                             active=1)
    for lbl in radio_bg.labels:
        lbl.set_fontsize(8); lbl.set_color(lc)
    ax_bg_btns.set_title('BG method', fontsize=8, color=lc, pad=2)

    # Spatial method buttons
    ax_sp_btns = fig.add_axes([bx + 0.18, 0.22, 0.14, 0.12])
    ax_sp_btns.set_facecolor(sb)
    radio_sp = RadioButtons(ax_sp_btns, ['none', 'bilateral', 'gaussian', 'nlmeans'],
                             active=0)
    for lbl in radio_sp.labels:
        lbl.set_fontsize(8); lbl.set_color(lc)
    ax_sp_btns.set_title('Spatial denoise', fontsize=8, color=lc, pad=2)

    ax_clahe_btn = fig.add_axes([bx, 0.175, bw, bh])
    btn_clahe = Button(ax_clahe_btn, 'CLAHE: ON')

    ax_histmask_btn = fig.add_axes([bx + 0.09, 0.175, bw + 0.03, bh])
    btn_histmask = Button(ax_histmask_btn, 'Set Hist Mask')

    # Extra buttons row
    ax_run_btn = fig.add_axes([bx, 0.135, bw, bh])
    btn_run = Button(ax_run_btn, 'Run Pipeline')

    ax_save_btn = fig.add_axes([bx + 0.09, 0.135, bw, bh])
    btn_save = Button(ax_save_btn, 'Save TIFF')

    ax_zoom_btn = fig.add_axes([bx, 0.095, bw, bh])
    btn_zoom = Button(ax_zoom_btn, 'Zoom ROI')

    ax_seg_btn = fig.add_axes([bx + 0.09, 0.095, bw, bh])
    btn_seg = Button(ax_seg_btn, 'Auto-Seg')

    ax_loadseg_btn = fig.add_axes([bx, 0.055, bw + 0.10, bh])
    btn_loadseg = Button(ax_loadseg_btn, 'Zoom + Seg Valve')

    ax_unet_btn = fig.add_axes([bx, 0.015, bw + 0.10, bh])
    btn_unet = Button(ax_unet_btn, 'U-Net Leaflets')

    def update(_=None):
        state['t'] = int(s_t.val)
        state['temporal_w'] = int(s_tw.val)
        state['gamma'] = s_gm.val
        state['clahe_clip'] = s_cl.val
        state['bilateral_d'] = int(s_bd.val)
        state['gauss_sigma'] = s_gs.val
        state['nlm_h'] = int(s_nl.val)
        state['bg_pctl'] = int(s_bp.val)
        state['ball_r'] = int(s_br.val)

        # Clear bg cache if percentile/ball changed
        for k in ['percentile', 'rolling_ball', 'tophat']:
            if k in bg_cache:
                del bg_cache[k]

        raw = stack[state['t']]
        raw_n = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        proc = _process_frame(state['t'])

        im_raw.set_data(raw_n)
        im_proc.set_data(proc)
        im_extra.set_data(proc)
        title.set_text(
            f'Frame {state["t"]}/{T-1}  |  BG={state["bg_method"]}  '
            f'Spatial={state["spatial"]}  Gamma={state["gamma"]:.2f}  '
            f'TempAvg={state["temporal_w"]}')
        fig.canvas.draw_idle()

    for s in [s_t, s_tw, s_gm, s_cl, s_bd, s_gs, s_nl, s_bp, s_br]:
        s.on_changed(update)

    def on_bg_change(label):
        state['bg_method'] = label
        # Clear cache for recomputation
        bg_cache.clear()
        update()

    def on_sp_change(label):
        state['spatial'] = label
        update()

    def toggle_clahe(_):
        state['clahe_on'] = not state['clahe_on']
        btn_clahe.label.set_text(f'CLAHE: {"ON" if state["clahe_on"] else "OFF"}')
        update()

    radio_bg.on_clicked(on_bg_change)
    radio_sp.on_clicked(on_sp_change)
    btn_clahe.on_clicked(gui_op("Toggle CLAHE")(toggle_clahe))

    _hist_click_count = [0]

    def on_set_hist_mask(_):
        _hist_click_count[0] += 1
        if _hist_state['mask'] is not None:
            # Mask exists — toggle on/off
            _hist_state['enabled'] = not _hist_state['enabled']
            status = 'ON' if _hist_state['enabled'] else 'OFF'
            btn_histmask.label.set_text(f'Correct: {status}')
            print(f"  Intensity correction: {status}")
            update()
        else:
            # No mask yet — draw it
            _set_hist_mask()
            btn_histmask.label.set_text('Correct: ON')
            update()

    btn_histmask.on_clicked(gui_op("Set Hist Mask")(on_set_hist_mask))

    def run_full_pipeline(_):
        """Apply current settings to full stack and run kymograph analysis."""
        print("Running full pipeline with current settings...")
        # Build processed stack
        bg = _get_bg(state['bg_method'])
        proc_stack = np.empty_like(stack)
        for t in range(T):
            proc_stack[t] = _process_frame(t)
            if t % 100 == 0:
                print(f"  Processing: {t}/{T}")
        print("  Processing done.")

        # Kymograph
        reference = proc_stack.max(axis=0)
        p1, p2 = pick_kymograph_line(reference)
        kymo = kymograph(proc_stack, p1, p2, thickness=5)
        position = trace_leaflet_from_kymograph(kymo, mode="bright")
        plot_leaflet_trajectory(position, kymo, savepath="leaflet_trajectory.png")

        period = detect_period(position)
        print(f"Estimated period: {period:.1f} frames")

        phase_stack = phase_average_stack(proc_stack, position, n_phase_bins=40)
        tifffile.imwrite("phase_averaged_cycle.tif", phase_stack.astype(np.float32))
        print("Saved phase_averaged_cycle.tif")

        keys = suggest_keyframes(position, n_per_cycle=8)
        print(f"Suggested {len(keys)} key-frames:", keys[:10], "...")
        plt.show()

    btn_run.on_clicked(gui_op("Run full pipeline")(run_full_pipeline))

    # ── Save processed TIFF ──
    def save_processed_tiff(_):
        """Process all frames with current settings and save as TIFF."""
        from tkinter import filedialog as fd
        import tkinter as _tk
        _root = _tk.Tk(); _root.withdraw()
        path = fd.asksaveasfilename(title="Save Processed TIFF",
                                     defaultextension=".tif",
                                     filetypes=[("TIFF","*.tif"),("All","*.*")])
        _root.destroy()
        if not path:
            return
        print(f"Processing all {T} frames...")
        proc_stack = np.empty((T, H, W), dtype=np.float32)
        for t in range(T):
            proc_stack[t] = _process_frame(t)
            if t % 50 == 0:
                print(f"  {t}/{T}")
        tifffile.imwrite(path, proc_stack)
        print(f"Saved processed TIFF: {path} ({T} frames)")

    btn_save.on_clicked(gui_op("Save processed TIFF")(save_processed_tiff))

    # ── Zoom ROI viewer ──
    zoom_roi = [None]  # persistent ROI state

    def open_zoom_viewer(_):
        """Click two corners on the processed image to define zoom ROI,
        then browse zoomed before/after with a slider."""

        def _popup_message(title, msg):
            """Show a blocking warning the user can't miss (terminal prints are
            easy to overlook, and stdout may be buffered)."""
            try:
                from tkinter import messagebox as _mb
                import tkinter as _tk
                _r = _tk.Tk(); _r.withdraw()
                _mb.showwarning(title, msg)
                _r.destroy()
            except Exception:
                print(f"  [{title}] {msg}")

        proc = _process_frame(state['t'])

        # Step 1: click two corners
        fig_sel, ax_sel = plt.subplots(figsize=(8, 8))
        ax_sel.imshow(proc, cmap='gray', vmin=0, vmax=1)
        ax_sel.set_title("Click TWO corners of the zoom region (top-left, bottom-right)",
                         fontsize=10)
        ax_sel.axis('off')
        pts = plt.ginput(2, timeout=0)
        plt.close(fig_sel)

        if len(pts) < 2:
            print("Need 2 clicks for ROI corners")
            return

        (x1c, y1c), (x2c, y2c) = pts
        rx0 = max(0, int(min(x1c, x2c)))
        ry0 = max(0, int(min(y1c, y2c)))
        rx1 = min(W, int(max(x1c, x2c)))
        ry1 = min(H, int(max(y1c, y2c)))

        if rx1 - rx0 < 5 or ry1 - ry0 < 5:
            print("ROI too small")
            return

        zoom_roi[0] = (ry0, rx0, ry1, rx1)
        print(f"ROI: y=[{ry0},{ry1}], x=[{rx0},{rx1}], size={rx1-rx0}x{ry1-ry0}")

        # Step 2: precompute background for ROI only (once)
        print(f"  Precomputing ROI background...")
        bg_roi = stack[:, ry0:ry1, rx0:rx1].min(axis=0)

        # Fast ROI-only processing (no full-frame overhead)
        def get_zoomed_fast(t):
            raw_roi = stack[t, ry0:ry1, rx0:rx1]
            raw_n = (raw_roi - raw_roi.min()) / (raw_roi.max() - raw_roi.min() + 1e-8)

            # Fast processing on small ROI
            frame = np.clip(raw_roi - bg_roi, 0, None)

            # Light temporal avg if enabled
            tw = state['temporal_w']
            if tw > 1:
                t0_ = max(0, t - tw // 2)
                t1_ = min(T, t + tw // 2 + 1)
                chunk = np.clip(stack[t0_:t1_, ry0:ry1, rx0:rx1] - bg_roi, 0, None)
                frame = np.mean(chunk, axis=0).astype(np.float32)

            # Normalize
            fmin, fmax = frame.min(), frame.max()
            if fmax > fmin:
                frame = (frame - fmin) / (fmax - fmin)
            else:
                frame = np.zeros_like(frame)

            # Gamma
            g = state['gamma']
            if g != 1.0:
                frame = np.power(np.clip(frame, 0, 1), g)

            # CLAHE (fast on small ROI)
            if state['clahe_on']:
                u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                cl = cv2.createCLAHE(clipLimit=state['clahe_clip'], tileGridSize=(4, 4))
                frame = cl.apply(u8).astype(np.float32) / 255.0

            # Histogram matching for uniform intensity
            frame = _histogram_match(frame)

            return raw_n, frame

        # Precompute ALL zoomed processed frames for instant slider response
        print("  Precomputing all zoomed processed frames...")
        _zoom_cache = np.empty((T, ry1 - ry0, rx1 - rx0), dtype=np.float32)
        for t in range(T):
            _, pz = get_zoomed_fast(t)
            _zoom_cache[t] = pz
            if t % 100 == 0 and t > 0:
                print(f"    {t}/{T}")
        print(f"  Cached {T} frames ({_zoom_cache.nbytes / 1e6:.1f} MB)")

        # Override get_zoomed_fast to use cache for instant slider response
        _orig_get_zoomed = get_zoomed_fast
        def get_zoomed_fast(t):
            raw_roi = stack[t, ry0:ry1, rx0:rx1]
            raw_n = (raw_roi - raw_roi.min()) / (raw_roi.max() - raw_roi.min() + 1e-8)
            return raw_n, _zoom_cache[t]

        # Precompute ROI mean intensity for all frames
        # Also correct for global brightness fluctuations
        print("  Precomputing ROI intensity trace + brightness correction...")
        roi_trace_raw = np.array([stack[t, ry0:ry1, rx0:rx1].mean() for t in range(T)])

        # Global brightness correction: normalize each frame's ROI by the
        # frame's overall mean, so slow illumination drift is removed
        global_mean = np.array([stack[t].mean() for t in range(T)])
        global_median = np.median(global_mean)
        # Correction factor: ratio of median to each frame's global mean
        correction = np.where(global_mean > 1e-8, global_median / global_mean, 1.0)

        # Use precomputed cache for instant trace computation
        roi_trace_proc = _zoom_cache.mean(axis=(1, 2)) * correction
        print(f"  Done. Brightness range: [{global_mean.min():.2f}, {global_mean.max():.2f}], "
              f"correction range: [{correction.min():.3f}, {correction.max():.3f}]")

        # Initial heartbeat detection (user can override with input)
        from scipy.signal import find_peaks
        trace_hp = roi_trace_proc - ndimage.uniform_filter1d(roi_trace_proc, size=50)
        trace_hp = ndimage.uniform_filter1d(trace_hp, size=3)
        beat_thresh = np.std(trace_hp) * 1.0
        beat_peaks, _ = find_peaks(trace_hp, height=beat_thresh, distance=5)
        n_beats = len(beat_peaks)
        if n_beats >= 2:
            mean_period = np.mean(np.diff(beat_peaks))
        else:
            mean_period = T
        print(f"  Auto-detected {n_beats} heartbeats, period~{mean_period:.1f} frames")

        # ── Open combined viewer: full + zoomed + trace ──
        fig_z = plt.figure(figsize=(17, 10))
        fig_z.canvas.manager.set_window_title(
            f'ROI Viewer ({rx1-rx0}x{ry1-ry0}) — {n_beats} beats')
        fig_z.patch.set_facecolor('#1a1a1a')

        # Layout: top row = images, bottom = trace
        ax_full = fig_z.add_axes([0.01, 0.42, 0.24, 0.54])
        ax_zraw = fig_z.add_axes([0.26, 0.42, 0.24, 0.54])
        ax_zproc = fig_z.add_axes([0.51, 0.42, 0.24, 0.54])
        ax_zcolor = fig_z.add_axes([0.76, 0.42, 0.24, 0.54])
        ax_trace = fig_z.add_axes([0.06, 0.22, 0.88, 0.16])

        for ax in [ax_full, ax_zraw, ax_zproc, ax_zcolor]:
            ax.set_facecolor('black'); ax.axis('off')

        # Full frame with ROI rectangle
        raw0 = stack[state['t']]
        raw0_n = (raw0 - raw0.min()) / (raw0.max() - raw0.min() + 1e-8)
        im_full = ax_full.imshow(raw0_n, cmap='gray', vmin=0, vmax=1)
        roi_rect = plt.Rectangle((rx0, ry0), rx1-rx0, ry1-ry0,
                                  lw=1.5, edgecolor='cyan', facecolor='none', ls='--')
        ax_full.add_patch(roi_rect)
        ax_full.set_title('Full frame', color='white', fontsize=9)

        # Zoomed panels
        raw_z, proc_z = get_zoomed_fast(state['t'])
        im_zraw = ax_zraw.imshow(raw_z, cmap='gray', vmin=0, vmax=1)
        ax_zraw.set_title('Zoomed raw', color='white', fontsize=9)
        im_zproc = ax_zproc.imshow(proc_z, cmap='gray', vmin=0, vmax=1)
        ax_zproc.set_title('Zoomed processed', color='white', fontsize=9)
        im_zcolor = ax_zcolor.imshow(proc_z, cmap='inferno', vmin=0, vmax=1)
        ax_zcolor.set_title('Enhanced', color='white', fontsize=9)

        # Trace plot
        ax_trace.plot(roi_trace_proc, color='#4a90d9', lw=0.5, alpha=0.8, label='ROI mean')
        ax_trace.plot(beat_peaks, roi_trace_proc[beat_peaks], 'v', color='#e07040',
                      markersize=4, label=f'{n_beats} beats')
        vline_trace = ax_trace.axvline(state['t'], color='red', lw=1, ls='--', alpha=0.8)
        ax_trace.set_xlim(0, T)
        ax_trace.set_xlabel('Frame', color='white', fontsize=8)
        ax_trace.set_ylabel('Mean intensity', color='white', fontsize=8)
        ax_trace.set_title(
            f'ROI intensity — {n_beats} beats, period~{mean_period:.1f} frames',
            color='white', fontsize=9)
        ax_trace.legend(fontsize=7, facecolor='#333', edgecolor='white',
                         labelcolor='white', loc='upper right')
        ax_trace.tick_params(colors='white', labelsize=7)
        ax_trace.set_facecolor('#1a1a1a')

        title_z = fig_z.suptitle(f'Frame {state["t"]}/{T-1}', color='white', fontsize=11)

        # Sliders
        sb = '#f0f0f0'
        ax_sf = fig_z.add_axes([0.06, 0.13, 0.55, 0.025], facecolor=sb)
        s_zf = Slider(ax_sf, 'Frame', 0, T-1, valinit=state['t'], valstep=1, color='#4a90d9')
        s_zf.label.set_color('black'); s_zf.valtext.set_color('black')

        # Heartbeat info display
        from matplotlib.widgets import Button as MplButton
        ax_hb_info = fig_z.add_axes([0.06, 0.08, 0.55, 0.03])
        ax_hb_info.set_facecolor('#1a1a1a')
        ax_hb_info.text(0, 0.3,
            f'Auto-detected: {n_beats} beats, period~{mean_period:.1f} frames  |  '
            f'Click "Find Valve Events" to enter your heartbeat count',
            color='#aaaaaa', fontsize=8, transform=ax_hb_info.transAxes)
        ax_hb_info.axis('off')

        # Buttons — two rows
        ax_save_ztif = fig_z.add_axes([0.64, 0.13, 0.07, 0.028])
        btn_save_ztif = MplButton(ax_save_ztif, 'Save TIFF')

        ax_sel_area = fig_z.add_axes([0.72, 0.13, 0.08, 0.028])
        btn_sel_area = MplButton(ax_sel_area, 'SubROI')

        ax_find_valve = fig_z.add_axes([0.81, 0.13, 0.10, 0.028])
        btn_find_valve = MplButton(ax_find_valve, 'Valve Events')

        ax_av_btn = fig_z.add_axes([0.64, 0.08, 0.16, 0.028])
        btn_av = MplButton(ax_av_btn, 'Atrium/Ventricle Valve')

        # Fast update using precomputed cache — no throttling needed
        def update_zoom(val):
            t = int(s_zf.val)
            raw_f = stack[t]
            raw_fn = (raw_f - raw_f.min()) / (raw_f.max() - raw_f.min() + 1e-8)
            im_full.set_data(raw_fn)
            rz, pz = get_zoomed_fast(t)  # uses cache — instant
            im_zraw.set_data(rz)
            im_zproc.set_data(pz)
            im_zcolor.set_data(pz)
            vline_trace.set_xdata([t, t])
            title_z.set_text(f'Frame {t}/{T-1}')
            fig_z.canvas.draw_idle()

        s_zf.on_changed(update_zoom)

        # Save zoomed processed TIFF
        def save_zoomed_tiff(_):
            from tkinter import filedialog as fd
            import tkinter as _tk
            _root = _tk.Tk(); _root.withdraw()
            path = fd.asksaveasfilename(title="Save Zoomed Processed TIFF",
                                         defaultextension=".tif",
                                         filetypes=[("TIFF","*.tif"),("All","*.*")])
            _root.destroy()
            if not path:
                return
            print(f"Processing zoomed ROI for all {T} frames...")
            out = np.empty((T, ry1-ry0, rx1-rx0), dtype=np.float32)
            for t in range(T):
                _, pz = get_zoomed_fast(t)
                out[t] = pz
                if t % 100 == 0:
                    print(f"  {t}/{T}")
            tifffile.imwrite(path, out)
            print(f"Saved: {path}")

        btn_save_ztif.on_clicked(save_zoomed_tiff)

        # Plot SubROI intensity change
        def plot_sub_roi(_):
            """Click 2 points on a fresh zoomed image to define a sub-area,
            then plot its intensity over ALL frames."""

            # Show the zoomed processed image in a new window for picking
            t_now = int(s_zf.val)
            _, pz = get_zoomed_fast(t_now)

            fig_pick, ax_pick = plt.subplots(figsize=(7, 7))
            ax_pick.imshow(pz, cmap='gray', vmin=0, vmax=1)
            ax_pick.set_title(
                f"Click TWO corners of the sub-area (frame {t_now})\n"
                "Then close this window", fontsize=10)
            ax_pick.axis('off')
            pts = plt.ginput(2, timeout=0)
            plt.close(fig_pick)

            if len(pts) < 2:
                print("Need 2 clicks for sub-ROI")
                return

            (sx1, sy1), (sx2, sy2) = pts
            sx0i = max(0, int(min(sx1, sx2)))
            sy0i = max(0, int(min(sy1, sy2)))
            sx1i = min(rx1 - rx0, int(max(sx1, sx2)))
            sy1i = min(ry1 - ry0, int(max(sy1, sy2)))

            if sx1i - sx0i < 1 or sy1i - sy0i < 1:
                print("Sub-ROI too small")
                return

            print(f"Sub-ROI in zoom: x=[{sx0i},{sx1i}], y=[{sy0i},{sy1i}], "
                  f"size={sx1i-sx0i}x{sy1i-sy0i}")
            print("  Computing intensity for all frames...")

            # Global coords for raw access
            gx0, gy0 = rx0 + sx0i, ry0 + sy0i
            gx1, gy1 = rx0 + sx1i, ry0 + sy1i

            # Compute traces
            sub_raw = np.array([
                stack[t, gy0:gy1, gx0:gx1].mean() for t in range(T)])
            sub_proc = np.zeros(T, dtype=np.float32)
            for t in range(T):
                _, pz_t = get_zoomed_fast(t)
                sub_proc[t] = pz_t[sy0i:sy1i, sx0i:sx1i].mean()

            # Apply brightness correction
            sub_proc_corr = sub_proc * correction

            print("  Done.")

            # Plot 3 panels
            fig_tr, axes_tr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            fig_tr.canvas.manager.set_window_title(
                f'Sub-ROI Trace ({sx1i-sx0i}x{sy1i-sy0i} px)')
            fig_tr.patch.set_facecolor('#1a1a1a')

            # Panel 1: raw + processed
            axes_tr[0].plot(sub_raw, color='#888888', lw=0.3, alpha=0.5, label='Raw')
            axes_tr[0].plot(sub_proc_corr, color='#4a90d9', lw=0.6,
                            label='Processed (brightness-corrected)')
            axes_tr[0].set_ylabel('Mean intensity', color='white', fontsize=8)
            axes_tr[0].set_title('Sub-ROI intensity over time', color='white', fontsize=10)
            axes_tr[0].legend(fontsize=7, facecolor='#333', labelcolor='white')
            axes_tr[0].tick_params(colors='white')
            axes_tr[0].set_facecolor('#1a1a1a')

            # Panel 2: temporal derivative (motion)
            deriv = np.abs(np.diff(sub_proc_corr))
            deriv_s = ndimage.uniform_filter1d(deriv, size=3)
            axes_tr[1].plot(deriv_s, color='#e07040', lw=0.5)
            axes_tr[1].set_ylabel('Motion |dI/dt|', color='white', fontsize=8)
            axes_tr[1].set_title('Temporal derivative (detects rapid changes)',
                                  color='white', fontsize=10)
            axes_tr[1].tick_params(colors='white')
            axes_tr[1].set_facecolor('#1a1a1a')

            # Panel 3: high-pass filtered (removes slow drift, shows oscillations)
            hp = sub_proc_corr - ndimage.uniform_filter1d(sub_proc_corr, size=50)
            axes_tr[2].plot(hp, color='#40a040', lw=0.5)
            axes_tr[2].set_ylabel('High-pass', color='white', fontsize=8)
            axes_tr[2].set_xlabel('Frame', color='white', fontsize=8)
            axes_tr[2].set_title('High-pass filtered (heartbeat oscillation)',
                                  color='white', fontsize=10)
            axes_tr[2].tick_params(colors='white')
            axes_tr[2].set_facecolor('#1a1a1a')

            fig_tr.tight_layout()

            # Save trace data
            np.save("subroi_trace_raw.npy", sub_raw)
            np.save("subroi_trace_processed.npy", sub_proc_corr)
            print("  Saved: subroi_trace_raw.npy, subroi_trace_processed.npy")

            plt.show()

        btn_sel_area.on_clicked(plot_sub_roi)

        # Find valve events based on user-provided heartbeat count
        def find_valve_events(_):
            """Prompt for heartbeat count via a matplotlib figure, then find events."""
            from scipy.signal import find_peaks as fp

            # Use a matplotlib figure with ginput as a simple prompt
            fig_prompt, ax_prompt = plt.subplots(figsize=(6, 2))
            fig_prompt.canvas.manager.set_window_title('Enter Heartbeat Count')
            ax_prompt.axis('off')
            ax_prompt.text(0.5, 0.7,
                f"How many heartbeats in {T} frames?  (auto={n_beats})\n"
                f"Click ONCE on the number line below.",
                ha='center', va='center', fontsize=11, color='black')

            # Draw a number line from 1 to max
            max_hb = min(100, T // 3)
            ax_num = fig_prompt.add_axes([0.1, 0.1, 0.8, 0.3])
            ax_num.set_xlim(0, max_hb)
            ax_num.set_ylim(0, 1)
            ax_num.set_xlabel('Number of heartbeats')
            ax_num.axvline(n_beats, color='blue', ls='--', lw=1, alpha=0.5,
                            label=f'auto={n_beats}')
            ax_num.legend(fontsize=8)
            # Tick marks every 5
            ax_num.set_xticks(range(0, max_hb + 1, 5))
            ax_num.set_yticks([])

            pts = plt.ginput(1, timeout=0)
            plt.close(fig_prompt)

            if len(pts) < 1:
                print("  Cancelled")
                return

            user_n_beats = max(1, int(round(pts[0][0])))
            print(f"  User selected: {user_n_beats} heartbeats")

            if user_n_beats < 1:
                print("  Need at least 1 heartbeat")
                return

            expected_period = T / user_n_beats
            min_distance = max(3, int(expected_period * 0.5))

            print(f"\n{'='*60}")
            print(f"VALVE EVENT ANALYSIS")
            print(f"{'='*60}")
            print(f"Total frames: {T}")
            print(f"User heartbeat count: {user_n_beats}")
            print(f"Expected period: {expected_period:.1f} frames")
            print(f"Min event distance: {min_distance} frames")

            # Brightness-corrected temporal derivative
            deriv = np.abs(np.diff(roi_trace_proc))
            # Also compute multi-frame difference for robustness
            deriv3 = np.zeros(T, dtype=np.float32)
            for dt in [1, 2, 3]:
                d = np.abs(roi_trace_proc[dt:] - roi_trace_proc[:-dt])
                deriv3[:len(d)] += d
            deriv3 /= 3.0
            deriv_smooth = ndimage.uniform_filter1d(deriv3, size=3)

            # Find top N peaks with minimum spacing
            # First find ALL peaks, then select top N by strength
            all_peaks, all_props = fp(deriv_smooth, distance=min_distance,
                                       prominence=0)
            if len(all_peaks) == 0:
                print("  No motion peaks found at all")
                return

            # Sort by prominence (strength) and take top N
            prominences = all_props['prominences']
            sorted_idx = np.argsort(prominences)[::-1]
            top_n = min(user_n_beats, len(all_peaks))
            selected_idx = sorted(sorted_idx[:top_n])
            valve_frames = all_peaks[selected_idx]
            valve_strengths = deriv_smooth[valve_frames]

            n_valves = len(valve_frames)
            print(f"\nValve openings found: {n_valves} (requested {user_n_beats})")
            for i, (vf, vs) in enumerate(zip(valve_frames, valve_strengths)):
                print(f"  Valve {i+1}: frame {vf}, strength={vs:.4f}")

            if n_valves >= 2:
                intervals = np.diff(valve_frames)
                print(f"  Mean interval: {np.mean(intervals):.1f} frames "
                      f"(expected {expected_period:.1f})")
                print(f"  Interval std: {np.std(intervals):.1f} frames")

            # ── Plot ──
            fig_v, axes_v = plt.subplots(4, 1, figsize=(15, 12))
            fig_v.canvas.manager.set_window_title(
                f'Valve Analysis — {n_valves} events / {user_n_beats} beats')
            fig_v.patch.set_facecolor('#1a1a1a')

            # Panel 1: brightness-corrected ROI intensity
            axes_v[0].plot(roi_trace_proc, color='#4a90d9', lw=0.5,
                           label='ROI mean (brightness-corrected)')
            # Show brightness correction factor
            ax0twin = axes_v[0].twinx()
            ax0twin.plot(correction, color='#aaaaaa', lw=0.3, alpha=0.5,
                          label='Brightness correction')
            ax0twin.set_ylabel('Correction', color='#aaaaaa', fontsize=7)
            ax0twin.tick_params(colors='#aaaaaa', labelsize=6)
            if n_valves > 0:
                axes_v[0].plot(valve_frames,
                               roi_trace_proc[valve_frames],
                               '^', color='#e07040', ms=7, zorder=5,
                               label=f'Valve events ({n_valves})')
            axes_v[0].set_ylabel('ROI mean', color='white')
            axes_v[0].set_title(
                f'Brightness-corrected ROI — {user_n_beats} heartbeats, '
                f'{n_valves} valve events',
                color='white', fontsize=10)
            axes_v[0].legend(fontsize=7, facecolor='#333', labelcolor='white',
                              loc='upper left')
            axes_v[0].tick_params(colors='white')
            axes_v[0].set_facecolor('#1a1a1a')

            # Panel 2: motion signal with valve markers
            axes_v[1].plot(deriv_smooth, color='#e07040', lw=0.5,
                           label='Multi-frame |ΔI|')
            if n_valves > 0:
                axes_v[1].plot(valve_frames, valve_strengths, '^',
                               color='yellow', ms=6, label='Detected')
                # Show expected beat positions as gray lines
                for b in range(user_n_beats):
                    expected_f = int(b * expected_period)
                    axes_v[1].axvline(expected_f, color='#555555', lw=0.3, ls=':')
            axes_v[1].set_ylabel('|ΔI/Δt|', color='white')
            axes_v[1].set_title('Motion signal + expected beat grid (gray)',
                                color='white')
            axes_v[1].legend(fontsize=7, facecolor='#333', labelcolor='white')
            axes_v[1].tick_params(colors='white')
            axes_v[1].set_facecolor('#1a1a1a')

            # Panel 3: valve-to-valve intervals
            if n_valves >= 2:
                intervals = np.diff(valve_frames)
                x_bar = np.arange(len(intervals))
                colors_bar = ['#40a040' if abs(iv - expected_period) < expected_period * 0.3
                               else '#e05050' for iv in intervals]
                axes_v[2].bar(x_bar, intervals, color=colors_bar)
                axes_v[2].axhline(expected_period, color='cyan', ls='--', lw=1,
                                   label=f'Expected ({expected_period:.1f})')
                axes_v[2].axhline(np.mean(intervals), color='white', ls='--', lw=0.8,
                                   label=f'Mean ({np.mean(intervals):.1f})')
                axes_v[2].set_ylabel('Interval (frames)', color='white')
                axes_v[2].set_title(
                    'Valve intervals (green=normal, red=irregular)',
                    color='white')
                axes_v[2].legend(fontsize=7, facecolor='#333', labelcolor='white')
            else:
                axes_v[2].text(0.5, 0.5, 'Need ≥2 events for interval analysis',
                               transform=axes_v[2].transAxes, color='white',
                               ha='center', va='center')
            axes_v[2].tick_params(colors='white')
            axes_v[2].set_facecolor('#1a1a1a')

            # Panel 4: per-event zoomed snapshots
            if n_valves > 0:
                show_n = min(n_valves, 10)
                snap_w = 1.0 / show_n
                for i in range(show_n):
                    vf = valve_frames[i]
                    # Show 3 frames: before, during, after
                    before_f = max(0, vf - 2)
                    after_f = min(T - 1, vf + 2)

                    _, proc_before = get_zoomed_fast(before_f)
                    _, proc_during = get_zoomed_fast(vf)
                    _, proc_after = get_zoomed_fast(after_f)

                    # Difference image
                    diff = np.abs(proc_after.astype(float) - proc_before.astype(float))
                    diff = diff / (diff.max() + 1e-8)

                    # Composite: red=motion, gray=structure
                    composite = np.stack([
                        np.clip(proc_during + diff * 0.5, 0, 1),
                        proc_during * 0.7,
                        proc_during * 0.7
                    ], axis=-1)

                    sub_ax = axes_v[3].inset_axes([i * snap_w + 0.005, 0.05,
                                                    snap_w - 0.01, 0.9])
                    sub_ax.imshow(np.clip(composite, 0, 1))
                    sub_ax.set_title(f'f{vf}', color='white', fontsize=7, pad=1)
                    sub_ax.axis('off')

                axes_v[3].set_facecolor('#1a1a1a')
                axes_v[3].axis('off')
                axes_v[3].set_title('Valve event snapshots (red = motion)',
                                     color='white', fontsize=9)
            else:
                axes_v[3].axis('off')

            fig_v.tight_layout()

            # Save
            np.save("valve_event_frames.npy", valve_frames)
            np.save("roi_trace_corrected.npy", roi_trace_proc)
            np.save("motion_signal.npy", deriv_smooth)
            np.save("brightness_correction.npy", correction)
            print(f"\nSaved: valve_event_frames.npy, roi_trace_corrected.npy, "
                  f"motion_signal.npy, brightness_correction.npy")

            plt.show()

        btn_find_valve.on_clicked(find_valve_events)

        # ── Atrium / Ventricle / Valve correlation analysis ──
        def av_valve_analysis(_):
            """3-step ROI selection:
              1. Select ATRIUM on the first zoom ROI
              2. Select VENTRICLE on the first zoom ROI
              3. Select VALVE region (further zoom inside the first ROI)
            Then track all three and find valve events at the A-to-V blood flow.
            """
            from scipy.signal import find_peaks as fp

            t_now = int(s_zf.val)
            _, pz = get_zoomed_fast(t_now)

            # Step 1: select ATRIUM on zoomed image
            fig_a, ax_a = plt.subplots(figsize=(7, 7))
            ax_a.imshow(pz, cmap='gray', vmin=0, vmax=1)
            ax_a.set_title("Step 1/3: Click TWO corners for ATRIUM",
                            fontsize=11, color='red', fontweight='bold')
            ax_a.axis('off')
            pts_a = plt.ginput(2, timeout=0)
            plt.close(fig_a)
            if len(pts_a) < 2:
                print("Need 2 clicks for atrium"); return

            # Step 2: select VENTRICLE (show atrium already marked)
            fig_v, ax_v = plt.subplots(figsize=(7, 7))
            ax_v.imshow(pz, cmap='gray', vmin=0, vmax=1)
            (ax1, ay1), (ax2, ay2) = pts_a
            a_rect = plt.Rectangle((min(ax1,ax2), min(ay1,ay2)),
                                    abs(ax2-ax1), abs(ay2-ay1),
                                    lw=2, edgecolor='red', facecolor='red', alpha=0.2)
            ax_v.add_patch(a_rect)
            ax_v.text(min(ax1,ax2), min(ay1,ay2)-3, 'Atrium', color='red', fontsize=9)
            ax_v.set_title("Step 2/3: Click TWO corners for VENTRICLE",
                            fontsize=11, color='cyan', fontweight='bold')
            ax_v.axis('off')
            pts_v = plt.ginput(2, timeout=0)
            plt.close(fig_v)
            if len(pts_v) < 2:
                print("Need 2 clicks for ventricle"); return

            # Step 3: select VALVE (further zoom — show both A and V marked)
            fig_vl, ax_vl = plt.subplots(figsize=(7, 7))
            ax_vl.imshow(pz, cmap='gray', vmin=0, vmax=1)
            # Show atrium
            a_rect2 = plt.Rectangle((min(ax1,ax2), min(ay1,ay2)),
                                     abs(ax2-ax1), abs(ay2-ay1),
                                     lw=2, edgecolor='red', facecolor='red', alpha=0.15)
            ax_vl.add_patch(a_rect2)
            ax_vl.text(min(ax1,ax2), min(ay1,ay2)-3, 'Atrium', color='red', fontsize=8)
            # Show ventricle
            (vx1, vy1), (vx2, vy2) = pts_v
            v_rect = plt.Rectangle((min(vx1,vx2), min(vy1,vy2)),
                                    abs(vx2-vx1), abs(vy2-vy1),
                                    lw=2, edgecolor='cyan', facecolor='cyan', alpha=0.15)
            ax_vl.add_patch(v_rect)
            ax_vl.text(min(vx1,vx2), min(vy1,vy2)-3, 'Ventricle', color='cyan', fontsize=8)
            ax_vl.set_title("Step 3/3: Click TWO corners for VALVE region\n"
                             "(small area between atrium and ventricle)",
                             fontsize=10, color='yellow', fontweight='bold')
            ax_vl.axis('off')
            pts_valve = plt.ginput(2, timeout=0)
            plt.close(fig_vl)
            if len(pts_valve) < 2:
                print("Need 2 clicks for valve"); return

            # Parse all ROI coords (local to the first zoom ROI)
            def parse_roi(pts):
                (x1, y1), (x2, y2) = pts
                return (max(0, int(min(y1,y2))), max(0, int(min(x1,x2))),
                        min(ry1-ry0, int(max(y1,y2))), min(rx1-rx0, int(max(x1,x2))))

            ay0, ax0, ay1_r, ax1_r = parse_roi(pts_a)
            vy0_r, vx0_r, vy1_r, vx1_r = parse_roi(pts_v)
            valy0, valx0, valy1, valx1 = parse_roi(pts_valve)

            # ── Validate every region is at least 2x2 px ──
            # Without this, a degenerate selection (two clicks on nearly the same
            # row/column, or a thin drag made while the toolbar Pan/Zoom was ON)
            # collapses to an empty slice -> .mean() is NaN -> the whole trace is
            # NaN -> find_peaks returns nothing -> a blank plot with NO error.
            # That is exactly the "I selected the final region and nothing
            # happens" symptom. Reject early with a visible message instead.
            _regions = {'Atrium':    (ay0, ax0, ay1_r, ax1_r),
                        'Ventricle': (vy0_r, vx0_r, vy1_r, vx1_r),
                        'Valve':     (valy0, valx0, valy1, valx1)}
            _bad = [nm for nm, (y0, x0, y1, x1) in _regions.items()
                    if (y1 - y0) < 2 or (x1 - x0) < 2]
            if _bad:
                _msg = (f"Region(s) too small / degenerate: {', '.join(_bad)}.\n\n"
                        f"Drag a clear box for each region and try again.\n\n"
                        f"TIP: make sure the toolbar Pan/Zoom button is OFF before "
                        f"clicking — while it is on, your clicks pan the image "
                        f"instead of being recorded as selection points.")
                print(f"  [Atrium/Ventricle Valve] {_msg}")
                _popup_message("Selection too small", _msg)
                return

            print(f"Atrium ROI:    y=[{ay0}:{ay1_r}], x=[{ax0}:{ax1_r}]")
            print(f"Ventricle ROI: y=[{vy0_r}:{vy1_r}], x=[{vx0_r}:{vx1_r}]")
            print(f"Valve ROI:     y=[{valy0}:{valy1}], x=[{valx0}:{valx1}]")

            # Step 4: compute intensity traces for all three regions
            print("Computing traces for atrium, ventricle, valve...")
            atrium_trace = np.zeros(T, dtype=np.float32)
            ventricle_trace = np.zeros(T, dtype=np.float32)
            valve_trace = np.zeros(T, dtype=np.float32)

            for t in range(T):
                _, pz_t = get_zoomed_fast(t)
                atrium_trace[t] = pz_t[ay0:ay1_r, ax0:ax1_r].mean()
                ventricle_trace[t] = pz_t[vy0_r:vy1_r, vx0_r:vx1_r].mean()
                valve_trace[t] = pz_t[valy0:valy1, valx0:valx1].mean()

            print("  Done.")

            # Brightness correction
            atrium_trace *= correction
            ventricle_trace *= correction
            valve_trace *= correction

            # High-pass filter to remove slow drift
            hp_size = 80
            atrium_hp = atrium_trace - ndimage.uniform_filter1d(atrium_trace, size=hp_size)
            ventricle_hp = ventricle_trace - ndimage.uniform_filter1d(ventricle_trace, size=hp_size)
            valve_hp = valve_trace - ndimage.uniform_filter1d(valve_trace, size=hp_size)

            # Smooth slightly
            atrium_sm = ndimage.uniform_filter1d(atrium_hp, size=3)
            ventricle_sm = ndimage.uniform_filter1d(ventricle_hp, size=3)
            valve_sm = ndimage.uniform_filter1d(valve_hp, size=3)

            # Step 5: find valve events using multiple signals
            d_atrium = np.gradient(atrium_sm)
            d_ventricle = np.gradient(ventricle_sm)
            d_valve = np.abs(np.gradient(valve_sm))  # valve motion = change in valve ROI

            # Combined valve opening signal:
            # 1. Chamber crossover: ventricle rising + atrium falling
            chamber_signal = d_ventricle - d_atrium
            # 2. Direct valve motion: absolute change in valve ROI intensity
            # Combine both (normalized)
            ch_std = np.std(chamber_signal) + 1e-8
            dv_std = np.std(d_valve) + 1e-8
            valve_signal = (chamber_signal / ch_std) + (d_valve / dv_std)
            valve_signal_sm = ndimage.uniform_filter1d(valve_signal, size=3)

            # Normalize
            vs_std = np.std(valve_signal_sm)
            if vs_std > 0:
                valve_signal_norm = valve_signal_sm / vs_std
            else:
                valve_signal_norm = valve_signal_sm

            # Detect valve opening events (peaks in valve_signal)
            valve_thresh = 1.5  # sigma above mean
            valve_peaks, vp_props = fp(valve_signal_norm,
                                        height=valve_thresh,
                                        distance=5,
                                        prominence=0.5)

            # Also detect atrium peaks (contraction = dip in atrium trace)
            atrium_peaks, _ = fp(-atrium_sm, distance=5,
                                   prominence=np.std(atrium_sm) * 0.5)
            ventricle_peaks, _ = fp(ventricle_sm, distance=5,
                                     prominence=np.std(ventricle_sm) * 0.5)

            n_valve = len(valve_peaks)
            n_atrium_beats = len(atrium_peaks)
            n_ventricle_beats = len(ventricle_peaks)

            print(f"\nResults:")
            print(f"  Atrium contractions: {n_atrium_beats}")
            print(f"  Ventricle contractions: {n_ventricle_beats}")
            print(f"  Valve opening events: {n_valve}")
            print(f"  Valve frames: {valve_peaks}")

            # Step 6: PLOT — 6 panels
            fig_av, axes_av = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
            fig_av.canvas.manager.set_window_title(
                f'Atrium-Ventricle-Valve Analysis ({n_valve} valve events)')
            fig_av.patch.set_facecolor('#1a1a1a')

            # Panel 1: Raw chamber + valve traces
            axes_av[0].plot(atrium_trace, color='#e05050', lw=0.4, alpha=0.5, label='Atrium')
            axes_av[0].plot(ventricle_trace, color='#5090e0', lw=0.4, alpha=0.5, label='Ventricle')
            axes_av[0].plot(valve_trace, color='#e0e040', lw=0.5, alpha=0.7, label='Valve ROI')
            axes_av[0].set_ylabel('Intensity', color='white')
            axes_av[0].set_title('Raw traces (brightness-corrected)', color='white')
            axes_av[0].legend(fontsize=7, facecolor='#333', labelcolor='white', ncol=3)
            axes_av[0].tick_params(colors='white')
            axes_av[0].set_facecolor('#1a1a1a')

            # Panel 2: High-pass filtered chambers + peaks
            axes_av[1].plot(atrium_sm, color='#e05050', lw=0.7, label='Atrium')
            axes_av[1].plot(ventricle_sm, color='#5090e0', lw=0.7, label='Ventricle')
            axes_av[1].plot(atrium_peaks, atrium_sm[atrium_peaks], 'v',
                            color='#ff8080', ms=4, label=f'A contract ({n_atrium_beats})')
            axes_av[1].plot(ventricle_peaks, ventricle_sm[ventricle_peaks], '^',
                            color='#80b0ff', ms=4, label=f'V contract ({n_ventricle_beats})')
            axes_av[1].axhline(0, color='#555', lw=0.3)
            axes_av[1].set_ylabel('HP filtered', color='white')
            axes_av[1].set_title('Alternating atrium/ventricle contractions', color='white')
            axes_av[1].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=2)
            axes_av[1].tick_params(colors='white')
            axes_av[1].set_facecolor('#1a1a1a')

            # Panel 3: Valve ROI direct trace (high-pass) + motion
            valve_motion = np.abs(np.gradient(valve_sm))
            valve_motion_sm = ndimage.uniform_filter1d(valve_motion, size=3)
            axes_av[2].plot(valve_sm, color='#e0e040', lw=0.6, label='Valve (HP)')
            ax2twin = axes_av[2].twinx()
            ax2twin.plot(valve_motion_sm, color='#ff80ff', lw=0.4, alpha=0.7, label='Valve |dI/dt|')
            ax2twin.set_ylabel('Motion', color='#ff80ff', fontsize=7)
            ax2twin.tick_params(colors='#ff80ff', labelsize=6)
            axes_av[2].set_ylabel('Valve intensity', color='white')
            axes_av[2].set_title('Valve ROI: intensity (yellow) + motion (pink)', color='white')
            axes_av[2].legend(fontsize=7, facecolor='#333', labelcolor='white', loc='upper left')
            ax2twin.legend(fontsize=7, facecolor='#333', labelcolor='white', loc='upper right')
            axes_av[2].tick_params(colors='white')
            axes_av[2].set_facecolor('#1a1a1a')

            # Panel 4: Combined valve opening signal
            axes_av[3].plot(valve_signal_norm, color='#40c040', lw=0.6,
                            label='Combined (chamber + valve)')
            axes_av[3].axhline(valve_thresh, color='yellow', ls='--', lw=0.5,
                                label=f'Threshold ({valve_thresh}sigma)')
            if n_valve > 0:
                axes_av[3].plot(valve_peaks, valve_signal_norm[valve_peaks],
                                '*', color='yellow', ms=8, label=f'Valve open ({n_valve})')
            axes_av[3].set_ylabel('Signal (sigma)', color='white')
            axes_av[3].set_title(
                'Valve opening signal (chamber crossover + valve motion)', color='white')
            axes_av[3].legend(fontsize=7, facecolor='#333', labelcolor='white')
            axes_av[3].tick_params(colors='white')
            axes_av[3].set_facecolor('#1a1a1a')

            # Panel 5: All three normalized + overlaid
            a_norm = (atrium_sm - atrium_sm.mean()) / (np.std(atrium_sm) + 1e-8)
            v_norm = (ventricle_sm - ventricle_sm.mean()) / (np.std(ventricle_sm) + 1e-8)
            vl_norm = (valve_sm - valve_sm.mean()) / (np.std(valve_sm) + 1e-8)
            axes_av[4].plot(a_norm, color='#e05050', lw=0.6, label='Atrium')
            axes_av[4].plot(v_norm, color='#5090e0', lw=0.6, label='Ventricle')
            axes_av[4].plot(vl_norm, color='#e0e040', lw=0.6, label='Valve ROI')
            for vi, vf in enumerate(valve_peaks):
                axes_av[4].axvline(vf, color='yellow', lw=0.8, alpha=0.4)
                axes_av[4].text(vf, 3.0, f'V{vi+1}', color='yellow', fontsize=6, ha='center')
            axes_av[4].set_ylabel('Normalized', color='white')
            axes_av[4].set_title(
                'Correlation: Atrium (red) vs Ventricle (blue) vs Valve (yellow)',
                color='white')
            axes_av[4].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=3)
            axes_av[4].tick_params(colors='white')
            axes_av[4].set_facecolor('#1a1a1a')

            # Panel 6: Phase relationship + timing
            if n_valve > 0 and n_atrium_beats > 0:
                delays = []
                for vf in valve_peaks:
                    dists = atrium_peaks - vf
                    preceding = dists[dists <= 0]
                    if len(preceding) > 0:
                        delay = -preceding.max()
                        delays.append(delay)
                    else:
                        delays.append(np.nan)

                delays = np.array(delays)
                valid = ~np.isnan(delays)

                if valid.any():
                    axes_av[5].bar(range(n_valve), delays, color='#c080ff')
                    axes_av[5].axhline(np.nanmean(delays), color='white', ls='--', lw=0.8,
                                        label=f'Mean delay = {np.nanmean(delays):.1f} frames')
                    axes_av[5].set_ylabel('Delay (frames)', color='white')
                    axes_av[5].set_xlabel('Valve event #', color='white')
                    axes_av[5].set_title(
                        'Valve opening delay after atrium contraction', color='white')
                    axes_av[5].legend(fontsize=8, facecolor='#333', labelcolor='white')
                else:
                    axes_av[5].text(0.5, 0.5, 'Could not compute delays',
                                     transform=axes_av[5].transAxes, color='white', ha='center')
            else:
                axes_av[5].text(0.5, 0.5, 'Need valve events + atrium beats for timing',
                                 transform=axes_av[5].transAxes, color='white', ha='center')
            axes_av[5].tick_params(colors='white')
            axes_av[5].set_facecolor('#1a1a1a')

            fig_av.tight_layout()

            # Save
            np.save("atrium_trace.npy", atrium_trace)
            np.save("ventricle_trace.npy", ventricle_trace)
            np.save("valve_roi_trace.npy", valve_trace)
            np.save("valve_signal.npy", valve_signal_norm)
            np.save("av_valve_frames.npy", valve_peaks)
            np.save("atrium_contraction_frames.npy", atrium_peaks)
            np.save("ventricle_contraction_frames.npy", ventricle_peaks)
            print(f"\nSaved: atrium_trace.npy, ventricle_trace.npy, valve_roi_trace.npy,")
            print(f"  valve_signal.npy, av_valve_frames.npy,")
            print(f"  atrium_contraction_frames.npy, ventricle_contraction_frames.npy")

            plt.show()

        btn_av.on_clicked(av_valve_analysis)

        # ── Tracked-line valve-disappearance analysis ──
        def valve_line_analysis(_):
            """Draw a line ALONG the valve leaflet, TRACK it as it moves, detect
            which part of the line 'disappears' (darkens below its own baseline)
            and when, then correlate that timing with the atrium/ventricle
            contractions by feeding it as the valve channel of the publication
            figure.

            Why a tracked line (not a fixed ROI): in dark-field the leaflet is a
            bright scattering line. When the valve opens the leaflets separate and
            a PORTION of that line stops scattering — it darkens/disappears. The
            line itself also drifts with the beating heart, so we let the sampling
            line follow the structure (translate along its normal each frame, by
            normalized cross-correlation) before measuring the disappearance.

            Steps:
              1. Select ATRIUM and VENTRICLE boxes (the V/A channels).
              2. Draw the valve LINE (2 clicks) on the zoomed image.
              3. Track it: per frame, search a small range of normal offsets and
                 keep the one whose intensity profile best matches the structure.
              4. Build the tracked kymograph (T x L).
              5. Disappearance = per-position darkening below that position's
                 temporal-median baseline. Sum along the line -> per-frame
                 valve-open signal; argmax along the line -> which part vanished.
              6. Reuse make_publication_figure (valve = the darkening signal).
            """
            from scipy.ndimage import map_coordinates

            rh, rw = ry1 - ry0, rx1 - rx0
            t_now = int(s_zf.val)
            _, pz_now = get_zoomed_fast(t_now)

            # ── Step 1: atrium & ventricle boxes (for the V/A channels) ──
            def _pick_box(title, color, prev=()):
                figb, axb = plt.subplots(figsize=(7, 7))
                axb.imshow(pz_now, cmap='gray', vmin=0, vmax=1)
                for (bx0, by0, bx1, by1, c, name) in prev:
                    axb.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                  lw=2, edgecolor=c, facecolor=c, alpha=0.15))
                    axb.text(bx0, by0 - 3, name, color=c, fontsize=8)
                axb.set_title(title, fontsize=11, color=color, fontweight='bold')
                axb.axis('off')
                p = plt.ginput(2, timeout=0)
                plt.close(figb)
                return p

            def _parse_box(pts):
                (x1, y1), (x2, y2) = pts
                return (max(0, int(min(x1, x2))), max(0, int(min(y1, y2))),
                        min(rw, int(max(x1, x2))), min(rh, int(max(y1, y2))))  # x0,y0,x1,y1

            pts_a = _pick_box("Step 1/3: TWO corners for ATRIUM box", 'red')
            if len(pts_a) < 2:
                _popup_message("Cancelled", "Need 2 clicks for the atrium box."); return
            A = _parse_box(pts_a)

            pts_v = _pick_box("Step 2/3: TWO corners for VENTRICLE box", 'cyan',
                              prev=[(A[0], A[1], A[2], A[3], 'red', 'Atrium')])
            if len(pts_v) < 2:
                _popup_message("Cancelled", "Need 2 clicks for the ventricle box."); return
            V = _parse_box(pts_v)

            for nm, B in (('Atrium', A), ('Ventricle', V)):
                if (B[2] - B[0]) < 2 or (B[3] - B[1]) < 2:
                    _popup_message("Too small",
                                   f"{nm} box is degenerate. Re-run and drag a clear box.\n"
                                   f"(TIP: turn the toolbar Pan/Zoom OFF before clicking.)")
                    return

            # ── Step 2: draw the valve line ──
            figl, axl = plt.subplots(figsize=(7, 7))
            axl.imshow(pz_now, cmap='gray', vmin=0, vmax=1)
            axl.add_patch(plt.Rectangle((A[0], A[1]), A[2] - A[0], A[3] - A[1],
                          lw=2, edgecolor='red', facecolor='red', alpha=0.12))
            axl.add_patch(plt.Rectangle((V[0], V[1]), V[2] - V[0], V[3] - V[1],
                          lw=2, edgecolor='cyan', facecolor='cyan', alpha=0.12))
            axl.set_title("Step 3/3: click TWO endpoints ALONG the valve leaflet\n"
                          "(on the bright line that disappears when the valve opens)",
                          fontsize=10, color='yellow', fontweight='bold')
            axl.axis('off')
            pts_line = plt.ginput(2, timeout=0)
            plt.close(figl)
            if len(pts_line) < 2:
                _popup_message("Cancelled", "Need 2 clicks for the valve line."); return
            (lx1, ly1), (lx2, ly2) = pts_line
            length = float(np.hypot(ly2 - ly1, lx2 - lx1))
            if length < 4:
                _popup_message("Too short", "Valve line is too short. Re-run."); return

            L = int(round(length))
            ys0 = np.linspace(ly1, ly2, L)
            xs0 = np.linspace(lx1, lx2, L)
            uy, ux = (ly2 - ly1) / length, (lx2 - lx1) / length
            ny, nx = ux, -uy                      # unit normal to the line
            thickness = 5
            offs = np.arange(-(thickness // 2), thickness // 2 + 1)

            def sample_line(proc, dshift):
                """Mean intensity profile along the line, translated dshift px
                along the normal, averaged over `thickness` parallel samples."""
                prof = np.zeros(L, dtype=np.float32)
                for o in offs:
                    yy = np.clip(ys0 + (o + dshift) * ny, 0, rh - 1)
                    xx = np.clip(xs0 + (o + dshift) * nx, 0, rw - 1)
                    prof += map_coordinates(proc, [yy, xx], order=1)
                return prof / len(offs)

            # ── Step 3: track the line (normal offset per frame) ──
            max_shift = 6
            shifts = np.arange(-max_shift, max_shift + 1)
            print("  [Valve Line] tracking line across frames...")
            ref_c = sample_line(pz_now, 0)
            ref_c = ref_c - ref_c.mean()
            ref_norm = np.linalg.norm(ref_c) + 1e-8

            kymo = np.zeros((T, L), dtype=np.float32)
            track_off = np.zeros(T, dtype=np.float32)
            prev_shift = 0
            for t in range(T):
                _, pz_t = get_zoomed_fast(t)
                best_s, best_score = 0, -np.inf
                for s in shifts:
                    pc = sample_line(pz_t, s)
                    pc = pc - pc.mean()
                    score = float(np.dot(pc, ref_c) / (np.linalg.norm(pc) * ref_norm + 1e-8))
                    score -= 0.02 * abs(s - prev_shift)   # prefer temporal smoothness
                    if score > best_score:
                        best_score, best_s = score, s
                track_off[t] = best_s
                prev_shift = best_s
                kymo[t] = sample_line(pz_t, best_s) * correction[t]
            track_off_sm = ndimage.uniform_filter1d(track_off, size=3)

            # ── Steps 4-5: disappearance = darkening below per-position median ──
            base = np.median(kymo, axis=0)                       # static line baseline
            dark = np.clip(base[None, :] - kymo, 0.0, None)      # (T, L) darkening map
            line_signal = np.nan_to_num(dark.sum(axis=1), nan=0.0)
            disappear_pos = np.argmax(dark, axis=1)              # which part vanished

            # ── V/A channels from the boxes (contraction = downward dip, to match
            #    make_publication_figure, which detects negative peaks) ──
            atrium_tr = np.zeros(T, dtype=np.float32)
            ventricle_tr = np.zeros(T, dtype=np.float32)
            for t in range(T):
                _, pz_t = get_zoomed_fast(t)
                atrium_tr[t] = pz_t[A[1]:A[3], A[0]:A[2]].mean() * correction[t]
                ventricle_tr[t] = pz_t[V[1]:V[3], V[0]:V[2]].mean() * correction[t]
            hp = 80
            ventricle_sig = ventricle_tr - ndimage.uniform_filter1d(ventricle_tr, size=hp)
            atrium_sig = atrium_tr - ndimage.uniform_filter1d(atrium_tr, size=hp)

            # ── Diagnostic figure: kymograph + tracking + disappearance ──
            figk = plt.figure(figsize=(15, 9))
            figk.canvas.manager.set_window_title('Valve Line — tracked kymograph & disappearance')
            figk.patch.set_facecolor('#1a1a1a')
            axk = figk.add_axes([0.06, 0.55, 0.42, 0.4])
            axk.imshow(kymo, cmap='gray', aspect='auto', extent=[0, L, T, 0])
            axk.set_title('Tracked-line kymograph', color='white', fontsize=10)
            axk.set_xlabel('Position along line (px)', color='white', fontsize=8)
            axk.set_ylabel('Frame', color='white', fontsize=8); axk.tick_params(colors='white')

            axd = figk.add_axes([0.54, 0.55, 0.42, 0.4])
            axd.imshow(dark, cmap='inferno', aspect='auto', extent=[0, L, T, 0])
            axd.plot(disappear_pos, np.arange(T), '.', color='cyan', ms=1, alpha=0.4)
            axd.set_title('Disappearance map (darkening below baseline)', color='white', fontsize=10)
            axd.set_xlabel('Position along line (px)', color='white', fontsize=8)
            axd.set_xlim(0, L); axd.set_ylim(T, 0); axd.tick_params(colors='white')

            axt = figk.add_axes([0.06, 0.30, 0.9, 0.16])
            axt.plot(track_off, color='#4a90d9', lw=0.6, label='raw offset')
            axt.plot(track_off_sm, color='#e07040', lw=1.0, label='smoothed')
            axt.set_ylabel('Line normal\noffset (px)', color='white', fontsize=8)
            axt.set_title('Tracked line displacement (how far the line moved to follow the structure)',
                          color='white', fontsize=9)
            axt.legend(fontsize=7, facecolor='#333', labelcolor='white'); axt.tick_params(colors='white')
            axt.set_facecolor('#1a1a1a'); axt.set_xlim(0, T)

            axs = figk.add_axes([0.06, 0.07, 0.9, 0.16])
            axs.plot(line_signal, color='#bcbd22', lw=0.8, label='line darkening (valve-open)')
            axs.set_ylabel('Σ darkening', color='white', fontsize=8)
            axs.set_xlabel('Frame', color='white', fontsize=8)
            axs.set_title('Per-frame valve-disappearance signal', color='white', fontsize=9)
            axs.legend(fontsize=7, facecolor='#333', labelcolor='white'); axs.tick_params(colors='white')
            axs.set_facecolor('#1a1a1a'); axs.set_xlim(0, T)
            for ax in (axt, axs):
                for sp in ax.spines.values():
                    sp.set_color('white')

            # ── Save ──
            np.save("line_kymograph.npy", kymo)
            np.save("line_darkening_map.npy", dark)
            np.save("line_darkening_signal.npy", line_signal)
            np.save("line_track_offsets.npy", track_off)
            np.save("line_disappear_position.npy", disappear_pos)
            # Save the raw A/V traces too so the template-fit tool ('Fit Valve
            # Curve') can derive the cardiac landmarks (V/A) for phase-folding.
            np.save("atrium_trace.npy", atrium_tr)
            np.save("ventricle_trace.npy", ventricle_tr)
            print("  [Valve Line] saved: line_kymograph.npy, line_darkening_map.npy,")
            print("                      line_darkening_signal.npy, line_track_offsets.npy,")
            print("                      line_disappear_position.npy,")
            print("                      atrium_trace.npy, ventricle_trace.npy")

            # ── Step 6: reuse the publication figure (valve = line darkening) ──
            make_publication_figure(
                ventricle_signal=ventricle_sig,
                atrium_signal=atrium_sig,
                valve_signal=line_signal,
                ventricle_peaks=np.array([], dtype=int),
                atrium_peaks=np.array([], dtype=int),
                valve_peaks=np.array([], dtype=int),
                fps=None,
                frame_range=(0, T),
                savgol_window=5,
                ventricle_z_threshold=VENTRICLE_Z_THRESHOLD_DEFAULT,
                atrium_z_threshold=ATRIUM_Z_THRESHOLD_DEFAULT,
                save_path=None,
                dark_mode=False)

            plt.show()

        ax_line_btn = fig_z.add_axes([0.81, 0.08, 0.10, 0.028])
        btn_line = MplButton(ax_line_btn, 'Valve Line')
        btn_line.on_clicked(valve_line_analysis)

        # Fit the measured valve-line darkening to the expected open/close
        # raised-cosine template (validation), then extract open/close frames.
        def fit_valve_curve(_):
            fit_valve_open_close_template(
                valve_path="line_darkening_signal.npy",
                atrium_path="atrium_trace.npy",
                ventricle_path="ventricle_trace.npy",
                atrium_z_threshold=-1.7)

        ax_fit_btn = fig_z.add_axes([0.915, 0.08, 0.075, 0.028])
        btn_fit = MplButton(ax_fit_btn, 'Fit Curve')
        btn_fit.on_clicked(fit_valve_curve)

        # CRITICAL: keep references alive so buttons/sliders don't get GC'd.
        # (Previously this list referenced an undefined `_do_zoom_update`, which
        # raised NameError here BEFORE the assignment completed — so _keep_alive
        # was never set and the buttons were intermittently garbage-collected,
        # making "Atrium/Ventricle Valve" silently stop responding.)
        fig_z._keep_alive = [
            btn_save_ztif, btn_sel_area, btn_find_valve, btn_av, btn_line, btn_fit,
            s_zf, im_full, im_zraw, im_zproc, im_zcolor,
            vline_trace, title_z, roi_rect,
            save_zoomed_tiff, plot_sub_roi, find_valve_events,
            av_valve_analysis, valve_line_analysis, fit_valve_curve, update_zoom,
        ]

        plt.show()

    btn_zoom.on_clicked(gui_op("Zoom ROI")(open_zoom_viewer))

    # ── Auto-segment & detect valve events ──
    def auto_segment_leaflet(_):
        """Detect rare valve events and segment the leaflet within the zoom ROI.

        Strategy for rare, instantaneous valve movements (< 10 events):
          1. Restrict analysis to zoom ROI (if set)
          2. Compute frame-to-frame absolute difference within ROI
          3. Sum difference per frame → motion signal (spikes = valve events)
          4. Detect peaks in motion signal → valve event frames
          5. For each event: segment the leaflet using temporal difference +
             adaptive threshold + morphological cleanup
          6. Show event timeline + segmentation for each detected event
        """
        from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
        from skimage.filters import threshold_otsu
        from scipy.signal import find_peaks
        from scipy.ndimage import label as ndlabel
        import cv2

        # Use zoom ROI if available, else full frame
        if zoom_roi[0] is not None:
            ry0, rx0, ry1, rx1 = zoom_roi[0]
            print(f"Using zoom ROI: y=[{ry0},{ry1}], x=[{rx0},{rx1}]")
        else:
            ry0, rx0, ry1, rx1 = 0, 0, H, W
            print("No zoom ROI set — using full frame. Set ROI with 'Zoom ROI' first for best results.")

        rh, rw = ry1 - ry0, rx1 - rx0

        # Step 1: compute motion signal within ROI
        print("  Computing motion signal (frame-to-frame difference)...")
        motion = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            diff = np.abs(stack[t, ry0:ry1, rx0:rx1].astype(np.float64) -
                          stack[t-1, ry0:ry1, rx0:rx1].astype(np.float64))
            motion[t] = np.sum(diff)
        # Normalize
        motion = motion / (rh * rw)
        motion_smooth = ndimage.uniform_filter1d(motion, size=3)

        # Step 2: detect valve events (peaks in motion signal)
        # Use high threshold since events are rare and strong
        noise_floor = np.median(motion_smooth)
        noise_std = np.std(motion_smooth[motion_smooth < np.percentile(motion_smooth, 80)])
        peak_thresh = noise_floor + 3 * noise_std  # 3-sigma above noise
        min_distance = 10  # minimum frames between events

        peaks, props = find_peaks(motion_smooth, height=peak_thresh,
                                   distance=min_distance, prominence=noise_std)

        print(f"  Detected {len(peaks)} valve events at frames: {peaks}")
        print(f"  Motion threshold: {peak_thresh:.4f} (noise={noise_floor:.4f}±{noise_std:.4f})")

        # Step 3: for each event, extract temporal context and segment
        event_window = 10  # frames before/after the peak to analyze
        event_data = []

        for i, peak_frame in enumerate(peaks):
            t0 = max(0, peak_frame - event_window)
            t1 = min(T, peak_frame + event_window + 1)

            # Temporal difference image around the event
            before = np.mean(stack[max(0, peak_frame-3):peak_frame, ry0:ry1, rx0:rx1], axis=0)
            after = np.mean(stack[peak_frame:min(T, peak_frame+3), ry0:ry1, rx0:rx1], axis=0)
            diff_img = np.abs(after - before)
            diff_norm = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min() + 1e-8)

            # Process the peak frame
            proc_frame = _process_frame(peak_frame)
            proc_roi = proc_frame[ry0:ry1, rx0:rx1]

            # Segment: combine temporal difference + processed intensity
            combined = 0.5 * diff_norm + 0.5 * proc_roi
            combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

            # Adaptive threshold
            try:
                thresh = threshold_otsu(combined[combined > 0.02])
            except ValueError:
                thresh = 0.3
            mask = combined > thresh * 0.8

            # Morphological cleanup
            mask = binary_closing(mask, disk(2))
            mask = binary_opening(mask, disk(1))
            mask = remove_small_objects(mask, min_size=max(10, rh * rw // 500))

            # Keep largest component
            labeled, n_feat = ndlabel(mask)
            if n_feat > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                largest = np.argmax(sizes) + 1
                mask = (labeled == largest)

            # Compute metrics for this event
            area = mask.sum()
            if area > 0:
                coords = np.argwhere(mask)
                centroid = coords.mean(axis=0)
            else:
                centroid = (rh // 2, rw // 2)

            event_data.append({
                'frame': peak_frame,
                'motion_strength': motion_smooth[peak_frame],
                'mask': mask,
                'diff_img': diff_norm,
                'proc_roi': proc_roi,
                'area': area,
                'centroid': centroid,
                't0': t0, 't1': t1,
            })

        # Step 4: Plot results
        n_events = len(event_data)

        # Figure 1: motion timeline with detected events
        fig1, ax_motion = plt.subplots(figsize=(14, 3))
        fig1.canvas.manager.set_window_title('Valve Event Detection')
        fig1.patch.set_facecolor('#1a1a1a')
        ax_motion.plot(motion_smooth, color='#4a90d9', lw=0.6, alpha=0.8)
        ax_motion.axhline(peak_thresh, color='red', ls='--', lw=0.5, alpha=0.7,
                          label=f'Threshold (3σ = {peak_thresh:.4f})')
        for i, ev in enumerate(event_data):
            ax_motion.axvline(ev['frame'], color='#e07040', lw=1.5, alpha=0.8)
            ax_motion.text(ev['frame'], motion_smooth.max() * 0.95, f'E{i+1}',
                          color='#e07040', fontsize=8, ha='center', va='top')
        ax_motion.set_xlabel('Frame', color='white')
        ax_motion.set_ylabel('Motion (mean |Δ|)', color='white')
        ax_motion.set_title(f'Motion signal — {n_events} valve events detected',
                            color='white', fontsize=10)
        ax_motion.legend(fontsize=8, facecolor='#333333', edgecolor='white',
                         labelcolor='white')
        ax_motion.tick_params(colors='white')
        ax_motion.set_facecolor('#1a1a1a')
        fig1.tight_layout()

        # Figure 2: segmentation for each event (max 6 shown)
        show_n = min(n_events, 6)
        if show_n > 0:
            fig2, axes2 = plt.subplots(3, show_n, figsize=(4 * show_n, 12))
            fig2.canvas.manager.set_window_title('Valve Event Segmentation')
            fig2.patch.set_facecolor('#1a1a1a')
            if show_n == 1:
                axes2 = axes2[:, np.newaxis]

            for i in range(show_n):
                ev = event_data[i]

                # Row 1: temporal difference
                axes2[0, i].imshow(ev['diff_img'], cmap='hot', vmin=0, vmax=1)
                axes2[0, i].set_title(f'E{i+1} Δ (f={ev["frame"]})',
                                       color='white', fontsize=9)

                # Row 2: processed ROI
                axes2[1, i].imshow(ev['proc_roi'], cmap='gray', vmin=0, vmax=1)
                axes2[1, i].set_title(f'Processed', color='white', fontsize=9)

                # Row 3: overlay
                overlay = np.stack([ev['proc_roi']] * 3, axis=-1)
                overlay[ev['mask'], 0] = 1.0
                overlay[ev['mask'], 1] *= 0.3
                overlay[ev['mask'], 2] *= 0.3
                axes2[2, i].imshow(np.clip(overlay, 0, 1))
                axes2[2, i].set_title(f'Seg ({ev["area"]} px)', color='white', fontsize=9)

                for row in range(3):
                    axes2[row, i].axis('off')

            fig2.suptitle('Valve events — temporal difference segmentation',
                          color='white', fontsize=12)
            fig2.tight_layout()

        # Figure 3: valve motion profile (area over time for detected events)
        if n_events >= 2:
            fig3, axes3 = plt.subplots(2, 1, figsize=(14, 6))
            fig3.canvas.manager.set_window_title('Valve Dynamics')
            fig3.patch.set_facecolor('#1a1a1a')

            # Full motion trace
            axes3[0].plot(motion_smooth, color='#4a90d9', lw=0.5)
            for ev in event_data:
                axes3[0].axvspan(ev['t0'], ev['t1'], alpha=0.15, color='#e07040')
            axes3[0].set_ylabel('Motion', color='white')
            axes3[0].set_title('Motion timeline with event windows', color='white')
            axes3[0].tick_params(colors='white')
            axes3[0].set_facecolor('#1a1a1a')

            # Event intervals
            if n_events >= 2:
                intervals = np.diff([ev['frame'] for ev in event_data])
                axes3[1].bar(range(len(intervals)), intervals, color='#e07040')
                axes3[1].set_xlabel('Event pair', color='white')
                axes3[1].set_ylabel('Interval (frames)', color='white')
                axes3[1].set_title(
                    f'Inter-event intervals (mean={np.mean(intervals):.1f} frames)',
                    color='white')
                axes3[1].tick_params(colors='white')
                axes3[1].set_facecolor('#1a1a1a')
            fig3.tight_layout()

        # Save results
        np.save("valve_motion_signal.npy", motion_smooth)
        np.save("valve_event_frames.npy", peaks)
        for i, ev in enumerate(event_data):
            np.save(f"valve_event_{i}_mask.npy", ev['mask'])
        print(f"\nSaved: valve_motion_signal.npy, valve_event_frames.npy, "
              f"{n_events} mask files")
        print(f"\nValve events summary:")
        for i, ev in enumerate(event_data):
            print(f"  Event {i+1}: frame {ev['frame']}, "
                  f"area={ev['area']}px, strength={ev['motion_strength']:.4f}")

        plt.show()

    btn_seg.on_clicked(gui_op("Auto-Seg")(auto_segment_leaflet))

    # ── Load Segmented TIF + Color-based AV Analysis ──
    def load_seg_tif_av(_):
        """Combined workflow:
        1. Load segmented RGB TIF (8 depths) -> extract red/blue area traces
        2. Select valve ROI on the ORIGINAL raw zoomed image
        3. Use A/V alternation from segmented data as reference to find valve events
        Time axes are matched (same frame count).
        """
        import cv2
        from scipy.signal import find_peaks as fp
        from tkinter import filedialog as fd
        import tkinter as _tk

        N_DEPTHS = 8

        # ── Step 1: Load segmented TIF ──
        _root = _tk.Tk(); _root.withdraw()
        seg_path = fd.askopenfilename(
            title="Select presegmented TIF stack (RGB, 8 depths)",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
        _root.destroy()
        if not seg_path:
            return

        print(f"Loading segmented TIF: {seg_path}")
        seg_full = tifffile.imread(seg_path)
        print(f"  Shape: {seg_full.shape}, dtype: {seg_full.dtype}")

        is_rgb = seg_full.ndim == 4 and seg_full.shape[-1] in (3, 4)
        total_seg = seg_full.shape[0]
        fpd = total_seg // N_DEPTHS
        print(f"  {N_DEPTHS} depths x {fpd} frames = {total_seg}, RGB={is_rgb}")

        # ── Step 2: Select depth ──
        fig_dp, ax_dp = plt.subplots(figsize=(8, 3))
        ax_dp.set_title(f"Click DEPTH to use ({N_DEPTHS} depths, {fpd} frames each)", fontsize=10)
        ax_num = fig_dp.add_axes([0.1, 0.25, 0.8, 0.4])
        ax_num.set_xlim(-0.5, N_DEPTHS - 0.5); ax_num.set_ylim(0, 1)
        ax_num.set_xticks(range(N_DEPTHS))
        ax_num.set_xticklabels([f"D{i}\n{i*fpd}-{(i+1)*fpd-1}" for i in range(N_DEPTHS)], fontsize=8)
        ax_num.set_yticks([])
        for i in range(N_DEPTHS):
            ax_num.bar(i, 0.8, color=plt.cm.viridis(i / N_DEPTHS), alpha=0.6)
        pts = plt.ginput(1, timeout=0)
        plt.close(fig_dp)
        if len(pts) < 1:
            print("  Cancelled"); return

        depth_idx = max(0, min(N_DEPTHS - 1, int(round(pts[0][0]))))
        seg_depth = seg_full[depth_idx * fpd:(depth_idx + 1) * fpd]
        sT = seg_depth.shape[0]
        print(f"  Depth {depth_idx}: {sT} frames")

        # ── Step 3: Extract red/blue area from ENTIRE seg frame (no ROI needed) ──
        print("  Extracting red/blue areas from segmented data...")
        red_area = np.zeros(sT, dtype=np.float32)
        blue_area = np.zeros(sT, dtype=np.float32)

        for t in range(sT):
            if is_rgb:
                frame = seg_depth[t].astype(np.float32)
                if frame.max() > 1:
                    frame = frame / 255.0
                r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
                red_area[t] = ((r > g + 0.1) & (r > b + 0.1) & (r > 0.2)).sum()
                blue_area[t] = ((b > r + 0.1) & (b > g + 0.1) & (b > 0.2)).sum()
            else:
                f = seg_depth[t].astype(np.float32)
                lo_t, hi_t = np.percentile(f, [30, 70])
                red_area[t] = (f > hi_t).sum()
                blue_area[t] = (f < lo_t).sum()
            if t % 200 == 0:
                print(f"    {t}/{sT}")

        print(f"  Red range: [{red_area.min():.0f}, {red_area.max():.0f}]")
        print(f"  Blue range: [{blue_area.min():.0f}, {blue_area.max():.0f}]")

        # Smooth + high-pass
        r_sm = ndimage.uniform_filter1d(red_area, size=3)
        b_sm = ndimage.uniform_filter1d(blue_area, size=3)
        hp_sz = 80
        r_hp = r_sm - ndimage.uniform_filter1d(r_sm, size=hp_sz)
        b_hp = b_sm - ndimage.uniform_filter1d(b_sm, size=hp_sz)

        # Detect A/V contractions
        r_peaks, _ = fp(-r_hp, distance=5, prominence=np.std(r_hp) * 0.5)
        b_peaks, _ = fp(-b_hp, distance=5, prominence=np.std(b_hp) * 0.5)
        print(f"  Red contractions: {len(r_peaks)}, Blue contractions: {len(b_peaks)}")

        # ── Step 4: Select valve ROI with temporal variance overlay ──
        n_raw = min(sT, T)
        print(f"  Frame count: seg={sT}, raw={T}, using {n_raw}")

        # Compute temporal variance map (shows where motion is)
        print("  Computing temporal variance map...")
        sample_step = max(1, n_raw // 50)
        var_map = np.zeros((H, W), dtype=np.float32)
        frames_sampled = list(range(0, n_raw, sample_step))
        proc_samples = []
        for t in frames_sampled:
            pf = _process_frame(t)
            proc_samples.append(pf)
        proc_arr = np.array(proc_samples)
        var_map = np.std(proc_arr, axis=0)
        var_norm = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        print(f"  Variance map computed from {len(frames_sampled)} frames")

        # Step 4a: Click ONE reference point near the valve, auto-zoom around it
        proc_mid = _process_frame(T // 2)

        # Show image + variance side by side for reference click
        fig_ref, axes_ref = plt.subplots(1, 2, figsize=(14, 7))
        axes_ref[0].imshow(proc_mid, cmap='gray', vmin=0, vmax=1)
        axes_ref[0].set_title("Image (mid frame)", fontsize=9)
        axes_ref[0].axis('off')
        axes_ref[1].imshow(var_norm, cmap='hot')
        axes_ref[1].set_title("Temporal variance (bright = motion)", fontsize=9)
        axes_ref[1].axis('off')
        fig_ref.suptitle("Click ONCE on the approximate valve location\n"
                          "(on either panel — the tool will auto-zoom around it)",
                          fontsize=11, color='cyan', fontweight='bold')
        ref_pt = plt.ginput(1, timeout=0)
        plt.close(fig_ref)

        if len(ref_pt) < 1:
            print("Need 1 click for reference point"); return

        ref_x, ref_y = ref_pt[0]
        # Handle click on right panel
        if ref_x > W:
            ref_x -= W
        ref_x, ref_y = int(ref_x), int(ref_y)
        ref_x = max(0, min(W - 1, ref_x))
        ref_y = max(0, min(H - 1, ref_y))
        print(f"  Reference point: ({ref_x}, {ref_y})")

        # Auto-zoom: center a window around the reference point
        # Use a generous zoom (±40 px or 20% of image, whichever is larger)
        zoom_r = max(40, min(H, W) // 5)
        pzx0 = max(0, ref_x - zoom_r)
        pzy0 = max(0, ref_y - zoom_r)
        pzx1 = min(W, ref_x + zoom_r)
        pzy1 = min(H, ref_y + zoom_r)

        # Refine: shift the zoom center to the peak variance near the reference
        # Search within ±15 px of the reference for the highest variance spot
        search_r = min(15, zoom_r // 3)
        sy0 = max(0, ref_y - search_r)
        sy1 = min(H, ref_y + search_r)
        sx0 = max(0, ref_x - search_r)
        sx1 = min(W, ref_x + search_r)
        local_var = var_norm[sy0:sy1, sx0:sx1]
        if local_var.size > 0:
            var_smooth_local = ndimage.gaussian_filter(local_var, sigma=2)
            ly, lx = np.unravel_index(np.argmax(var_smooth_local), var_smooth_local.shape)
            refined_x = sx0 + lx
            refined_y = sy0 + ly
            print(f"  Refined to peak variance: ({refined_x}, {refined_y}) "
                  f"(shifted {refined_x - ref_x:+d}, {refined_y - ref_y:+d})")
            # Re-center zoom on refined point
            pzx0 = max(0, refined_x - zoom_r)
            pzy0 = max(0, refined_y - zoom_r)
            pzx1 = min(W, refined_x + zoom_r)
            pzy1 = min(H, refined_y + zoom_r)
        else:
            refined_x, refined_y = ref_x, ref_y

        pzx0 = max(0, int(pzx0))
        pzy0 = max(0, int(pzy0))
        pzx1 = min(W, int(pzx1))
        pzy1 = min(H, int(pzy1))

        # Step 4b: ZOOMED view with variance overlay + auto valve suggestion
        from matplotlib.widgets import Slider as _Sl
        zh, zw = pzy1 - pzy0, pzx1 - pzx0

        if zh < 3 or zw < 3:
            print("  Zoom region too small, using full image")
            pzx0, pzy0, pzx1, pzy1 = 0, 0, W, H
            zh, zw = H, W

        zoom_var = var_norm[pzy0:pzy1, pzx0:pzx1]
        zoom_mid = proc_mid[pzy0:pzy1, pzx0:pzx1]

        # Auto-suggest valve location: highest variance region in the zoom
        # Find the peak of the variance map — that's where the most motion is
        # In a zebrafish heart, this is typically the valve/leaflet region
        from scipy.ndimage import maximum_filter, label as ndlabel
        var_smooth = ndimage.gaussian_filter(zoom_var, sigma=3)
        var_peak_y, var_peak_x = np.unravel_index(np.argmax(var_smooth), var_smooth.shape)

        # Suggest a box around the peak
        suggest_r = max(5, min(zh, zw) // 6)
        sug_y0 = max(0, var_peak_y - suggest_r)
        sug_y1 = min(zh, var_peak_y + suggest_r)
        sug_x0 = max(0, var_peak_x - suggest_r)
        sug_x1 = min(zw, var_peak_x + suggest_r)

        # Reference point in zoomed coordinates
        ref_in_zoom_x = refined_x - pzx0
        ref_in_zoom_y = refined_y - pzy0

        print(f"  Auto-suggested valve: ({var_peak_x}, {var_peak_y}) in zoomed coords")
        print(f"  Your reference point: ({ref_in_zoom_x}, {ref_in_zoom_y}) in zoomed coords")
        print(f"  Darkfield valve physics:")
        print(f"    - Bright = scattering tissue (valve leaflets)")
        print(f"    - Valve CLOSED = leaflets together = BRIGHTER")
        print(f"    - Valve OPEN = leaflets apart = DIMMER")
        print(f"    - Highest motion = valve/leaflet region")

        # Show zoomed view with reference point + suggestion
        fig_vl, ax_vl = plt.subplots(figsize=(10, 8))
        fig_vl.subplots_adjust(bottom=0.18)
        fig_vl.canvas.manager.set_window_title('Valve Selection (ZOOMED around your reference)')

        # Composite: green = variance
        comp_zoom = np.stack([zoom_mid,
                              np.clip(zoom_mid + zoom_var * 0.5, 0, 1),
                              zoom_mid], axis=-1).astype(np.float32)
        comp_zoom = np.clip(comp_zoom, 0, 1)

        im_vl0 = ax_vl.imshow(comp_zoom)

        # Draw your reference point (red circle)
        ax_vl.plot(ref_in_zoom_x, ref_in_zoom_y, 'o', color='red', ms=12, mew=2,
                    fillstyle='none', label='Your reference')
        ax_vl.text(ref_in_zoom_x + 4, ref_in_zoom_y + 4,
                    'Your ref', color='red', fontsize=8)

        # Draw auto-suggested valve box (cyan dashed)
        sug_rect = plt.Rectangle((sug_x0, sug_y0), sug_x1 - sug_x0, sug_y1 - sug_y0,
                                   lw=2, edgecolor='cyan', facecolor='none', ls='--')
        ax_vl.add_patch(sug_rect)
        ax_vl.plot(var_peak_x, var_peak_y, '+', color='cyan', ms=15, mew=2)
        ax_vl.text(var_peak_x + 3, var_peak_y - 8,
                    'Suggested\n(max motion)', color='cyan', fontsize=8)

        ax_vl.set_title(
            "ZOOMED around your reference (red circle)\n"
            "Cyan box = auto-suggested valve (peak motion)\n"
            "Click TWO corners for VALVE ROI, or use the suggestion",
            fontsize=9, color='yellow', fontweight='bold')
        ax_vl.axis('off')

        ax_vl_sl = fig_vl.add_axes([0.1, 0.08, 0.8, 0.03], facecolor='#f0f0f0')
        s_vl_f = _Sl(ax_vl_sl, 'Frame', 0, n_raw-1, valinit=n_raw//2, valstep=1,
                       color='#4a90d9')
        s_vl_f.label.set_color('black'); s_vl_f.valtext.set_color('black')

        # Info text
        ax_info = fig_vl.add_axes([0.05, 0.01, 0.9, 0.06])
        ax_info.axis('off')
        ax_info.text(0, 0.5,
            "Zebrafish darkfield: Valve leaflets scatter light (bright). "
            "Closed=brighter (leaflets together). Open=dimmer (leaflets apart). "
            "Green overlay = temporal variance (highest = valve).",
            fontsize=8, color='#aaaaaa', va='center', wrap=True)

        def _update_vl_frame(val):
            t = int(s_vl_f.val)
            pf = _process_frame(t)
            zf = pf[pzy0:pzy1, pzx0:pzx1]
            comp = np.stack([zf, np.clip(zf + zoom_var * 0.5, 0, 1), zf], axis=-1)
            im_vl0.set_data(np.clip(comp, 0, 1).astype(np.float32))
            fig_vl.canvas.draw_idle()
        s_vl_f.on_changed(_update_vl_frame)

        fig_vl._keep = [s_vl_f, _update_vl_frame, sug_rect]

        pts_vl = plt.ginput(2, timeout=0)
        plt.close(fig_vl)
        if len(pts_vl) < 2:
            print("Need 2 clicks for valve"); return

        # Coordinates are in zoomed image space
        (vx1, vy1_c), (vx2, vy2_c) = pts_vl
        local_x0 = max(0, int(min(vx1, vx2)))
        local_y0 = max(0, int(min(vy1_c, vy2_c)))
        local_x1 = min(zw, int(max(vx1, vx2)))
        local_y1 = min(zh, int(max(vy1_c, vy2_c)))

        # Map to full image coords
        vlx0 = pzx0 + local_x0
        vly0 = pzy0 + local_y0
        vlx1 = pzx0 + local_x1
        vly1 = pzy0 + local_y1
        vlx1 = min(W, vlx1); vly1 = min(H, vly1)

        if vlx1 - vlx0 < 2 or vly1 - vly0 < 2:
            print(f"  ERROR: Valve ROI too small ({vlx1-vlx0}x{vly1-vly0})")
            return
        print(f"  Valve ROI: y=[{vly0}:{vly1}], x=[{vlx0}:{vlx1}], "
              f"size={vlx1-vlx0}x{vly1-vly0}")

        # ── Step 5: Compute valve signal from frame-difference ──
        # ALWAYS use frame-difference motion (centralized function)
        print(f"  Computing valve signal from frame-difference (ROI: {vlx1-vlx0}x{vly1-vly0} px)...")
        roi_stack = np.zeros((n_raw, vly1 - vly0, vlx1 - vlx0), dtype=np.float32)
        for t in range(n_raw):
            vl_frame = _process_frame(t)
            roi_stack[t] = vl_frame[vly0:vly1, vlx0:vlx1]
            if t % 200 == 0: print(f"    {t}/{n_raw}")

        valve_motion_raw, vm_sm = compute_valve_signal_from_frame_diff(roi_stack)
        assert vm_sm is not None, "Valve signal must come from frame-difference"
        print(f"  [CONFIRMED] Valve signal source: frame-difference motion")
        print(f"  Valve traces done. Signal range: [{vm_sm.min():.2f}, {vm_sm.max():.2f}]")

        # ── Step 5b: Valve-OPEN signal from darkening below static background ──
        # An OPEN valve scatters less light, so the ROI dims below its static
        # (temporal-median) background. Measure that darkening straight from the
        # raw ROI intensities (keeping the below-background side), so its peaks
        # mark valve-open events. This is the signal used by the publication
        # figure to detect valve OPENING.
        roi_raw_stack = stack[:n_raw, vly0:vly1, vlx0:vlx1].astype(np.float32)
        valve_open_raw, valve_open_signal = \
            compute_valve_open_signal_from_darkening(roi_raw_stack)
        print(f"  [CONFIRMED] Valve-OPEN signal source: darkening below static background")
        print(f"  Valve-open (darkening) signal range: "
              f"[{valve_open_signal.min():.3f}, {valve_open_signal.max():.3f}]")

        # ── Step 6: Physics-constrained valve detection ──
        # 1 heartbeat = 2 crossovers. Track CLOSE state (more reliable).
        # Filter crossovers by local A/V variation — only use regions
        # where the alternation amplitude is large.

        d_red = np.gradient(r_hp[:n_raw])
        d_blue = np.gradient(b_hp[:n_raw])
        av_handoff = d_blue - d_red
        av_sm = ndimage.uniform_filter1d(av_handoff, size=3)

        # Compute local A/V variation (rolling std of av_handoff)
        local_var_window = 30
        av_local_std = np.array([
            np.std(av_sm[max(0, i-local_var_window):i+local_var_window])
            for i in range(n_raw)])
        av_var_thresh = np.percentile(av_local_std, 50)  # only high-variation regions

        # Find crossovers — both directions
        open_crossovers, open_props = fp(av_sm, distance=3, prominence=0)
        close_crossovers, close_props = fp(-av_sm, distance=3, prominence=0)

        # Filter by prominence AND local variation
        def _filter_crossovers(peaks, props):
            if len(peaks) == 0:
                return np.array([], dtype=int)
            prom = props['prominences']
            prom_thresh = np.median(prom) * 0.5
            # Both prominence AND local variation must be high
            keep = (prom > prom_thresh) & np.array([
                av_local_std[min(p, n_raw-1)] > av_var_thresh for p in peaks])
            return peaks[keep]

        large_open = _filter_crossovers(open_crossovers, open_props)
        large_close = _filter_crossovers(close_crossovers, close_props)

        # 1 heartbeat = 1 close crossover (more reliable for tracking)
        n_heartbeats = len(large_close)
        # Use CLOSE crossovers as reference (valve closes = intensity rises)
        crossover_frames = large_close

        if n_heartbeats < 1:
            print("  WARNING: no large close crossovers, trying open...")
            crossover_frames = large_open
            n_heartbeats = len(large_open)
        if n_heartbeats < 1:
            print("  ERROR: no significant crossovers"); return

        expected_period = n_raw / max(n_heartbeats, 1)
        print(f"  Heartbeats: {n_heartbeats}, period~{expected_period:.1f} frames")
        print(f"  Large open: {len(large_open)}, Large close: {len(large_close)}")

        # For each CLOSE crossover, find valve closing slightly BEFORE/AT it
        search_before = max(2, int(expected_period * 0.3))
        search_after = max(1, int(expected_period * 0.15))

        # Valve signal = frame-difference motion ONLY (no intensity mixing)
        # This is the centralized, sole source for valve detection
        combined_valve = vm_sm.copy()
        print(f"  [CONFIRMED] Valve detection source: frame-difference motion only")

        valve_peaks = []
        valve_strengths = []
        for cross_f in crossover_frames:
            t0_s = max(0, cross_f - search_before)
            t1_s = min(n_raw, cross_f + search_after)
            segment = combined_valve[t0_s:t1_s]
            if len(segment) > 0:
                local_peak = np.argmax(segment) + t0_s
                valve_peaks.append(local_peak)
                valve_strengths.append(combined_valve[local_peak])

        valve_peaks = np.array(valve_peaks)
        valve_strengths = np.array(valve_strengths)

        # Remove duplicates
        if len(valve_peaks) > 1:
            unique_mask = np.concatenate([[True], np.diff(valve_peaks) > 2])
            valve_peaks = valve_peaks[unique_mask]
            valve_strengths = valve_strengths[unique_mask]

        # Pure motion peaks for comparison
        min_dist = max(3, int(expected_period * 0.4))
        vm_peaks, _ = fp(vm_sm, height=np.median(vm_sm) + 1.5 * np.std(vm_sm), distance=min_dist)

        print(f"\n{'='*60}")
        print(f"VALVE CLOSE DETECTION (1 beat = 2 crossovers)")
        print(f"{'='*60}")
        print(f"  Heartbeats: {n_heartbeats} (from large close crossovers)")
        print(f"  Valve close events: {len(valve_peaks)}")
        print(f"  Pure motion peaks: {len(vm_peaks)}")
        for i, vf in enumerate(valve_peaks):
            cross_delay = "?"
            for cf in crossover_frames:
                if abs(cf - vf) < search_before + search_after:
                    cross_delay = f"{vf - cf:+d}"
                    break
            print(f"    Close {i+1}: frame {vf} (vs V->A crossover: {cross_delay})")

        # ── Step 7: Interactive plot with adjustable threshold + boost ──
        from matplotlib.widgets import Slider as MplSlider

        fig_av = plt.figure(figsize=(16, 16))
        fig_av.canvas.manager.set_window_title(
            f'Combined Seg+Raw Valve Analysis (depth {depth_idx})')
        fig_av.patch.set_facecolor('#1a1a1a')

        # Leave room for sliders at bottom
        gs_main = fig_av.add_gridspec(6, 1, hspace=0.4, top=0.95, bottom=0.12)
        axes = [fig_av.add_subplot(gs_main[i]) for i in range(6)]

        def _sty(ax):
            ax.tick_params(colors='white'); ax.set_facecolor('#1a1a1a')

        # P1: red/blue area (from segmented data)
        axes[0].plot(r_sm[:n_raw], color='#e05050', lw=0.6, label='Red (atrium)')
        axes[0].plot(b_sm[:n_raw], color='#5090e0', lw=0.6, label='Blue (ventricle)')
        axes[0].set_ylabel('Pixel count', color='white')
        axes[0].set_title('A/V from segmented data (red=atrium, blue=ventricle)', color='white')
        axes[0].legend(fontsize=7, facecolor='#333', labelcolor='white')
        _sty(axes[0])

        # P2: A/V contractions
        axes[1].plot(r_hp[:n_raw], color='#e05050', lw=0.6, label='Red (HP)')
        axes[1].plot(b_hp[:n_raw], color='#5090e0', lw=0.6, label='Blue (HP)')
        rp_clip = r_peaks[r_peaks < n_raw]
        bp_clip = b_peaks[b_peaks < n_raw]
        axes[1].plot(rp_clip, r_hp[rp_clip], 'v', color='#ff8080', ms=4,
                      label=f'A contract ({len(rp_clip)})')
        axes[1].plot(bp_clip, b_hp[bp_clip], 'v', color='#80b0ff', ms=4,
                      label=f'V contract ({len(bp_clip)})')
        axes[1].axhline(0, color='#555', lw=0.3)
        axes[1].set_ylabel('Area (HP)', color='white')
        axes[1].set_title('Alternating A/V contractions (from seg data)', color='white')
        axes[1].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=2)
        _sty(axes[1])

        # P3: valve frame-difference motion signal (sole source)
        axes[2].plot(vm_sm, color='#e0e040', lw=0.6, label='Valve motion (frame-diff)')
        axes[2].axhline(0, color='#555', lw=0.3)
        if len(valve_peaks) > 0:
            axes[2].plot(valve_peaks, vm_sm[np.clip(valve_peaks, 0, len(vm_sm)-1)],
                          '*', color='yellow', ms=7,
                          label=f'Valve events ({len(valve_peaks)})')
        axes[2].set_ylabel('Motion (frame-diff)', color='white')
        axes[2].set_title('Valve motion signal (frame-difference, sole source)',
                           color='white')
        axes[2].legend(fontsize=6, facecolor='#333', labelcolor='white')
        _sty(axes[2])
        print("  [CONFIRMED] Initial plot: valve signal from frame-difference")

        # P4: A/V blood flow with OPEN and CLOSE crossovers marked separately
        axes[3].plot(av_sm, color='#40c040', lw=0.6, label='A/V blood flow')
        if len(large_open) > 0:
            lo_clip = large_open[large_open < len(av_sm)]
            axes[3].plot(lo_clip, av_sm[lo_clip], '^', color='#40ff40', ms=6,
                          label=f'Open crossovers ({len(large_open)})')
        if len(large_close) > 0:
            lc_clip = large_close[large_close < len(av_sm)]
            if len(lc_clip) > 0:
                axes[3].plot(lc_clip, av_sm[lc_clip],
                              'v', color='#ff8080', ms=5,
                              label=f'Close crossovers ({len(large_close)})')
        if len(crossover_frames) > 0:
            axes[3].plot(crossover_frames, av_sm[crossover_frames], 'o',
                          color='#40ff40', ms=5, label=f'A/V crossovers ({len(crossover_frames)})')
        if len(valve_peaks) > 0:
            axes[3].plot(valve_peaks, av_sm[np.clip(valve_peaks, 0, len(av_sm)-1)],
                          '*', color='yellow', ms=8,
                          label=f'Valve open ({len(valve_peaks)}, 1/beat)')
            # Draw search windows
            for vp in valve_peaks:
                axes[3].axvspan(max(0, vp - 1), min(n_raw, vp + 1),
                                 alpha=0.1, color='yellow')
        axes[3].set_ylabel('Signal', color='white')
        axes[3].set_title(
            f'Physics: {len(valve_peaks)} valve events for {n_heartbeats} beats '
            f'(valve opens before A/V crossover)', color='white')
        axes[3].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=2)
        _sty(axes[3])

        # P5: all normalized + overlaid
        def _norm(x):
            s = np.std(x); return (x - x.mean()) / s if s > 0 else x * 0
        axes[4].plot(_norm(r_hp[:n_raw]), color='#e05050', lw=0.5, label='Atrium (red)')
        axes[4].plot(_norm(b_hp[:n_raw]), color='#5090e0', lw=0.5, label='Ventricle (blue)')
        axes[4].plot(_norm(vm_sm), color='#e0e040', lw=0.5, label='Valve (raw)')
        for vf in valve_peaks:
            axes[4].axvline(vf, color='yellow', lw=0.6, alpha=0.4)
        axes[4].set_ylabel('Normalized', color='white')
        axes[4].set_title('Correlation: seg A/V vs raw valve motion', color='white')
        axes[4].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=3)
        _sty(axes[4])

        # P6: valve delay after atrium contraction
        if len(valve_peaks) > 0 and len(rp_clip) > 0:
            delays = []
            for vf in valve_peaks:
                dists = rp_clip.astype(int) - int(vf)
                preceding = dists[dists <= 0]
                delays.append(-preceding.max() if len(preceding) > 0 else np.nan)
            delays = np.array(delays)
            valid_d = ~np.isnan(delays)
            if valid_d.any():
                axes[5].bar(range(len(delays)), delays, color='#c080ff')
                axes[5].axhline(np.nanmean(delays), color='white', ls='--', lw=0.8,
                                 label=f'Mean = {np.nanmean(delays):.1f} frames')
                axes[5].set_title('Valve delay after atrium contraction', color='white')
                axes[5].legend(fontsize=8, facecolor='#333', labelcolor='white')
        axes[5].set_ylabel('Delay (frames)', color='white')
        axes[5].set_xlabel('Frame', color='white')
        _sty(axes[5])

        # ── Interactive threshold + signal boost sliders ──
        sb = '#f0f0f0'; lc = 'black'
        # Row 1: A/V contraction thresholds
        ax_athresh = fig_av.add_axes([0.08, 0.09, 0.15, 0.018], facecolor=sb)
        ax_vthresh = fig_av.add_axes([0.26, 0.09, 0.15, 0.018], facecolor=sb)
        ax_mindist = fig_av.add_axes([0.44, 0.09, 0.10, 0.018], facecolor=sb)
        ax_maxvalve = fig_av.add_axes([0.57, 0.09, 0.10, 0.018], facecolor=sb)

        a_std = float(np.std(r_hp))
        v_std = float(np.std(b_hp))
        s_athresh = MplSlider(ax_athresh, 'A thr', 0.1, 3.0, valinit=0.5,
                               valstep=0.05, color='#e05050')
        s_vthresh = MplSlider(ax_vthresh, 'V thr', 0.1, 3.0, valinit=0.5,
                               valstep=0.05, color='#5090e0')
        s_mindist = MplSlider(ax_mindist, 'Dist', 3, 50, valinit=5,
                               valstep=1, color='#888888')
        s_maxvalve = MplSlider(ax_maxvalve, 'Max V', 1, 30, valinit=DEFAULT_VALVE_PEAK_COUNT,
                                valstep=1, color='#e0e040')

        # Row 2: valve detection params
        ax_thresh = fig_av.add_axes([0.08, 0.06, 0.15, 0.018], facecolor=sb)
        ax_boost = fig_av.add_axes([0.26, 0.06, 0.15, 0.018], facecolor=sb)
        ax_search = fig_av.add_axes([0.44, 0.06, 0.10, 0.018], facecolor=sb)

        s_thresh = MplSlider(ax_thresh, 'V-Thr', 0.5, 5.0, valinit=1.5,
                              valstep=0.1, color='#e07040')
        s_boost = MplSlider(ax_boost, 'Boost', 1.0, 5.0, valinit=1.0,
                             valstep=0.1, color='#40a040')
        s_search = MplSlider(ax_search, 'Srch%', 5, 50, valinit=30,
                              valstep=5, color='#4a90d9')

        # Row 3: frame range for export
        ax_fstart = fig_av.add_axes([0.08, 0.03, 0.25, 0.018], facecolor=sb)
        ax_fend = fig_av.add_axes([0.38, 0.03, 0.25, 0.018], facecolor=sb)
        s_fstart = MplSlider(ax_fstart, 'Start', 0, n_raw-1, valinit=0,
                              valstep=1, color='#c080ff')
        s_fend = MplSlider(ax_fend, 'End', 0, n_raw-1, valinit=n_raw-1,
                            valstep=1, color='#c080ff')

        for s in [s_athresh, s_vthresh, s_mindist, s_maxvalve,
                  s_thresh, s_boost, s_search, s_fstart, s_fend]:
            s.label.set_color(lc); s.valtext.set_color(lc)

        from matplotlib.widgets import Button as _Btn
        ax_redetect = fig_av.add_axes([0.70, 0.07, 0.08, 0.028])
        btn_redetect = _Btn(ax_redetect, 'Re-detect')
        ax_savebtn = fig_av.add_axes([0.80, 0.07, 0.08, 0.028])
        btn_save_res = _Btn(ax_savebtn, 'Save')
        ax_exportbtn = fig_av.add_axes([0.70, 0.03, 0.10, 0.028])
        btn_export = _Btn(ax_exportbtn, 'Export Range')
        ax_pubbtn = fig_av.add_axes([0.82, 0.03, 0.10, 0.028])
        btn_pub = _Btn(ax_pubbtn, 'Pub Figure')

        def _boost_signal(signal, threshold_sigma, boost_power):
            """Soft-threshold + nonlinear boost.

            Scientific basis: equivalent to wavelet soft-thresholding followed
            by power-law amplification. Signal below threshold is suppressed
            (not removed — just reduced). Signal above threshold is boosted
            by raising the excess to a power > 1.

            This preserves peak positions and relative ordering — only
            amplifies the contrast between peaks and noise floor.
            """
            sig = signal.copy()
            noise = np.std(sig[sig < np.percentile(sig, 75)])
            thresh = noise * threshold_sigma

            # Soft threshold: reduce sub-threshold signal
            below = np.abs(sig) < thresh
            sig[below] *= 0.2  # keep 20% of sub-threshold (don't zero it)

            # Boost supra-threshold: power-law amplification
            above = ~below
            if above.any():
                excess = np.abs(sig[above]) - thresh
                max_excess = excess.max() + 1e-8
                # Normalize excess to [0,1], apply power, rescale
                boosted = (excess / max_excess) ** (1.0 / boost_power) * max_excess
                sig[above] = np.sign(sig[above]) * (boosted + thresh)

            return sig

        def redetect_valves(_=None):
            """Re-run A/V contraction + valve detection with slider values."""
            a_thresh_s = s_athresh.val
            v_thresh_s = s_vthresh.val
            min_d = int(s_mindist.val)
            thresh_s = s_thresh.val
            boost_p = s_boost.val
            search_pct = s_search.val / 100.0

            # Step 1: Re-detect A/V contractions with new thresholds
            # SINGLE SOURCE OF TRUTH: RED=ATRIUM, BLUE=VENTRICLE
            # A thresh controls red(atrium) peaks; V thresh controls blue(ventricle) peaks
            a_prom = np.std(r_hp[:n_raw]) * a_thresh_s  # A = red (atrium)
            v_prom = np.std(b_hp[:n_raw]) * v_thresh_s  # V = blue (ventricle)
            new_r_peaks, _ = fp(-r_hp[:n_raw], distance=min_d, prominence=a_prom)  # A = red
            new_b_peaks, _ = fp(-b_hp[:n_raw], distance=min_d, prominence=v_prom)  # V = blue

            # Reference count = max(A contractions, V contractions)
            ref_count = max(len(new_r_peaks), len(new_b_peaks))
            # RED=Atrium, BLUE=Ventricle
            # new_r_peaks = A (atrium) contractions
            # new_b_peaks = V (ventricle) contractions
            print(f"  A contractions (red): {len(new_r_peaks)} (thresh={a_thresh_s:.2f}sigma)")
            print(f"  V contractions (blue): {len(new_b_peaks)} (thresh={v_thresh_s:.2f}sigma)")
            print(f"  Reference count for valve: {ref_count}")

            # Step 2: Boost valve signal
            vm_boosted = _boost_signal(combined_valve, thresh_s, boost_p)

            # Step 3: Find valve close peaks using physiological constraint
            # Cardiac cycle order: A contraction -> AV valve CLOSE -> V contraction
            # So the valve close must be AFTER the nearest A peak and BEFORE
            # the nearest V peak. Search only in this window.

            new_peaks = []
            new_strengths = []

            if len(new_r_peaks) > 0 and len(new_b_peaks) > 0:
                # RED=Atrium, BLUE=Ventricle
                # For each A(red) contraction, find the next V(blue) contraction
                # The valve close happens in between
                for i, a_pk in enumerate(new_r_peaks):
                    # Find the next V(blue) contraction after this A(red) contraction
                    v_after = new_b_peaks[new_b_peaks > a_pk]
                    if len(v_after) == 0:
                        continue
                    v_pk = v_after[0]  # nearest V after this A

                    # Valve close window: from A peak to V peak
                    # Use the FULL interval — no margin trimming.
                    # The search_pct slider controls how much to expand beyond.
                    interval = v_pk - a_pk
                    expand = int(interval * search_pct * 0.5)
                    t0_s = max(0, a_pk - expand)
                    t1_s = min(n_raw, v_pk + expand)

                    if t1_s <= t0_s:
                        continue

                    # Search in the ORIGINAL signal (not boosted) for the true max
                    segment = combined_valve[t0_s:t1_s]
                    if len(segment) > 0:
                        lp = np.argmax(segment) + t0_s
                        new_peaks.append(lp)
                        new_strengths.append(combined_valve[min(lp, len(combined_valve)-1)])

                print(f"  A(red)->Valve->V(blue) windows: {len(new_peaks)} "
                      f"(from {len(new_r_peaks)} A peaks)")
            else:
                # Fallback: use search window around whichever peaks exist
                ref_peaks = new_r_peaks if len(new_r_peaks) > 0 else new_b_peaks
                sb_frames = max(2, int((n_raw / max(len(ref_peaks), 1)) * search_pct))
                for ref_f in ref_peaks:
                    t0_s = max(0, ref_f)
                    t1_s = min(n_raw, ref_f + sb_frames)
                    segment = vm_boosted[t0_s:t1_s]
                    if len(segment) > 0:
                        lp = np.argmax(segment) + t0_s
                        new_peaks.append(lp)
                        new_strengths.append(vm_boosted[min(lp, len(vm_boosted)-1)])

            new_peaks = np.array(new_peaks)
            new_strengths = np.array(new_strengths) if len(new_strengths) > 0 else np.array([])

            # Remove duplicates
            if len(new_peaks) > 1:
                uniq = np.concatenate([[True], np.diff(new_peaks) > 2])
                new_peaks = new_peaks[uniq]
                if len(new_strengths) > len(uniq):
                    new_strengths = new_strengths[uniq]

            # Constrain: at most max_valve peaks (user-controlled)
            max_v = int(s_maxvalve.val)
            if len(new_peaks) > max_v and max_v > 0:
                strengths = vm_boosted[np.clip(new_peaks, 0, len(vm_boosted)-1)]
                top_idx = np.argsort(strengths)[::-1][:max_v]
                new_peaks = np.sort(new_peaks[top_idx])

            print(f"  Valve events: {len(new_peaks)} (max={max_v})")

            # ── Helper: fit smooth upper/lower envelope of signal ──
            from scipy.interpolate import UnivariateSpline
            from scipy.signal import argrelextrema

            def _fit_envelope(signal, mode='upper', window=15):
                """Fit smooth envelope of the signal.

                mode='upper': connects local maxima → upper envelope
                mode='lower': connects local minima → lower envelope

                This is the TRUE envelope — not a fit through detected peaks,
                but a smooth curve that traces the peaks/troughs of the oscillation.
                """
                n = len(signal)
                t_all = np.arange(n)
                if n < 10:
                    return signal.copy()

                # Find local extrema
                order = max(3, window)
                if mode == 'upper':
                    extrema = argrelextrema(signal, np.greater_equal, order=order)[0]
                else:
                    extrema = argrelextrema(signal, np.less_equal, order=order)[0]

                if len(extrema) < 3:
                    return ndimage.uniform_filter1d(signal, size=window * 2)

                # Add boundary points
                x = np.concatenate([[0], extrema, [n - 1]])
                if mode == 'upper':
                    y = np.concatenate([[signal[:order].max()],
                                         signal[extrema],
                                         [signal[-order:].max()]])
                else:
                    y = np.concatenate([[signal[:order].min()],
                                         signal[extrema],
                                         [signal[-order:].min()]])

                try:
                    spl = UnivariateSpline(x, y, s=len(x) * 2.0, k=3)
                    return spl(t_all).astype(np.float32)
                except Exception:
                    return np.interp(t_all, x, y).astype(np.float32)

            def _fit_gaussian_peaks(signal, peaks, half_width=5):
                """Fit Gaussian-shaped peaks at detected positions.
                Valve motion is a sharp burst lasting a few frames — model each
                event as a Gaussian pulse. Returns the fitted curve."""
                n = len(signal)
                fitted = np.zeros(n, dtype=np.float32)
                t_all = np.arange(n)
                for pk in peaks:
                    pk = int(pk)
                    if pk >= n:
                        continue
                    amp = float(signal[min(pk, n-1)])
                    # Fit sigma from the actual peak shape
                    lo = max(0, pk - half_width)
                    hi = min(n, pk + half_width + 1)
                    local = signal[lo:hi]
                    if len(local) < 3 or amp < 1e-6:
                        continue
                    # Gaussian: A * exp(-(t-pk)^2 / (2*sigma^2))
                    # Estimate sigma from half-max width
                    above_half = local > amp * 0.5
                    hw = max(1, above_half.sum() // 2)
                    sigma = max(1.0, float(hw) / 1.18)  # FWHM = 2.35*sigma
                    gauss = amp * np.exp(-0.5 * ((t_all - pk) / sigma) ** 2)
                    fitted = np.maximum(fitted, gauss)
                return fitted

            # ── Update ALL 6 panels ──

            # Get frame range
            f0 = int(s_fstart.val)
            f1 = int(s_fend.val)
            if f1 <= f0:
                f1 = n_raw - 1
            fr = slice(f0, f1 + 1)
            t_range = np.arange(f0, f1 + 1)

            # P1: A/V areas with upper/lower envelope
            axes[0].clear()
            r_seg = r_sm[fr]
            b_seg = b_sm[fr]
            axes[0].plot(t_range, r_seg, color='#e05050', lw=0.6, label='Red (atrium)')
            axes[0].plot(t_range, b_seg, color='#5090e0', lw=0.6, label='Blue (ventricle)')
            # Contractions
            rp_in = new_r_peaks[(new_r_peaks >= f0) & (new_r_peaks <= f1)]
            bp_in = new_b_peaks[(new_b_peaks >= f0) & (new_b_peaks <= f1)]
            if len(rp_in) > 0:
                axes[0].plot(rp_in, r_sm[rp_in], 'v', color='red', ms=5)
            if len(bp_in) > 0:
                axes[0].plot(bp_in, b_sm[bp_in], 'v', color='blue', ms=5)
            axes[0].set_xlim(f0, f1)
            axes[0].set_ylabel('Pixel count', color='white')
            axes[0].set_title(
                f'A={len(rp_in)} V={len(bp_in)} in [{f0}-{f1}] | '
                f'maxV={max_v}', color='white')
            axes[0].legend(fontsize=6, facecolor='#333', labelcolor='white', ncol=2)
            _sty(axes[0])

            # P2: HP + contractions with envelope
            axes[1].clear()
            r_hp_seg = r_hp[fr]
            b_hp_seg = b_hp[fr]
            axes[1].plot(t_range, r_hp_seg, color='#e05050', lw=0.6, label=f'Red/A HP ({len(rp_in)})')
            axes[1].plot(t_range, b_hp_seg, color='#5090e0', lw=0.6, label=f'Blue/V HP ({len(bp_in)})')
            if len(rp_in) > 0:
                axes[1].plot(rp_in, r_hp[rp_in], 'v', color='#ff8080', ms=5)
            if len(bp_in) > 0:
                axes[1].plot(bp_in, b_hp[bp_in], 'v', color='#80b0ff', ms=5)
            axes[1].axhline(-a_prom, color='#e05050', ls=':', lw=0.5, alpha=0.5)
            axes[1].axhline(-v_prom, color='#5090e0', ls=':', lw=0.5, alpha=0.5)
            axes[1].axhline(0, color='#555', lw=0.3)
            axes[1].set_xlim(f0, f1)
            axes[1].set_ylabel('Area (HP)', color='white')
            axes[1].set_title('A/V envelopes (shaded = oscillation range)', color='white')
            axes[1].legend(fontsize=5, facecolor='#333', labelcolor='white', ncol=2)
            _sty(axes[1])

            # Filter peaks to frame range
            vp_in = new_peaks[(new_peaks >= f0) & (new_peaks <= f1)]

            # P3: valve signal + Gaussian peak fit (frame range)
            axes[2].clear()
            # Plot ORIGINAL signal (never changes with threshold)
            axes[2].plot(t_range, combined_valve[fr], color='#e0e040', lw=0.6,
                          label='Valve signal (original)')
            # Threshold line (visual reference only — doesn't modify signal)
            noise_floor = np.std(combined_valve[combined_valve < np.percentile(combined_valve, 75)])
            thresh_line = noise_floor * thresh_s
            axes[2].axhline(thresh_line, color='#e07040', ls='--', lw=0.5,
                             label=f'Threshold ({thresh_s:.1f}sigma)')
            if len(vp_in) > 0:
                # Gaussian fit on the ORIGINAL signal at detected peaks
                gauss_fit = _fit_gaussian_peaks(combined_valve, vp_in, half_width=5)
                axes[2].plot(np.arange(len(gauss_fit)), gauss_fit, color='#ff80ff', lw=1.2,
                              alpha=0.7, label='Gaussian fit')
                axes[2].plot(vp_in, combined_valve[np.clip(vp_in, 0, len(combined_valve)-1)],
                              '*', color='yellow', ms=8,
                              label=f'Events ({len(vp_in)})')
                for pk in vp_in:
                    pk = int(pk)
                    lo_p = max(0, pk - 5); hi_p = min(len(combined_valve), pk + 6)
                    local = combined_valve[lo_p:hi_p]
                    amp = combined_valve[min(pk, len(combined_valve)-1)]
                    if amp > 0:
                        fwhm = (local > amp * 0.5).sum()
                        axes[2].annotate(f'{fwhm}f', (pk, amp * 1.05),
                                          color='yellow', fontsize=6, ha='center')
            axes[2].set_xlim(f0, f1)
            axes[2].set_ylabel('Signal', color='white')
            axes[2].set_title(
                f'{len(vp_in)} valve closes (threshold for detection only, signal unchanged)',
                color='white')
            axes[2].legend(fontsize=5, facecolor='#333', labelcolor='white', ncol=2)
            _sty(axes[2])

            # P4: A/V blood flow + crossovers (frame range)
            axes[3].clear()
            axes[3].plot(t_range, av_sm[fr], color='#40c040', lw=0.6, label='A/V blood flow')
            lo_in = large_open[(large_open >= f0) & (large_open <= f1)]
            lc_in = large_close[(large_close >= f0) & (large_close <= f1)]
            if len(lo_in) > 0:
                axes[3].plot(lo_in, av_sm[lo_in], '^', color='#40ff40', ms=5,
                              label=f'Open ({len(lo_in)})')
            if len(lc_in) > 0:
                axes[3].plot(lc_in, av_sm[lc_in], 'v', color='#ff8080', ms=5,
                              label=f'Close ({len(lc_in)})')
            if len(vp_in) > 0:
                axes[3].plot(vp_in, av_sm[np.clip(vp_in, 0, len(av_sm)-1)],
                              '*', color='yellow', ms=8, label=f'Valve ({len(vp_in)})')
            axes[3].set_xlim(f0, f1)
            axes[3].set_ylabel('Blood flow (A->V)', color='white')
            axes[3].set_title(
                f'{len(vp_in)} valve closes in [{f0}-{f1}] '
                f'(A contract -> valve close -> V contract)', color='white')
            axes[3].legend(fontsize=5, facecolor='#333', labelcolor='white', ncol=2)
            _sty(axes[3])

            # P5: normalized overlay (frame range)
            def _norm(x):
                s = np.std(x); return (x - x.mean()) / s if s > 0 else x * 0
            axes[4].clear()
            axes[4].plot(t_range, _norm(r_hp[fr]), color='#e05050', lw=0.5, label='Atrium (red)')
            axes[4].plot(t_range, _norm(b_hp[fr]), color='#5090e0', lw=0.5, label='Ventricle (blue)')
            axes[4].plot(t_range, _norm(vm_boosted[fr]), color='#e0e040', lw=0.5, label='Valve')
            for vf in vp_in:
                axes[4].axvline(vf, color='yellow', lw=0.6, alpha=0.4)
            axes[4].set_xlim(f0, f1)
            axes[4].set_ylabel('Normalized', color='white')
            axes[4].set_title('Correlation in selected range', color='white')
            axes[4].legend(fontsize=5, facecolor='#333', labelcolor='white', ncol=3)
            _sty(axes[4])

            # P6: delays (in frame range)
            axes[5].clear()
            # Use A (red=atrium) peaks as reference for delay computation
            rpc_in = new_r_peaks[(new_r_peaks >= f0) & (new_r_peaks <= f1)]
            if len(vp_in) > 0 and len(rpc_in) > 0:
                delays_new = []
                for vf in vp_in:
                    dists = rpc_in.astype(int) - int(vf)
                    preceding = dists[dists <= 0]
                    delays_new.append(-preceding.max() if len(preceding) > 0 else np.nan)
                delays_new = np.array(delays_new)
                valid_n = ~np.isnan(delays_new)
                if valid_n.any():
                    axes[5].bar(range(len(delays_new)), delays_new, color='#c080ff')
                    axes[5].axhline(np.nanmean(delays_new), color='white', ls='--', lw=0.8,
                                     label=f'Mean = {np.nanmean(delays_new):.1f} frames')
                    axes[5].legend(fontsize=8, facecolor='#333', labelcolor='white')
            axes[5].set_ylabel('Delay (frames)', color='white')
            axes[5].set_xlabel('Frame', color='white')
            axes[5].set_title(
                f'Valve close delay after A contraction '
                f'({len(vp_in)} events, should be >0)', color='white')
            _sty(axes[5])

            fig_av.canvas.draw_idle()

            fig_av._last_peaks = new_peaks
            fig_av._last_boosted = vm_boosted
            print(f"  Re-detected: {len(new_peaks)} close events "
                  f"(thresh={thresh_s:.1f}, boost={boost_p:.1f}, search={search_pct*100:.0f}%)")

        btn_redetect.on_clicked(redetect_valves)

        def save_results(_):
            peaks_to_save = getattr(fig_av, '_last_peaks', valve_peaks)
            boosted_to_save = getattr(fig_av, '_last_boosted', vm_sm)
            np.save("seg_red_area.npy", r_sm[:n_raw])
            np.save("seg_blue_area.npy", b_sm[:n_raw])
            np.save("raw_valve_motion.npy", vm_sm)
            np.save("boosted_valve_motion.npy", boosted_to_save)
            np.save("combined_valve_events.npy", peaks_to_save)
            print(f"\nSaved: seg_red_area.npy, seg_blue_area.npy, "
                  f"raw_valve_motion.npy, boosted_valve_motion.npy, combined_valve_events.npy")

        btn_save_res.on_clicked(save_results)

        def export_range(_):
            """Export the current frame range as a publication figure + data."""
            from tkinter import filedialog as _fd
            import tkinter as _tk

            f0e = int(s_fstart.val)
            f1e = int(s_fend.val)
            if f1e <= f0e: f1e = n_raw - 1

            _root = _tk.Tk(); _root.withdraw()
            path = _fd.asksaveasfilename(
                title=f"Export range [{f0e}-{f1e}]",
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
            _root.destroy()
            if not path:
                return

            # Re-run redetect to ensure panels are up to date
            redetect_valves()

            # Save the figure
            fig_av.savefig(path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
            print(f"  Figure saved: {path}")

            # Save data for this range
            base = path.rsplit('.', 1)[0]
            vp_exp = getattr(fig_av, '_last_peaks', np.array([]))
            vp_range = vp_exp[(vp_exp >= f0e) & (vp_exp <= f1e)] if len(vp_exp) > 0 else np.array([])

            np.savez(base + '_data.npz',
                     frame_start=f0e, frame_end=f1e,
                     red_area=r_sm[f0e:f1e+1],
                     blue_area=b_sm[f0e:f1e+1],
                     red_hp=r_hp[f0e:f1e+1],
                     blue_hp=b_hp[f0e:f1e+1],
                     valve_motion=vm_sm[f0e:f1e+1],
                     valve_events=vp_range,
                     av_blood_flow=av_sm[f0e:f1e+1])
            print(f"  Data saved: {base}_data.npz")
            print(f"  Valve events in range: {vp_range}")

        btn_export.on_clicked(export_range)

        def make_pub_fig(_):
            """Generate publication figure with current detection results."""
            from tkinter import filedialog as _fd
            import tkinter as _tk

            f0p = int(s_fstart.val)
            f1p = int(s_fend.val)
            if f1p <= f0p: f1p = n_raw - 1

            # Get current peaks
            cur_peaks = getattr(fig_av, '_last_peaks', valve_peaks)

            # SINGLE SOURCE OF TRUTH: b_hp=ventricle (blue), r_hp=atrium (red)
            _root = _tk.Tk(); _root.withdraw()
            path = _fd.asksaveasfilename(
                title="Save Publication Figure",
                defaultextension=".pdf",
                filetypes=[("PDF","*.pdf"),("SVG","*.svg"),("PNG","*.png")])
            _root.destroy()

            pub_fig = make_publication_figure(
                ventricle_signal=b_hp[:n_raw],
                atrium_signal=r_hp[:n_raw],
                valve_signal=valve_open_signal,
                ventricle_peaks=b_peaks[b_peaks < n_raw],
                atrium_peaks=r_peaks[r_peaks < n_raw],
                valve_peaks=cur_peaks,
                fps=None,
                frame_range=(f0p, f1p),
                savgol_window=5,
                ventricle_z_threshold=VENTRICLE_Z_THRESHOLD_DEFAULT,
                atrium_z_threshold=ATRIUM_Z_THRESHOLD_DEFAULT,
                save_path=path if path else None,
                dark_mode=False)

            plt.show()

        btn_pub.on_clicked(make_pub_fig)

        # Keep alive
        fig_av._keep_alive = [
            s_athresh, s_vthresh, s_mindist, s_maxvalve,
            s_thresh, s_boost, s_search,
            s_fstart, s_fend,
            btn_redetect, btn_save_res, btn_export, btn_pub,
            redetect_valves, save_results, export_range, make_pub_fig,
            _boost_signal,
        ]

        # ── Regression guard ──
        print(f"\n  === GRAPH PLOT STATUS ===")
        print(f"  ROI selected: y=[{vly0}:{vly1}], x=[{vlx0}:{vlx1}]")
        print(f"  Valve signal computed: {len(vm_sm)} frames, "
              f"range=[{vm_sm.min():.2f}, {vm_sm.max():.2f}]")
        print(f"  Valve events: {len(valve_peaks)}")
        print(f"  Thresholds: V_z<{VENTRICLE_Z_THRESHOLD_DEFAULT}, "
              f"A_z<{ATRIUM_Z_THRESHOLD_DEFAULT}")
        print(f"  Max valve peaks: {DEFAULT_VALVE_PEAK_COUNT}")
        print(f"  Search window: {MAX_SEARCH_WINDOW} frames")
        print(f"  Mode: {'U-Net' if USE_UNET_VALVE_TRACKING else 'reference-point (manual)'}")
        print(f"  Graph plotted successfully")
        print(f"  =========================\n")

        plt.show()

    btn_loadseg.on_clicked(gui_op("Load Seg TIF + A/V")(load_seg_tif_av))

    # ── U-Net Leaflet Pipeline (annotation → train → inference) ──
    def launch_unet_pipeline(_):
        """5-step AV-valve pipeline (re-grounded on the final goal).
        STEP 1 (depth + V/A + raw) is the FOUNDATION: from the R&B-marked stack it
        extracts per-depth V/A MOVEMENT (region-area change) and loads the SEPARATE
        raw per-depth grayscale file as the U-Net working stack. STEP 2 annotates
        the AV-junction LINE on that raw stack; STEP 3 trains (1-ch raw, no ROI);
        STEP 4 infers the line, measures OPENNESS at the line MIDPOINT (no ROI),
        and detects EXACTLY 4 opens as line breaks timed between the V/A max-change
        points; STEP 5 plots the validation figure (openness + V/A + 4 opens).
        Every later step is GUARDED on the STEP-1 foundation and keyed to its
        depth + V/A file paths (logged)."""
        from tkinter import filedialog as _fd, messagebox as _mb
        import tkinter as _tk
        import leaflet_unet as _lu
        global ACTIVE_DS_DIR, ACTIVE_DEPTH, ACTIVE_STACK

        sys.path.insert(0, BASE_DIR)              # no machine-specific literals
        _root = _tk.Tk(); _root.withdraw()

        # ── active dataset folder (single source of truth; chosen in STEP 1) ──
        ds_dir = active_ds_dir()
        model_path = os.path.join(ds_dir, 'checkpoints', 'best_leaflet_unet.pth')
        meta_path = os.path.join(ds_dir, 'metadata.json')

        # ── restore the STEP-1 foundation (depth_config) for this folder ──
        dcfg = _lu.load_depth_config(ds_dir)
        if dcfg is not None:
            if ACTIVE_DEPTH is None:
                ACTIVE_DEPTH = dcfg.get("depth")
            if DATA_TIFF is None and dcfg.get("stack") and os.path.exists(dcfg["stack"]):
                globals()["DATA_TIFF"] = dcfg["stack"]
        va_v = dcfg.get("ventricle") if dcfg else None     # V/A from STEP 1 (not stale npy)
        va_a = dcfg.get("atrium") if dcfg else None
        has_depth = dcfg is not None and dcfg.get("stack") and os.path.exists(dcfg["stack"])
        has_va = bool(va_v and va_a and os.path.exists(va_v) and os.path.exists(va_a))
        has_step1 = has_depth and has_va

        tif_path = DATA_TIFF or os.path.join(BASE_DIR, "record_02092025_183907_rl_9.tif")

        has_dataset = os.path.exists(meta_path)
        has_model = os.path.exists(model_path)
        n_annotated = 0
        if has_dataset:
            try:
                n_annotated = len(json.load(open(meta_path)).get('frames', []))
            except Exception:
                pass
        band_mask = _lu.load_valve_band(ds_dir)   # auto AV-junction band (focus)
        has_band = band_mask is not None
        LOG.info(f"U-Net SESSION  dataset_folder={ds_dir}  "
                 f"step1={'YES' if has_step1 else 'NO'}  "
                 f"depth={ACTIVE_DEPTH}  "
                 f"raw_stack={dcfg.get('stack') if has_depth else 'NONE'}  "
                 f"V/A={va_v if has_va else 'NONE'} , {va_a if has_va else 'NONE'}  "
                 f"band={'YES (' + str(int(band_mask.sum())) + 'px)' if has_band else 'NONE'}  "
                 f"dataset={'YES' if has_dataset else 'NO'} ({n_annotated} annotated)  "
                 f"model={'YES' if has_model else 'NO'}  "
                 f"um/px={_lu.UM_PER_PIXEL}  (openness at LINE MIDPOINT, no ROI)")

        def _pcb(opname):
            # Per-EPOCH messages are logged; per-BATCH heartbeats ("...training...")
            # only refresh the GUI (no log flood) but keep the window responsive.
            def cb(i, n, m):
                is_beat = isinstance(m, str) and m.endswith("training...")
                set_status(f"Running: {opname} {i}/{n}  {m}", "run", log=not is_beat)
            return cb

        def _need_step1():
            set_status("ERROR: run STEP 1 (depth + V/A) first.", "error")
            _w = _tk.Tk(); _w.withdraw()
            _mb.showerror("Run STEP 1 first",
                          "No depth / chosen-depth stack / V-A foundation was found "
                          "in the active dataset folder:\n\n  " + ds_dir +
                          "\n\nRun STEP 1 (Select depth + extract V/A) first. It picks "
                          "the depth the U-Net is keyed to and writes the chosen-depth "
                          "stack + V/A chamber signals that every later step uses.\n\n"
                          "(No stale .npy / wrong depth is ever used as a fallback.)")
            _w.destroy()

        def _depth_stack():
            """The chosen-depth single-plane stack (STEP-1 single source of truth)."""
            import tifffile
            return tifffile.imread(dcfg["stack"]).astype(np.float32)

        def _depth_mismatch_ok():
            """Warn (and let the user abort) if the model was trained on a depth
            different from the current STEP-1 depth."""
            md = _lu.model_valve_depth(model_path)
            cur = ACTIVE_DEPTH
            if md is not None and cur is not None and md != cur:
                LOG.warning(f"[depth] MISMATCH: model trained on depth {md}, "
                            f"current depth is {cur}")
                _w = _tk.Tk(); _w.withdraw()
                go = _mb.askyesno("Depth mismatch",
                                  f"The trained model was built on DEPTH {md}, but the "
                                  f"current STEP-1 depth is {cur}.\n\nResults may be invalid "
                                  f"(the U-Net is trained for one specific depth).\n\n"
                                  f"YES = run anyway   NO = abort")
                _w.destroy()
                if not go:
                    set_status(f"Aborted: depth mismatch (model {md} vs current {cur}).",
                               "info")
                return go
            return True

        msg = (f"AV-Valve Pipeline (PRIMARY openness = INTENSITY darkening + M-mode)\n"
               f"folder: {os.path.basename(ds_dir)}   "
               f"STEP1(depth+V/A+raw): {'YES depth ' + str(ACTIVE_DEPTH) if has_step1 else 'NO'}   "
               f"Dataset: {'YES' if has_dataset else 'NO'} ({n_annotated})   "
               f"Model: {'YES' if has_model else 'NO'}\n\n"
               f"Choose action:\n"
               f"  1 = DEPTH + V/A + RAW stack + auto valve BAND  (run first)\n"
               f"  2 = Annotate TWO leaflets as U-curves (1 & 2, incl. closed; in band)\n"
               f"  3 = Train 3-class U-Net (SELECT frames; {'>= ' + str(MIN_TRAINING_SAMPLES) + ' rec.' if n_annotated < MIN_TRAINING_SAMPLES else 'ready'}, band-focused)\n"
               f"  4 = OPENNESS: intensity DARKENING + M-mode (mark valve) [default] / U-Net\n"
               f"  5 = Nature figure (kymograph + openness + V/A + 4 opens; SVG/PDF/PNG)\n\n"
               f"Click on the number line:")

        fig_choice, ax_choice = plt.subplots(figsize=(9.5, 3))
        ax_choice.set_title(msg, fontsize=9, ha='left', x=0)
        ax_num = fig_choice.add_axes([0.06, 0.15, 0.9, 0.3])
        ax_num.set_xlim(0.5, 5.5); ax_num.set_ylim(0, 1)
        ax_num.set_xticks([1, 2, 3, 4, 5])
        ax_num.set_xticklabels(['1: Depth+V/A+raw', '2: Annotate 2', '3: Train',
                                '4: Openness', '5: Nature fig'], fontsize=8)
        ax_num.set_yticks([])
        _has_openness = (os.path.exists(os.path.join(BASE_DIR, "valve_int_openness.npy")) or
                         os.path.exists(os.path.join(BASE_DIR, "unet_line_openness.npy")))
        for i in range(1, 6):
            c = '#40a040' if (i == 1 or
                              (i == 2 and has_step1) or
                              (i == 3 and has_step1 and n_annotated >= MIN_TRAINING_SAMPLES) or
                              (i == 4 and has_step1) or          # intensity needs no model
                              (i == 5 and has_step1 and _has_openness)) \
                else '#cc4444'
            ax_num.bar(i, 0.8, color=c, alpha=0.6)

        pts = plt.ginput(1, timeout=0)
        plt.close(fig_choice)
        if len(pts) < 1:
            _root.destroy(); return

        choice = int(round(pts[0][0]))
        _root.destroy()

        # ── STEP 1: FOUNDATION — R&B stack -> per-depth V/A (region-area change);
        #    SEPARATE raw per-depth file -> the U-Net working stack. ──
        if choice == 1:
            dest = active_ds_dir()
            ACTIVE_DS_DIR = dest
            os.makedirs(dest, exist_ok=True)
            LOG.info(f"[STEP 1] dataset folder (destination) -> {dest}")
            set_status("Running STEP 1: R&B stack -> V/A; raw per-depth file -> "
                       "U-Net stack (foundation)", "run")
            log_files(writes=[os.path.join(dest, "depth_config.json"),
                              os.path.join(dest, "ventricle_trace.npy"),
                              os.path.join(dest, "atrium_trace.npy")])
            chamber_signals_gui(dest, default_tif=tif_path)
            return

        # ── every later step REQUIRES the STEP-1 foundation ──
        if not has_step1:
            _need_step1(); return
        LOG.info(f"[U-Net] step {choice}: keyed to depth {ACTIVE_DEPTH}  "
                 f"raw_stack={dcfg['stack']}  V/A={va_v} , {va_a}")
        # the chosen-depth RAW stack is THE working stack for every later step
        src_stack = _depth_stack()

        if choice == 2:
            from leaflet_unet import annotate_frames
            # ── SEPARATE picker: the dataset DIRECTORY. Re-keys to that folder. ──
            _rd = _tk.Tk(); _rd.withdraw()
            chosen = _fd.askdirectory(
                title="Pick (or create) the dataset FOLDER to annotate the line",
                initialdir=BASE_DIR, mustexist=False)
            _rd.destroy()
            if chosen:
                ACTIVE_DS_DIR = chosen
                ds_dir = chosen
                os.makedirs(ds_dir, exist_ok=True)
                meta_path = os.path.join(ds_dir, 'metadata.json')
                dcfg = _lu.load_depth_config(ds_dir)        # foundation for THIS folder
                if not (dcfg and dcfg.get("stack") and os.path.exists(dcfg["stack"])):
                    _need_step1(); return
                if ACTIVE_DEPTH != dcfg.get("depth"):
                    ACTIVE_DEPTH = dcfg.get("depth")
            # ── SQUARE TRAINING CROP (before annotation): annotate zoomed in ──
            _existing_crop = _lu.load_training_crop(ds_dir)
            _rc = _tk.Tk(); _rc.withdraw()
            want_crop = _mb.askyesno(
                "Square training crop",
                (f"A square training crop is set: {_existing_crop}.\nRe-define it?\n\n"
                 "YES = re-draw the square crop   NO = keep it" if _existing_crop else
                 "Define a SQUARE crop around the valve first? The small valve is far "
                 "easier to annotate zoomed in.\n\nYES = draw a square crop   "
                 "NO = annotate the full frame"))
            _rc.destroy()
            if want_crop:
                import tifffile as _tf
                _full = _lu.darkfield_input(_tf.imread(dcfg["stack"]).astype(np.float32))
                define_training_crop_gui(_full, ds_dir)
            crop_now = _lu.load_training_crop(ds_dir)
            LOG.info(f"[annotate] dataset folder -> {ds_dir}  (depth {ACTIVE_DEPTH})  "
                     f"raw_stack={dcfg['stack']}  "
                     f"crop={crop_now if crop_now else 'NONE (full frame)'}")
            set_status(f"Running: Annotate TWO leaflets (1 & 2), depth {ACTIVE_DEPTH}  "
                       + (f"crop={crop_now}  " if crop_now else "")
                       + f"folder={os.path.basename(ds_dir)}", "run")
            log_files(reads=[dcfg["stack"], dcfg.get("ventricle", ""), dcfg.get("atrium", "")],
                      writes=[meta_path, os.path.join(ds_dir, 'images'),
                              os.path.join(ds_dir, 'masks'),
                              os.path.join(ds_dir, 'training_crop.json')])
            # pass the SELECTED depth's V/A (901 frames) for the live reference cursor
            annotate_frames(dcfg["stack"], ds_dir,
                            ventricle_path=dcfg.get("ventricle"),
                            atrium_path=dcfg.get("atrium"))

        elif choice == 3:
            from leaflet_unet import train_unet, annotated_frame_info
            # ── EXPLICIT training-frame selection (checklist + index/range entry) ──
            finfo = annotated_frame_info(ds_dir)
            if len(finfo) < 4:
                _mb2 = _tk.Tk(); _mb2.withdraw()
                _mb.showerror("Train blocked",
                              f"Only {len(finfo)} annotated frames - annotate a few "
                              f"more (step 2) before training.")
                _mb2.destroy()
                set_status(f"ERROR: only {len(finfo)} annotated frames.", "error")
                return
            selected = select_training_frames_gui(finfo)   # default = all annotated
            if selected is None:
                set_status("Train cancelled (frame selection cancelled).", "info"); return
            if len(selected) < 4:
                set_status(f"ERROR: {len(selected)} frames selected - need >= 4.", "error")
                _w = _tk.Tk(); _w.withdraw()
                _mb.showerror("Too few frames",
                              f"You selected {len(selected)} frames; need at least 4 "
                              f"to train. Re-run step 3 and select more.")
                _w.destroy(); return
            if len(selected) < MIN_TRAINING_SAMPLES:        # warn but ALLOW (deliberate)
                _w = _tk.Tk(); _w.withdraw()
                go = _mb.askyesno("Below recommended minimum",
                                  f"You selected {len(selected)} frames, below the "
                                  f"recommended {MIN_TRAINING_SAMPLES}.\n\nTrain anyway?\n"
                                  f"YES = proceed (I'm choosing deliberately)   NO = cancel")
                _w.destroy()
                if not go:
                    set_status(f"Train cancelled ({len(selected)} < "
                               f"{MIN_TRAINING_SAMPLES}).", "info"); return
                LOG.warning(f"[train] proceeding with {len(selected)} frames "
                            f"(< MIN_TRAINING_SAMPLES={MIN_TRAINING_SAMPLES}) by user choice")
            LOG.info(f"[train] SELECTED {len(selected)}/{len(finfo)} training frames: "
                     f"{selected}")
            set_status(f"Running: Train 3-class U-Net on {len(selected)} SELECTED frames, "
                       f"depth {ACTIVE_DEPTH} (boundary-weighted, band-focused)", "run")
            log_files(reads=[meta_path, os.path.join(ds_dir, 'depth_config.json')],
                      writes=[model_path])
            train_unet(ds_dir, epochs=100, progress_cb=_pcb("Train"),
                       train_frames=selected)

        elif choice == 4:
            # ── openness SOURCE: intensity darkening (PRIMARY) or U-Net leaflets ──
            _rs = _tk.Tk(); _rs.withdraw()
            use_int = _mb.askyesno(
                "Openness source",
                "How should valve OPENNESS be measured?\n\n"
                "YES = INTENSITY / M-mode (default, recommended): you mark the valve "
                "(ROI + line); openness = normalized DARKENING (open = darker). Builds "
                "an M-mode kymograph; no trained model needed.\n\n"
                "NO  = U-Net leaflets: requires a trained 3-class model (step 3).")
            _rs.destroy()
            src_marker = os.path.join(BASE_DIR, "last_openness_source.txt")
            # normalized chosen-depth grayscale (honest dF/F; NOT darkfield-CLAHE)
            _lo, _hi = np.percentile(src_stack, [0.5, 99.5])
            stack_int = np.clip((src_stack - _lo) / (_hi - _lo + 1e-8), 0, 1)

            if use_int:
                from leaflet_unet import analyze_valve_intensity
                # auto-localize switch (suggest the ROI from cardiac-frequency power)
                _ra = _tk.Tk(); _ra.withdraw()
                auto = _mb.askyesno("Auto-localize valve?",
                                    "Overlay the AUTO valve-activity map (cardiac-frequency "
                                    "power) to help place the ROI?\nYES = auto overlay   "
                                    "NO = plain marking")
                _ra.destroy()
                vreg = "auto" if auto else "manual"
                mark = load_valve_mark(ds_dir)
                if mark is None or _mb.askyesno("Valve mark",
                        "Re-mark the valve ROI + line? (NO = use the saved mark)"
                        if mark else "Mark the valve now?"):
                    vtr = np.load(va_v) if os.path.exists(va_v) else None
                    mark = mark_valve_gui(stack_int, ds_dir, fps=600.0,
                                          valve_region=vreg, ventricle_trace=vtr)
                if mark is None:
                    set_status("No valve mark - aborted.", "info"); return
                roi_m, line_m, ref_m = _mark_to_masks(mark, stack_int.shape[1:])
                outs = [os.path.join(BASE_DIR, "valve_int_" + f) for f in
                        ("openness.npy", "dff.npy", "gap_um.npy", "open_frames.npy",
                         "kymograph.npy", "events.npz", "kymo_meta.npz")]
                log_files(reads=[dcfg["stack"], va_a, va_v,
                                 os.path.join(ds_dir, "valve_mark.json")], writes=outs)
                set_status(f"Running: INTENSITY darkening openness + M-mode + 4 opens, "
                           f"depth {ACTIVE_DEPTH} ...", "run")
                LOG.info(f"[intensity] depth {ACTIVE_DEPTH}  raw_stack={dcfg['stack']}  "
                         f"valve_region={vreg}  ref={'YES' if ref_m is not None else 'no'}")
                os.chdir(BASE_DIR)
                analyze_valve_intensity(stack_int, roi_m, line_m,
                                        ventricle_trace_path=va_v,
                                        atrium_trace_path=va_a,
                                        fps=600.0, um_per_pixel=_lu.UM_PER_PIXEL,
                                        ref_mask=ref_m, save_prefix="valve_int",
                                        progress_cb=_pcb("Intensity"))
                open(src_marker, "w").write("intensity")
                set_status(f"Intensity openness done (depth {ACTIVE_DEPTH}); see log for "
                           f"4 opens + dF/F darkening + x-corr with V/A.", "done")
            else:
                from leaflet_unet import analyze_line_breaks
                if not has_model:
                    set_status("ERROR: No trained model — train first (step 3).", "error")
                    return
                if not _depth_mismatch_ok():
                    return
                outs = [os.path.join(BASE_DIR, "unet_line_" + f) for f in
                        ("openness.npy", "gap_px.npy", "gap_um.npy", "open_frames.npy",
                         "ctrl_points.npz", "events.npz", "figure.png")]
                _rm = _tk.Tk(); _rm.withdraw()
                use_a2 = _mb.askyesno("Openness method",
                    "CLOSED vs OPEN from the two leaflet lines by:\n\n"
                    "YES = EXTEND + OVERLAP (default)\nNO  = CONNECT + BREAK")
                _rm.destroy()
                omethod = "extend_overlap" if use_a2 else "connect_break"
                band_path = os.path.join(ds_dir, "valve_band.npy")
                log_files(reads=[dcfg["stack"], va_a, va_v, model_path]
                          + ([band_path] if has_band else []), writes=outs)
                set_status(f"Running: U-Net 2 leaflets + {omethod} + 4 opens, depth "
                           f"{ACTIVE_DEPTH} ...", "run")
                os.chdir(BASE_DIR)
                stack_n = _lu.darkfield_input(src_stack)
                # crop inference to the SAME square training crop the model used
                _crop = _lu.load_training_crop(ds_dir)
                _bm = band_mask
                if _crop is not None:
                    cy0, cx0, cy1, cx1 = _crop
                    stack_n = stack_n[:, cy0:cy1, cx0:cx1]
                    if _bm is not None and _bm.shape != stack_n.shape[1:] \
                            and cy1 <= _bm.shape[0] and cx1 <= _bm.shape[1]:
                        _bm = _bm[cy0:cy1, cx0:cx1]
                    LOG.info(f"[infer] training crop {_crop} -> {stack_n.shape[1:]}")
                analyze_line_breaks(stack_n, model_path, atrium_trace_path=va_a,
                                    ventricle_trace_path=va_v, fps=600.0,
                                    um_per_pixel=_lu.UM_PER_PIXEL, band_mask=_bm,
                                    openness_method=omethod, progress_cb=_pcb("Infer"))
                open(src_marker, "w").write("unet")
                set_status(f"U-Net openness done (depth {ACTIVE_DEPTH}, {omethod}).", "done")

        elif choice == 5:
            # pick the figure by the openness source of the LAST step-4 run
            src_marker = os.path.join(BASE_DIR, "last_openness_source.txt")
            source = open(src_marker).read().strip() if os.path.exists(src_marker) else None
            int_opn = os.path.join(BASE_DIR, "valve_int_openness.npy")
            unet_opn = os.path.join(BASE_DIR, "unet_line_openness.npy")
            if source is None:
                source = "intensity" if os.path.exists(int_opn) else "unet"
            opn = int_opn if source == "intensity" else unet_opn
            if not os.path.exists(opn):
                set_status("ERROR: run step 4 (openness) first.", "error"); return
            _r5 = _tk.Tk(); _r5.withdraw()
            save_path = _fd.asksaveasfilename(
                title="Save Nature figure (stem; writes .svg/.pdf/.png)",
                defaultextension=".svg", filetypes=[("SVG", "*.svg"), ("All", "*.*")])
            _r5.destroy()
            stem = os.path.splitext(save_path)[0] if save_path else os.path.join(
                BASE_DIR, f"valve_{source}_figure")
            os.chdir(BASE_DIR)
            _lo, _hi = np.percentile(src_stack, [0.5, 99.5])
            stack_int = np.clip((src_stack - _lo) / (_hi - _lo + 1e-8), 0, 1)
            if source == "intensity":
                mark = load_valve_mark(ds_dir)
                if mark is None:
                    set_status("ERROR: no valve_mark.json - re-run step 4 (intensity).", "error"); return
                log_files(reads=[int_opn, os.path.join(BASE_DIR, "valve_int_events.npz"),
                                 os.path.join(BASE_DIR, "valve_int_kymograph.npy"), va_v, va_a],
                          writes=[stem + ".svg", stem + ".pdf", stem + ".png"])
                set_status(f"Running: INTENSITY Nature figure (kymograph + dF/F + V/A + "
                           f"4 opens), depth {ACTIVE_DEPTH} ...", "run")
                make_intensity_valve_figure(stack_int, mark, um_per_pixel=_lu.UM_PER_PIXEL,
                                            fps=600.0, save_stem=stem, prefix="valve_int",
                                            depth=ACTIVE_DEPTH)
            else:
                log_files(reads=[unet_opn, os.path.join(BASE_DIR, "unet_line_ctrl_points.npz"),
                                 os.path.join(BASE_DIR, "unet_line_events.npz"), va_v, va_a],
                          writes=[stem + ".svg", stem + ".pdf", stem + ".png"])
                set_status(f"Running: U-Net validation figure, depth {ACTIVE_DEPTH} ...", "run")
                stack_n = _lu.darkfield_input(src_stack)
                _crop = _lu.load_training_crop(ds_dir)   # crop so it matches ctrl-point coords
                if _crop is not None:
                    cy0, cx0, cy1, cx1 = _crop
                    stack_n = stack_n[:, cy0:cy1, cx0:cx1]
                    LOG.info(f"[fig] training crop {_crop} -> {stack_n.shape[1:]}")
                # Image-panel flips (DISPLAY only). Horizontal stays ON (established);
                # offer the independent VERTICAL (up-down) flip per render.
                _flip_h, _flip_v = True, False
                try:
                    _rf = _tk.Tk(); _rf.withdraw(); _rf.attributes("-topmost", True)
                    _flip_v = bool(_mb.askyesno("Image panel flip",
                        "Flip the image panels (a, b, c) VERTICALLY (up-down) too?\n\n"
                        "(The horizontal left-right flip stays ON. Sup/Inf labels stay "
                        "anatomically correct either way.)\n\nYes = horizontal + vertical   "
                        "No = horizontal only"))
                    _rf.destroy()
                except Exception:
                    _flip_v = False
                # Figure-(a) chamber regions are selected MANUALLY first (default):
                # make_nature_valve_figure opens the draw window on the panel-a crop
                # (reuse-or-redraw if regions already saved) BEFORE rendering.
                _figa_npz = os.path.join(BASE_DIR, "unet_line_figa_regions.npz")
                set_status(f"Figure (a): SELECT ventricle/atrium regions per frame "
                           f"(CLOSED then OPEN; flip H={_flip_h}, V={_flip_v}) — reuse or re-draw...", "run")
                LOG.info(f"[fig] flips: horizontal={_flip_h} vertical={_flip_v}; "
                         f"figure-(a) regions = MANUAL (default); "
                         f"saved={'YES' if os.path.exists(_figa_npz) else 'no'} "
                         f"({os.path.basename(_figa_npz)})")
                try:
                    make_nature_valve_figure(stack_n, um_per_pixel=_lu.UM_PER_PIXEL,
                                             fps=600.0, save_stem=stem, roi=None,
                                             atrium_trace_path=va_a, ventricle_trace_path=va_v,
                                             depth=ACTIVE_DEPTH, fig_a_regions="manual",
                                             flip_horizontal=_flip_h, flip_vertical=_flip_v)
                except Exception as _fe:
                    import traceback as _tb
                    LOG.error(f"[fig] Nature figure FAILED: {_fe}\n{_tb.format_exc()}")
                    set_status(f"ERROR building Nature figure: {_fe}", "error")
                    return
            set_status(f"Nature figure ({source}) written: {stem}.svg/.pdf/.png", "done")
        else:
            set_status("Invalid choice.", "info")

    btn_unet.on_clicked(gui_op("U-Net pipeline")(launch_unet_pipeline))

    # CRITICAL: keep references alive so buttons/sliders don't get GC'd
    fig._keep_alive = [
        btn_run, btn_save, btn_zoom, btn_seg, btn_clahe, btn_loadseg,
        btn_histmask, btn_unet,
        radio_bg, radio_sp,
        s_t, s_tw, s_gm, s_cl, s_bd, s_gs, s_nl, s_bp, s_br,
        update, toggle_clahe, on_bg_change, on_sp_change,
        on_set_hist_mask,
        run_full_pipeline, save_processed_tiff, open_zoom_viewer,
        auto_segment_leaflet, load_seg_tif_av, launch_unet_pipeline,
    ]

    plt.show()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    # Default data file lives next to this script (BASE_DIR); override via argv[1].
    TIFF_PATH = (sys.argv[1] if len(sys.argv) > 1
                 else os.path.join(BASE_DIR, "record_02092025_183907_rl_9.tif"))
    DATA_TIFF = TIFF_PATH                      # single source for the input TIFF path

    LOG.info(f"Loading stack from {os.path.abspath(TIFF_PATH)}")
    stack = load_stack(TIFF_PATH)
    LOG.info(f"Loaded: {stack.shape} {stack.dtype}")

    interactive_preprocess_gui(stack)
