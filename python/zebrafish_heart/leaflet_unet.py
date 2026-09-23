# NIR-II SLIM - zebrafish valve leaflet segmentation - produces Fig. 2i-m
# environment: heart_valve_py314
"""
leaflet_unet.py
===============
Thin-structure U-Net segmentation and line-structure labeling of the zebrafish
atrioventricular (AV) valve for NIR-II SLIM cardiac imaging — rebuilt to a
publication ("Nature main figure") standard.

This module is the back-end for the four entry points called by
``valve_analysis.py``:

    annotate_frames(tif_path, ds_dir)
    train_unet(ds_dir, epochs=100)
    analyze_line_breaks(stack, model_path, atrium_trace_path, ventricle_trace_path)
    render_leaflet_overlay(stack, model_path, crop_roi, save_path, save_labels)

What is "upgraded" relative to a blob-segmentation U-Net
--------------------------------------------------------
1.  THIN-STRUCTURE-AWARE SEGMENTATION.  The leaflet is a ~30 um curvilinear
    structure only 5-15 um thick — at or below the axial resolution limit.
    A plain Dice/BCE loss happily drops the thinnest, most informative pixels.
    We add a soft-clDice (centerline-Dice) term (Shit et al., CVPR 2021) so the
    network is rewarded for preserving the *connected centerline*, not just area.

2.  LINE-STRUCTURE LABELING, not pixel blobs.  Each prediction is reduced to an
    ordered, spline-smoothed centerline and split into the two leaflets at the
    coaptation point.  Valve opening is read off as a geometric "line break"
    (the centerline separates into two components / a tip-to-tip gap opens),
    complemented by the dark-field darkening signal.

3.  ONE EVENT PER CARDIAC CYCLE.  Opening detection is segmented by the
    ventricle/atrium contraction landmarks and takes one maximum per V->A
    diastolic window, so the count equals the number of cycles (4 expected),
    matching the convention already used in valve_analysis.make_publication_figure.

4.  FIGURE-GRADE OUTPUT.  All figures use an embedded-font, vector-friendly
    Nature style and export to SVG + PDF + 600-dpi PNG for Adobe Illustrator.
    Overlays are drawn as the two-leaflet centerlines (solid or dashed) over the
    restored grayscale image and written frame-by-frame to a contiguous RGB TIFF.

Pixel calibration
-----------------
Pixel size is defined ONCE by the module constant UM_PER_PIXEL (isotropic,
um/pixel). Every um scale bar, spatial axis, and physical measurement in this
module and in valve_analysis.make_nature_valve_figure derives from it. Nothing
about pixel size is hardcoded anywhere else.

Dependencies
------------
    pip install numpy scipy scikit-image tifffile matplotlib tqdm
    pip install torch            # for train_unet / inference (CUDA build for GPU)
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from skimage.draw import line as sk_line, line_aa, disk as draw_disk
from skimage.morphology import (skeletonize, remove_small_objects,
                                binary_closing, binary_dilation, disk as morph_disk)
from skimage.measure import label as cc_label

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Configuration (edit these)
# ============================================================
# ─────────────────────────────────────────────────────────────────────────────
# PIXEL CALIBRATION — THE SINGLE SOURCE OF TRUTH for pixel size (isotropic).
# Every um scale bar, every spatial axis, and every physical measurement
# (leaflet length, tip-to-tip gap, opening width) in this module AND in
# valve_analysis.make_nature_valve_figure derives from this one constant.
# Do NOT hardcode a pixel size anywhere else.
# NOTE: at 10 um/px a 200 um scale bar = 20 px; a ~30 um leaflet spans ~3 px;
# short scale bars and a PSF-broadened centerline are EXPECTED — the um
# conversion is for labeling.
# ─────────────────────────────────────────────────────────────────────────────
UM_PER_PIXEL: float = 10.0                     # micrometers per pixel

# Acquisition
FPS_DEFAULT: float = 600.0                     # volumetric / frame rate (Hz)

# Compute (single source of truth for batch sizes)
INFER_BATCH_GPU: int = 24                       # frames per GPU inference mini-batch
INFER_BATCH_CPU: int = 8                        # frames per CPU inference mini-batch
TRAIN_BATCH_GPU: int = 8                        # training batch on GPU
# (TRAIN_BATCH below is the CPU default)

# Segmentation / labeling
PROB_THRESHOLD: float = 0.5                    # leaflet probability -> mask
MASK_THICKNESS_PX: int = 3                     # rasterized half-thickness of GT line
MIN_SKELETON_PX: int = 6                       # prune skeleton fragments shorter than this
ROI_MARGIN: int = 8                            # px margin around the valve ROI bbox crop
SPLINE_SMOOTH: float = 8.0                     # centerline spline smoothing (s in splprep)
GAP_OPEN_PX: float = 2.0                       # tip gap (px) above which we call "broken"

# Segmentation: 3-class softmax (bg / leaflet1 / leaflet2), non-overlapping, with
# a Ronneberger boundary-weight map up-weighting the inter-leaflet CONTACT.
N_CLASSES: int = 3                             # bg, leaflet1, leaflet2 (mutually exclusive)
BOUNDARY_W0: float = 10.0                      # contact-boundary weight amplitude
BOUNDARY_SIGMA: float = 3.0                    # contact-boundary weight falloff (px)

# CONNECTING-LINE detection model (current approach): the U-Net segments THREE
# INDEPENDENT (multi-label sigmoid) structures the user draws per frame -
#   ch0 = leaflet 1 line   ch1 = leaflet 2 line   ch2 = CONNECTING line.
# The two leaflet lines are for REPRESENTATION (label + U-shape schematic); the
# CONNECTOR is the DETECTION structure: continuous -> CLOSED, broken -> OPEN.
STRUCT_NAMES = ["leaflet1", "leaflet2", "connector"]
N_STRUCT: int = 3                              # multi-label channels (sigmoid, may overlap)
CONNECTOR_CH: int = 2                          # index of the connecting-line channel
LEAFLET_CHS = (0, 1)                           # leaflet channels (clDice continuity applies)
# class-imbalance: OPEN frames (connector broken) are rare (~4/clip) -> oversample.
OPEN_OVERSAMPLE: int = 6                       # extra repeats of each OPEN frame in training
# connector-break openness: prob >= this counts as "structure present" along the
# tip-to-tip connector; a missing run longer than CONNECTOR_BREAK_PX -> OPEN.
CONNECTOR_PROB_THRESH: float = 0.35
CONNECTOR_BREAK_PX: float = 2.5

# CLOSED/OPEN post-processing at the ORIFICE (free-tip gap / contact continuity).
#   'extend_overlap' (A): extend each U-tip along its tangent; CLOSED if they meet
#   'connect_break'  (B): test contact continuity along the tip-to-tip connector
OPENNESS_METHOD: str = "extend_overlap"        # 'extend_overlap' | 'connect_break'
# Leaflet identity / orifice direction: the two free tips face each other
# VERTICALLY (upper leaflet's bottom tip <-> lower leaflet's top tip).
UPPER_IS_SMALLER_Y: bool = True                # leaflet1=UPPER=smaller y (False if flipped)
EXTEND_LEN_PX: float = 4.0                     # (A) SHORT U-tip extrapolation length (px)
CLOSED_GAP_PX: float = 2.0                     # (A) residual gap (px) <= this -> CLOSED
TIP_FIT_PTS: int = 6                           # points near the tip used for the tangent
BRIDGE_PROB_THRESH: float = 0.30               # (B) leaflet-prob counted as "structure"
BRIDGE_BREAK_PX: float = 2.0                   # (B) break run (px) > this -> OPEN

# ── SINGLE SOURCE OF TRUTH: which marker COLOR is which heart CHAMBER ──
# Verified 2026-06-17 by valve-open alignment (the ATRIUM area empties, peak->trough,
# exactly as the valve opens) + morphology (the larger blue region is the ventricle):
#     RED marker  -> ATRIUM     BLUE marker -> VENTRICLE
# Everything that maps colour<->chamber (V/A extraction, signals, figure labels and
# colours, the atrium-anchored expected-open window, the timing prior) must read these.
CHAMBER_OF_COLOR = {"red": "atrium", "blue": "ventricle"}
COLOR_OF_CHAMBER = {"atrium": "red", "ventricle": "blue"}
# Display colours keyed to the SOURCE marker, so labels and colours stay consistent.
CHAMBER_COLOR = {"atrium": "#d62728", "ventricle": "#1f77b4"}   # atrium=red, ventricle=blue

# Cardiac landmarks (mirror valve_analysis defaults so events agree)
VENTRICLE_Z_THRESHOLD: float = -1.7
ATRIUM_Z_THRESHOLD: float = -2.0
EXPECTED_OPENINGS: int = 4
# Expected-open window = atrium PEAK -> just-before-TROUGH (atrial emptying limb).
ATRIUM_OPEN_START_OFFSET: int = 0              # frames from the atrium peak to START (0 = at peak)
ATRIUM_OPEN_END_LEAD: int = -1                 # frames before the trough to END (-1 = auto ~6% period)

# U-Net
UNET_BASE_CH: int = 32
UNET_DEPTH: int = 4
TRAIN_BATCH: int = 4
TRAIN_VAL_SPLIT: float = 0.2
TRAIN_LR: float = 1e-3
# SINGLE SOURCE OF TRUTH for the minimum annotated-frame count needed to train.
# valve_analysis.py imports this name (does not redefine it).
MIN_TRAINING_SAMPLES: int = 50
CLDICE_WEIGHT: float = 0.15                    # weight of centerline-Dice (clDice) in the
#   loss. LOWERED from 0.4: clDice rewards a connected line, which fights the OPEN/broken
#   valve state we must learn. Down-weighted so the gap can form; >0 still curbs spurious
#   fragmentation on closed frames.
SOFT_SKEL_ITERS: int = 8                       # iterations of the soft-skeletonization

# ---- Nature-style palette (NPG / ggsci-style, colorblind-aware) ----
C_BG = "white"
C_FG = "black"
C_MUTED = "#5b5b5b"
C_LEAFLET_A = "#E64B35"      # warm red   — leaflet 1
C_LEAFLET_B = "#4DBBD5"      # cyan       — leaflet 2
CONNECTOR_COLOR = "#F7C948"  # amber      — CONNECTING line (detection structure)
C_VENTRICLE = "#D62728"      # keep consistent with valve_analysis figure
C_ATRIUM = "#1F77B4"
C_VALVE = "#3C5488"          # opening / gap trace
C_OPEN = "#E64B35"           # detected opening markers
FONT_FAMILY = "Arial"

# ---- overlay rendering ----
OVERLAY_LINE_PX: int = 3        # solid leaflet centerline thickness (px) in the overlay TIFF
OVERLAY_DASHED: bool = False    # SOLID leaflet lines (not dashed)
OVERLAY_DASH: int = 7
OVERLAY_GAP: int = 5


# ============================================================
# Publication style helpers
# ============================================================
def set_pub_style() -> None:
    """Apply a Nature-style, vector-friendly matplotlib rcParams profile.

    Fonts are kept as editable text in SVG/PDF (``svg.fonttype='none'`` and
    TrueType embedding) so labels remain selectable/editable in Illustrator.
    """
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_letter(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", family=FONT_FAMILY)


def add_scale_bar(ax, length_um: float, img_w_px: int,
                  pixel_size_um: Optional[float] = None,
                  loc: str = "lower right", color: str = "white",
                  thickness_frac: float = 0.012, pad_frac: float = 0.05,
                  label: bool = True) -> None:
    """Draw a scale bar on an image axis. Falls back to a pixel bar if the
    pixel size is unknown."""
    px = pixel_size_um if pixel_size_um is not None else UM_PER_PIXEL
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    H = abs(yhi - ylo)
    W = abs(xhi - xlo)
    if px:
        bar_px = length_um / px
        text = f"{length_um:g} \u00b5m"
    else:
        bar_px = max(10.0, 0.15 * img_w_px)
        text = f"{bar_px:.0f} px"
    pad = pad_frac * W
    h = thickness_frac * H
    if "right" in loc:
        x0 = max(xlo, xhi) - pad - bar_px
    else:
        x0 = min(xlo, xhi) + pad
    if "lower" in loc:
        y0 = max(ylo, yhi) - pad
    else:
        y0 = min(ylo, yhi) + pad + h
    ax.add_patch(Rectangle((x0, y0 - h), bar_px, h, color=color,
                           ec="none", zorder=10))
    if label:
        ax.text(x0 + bar_px / 2, y0 - h - 0.012 * H, text, color=color,
                ha="center", va="bottom", fontsize=6.5, family=FONT_FAMILY,
                zorder=10)


def save_vector(fig, save_path: str, also: Sequence[str] = ("svg", "pdf", "png")) -> list:
    """Save a figure to SVG + PDF + PNG (for Illustrator). Returns paths."""
    stem = os.path.splitext(save_path)[0]
    out = []
    for ext in also:
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=(600 if ext == "png" else None), facecolor="white",
                    transparent=False)
        out.append(p)
    return out


# ============================================================
# Compute device (single source of truth for CUDA vs CPU)
# ============================================================
def get_device():
    """Return the torch.device to use everywhere (cuda if available, else cpu)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_banner(op: str):
    """Print a clear device report at the start of a torch operation and return
    (device, is_cuda). If CUDA is unavailable, tell the user how to enable it."""
    import torch
    dev = get_device()
    is_cuda = dev.type == "cuda"
    print(f"[{op}] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"[{op}] USING GPU: {torch.cuda.get_device_name(0)}  (device={dev})")
    else:
        print(f"[{op}] USING CPU (device={dev}) - this is slow for the 901-frame stack.")
        print(f"[{op}] To enable GPU, install a CUDA build of torch, e.g.:")
        print(f"[{op}]   pip install torch --index-url "
              f"https://download.pytorch.org/whl/cu121   (match your CUDA version)")
    return dev, is_cuda


def infer_batch(is_cuda: bool) -> int:
    """Single source of truth for the inference mini-batch size."""
    return INFER_BATCH_GPU if is_cuda else INFER_BATCH_CPU


def _pid_alive(pid: int) -> bool:
    """Best-effort: is process ``pid`` still running? Used to detect a STALE
    training lock (crashed process) vs a live concurrent training. If we cannot
    tell, return True (assume alive -> safer to refuse than to clobber)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(int(pid), 0)                       # POSIX: no signal, just probe
        return True
    except (ProcessLookupError, OSError):
        return False
    except Exception:
        return True


def _acquire_training_lock(ds_dir: str):
    """Refuse to start a SECOND concurrent training on the SAME dataset folder
    (two trainings clobber each other's best_leaflet_unet.pth and starve the
    GPU). Returns the lock path on success, or None if another LIVE training in a
    DIFFERENT process already holds it. A lock left by a crashed/dead process (or
    by THIS process) is reclaimed. Released at process exit via atexit."""
    import atexit
    lock = os.path.join(ds_dir, ".training.lock")
    if os.path.exists(lock):
        try:
            info = json.load(open(lock)); opid = int(info.get("pid", -1))
        except Exception:
            opid = -1
        if opid > 0 and opid != os.getpid() and _pid_alive(opid):
            print(f"[train] REFUSED: a training is already running on this folder "
                  f"(pid {opid}, since {info.get('time_str', '?')}). Close the other "
                  f"window/process, or delete:\n   {lock}")
            return None
        if opid == os.getpid():
            print("[train] re-acquiring this process's own training lock.")
        else:
            print(f"[train] reclaiming stale training lock (pid {opid} not running).")
    try:
        json.dump({"pid": os.getpid(), "time": time.time(),
                   "time_str": time.strftime("%Y-%m-%d %H:%M:%S")}, open(lock, "w"))
        atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    except Exception as e:
        print(f"[train] WARNING could not write training lock ({e}); proceeding.")
    return lock


# ============================================================
# Architecture (torch imported lazily so the figure/geometry code
# runs on machines without torch)
# ============================================================
def _torch():
    import torch  # noqa: F401
    return torch


def build_unet(in_ch: int = 1, base: int = UNET_BASE_CH, depth: int = UNET_DEPTH,
               out_ch: int = 1):
    """Construct a residual, attention-gated, deeply-supervised U-Net for thin
    curvilinear segmentation. ``out_ch`` output channels (2 = the two leaflets as
    independent sigmoid maps, so they can touch/overlap when closed).

    - Residual ConvBlocks (skip-add) ease optimization of the deep, thin-feature net.
    - GroupNorm (not BatchNorm) so it trains stably at the small batch sizes used here.
    - Additive attention gates (Oktay et al., 2018) sharpen the thin foreground in
      each skip connection.
    - Deep supervision: an auxiliary 1x1 head at every decoder scale; ``forward(x,
      deep=True)`` returns ``(main_logits, [aux_logits, ...])`` (all upsampled to
      input size) so the trainer can add auxiliary losses.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def gn(c):                                   # safe GroupNorm group count
        g = 8
        while c % g and g > 1:
            g //= 2
        return nn.GroupNorm(g, c)

    class ResBlock(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.c1 = nn.Conv2d(ci, co, 3, padding=1, bias=False); self.n1 = gn(co)
            self.c2 = nn.Conv2d(co, co, 3, padding=1, bias=False); self.n2 = gn(co)
            self.skip = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            s = self.skip(x)
            x = self.act(self.n1(self.c1(x)))
            x = self.n2(self.c2(x))
            return self.act(x + s)

    class AttnGate(nn.Module):
        def __init__(self, fg, fl, fint):
            super().__init__()
            self.wg = nn.Sequential(nn.Conv2d(fg, fint, 1), gn(fint))
            self.wx = nn.Sequential(nn.Conv2d(fl, fint, 1), gn(fint))
            self.psi = nn.Sequential(nn.Conv2d(fint, 1, 1), nn.Sigmoid())
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            a = self.relu(self.wg(g) + self.wx(x))
            return x * self.psi(a)

    class AttResUNet(nn.Module):
        def __init__(self, in_ch, base, depth, out_ch=1):
            super().__init__()
            self.depth = depth
            chs = [base * (2 ** i) for i in range(depth + 1)]
            self.enc = nn.ModuleList()
            self.pool = nn.MaxPool2d(2)
            prev = in_ch
            for c in chs[:-1]:
                self.enc.append(ResBlock(prev, c)); prev = c
            self.bottleneck = ResBlock(chs[-2], chs[-1])
            self.up = nn.ModuleList(); self.gates = nn.ModuleList()
            self.dec = nn.ModuleList(); self.aux = nn.ModuleList()
            for i in range(depth - 1, -1, -1):
                cup, cskip = chs[i + 1], chs[i]
                self.up.append(nn.ConvTranspose2d(cup, cskip, 2, stride=2))
                self.gates.append(AttnGate(cskip, cskip, max(cskip // 2, 1)))
                self.dec.append(ResBlock(cskip * 2, cskip))
                self.aux.append(nn.Conv2d(cskip, out_ch, 1))
            self.head = nn.Conv2d(chs[0], out_ch, 1)

        def forward(self, x, deep: bool = False):
            skips = []
            for e in self.enc:
                x = e(x); skips.append(x); x = self.pool(x)
            x = self.bottleneck(x)
            auxs = []
            for up, gate, dec, aux, s in zip(self.up, self.gates, self.dec,
                                             self.aux, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != s.shape[-2:]:
                    x = F.interpolate(x, size=s.shape[-2:], mode="bilinear",
                                      align_corners=False)
                x = dec(torch.cat([gate(x, s), x], dim=1))
                if deep:
                    auxs.append(aux(x))
            main = self.head(x)
            if deep:
                auxs = [F.interpolate(a, size=main.shape[-2:], mode="bilinear",
                                      align_corners=False) for a in auxs]
                return main, auxs
            return main

    return AttResUNet(in_ch, base, depth, out_ch)


def _pad_to_multiple(x, m: int = 16):
    """Pad a torch tensor (N,C,H,W) so H,W are multiples of m. Returns (x, (ph, pw))."""
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


# ============================================================
# Losses (soft-Dice + BCE + soft-clDice for thin centerlines)
# ============================================================
def make_loss(cldice_weight: float = CLDICE_WEIGHT, iters: int = SOFT_SKEL_ITERS):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def soft_skeleton(p, iters):
        """Differentiable morphological skeleton (Shit et al. 2021)."""
        def mp(x):  # min-pool via -maxpool(-x)
            return -F.max_pool2d(-x, 3, 1, 1)
        def op(x):  # opening
            return F.max_pool2d(mp(x), 3, 1, 1)
        skel = F.relu(p - op(p))
        for _ in range(iters):
            p = mp(p)
            delta = F.relu(p - op(p))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def soft_dice(p, y, eps=1e-5):
        num = 2 * (p * y).sum(dim=(2, 3)) + eps
        den = (p + y).sum(dim=(2, 3)) + eps
        return 1 - (num / den).mean()

    def soft_cldice(p, y, eps=1e-5):
        sp, sy = soft_skeleton(p, iters), soft_skeleton(y, iters)
        tprec = (sp * y).sum(dim=(2, 3)) + eps
        tprec = tprec / (sp.sum(dim=(2, 3)) + eps)
        tsens = (sy * p).sum(dim=(2, 3)) + eps
        tsens = tsens / (sy.sum(dim=(2, 3)) + eps)
        cl = 2 * (tprec * tsens) / (tprec + tsens)
        return 1 - cl.mean()

    def loss_fn(logits, y):
        """L = 0.5*softDice + 0.5*BCE + cldice_weight*clDice, with
        inverse-frequency (pos_weight) BCE so the few thin leaflet pixels aren't
        drowned by background.

        NOTE: clDice rewards a CONNECTED centerline. For the AV valve the
        biologically meaningful signal is the OPEN/BROKEN state (a gap at the
        orifice), so the connectivity term is down-weighted (CLDICE_WEIGHT) - a
        strong clDice would actively fight the very break we want the model to
        learn. It still helps suppress spurious fragmentation on CLOSED frames."""
        p = torch.sigmoid(logits)
        dice = soft_dice(p, y)
        cl = soft_cldice(p, y)
        pos = y.sum()
        neg = y.numel() - pos
        pw = torch.clamp(neg / (pos + 1.0), max=200.0)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        return 0.5 * dice + 0.5 * bce + cldice_weight * cl

    def cldice_score(logits, y):
        """Connectivity metric in [0,1] (higher better) for model selection."""
        with torch.no_grad():
            return float(1.0 - soft_cldice(torch.sigmoid(logits), y))

    loss_fn.cldice_score = cldice_score
    return loss_fn


def leaflet_label_and_weight(m1, m2, w0: float = BOUNDARY_W0,
                             sigma: float = BOUNDARY_SIGMA):
    """Build the 3-class LABEL map (0=bg, 1=leaflet1, 2=leaflet2; MUTUALLY
    EXCLUSIVE - the leaflets abut, never overlap) and a Ronneberger BOUNDARY
    WEIGHT map from two leaflet masks.

    Overlapping annotation pixels (closed frames, where the two thin curves
    touch) are split to the nearer leaflet core, so the inter-leaflet boundary
    becomes the CONTACT. The weight map up-weights pixels near BOTH leaflets
    (the contact / orifice) - w = 1 + w0*exp(-(d1+d2)^2 / 2 sigma^2) - so the
    model precisely localizes where the leaflets meet/part."""
    from scipy import ndimage as ndi
    m1 = np.asarray(m1, bool); m2 = np.asarray(m2, bool)
    H, W = m1.shape
    lab = np.zeros((H, W), np.int64)
    lab[m1] = 1; lab[m2] = 2
    both = m1 & m2
    if both.any():                                # split overlap to the nearer core
        core1, core2 = m1 & ~m2, m2 & ~m1
        BIG = np.float32(1e6)
        d1 = ndi.distance_transform_edt(~core1) if core1.any() else np.full((H, W), BIG)
        d2 = ndi.distance_transform_edt(~core2) if core2.any() else np.full((H, W), BIG)
        lab[both] = np.where(d1[both] <= d2[both], 1, 2)
    BIG = np.float32(1e6)
    d_to_1 = ndi.distance_transform_edt(~m1) if m1.any() else np.full((H, W), BIG)
    d_to_2 = ndi.distance_transform_edt(~m2) if m2.any() else np.full((H, W), BIG)
    wmap = 1.0 + w0 * np.exp(-((d_to_1 + d_to_2) ** 2) / (2.0 * sigma * sigma))
    return lab.astype(np.int64), wmap.astype(np.float32)


def make_loss3(cldice_weight: float = CLDICE_WEIGHT, iters: int = SOFT_SKEL_ITERS,
               n_classes: int = N_CLASSES):
    """3-CLASS (softmax) loss for the touch-aware, non-overlapping leaflets:
      * BOUNDARY-WEIGHTED cross-entropy (Ronneberger weight map emphasizes the
        inter-leaflet contact / orifice),
      * soft Dice over the two foreground classes,
      * soft clDice PER LEAFLET (each U-curve stays continuous; the two leaflets
        are NOT forced to connect).
    ``loss_fn(logits, target_long, wmap)``; ``.leaflet_score`` = mean per-leaflet
    clDice (higher better) for checkpoint selection."""
    import torch
    import torch.nn.functional as F

    def soft_skeleton(p, it):
        def mp(x):
            return -F.max_pool2d(-x, 3, 1, 1)
        def op(x):
            return F.max_pool2d(mp(x), 3, 1, 1)
        skel = F.relu(p - op(p))
        for _ in range(it):
            p = mp(p)
            delta = F.relu(p - op(p))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def _cldice1(p, y, eps=1e-5):                 # p,y: (B,1,H,W) for ONE leaflet
        sp, sy = soft_skeleton(p, iters), soft_skeleton(y, iters)
        tprec = ((sp * y).sum((2, 3)) + eps) / (sp.sum((2, 3)) + eps)
        tsens = ((sy * p).sum((2, 3)) + eps) / (sy.sum((2, 3)) + eps)
        return 1 - (2 * tprec * tsens / (tprec + tsens)).mean()

    def loss_fn(logits, target, wmap, eps=1e-5):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, reduction="none")       # (B,H,W)
        wce = (ce * wmap).sum() / (wmap.sum() + eps)          # boundary-weighted CE
        prob = logp.exp()
        dice = 0.0; cl = 0.0
        for c in range(1, n_classes):                         # foreground classes only
            p = prob[:, c:c + 1]
            y = (target == c).float().unsqueeze(1)
            num = 2 * (p * y).sum((2, 3)) + eps
            den = (p + y).sum((2, 3)) + eps
            dice = dice + (1 - (num / den).mean())
            cl = cl + _cldice1(p, y)
        k = max(1, n_classes - 1)
        return wce + 0.5 * (dice / k) + cldice_weight * (cl / k)

    def leaflet_score(logits, target):
        with torch.no_grad():
            prob = F.softmax(logits, dim=1)
            s = 0.0
            for c in range(1, n_classes):
                p = prob[:, c:c + 1]; y = (target == c).float().unsqueeze(1)
                s += float(1.0 - _cldice1(p, y))
            return s / max(1, n_classes - 1)

    loss_fn.leaflet_score = leaflet_score
    return loss_fn


# ============================================================
# CONNECTING-LINE approach: multi-label (sigmoid) targets, loss, augmentation
# ============================================================
def structures_label_and_weight(m1, m2, mc, w0: float = BOUNDARY_W0,
                                sigma: float = BOUNDARY_SIGMA):
    """Build the multi-LABEL target (N_STRUCT, H, W) float = [leaflet1, leaflet2,
    CONNECTOR] (INDEPENDENT channels; they may overlap) plus a Ronneberger
    BOUNDARY-WEIGHT map (H, W) up-weighting the inter-leaflet CONTACT / orifice -
    i.e. exactly where the connecting line lives - so the model localizes the
    connector break precisely."""
    from scipy import ndimage as ndi
    m1 = np.asarray(m1, bool); m2 = np.asarray(m2, bool); mc = np.asarray(mc, bool)
    H, W = m1.shape
    target = np.stack([m1, m2, mc], 0).astype(np.float32)      # (3,H,W)
    BIG = np.float32(1e6)
    d1 = ndi.distance_transform_edt(~m1) if m1.any() else np.full((H, W), BIG)
    d2 = ndi.distance_transform_edt(~m2) if m2.any() else np.full((H, W), BIG)
    wmap = 1.0 + w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma * sigma))
    return target, wmap.astype(np.float32)


def make_loss_multilabel(n_ch: int = N_STRUCT, cldice_ch=LEAFLET_CHS,
                         cldice_weight: float = CLDICE_WEIGHT,
                         iters: int = SOFT_SKEL_ITERS):
    """MULTI-LABEL (independent sigmoid) loss for the THREE structures:
      * BOUNDARY-WEIGHTED BCE-with-logits per channel (weight map emphasizes the
        leaflet contact / orifice where the connector lives),
      * soft Dice per channel,
      * soft clDice ONLY on the LEAFLET channels (``cldice_ch``) - the CONNECTOR is
        deliberately NOT pushed to stay connected, so it is free to BREAK when the
        valve is open (the break IS the openness signal).
    ``loss_fn(logits, target, wmap)``; target is (B, n_ch, H, W) float. The
    checkpoint-selection metric ``.leaflet_score`` = mean leaflet clDice."""
    import torch
    import torch.nn.functional as F

    def soft_skeleton(p, it):
        def mp(x):
            return -F.max_pool2d(-x, 3, 1, 1)
        def op(x):
            return F.max_pool2d(mp(x), 3, 1, 1)
        skel = F.relu(p - op(p))
        for _ in range(it):
            p = mp(p)
            delta = F.relu(p - op(p))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def _cldice1(p, y, eps=1e-5):
        sp, sy = soft_skeleton(p, iters), soft_skeleton(y, iters)
        tprec = ((sp * y).sum((2, 3)) + eps) / (sp.sum((2, 3)) + eps)
        tsens = ((sy * p).sum((2, 3)) + eps) / (sy.sum((2, 3)) + eps)
        return 1 - (2 * tprec * tsens / (tprec + tsens)).mean()

    def loss_fn(logits, target, wmap, eps=1e-5):
        # boundary-weighted BCE per channel (wmap broadcast over channels)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        w = wmap.unsqueeze(1)                                  # (B,1,H,W)
        wbce = (bce * w).sum() / (w.sum() * logits.shape[1] + eps)
        prob = torch.sigmoid(logits)
        dice = 0.0; cl = 0.0
        for c in range(logits.shape[1]):
            p = prob[:, c:c + 1]; y = target[:, c:c + 1]
            num = 2 * (p * y).sum((2, 3)) + eps
            den = (p + y).sum((2, 3)) + eps
            dice = dice + (1 - (num / den).mean())
            if c in cldice_ch:                                # leaflets only
                cl = cl + _cldice1(p, y)
        nk = max(1, len(cldice_ch))
        return wbce + 0.5 * (dice / logits.shape[1]) + cldice_weight * (cl / nk)

    def leaflet_score(logits, target):
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            s = 0.0
            for c in cldice_ch:
                p = prob[:, c:c + 1]; y = target[:, c:c + 1]
                s += float(1.0 - _cldice1(p, y))
            return s / max(1, len(cldice_ch))

    loss_fn.leaflet_score = leaflet_score
    return loss_fn


def _augment_multi(img: np.ndarray, chans, rng: np.random.Generator):
    """Like ``_augment`` but transforms a LIST of label channels (e.g. leaflet1,
    leaflet2, connector) with the SAME geometric ops, so all stay aligned with the
    image. Intensity/noise ops apply to the image only. Returns (img, [chans])."""
    H, W = img.shape
    chans = [np.asarray(c, np.float32) for c in chans]

    def _geo_mask(m, fn):
        return (fn(m) > 0.5).astype(np.float32)

    if rng.random() < 0.5:
        img = img[:, ::-1].copy(); chans = [c[:, ::-1].copy() for c in chans]
    if rng.random() < 0.5:
        img = img[::-1].copy(); chans = [c[::-1].copy() for c in chans]
    if rng.random() < 0.7:                                    # rotation
        ang = rng.uniform(-12, 12)
        img = ndimage.rotate(img, ang, reshape=False, order=1, mode="reflect")
        chans = [_geo_mask(c, lambda x: ndimage.rotate(x, ang, reshape=False,
                 order=0, mode="constant")) for c in chans]
    if rng.random() < 0.5:                                    # scale (zoom) + refit
        z = rng.uniform(0.85, 1.18)
        img = _center_fit(ndimage.zoom(img, z, order=1), (H, W), 0.0)
        chans = [_geo_mask(c, lambda x: _center_fit(ndimage.zoom(x, z, order=0),
                 (H, W), 0.0)) for c in chans]
    if rng.random() < 0.4:                                    # elastic (shared field)
        dy = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1,
                                     rng.uniform(4, 7)) * rng.uniform(8, 18)
        dx = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1,
                                     rng.uniform(4, 7)) * rng.uniform(8, 18)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords = np.array([yy + dy, xx + dx])
        img = ndimage.map_coordinates(img, coords, order=1, mode="reflect")
        chans = [(ndimage.map_coordinates(c, coords, order=0, mode="constant") > 0.5)
                 .astype(np.float32) for c in chans]
    if rng.random() < 0.6:                                    # gamma / intensity jitter
        img = np.clip(np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
                      * rng.uniform(0.8, 1.2), 0, 1)
    if rng.random() < 0.3:
        img = ndimage.gaussian_filter(img, sigma=rng.uniform(0.4, 0.9))
    if rng.random() < 0.5:                                    # Poisson shot noise
        peak = rng.uniform(300, 1500)
        img = np.clip(rng.poisson(np.clip(img, 0, 1) * peak) / peak, 0, 1)
    if rng.random() < 0.4:
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    return img.astype(np.float32), chans


def ushape_around_line(line_yx, orifice_yx, width_px: float = 7.0,
                       n: int = 40):
    """SCHEMATIC 'virtual leaflet' U-shape built AROUND a drawn leaflet centerline
    (NOT a segmented shape). The U's CLOSED bottom (a ROUNDED cap) faces the
    orifice (``orifice_yx``, where the two leaflets meet); the OPEN side faces away.

    Returns a dense, SMOOTH ordered (M,2) [y,x] boundary polygon: the centerline is
    SPLINE-smoothed, offset by +/- width/2 along its smooth normal into two curved
    strands, joined at the orifice end by a SEMICIRCULAR rounded cap (no sharp
    corners). Purely for display / labeling."""
    P = np.asarray(line_yx, float)
    if len(P) < 2:
        return np.empty((0, 2))
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if seg[-1] <= 0:
        return np.empty((0, 2))
    # ── SMOOTH curved centerline (spline; fall back to dense linear) ──
    try:
        from scipy.interpolate import splprep, splev
        kk = min(3, len(P) - 1)
        tck, _ = splprep([P[:, 0], P[:, 1]], u=seg / seg[-1],
                         s=max(1.0, len(P) * 0.5), k=kk)
        Y, X = splev(np.linspace(0, 1, n), tck)
        C = np.stack([Y, X], 1)
    except Exception:
        u = np.linspace(0, seg[-1], n)
        C = np.stack([np.interp(u, seg, P[:, 0]), np.interp(u, seg, P[:, 1])], 1)
    # orient so C[-1] is the ORIFICE end (rounded cap there)
    o = np.asarray(orifice_yx, float)
    if np.linalg.norm(C[0] - o) < np.linalg.norm(C[-1] - o):
        C = C[::-1]
    tang = np.gradient(C, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    r = width_px / 2.0
    left = C + nrm * r
    right = C - nrm * r
    # SEMICIRCULAR rounded cap at the orifice end: left[-1] -> bulge(+tangent) -> right[-1]
    t_end = C[-1] - C[-2]
    t_end = t_end / (np.linalg.norm(t_end) + 1e-9)        # tangent toward orifice
    nh = nrm[-1]
    th = np.linspace(0.0, np.pi, 14)
    cap = C[-1] + r * (np.cos(th)[:, None] * nh + np.sin(th)[:, None] * t_end)
    return np.vstack([left, cap, right[::-1]])


# ============================================================
# Line-structure geometry  (the core of "line-structure labeling")
# ============================================================
def rasterize_polyline(pts: np.ndarray, shape: tuple[int, int],
                       thickness: int = MASK_THICKNESS_PX) -> np.ndarray:
    """Rasterize an ordered polyline [(y, x), ...] to a thin binary mask."""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        if len(pts) == 1:
            yy, xx = draw_disk((pts[0, 0], pts[0, 1]), max(1, thickness), shape=shape)
            m[yy, xx] = True
        return m.astype(np.uint8)
    for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
        rr, cc = sk_line(int(round(y0)), int(round(x0)),
                         int(round(y1)), int(round(x1)))
        rr = np.clip(rr, 0, H - 1); cc = np.clip(cc, 0, W - 1)
        m[rr, cc] = True
    if thickness > 1:
        m = ndimage.binary_dilation(m, structure=morph_disk(thickness // 2))
    return m.astype(np.uint8)


def _skeleton_adjacency(skel: np.ndarray):
    """8-connectivity adjacency over skeleton pixels. Returns coords (N,2) and
    a dict idx -> list[idx]."""
    ys, xs = np.nonzero(skel)
    coords = np.stack([ys, xs], axis=1)
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    adj = {i: [] for i in range(len(coords))}
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    adj[i].append(j)
    return coords, adj


def _bfs_farthest(adj, src):
    """Return (farthest_node, parent_map) from src via BFS (unit edges)."""
    from collections import deque
    dist = {src: 0}
    parent = {src: -1}
    dq = deque([src])
    far, fard = src, 0
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                if dist[v] > fard:
                    fard, far = dist[v], v
                dq.append(v)
    return far, parent


def order_skeleton(skel: np.ndarray) -> np.ndarray:
    """Order skeleton pixels into a single open path (the leaflet centerline),
    longest-shortest-path between the two most distant endpoints (tree diameter).
    Falls back to PCA ordering for loopy skeletons. Returns (M,2) [(y,x), ...]."""
    coords, adj = _skeleton_adjacency(skel)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    if len(coords) == 1:
        return coords.astype(float)
    # double BFS -> diameter path
    a, _ = _bfs_farthest(adj, 0)
    b, parent = _bfs_farthest(adj, a)
    path = []
    node = b
    seen = set()
    while node != -1 and node not in seen:
        seen.add(node)
        path.append(node)
        node = parent.get(node, -1)
    if len(path) < 2:  # fallback: PCA projection ordering
        c = coords - coords.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        order = np.argsort(c @ vt[0])
        return coords[order].astype(float)
    return coords[path[::-1]].astype(float)


def smooth_centerline(path: np.ndarray, smooth: float = SPLINE_SMOOTH,
                      n: int = 200) -> np.ndarray:
    """Fit a smoothing spline through an ordered centerline and resample it."""
    if len(path) < 4:
        return path.astype(float)
    try:
        # de-duplicate consecutive points (splprep dislikes repeats)
        keep = np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]
        p = path[keep]
        if len(p) < 4:
            return path.astype(float)
        tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, k=min(3, len(p) - 1))
        u = np.linspace(0, 1, n)
        yy, xx = splev(u, tck)
        return np.stack([yy, xx], axis=1)
    except Exception:
        return path.astype(float)


def _arc_length(path: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.zeros(len(path))
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))]
    return d


def split_two_leaflets(mask: np.ndarray, smooth: float = SPLINE_SMOOTH):
    """Reduce a leaflet mask to its line structure and split into TWO leaflets.

    Returns dict with:
        n_components : number of skeleton fragments (>=2 => clearly "broken"/open)
        leaflet_a, leaflet_b : (M,2) ordered, spline-smoothed centerlines (or None)
        gap_px       : tip-to-tip separation between the two leaflet free edges (px)
        open_len_px  : total skeleton length labeled (proxy for visible leaflet)
    Strategy:
      * skeletonize; prune tiny fragments
      * >=2 fragments  -> the valve is OPEN; the two longest fragments are the
        two leaflets, gap = min distance between their endpoints near the centre
      * 1 fragment     -> CLOSED/coapted; order it and split at the coaptation
        point (mid arc-length) into two leaflets, gap ~ 0
    """
    skel = skeletonize(mask > 0)
    lbl = cc_label(skel, connectivity=2)
    if lbl.max() > 0:                            # prune tiny fragments (version-proof)
        sizes = np.bincount(lbl.ravel())
        small = np.where(sizes < MIN_SKELETON_PX)[0]
        small = small[small != 0]                # never the background label
        if len(small):
            skel = skel & ~np.isin(lbl, small)
            lbl = cc_label(skel, connectivity=2)
    n = int(lbl.max())
    out = dict(n_components=n, leaflet_a=None, leaflet_b=None,
               gap_px=0.0, open_len_px=0.0)
    if n == 0:
        return out

    frags = []
    for k in range(1, n + 1):
        path = order_skeleton(lbl == k)
        if len(path) >= 2:
            frags.append(path)
    if not frags:
        return out
    frags.sort(key=lambda p: _arc_length(p)[-1], reverse=True)
    out["open_len_px"] = float(sum(_arc_length(p)[-1] for p in frags))

    if len(frags) >= 2:
        from scipy.spatial.distance import cdist
        a = smooth_centerline(frags[0], smooth)
        b = smooth_centerline(frags[1], smooth)
        gap = float(cdist(a, b).min())
        if gap >= GAP_OPEN_PX:                    # genuine opening -> keep two leaflets
            out["leaflet_a"], out["leaflet_b"] = a, b
            out["gap_px"] = gap
            return out
        # min-gap guard: a sub-threshold gap is noise -> stitch the two longest
        # fragments end-to-end and treat as ONE continuous (closed) leaflet.
        ea, eb = np.array([frags[0][0], frags[0][-1]]), np.array([frags[1][0], frags[1][-1]])
        i, j = np.unravel_index(int(cdist(ea, eb).argmin()), (2, 2))
        aa = frags[0] if i == 1 else frags[0][::-1]
        bb = frags[1] if j == 0 else frags[1][::-1]
        frags = [np.vstack([aa, bb])]

    # single fragment (or stitched) -> closed, split at coaptation (mid arc-length)
    path = smooth_centerline(frags[0], smooth, n=200)
    s = _arc_length(path)
    mid = np.searchsorted(s, s[-1] / 2.0)
    a, b = path[:mid + 1], path[mid:]
    out["leaflet_a"], out["leaflet_b"] = a, b
    out["gap_px"] = 0.0
    return out


# ============================================================
# Quantitative geometry (px + um) and temporal consistency
# ============================================================
def resample_path(path: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Resample an ordered centerline to ``n`` equal-arc-length points (n,2)."""
    if path is None or len(path) < 2:
        return None
    s = _arc_length(path)
    if s[-1] <= 0:
        return None
    si = np.linspace(0, s[-1], n)
    y = np.interp(si, s, path[:, 0])
    x = np.interp(si, s, path[:, 1])
    return np.stack([y, x], axis=1)


def leaflet_geometry(info: dict, um_per_pixel: Optional[float] = None) -> dict:
    """Per-frame leaflet metrics from a split_two_leaflets() result.

    gap          tip-to-tip opening width (px, um)
    length       total labeled leaflet centerline length (px, um)
    angle_deg    coaptation opening angle: 0 deg = coapted/straight, larger =
                 more splayed open (180 - angle between the two leaflet bodies
                 measured at their mutually-closest tips)
    """
    a, b = info.get("leaflet_a"), info.get("leaflet_b")
    gap_px = float(info.get("gap_px", 0.0))
    length_px = float(info.get("open_len_px", 0.0))
    angle = float("nan")
    if a is not None and b is not None and len(a) >= 3 and len(b) >= 3:
        from scipy.spatial.distance import cdist
        ea = np.array([a[0], a[-1]]); eb = np.array([b[0], b[-1]])
        i, j = np.unravel_index(int(cdist(ea, eb).argmin()), (2, 2))

        def bodydir(path, tip_is_start):
            k = min(len(path) - 1, 5)
            v = (path[k] - path[0]) if tip_is_start else (path[-1 - k] - path[-1])
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 1e-6 else v
        va, vb = bodydir(a, i == 0), bodydir(b, j == 0)
        between = np.degrees(np.arccos(np.clip(np.dot(va, vb), -1, 1)))
        angle = float(180.0 - between)
    upp = um_per_pixel
    return dict(gap_px=gap_px, gap_um=(gap_px * upp if upp else float("nan")),
                length_px=length_px,
                length_um=(length_px * upp if upp else float("nan")),
                angle_deg=angle)


def hampel_clean(x: np.ndarray, window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace isolated outliers (physically-impossible single-frame geometry
    jumps) by the local median, WITHOUT blurring genuine multi-frame openings.
    A Hampel filter: flag points > n_sigmas robust-std from the local median."""
    x = np.asarray(x, float).copy()
    n = len(x)
    k = window // 2
    out = x.copy()
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        seg = x[lo:hi]
        med = np.nanmedian(seg)
        mad = np.nanmedian(np.abs(seg - med)) * 1.4826
        if mad > 1e-9 and abs(x[i] - med) > n_sigmas * mad:
            out[i] = med
    return out


def temporal_smooth_trace(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Hampel outlier removal + light median smoothing over a short time window.
    Window stays small (3-5 frames) so the brief valve openings survive."""
    x = hampel_clean(np.nan_to_num(np.asarray(x, float)), window=window)
    return ndimage.median_filter(x, size=3, mode="nearest")


def smooth_centerline_track(cp: np.ndarray, window: int = 5) -> np.ndarray:
    """Temporally smooth a (T, n, 2) stack of per-frame resampled control points.
    Uses a short Savitzky-Golay window along time over runs of valid (non-NaN)
    frames; isolated valid frames are left untouched. Brief openings (a few
    consecutive frames) are preserved because the window is short."""
    from scipy.signal import savgol_filter
    cp = np.array(cp, float)
    T = cp.shape[0]
    valid = ~np.isnan(cp).any(axis=(1, 2))
    out = cp.copy()
    i = 0
    while i < T:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < T and valid[j]:
            j += 1
        run = slice(i, j)
        L = j - i
        if L >= 5:
            w = min(window if window % 2 == 1 else window + 1, L if L % 2 == 1 else L - 1)
            if w >= 5:
                out[run] = savgol_filter(cp[run], w, 2, axis=0)
        i = j
    return out


# ============================================================
# Dataset persistence
# ============================================================
def _ds_paths(ds_dir: str):
    d = Path(ds_dir)
    return dict(root=d, images=d / "images", masks=d / "masks",
               lines=d / "lines", ckpt=d / "checkpoints",
               meta=d / "metadata.json", roi=d / "valve_roi.json",
               depth=d / "depth_config.json", band=d / "valve_band.npy",
               crop=d / "training_crop.json")


def _load_metadata(ds_dir: str) -> dict:
    p = _ds_paths(ds_dir)["meta"]
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"frames": [], "tif_path": None, "image_shape": None,
            "um_per_pixel": UM_PER_PIXEL, "mask_thickness_px": MASK_THICKNESS_PX}


def _save_metadata(ds_dir: str, meta: dict) -> None:
    p = _ds_paths(ds_dir)
    with open(p["meta"], "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Valve ROI (single source of truth: <dataset>/valve_roi.json)
# A user-drawn spatial prior used by annotate / train / infer / openness / figure.
# ============================================================
def save_valve_roi(ds_dir: str, polygon_yx, image_shape, margin: int = ROI_MARGIN) -> dict:
    """Persist a user-drawn valve ROI to <dataset>/valve_roi.json: polygon
    vertices (full-frame y,x), bbox (+margin, clipped), and the binary mask
    cropped to the bbox. Returns the roi dict."""
    from skimage.draw import polygon as sk_polygon
    poly = np.asarray(polygon_yx, float)
    H, W = int(image_shape[0]), int(image_shape[1])
    y0 = max(0, int(np.floor(poly[:, 0].min())) - margin)
    y1 = min(H, int(np.ceil(poly[:, 0].max())) + margin)
    x0 = max(0, int(np.floor(poly[:, 1].min())) - margin)
    x1 = min(W, int(np.ceil(poly[:, 1].max())) + margin)
    full = np.zeros((H, W), np.uint8)
    rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=(H, W))
    full[rr, cc] = 1
    roi = {"polygon_yx": [[float(y), float(x)] for y, x in poly],
           "bbox": [int(y0), int(x0), int(y1), int(x1)], "margin": int(margin),
           "image_shape": [H, W],
           "mask_crop": full[y0:y1, x0:x1].astype(int).tolist()}
    with open(_ds_paths(ds_dir)["roi"], "w") as f:
        json.dump(roi, f)
    print(f"[roi] SAVED valve ROI -> {_ds_paths(ds_dir)['roi']}  "
          f"bbox={roi['bbox']} ({len(roi['polygon_yx'])} vertices)")
    return roi


def load_valve_roi(ds_dir: str):
    """Load the valve ROI (or None). Logs which ROI is in use."""
    p = _ds_paths(ds_dir)["roi"]
    if not Path(p).exists():
        print(f"[roi] no valve ROI at {p} -> whole-frame behavior")
        return None
    try:
        roi = json.load(open(p))
        print(f"[roi] USING valve ROI: bbox={roi.get('bbox')} "
              f"({len(roi.get('polygon_yx', []))} vertices) from {p}")
        return roi
    except Exception as e:
        print(f"[roi] failed to load {p}: {e} -> whole-frame behavior")
        return None


def roi_bbox(roi) -> tuple:
    return tuple(int(v) for v in roi["bbox"])


def roi_full_mask(roi, shape) -> np.ndarray:
    """Rasterize the ROI polygon to a full-frame binary mask of `shape`."""
    from skimage.draw import polygon as sk_polygon
    H, W = int(shape[0]), int(shape[1])
    m = np.zeros((H, W), np.uint8)
    poly = np.asarray(roi["polygon_yx"], float)
    rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=(H, W))
    m[rr, cc] = 1
    return m


def roi_crop_mask(roi) -> np.ndarray:
    """Binary ROI mask cropped to the bbox (the model's spatial-prior channel)."""
    mc = roi.get("mask_crop")
    if mc is not None:
        return np.asarray(mc, np.uint8)
    y0, x0, y1, x1 = roi_bbox(roi)
    return roi_full_mask(roi, roi["image_shape"])[y0:y1, x0:x1]


# ============================================================
# Depth config (single source: <dataset>/depth_config.json)
# Records the chosen depth, interleave order, working single-depth stack, and the
# ventricle/atrium signal paths so EVERY step keys off the same depth + timeline.
# ============================================================
def save_depth_config(ds_dir: str, cfg: dict) -> dict:
    """Write the depth/V-A foundation config to <dataset>/depth_config.json."""
    with open(_ds_paths(ds_dir)["depth"], "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[depth] SAVED depth config -> {_ds_paths(ds_dir)['depth']}  "
          f"depth={cfg.get('depth')} order='{cfg.get('order')}'")
    return cfg


def load_depth_config(ds_dir: str):
    """Load the depth/V-A config (or None). Logs the depth + V/A path in use."""
    p = _ds_paths(ds_dir)["depth"]
    if not Path(p).exists():
        return None
    try:
        cfg = json.load(open(p))
        print(f"[depth] USING depth {cfg.get('depth')} (order '{cfg.get('order')}'); "
              f"V/A: {cfg.get('ventricle')} / {cfg.get('atrium')}")
        return cfg
    except Exception as e:
        print(f"[depth] failed to load {p}: {e}")
        return None


# ============================================================
# Auto valve BAND (single source of truth: <dataset>/valve_band.npy)
# Derived per depth from the temporal overlap of the red(V)/blue(A) regions; the
# AV-junction band that focuses U-Net training / line extraction / openness.
# ============================================================
def save_valve_band(ds_dir: str, mask: np.ndarray) -> str:
    """Persist the auto valve band (H,W bool) to <dataset>/valve_band.npy."""
    p = str(_ds_paths(ds_dir)["band"])
    np.save(p, np.asarray(mask, bool))
    print(f"[band] SAVED valve band -> {p}  ({int(np.asarray(mask, bool).sum())} px)")
    return p


def load_valve_band(ds_dir: str):
    """Load the auto valve band (H,W bool) or None."""
    p = _ds_paths(ds_dir)["band"]
    if not Path(p).exists():
        return None
    try:
        m = np.load(p).astype(bool)
        print(f"[band] USING valve band {p}  ({int(m.sum())} px)")
        return m
    except Exception as e:
        print(f"[band] failed to load {p}: {e}")
        return None


# ============================================================
# Square TRAINING CROP (single source of truth: <dataset>/training_crop.json)
# A square (1:1) region around the valve. Annotate / train / infer all operate on
# this crop at NATIVE resolution (display is enlarged; labels stay in crop px).
# ============================================================
def save_training_crop(ds_dir: str, bbox) -> str:
    """Persist the SQUARE training crop bbox = [y0, x0, y1, x1] (y1-y0 == x1-x0)."""
    y0, x0, y1, x1 = [int(round(v)) for v in bbox]
    if (y1 - y0) != (x1 - x0):                 # enforce square
        side = min(y1 - y0, x1 - x0)
        y1, x1 = y0 + side, x0 + side
    p = str(_ds_paths(ds_dir)["crop"])
    with open(p, "w") as f:
        json.dump({"bbox": [y0, x0, y1, x1], "side": int(y1 - y0)}, f, indent=2)
    print(f"[crop] SAVED training crop -> {p}  bbox=[{y0},{x0},{y1},{x1}] "
          f"side={y1 - y0}px")
    return p


def load_training_crop(ds_dir: str):
    """Load the square training-crop bbox [y0,x0,y1,x1] (enforced square), or None."""
    p = _ds_paths(ds_dir)["crop"]
    if not Path(p).exists():
        return None
    try:
        d = json.load(open(p))
        b = [int(v) for v in d["bbox"]]
        if (b[2] - b[0]) != (b[3] - b[1]):
            print(f"[crop] WARNING non-square crop {b}; ignoring."); return None
        return b
    except Exception as e:
        print(f"[crop] failed to load {p}: {e}")
        return None


def apply_training_crop(stack, ds_dir):
    """Crop a (T,H,W) (or (H,W)) array to the saved square training crop, if any.
    Returns (cropped, bbox|None). Logs the crop. Invalid/oversized crops are skipped."""
    crop = load_training_crop(ds_dir)
    if crop is None:
        return stack, None
    y0, x0, y1, x1 = crop
    H, W = stack.shape[-2:]
    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        print(f"[crop] WARNING crop {crop} out of bounds for {(H, W)}; using full frame.")
        return stack, None
    out = stack[..., y0:y1, x0:x1]
    print(f"[crop] TRAINING CROP active: bbox=[{y0},{x0},{y1},{x1}] -> "
          f"{out.shape[-2]}x{out.shape[-1]} (native crop coords)")
    return out, crop


def _frame_idx(rec) -> int:
    """Integer frame index from a record that may be a plain int OR a (legacy)
    dict like {'index': i, ...} / {'frame': i, ...}. NEVER hash/sort the dict."""
    if isinstance(rec, dict):
        return int(rec.get("frame", rec.get("index")))
    return int(rec)


def _frame_indices(meta: dict):
    """Sorted, de-duplicated list of INTEGER frame indices from meta['frames'].
    Records may be ints or legacy dicts; de-dup keys on the hashable index, never
    on the record itself."""
    seen = {}
    for rec in meta.get("frames", []):
        try:
            seen[_frame_idx(rec)] = True
        except Exception:
            pass
    return sorted(seen)


def _migrate_legacy_dataset(ds_dir: str, proc_stack: Optional[np.ndarray] = None) -> dict:
    """Convert a LEGACY dataset (PNG images under frames/, masks under
    masks/line1[/line2], dict frame-records keyed by 'index') to the CURRENT
    schema that train_unet/validate read: float32 npy image + union npy mask per
    frame, and meta['frames'] as a plain list of integer indices.

    Mask GEOMETRY is preserved from the saved masks (or polyline json). Images
    are RE-DERIVED from `proc_stack` (the dark-field-preprocessed stack) when
    given so the training domain matches inference; otherwise re-derived from the
    source TIFF, else read from the saved PNG. Idempotent and logged."""
    from PIL import Image
    p = _ds_paths(ds_dir)
    meta = _load_metadata(ds_dir)
    recs = meta.get("frames", [])
    idxs = _frame_indices(meta)
    if not idxs:
        meta["frames"] = []
        return meta
    legacy_records = any(isinstance(r, dict) for r in recs)
    need = legacy_records or any(
        not (p["images"] / f"frame_{i:04d}.npy").exists() or
        not (p["masks"] / f"frame_{i:04d}.npy").exists() for i in idxs)
    if not need:
        meta["frames"] = idxs
        return meta

    if proc_stack is None:                       # re-derive darkfield from source TIFF
        tif = meta.get("tif_path") or meta.get("source")
        if tif and os.path.exists(tif):
            try:
                import tifffile
                st = tifffile.imread(tif).astype(np.float32)
                if st.ndim == 2:
                    st = st[None]
                proc_stack = darkfield_input(st)
                print("[migrate] re-derived dark-field images from source TIFF.")
            except Exception as e:
                print(f"[migrate] could not load source TIFF ({e}); using saved PNGs.")

    p["images"].mkdir(parents=True, exist_ok=True)
    p["masks"].mkdir(parents=True, exist_ok=True)
    line1, line2 = p["masks"] / "line1", p["masks"] / "line2"
    migrated = 0
    for i in idxs:
        ni, nm = p["images"] / f"frame_{i:04d}.npy", p["masks"] / f"frame_{i:04d}.npy"
        if ni.exists() and nm.exists() and not legacy_records:
            continue
        img = None
        if proc_stack is not None and i < len(proc_stack):
            img = proc_stack[i].astype(np.float32)
        else:
            for src in (p["root"] / "frames" / f"frame_{i:05d}.png",
                        p["images"] / f"frame_{i:05d}.png",
                        p["images"] / f"frame_{i:04d}.png"):
                if src.exists():
                    img = np.asarray(Image.open(src)).astype(np.float32) / 255.0
                    break
        if img is None:
            print(f"[migrate] frame {i}: no source image, skipped.")
            continue
        m = np.zeros(img.shape[:2], np.uint8)
        got = False
        for sub in (line1, line2):
            for mp in (sub / f"frame_{i:05d}.png", sub / f"frame_{i:04d}.png"):
                if mp.exists():
                    m |= (np.asarray(Image.open(mp)) > 127).astype(np.uint8); got = True
        if not got:
            for jn in (p["lines"] / f"frame_{i:05d}.json", p["lines"] / f"frame_{i:04d}.json"):
                if jn.exists():
                    d = json.load(open(jn))
                    for key in ("leaflet1_yx", "leaflet2_yx", "points_yx"):
                        pts = d.get(key, [])
                        if len(pts) >= 2:
                            m |= rasterize_polyline(np.array(pts), img.shape[:2], MASK_THICKNESS_PX)
                    got = True
                    break
        np.save(ni, img); np.save(nm, m.astype(np.uint8)); migrated += 1
    meta["frames"] = idxs                        # normalize to integer indices
    _save_metadata(ds_dir, meta)
    print(f"[migrate] converted {migrated}/{len(idxs)} legacy annotations -> npy; "
          f"frames normalized to integer indices "
          f"(image domain: {'dark-field' if proc_stack is not None else 'saved PNG'}).")
    return meta


def _norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)


def darkfield_input(stack: np.ndarray) -> np.ndarray:
    """Dark-field preprocessing applied to the WHOLE stack, used for BOTH
    annotation and inference so the train/test input domains match exactly.
    Normalizes to [0,1], then runs valve_analysis.preprocess(darkfield=True)
    (temporal-min background, temporal denoise, bilateral, gamma+CLAHE). Falls
    back to a plain percentile norm if valve_analysis isn't importable."""
    s = np.asarray(stack, np.float32)
    lo, hi = np.percentile(s, [0.5, 99.5])
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)
    try:
        from valve_analysis import preprocess
        print("[darkfield] running preprocess(darkfield=True) on the stack ...")
        return np.clip(preprocess(s, darkfield=True).astype(np.float32), 0, 1)
    except Exception as e:
        print(f"[darkfield] preprocess unavailable ({e}); using percentile norm.")
        return s


# ============================================================
# Stage 1: annotate  (draw ONE leaflet polyline per frame)
# ============================================================
def suggest_annotation_frames(stack: np.ndarray, done=(), every: int = 30):
    """Propose frames to annotate, OVERSAMPLING the brief valve-OPENING events
    (frames near dark-field darkening peaks) plus a uniform spread of
    closed/intermediate frames, so the training set isn't dominated by the ~80%
    closed frames. Returns (ordered_frames, opening_peak_frames)."""
    T = stack.shape[0]
    done = set(int(d) for d in done)
    try:
        from valve_analysis import compute_valve_open_signal_from_darkening
        _, dk = compute_valve_open_signal_from_darkening(stack)
    except Exception as e:
        print(f"[annotate] darkening cue unavailable ({e}); uniform sampling.")
        dk = stack.reshape(T, -1).mean(1)
    z = (dk - dk.mean()) / (dk.std() + 1e-9)
    peaks, _ = find_peaks(z, prominence=0.5, distance=max(3, T // 40))
    ordered, seen = [], set()
    # openings first (priority): each peak +/- 2 frames
    for f in peaks:
        for d in (0, -1, 1, -2, 2):
            ff = int(f + d)
            if 0 <= ff < T and ff not in done and ff not in seen:
                seen.add(ff); ordered.append(ff)
    # then a uniform spread of (mostly closed) frames
    for ff in range(0, T, max(1, every)):
        if ff not in done and ff not in seen:
            seen.add(ff); ordered.append(ff)
    return ordered, np.asarray(peaks, int)


def annotate_frames(tif_path: str, ds_dir: str, every: int = 30,
                    mode: Optional[str] = None, ventricle_path: Optional[str] = None,
                    atrium_path: Optional[str] = None, fps: float = FPS_DEFAULT) -> None:
    """Resumable TWO-LEAFLET annotation tool (save-as-you-go).

    Draw BOTH leaflets per frame as separate labels (leaflet 1 = red,
    leaflet 2 = cyan), point-by-point, on the SAME dark-field-preprocessed image
    the model trains on. Controls:
        L-click            add a point to the ACTIVE leaflet
        1 / 2              select active leaflet 1 / 2
        u / backspace      undo last point (active leaflet)
        c                  clear the active leaflet
        enter              SAVE this frame and advance
        n / right          skip to next frame (no save)
        p / left           previous frame
        j                  jump to a frame index
        q                  quit (everything is already saved)

    On launch, if a dataset already exists you choose ADD (append) or FRESH
    (clear). Each saved frame immediately updates metadata.json so the GUI count
    is always correct. Saves per-frame image (.npy), union mask (.npy) and both
    polylines (.json) — the mask format matches what train_unet() loads.
    """
    import tifffile
    p = _ds_paths(ds_dir)
    for k in ("images", "masks", "lines", "ckpt"):
        p[k].mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(ds_dir)
    n_prior = len(_frame_indices(meta))          # integer-index count (records may be dicts)
    prior_shape = meta.get("image_shape")        # (H,W) the existing annotations were drawn on

    # ── ADD vs FRESH mode ──
    print(f"[annotate] dataset folder: {p['root'].resolve()}")
    if n_prior:
        if mode is None:
            try:
                import tkinter as _tk
                from tkinter import messagebox as _mb
                _r = _tk.Tk(); _r.withdraw()
                add = _mb.askyesno(
                    "Annotation mode",
                    f"{n_prior} frames already annotated in\n{p['root'].resolve()}\n\n"
                    f"YES  = ADD to existing (append, keep prior frames)\n"
                    f"NO   = START FRESH (archive the old set to a timestamped "
                    f"backup, then annotate a NEW set)")
                _r.destroy()
                mode = "add" if add else "fresh"
            except Exception:
                mode = "add"
        if mode == "fresh":
            # Archive (MOVE, never silently delete) ONLY the annotation artifacts
            # (images / masks / lines / metadata). PRESERVE the STEP-1 foundation
            # in place - depth_config.json, depth*_stack.tif, ventricle/atrium_
            # trace.npy, valve_roi.json - so a fresh annotation still has the
            # chosen-depth stack + V/A it is keyed to. (Moving the whole folder
            # used to sweep the foundation into the backup and then crash reading
            # the just-moved depth stack.)
            import shutil
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = Path(f"{str(p['root'])}_backup_{ts}")
            try:
                backup.mkdir(parents=True, exist_ok=True)
                for sub in ("images", "masks", "lines"):
                    if p[sub].exists():
                        shutil.move(str(p[sub]), str(backup / p[sub].name))
                if Path(p["meta"]).exists():
                    shutil.move(str(p["meta"]), str(backup / Path(p["meta"]).name))
                print(f"[annotate] MODE: START FRESH - archived {n_prior} annotated "
                      f"frames (images/masks/lines/metadata) to {backup.resolve()}; "
                      f"KEPT the STEP-1 foundation (stack / V-A / depth_config / ROI).")
            except Exception as e:
                print(f"[annotate] archive failed ({e}); deleting frame files in place.")
                for sub in ("images", "masks", "lines"):
                    for f in p[sub].glob("frame_*.*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
            for k in ("images", "masks", "lines", "ckpt"):   # recreate empty annot dirs
                p[k].mkdir(parents=True, exist_ok=True)
            meta = _load_metadata(ds_dir)         # metadata.json gone -> frames == []
            meta["frames"] = []
            _save_metadata(ds_dir, meta)          # count is now 0 on disk
            n_prior = 0
        else:
            print(f"[annotate] MODE: ADD to existing ({n_prior} prior frames). "
                  f"MERGE POLICY: keyed on integer frame index; re-annotating an "
                  f"existing frame overwrites that frame's files.")
    else:
        mode = "fresh"
        print("[annotate] MODE: START FRESH (empty dataset)")

    stack = tifffile.imread(tif_path).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[None]
    stack = darkfield_input(stack)            # same domain as train/infer
    # ── SQUARE TRAINING CROP (single source of truth): annotate on the cropped,
    #    enlarged image. imshow auto-enlarges the small crop to fill the axes, and
    #    clicks land in NATIVE crop pixels, so labels round-trip correctly. ──
    stack, train_crop = apply_training_crop(stack, ds_dir)
    if train_crop is not None:
        print(f"[annotate] annotating the TRAINING CROP (enlarged); leaflet labels "
              f"are saved in native crop pixels.")
    T, H, W = stack.shape
    # ── resolution-consistency guard ──
    # Existing annotations are tied to the resolution they were drawn on. If the
    # current depth stack has a DIFFERENT (H,W), appending to them silently mixes
    # incompatible coordinates -> the ROI crop lands on background -> the model
    # trains on empty masks (loss 0.0, valDice 0.0). Refuse clearly instead.
    if mode != "fresh" and prior_shape and tuple(prior_shape) != (int(H), int(W)):
        raise ValueError(
            f"Existing {n_prior} annotations were drawn on a {tuple(prior_shape)} "
            f"stack, but the chosen-depth stack is {(int(H), int(W))}. They are "
            f"incompatible (annotations would not line up; training would collapse "
            f"to empty masks). Re-run Annotate and choose START FRESH to annotate "
            f"this depth's stack, or re-run STEP 1 with the matching depth/source.")
    meta.update(tif_path=str(tif_path), image_shape=[int(H), int(W)],
                um_per_pixel=UM_PER_PIXEL, mask_thickness_px=MASK_THICKNESS_PX)
    # Migrate any legacy (PNG/dict) records to npy + integer indices, re-deriving
    # images from THIS dark-field stack so old + new annotations share one domain.
    if mode != "fresh":
        meta = _migrate_legacy_dataset(ds_dir, proc_stack=stack)
        meta.update(tif_path=str(tif_path), image_shape=[int(H), int(W)],
                    um_per_pixel=UM_PER_PIXEL, mask_thickness_px=MASK_THICKNESS_PX)
    done = set(_frame_indices(meta))          # set of INTS, never dict records

    suggested, peaks = suggest_annotation_frames(stack, done=done, every=every)
    if not suggested:
        suggested = [t for t in range(0, T, max(1, every))]
    print(f"[annotate] {len(suggested)} suggested frames "
          f"({len(peaks)} opening events oversampled); {len(done)} already done; "
          f"target {MIN_TRAINING_SAMPLES}.")
    set_pub_style()

    # auto valve BAND: constrain annotation to the AV-junction band + optional seed
    # (the deprecated separate valve ROI is gone - the band is the only focus region)
    band = load_valve_band(ds_dir)
    if band is not None and train_crop is not None and band.shape != (H, W):
        cy0, cx0, cy1, cx1 = train_crop          # crop the full-frame band to match
        if cy1 <= band.shape[0] and cx1 <= band.shape[1]:
            band = band[cy0:cy1, cx0:cx1]
    if band is not None and band.shape != (H, W):
        print(f"[annotate] WARNING band {band.shape} != frame {(H, W)}; ignoring.")
        band = None
    band_seed = None
    if band is not None:
        sk = skeletonize(band)
        if sk.any():
            try:
                bs = order_skeleton(sk)          # ordered medial line of the band
                band_seed = [(float(y), float(x)) for y, x in bs] if bs is not None and len(bs) >= 2 else None
            except Exception:
                band_seed = None
        print(f"[annotate] valve BAND active ({int(band.sum())} px); annotations "
              f"clipped to band" + ("; press 's' to seed the active leaflet with the "
              "band centerline" if band_seed else ""))

    # FULL range is navigable by default; the smart suggestions are kept ONLY as
    # optional guidance (highlighted + a "next suggested" shortcut), never a limit.
    suggested_set = set(int(s) for s in suggested)          # opening-oversampled refs
    peaks_set = set(int(p) for p in peaks)                  # opening-event frames
    all_frames = list(range(T))
    sugg_sorted = sorted(suggested_set)
    start_idx = sugg_sorted[0] if sugg_sorted else 0        # land on the 1st suggestion
    state = {"frames": all_frames, "all_frames": all_frames,
             "sugg_frames": (sugg_sorted or all_frames), "mode": "all",
             "idx": start_idx, "p1": [], "p2": [], "pc": [], "open": False,
             "active": 1}
    print(f"[annotate] FULL range navigable: {T} frames (0..{T-1}). "
          f"{len(suggested_set)} suggested (highlighted); press 'a' to toggle "
          f"all/suggested, 'g' = next suggested frame.")

    def _load_saved(t):
        jp = p["lines"] / f"frame_{t:04d}.json"
        if jp.exists():
            try:
                d = json.load(open(jp))
                return ([tuple(x) for x in d.get("leaflet1_yx", [])],
                        [tuple(x) for x in d.get("leaflet2_yx", d.get("points_yx", []))],
                        [tuple(x) for x in d.get("connector_yx", [])],
                        bool(d.get("open", False)))
            except Exception:
                pass
        return [], [], [], False

    # ── reference V/A signals (SELECTED depth, 901 frames) for the live cursor ──
    def _load_tr(pth):
        try:
            return np.load(pth).astype(float) if pth and os.path.exists(pth) else None
        except Exception:
            return None
    va_v = _load_tr(ventricle_path); va_a = _load_tr(atrium_path)
    va_vchg = va_achg = np.array([], int)
    if va_v is not None:
        try:
            va_vchg, _ = signal_max_change_points(va_v)
        except Exception:
            pass
    if va_a is not None:
        try:
            va_achg, _ = signal_max_change_points(va_a)
        except Exception:
            pass
    va_opens = np.array([], int)                  # previously-detected opens (context)
    for _op in ("unet_line_open_frames.npy", "valve_int_open_frames.npy"):
        if os.path.exists(_op):
            try:
                va_opens = np.load(_op).astype(int); break
            except Exception:
                pass
    have_va = (va_v is not None) or (va_a is not None)
    if have_va:
        print(f"[annotate] V/A reference loaded (V={va_v is not None}, A={va_a is not None}); "
              f"live current-frame cursor on the V/A graph.")

    def _mm(x):
        x = np.asarray(x, float); lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-9)

    def _fit(tr):                                 # normalize + length-match to T
        tr = np.asarray(tr, float)
        if len(tr) >= T:
            tr = tr[:T]
        else:
            tr = np.pad(tr, (0, T - len(tr)), "edge")
        return _mm(tr)

    from matplotlib.widgets import Button
    if have_va:
        fig = plt.figure(figsize=(8.6, 9.8))
        ax = fig.add_axes([0.05, 0.36, 0.90, 0.58])      # enlarged (crop) annotation view
        ax_va = fig.add_axes([0.08, 0.165, 0.87, 0.125])  # V/A reference strip (901 frames)
    else:
        fig = plt.figure(figsize=(7.8, 8.4))
        ax = fig.add_axes([0.04, 0.15, 0.92, 0.80])
        ax_va = None
    fig.canvas.manager.set_window_title(
        "Annotate 3 structures: leaflet 1 / leaflet 2 / CONNECTOR (+ OPEN/CLOSED)")

    # static V/A reference plot + live cursor / dots (updated in draw())
    # colours match the SOURCE marker (single source of truth): atrium=RED, ventricle=BLUE
    CV_, CA_ = CHAMBER_COLOR["ventricle"], CHAMBER_COLOR["atrium"]   # ventricle blue / atrium red
    va_disp = {"v": None, "a": None}
    va_cursor = va_dotv = va_dota = None
    if ax_va is not None:
        fr = np.arange(T)
        if va_v is not None:
            va_disp["v"] = _fit(va_v)
            ax_va.plot(fr, va_disp["v"], color=CV_, lw=0.8, label="ventricle")
        if va_a is not None:
            va_disp["a"] = _fit(va_a)
            ax_va.plot(fr, va_disp["a"], color=CA_, lw=0.8, label="atrium")
        # EXPECTED-OPEN windows = atrium PEAK -> just-before-TROUGH (emptying limb)
        if va_a is not None:
            try:
                _aw, _apk, _ = atrium_open_windows(
                    va_a[:T], fps, start_offset=ATRIUM_OPEN_START_OFFSET,
                    end_lead=ATRIUM_OPEN_END_LEAD)
                for k, (ws, we) in enumerate(_aw):
                    ax_va.axvspan(ws, we, color="#f4c542", alpha=0.28, zorder=0,
                                  label=("expected open (atrial emptying)" if k == 0 else None))
                for pp in _apk:                    # atrium maximum = window start
                    if 0 <= pp < T:
                        ax_va.plot([pp], [1.06], marker="v", color="#b8860b", ms=4,
                                   clip_on=False)
                print(f"[annotate] expected-open windows (atrium peak->trough): {_aw}")
            except Exception as _e:
                print(f"[annotate] atrium open-windows failed: {_e}")
        for cf in list(va_vchg) + list(va_achg):  # V/A max-change (light context)
            if 0 <= cf < T:
                ax_va.axvline(cf, color="0.55", lw=0.5, alpha=0.28)
        for of in va_opens:                        # DETECTED opens (distinct: green line)
            if 0 <= of < T:
                ax_va.axvline(of, color="#2ca02c", lw=1.1, alpha=0.85)
        ax_va.set_xlim(0, T - 1); ax_va.set_ylim(-0.05, 1.12)
        ax_va.set_xlabel("frame", fontsize=7); ax_va.set_ylabel("V/A (norm)", fontsize=7)
        ax_va.tick_params(labelsize=6)
        ax_va.legend(loc="upper right", frameon=False, fontsize=6, ncol=2)
        ax_va.set_title("ventricle (blue) / atrium (red) - selected depth.  GOLD = expected "
                        "open (atrium peak->just-before-trough); green = detected open; "
                        "black cursor = current frame", fontsize=6.2)
        va_cursor = ax_va.axvline(0, color="k", lw=1.4, alpha=0.9)   # distinct from red atrium
        if va_disp["v"] is not None:
            va_dotv, = ax_va.plot([0], [va_disp["v"][0]], "o", color=CV_, ms=4, mec="k", mew=0.3)
        if va_disp["a"] is not None:
            va_dota, = ax_va.plot([0], [va_disp["a"][0]], "o", color=CA_, ms=4, mec="k", mew=0.3)

    def _update_va_cursor(t):
        if ax_va is None:
            return
        tc = int(min(max(0, t), T - 1))
        va_cursor.set_xdata([tc, tc])
        if va_dotv is not None:
            va_dotv.set_data([tc], [va_disp["v"][tc]])
        if va_dota is not None:
            va_dota.set_data([tc], [va_disp["a"][tc]])

    def draw():
        ax.clear()
        t = state["frames"][state["idx"]]
        ax.imshow(stack[t], cmap="gray", vmin=0, vmax=1)
        if band is not None:                      # AV-junction band (annotate inside it)
            ax.imshow(np.where(band, 0.18, np.nan), cmap="cool", vmin=0, vmax=1, alpha=0.35)
            ax.contour(band, levels=[0.5], colors="#00e5ff", linewidths=0.8, alpha=0.9)
        for pts, c, lab in ((state["p1"], C_LEAFLET_A, 1), (state["p2"], C_LEAFLET_B, 2),
                            (state["pc"], CONNECTOR_COLOR, 3)):
            if pts:
                pa = np.array(pts)
                lw = 2.4 if state["active"] == lab else 1.1
                ls = "--" if lab == 3 else "-"        # connector dashed (detection line)
                ax.plot(pa[:, 1], pa[:, 0], ls + "o", color=c, ms=3, lw=lw)
        n_done = len(meta["frames"])
        opened = sum(1 for f in _frame_indices(meta)
                     if len(peaks) and np.min(np.abs(peaks - f)) <= 2)
        act_c = {1: C_LEAFLET_A, 2: C_LEAFLET_B, 3: CONNECTOR_COLOR}[state["active"]]
        act_name = {1: "LEAFLET 1", 2: "LEAFLET 2", 3: "CONNECTOR"}[state["active"]]
        # suggestion / opening-event guidance for the CURRENT frame (not a limit)
        tag = ""
        if t in peaks_set:
            tag = "  [OPENING EVENT - suggested]"
        elif t in suggested_set:
            tag = "  [suggested]"
        done_tag = "  (saved)" if t in set(_frame_indices(meta)) else ""
        ax.set_title(
            f"frame {t}  ({state['idx']+1}/{len(state['frames'])}, mode={state['mode']})"
            f"{tag}{done_tag}\n"
            f"draw leaflet 1/2 lines + the CONNECTING line joining them | "
            f"1/2/3 switch structure | o OPEN/CLOSED | u undo | c clear | s seed | "
            f"enter save | n/p/j nav | g next-sugg | a all/sugg | q quit   "
            f"saved {n_done}/{MIN_TRAINING_SAMPLES}", fontsize=7.0)
        # prominent ACTIVE-structure banner + OPEN/CLOSED judgment
        ax.text(0.30, 1.012, f"ACTIVE: {act_name}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=act_c)
        oc = state["open"]
        ax.text(0.78, 1.012, f"judgment: {'OPEN (connector broken)' if oc else 'CLOSED'}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=("#2ca02c" if oc else "#888888"))
        # timeline tick-strip of suggested frames along the bottom edge
        # (cyan = suggested, orange = opening event); red caret = current frame
        ss = np.asarray(state["sugg_frames"], float)
        if len(ss):
            xs = ss / max(1, T - 1) * (W - 1)
            ispk = np.array([int(s) in peaks_set for s in state["sugg_frames"]])
            ax.scatter(xs[~ispk], np.full((~ispk).sum(), H - 1), marker="|", s=28,
                       c="#00e5ff", clip_on=False, linewidths=0.8)
            if ispk.any():
                ax.scatter(xs[ispk], np.full(ispk.sum(), H - 1), marker="|", s=44,
                           c="#ff7f0e", clip_on=False, linewidths=1.1)
        ax.scatter([t / max(1, T - 1) * (W - 1)], [H - 1], marker="^", s=30,
                   c="#ff2d2d", clip_on=False)
        ax.axis("off")
        _update_va_cursor(t)                      # live current-frame cursor on the V/A graph
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        key = {1: "p1", 2: "p2", 3: "pc"}[state["active"]]
        if ev.button == 1:
            state[key].append((ev.ydata, ev.xdata))
        elif ev.button == 3 and state[key]:
            state[key].pop()
        draw()

    def save_current():
        t = state["frames"][state["idx"]]
        if len(state["p1"]) < 2 and len(state["p2"]) < 2:
            print(f"  frame {t}: need >=2 points on at least one leaflet."); return False
        if len(state["p1"]) < 2 or len(state["p2"]) < 2:
            print(f"  frame {t}: NOTE only one leaflet drawn - representation needs BOTH "
                  f"leaflet 1 and leaflet 2 labeled (incl. closed frames).")
        is_open = bool(state["open"])
        if not is_open and len(state["pc"]) < 2:
            print(f"  frame {t}: NOTE CLOSED frame with NO connecting line - draw the "
                  f"connector (key 3) joining the leaflets, or toggle OPEN (key o).")
        # THREE-channel mask: ch0=leaflet1, ch1=leaflet2, ch2=CONNECTOR, clipped to band
        m = np.zeros((N_STRUCT, H, W), np.uint8)
        if len(state["p1"]) >= 2:
            m[0] = rasterize_polyline(np.array(state["p1"]), (H, W), MASK_THICKNESS_PX)
        if len(state["p2"]) >= 2:
            m[1] = rasterize_polyline(np.array(state["p2"]), (H, W), MASK_THICKNESS_PX)
        if len(state["pc"]) >= 2:                     # connector (broken/absent => open)
            m[CONNECTOR_CH] = rasterize_polyline(np.array(state["pc"]), (H, W), MASK_THICKNESS_PX)
        if band is not None:
            m = (m & band[None]).astype(np.uint8)
        np.save(p["images"] / f"frame_{t:04d}.npy", stack[t].astype(np.float32))
        np.save(p["masks"] / f"frame_{t:04d}.npy", m.astype(np.uint8))
        with open(p["lines"] / f"frame_{t:04d}.json", "w") as f:
            json.dump({"frame": int(t), "leaflet1_yx": state["p1"],
                       "leaflet2_yx": state["p2"], "connector_yx": state["pc"],
                       "open": is_open}, f)
        if t not in meta["frames"]:
            meta["frames"].append(int(t))
        _save_metadata(ds_dir, meta)                 # save-as-you-go
        print(f"  saved frame {t}  [{'OPEN' if is_open else 'CLOSED'}, "
              f"connector {len(state['pc'])}pts]  ({len(meta['frames'])} total)")
        return True

    def goto(idx):
        state["idx"] = max(0, min(idx, len(state["frames"]) - 1))
        t = state["frames"][state["idx"]]
        state["p1"], state["p2"], state["pc"], state["open"] = _load_saved(t)  # resume
        set_active(state["active"] if state["active"] in (1, 2, 3) else 1)     # redraw

    def on_key(ev):
        if ev.key == "enter":
            if save_current() and state["idx"] < len(state["frames"]) - 1:
                goto(state["idx"] + 1)
            else:
                draw()
        elif ev.key in ("n", "right"):
            goto(state["idx"] + 1)
        elif ev.key in ("p", "left"):
            goto(state["idx"] - 1)
        elif ev.key in ("1", "2", "3"):
            set_active(int(ev.key))
        elif ev.key in ("o", " "):                    # toggle my OPEN/CLOSED judgment
            state["open"] = not state["open"]
            print(f"  frame {state['frames'][state['idx']]}: judgment = "
                  f"{'OPEN' if state['open'] else 'CLOSED'}")
            draw()
        elif ev.key in ("u", "backspace"):
            key = {1: "p1", 2: "p2", 3: "pc"}[state["active"]]
            if state[key]:
                state[key].pop(); draw()
        elif ev.key == "c":
            state[{1: "p1", 2: "p2", 3: "pc"}[state["active"]]] = []; draw()
        elif ev.key == "s":                          # seed active structure w/ band centerline
            if band_seed:
                step = max(1, len(band_seed) // 12)
                state[{1: "p1", 2: "p2", 3: "pc"}[state["active"]]] = band_seed[::step]
                print("  seeded active structure with the band centerline (edit as needed)")
                draw()
            else:
                print("  no band seed available")
        elif ev.key == "j":
            try:
                import tkinter as _tk
                from tkinter import simpledialog as _sd
                _r = _tk.Tk(); _r.withdraw()
                f = _sd.askinteger("Jump", f"frame index 0..{T-1} (ANY frame)")
                _r.destroy()
                if f is not None:
                    f = int(np.clip(f, 0, T - 1))
                    if f not in state["frames"]:        # reachable even in 'suggested' mode
                        state["frames"].append(f); state["frames"].sort()
                    goto(state["frames"].index(f))
            except Exception as e:
                print(f"  jump failed: {e}")
        elif ev.key == "g":                              # jump to the NEXT suggested frame
            cur = state["frames"][state["idx"]]
            sg = state["sugg_frames"]
            nxt = [s for s in sg if s > cur]
            target = nxt[0] if nxt else (sg[0] if sg else cur)
            if target not in state["frames"]:
                state["frames"].append(target); state["frames"].sort()
            goto(state["frames"].index(target))
            print(f"  -> next suggested frame {target}")
        elif ev.key == "a":                              # toggle ALL <-> suggested-only
            cur = state["frames"][state["idx"]]
            if state["mode"] == "all":
                state["mode"] = "sugg"; state["frames"] = list(state["sugg_frames"])
            else:
                state["mode"] = "all"; state["frames"] = list(state["all_frames"])
            fr = state["frames"]
            near = min(range(len(fr)), key=lambda k: abs(fr[k] - cur)) if fr else 0
            print(f"  mode = {state['mode']} ({len(fr)} frames navigable)")
            goto(near)
        elif ev.key == "q":
            plt.close(fig)

    # ── ACTIVE-structure selector: leaflet1 / leaflet2 / CONNECTOR + OPEN toggle ──
    ax_b1 = fig.add_axes([0.06, 0.04, 0.21, 0.07])
    ax_b2 = fig.add_axes([0.28, 0.04, 0.21, 0.07])
    ax_b3 = fig.add_axes([0.50, 0.04, 0.23, 0.07])
    ax_bo = fig.add_axes([0.75, 0.04, 0.19, 0.07])
    b1 = Button(ax_b1, "Leaflet 1 (1)", color=C_LEAFLET_A, hovercolor="#ffb3a0")
    b2 = Button(ax_b2, "Leaflet 2 (2)", color="0.85", hovercolor="#a0d8ff")
    b3 = Button(ax_b3, "Connector (3)", color="0.85", hovercolor="#ffe9a0")
    bo = Button(ax_bo, "OPEN/CLOSED (o)", color="0.85", hovercolor="#bdf0bd")
    _INACT = "0.85"

    def set_active(lab):
        """Switch the ACTIVE structure (1=leaflet1, 2=leaflet2, 3=connector).
        Highlights the active button; NEVER touches the other structures' points."""
        state["active"] = int(lab)
        b1.color = C_LEAFLET_A if lab == 1 else _INACT
        b2.color = C_LEAFLET_B if lab == 2 else _INACT
        b3.color = CONNECTOR_COLOR if lab == 3 else _INACT
        b1.ax.set_facecolor(b1.color); b2.ax.set_facecolor(b2.color)
        b3.ax.set_facecolor(b3.color)
        draw()

    def toggle_open(_e):
        state["open"] = not state["open"]
        print(f"  frame {state['frames'][state['idx']]}: judgment = "
              f"{'OPEN' if state['open'] else 'CLOSED'}")
        draw()

    b1.on_clicked(lambda _e: set_active(1))
    b2.on_clicked(lambda _e: set_active(2))
    b3.on_clicked(lambda _e: set_active(3))
    bo.on_clicked(toggle_open)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    # keep widget refs alive (else matplotlib GCs the buttons and they go dead)
    fig._keep_alive = [b1, b2, b3, bo, set_active, toggle_open]
    goto(state["idx"])                                # land on the first suggested frame
    plt.show()

    n_done = len(_frame_indices(meta))
    opened = sum(1 for f in _frame_indices(meta)
                 if len(peaks) and np.min(np.abs(peaks - f)) <= 2)
    print(f"[annotate] session end: {n_done}/{MIN_TRAINING_SAMPLES} frames annotated, "
          f"{opened} opening events covered (of {len(peaks)} detected).")


# ============================================================
# Stage 2: train
# ============================================================
def _center_fit(a: np.ndarray, shape: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    """Center-crop or center-pad ``a`` to ``shape``."""
    H, W = shape; h, w = a.shape
    out = np.full(shape, fill, dtype=a.dtype)
    y0s, x0s = max(0, (h - H) // 2), max(0, (w - W) // 2)
    y0d, x0d = max(0, (H - h) // 2), max(0, (W - w) // 2)
    hh, ww = min(H, h), min(W, w)
    out[y0d:y0d + hh, x0d:x0d + ww] = a[y0s:y0s + hh, x0s:x0s + ww]
    return out


def _elastic(img, mask, rng, alpha, sigma):
    """Elastic deformation (Simard 2003): smoothed random displacement fields."""
    H, W = img.shape
    dy = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1, sigma) * alpha
    dx = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([yy + dy, xx + dx])
    im2 = ndimage.map_coordinates(img, coords, order=1, mode="reflect")
    mk2 = ndimage.map_coordinates(mask.astype(float), coords, order=0, mode="constant")
    return im2.astype(np.float32), (mk2 > 0.5).astype(np.uint8)


def _augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, roi=None):
    """Low-SNR-aware, label-preserving augmentation (numpy/scipy only). The
    optional ``roi`` spatial-prior channel is transformed by the SAME geometric
    ops (flip/rotate/scale/elastic) as the mask so it stays aligned with the
    image; intensity/noise ops apply to the image only. Returns (img, mask) or
    (img, mask, roi)."""
    H, W = img.shape
    has_roi = roi is not None
    if not has_roi:
        roi = np.zeros((H, W), np.float32)

    if rng.random() < 0.5:
        img, mask, roi = img[:, ::-1].copy(), mask[:, ::-1].copy(), roi[:, ::-1].copy()
    if rng.random() < 0.5:
        img, mask, roi = img[::-1].copy(), mask[::-1].copy(), roi[::-1].copy()
    if rng.random() < 0.7:                        # rotation
        ang = rng.uniform(-12, 12)
        img = ndimage.rotate(img, ang, reshape=False, order=1, mode="reflect")
        mask = (ndimage.rotate(mask.astype(float), ang, reshape=False, order=0,
                               mode="constant") > 0.5).astype(np.uint8)
        roi = (ndimage.rotate(roi.astype(float), ang, reshape=False, order=0,
                              mode="constant") > 0.5).astype(np.float32)
    if rng.random() < 0.5:                        # random scale (zoom) + refit
        z = rng.uniform(0.85, 1.18)
        img = _center_fit(ndimage.zoom(img, z, order=1), (H, W), 0.0)
        mask = (_center_fit(ndimage.zoom(mask.astype(float), z, order=0), (H, W), 0.0) > 0.5).astype(np.uint8)
        roi = (_center_fit(ndimage.zoom(roi.astype(float), z, order=0), (H, W), 0.0) > 0.5).astype(np.float32)
    if rng.random() < 0.4:                        # elastic deformation
        img, mask, roi = _elastic_roi(img, mask, roi, rng, alpha=rng.uniform(8, 18),
                                      sigma=rng.uniform(4, 7))
    if rng.random() < 0.6:                        # gamma / intensity jitter
        g = rng.uniform(0.7, 1.4)
        img = np.clip(img, 0, 1) ** g
        img = np.clip(img * rng.uniform(0.8, 1.2), 0, 1)
    if rng.random() < 0.3:                        # mild blur
        img = ndimage.gaussian_filter(img, sigma=rng.uniform(0.4, 0.9))
    if rng.random() < 0.5:                        # Poisson (shot) noise, ~14-bit
        peak = rng.uniform(300, 1500)
        img = np.clip(rng.poisson(np.clip(img, 0, 1) * peak) / peak, 0, 1)
    if rng.random() < 0.4:                        # additive read noise
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    if has_roi:
        return img.astype(np.float32), mask.astype(np.uint8), roi.astype(np.float32)
    return img.astype(np.float32), mask.astype(np.uint8)


def _elastic_roi(img, mask, roi, rng, alpha, sigma):
    """Elastic deformation applied jointly to image, mask, and ROI prior."""
    H, W = img.shape
    dy = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1, sigma) * alpha
    dx = ndimage.gaussian_filter(rng.random((H, W)) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([yy + dy, xx + dx])
    im2 = ndimage.map_coordinates(img, coords, order=1, mode="reflect")
    mk2 = ndimage.map_coordinates(mask.astype(float), coords, order=0, mode="constant")
    rm2 = ndimage.map_coordinates(roi.astype(float), coords, order=0, mode="constant")
    return (im2.astype(np.float32), (mk2 > 0.5).astype(np.uint8),
            (rm2 > 0.5).astype(np.float32))


def annotated_frame_info(ds_dir: str):
    """For the training-frame selector: list of {idx, has_l1, has_l2, has_conn,
    open} for every annotated frame (read from lines/*.json)."""
    p = _ds_paths(ds_dir)
    meta = _load_metadata(ds_dir)
    out = []
    for t in _frame_indices(meta):
        a = b = c = 0; op = False
        jp = p["lines"] / f"frame_{t:04d}.json"
        if jp.exists():
            try:
                d = json.load(open(jp))
                a = len(d.get("leaflet1_yx", []))
                b = len(d.get("leaflet2_yx", d.get("points_yx", [])))
                c = len(d.get("connector_yx", []))
                op = bool(d.get("open", False))
            except Exception:
                pass
        out.append(dict(idx=int(t), has_l1=a >= 2, has_l2=b >= 2,
                        has_conn=c >= 2, open=op))
    return out


def train_unet(ds_dir: str, epochs: int = 100, batch: Optional[int] = None,
               lr: float = TRAIN_LR, seed: int = 0, progress_cb=None,
               train_frames=None) -> str:
    """Train the attention U-Net with the thin-structure (BCE+Dice+clDice) loss.
    GPU-accelerated (mixed precision when CUDA is available). Saves the best
    checkpoint (by val clDice) and a training-curve figure. Returns the ckpt path.

    ``train_frames``: explicit list of annotated frame indices to train on (the
    SINGLE SOURCE OF TRUTH for the training set). None -> all annotated frames.
    The selection is logged + persisted to metadata.json['training_frames']; the
    validation split is taken from WITHIN the selected set.

    progress_cb(i, n, msg) is called once per epoch for GUI status updates.
    """
    from tqdm import tqdm
    import torch
    from torch.utils.data import Dataset, DataLoader

    p = _ds_paths(ds_dir)
    meta = _migrate_legacy_dataset(ds_dir)        # legacy PNG/dict -> npy + int frames
    all_frames = _frame_indices(meta)             # sorted unique INTEGER indices
    if train_frames is not None:                  # SELECTED subset only
        avail = set(all_frames)
        sel = sorted(set(int(t) for t in train_frames if int(t) in avail))
        missing = sorted(set(int(t) for t in train_frames) - avail)
        if missing:
            print(f"[train] WARNING {len(missing)} selected frames are not annotated "
                  f"and are ignored: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        frames = sel
        print(f"[train] training on {len(frames)} SELECTED frames of "
              f"{len(all_frames)} annotated: {frames}")
    else:
        frames = all_frames
        print(f"[train] training on ALL {len(frames)} annotated frames")
    if len(frames) < 4:
        print(f"[train] only {len(frames)} training frames - need a handful more.")
        return ""
    # ── single training per dataset folder (no concurrent clobber / GPU starve) ──
    if _acquire_training_lock(ds_dir) is None:
        raise RuntimeError(
            "Another training is already running on this dataset folder. Close the "
            "other pipeline window (or wait for it to finish) and re-run Train. "
            "Running two trainings on the same folder corrupts the checkpoint and "
            "starves the GPU.")
    meta["training_frames"] = list(frames)        # PERSIST the selection (reproducible)
    _save_metadata(ds_dir, meta)
    print(f"[train] persisted training_frames -> metadata.json ({len(frames)} frames)")
    imgs = [np.load(p["images"] / f"frame_{t:04d}.npy") for t in frames]
    shape = imgs[0].shape

    _dcfg = load_depth_config(ds_dir)             # record the training depth in the ckpt
    train_depth = _dcfg.get("depth") if _dcfg else None
    print(f"[train] training depth = {train_depth}")

    roi = None
    in_ch = 1
    out_ch = N_STRUCT                             # 3 multi-label sigmoid: leaflet1/leaflet2/CONNECTOR
    train_crop = load_training_crop(ds_dir)       # square crop the labels were drawn on
    if train_crop is not None:
        print(f"[train] TRAINING CROP active: bbox={train_crop} (crop-native {shape})")
    band = load_valve_band(ds_dir)
    if band is not None and train_crop is not None and tuple(band.shape) != tuple(shape):
        cy0, cx0, cy1, cx1 = train_crop           # crop the full-frame band to match
        if cy1 <= band.shape[0] and cx1 <= band.shape[1]:
            band = band[cy0:cy1, cx0:cx1]
    if band is not None and tuple(band.shape) != tuple(shape):
        print(f"[train] WARNING band shape {band.shape} != frame {shape}; ignoring band.")
        band = None

    # ── THREE structures per frame (multi-label): rasterize LEAFLET 1, LEAFLET 2,
    #    and the CONNECTING line SEPARATELY from lines/frame_*.json, each clipped to
    #    the band. The connector is the DETECTION structure (continuous=closed,
    #    broken/absent=open); leaflets are the representation. Also read the
    #    per-frame OPEN judgment ('open'). Input = 1-channel raw. ──
    def _structures(t):
        m1 = np.zeros(shape, np.uint8); m2 = np.zeros(shape, np.uint8)
        mc = np.zeros(shape, np.uint8); is_open = False
        jp = p["lines"] / f"frame_{t:04d}.json"
        if jp.exists():
            d = json.load(open(jp))
            a = d.get("leaflet1_yx", [])
            b = d.get("leaflet2_yx", d.get("points_yx", []))
            c = d.get("connector_yx", [])
            is_open = bool(d.get("open", False))
            if len(a) >= 2:
                m1 = rasterize_polyline(np.array(a), shape, MASK_THICKNESS_PX)
            if len(b) >= 2:
                m2 = rasterize_polyline(np.array(b), shape, MASK_THICKNESS_PX)
            if len(c) >= 2:
                mc = rasterize_polyline(np.array(c), shape, MASK_THICKNESS_PX)
        else:                                     # fallback: legacy single mask -> ch1
            mp = p["masks"] / f"frame_{t:04d}.npy"
            if mp.exists():
                mm = np.load(mp)
                m1 = (mm[0] if mm.ndim == 3 else mm).astype(np.uint8)
        if band is not None:
            m1 = (m1 & band).astype(np.uint8); m2 = (m2 & band).astype(np.uint8)
            mc = (mc & band).astype(np.uint8)
        return m1, m2, mc, is_open

    masks1, masks2, masksc, opens = [], [], [], []
    for t in frames:
        a, b, c, op = _structures(t)
        masks1.append(a); masks2.append(b); masksc.append(c); opens.append(op)
    opens = np.asarray(opens, bool)
    n_open = int(opens.sum()); n_closed = len(opens) - n_open
    print(f"[train] 3 multi-label structures (leaflet1/leaflet2/CONNECTOR) sigmoid "
          f"targets + boundary-weight map, 1-channel raw input"
          + (f"; BAND focus ({int(band.sum())} px)" if band is not None else
             " (no band -> whole-frame)"))
    print(f"[train] CLASS BALANCE: {n_open} OPEN (connector broken) vs {n_closed} "
          f"CLOSED frames; OPEN frames oversampled x{1 + OPEN_OVERSAMPLE}.")

    # ── guards: leaflets must be drawn; connector present in CLOSED frames. ──
    fg1 = sum(int(m.sum()) > 0 for m in masks1)
    fg2 = sum(int(m.sum()) > 0 for m in masks2)
    if fg1 < max(2, len(frames) // 10) or fg2 < max(2, len(frames) // 10):
        raise ValueError(
            f"Per-leaflet foreground too sparse: leaflet1 non-empty {fg1}/{len(frames)}, "
            f"leaflet2 {fg2}/{len(frames)}. Both leaflets must be annotated (incl. "
            f"closed frames, where they touch). Re-annotate within the band (START "
            f"FRESH); the band must match the chosen-depth stack.")
    fgc_closed = sum(int(masksc[i].sum()) > 0 for i in range(len(frames)) if not opens[i])
    fgc = sum(int(m.sum()) > 0 for m in masksc)
    if fgc < max(2, len(frames) // 8):
        raise ValueError(
            f"CONNECTING-line foreground too sparse: connector non-empty in only "
            f"{fgc}/{len(frames)} frames. Draw the connecting line joining the two "
            f"leaflet lines in CLOSED frames (and break/omit it in OPEN frames). "
            f"Re-annotate (START FRESH) including the connector + OPEN/CLOSED toggle.")
    print(f"[train] leaflet1 {fg1}/{len(frames)}, leaflet2 {fg2}/{len(frames)}, "
          f"connector {fgc}/{len(frames)} (closed-frame connector {fgc_closed}/{n_closed}).")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    n_val = max(1, int(round(TRAIN_VAL_SPLIT * len(frames))))
    val_idx, tr_idx = list(idx[:n_val].tolist()), list(idx[n_val:].tolist())
    # CLASS IMBALANCE: oversample the rare OPEN (connector-broken) training frames.
    tr_open = [j for j in tr_idx if opens[j]]
    tr_idx_os = list(tr_idx) + tr_open * OPEN_OVERSAMPLE
    print(f"[train] train set {len(tr_idx)} frames ({len(tr_open)} open) -> "
          f"{len(tr_idx_os)} after open-oversampling; val {len(val_idx)} frames.")

    class DS(Dataset):
        def __init__(self, indices, train):
            self.indices, self.train = list(indices), train

        def __len__(self):
            return len(self.indices) * (8 if self.train else 1)

        def __getitem__(self, i):
            j = self.indices[i % len(self.indices)]
            im = imgs[j].copy()
            m1, m2, mc = masks1[j].copy(), masks2[j].copy(), masksc[j].copy()
            if self.train:
                # all THREE structures ride the SAME geometric transform as the image
                im, (m1, m2, mc) = _augment_multi(
                    im, [m1.astype(np.float32), m2.astype(np.float32),
                         mc.astype(np.float32)], rng)
            # multi-label (3,H,W) float target + boundary-weight map AFTER augmentation
            target, wmap = structures_label_and_weight(
                np.asarray(m1) > 0.5, np.asarray(m2) > 0.5, np.asarray(mc) > 0.5)
            x = im[None].astype(np.float32)
            return (torch.from_numpy(x), torch.from_numpy(target),
                    torch.from_numpy(wmap))

    dev, is_cuda = device_banner("train")
    if batch is None:
        batch = TRAIN_BATCH_GPU if is_cuda else TRAIN_BATCH
    net = build_unet(in_ch=in_ch, out_ch=out_ch).to(dev)
    loss_fn = make_loss_multilabel()              # multi-label sigmoid (connector free to break)
    import torch.nn.functional as _F

    def _pad_lw(lab, wmap, hp, wp):               # pad label/weight to (hp,wp), neutral
        pl = (0, wp - lab.shape[-1], 0, hp - lab.shape[-2])
        return _F.pad(lab, pl, value=0), _F.pad(wmap, pl, value=0.0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # In-memory tiny dataset -> num_workers=0 (the local Dataset isn't picklable,
    # and the augmentation is cheap); pin_memory speeds host->GPU copies.
    tr = DataLoader(DS(tr_idx_os, True), batch_size=batch, shuffle=True,
                    pin_memory=is_cuda, num_workers=0)
    va = DataLoader(DS(sorted(val_idx), False), batch_size=1, pin_memory=is_cuda)

    # mixed precision (no-op on CPU)
    try:
        from torch.amp import autocast as _ac, GradScaler as _GS
        amp_ctx = lambda: _ac(device_type=dev.type, enabled=is_cuda)
        scaler = _GS(dev.type, enabled=is_cuda)
    except Exception:
        from torch.cuda.amp import autocast as _ac, GradScaler as _GS
        amp_ctx = lambda: _ac(enabled=is_cuda)
        scaler = _GS(enabled=is_cuda)

    hist = {"train": [], "val": [], "val_cldice": []}
    best = -math.inf                              # best by val per-leaflet clDice (higher better)
    ds_w = 0.4                                    # deep-supervision auxiliary weight
    p["ckpt"].mkdir(parents=True, exist_ok=True)
    ckpt_path = str(p["ckpt"] / "best_leaflet_unet.pth")
    print(f"[train] device={dev.type}  batch={batch}  train={len(tr_idx)} "
          f"val={len(val_idx)}  epochs={epochs}")
    sys.stdout.flush()
    t_start = time.time()

    # PERSISTENT epoch progress bar; live metrics go in its postfix (never a bare
    # print() inside the loop, which would erase/scramble the bar).
    epbar = tqdm(range(epochs), desc="train", unit="ep", dynamic_ncols=True)
    for ep in epbar:
        ep_t0 = time.time()
        net.train(); tl = 0.0
        n_batches = len(tr)
        for bi, (x, lab, wmap) in enumerate(tqdm(tr, desc=f"epoch {ep + 1}/{epochs}",
                                                 leave=False, unit="batch",
                                                 dynamic_ncols=True)):
            x = x.to(dev, non_blocking=is_cuda)
            lab = lab.to(dev, non_blocking=is_cuda); wmap = wmap.to(dev, non_blocking=is_cuda)
            x, (h, w) = _pad_to_multiple(x)
            hp, wp = x.shape[-2:]
            lab_p, w_p = _pad_lw(lab, wmap, hp, wp)
            opt.zero_grad()
            with amp_ctx():
                main, auxs = net(x, deep=True)        # deep supervision; (B,3,H,W)
                loss = loss_fn(main, lab_p, w_p)      # boundary-weighted CE + Dice + clDice
                if auxs:
                    loss = loss + ds_w * sum(loss_fn(a, lab_p, w_p) for a in auxs) / len(auxs)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += float(loss.detach())
            # per-batch HEARTBEAT: keep the GUI status line live + the window
            # responsive DURING the epoch (otherwise it looks frozen between epochs).
            if progress_cb and (bi % 4 == 0 or bi == n_batches - 1):
                progress_cb(ep + 1, epochs,
                            f"epoch {ep + 1}/{epochs}  batch {bi + 1}/{n_batches}  "
                            f"[{dev.type}] training...")
        sched.step()
        net.eval(); vl = 0.0; vcl = 0.0
        with torch.no_grad():
            for x, lab, wmap in va:
                x = x.to(dev, non_blocking=is_cuda)
                lab = lab.to(dev, non_blocking=is_cuda); wmap = wmap.to(dev, non_blocking=is_cuda)
                x, _ = _pad_to_multiple(x)
                hp, wp = x.shape[-2:]
                lab_p, w_p = _pad_lw(lab, wmap, hp, wp)
                lo = net(x)
                vl += float(loss_fn(lo, lab_p, w_p))
                vcl += loss_fn.leaflet_score(lo, lab_p)
        tl /= max(1, len(tr)); vl /= max(1, len(va)); vcl /= max(1, len(va))
        hist["train"].append(tl); hist["val"].append(vl); hist["val_cldice"].append(vcl)
        if vcl > best:                           # select best by val per-leaflet clDice
            best = vcl
            torch.save({"state_dict": net.state_dict(),
                        "config": {"base": UNET_BASE_CH, "depth": UNET_DEPTH,
                                   "in_ch": in_ch, "out_ch": out_ch, "softmax": False,
                                   "structures": list(STRUCT_NAMES)},
                        "roi": roi, "train_crop": train_crop,
                        "band": (band.astype(bool) if band is not None else None),
                        "valve_depth": train_depth, "meta": meta,
                        "val_loss": vl, "val_cldice": vcl, "epoch": ep}, ckpt_path)
        dt = time.time() - ep_t0
        eta = (time.time() - t_start) / (ep + 1) * (epochs - ep - 1)
        # live console bar: per-epoch metrics ride the PERSISTENT epoch bar's
        # postfix, so the bar never disappears mid-training.
        epbar.set_postfix_str(
            f"loss {tl:.4f} vDice/clDice {vl:.3f}/{vcl:.3f} best {best:.3f} "
            f"{dt:.1f}s/ep ETA {eta:.0f}s [{dev.type}]")
        msg = (f"epoch {ep + 1}/{epochs}  loss {tl:.4f}  valDice/clDice "
               f"{vl:.3f}/{vcl:.3f}  best {best:.3f}  {dt:.1f}s/ep  ETA {eta:.0f}s  "
               f"[{dev.type}]")
        if ep % 5 == 0 or ep == epochs - 1:
            epbar.write("  " + msg)               # milestone line, won't break the bar
        if progress_cb:                           # GUI status line, EVERY epoch
            progress_cb(ep + 1, epochs, msg)
        sys.stdout.flush()
    epbar.close()

    # training curve figure
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(hist["train"], color=C_MUTED, label="train loss")
    ax.plot(hist["val"], color=C_LEAFLET_A, label="val loss")
    ax2 = ax.twinx()
    ax2.plot(hist["val_cldice"], color=C_LEAFLET_B, label="val clDice")
    ax2.set_ylabel("val clDice")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(frameon=False, loc="upper right")
    save_vector(fig, str(p["ckpt"] / "training_curve.png"))
    plt.close(fig)
    print(f"[train] best val clDice {best:.3f} -> {ckpt_path}")
    return ckpt_path


# ============================================================
# Inference
# ============================================================
def model_valve_depth(model_path: str):
    """Cheap read of the depth the model was trained on (or None) - for the
    depth-mismatch guard, without building the network."""
    import torch
    try:
        return torch.load(model_path, map_location="cpu").get("valve_depth")
    except Exception:
        return None


def load_model(model_path: str):
    """Returns (net, meta, roi, in_ch, band, out_ch, structures). ``structures`` is
    the list of channel names for the CONNECTING-LINE multi-label model
    (['leaflet1','leaflet2','connector']) or None for the legacy softmax model.
    ``band`` is the auto valve band (or None); ``roi`` is a legacy ROI (or None)."""
    import torch
    ck = torch.load(model_path, map_location="cpu")
    cfg = ck.get("config", {"base": UNET_BASE_CH, "depth": UNET_DEPTH, "in_ch": 1})
    in_ch = int(cfg.get("in_ch", 1))
    out_ch = int(cfg.get("out_ch", 1))
    structures = cfg.get("structures")            # None unless the connector model
    net = build_unet(in_ch=in_ch, base=cfg["base"], depth=cfg["depth"], out_ch=out_ch)
    try:
        net.load_state_dict(ck["state_dict"])
    except Exception as e:
        raise RuntimeError(
            "This checkpoint does not match the current U-Net architecture / output "
            "channels - it predates the 3-structure (leaflet1/leaflet2/CONNECTOR "
            "multi-label) change. Please RE-TRAIN (U-Net step 'Train'); your "
            f"annotations are intact. [err: {e}]")
    net.eval()
    bnd = ck.get("band")
    bnd = np.asarray(bnd, bool) if bnd is not None else None
    return net, ck.get("meta", {}), ck.get("roi"), in_ch, bnd, out_ch, structures


def predict_probabilities(stack: np.ndarray, model_path: str,
                          batch: Optional[int] = None, tta: bool = True,
                          op: str = "infer", progress_cb=None, roi=None,
                          band_mask=None) -> np.ndarray:
    """Run the U-Net over the stack in GPU mini-batches; returns (T,H,W) float32
    probs in [0,1]. Predictions are ZEROED outside the auto valve BAND (the
    AV-junction focus): ``band_mask`` (H,W bool) overrides the checkpoint's band;
    if None the checkpoint band is used. ``tta`` averages 4 flip views. (Legacy
    2-channel ROI-prior models still run on the ROI crop.)"""
    import torch
    from tqdm import tqdm
    net, _, ck_roi, in_ch, ck_band, out_ch, _structures = load_model(model_path)
    if roi is None:
        roi = ck_roi
    band = band_mask if band_mask is not None else ck_band
    if band is not None:
        band = np.asarray(band, bool)
    dev, is_cuda = device_banner(op)
    net.to(dev)
    if batch is None:
        batch = infer_batch(is_cuda)
    T, H, W = stack.shape
    if band is not None and band.shape != (H, W):
        print(f"[{op}] WARNING band shape {band.shape} != frame {(H, W)}; ignoring band.")
        band = None
    # (T,H,W) for a 1-channel model; (T,out_ch,H,W) for the two-leaflet model
    out = (np.zeros((T, out_ch, H, W), np.float32) if out_ch > 1
           else np.zeros((T, H, W), np.float32))
    views = [(False, False), (True, False), (False, True), (True, True)] if tta \
        else [(False, False)]
    n_batches = (T + batch - 1) // batch

    if in_ch == 2:
        if roi is not None:
            y0, x0, y1, x1 = roi_bbox(roi)
            rmask = roi_crop_mask(roi).astype(np.float32)
        else:                                            # 2-ch model but no ROI -> whole frame
            y0, x0, y1, x1 = 0, 0, H, W
            rmask = np.ones((H, W), np.float32)
        cropT = stack[:, y0:y1, x0:x1].astype(np.float32)
        ch_out = np.zeros_like(cropT, dtype=np.float32)
        print(f"[{op}] ROI 2-ch crop {y1 - y0}x{x1 - x0}, {T} frames, "
              f"batch={batch} ({n_batches} batches), tta={tta}")
        with torch.no_grad():
            for bi, s in enumerate(tqdm(range(0, T, batch), total=n_batches,
                                        desc=f"{op} batches", unit="batch")):
                chunk = cropT[s:s + batch]
                b = chunk.shape[0]
                rmb = np.broadcast_to(rmask, (b,) + rmask.shape)
                acc = np.zeros((b,) + rmask.shape, np.float32)
                for flr, fud in views:
                    ci, cr = chunk, rmb
                    if flr:
                        ci = ci[:, :, ::-1]; cr = cr[:, :, ::-1]
                    if fud:
                        ci = ci[:, ::-1, :]; cr = cr[:, ::-1, :]
                    xin = np.stack([ci, cr], axis=1)     # (b, 2, h, w)
                    xt = torch.from_numpy(np.ascontiguousarray(xin)).to(
                        dev, non_blocking=is_cuda)
                    xt, (h, w) = _pad_to_multiple(xt)
                    pr = torch.sigmoid(net(xt))[..., :h, :w][:, 0].cpu().numpy()
                    if fud:
                        pr = pr[:, ::-1, :]
                    if flr:
                        pr = pr[:, :, ::-1]
                    acc += pr
                ch_out[s:s + batch] = (acc / len(views)) * rmask   # zero outside ROI
                if progress_cb:
                    progress_cb(bi + 1, n_batches, f"inference batch {bi + 1}/{n_batches}")
                sys.stdout.flush()
        out[:, y0:y1, x0:x1] = ch_out
        return out

    # ── 1-channel input. The CONNECTOR model (cfg['structures'] set) is MULTI-LABEL
    #    sigmoid even at out_ch=3; the legacy bg/l1/l2 model is softmax at out_ch=3;
    #    out_ch<=2 is legacy sigmoid. Predictions zeroed outside the BAND (or legacy
    #    ROI); for softmax only the FOREGROUND classes are zeroed (bg stays). ──
    softmax_out = (out_ch >= 3) and (_structures is None)
    full_poly = roi_full_mask(roi, (H, W)).astype(np.float32) if roi is not None else None
    if band is not None:                                  # band takes precedence
        full_poly = band.astype(np.float32)
    print(f"[{op}] {T} frames, out_ch={out_ch} ({'softmax' if softmax_out else 'sigmoid'}), "
          f"batch={batch} ({n_batches} batches), tta={tta}"
          + (" (BAND-masked)" if band is not None else
             " (ROI-masked)" if full_poly is not None else ""))
    with torch.no_grad():
        for bi, s in enumerate(tqdm(range(0, T, batch), total=n_batches,
                                    desc=f"{op} batches", unit="batch")):
            chunk = stack[s:s + batch].astype(np.float32)
            b = chunk.shape[0]
            acc = np.zeros((b, out_ch, H, W), np.float32)
            for flr, fud in views:
                c = chunk
                if flr:
                    c = c[:, :, ::-1]
                if fud:
                    c = c[:, ::-1, :]
                x = torch.from_numpy(np.ascontiguousarray(c[:, None])).to(
                    dev, non_blocking=is_cuda)
                x, (h, w) = _pad_to_multiple(x)
                lo = net(x)[..., :h, :w]
                pr = (torch.softmax(lo, dim=1) if softmax_out
                      else torch.sigmoid(lo)).cpu().numpy()       # (b,out_ch,h,w)
                if fud:
                    pr = pr[:, :, ::-1, :]
                if flr:
                    pr = pr[:, :, :, ::-1]
                acc += pr
            res = acc / len(views)                        # (b, out_ch, H, W)
            if full_poly is not None:
                if softmax_out:
                    res[:, 1:] = res[:, 1:] * full_poly[None, None]   # zero leaflets, keep bg
                else:
                    res = res * full_poly[None, None]
            out[s:s + batch] = res if out_ch > 1 else res[:, 0]
            if progress_cb:
                progress_cb(bi + 1, n_batches, f"inference batch {bi + 1}/{n_batches}")
            sys.stdout.flush()
    return out


# ============================================================
# Held-out validation against manual annotations
# ============================================================
def validate_model(ds_dir: str, model_path: str,
                   um_per_pixel: Optional[float] = None, seed: int = 0) -> dict:
    """Evaluate on the held-out validation split (same seed as train_unet):
    Dice, clDice, and mean symmetric centerline distance (px and um). Prints a
    report and returns the metrics."""
    from scipy.ndimage import distance_transform_edt
    if um_per_pixel is None:
        um_per_pixel = UM_PER_PIXEL
    p = _ds_paths(ds_dir)
    meta = _migrate_legacy_dataset(ds_dir)        # legacy PNG/dict -> npy + int frames
    frames = _frame_indices(meta)                 # sorted unique INTEGER indices
    if len(frames) < 2:
        print("[validate] too few annotations.")
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    n_val = max(1, int(round(TRAIN_VAL_SPLIT * len(frames))))
    val = [frames[k] for k in idx[:n_val]]
    imgs = np.stack([np.load(p["images"] / f"frame_{t:04d}.npy") for t in val])
    gts = [np.load(p["masks"] / f"frame_{t:04d}.npy") > 0 for t in val]
    probs = predict_probabilities(imgs, model_path, tta=True)

    dices, clds, cdist_px = [], [], []
    for pr, G in zip(probs, gts):
        P = pr > PROB_THRESHOLD
        dices.append(2 * (P & G).sum() / (P.sum() + G.sum() + 1e-8))
        sp, sg = skeletonize(P), skeletonize(G)
        if sp.sum() > 0 and sg.sum() > 0:
            tprec = (sp & G).sum() / (sp.sum() + 1e-8)
            tsens = (sg & P).sum() / (sg.sum() + 1e-8)
            clds.append(2 * tprec * tsens / (tprec + tsens + 1e-8))
            dg, dp = distance_transform_edt(~sg), distance_transform_edt(~sp)
            cdist_px.append(0.5 * (dg[sp].mean() + dp[sg].mean()))

    mD = float(np.mean(dices)) if dices else float("nan")
    mC = float(np.mean(clds)) if clds else float("nan")
    mCd = float(np.mean(cdist_px)) if cdist_px else float("nan")
    print("=" * 56)
    print(f"HELD-OUT VALIDATION  ({len(val)} frames: {val})")
    print(f"  Dice             : {mD:.3f}")
    print(f"  clDice           : {mC:.3f}")
    cmsg = f"{mCd:.2f} px" + (f" = {mCd * um_per_pixel:.2f} um" if um_per_pixel else "")
    print(f"  centerline dist  : {cmsg}")
    print("=" * 56)
    return dict(dice=mD, cldice=mC, centerline_px=mCd,
                centerline_um=(mCd * um_per_pixel if um_per_pixel else None),
                val_frames=val)


# ============================================================
# Signals & cardiac-cycle event detection
# ============================================================
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-10 else x - x.mean()


def robust_cardiac_period(traces, n: int, expected: int = EXPECTED_OPENINGS,
                          fps: float = FPS_DEFAULT, hr_bpm=(90.0, 260.0)) -> float:
    """FUNDAMENTAL cardiac period (frames), HARMONIC-REJECTED. Windowed
    autocorrelation restricted to the plausible-period band
    [n/(expected+1.5), n/(expected-1.5)] (further bounded by a physiological
    heart-rate range), so a 2nd/5th harmonic (a SHORTER lag) cannot win. Uses the
    strongest-supported trace among ``traces``; falls back to n/expected.

    For 901 frames @ 600 Hz, expected=4 -> band ~[164, 360] -> period ~225."""
    band_lo = max(6, int(round(n / (expected + 1.5))))
    band_hi = min(n - 2, int(round(n / max(1.0, expected - 1.5))))
    if fps and hr_bpm:                                # clamp to a plausible HR range
        band_lo = max(band_lo, int(round(fps * 60.0 / hr_bpm[1])))
        band_hi = min(band_hi, int(round(fps * 60.0 / hr_bpm[0])))
    if band_hi <= band_lo:
        return float(n) / max(1, expected)
    best = None
    for tr in traces:
        if tr is None:
            continue
        x = np.asarray(tr, float).ravel()
        if len(x) < n:
            continue
        x = x[:n] - ndimage.uniform_filter1d(np.asarray(x[:n], float),
                                             size=min(80, max(3, n // 8)))
        x = x - x.mean()
        if x.std() < 1e-9:
            continue
        ac = np.correlate(x, x, "full")[n - 1:]
        ac = ac / (ac[0] + 1e-9)
        w = ac[band_lo:band_hi + 1]                  # ONLY the fundamental band
        lag = band_lo + int(np.argmax(w))
        score = float(w.max())
        if best is None or score > best[0]:
            best = (score, lag)
    return float(best[1]) if best else float(n) / max(1, expected)


def detect_contractions(trace: np.ndarray, z_threshold: float,
                        hp_size: int = 80, prominence: float = 0.3,
                        distance: int = 5, expected: int = EXPECTED_OPENINGS,
                        fps: float = FPS_DEFAULT):
    """High-pass + z-score a chamber trace; return contraction frames (downward
    dips with z < z_threshold). The minimum inter-peak distance is tied to the
    ROBUST fundamental period (>= ~half a cardiac cycle) so it cannot fire ~20x
    on noise / sub-beat features. Matches valve_analysis._detect_contractions."""
    x = np.asarray(trace, float)
    x = x - ndimage.uniform_filter1d(x, size=hp_size)
    z = _zscore(x)
    n = len(z)
    P = robust_cardiac_period([trace], n, expected, fps)
    dist = max(int(distance), int(round(0.55 * P)))   # >= ~half a cardiac cycle
    raw, _ = find_peaks(-z, prominence=prominence, distance=dist)
    strong = np.array([p for p in raw if z[p] < z_threshold], dtype=int)
    if len(strong) >= max(2, expected - 1):
        return strong, z
    # weak signal (area dips shallower than the fixed z-threshold): take the
    # ``expected`` DEEPEST well-separated dips so we still get ~expected/cycle.
    if len(raw) == 0:
        return strong, z
    deepest = raw[np.argsort(z[raw])][:expected]
    return np.array(sorted(deepest), dtype=int), z


def one_event_per_cycle(opening_signal: np.ndarray,
                        v_peaks: np.ndarray, a_peaks: np.ndarray,
                        expected: int = EXPECTED_OPENINGS) -> np.ndarray:
    """Take exactly one opening per cardiac cycle.

    Segments the trace by the midpoints between consecutive contraction
    landmarks (atrial preferred; ventricular fallback) and returns the frame of
    maximum opening within each segment — guaranteeing one event per cycle and
    covering weak/off-phase cycles. Identical philosophy to the segmented
    argmax used in valve_analysis.make_publication_figure.
    """
    n = len(opening_signal)
    a, v = np.sort(np.asarray(a_peaks, int)), np.sort(np.asarray(v_peaks, int))
    landmarks = a if len(a) >= len(v) else v
    if len(landmarks) >= 2:
        mids = ((landmarks[:-1] + landmarks[1:]) // 2).astype(int)
        bounds = np.concatenate([[0], mids, [n]])
    elif len(landmarks) == 1:
        bounds = np.array([0, n], dtype=int)
    else:
        # no landmarks: fall back to the `expected` strongest, well-separated peaks
        env = ndimage.uniform_filter1d(opening_signal, 3)
        pk, _ = find_peaks(env, distance=max(3, n // (expected * 2)))
        if len(pk) == 0:
            return np.array([int(np.argmax(env))])
        order = pk[np.argsort(env[pk])[::-1]][:expected]
        return np.array(sorted(order), dtype=int)
    events = []
    for i in range(len(bounds) - 1):
        s0, s1 = int(bounds[i]), int(bounds[i + 1])
        if s1 > s0:
            events.append(s0 + int(np.argmax(opening_signal[s0:s1])))
    return np.array(sorted(set(events)), dtype=int)


def darkening_signal(stack: np.ndarray, roi: Optional[tuple] = None) -> np.ndarray:
    """Dark-field valve-OPEN signal: per-frame sum of intensity dropping BELOW
    the per-pixel temporal median within an ROI. Complements the geometric gap.
    """
    sub = stack if roi is None else stack[:, roi[0]:roi[2], roi[1]:roi[3]]
    bg = np.median(sub, axis=0)
    dark = np.clip(bg[None] - sub, 0, None).sum(axis=(1, 2))
    return np.nan_to_num(dark)


# ============================================================
# Drawing helpers for the two-leaflet overlay
# ============================================================
def _draw_polyline_color(rgb: np.ndarray, path: np.ndarray, color, thickness: int,
                         dashed: bool = False, dash: int = OVERLAY_DASH,
                         gap: int = OVERLAY_GAP) -> None:
    """Anti-aliased colored polyline into an (H,W,3) uint8 image, in place."""
    if path is None or len(path) < 2:
        return
    H, W, _ = rgb.shape
    col = np.array(color, dtype=np.float32)
    s = _arc_length(path)
    run = 0.0
    for (y0, x0), (y1, x1), L in zip(path[:-1], path[1:], np.diff(s)):
        if dashed and (int(run // (dash + gap)) * (dash + gap) + dash) < run:
            run += L
            continue
        run += L
        rr, cc, val = line_aa(int(round(y0)), int(round(x0)),
                              int(round(y1)), int(round(x1)))
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc, val = rr[ok], cc[ok], val[ok]
        for d in range(-(thickness // 2), thickness // 2 + 1):
            rr2 = np.clip(rr + d, 0, H - 1)
            a = val[:, None]
            rgb[rr2, cc] = ((1 - a) * rgb[rr2, cc] + a * col[None]).astype(np.uint8)


def _restored_rgb(frame01: np.ndarray) -> np.ndarray:
    g = (np.clip(frame01, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


# ============================================================
# Nature-style multi-panel figure
# ============================================================
def make_leaflet_figure(stack: np.ndarray, probs: np.ndarray,
                        gap_signal: np.ndarray, open_frames: np.ndarray,
                        v_z: Optional[np.ndarray], a_z: Optional[np.ndarray],
                        v_peaks: np.ndarray, a_peaks: np.ndarray,
                        fps: float, save_path: str,
                        closed_frame: Optional[int] = None) -> str:
    """Assemble the publication figure:
       a  closed-valve frame + two-leaflet centerline overlay (+ scale bar)
       b  open-valve frame (strongest opening) + overlay
       c  leaflet-gap trace over time with V/A contractions and the detected
          openings; chamber z-traces underneath for cardiac phase context.
    Exports SVG + PDF + 600-dpi PNG.
    """
    set_pub_style()
    T = stack.shape[0]
    open_main = int(open_frames[np.argmax(gap_signal[open_frames])]) if len(open_frames) else int(np.argmax(gap_signal))
    if closed_frame is None:
        # most-closed = smallest gap, away from the opening frames
        mask_far = np.ones(T, bool)
        for f in open_frames:
            mask_far[max(0, f - 5):f + 6] = False
        closed_frame = int(np.argmin(np.where(mask_far, gap_signal, gap_signal.max() + 1)))

    t = np.arange(T) / fps if fps else np.arange(T)
    xlabel = "Time (s)" if fps else "Frame"

    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.35, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    def _panel_image(ax, frame, title):
        ax.imshow(np.clip(stack[frame], 0, 1), cmap="gray", vmin=0, vmax=1)
        info = (frame_leaflets(probs[frame])[0]
                if probs.ndim == 4 else split_two_leaflets(probs[frame] > PROB_THRESHOLD))
        # EXTENDED centerlines: CLOSED reads as joined, OPEN keeps a gap
        ea, eb = extend_leaflet_pair(info["leaflet_a"], info["leaflet_b"])
        for cl, c in ((ea, C_LEAFLET_A), (eb, C_LEAFLET_B)):
            if cl is not None:
                ax.plot(cl[:, 1], cl[:, 0], color=c, lw=1.4)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        add_scale_bar(ax, 20.0, stack.shape[2])

    _panel_image(ax_a, closed_frame, "Closed (coapted)")
    panel_letter(ax_a, "a", x=-0.04)
    _panel_image(ax_b, open_main, "Open (line break)")
    panel_letter(ax_b, "b", x=-0.04)
    ax_a.legend(handles=[Line2D([0], [0], color=C_LEAFLET_A, lw=1.4, label="leaflet 1"),
                         Line2D([0], [0], color=C_LEAFLET_B, lw=1.4, label="leaflet 2")],
                loc="upper left", frameon=False, fontsize=6, labelcolor="white")

    # Panel c — gap trace + cardiac context
    g = _zscore(gap_signal)
    ax_c.plot(t, g, color=C_VALVE, lw=1.1, label="leaflet gap (open)")
    if v_z is not None:
        ax_c.plot(t, v_z - (g.min() - 1.5), color=C_VENTRICLE, lw=0.7, alpha=0.7)
    if a_z is not None:
        ax_c.plot(t, a_z - (g.min() - 1.5), color=C_ATRIUM, lw=0.7, alpha=0.7)
    if len(v_peaks):
        ax_c.plot(t[v_peaks], (v_z[v_peaks] - (g.min() - 1.5)) if v_z is not None
                  else np.full(len(v_peaks), g.min()), "v", color=C_VENTRICLE, ms=4,
                  mec="k", mew=0.3)
    if len(a_peaks):
        ax_c.plot(t[a_peaks], (a_z[a_peaks] - (g.min() - 1.5)) if a_z is not None
                  else np.full(len(a_peaks), g.min()), "^", color=C_ATRIUM, ms=4,
                  mec="k", mew=0.3)
    for f in open_frames:
        ax_c.axvline(t[f], color=C_OPEN, lw=0.6, ls="--", alpha=0.6)
        ax_c.plot(t[f], g[f], "o", color=C_OPEN, ms=4.5, mec="k", mew=0.3, zorder=5)
    ax_c.set_xlabel(xlabel); ax_c.set_ylabel("z-scored signal")
    ax_c.set_title(f"Valve opening: {len(open_frames)} events "
                   f"(one per cardiac cycle)", fontsize=7.5)
    ax_c.legend(handles=[
        Line2D([0], [0], color=C_VALVE, lw=1.1, label="leaflet gap"),
        Line2D([0], [0], color=C_VENTRICLE, lw=0.9, label="ventricle"),
        Line2D([0], [0], color=C_ATRIUM, lw=0.9, label="atrium"),
        Line2D([0], [0], marker="o", color=C_OPEN, lw=0, mec="k", mew=0.3,
               label="opening")], loc="center left", bbox_to_anchor=(1.005, 0.5),
        frameon=False)
    panel_letter(ax_c, "c", x=-0.06)

    paths = save_vector(fig, save_path)
    plt.close(fig)
    print(f"[figure] saved: {', '.join(os.path.basename(p) for p in paths)}")
    return paths[0]


# ============================================================
# Open-gap diagnostic (TRAINING vs DISPLAY)
# ============================================================
def diagnose_open_gap(stack: np.ndarray, model_path: str, n_show: int = 4,
                      prob_threshold: Optional[float] = None,
                      save_path: str = "gap_diagnosis.png",
                      roi_margin: int = 16) -> dict:
    """Instrument the open-gap pipeline. For a few OPEN frames (dark-field
    darkening peaks, chosen independently of the model) save a grid:
        [ sigmoid prob | binary mask>thr | skeleton | rendered centerline ]
    so we can tell whether the RAW mask already breaks (=> DISPLAY/post-proc:
    the spline is bridging a real gap) or stays continuous (=> TRAINING: the
    model never learned the open/broken state). Prints + returns per-frame
    skeleton component counts."""
    import matplotlib.pyplot as plt
    from skimage.morphology import skeletonize
    from skimage.measure import label as cc_label
    from scipy.signal import find_peaks

    stack = np.clip(stack.astype(np.float32), 0, 1)
    T, H, W = stack.shape
    thr = PROB_THRESHOLD if prob_threshold is None else prob_threshold
    try:
        from valve_analysis import compute_valve_open_signal_from_darkening
        _, dk = compute_valve_open_signal_from_darkening(stack)
    except Exception:
        dk = stack.reshape(T, -1).mean(1)
    z = (dk - dk.mean()) / (dk.std() + 1e-9)
    pk, _ = find_peaks(z, prominence=0.5, distance=max(3, T // 40))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(z))])
    frames = sorted(int(f) for f in pk[np.argsort(z[pk])[::-1]][:n_show])
    print(f"[diagnose] OPEN candidate frames (darkening peaks): {frames}")

    probs = predict_probabilities(stack[frames], model_path, tta=True, op="diagnose")
    fg = probs.max(0) > thr
    if fg.any():
        ys, xs = np.where(fg)
        ry0, rx0 = max(0, ys.min() - roi_margin), max(0, xs.min() - roi_margin)
        ry1, rx1 = min(H, ys.max() + roi_margin), min(W, xs.max() + roi_margin)
    else:
        ry0, rx0, ry1, rx1 = 0, 0, H, W

    set_pub_style()
    nf = len(frames)
    fig, axes = plt.subplots(nf, 4, figsize=(10, 2.5 * nf), squeeze=False)
    summary = []
    for r, (f, pr) in enumerate(zip(frames, probs)):
        m = pr > thr
        skel = skeletonize(m)
        ncomp = int(cc_label(skel, connectivity=2).max())
        info = split_two_leaflets(m.astype(np.uint8))
        summary.append(dict(frame=f, prob_max=float(pr.max()), mask_px=int(m.sum()),
                            skel_components=ncomp, split_components=info["n_components"],
                            gap_px=float(info["gap_px"])))
        axes[r, 0].imshow(pr[ry0:ry1, rx0:rx1], cmap="magma", vmin=0, vmax=1)
        axes[r, 0].set_title(f"f{f}: prob (max {pr.max():.2f})", fontsize=7)
        axes[r, 1].imshow(m[ry0:ry1, rx0:rx1], cmap="gray")
        axes[r, 1].set_title(f"mask>thr ({int(m.sum())} px)", fontsize=7)
        axes[r, 2].imshow(skel[ry0:ry1, rx0:rx1], cmap="gray")
        axes[r, 2].set_title(f"skeleton (components={ncomp})", fontsize=7,
                             color=("#1a7a3a" if ncomp >= 2 else "#b22222"))
        axes[r, 3].imshow(stack[f][ry0:ry1, rx0:rx1], cmap="gray", vmin=0, vmax=1)
        for cl, c in ((info["leaflet_a"], C_LEAFLET_A), (info["leaflet_b"], C_LEAFLET_B)):
            if cl is not None:
                axes[r, 3].plot(cl[:, 1] - rx0, cl[:, 0] - ry0, "-", lw=1.8, color=c)
        axes[r, 3].set_title(f"centerline (split={info['n_components']}, "
                             f"gap {info['gap_px']:.1f}px)", fontsize=7)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Open-gap diagnosis:  prob | mask | skeleton | centerline   "
                 "(skeleton components>=2 => raw mask breaks)", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    any_break = any(s["skel_components"] >= 2 for s in summary)
    verdict = ("DISPLAY (raw mask breaks but rendering may bridge it)" if any_break
               else "TRAINING (raw mask is continuous on open frames - model never "
                    "learned the gap)")
    print(f"[diagnose] saved {save_path}")
    for s in summary:
        print("  ", s)
    print(f"[diagnose] VERDICT: {verdict}")
    return dict(frames=frames, summary=summary, verdict=verdict,
                any_break=any_break, save_path=save_path)


# ============================================================
# Stage 3: TWO-LEAFLET separation  (entry point)
# ============================================================
def leaflet_centerline(prob_mask: np.ndarray, smooth: float = SPLINE_SMOOTH):
    """Centerline of ONE leaflet probability map: skeletonize, prune tiny
    fragments, order the longest fragment, spline-smooth. Returns (M,2) or None.
    (Per-leaflet - the two leaflets are separate channels, so no break-splitting.)"""
    skel = skeletonize(prob_mask > 0)
    lbl = cc_label(skel, connectivity=2)
    n = int(lbl.max())
    if n == 0:
        return None
    if n > 1:                                     # prune tiny fragments
        sizes = np.bincount(lbl.ravel())
        small = [k for k in range(1, n + 1) if sizes[k] < MIN_SKELETON_PX]
        if small:
            skel = skel & ~np.isin(lbl, small)
            lbl = cc_label(skel, connectivity=2); n = int(lbl.max())
        if n == 0:
            return None
    frags = [order_skeleton(lbl == k) for k in range(1, n + 1)]
    frags = [f for f in frags if len(f) >= 2]
    if not frags:
        return None
    frags.sort(key=lambda pp: _arc_length(pp)[-1], reverse=True)
    return smooth_centerline(frags[0], smooth)


def _tip_tangent(path: np.ndarray, end: int, n: int = TIP_FIT_PTS) -> np.ndarray:
    """Unit tangent at a leaflet's ORIFICE tip, pointing OUTWARD (away from the
    leaflet body, toward the junction). ``end``=0 -> tip is path[0]; 1 -> path[-1].
    Uses the local segment from an inner point to the tip (low-order fit)."""
    n = int(min(max(1, n), len(path) - 1))
    if end == 0:
        tip, inner = path[0], path[n]
    else:
        tip, inner = path[-1], path[-1 - n]
    v = np.asarray(tip, float) - np.asarray(inner, float)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-6 else np.zeros(2)


def _y_extreme_endpoint(path: np.ndarray, want_max_y: bool):
    """The U-curve ENDPOINT (path[0] or path[-1]) with the largest (want_max_y) or
    smallest y - i.e. the leaflet's free tip that faces the orifice vertically."""
    y0, y1 = float(path[0][0]), float(path[-1][0])
    pick_last = (y1 >= y0) if want_max_y else (y1 <= y0)
    return (path[-1] if pick_last else path[0]), (1 if pick_last else 0)


def two_channel_leaflets(prob1: np.ndarray, prob2: np.ndarray,
                         thr: float = PROB_THRESHOLD, smooth: float = SPLINE_SMOOTH,
                         upper_is_smaller_y: bool = UPPER_IS_SMALLER_Y):
    """Per-frame geometry of the two U-shaped leaflets, ORDERED + measured by
    VERTICAL (up/down) position so identity and the orifice tips are stable.

    * leaflet_a = UPPER, leaflet_b = LOWER (by mean y; ``upper_is_smaller_y``
      flips the convention if the image is inverted).
    * FREE (orifice) tips face each other vertically: the UPPER leaflet's LOWEST
      endpoint (tip_a) and the LOWER leaflet's HIGHEST endpoint (tip_b).
    * gap_px = the VERTICAL gap between those tips (lower-top.y - upper-bottom.y;
      ~0 closed, > 0 open). dir_a/dir_b point toward each other vertically (upper
      DOWN, lower UP) for the extend+overlap test."""
    c1 = leaflet_centerline(prob1 > thr, smooth)
    c2 = leaflet_centerline(prob2 > thr, smooth)
    out = dict(n_components=int(c1 is not None) + int(c2 is not None),
               leaflet_a=None, leaflet_b=None, gap_px=0.0, open_len_px=0.0,
               midpoint=None, tip_a=None, tip_b=None, dir_a=None, dir_b=None)
    out["open_len_px"] = float((_arc_length(c1)[-1] if c1 is not None else 0.0)
                               + (_arc_length(c2)[-1] if c2 is not None else 0.0))

    if c1 is not None and c2 is not None:
        # ORDER by vertical position: leaflet_a = UPPER (smaller mean y by default)
        y1, y2 = float(np.mean(c1[:, 0])), float(np.mean(c2[:, 0]))
        c1_is_upper = (y1 <= y2) if upper_is_smaller_y else (y1 >= y2)
        a, b = (c1, c2) if c1_is_upper else (c2, c1)        # a=UPPER, b=LOWER
        out["leaflet_a"], out["leaflet_b"] = a, b
        # free tips: UPPER's lowest endpoint, LOWER's highest endpoint
        tip_a, _ = _y_extreme_endpoint(a, want_max_y=upper_is_smaller_y)   # upper -> max y
        tip_b, _ = _y_extreme_endpoint(b, want_max_y=not upper_is_smaller_y)  # lower -> min y
        out["tip_a"], out["tip_b"] = tip_a, tip_b
        # extension directions: UPPER tip DOWN, LOWER tip UP (toward each other)
        down = 1.0 if upper_is_smaller_y else -1.0
        out["dir_a"] = np.array([down, 0.0])      # upper goes toward larger y (down)
        out["dir_b"] = np.array([-down, 0.0])     # lower goes toward smaller y (up)
        # VERTICAL gap (lower-top.y - upper-bottom.y); 0 when tips meet/cross (closed)
        out["gap_px"] = float(max(0.0, (tip_b[0] - tip_a[0]) * down))
        out["midpoint"] = (tip_a + tip_b) / 2.0
    elif c1 is not None:
        out["leaflet_a"] = c1; out["midpoint"] = c1[len(c1) // 2]
    elif c2 is not None:
        out["leaflet_b"] = c2; out["midpoint"] = c2[len(c2) // 2]
    return out


def frame_leaflets(pt: np.ndarray, upper_is_smaller_y: bool = UPPER_IS_SMALLER_Y):
    """Per-frame leaflet geometry from a prediction map, for BOTH the 3-class
    softmax model (pt = (3,H,W): bg/l1/l2 -> argmax masks, mutually exclusive,
    NON-overlapping) and the legacy 2-channel sigmoid model (pt = (2,H,W)).
    Leaflets are ordered UPPER/LOWER by y (see two_channel_leaflets). Returns
    (info, lp1, lp2): ``info`` = geometry; lp1/lp2 = the per-class probability
    maps used (symmetrically) for the contact-continuity test."""
    pt = np.asarray(pt, np.float32)
    if pt.shape[0] >= 3:                          # 3-class softmax -> argmax masks
        lab = pt.argmax(0)
        m1 = (lab == 1).astype(np.float32); m2 = (lab == 2).astype(np.float32)
        info = two_channel_leaflets(m1, m2, thr=0.5, upper_is_smaller_y=upper_is_smaller_y)
        return info, pt[1], pt[2]                 # class-1 / class-2 probabilities
    info = two_channel_leaflets(pt[0], pt[1], upper_is_smaller_y=upper_is_smaller_y)
    return info, pt[0], pt[1]


def _seg_seg_dist(a0, a1, b0, b1, k: int = 16) -> float:
    """Min distance between 2D segments a0-a1 and b0-b1 (0 if they cross/overlap).
    Sampled (robust + simple); resolution ~ seg_len/k."""
    from scipy.spatial.distance import cdist
    ta = np.linspace(0, 1, k)
    A = np.asarray(a0, float) + ta[:, None] * (np.asarray(a1, float) - a0)
    B = np.asarray(b0, float) + ta[:, None] * (np.asarray(b1, float) - b0)
    return float(cdist(A, B).min())


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous run of True in a boolean array."""
    best = cur = 0
    for v in np.asarray(mask, bool):
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def leaflet_openness(info: dict, prob1: np.ndarray, prob2: np.ndarray,
                     method: str = OPENNESS_METHOD,
                     extend_len: float = EXTEND_LEN_PX,
                     closed_gap_px: float = CLOSED_GAP_PX,
                     bridge_prob: float = BRIDGE_PROB_THRESH,
                     bridge_break_px: float = BRIDGE_BREAK_PX):
    """POST-PROCESS the two U-leaflets into a CLOSED/OPEN state + openness
    MAGNITUDE, NO retraining. Magnitude = the VERTICAL gap between the free tips
    (upper-leaflet bottom tip <-> lower-leaflet top tip; ~0 closed, > 0 open).
    The CLOSED/OPEN decision uses one of:

      'extend_overlap' (A): extend the UPPER tip DOWN and the LOWER tip UP by
        ``extend_len`` (toward each other); CLOSED if they meet / residual gap
        <= ``closed_gap_px``, else OPEN.
      'connect_break'  (B): walk the tip-to-tip connector and test contact
        CONTINUITY (leaflet prediction bridging); CLOSED if continuous, OPEN if a
        break run > ``bridge_break_px``.

    Returns (is_open: bool, vertical_gap_px: float, detail: dict)."""
    ta, tb = info.get("tip_a"), info.get("tip_b")
    if ta is None or tb is None:                  # need BOTH leaflets to judge a gap
        return False, 0.0, {"reason": "one_leaflet_missing"}
    ta = np.asarray(ta, float); tb = np.asarray(tb, float)
    vgap = float(info.get("gap_px", 0.0))         # VERTICAL free-tip gap = the openness

    if method == "connect_break":
        H, W = prob1.shape
        L = float(np.hypot(*(ta - tb)))
        n = max(2, int(round(L)))
        ts = np.linspace(0.0, 1.0, n)
        pts = ta + ts[:, None] * (tb - ta)        # connector between the free tips
        yy = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
        xx = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)
        struct = np.maximum(prob1[yy, xx], prob2[yy, xx]) >= bridge_prob   # contact?
        step = L / (n - 1) if n > 1 else 0.0
        break_px = float(_longest_true_run(~struct) * step)
        is_open = break_px > bridge_break_px
        return is_open, vgap, {"method": "connect_break", "break_px": break_px,
                               "bridged_frac": float(struct.mean()), "vgap": vgap}

    # default: extend_overlap (A) - extend UPPER down + LOWER up (vertical)
    da = np.asarray(info.get("dir_a", [1.0, 0.0]), float)
    db = np.asarray(info.get("dir_b", [-1.0, 0.0]), float)
    ext_a = ta + da * extend_len
    ext_b = tb + db * extend_len
    resid = _seg_seg_dist(ta, ext_a, tb, ext_b)   # 0 if the extensions cross/overlap
    is_open = resid > closed_gap_px
    return is_open, vgap, {"method": "extend_overlap", "residual_px": resid, "vgap": vgap}


def connector_openness(info: dict, prob_c: np.ndarray,
                       prob_thresh: float = CONNECTOR_PROB_THRESH,
                       break_px: float = CONNECTOR_BREAK_PX):
    """CONNECTING-LINE openness (current approach). The PREDICTED connector channel
    ``prob_c`` (H,W) is the detection structure: CONTINUOUS along the orifice ->
    CLOSED; a missing run (break) -> OPEN. openness MAGNITUDE = the longest break
    length (px) along the connector.

    Sampling path: the segment between the two leaflet FREE TIPS (where the
    connector lives). If a tip is missing, fall back to the connector mask's own
    extent (longest gap along its principal axis). If the connector channel is
    essentially ABSENT, the valve is fully OPEN. Returns (is_open, break_px, detail)."""
    prob_c = np.asarray(prob_c, np.float32)
    H, W = prob_c.shape
    ta, tb = info.get("tip_a"), info.get("tip_b")
    if ta is not None and tb is not None:
        ta = np.asarray(ta, float); tb = np.asarray(tb, float)
        L = float(np.hypot(*(ta - tb)))
        n = max(2, int(round(L)))
        ts = np.linspace(0.0, 1.0, n)
        pts = ta + ts[:, None] * (tb - ta)
        yy = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
        xx = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)
        profile = prob_c[yy, xx].astype(np.float32)   # connector prob ALONG the path
        present = profile >= prob_thresh
        step = L / (n - 1) if n > 1 else 0.0
        # locate the LONGEST missing run -> its centre = the break LOCATION (0..1)
        miss = ~present
        b0 = b1 = bc01 = -1
        cur = cs = 0
        for k, v in enumerate(miss):
            if v:
                cs = k if cur == 0 else cs; cur += 1
                if cur > (b1 - b0 + 1):
                    b0, b1 = cs, k
            else:
                cur = 0
        brk = float((b1 - b0 + 1) * step) if b1 >= 0 else 0.0
        bc01 = float(((b0 + b1) / 2.0) / (n - 1)) if (b1 >= 0 and n > 1) else float("nan")
        frac = float(present.mean())
        is_open = (brk > break_px) or (frac < 0.25)
        return is_open, brk, {"method": "connector", "break_px": brk,
                              "present_frac": frac, "path": "tip_to_tip",
                              "profile": profile, "break_center01": bc01}
    # fallback: no leaflet tips -> use the connector mask's own longest gap
    cm = prob_c >= prob_thresh
    if not cm.any():                              # connector absent -> fully OPEN
        return True, float(max(H, W)), {"method": "connector", "path": "absent"}
    try:
        sk = skeletonize(cm)
        op = order_skeleton(sk) if sk.any() else None
        if op is not None and len(op) >= 2:
            P = np.asarray(op, float)
            d = np.hypot(*(P[-1] - P[0]))
            span = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            brk = float(max(0.0, d - span))       # rough missing length
            return brk > break_px, brk, {"method": "connector", "path": "mask"}
    except Exception:
        pass
    return False, 0.0, {"method": "connector", "path": "mask_continuous"}


def model_structures(model_path: str):
    """Cheap read of the model's structure-channel names (the CONNECTING-LINE
    multi-label model) or None for the legacy softmax model."""
    import torch
    try:
        return torch.load(model_path, map_location="cpu").get("config", {}).get("structures")
    except Exception:
        return None


def extend_leaflet_pair(a, b, extend_len: float = EXTEND_LEN_PX, n_fit: int = TIP_FIT_PTS,
                        upper_is_smaller_y: bool = UPPER_IS_SMALLER_Y):
    """Return the two leaflet centerlines with their free tips EXTRAPOLATED toward
    each other VERTICALLY (UPPER tip DOWN, LOWER tip UP) by ``extend_len``, for
    RENDERING: CLOSED -> the extensions meet so the leaflets read JOINED; OPEN ->
    a gap remains. Input/return are (M,2) yx arrays (NaN rows dropped)."""
    def _clean(p):
        if p is None:
            return None
        p = np.asarray(p, float)
        p = p[~np.isnan(p).any(1)] if p.ndim == 2 else p
        return p if len(p) >= 2 else None
    a, b = _clean(a), _clean(b)
    if a is None or b is None:
        return a, b
    # order UPPER/LOWER by mean y, then extend the facing tips toward each other
    up, lo = (a, b) if ((np.mean(a[:, 0]) <= np.mean(b[:, 0])) == upper_is_smaller_y) else (b, a)
    down = 1.0 if upper_is_smaller_y else -1.0
    tip_up, i_up = _y_extreme_endpoint(up, want_max_y=upper_is_smaller_y)   # upper bottom
    tip_lo, i_lo = _y_extreme_endpoint(lo, want_max_y=not upper_is_smaller_y)  # lower top
    up_ext = np.vstack([up, tip_up + np.array([down, 0.0]) * extend_len]) if i_up == 1 \
        else np.vstack([tip_up + np.array([down, 0.0]) * extend_len, up])
    lo_ext = np.vstack([lo, tip_lo + np.array([-down, 0.0]) * extend_len]) if i_lo == 1 \
        else np.vstack([tip_lo + np.array([-down, 0.0]) * extend_len, lo])
    # return in the SAME order as the inputs (a, b)
    return (up_ext, lo_ext) if up is a else (lo_ext, up_ext)


def signal_max_change_points(sig: np.ndarray, expected: int = EXPECTED_OPENINGS,
                             min_sep: int = 10):
    """Frames of MAXIMUM CHANGE (steepest slope) of a chamber MOVEMENT signal -
    the contraction/relaxation transitions. Returns (sorted change-frame indices,
    signed derivative). The valve opening is expected to fall BETWEEN the
    ventricular and atrial max-change frames of a cycle."""
    x = ndimage.uniform_filter1d(_zscore(np.asarray(sig, float)), 3)
    d = np.gradient(x)
    mag = np.abs(d)
    n = len(x)
    pk, _ = find_peaks(mag, distance=max(min_sep, n // (expected * 2 + 1)))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(mag))])
    keep = pk[np.argsort(mag[pk])[::-1]][:max(expected * 2, 4)]
    return np.sort(keep), d


def cardiac_period_frames(v_peaks, a_peaks, openness, n, expected=EXPECTED_OPENINGS):
    """Cardiac period (frames), harmonic-rejected. Prefer the median spacing of
    chamber landmarks ONLY if it lands in the plausible band; otherwise use the
    robust (windowed-autocorrelation) fundamental period of the openness signal."""
    lm = a_peaks if len(a_peaks) >= len(v_peaks) else v_peaks
    if len(lm) >= 2:
        P = float(np.median(np.diff(np.sort(lm))))
        if n / (expected + 1.5) <= P <= n / max(1, expected - 1.5):
            return P                                  # landmarks agree with ~expected cycles
    return robust_cardiac_period([openness], n, expected)


def atrium_open_windows(a_sig, fps: float = FPS_DEFAULT,
                        expected: int = EXPECTED_OPENINGS,
                        start_offset: int = 0, end_lead: int = -1):
    """Per-cardiac-cycle EXPECTED-OPEN windows defined by the ATRIUM signal: from
    the atrium MAXIMUM (peak, just before it contracts) to JUST BEFORE the
    following atrium MINIMUM (trough) - i.e. the atrial contraction / EMPTYING
    (descending) limb. This replaces a fixed-duration band.

      start_offset : frames from the peak to START (0 = at the maximum; small
                     negative = just before it).
      end_lead     : frames before the trough to END (>=0; -1 -> auto ~6% period,
                     so it closes just before the minimum).

    Returns (windows, peaks, troughs): ``windows`` = list of (start, end) frame
    pairs (~expected, one per cycle)."""
    a = np.asarray(a_sig, float)
    n = len(a)
    if n < 8:
        return [], np.array([], int), np.array([], int)
    P = robust_cardiac_period([a], n, expected, fps)
    if end_lead is None or end_lead < 0:
        end_lead = max(2, int(round(0.06 * P)))
    dist = max(3, int(round(0.55 * P)))
    az = ndimage.uniform_filter1d(a, max(3, int(round(0.04 * P))) | 1)   # light smooth
    pk, _ = find_peaks(az, distance=dist)         # atrium MAXIMA
    tr, _ = find_peaks(-az, distance=dist)        # atrium MINIMA
    if len(pk) == 0:
        return [], np.sort(pk), np.sort(tr)
    # keep the strongest ~expected peaks (the real atrial contractions)
    if len(pk) > expected:
        pk = np.sort(pk[np.argsort(az[pk])[::-1]][:expected])
    else:
        pk = np.sort(pk)
    windows = []
    for p in pk:
        after = tr[tr > p]                        # the trough that FOLLOWS this peak
        t0 = int(after[0]) if len(after) else min(n - 1, int(p + round(0.5 * P)))
        s = int(np.clip(p + start_offset, 0, n - 1))
        e = int(np.clip(t0 - end_lead, 0, n - 1))
        if e > s:
            windows.append((s, e))
    return sorted(windows), pk, np.sort(tr)


def exactly_n_opens(openness, break_flag, v_peaks, a_peaks, n,
                    expected=EXPECTED_OPENINGS):
    """Return EXACTLY ``expected`` opening frames - one per cardiac cycle.

    Build ``expected`` cycle windows from the cardiac period (anchored on a
    chamber landmark when available, else evenly), and in each window take the
    frame of maximum openness, PREFERRING frames flagged as a genuine line break
    at the midpoint. Guarantees exactly ``expected`` events (4 opens = 4 breaks).
    """
    P = cardiac_period_frames(v_peaks, a_peaks, openness, n, expected)
    lm = np.sort(a_peaks if len(a_peaks) >= len(v_peaks) else v_peaks)
    # cycle CENTERS
    if len(lm) >= 1:
        a0 = int(lm[0]) % int(round(P))
        centers = [a0 + int(round((k + 0.5) * P)) for k in range(expected + 2)]
        centers = [c for c in centers if 0 <= c < n]
    else:
        centers = []
    if len(centers) < expected:                       # evenly-spaced fallback
        centers = [int((k + 0.5) * n / expected) for k in range(expected)]
    centers = sorted(centers)[:expected]
    half = max(2, int(round(P / 2)))
    opens, used = [], np.zeros(n, bool)
    for c in centers:
        s0, s1 = max(0, c - half), min(n, c + half + 1)
        seg = np.array(openness[s0:s1], float)
        bf = np.asarray(break_flag[s0:s1], bool)
        cand = np.where(bf, seg, -np.inf) if bf.any() else seg
        f = s0 + int(np.argmax(cand))
        opens.append(f)
    # dedup while keeping exactly `expected`: if a collision dropped one, fill the
    # widest remaining gap with its openness argmax
    opens = sorted(set(int(o) for o in opens))
    while len(opens) < expected:
        bnds = [0] + opens + [n]
        gaps = [(bnds[i + 1] - bnds[i], bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)]
        _, g0, g1 = max(gaps)
        seg = np.array(openness[g0:g1], float)
        for o in opens:
            if g0 <= o < g1:
                seg[o - g0] = -np.inf
        opens = sorted(set(opens + [g0 + int(np.argmax(seg))]))
    return np.array(sorted(opens)[:expected], dtype=int)


def analyze_line_breaks(stack: np.ndarray, model_path: str,
                        atrium_trace_path: str = "__none__",
                        ventricle_trace_path: str = "__none__",
                        fps: float = FPS_DEFAULT,
                        um_per_pixel: Optional[float] = None,
                        n_ctrl: int = 60, tta: bool = True, progress_cb=None,
                        save_prefix: str = "unet_line",
                        mid_win_px: int = 7, band_mask=None,
                        expected_opens: int = EXPECTED_OPENINGS,
                        openness_method: str = OPENNESS_METHOD,
                        extend_len: float = EXTEND_LEN_PX,
                        closed_gap_px: float = CLOSED_GAP_PX,
                        bridge_prob: float = BRIDGE_PROB_THRESH,
                        bridge_break_px: float = BRIDGE_BREAK_PX,
                        upper_is_smaller_y: bool = UPPER_IS_SMALLER_Y) -> dict:
    """TWO-LEAFLET pipeline. Infer the two leaflet channels, extract each
    leaflet's centerline SEPARATELY, then POST-PROCESS a CLOSED/OPEN state +
    openness MAGNITUDE per frame from the two centerlines (NO retraining), via
    ``openness_method``:
      'extend_overlap' (A, default): extrapolate each orifice tip by ``extend_len``
        along its tangent; CLOSED if the extended tips overlap / residual gap <=
        ``closed_gap_px``. magnitude = residual gap.
      'connect_break'  (B): test leaflet-prediction bridging along a tip-to-tip
        connector; OPEN if a break run > ``bridge_break_px`` (prob >= ``bridge_prob``
        counts as structure). magnitude = break length.
    The resulting openness(t) feeds the SAME logic: EXACTLY ``expected_opens`` opens
    (one per cardiac cycle), each timed BETWEEN the V/A max-change points.

    Saves (npy): <prefix>_{openness,break_signal,gap_px,gap_um,angle_deg,
    open_frames}.npy, <prefix>_ctrl_points.npz, and <prefix>_events.npz.
    """
    if um_per_pixel is None:
        um_per_pixel = UM_PER_PIXEL
    from tqdm import tqdm
    stack = np.clip(stack.astype(np.float32), 0, 1)
    T, H, W = stack.shape
    if band_mask is not None:
        band_mask = np.asarray(band_mask, bool)
        print(f"[analyze] AV-junction BAND focus ({int(band_mask.sum())} px); "
              f"line/break/openness restricted to the band")
    print(f"[analyze] inferring leaflet segmentation for {T} frames ...")
    probs = predict_probabilities(stack, model_path, tta=tta, op="analyze",
                                  progress_cb=progress_cb, roi=None, band_mask=band_mask)
    if probs.ndim != 4 or probs.shape[1] < 2:
        raise ValueError(
            "This model has a SINGLE output channel; the two-leaflet pipeline needs "
            "a multi-class model. RE-TRAIN (step 3) with leaflet 1 and leaflet 2 "
            "annotated as U-curves (incl. closed frames).")
    structures = model_structures(model_path)        # connecting-line model?
    is_ml = structures is not None and probs.shape[1] >= N_STRUCT
    print(f"[analyze] model output: {probs.shape[1]} channels "
          + (f"(MULTI-LABEL {structures})" if is_ml else
             ('3-class softmax bg/l1/l2' if probs.shape[1] >= 3 else '2-channel sigmoid')))

    if is_ml:                                         # CONNECTING-LINE break is primary
        openness_method = "connector"
    elif openness_method not in ("extend_overlap", "connect_break"):
        print(f"[analyze] unknown openness_method '{openness_method}' -> 'extend_overlap'")
        openness_method = "extend_overlap"
    print(f"[analyze] CLOSED/OPEN method = '{openness_method}'  "
          + (f"(connector prob>={CONNECTOR_PROB_THRESH}, break>{CONNECTOR_BREAK_PX}px)"
             if openness_method == "connector"
             else (f"(extend_len={extend_len}px, closed_gap<={closed_gap_px}px)"
                   if openness_method == "extend_overlap"
                   else f"(bridge_prob>={bridge_prob}, break>{bridge_break_px}px)")))

    gap = np.zeros(T, np.float32)                     # openness MAGNITUDE (method-dependent)
    open_bool = np.zeros(T, bool)                     # per-frame CLOSED/OPEN decision
    length = np.zeros(T, np.float32)
    angle = np.full(T, np.nan, np.float32)
    n_comp = np.zeros(T, np.int16)
    mid_yx = np.full((T, 2), np.nan, np.float32)      # per-frame orifice MIDPOINT
    cp_a = np.full((T, n_ctrl, 2), np.nan, np.float32)
    cp_b = np.full((T, n_ctrl, 2), np.nan, np.float32)
    KYMO_L = 48                                       # connecting-path kymograph length
    kymo = np.full((T, KYMO_L), np.nan, np.float32)   # connector prob along the path / frame
    break_pos = np.full(T, np.nan, np.float32)        # break-centre location along path (0..1)
    print(f"[analyze] leaflet identity by y: leaflet1=UPPER, leaflet2=LOWER "
          f"(upper_is_smaller_y={upper_is_smaller_y}); openness = VERTICAL free-tip gap")
    for t in tqdm(range(T), desc="analyze openness", unit="frame"):
        if is_ml:                                     # multi-label: leaflets ch0/ch1, connector ch2
            info = two_channel_leaflets(probs[t][0], probs[t][1],
                                        upper_is_smaller_y=upper_is_smaller_y)
        else:
            info, lp1, lp2 = frame_leaflets(probs[t], upper_is_smaller_y)
        g = leaflet_geometry(info, um_per_pixel)
        # CLOSED/OPEN + magnitude: connector-break (current) or legacy tip methods
        if openness_method == "connector":
            is_open, mag, _cd = connector_openness(info, probs[t][CONNECTOR_CH])
            prof = _cd.get("profile")                 # connector prob along the path
            if prof is not None and len(prof) >= 2:   # resample to fixed kymograph length
                kymo[t] = np.interp(np.linspace(0, 1, KYMO_L),
                                    np.linspace(0, 1, len(prof)), prof)
            break_pos[t] = _cd.get("break_center01", np.nan)
        else:
            is_open, mag, _ = leaflet_openness(
                info, lp1, lp2, method=openness_method,
                extend_len=extend_len, closed_gap_px=closed_gap_px,
                bridge_prob=bridge_prob, bridge_break_px=bridge_break_px)
        gap[t], open_bool[t] = mag, is_open
        length[t], angle[t] = g["length_px"], g["angle_deg"]
        n_comp[t] = info["n_components"]
        ra = resample_path(info["leaflet_a"], n_ctrl)
        rb = resample_path(info["leaflet_b"], n_ctrl)
        if ra is not None:
            cp_a[t] = ra
        if rb is not None:
            cp_b[t] = rb
        if info["midpoint"] is not None:                  # orifice centre (between leaflets)
            mid_yx[t] = info["midpoint"]

    gap_s = temporal_smooth_trace(gap)
    length_s = temporal_smooth_trace(length)
    angle_s = temporal_smooth_trace(np.nan_to_num(angle))
    cp_a = smooth_centerline_track(cp_a)
    cp_b = smooth_centerline_track(cp_b)

    # ── openness at the LINE MIDPOINT (no ROI) ──
    # stable anchor = temporal median of the per-frame midpoints; darkening is
    # summed in a small disk around it (the opening occurs in this neighborhood).
    if np.isfinite(mid_yx).any():
        my = float(np.nanmedian(mid_yx[:, 0])); mx = float(np.nanmedian(mid_yx[:, 1]))
    else:                                             # no line ever found -> frame centre
        my, mx = H / 2.0, W / 2.0
    yy, xx = np.ogrid[:H, :W]
    disk = ((yy - my) ** 2 + (xx - mx) ** 2) <= (mid_win_px ** 2)
    ys, xs = np.where(disk)
    sub = stack[:, ys, xs]                            # (T, Npix) around the midpoint
    bg = np.median(sub, axis=0)
    dark_mid = np.nan_to_num(np.clip(bg[None] - sub, 0.0, None).sum(axis=1))
    print(f"[analyze] orifice midpoint=({my:.0f},{mx:.0f}); openness = CLOSED/OPEN "
          f"magnitude from the two leaflet centerlines ('{openness_method}'). "
          f"midpoint darkening (r={mid_win_px}px, {int(disk.sum())}px) saved as cross-check.")

    # openness(t) = the post-processed openness MAGNITUDE (residual gap / break
    # length). The midpoint darkening is kept only as an independent cross-check.
    openness = ndimage.uniform_filter1d(_zscore(gap_s), 3)
    # a frame is OPEN per the chosen method's CLOSED/OPEN decision
    break_flag = open_bool
    print(f"[analyze] frames OPEN ({openness_method}): {int(open_bool.sum())}/{T}")

    # ── chamber MOVEMENT signals: contraction landmarks + MAX-CHANGE points ──
    v_peaks = a_peaks = np.array([], dtype=int)
    v_z = a_z = None
    v_sig = a_sig = None
    v_chg = a_chg = np.array([], dtype=int)
    if os.path.exists(ventricle_trace_path):
        v_sig = np.load(ventricle_trace_path)
        v_peaks, v_z = detect_contractions(v_sig, VENTRICLE_Z_THRESHOLD)
        v_chg, _ = signal_max_change_points(v_sig, expected_opens)
        np.save(f"{save_prefix}_ventricle.npy", v_sig)
    if os.path.exists(atrium_trace_path):
        a_sig = np.load(atrium_trace_path)
        a_peaks, a_z = detect_contractions(a_sig, ATRIUM_Z_THRESHOLD)
        a_chg, _ = signal_max_change_points(a_sig, expected_opens)
        np.save(f"{save_prefix}_atrium.npy", a_sig)

    # ── EXACTLY 4 opens (4 line-breaks), one per cardiac cycle, V/A-timed ──
    open_frames = exactly_n_opens(openness, break_flag, v_peaks, a_peaks, T,
                                  expected_opens)
    period = cardiac_period_frames(v_peaks, a_peaks, openness, T, expected_opens)

    # ── per-event metrics: gap width, duration, lags vs V/A max-change, leaflets ─
    def _nearest(arr, f):
        if len(arr) == 0:
            return None
        return int(arr[int(np.argmin(np.abs(np.asarray(arr) - f)))])

    open_thr = np.percentile(openness, 60)            # "above closed baseline"
    events = []
    for f in open_frames:
        # open duration: contiguous run around f where openness stays elevated
        s = f
        while s > 0 and openness[s - 1] >= open_thr:
            s -= 1
        e = f
        while e < T - 1 and openness[e + 1] >= open_thr:
            e += 1
        dur_frames = int(e - s + 1)
        vt, at = _nearest(v_chg, f), _nearest(a_chg, f)
        # is the open BETWEEN the V and A max-change points of its cycle?
        between = (vt is not None and at is not None and
                   (min(vt, at) - 3) <= f <= (max(vt, at) + 3))
        events.append(dict(
            frame=int(f),
            gap_um=float(gap_s[f] * um_per_pixel) if um_per_pixel else float(gap_s[f]),
            gap_px=float(gap_s[f]),
            duration_frames=dur_frames,
            duration_ms=float(dur_frames / fps * 1e3),
            lag_v_ms=(float((f - vt) / fps * 1e3) if vt is not None else float("nan")),
            lag_a_ms=(float((f - at) / fps * 1e3) if at is not None else float("nan")),
            v_change_frame=(vt if vt is not None else -1),
            a_change_frame=(at if at is not None else -1),
            between_va=bool(between),
            upper_is_a=bool(np.nanmean(cp_a[f, :, 0]) <= np.nanmean(cp_b[f, :, 0])),
        ))

    # recurrence: spacing between consecutive opens vs the cardiac period
    inter = np.diff(open_frames) if len(open_frames) >= 2 else np.array([])
    recur_frames = float(np.mean(inter)) if len(inter) else float("nan")
    n_between = sum(e["between_va"] for e in events)

    # fraction of frames essentially CLOSED (openness near its floor)
    thr_closed = np.percentile(openness, 25) + 0.15 * (openness.max() - np.percentile(openness, 25))
    frac_closed = float(np.mean(openness <= thr_closed))

    # ── save quantitative outputs ──
    np.save(f"{save_prefix}_break_signal.npy", openness)
    np.save(f"{save_prefix}_openness.npy", openness)
    np.save(f"{save_prefix}_gap_px.npy", gap_s)
    np.save(f"{save_prefix}_open_frames.npy", open_frames)
    np.save(f"{save_prefix}_angle_deg.npy", angle_s)
    np.save(f"{save_prefix}_midpoint_darkening.npy", dark_mid)
    if um_per_pixel:
        np.save(f"{save_prefix}_gap_um.npy", gap_s * um_per_pixel)
        np.save(f"{save_prefix}_length_um.npy", length_s * um_per_pixel)
    np.savez(f"{save_prefix}_ctrl_points.npz", leaflet_a=cp_a, leaflet_b=cp_b,
             n_components=n_comp, midpoint=np.array([my, mx], np.float32))
    np.save(f"{save_prefix}_open_state.npy", open_bool)        # per-frame CLOSED/OPEN
    np.save(f"{save_prefix}_connector_kymo.npy", kymo)         # (T,L) connecting-path M-mode
    np.save(f"{save_prefix}_break_pos.npy", break_pos)         # break-centre location (0..1)
    np.savez(f"{save_prefix}_events.npz",
             open_frames=open_frames,
             v_change=v_chg, a_change=a_chg,
             period_frames=np.array([period], np.float32),
             openness_method=np.array(openness_method),
             gap_um=np.array([e["gap_um"] for e in events], np.float32),
             duration_ms=np.array([e["duration_ms"] for e in events], np.float32),
             lag_v_ms=np.array([e["lag_v_ms"] for e in events], np.float32),
             lag_a_ms=np.array([e["lag_a_ms"] for e in events], np.float32),
             between_va=np.array([e["between_va"] for e in events], bool))

    fig_path = make_leaflet_figure(stack, probs, openness, open_frames,
                                   v_z, a_z, v_peaks, a_peaks, fps,
                                   f"{save_prefix}_figure.png")

    # ── report (the validation goal) ──
    print(f"[analyze] OPENS (openness peaks, '{openness_method}'): {len(open_frames)} "
          f"(target {expected_opens}) at frames {list(open_frames)}")
    print(f"[analyze] cardiac period ~{period:.0f} frames "
          f"({period / fps * 1e3:.0f} ms); recurrence ~{recur_frames:.0f} frames "
          f"({recur_frames / fps * 1e3:.0f} ms)")
    print(f"[analyze] V max-change frames {list(v_chg)}  A max-change frames {list(a_chg)}")
    print(f"[analyze] opens between V/A max-change: {n_between}/{len(open_frames)}  "
          f"(CONSISTENCY CHECK only: the connector detector reproduces the human "
          f"OPEN/CLOSED labels, so V/A agreement tests sensible cardiac phasing - "
          f"NOT an independent physical measurement)")
    for k, e in enumerate(events, 1):
        print(f"  open {k}: frame {e['frame']}  gap {e['gap_um']:.1f} um  "
              f"dur {e['duration_ms']:.1f} ms  lagV {e['lag_v_ms']:+.1f} ms  "
              f"lagA {e['lag_a_ms']:+.1f} ms  "
              f"{'BETWEEN V/A' if e['between_va'] else 'off-window'}  "
              f"upper={'A' if e['upper_is_a'] else 'B'}")
    print(f"[analyze] valve closed for {frac_closed*100:.0f}% of frames (expected 75-85%)")
    return dict(open_frames=open_frames, gap_px=gap_s, openness=openness,
                length_px=length_s, angle_deg=angle_s, ctrl_a=cp_a, ctrl_b=cp_b,
                v_peaks=v_peaks, a_peaks=a_peaks, v_change=v_chg, a_change=a_chg,
                midpoint=(my, mx), midpoint_darkening=dark_mid,
                events=events, period_frames=period, recur_frames=recur_frames,
                n_between=n_between, figure=fig_path, n_components=n_comp,
                frac_closed=frac_closed, um_per_pixel=um_per_pixel)


# ============================================================
# PRIMARY openness: INTENSITY M-MODE / DARKENING (no leaflet segmentation needed)
# ============================================================
def _xcorr_lag(a, b, fps, max_lag_frames):
    """Best Pearson correlation of a vs b over integer lags in
    [-max_lag, +max_lag]. Returns (best_corr, lag_ms) where a positive lag means
    ``a`` follows ``b`` by that many ms."""
    a = _zscore(np.asarray(a, float)); b = _zscore(np.asarray(b, float))
    n = len(a); best_c, best_l = 0.0, 0
    for lag in range(-int(max_lag_frames), int(max_lag_frames) + 1):
        if lag >= 0:
            x, y = a[lag:], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:]
        if len(x) > 4 and x.std() > 1e-9 and y.std() > 1e-9:
            c = float(np.corrcoef(x, y)[0, 1])
            if abs(c) > abs(best_c):
                best_c, best_l = c, lag
    return best_c, best_l / fps * 1e3


def analyze_valve_intensity(stack: np.ndarray, roi_mask: np.ndarray, line_pts,
                            ventricle_trace_path: str = "__none__",
                            atrium_trace_path: str = "__none__",
                            fps: float = FPS_DEFAULT,
                            um_per_pixel: Optional[float] = None,
                            ref_mask=None, save_prefix: str = "valve_int",
                            detrend: bool = True, baseline_pct: float = 90.0,
                            open_thr_sd: float = 1.0, kymo_halfwidth: int = 1,
                            expected_opens: int = EXPECTED_OPENINGS,
                            progress_cb=None) -> dict:
    """INTENSITY / M-MODE openness (the reliably-observable signal). openness(t) =
    NORMALIZED DARKENING of the marked valve ROI: dF/F against a bright baseline,
    optionally REFERENCE-corrected (ratio to a nearby region) and DETRENDED
    (photobleaching/motion). OPEN = darker. Builds the M-mode KYMOGRAPH along the
    valve LINE, detects EXACTLY ``expected_opens`` darkening peaks timed between the
    V/A max-change points, and CROSS-CORRELATES + phase-locks openness with V/A.

    Saves (npy/npz): <prefix>_{openness,dff,gap_um,open_frames,kymograph,
    ventricle,atrium}.npy and <prefix>_{events,kymo_meta}.npz. Returns a summary."""
    if um_per_pixel is None:
        um_per_pixel = UM_PER_PIXEL
    stack = np.clip(np.asarray(stack, np.float32), 0, 1)
    T, H, W = stack.shape
    roi = np.asarray(roi_mask, bool)
    if roi.shape != (H, W):
        raise ValueError(f"roi_mask shape {roi.shape} != frame {(H, W)}")
    if int(roi.sum()) < 4:
        raise ValueError("valve ROI has < 4 pixels - mark a larger region.")
    print(f"[intensity] valve ROI {int(roi.sum())} px; reference="
          f"{'YES' if ref_mask is not None else 'none'}; detrend={detrend}")

    # ── raw valve-region intensity over time + optional reference correction ──
    F = stack[:, roi].mean(axis=1).astype(np.float64)         # (T,)
    if ref_mask is not None and np.asarray(ref_mask, bool).sum() >= 4:
        R = stack[:, np.asarray(ref_mask, bool)].mean(axis=1).astype(np.float64)
        Fc = F / (R + 1e-6)                                   # remove common-mode
    else:
        Fc = F.copy()

    # ── V/A movement signals: contractions, max-change, robust cardiac period ──
    v_peaks = a_peaks = np.array([], dtype=int)
    v_z = a_z = None; v_sig = a_sig = None
    v_chg = a_chg = np.array([], dtype=int)
    if os.path.exists(ventricle_trace_path):
        v_sig = np.load(ventricle_trace_path)
        v_peaks, v_z = detect_contractions(v_sig, VENTRICLE_Z_THRESHOLD)
        v_chg, _ = signal_max_change_points(v_sig, expected_opens)
        np.save(f"{save_prefix}_ventricle.npy", v_sig)
    if os.path.exists(atrium_trace_path):
        a_sig = np.load(atrium_trace_path)
        a_peaks, a_z = detect_contractions(a_sig, ATRIUM_Z_THRESHOLD)
        a_chg, _ = signal_max_change_points(a_sig, expected_opens)
        np.save(f"{save_prefix}_atrium.npy", a_sig)
    period = cardiac_period_frames(v_peaks, a_peaks, -Fc, T, expected_opens)

    # ── dF/F: detrend slow photobleaching, then a bright baseline (closed) ──
    if detrend:
        win = max(15, int(round(1.6 * period)) | 1)          # > 1 cycle -> keeps cardiac
        trend = ndimage.uniform_filter1d(Fc, size=win, mode="nearest")
        Fc_dt = Fc - trend + float(np.mean(Fc))
    else:
        Fc_dt = Fc
    F0 = float(np.percentile(Fc_dt, baseline_pct))           # bright = closed baseline
    dff = (Fc_dt - F0) / (abs(F0) + 1e-9)                    # negative when darker
    openness = ndimage.uniform_filter1d(_zscore(-dff), 3)    # OPEN = darker -> positive
    # a frame is a clear OPEN (darkening) when openness clears a tunable SD threshold
    thr = openness.mean() + open_thr_sd * openness.std()
    break_flag = openness >= thr

    # ── exactly N opens, one per cardiac cycle, V/A-timed ──
    open_frames = exactly_n_opens(openness, break_flag, v_peaks, a_peaks, T, expected_opens)

    # ── M-mode kymograph along the valve LINE (space x time) ──
    (y0, x0), (y1, x1) = np.asarray(line_pts, float)
    L = max(2, int(round(np.hypot(y1 - y0, x1 - x0))))
    ts = np.linspace(0, 1, L)
    ly = y0 + ts * (y1 - y0); lx = x0 + ts * (x1 - x0)
    # perpendicular unit (for a small averaging band of half-width kymo_halfwidth)
    d = np.array([y1 - y0, x1 - x0], float); d /= (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0]])
    kymo = np.zeros((L, T), np.float32)
    offs = range(-kymo_halfwidth, kymo_halfwidth + 1)
    for k, (yy, xx) in enumerate(zip(ly, lx)):
        acc = np.zeros(T, np.float64); nb = 0
        for o in offs:
            yi = int(round(yy + o * perp[0])); xi = int(round(xx + o * perp[1]))
            if 0 <= yi < H and 0 <= xi < W:
                acc += stack[:, yi, xi]; nb += 1
        kymo[k] = (acc / max(1, nb)).astype(np.float32)

    # ── cross-correlation of openness with V and A (corr + lag ms) ──
    maxlag = int(round(0.5 * period))
    corr_v = corr_a = float("nan"); lag_v_corr = lag_a_corr = float("nan")
    if v_z is not None:
        corr_v, lag_v_corr = _xcorr_lag(openness, v_z[:T], fps, maxlag)
    if a_z is not None:
        corr_a, lag_a_corr = _xcorr_lag(openness, a_z[:T], fps, maxlag)

    # ── per-event metrics (darkening magnitude, duration, lag vs V/A max-change) ──
    def _nearest(arr, f):
        return None if len(arr) == 0 else int(arr[int(np.argmin(np.abs(np.asarray(arr) - f)))])
    open_thr2 = np.percentile(openness, 60)
    events = []
    for f in open_frames:
        s = f
        while s > 0 and openness[s - 1] >= open_thr2:
            s -= 1
        e = f
        while e < T - 1 and openness[e + 1] >= open_thr2:
            e += 1
        vt, at = _nearest(v_chg, f), _nearest(a_chg, f)
        between = (vt is not None and at is not None and
                   (min(vt, at) - 3) <= f <= (max(vt, at) + 3))
        events.append(dict(frame=int(f),
                           darkening_pct=float(-dff[f] * 100.0),
                           gap_um=float(-dff[f] * 100.0),     # "magnitude" proxy (% darkening)
                           duration_frames=int(e - s + 1),
                           duration_ms=float((e - s + 1) / fps * 1e3),
                           lag_v_ms=(float((f - vt) / fps * 1e3) if vt is not None else float("nan")),
                           lag_a_ms=(float((f - at) / fps * 1e3) if at is not None else float("nan")),
                           v_change_frame=(vt if vt is not None else -1),
                           a_change_frame=(at if at is not None else -1),
                           between_va=bool(between)))
    inter = np.diff(open_frames) if len(open_frames) >= 2 else np.array([])
    recur = float(np.mean(inter)) if len(inter) else float("nan")
    n_between = sum(e["between_va"] for e in events)
    thr_closed = np.percentile(openness, 25) + 0.15 * (openness.max() - np.percentile(openness, 25))
    frac_closed = float(np.mean(openness <= thr_closed))

    # ── save outputs (figure-compatible names + intensity extras) ──
    np.save(f"{save_prefix}_openness.npy", openness)
    np.save(f"{save_prefix}_break_signal.npy", openness)
    np.save(f"{save_prefix}_dff.npy", dff.astype(np.float32))
    np.save(f"{save_prefix}_open_frames.npy", open_frames)
    np.save(f"{save_prefix}_gap_um.npy", (-dff * 100.0).astype(np.float32))   # % darkening
    np.save(f"{save_prefix}_kymograph.npy", kymo)
    np.savez(f"{save_prefix}_kymo_meta.npz", line=np.asarray(line_pts, float),
             L=np.array([L]), um_per_pixel=np.array([um_per_pixel]),
             roi=roi, ref=(np.asarray(ref_mask, bool) if ref_mask is not None
                           else np.zeros((H, W), bool)))
    np.savez(f"{save_prefix}_events.npz",
             open_frames=open_frames, v_change=v_chg, a_change=a_chg,
             period_frames=np.array([period], np.float32),
             openness_source=np.array("intensity"),
             gap_um=np.array([e["gap_um"] for e in events], np.float32),
             duration_ms=np.array([e["duration_ms"] for e in events], np.float32),
             lag_v_ms=np.array([e["lag_v_ms"] for e in events], np.float32),
             lag_a_ms=np.array([e["lag_a_ms"] for e in events], np.float32),
             between_va=np.array([e["between_va"] for e in events], bool),
             corr_v=np.array([corr_v]), lag_v_corr_ms=np.array([lag_v_corr]),
             corr_a=np.array([corr_a]), lag_a_corr_ms=np.array([lag_a_corr]))

    # ── report (the scientific check) ──
    print(f"[intensity] OPENS (darkening peaks): {len(open_frames)} (target "
          f"{expected_opens}) at {list(open_frames)}")
    print(f"[intensity] cardiac period ~{period:.0f} frames "
          f"({period / fps * 1e3:.0f} ms); recurrence ~{recur:.0f} frames")
    print(f"[intensity] opens between V/A max-change: {n_between}/{len(open_frames)}")
    print(f"[intensity] x-corr openness vs V: r={corr_v:+.2f} at lag {lag_v_corr:+.0f} ms; "
          f"vs A: r={corr_a:+.2f} at lag {lag_a_corr:+.0f} ms")
    for k, e in enumerate(events, 1):
        print(f"  open {k}: frame {e['frame']}  darkening {e['darkening_pct']:.1f}%  "
              f"dur {e['duration_ms']:.0f} ms  lagV {e['lag_v_ms']:+.0f}  "
              f"lagA {e['lag_a_ms']:+.0f} ms  "
              f"{'BETWEEN V/A' if e['between_va'] else 'off-window'}")
    print(f"[intensity] valve clearly closed ~{frac_closed*100:.0f}% of frames")
    return dict(openness=openness, dff=dff, open_frames=open_frames, kymograph=kymo,
                v_peaks=v_peaks, a_peaks=a_peaks, v_change=v_chg, a_change=a_chg,
                period_frames=period, recur_frames=recur, events=events,
                corr_v=corr_v, lag_v_corr_ms=lag_v_corr, corr_a=corr_a,
                lag_a_corr_ms=lag_a_corr, n_between=n_between, frac_closed=frac_closed,
                um_per_pixel=um_per_pixel)


# ============================================================
# Stage 4: render two-leaflet overlay  (entry point)
# ============================================================
def render_leaflet_overlay(stack: np.ndarray, model_path: str,
                           crop_roi: Optional[tuple] = None,
                           save_path: str = "leaflet_overlay.tif",
                           save_labels: bool = True,
                           dashed: bool = OVERLAY_DASHED,
                           thickness: int = OVERLAY_LINE_PX,
                           montage_n: int = 6, tta: bool = True,
                           mark_open: bool = True,
                           open_threshold_px: Optional[float] = None,
                           progress_cb=None) -> str:
    """Write a contiguous RGB TIFF of the two-leaflet centerline overlay on the
    restored grayscale image, frame-by-frame (memory-safe). Two distinct colors;
    OPEN frames (tip-to-tip gap >= open_threshold_px, default GAP_OPEN_PX) get a
    dot on each orifice tip and a green frame border so open vs closed is obvious.
    Optionally also a uint8 label TIFF (1/2 per leaflet). Returns the TIFF path."""
    from scipy.spatial.distance import cdist
    if open_threshold_px is None:
        open_threshold_px = GAP_OPEN_PX
    import tifffile
    from tqdm import tqdm
    stack = np.clip(stack.astype(np.float32), 0, 1)
    if crop_roi is not None:
        y0, x0, y1, x1 = crop_roi
        stack = stack[:, y0:y1, x0:x1]
    T, H, W = stack.shape
    print(f"[overlay] inferring {T} frames ...")
    probs = predict_probabilities(stack, model_path, tta=tta, op="overlay",
                                  progress_cb=progress_cb)

    def _frame_info(pr):                          # 3-class/2-channel leaflets (or legacy 1-ch)
        return (frame_leaflets(pr)[0] if pr.ndim == 3
                else split_two_leaflets(pr > PROB_THRESHOLD))

    label_path = os.path.splitext(save_path)[0] + "_labels.tif"
    montage_frames, montage_probs = [], []
    pick = np.linspace(0, T - 1, montage_n).astype(int)

    with tifffile.TiffWriter(save_path, bigtiff=True) as tw, \
         (tifffile.TiffWriter(label_path, bigtiff=True) if save_labels else _Null()) as lw:
        for t in tqdm(range(T), desc="overlay write", unit="frame"):
            info = _frame_info(probs[t])
            rgb = _restored_rgb(stack[t])
            # EXTENDED centerlines: CLOSED reads as joined, OPEN keeps a gap
            _ea, _eb = extend_leaflet_pair(info["leaflet_a"], info["leaflet_b"])
            _draw_polyline_color(rgb, _ea, _hex2rgb(C_LEAFLET_A), thickness, dashed)
            _draw_polyline_color(rgb, _eb, _hex2rgb(C_LEAFLET_B), thickness, dashed)
            # OPEN-state cues: dot each orifice tip + green border so the gap reads
            a, b = info["leaflet_a"], info["leaflet_b"]
            if (mark_open and info["gap_px"] >= open_threshold_px
                    and a is not None and b is not None):
                ea = np.array([a[0], a[-1]]); eb = np.array([b[0], b[-1]])
                i, j = np.unravel_index(int(cdist(ea, eb).argmin()), (2, 2))
                for tip in (ea[i], eb[j]):
                    rr, cc = draw_disk((float(tip[0]), float(tip[1])),
                                       max(2, thickness), shape=(H, W))
                    rgb[rr, cc] = (255, 212, 0)
                gb = (44, 160, 44)
                rgb[:2, :] = gb; rgb[-2:, :] = gb; rgb[:, :2] = gb; rgb[:, -2:] = gb
            tw.write(rgb, contiguous=True, photometric="rgb")
            if save_labels:
                lab = np.zeros((H, W), np.uint8)
                if info["leaflet_a"] is not None:
                    lab |= rasterize_polyline(info["leaflet_a"], (H, W), thickness)
                if info["leaflet_b"] is not None:
                    lab |= rasterize_polyline(info["leaflet_b"], (H, W), thickness) * 2
                lw.write(lab, contiguous=True)
            if t in pick:
                montage_frames.append(t); montage_probs.append(probs[t])

    # still montage figure
    set_pub_style()
    ncol = len(montage_frames)
    fig, axes = plt.subplots(1, ncol, figsize=(1.15 * ncol, 1.35))
    if ncol == 1:
        axes = [axes]
    for ax, t, pr in zip(axes, montage_frames, montage_probs):
        ax.imshow(stack[t], cmap="gray", vmin=0, vmax=1)
        info = _frame_info(pr)
        ea, eb = extend_leaflet_pair(info["leaflet_a"], info["leaflet_b"])
        for cl, c in ((ea, C_LEAFLET_A), (eb, C_LEAFLET_B)):
            if cl is not None:
                ax.plot(cl[:, 1], cl[:, 0], color=c, lw=1.8, solid_capstyle="round")
        ax.set_title(f"{t/FPS_DEFAULT*1e3:.0f} ms" if FPS_DEFAULT else f"f{t}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    add_scale_bar(axes[-1], 20.0, W)
    save_vector(fig, os.path.splitext(save_path)[0] + "_montage.png")
    plt.close(fig)
    print(f"[overlay] wrote {save_path}" + (f" + {label_path}" if save_labels else ""))
    return save_path


def _hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, *a, **k): pass


# ============================================================
# Optional standalone CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Leaflet U-Net line-structure pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("annotate"); pa.add_argument("tif"); pa.add_argument("ds")
    pa.add_argument("--every", type=int, default=30)
    pt = sub.add_parser("train"); pt.add_argument("ds"); pt.add_argument("--epochs", type=int, default=100)
    pn = sub.add_parser("analyze"); pn.add_argument("tif"); pn.add_argument("model")
    pn.add_argument("--atrium", default="__none__"); pn.add_argument("--ventricle", default="__none__")
    po = sub.add_parser("overlay"); po.add_argument("tif"); po.add_argument("model")
    po.add_argument("--out", default="leaflet_overlay.tif")
    args = ap.parse_args()

    if args.cmd == "annotate":
        annotate_frames(args.tif, args.ds, every=args.every)
    elif args.cmd == "train":
        train_unet(args.ds, epochs=args.epochs)
    else:
        import tifffile
        st = tifffile.imread(args.tif).astype(np.float32)
        lo, hi = np.percentile(st, [1, 99.5]); st = np.clip((st - lo) / (hi - lo + 1e-8), 0, 1)
        if args.cmd == "analyze":
            analyze_line_breaks(st, args.model, args.atrium, args.ventricle)
        else:
            render_leaflet_overlay(st, args.model, save_path=args.out)
