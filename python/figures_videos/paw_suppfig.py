#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - paw supplementary figure + SBR/CNR/FWHM vs depth - produces Supp. Fig. 6 (SBR/CNR vs depth), Supp. Fig. 7
# environment: heart_valve_py314
"""
paw_suppfig.py (v2) — NIR-II SLIM: Supplementary Figure panels + Supplementary Video from an
ALREADY-reconstructed time series (record_*_rl.mat), using ONE representative volume (middle frame).

ONE-COMMAND BATCH (sweep -> automatic pick -> your picks, each setting in its own folder):
  python paw_suppfig.py --batch Dynamic --fps 10 --lat-res-um 22 --mip-smooth-px 1.0
     -> <Dynamic>/suppfig_batch/<stem>/sweep/ (+ mip_contact_sheet_sweep.png), auto_<setting>/ (full outputs with the
        automatic pick), pick_<setting>/ (your picks typed at the prompt, or from --picks-file id,bg,min,gamma).
        Masks are picked up from <stem>_suppfig/Mask.tif etc. (--mask-pattern). Re-running reuses finished sweeps.

Workflow (per file)
  0) STUDIO (recommended): python paw_suppfig.py --recon record_..._rl.mat --raw record_....raw --fps 10 --gui
                   -> one window: draw/refine the mask (polygon / ellipse / brush), set background removal, |z| range,
                      min/max, gamma, smoothing, upsampling by eye (Gray / Depth-coded / Binned previews),
                      then EXPORT -> writes mask.png + mip_display.json and runs the unchanged pipeline below.
  1) inspect       python paw_suppfig.py --recon record_..._rl.mat --raw record_....raw --dry-run
                   -> prints .mat variables/shape/chunking, T, chosen frame, D, dz, raw frame count, fps evidence
  2) representative-frame mask (choose one)
       a) built-in drawing tool:   ... --draw-mask            (polygon: click points, click first point to close;
                                                               press 'e' for ellipse mode, drag; Enter = done)
       b) Fiji:                    ... --export-frame          -> rep_frame_mip_16bit.tif; draw selection in Fiji,
                                                               Edit > Selection > Create Mask, save mask.png/.tif
                                   ... --mask-file mask.png
  3) full run      python paw_suppfig.py --recon ..._rl.mat --raw ....raw --mask-file mask.png --px-um 4.0
                   (add --no-video to iterate on the panels first)

Geometry conventions (as agreed)
  * plane spacing dz: the .mat step variable (depth_size / dz / z_step ...) when present, else the D planes are
    assumed to span --z-range-um (600 µm): dz = span/(D-1).  Explicit --dz-um <µm> overrides both.
  * z = 0 at the CENTRE plane (--surface-index center); planes run from -z_range/2 to +z_range/2
  * representative frame = middle frame of the time series (--t mid); no time averaging
  * .mat v7.3 (HDF5) is read LAZILY: only the chosen frame is loaded (needs h5py: pip install h5py)

Outputs (all in --out, default <stem>_suppfig/)
  panels as SVG/PDF/PNG (Arial 7 pt, svg.fonttype none) + clean 8-bit PNG / 16-bit TIF image layers
  source-data CSVs (FWHM measurements + per-plane summary, SBR/CNR per plane)
  <stem>_suppfig.pptx : native XY charts (d, e), pictures with native scale bars / labels, parameter table  (python-pptx)
  panel_d_detections_overlay.png/.svg : where the FWHM profiles were measured (transparent overlay)
  SuppVideo_*.mp4 : fly-through -> MIP -> depth-coded MIP -> 360° turntable (1920x1080, H.264)
  figure_metadata.json, run_log.txt

MIP display (b, b', binned MIPs, video MIP/turntable) — DISPLAY ONLY, never touches the quantification:
  --mip-bg-method gauss|tophat + --mip-bg-radius-um (per-plane local background removal), --mip-z-range (drop boundary planes),
  MIN/MAX black-white levels (--mip-min/--mip-max, floats or 'pNN' percentiles of the MIP inside the mask), --mip-gamma,
  per-plane 2-D Gaussian smoothing before the MIP (--mip-smooth-px), bicubic upsampling (--mip-upsample, default 2x).
  Trial grid: --sweep [--sweep-gamma 0.7:1.5:0.1 --sweep-min 40:70:5 --sweep-bg none,tophat] -> <out>/sweep/ PNGs +
  contact sheets + sweep_index.csv with the CLI flags reproducing each image; pick one and re-run with those flags.
  Check by eye with --tune-mip (sliders incl. bg radius, 'b' cycles the bg method; Enter = accept -> mip_display.json,
  auto-reused) or --mip-sheet (contact sheet: bg method x gamma x min level). The log prints the MIP percentiles so the
  MIN/MAX can also be typed by hand. FOV mask: --draw-mask (polygon/ellipse) or --paint-mask (brush, refines --mask-file).

Conventions copied from main.py: (H,W,D,T) axis order and .mat variable candidates; percentile clip 0.5/99.9;
depth-coded MIP = turbo(z) x MIP^gamma with intensity-weighted depth smoothing (r=3, floor 0.03, median 3);
_perp_profile / _fwhm_from_profile; SBR = mean(top 1 %)/median(background), CNR with MAD sigma (Supplementary Fig. 6).
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
from scipy import ndimage as ndi

try:
    import cv2
except Exception:
    cv2 = None

MM = 1.0 / 25.4
OI = dict(blue='#0072B2', orange='#E69F00', green='#009E73', vermilion='#D55E00',
          sky='#56B4E9', purple='#CC79A7', yellow='#F0E442', black='#000000', grey='#8c8c8c')
MAT_CANDIDATES = ['recon', 'volume', 'result', 'data', 'vol', 'reconstruction', 'img', 'image',
                  'Recon', 'Volume', 'Result', 'Data', 'Vol']
PARAM_NAMES = dict(fps=('fps', 'frame_rate', 'framerate', 'FPS', 'hz', 'Hz', 'frameRate'),
                   dz=('depth_size', 'dz', 'z_step', 'zstep', 'dz_um', 'step_um', 'zStep', 'depth_step', 'z_size', 'zsize', 'plane_spacing'),
                   zrange=('z_range', 'zrange', 'zRange', 'z_range_um', 'depth_range'),
                   iters=('iter', 'iters', 'n_iter', 'niter', 'iterations', 'rl_iter'),
                   px=('pixel_size', 'px_um', 'pixelsize', 'pixel_size_um', 'pixelSize', 'dx', 'dx_um'))


# ════════════════════════════════════════════════════════════════════════════
# 1. Source inspection + lazy loading of ONE representative volume
# ════════════════════════════════════════════════════════════════════════════
def _is_mat73(path):
    with open(path, 'rb') as f:
        head = f.read(128)
    return b'MATLAB 7.3' in head or b'HDF' in head[:16]


def _pick_dataset(names_shapes, var):
    """names_shapes: {name: shape}. Returns chosen name."""
    if var and var in names_shapes:
        return var
    for c in MAT_CANDIDATES:
        if c in names_shapes and len(names_shapes[c]) >= 3:
            return c
    best, bname = 0, None
    for k, shp in names_shapes.items():
        if len(shp) >= 3 and int(np.prod(shp)) > best:
            best, bname = int(np.prod(shp)), k
    return bname


def _axis_roles(shape_matlab, axes_override, log):
    """shape_matlab: shape in MATLAB (column-major) order. Returns dict role->axis index (H,W,D[,T])."""
    nd = len(shape_matlab)
    if axes_override:
        a = axes_override.upper()
        if len(a) != nd:
            raise SystemExit(f"--axes '{a}' has {len(a)} letters but the array is {nd}-D (MATLAB order {shape_matlab})")
        roles = {c: a.index(c) for c in 'HWDT' if c in a}
        log(f"  axes override (MATLAB order): {a}")
        return roles
    if nd == 3:
        return dict(H=0, W=1, D=2)
    s = list(shape_matlab)
    # main.py accepts (H,W,D,T) [recon app default] or (T,H,W,D). Default to (H,W,D,T); switch to (T,H,W,D)
    # only when axis 0 is the largest AND axis 3 the smallest extent.
    if s[0] > max(s[1], s[2]) and s[3] < min(s[1], s[2]):
        roles = dict(T=0, H=1, W=2, D=3); order = '(T,H,W,D)'
    else:
        roles = dict(H=0, W=1, D=2, T=3); order = '(H,W,D,T)'
    log(f"  4-D order assumed {order}: H={s[roles['H']]}, W={s[roles['W']]}, D={s[roles['D']]}, T={s[roles['T']]}  (override with --axes)")
    if s[roles['D']] > s[roles['T']]:
        log("  !! D > T: check the order — pass --axes if this is wrong")
    return roles


def _scalar_params(getter_items, log):
    """Collect small numeric/string variables that look like acquisition/recon parameters."""
    found = {}
    for k, v in getter_items:
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.size == 0 or arr.size > 16:
            continue
        if arr.dtype.kind in 'iuf':
            val = arr.astype(float).ravel().tolist()
            found[k] = val[0] if len(val) == 1 else val
        elif arr.dtype.kind in 'SU':
            found[k] = str(arr.ravel()[0])
    if found:
        log(f"  small variables (possible parameters): {found}")
    return found


def load_representative_volume(path, var=None, axes=None, t_mode='mid', log=print):
    """Returns raw (H,W,D) float32 for ONE frame + info dict. v7.3 files are read lazily."""
    ext = os.path.splitext(path)[1].lower()
    info = dict(path=path, params={})
    log(f"[load] {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    if ext == '.mat' and _is_mat73(path):
        try:
            import h5py
        except ImportError:
            raise SystemExit("This is a MATLAB v7.3 (HDF5) file; install h5py:  pip install h5py")
        with h5py.File(path, 'r') as f:
            names = {}
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    names[k] = tuple(f[k].shape)
                    log(f"  dataset '{k}': stored {f[k].shape} {f[k].dtype} chunks={f[k].chunks} compression={f[k].compression}")
            name = _pick_dataset(names, var)
            if name is None:
                raise SystemExit("no >=3-D dataset found")
            ds = f[name]
            stored = tuple(ds.shape)
            matlab_shape = stored[::-1]                       # HDF5 stores MATLAB arrays reversed
            log(f"  using '{name}': MATLAB shape {matlab_shape} (stored {stored})")
            roles = _axis_roles(matlab_shape, axes, log)
            nd = len(matlab_shape)
            if 'T' in roles:
                T = matlab_shape[roles['T']]
                t = T // 2 if str(t_mode) == 'mid' else int(t_mode)
                t = max(0, min(t, T - 1))
                sl = [slice(None)] * nd
                sl[nd - 1 - roles['T']] = t                   # stored axis of T
                t0 = time.time()
                vol = np.asarray(ds[tuple(sl)], dtype=np.float32)   # only this frame is read
                log(f"  read frame {t}/{T - 1} in {time.time() - t0:.1f} s (stored slice axis {nd - 1 - roles['T']})")
                rem = [ax for ax in range(nd) if ax != roles['T']]  # MATLAB axes left, in MATLAB order
                stored_order = rem[::-1]                             # ...as they appear in the stored (reversed) array
                perm = [stored_order.index(roles[c]) for c in ('H', 'W', 'D')]
                raw = vol.transpose(perm)
                info.update(T=int(T), t_used=int(t))
            else:
                vol = np.asarray(ds[()], dtype=np.float32)
                stored_order = list(range(nd))[::-1]
                perm = [stored_order.index(roles[c]) for c in ('H', 'W', 'D')]
                raw = vol.transpose(perm); info.update(T=1, t_used=0)
            info['params'] = _scalar_params(((k, f[k][()]) for k in f.keys()
                                             if isinstance(f[k], h5py.Dataset) and f[k].size <= 16), log)
            info.update(variable=name, format='mat v7.3 (HDF5, lazy)')
    elif ext == '.mat':
        import scipy.io as sio
        mat = sio.loadmat(path)
        names = {k: np.asarray(v).shape for k, v in mat.items() if not k.startswith('__') and isinstance(v, np.ndarray)}
        log(f"  variables: {names}")
        name = _pick_dataset(names, var)
        if name is None:
            raise SystemExit("no >=3-D variable found")
        vol = np.asarray(mat[name])
        roles = _axis_roles(vol.shape, axes, log)
        if 'T' in roles:
            T = vol.shape[roles['T']]
            t = T // 2 if str(t_mode) == 'mid' else int(t_mode); t = max(0, min(t, T - 1))
            vol = np.take(vol, t, axis=roles['T'])
            rem = [ax for ax in range(4) if ax != roles['T']]
            perm = [rem.index(roles[c]) for c in ('H', 'W', 'D')]
            raw = vol.transpose(perm); info.update(T=int(T), t_used=int(t))
        else:
            raw = vol.transpose([roles['H'], roles['W'], roles['D']]); info.update(T=1, t_used=0)
        info['params'] = _scalar_params(((k, v) for k, v in mat.items() if not k.startswith('__')), log)
        info.update(variable=name, format='mat v5 (scipy)')
    elif ext in ('.tif', '.tiff'):
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 3:
            raw = arr.transpose(1, 2, 0); info.update(T=1, t_used=0)
        elif arr.ndim == 4:                                   # (T,D,H,W)
            T = arr.shape[0]; t = T // 2 if str(t_mode) == 'mid' else int(t_mode)
            raw = arr[t].transpose(1, 2, 0); info.update(T=int(T), t_used=int(t))
        else:
            raise SystemExit(f"unexpected TIFF shape {arr.shape}")
        info.update(variable='tiff', format='tiff (ImageJ order)')
    elif ext == '.npy':
        arr = np.load(path, mmap_mode='r')
        roles = _axis_roles(arr.shape, axes, log)
        if 'T' in roles:
            T = arr.shape[roles['T']]; t = T // 2 if str(t_mode) == 'mid' else int(t_mode)
            vol = np.take(arr, t, axis=roles['T']); rem = [ax for ax in range(4) if ax != roles['T']]
            raw = np.asarray(vol).transpose([rem.index(roles[c]) for c in ('H', 'W', 'D')]); info.update(T=int(T), t_used=int(t))
        else:
            raw = np.asarray(arr).transpose([roles['H'], roles['W'], roles['D']]); info.update(T=1, t_used=0)
        info.update(variable='npy', format='npy')
    else:
        raise SystemExit(f"unsupported extension {ext}")
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    raw[~np.isfinite(raw)] = 0
    info['shape_HWD'] = list(raw.shape)
    log(f"  -> representative volume (H,W,D) = {raw.shape}; frame {info['t_used']} of {info['T']}; range [{raw.min():g}, {raw.max():g}]")
    return raw, info


def display_volume(raw, pct_lo=0.5, pct_hi=99.9, bg_pct=None, mask=None):
    """main.py normalisation: percentile clip -> [0,1]. bg_pct = DISPLAY-only per-plane background subtraction."""
    raw = raw.astype(np.float32, copy=True)
    if bg_pct is not None:
        m = mask if mask is not None else np.ones(raw.shape[:2], bool)
        for z in range(raw.shape[2]):
            b = float(np.percentile(raw[:, :, z][m], bg_pct))
            raw[:, :, z] = np.maximum(raw[:, :, z] - b, 0)
    sample = raw.ravel()[::max(1, raw.size // 1_000_000)]
    vmin = float(np.percentile(sample, pct_lo)); vmax = float(np.percentile(sample, pct_hi))
    disp = np.clip(raw, vmin, vmax); disp -= vmin
    if vmax > vmin:
        disp /= (vmax - vmin)
    return disp.astype(np.float32), (vmin, vmax)


# ════════════════════════════════════════════════════════════════════════════
# 2. Raw recording check (frame count, tag diagnostics, fps evidence)
# ════════════════════════════════════════════════════════════════════════════
def check_raw(raw_path, shape=(640, 512), dtype='int16', n_probe=64, log=print):
    """Headerless C-RED 2 raw (int16, W fastest; main.py _save_denoised writes the same layout)."""
    W, H = int(shape[0]), int(shape[1])
    dt = np.dtype(dtype); fb = W * H * dt.itemsize
    size = os.path.getsize(raw_path)
    n_frames, rem = size // fb, size % fb
    out = dict(path=raw_path, bytes=int(size), frame_shape_WH=[W, H], dtype=str(dt), n_frames=int(n_frames), remainder_bytes=int(rem))
    log(f"[raw] {os.path.basename(raw_path)}: {size:,} bytes -> {n_frames} frames of {W}x{H} {dt}"
        + ("" if rem == 0 else f"  (!! {rem} bytes remainder: header/ROI/dtype differs from the assumed layout)"))
    n = int(min(n_probe, n_frames))
    if n >= 3:
        arr = np.fromfile(raw_path, dtype=dt, count=n * W * H).reshape(n, H, W)
        mid = arr[n // 2].astype(np.float64)
        out.update(probe_mean=float(mid.mean()), probe_min=float(mid.min()), probe_max=float(mid.max()))
        log(f"  probe frame stats: mean {mid.mean():.1f}, min {mid.min():g}, max {mid.max():g}")
        first = arr[:, 0, :8].astype(np.int64)
        log(f"  first 8 pixels of row 0, frames 0-2: {first[:3].tolist()}")
        tags = []
        for c in range(8):
            d = np.diff(first[:, c])
            if np.all(d > 0) and np.ptp(d) <= max(1, 0.01 * abs(d.mean())):
                tags.append(dict(pixel=c, increment=float(d.mean())))
        if tags:
            log(f"  monotonic leading pixels (camera tags?): {tags}  -> if a timestamp in µs, fps = 1e6/increment")
        out['tag_candidates'] = tags
    return out


def resolve_fps(args, raw_info, mat_params, log):
    """Priority: --fps > filename token (_100hz) > sidecar text files > .mat variables > None."""
    if args.fps:
        log(f"[fps] {args.fps} Hz (from --fps)"); return float(args.fps), '--fps'
    for p in (args.raw, args.recon):
        if not p:
            continue
        m = re.search(r'(\d+(?:\.\d+)?)\s*(hz|fps)', os.path.basename(p), re.IGNORECASE)
        if m:
            log(f"[fps] {m.group(1)} Hz (filename token in {os.path.basename(p)})"); return float(m.group(1)), 'filename'
    for p in (args.raw, args.recon):
        if not p:
            continue
        stem = os.path.splitext(p)[0]
        for ext in ('.txt', '.json', '.ini', '.log', '.xml', '.csv', '.md'):
            side = stem + ext
            if os.path.exists(side):
                try:
                    txt = open(side, errors='ignore').read()
                except Exception:
                    continue
                m = re.search(r'(?:fps|frame ?rate|hz)\D{0,20}(\d+(?:\.\d+)?)', txt, re.IGNORECASE) or \
                    re.search(r'(\d+(?:\.\d+)?)\s*(?:hz|fps)', txt, re.IGNORECASE)
                if m:
                    log(f"[fps] {m.group(1)} Hz (sidecar {os.path.basename(side)})"); return float(m.group(1)), os.path.basename(side)
    if args.sidecar and os.path.exists(args.sidecar):
        txt = open(args.sidecar, errors='ignore').read()
        m = re.search(r'(?:fps|frame ?rate|hz)\D{0,20}(\d+(?:\.\d+)?)', txt, re.IGNORECASE) or re.search(r'(\d+(?:\.\d+)?)\s*(?:hz|fps)', txt, re.IGNORECASE)
        if m:
            log(f"[fps] {m.group(1)} Hz (--sidecar {os.path.basename(args.sidecar)})"); return float(m.group(1)), os.path.basename(args.sidecar)
        log(f"[fps] --sidecar {args.sidecar}: no fps/frame-rate/Hz token found")
    if args.raw:
        tags = (raw_info or {}).get('tag_candidates') or []
        stem = os.path.splitext(args.raw)[0]
        log(f"[fps] raw is headerless (no fps inside); no same-stem sidecar found ({os.path.basename(stem)}.txt/.json/.ini/.log/.xml/.csv/.md); "
            + (f"camera tag candidates: {tags}" if tags else "no monotonic tag pixels in row 0"))
    for k in PARAM_NAMES['fps']:
        if k in mat_params and isinstance(mat_params[k], (int, float)):
            log(f"[fps] {mat_params[k]} Hz (.mat variable '{k}' written by the recon app — NOT read from the raw)"); return float(mat_params[k]), f".mat:{k}"
    log("[fps] NOT FOUND: pass --fps <Hz> for the caption (check the acquisition app/log).")
    return None, 'unknown'


# ════════════════════════════════════════════════════════════════════════════
# 3. Geometry / mask
# ════════════════════════════════════════════════════════════════════════════
def z_axis(D, dz_um, surface_index):
    return (np.arange(D) - float(surface_index)) * dz_um


def load_mask_file(path, H, W, log):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        m = np.load(path)
    elif ext in ('.tif', '.tiff'):
        import tifffile; m = tifffile.imread(path)
    else:
        from PIL import Image; m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (H, W):
        raise SystemExit(f"mask {m.shape} does not match volume (H,W)=({H},{W}); draw it on rep_frame_mip_16bit.tif of THIS file")
    hard = m > 0
    log(f"[mask] {os.path.basename(path)}: {hard.sum()} px inside ({100 * hard.mean():.1f} % of frame)")
    return hard


def build_mask(H, W, spec, raw=None, log=print):
    if spec is None or str(spec).lower() in ('none', ''):
        return np.ones((H, W), bool)
    if str(spec).lower() == 'detect':
        mip = raw.max(axis=2); sup = mip > (1e-6 * max(float(mip.max()), 1e-12))
        if sup.mean() > 0.995:
            log("  mask=detect: no masked region found -> 'auto'")
            cx, cy, r = W / 2.0, H / 2.0, min(H, W) / 2.0 - 5
        else:
            ys, xs = np.nonzero(sup); cx, cy = float(xs.mean()), float(ys.mean()); r = float(np.sqrt(sup.sum() / np.pi)) - 2
            log(f"  mask=detect: circle cx={cx:.1f}, cy={cy:.1f}, r={r:.1f} px")
    elif str(spec).lower() == 'auto':
        cx, cy, r = W / 2.0, H / 2.0, min(H, W) / 2.0 - 5
    else:
        cx, cy, r = [float(v) for v in str(spec).split(',')]
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(xx - cx, yy - cy) <= r


def soft_edge(hard, feather_px=3.0):
    if hard.all():
        return np.ones(hard.shape, np.float32)
    inside = ndi.distance_transform_edt(hard)
    return np.clip(inside / feather_px, 0, 1).astype(np.float32)


def _interactive_backend(log):
    """Switch matplotlib to a GUI backend that actually works here (probes by creating a figure)."""
    import matplotlib
    for be in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX', 'WXAgg'):
        try:
            matplotlib.use(be, force=True)
            import matplotlib.pyplot as plt
            f = plt.figure(); plt.close(f)
            return True
        except Exception as e:
            log(f"  backend {be} unavailable: {e.__class__.__name__}")
    return False


def draw_mask_interactive(img01, out_dir, log):
    """Polygon (default) or ellipse ('e') drawn on the representative MIP. Enter = accept."""
    if not _interactive_backend(log):
        raise SystemExit("no GUI backend (tkinter/Qt) available for --draw-mask: use --export-frame -> Fiji -> --mask-file instead")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector, EllipseSelector
    from matplotlib.path import Path as MplPath
    H, W = img01.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(img01, 0, 1) ** 0.5, cmap='gray', vmin=0, vmax=1)
    ax.set_title("polygon: click vertices, click the first vertex to close | 'e' ellipse (drag) | 'p' polygon | Enter = accept", fontsize=8)
    state = dict(mode='polygon', poly=None)

    def on_poly(verts):
        state['poly'] = [(float(x), float(y)) for x, y in verts]
    ps = PolygonSelector(ax, on_poly, useblit=False,
                         props=dict(color='yellow', linestyle='-', linewidth=1.5, alpha=0.9),
                         handle_props=dict(markersize=6, markerfacecolor='yellow', markeredgecolor='k'))
    es = EllipseSelector(ax, lambda a, b: None, useblit=False, interactive=True,
                         props=dict(facecolor='none', edgecolor='cyan', linewidth=1.5))
    es.set_active(False); es.set_visible(False)

    def on_key(ev):
        if ev.key == 'e':
            state['mode'] = 'ellipse'; ps.set_active(False); ps.set_visible(False); es.set_active(True); es.set_visible(True)
        elif ev.key == 'p':
            state['mode'] = 'polygon'; es.set_active(False); es.set_visible(False); ps.set_active(True); ps.set_visible(True)
        elif ev.key in ('enter', 'return'):
            plt.close(fig)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    yy, xx = np.mgrid[0:H, 0:W]
    if state['mode'] == 'ellipse' and es.extents and (es.extents[1] - es.extents[0]) > 2:
        x0, x1, y0, y1 = es.extents
        cx, cy, a, b = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
        spec = dict(type='ellipse', cx=cx, cy=cy, a=a, b=b)
    elif state['poly'] and len(state['poly']) >= 3:
        mask = MplPath(state['poly']).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(H, W)
        spec = dict(type='polygon', vertices=state['poly'])
    else:
        raise SystemExit("no mask drawn (close the polygon by clicking its first vertex, then press Enter)")
    from PIL import Image
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, 'mask.png'))
    np.save(os.path.join(out_dir, 'mask.npy'), mask)
    json.dump(spec, open(os.path.join(out_dir, 'mask.json'), 'w'), indent=1)
    log(f"[mask] drawn {spec['type']}: {mask.sum()} px -> mask.png / mask.npy / mask.json  (reuse with --mask-file)")
    return mask


# ════════════════════════════════════════════════════════════════════════════
# 4. Depth-coded MIP (Movie Viewer convention)
# ════════════════════════════════════════════════════════════════════════════
def parse_level(spec, mip, mask):
    """'p60' -> 60th percentile of the MIP inside the mask; otherwise a float in display units (0-1)."""
    if isinstance(spec, str) and spec.lower().startswith('p'):
        return float(np.percentile(mip[mask], float(spec[1:])))
    return float(spec)


def upsample_img(img, factor, order=3):
    """Bicubic (order 3) / linear (order 1) upsampling of a 2-D or (H,W,3) image; display only."""
    if factor <= 1:
        return img
    if img.ndim == 3:
        return np.stack([upsample_img(img[..., c], factor, order) for c in range(img.shape[2])], axis=2)
    if cv2 is not None:
        interp = cv2.INTER_CUBIC if order >= 3 else cv2.INTER_LINEAR
        return cv2.resize(np.ascontiguousarray(img, dtype=np.float32), None, fx=factor, fy=factor, interpolation=interp)
    return ndi.zoom(img, factor, order=order)


def mip_and_depth(disp, smooth_px=0.0, mode='max', focus_win_px=7.0, focus_sigma_px=1.5):
    """Projection of the display volume (after an optional per-plane 2-D Gaussian):
    'max'   : classic MIP (intensity max over z, depth = argmax)            — unchanged default
    'focus' : extended-depth-of-field: per pixel take the plane with the highest LOCAL FOCUS
              (|Laplacian-of-Gaussian| averaged over a window), intensity from that plane.
              Suppresses laterally shifted defocus ghosts of light-field reconstructions in the MIP
              and gives a depth map from the in-focus plane rather than the brightest copy."""
    vol = disp if smooth_px <= 0 else ndi.gaussian_filter(disp, sigma=(smooth_px, smooth_px, 0))
    if mode != 'focus':
        return vol.max(axis=2), vol.argmax(axis=2).astype(np.float32)
    F = np.empty_like(vol)
    for z in range(vol.shape[2]):
        F[:, :, z] = ndi.gaussian_filter(np.abs(ndi.gaussian_laplace(vol[:, :, z], focus_sigma_px)), max(focus_win_px, 0.5))
    zi = F.argmax(axis=2)
    proj = np.take_along_axis(vol, zi[:, :, None], axis=2)[:, :, 0]
    return proj, zi.astype(np.float32)


def proj_kw(mipd, px_um):
    """Projection keyword arguments from the MIP display settings (mip_and_depth)."""
    return dict(mode=mipd.get('proj_mode', 'max'), focus_win_px=float(mipd.get('focus_win_um', 40.0)) / px_um)


def colour_brightness(proj_n, mode='intensity', gamma=0.5, floor=0.5, thr=0.1, local_px=17):
    """Brightness that modulates the depth colour (proj_n = projection after black/white levels, 0-1).
    'intensity' : I^gamma (classic; dim vessels stay dim)
    'floor'     : smooth lift — floor*smoothstep(I/thr) + (1-floor)*I^gamma (no step; dim vessels keep their hue)
    'local'     : local normalisation — each vessel scaled to its neighbourhood maximum, so dim and bright
                  vessels reach similar brightness; speckle below thr is suppressed by the same smoothstep
    'flat'      : uniform colour for everything above thr (smoothstep edge)
    Bright pixels stay at full brightness in every mode; background (I = 0) stays black."""
    p = np.clip(proj_n, 0, 1)
    sm = np.clip(p / max(thr, 1e-6), 0, 1); sm = sm * sm * (3 - 2 * sm)
    if mode == 'floor':
        return np.clip(floor * sm + (1.0 - floor) * p ** gamma, 0, 1)
    if mode == 'local':
        lm = ndi.maximum_filter(p, size=2 * int(local_px) + 1)
        lm = ndi.gaussian_filter(lm, max(local_px / 2.0, 0.5))
        return np.clip(sm * (p / np.maximum(lm, thr)) ** gamma, 0, 1)
    if mode == 'flat':
        return sm
    return p ** gamma


def colour_kw(mipd):
    """Depth-coded brightness keyword arguments (depth_coded_mip / video)."""
    return dict(bright_mode=mipd.get('depth_bright', 'intensity'), bright_floor=float(mipd.get('depth_floor', 0.5) or 0.0),
                bright_gamma=mipd.get('depth_gamma'), bright_thr=float(mipd.get('depth_floor_thr', 0.1) or 0.0),
                bright_local_px=mipd.get('depth_local_px', 17))


def apply_levels(mip, lo, hi, gamma):
    """Black level lo -> 0, white level hi -> 1 (display units), then gamma."""
    return np.clip((mip - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma


def depth_coded_mip(disp, gamma=0.5, smooth_r=3, soft_mask=None, cmap_name='turbo', floor=0.0,
                    lo=0.0, hi=1.0, smooth_px=0.0, upsample=1, mode='max', focus_win_px=7.0, bright_mode='intensity', bright_floor=0.5,
                    bright_gamma=None, bright_thr=0.1, bright_local_px=17):
    """bright_floor: colour brightness = floor + (1 - floor) * I^bright_gamma for every pixel above the black level
    (0 = classic intensity modulation; 0.4-0.6 keeps the hue readable on dim vessels; 1 = flat colour). Background stays black."""
    """Movie Viewer convention (turbo(z) x MIP^gamma, intensity-weighted depth smoothing r=3, floor 0.03, median 3)
    + black/white levels (lo/hi), per-plane 2-D Gaussian smoothing and bicubic upsampling — all DISPLAY only.
    Returns rgb (h,w,3), gray MIP (h,w), depth_norm (h,w) at the upsampled size."""
    import matplotlib.pyplot as plt
    H, W, D = disp.shape
    cmap = plt.colormaps.get_cmap(cmap_name)
    proj_val, depth_idx = mip_and_depth(disp, smooth_px, mode=mode, focus_win_px=focus_win_px)
    proj_n = apply_levels(proj_val, lo, hi, 1.0)
    if floor > 0:
        proj_n = np.where(proj_n >= floor, proj_n, 0.0)
    proj_g_raw = proj_n ** gamma                                   # grey MIP brightness (returned)
    bgm = gamma if bright_gamma is None else bright_gamma
    colour_b = colour_brightness(proj_n, bright_mode, bgm, bright_floor, bright_thr, bright_local_px)
    soft = soft_mask if soft_mask is not None else np.ones((H, W), np.float32)
    if smooth_r > 1:
        weight = proj_g_raw.copy(); weight[weight < 0.03] = 0
        k = 2 * smooth_r + 1
        dsm = ndi.uniform_filter(depth_idx * weight, size=k); wsm = ndi.uniform_filter(weight, size=k)
        depth_final = np.divide(dsm, wsm, out=depth_idx.copy(), where=wsm > 1e-8)
        depth_u8 = (np.clip(depth_final / max(D - 1, 1), 0, 1) * 255).astype(np.uint8)
        mk = max(3, smooth_r | 1)
        depth_u8 = cv2.medianBlur(depth_u8, mk) if cv2 is not None else ndi.median_filter(depth_u8, size=mk)
        depth_final = depth_u8.astype(np.float32) / 255.0 * (D - 1)
    else:
        depth_final = depth_idx
    depth_norm = np.clip(depth_final / max(D - 1, 1), 0, 1)
    if upsample > 1:
        proj_g_raw = np.clip(upsample_img(proj_g_raw, upsample, 3), 0, 1)
        colour_b = np.clip(upsample_img(colour_b, upsample, 3), 0, 1)
        depth_norm = np.clip(upsample_img(depth_norm, upsample, 1), 0, 1)
        soft = np.clip(upsample_img(soft, upsample, 1), 0, 1)
    proj_g = (proj_g_raw * soft).astype(np.float32)
    rgb = (cmap(depth_norm)[:, :, :3] * (colour_b * soft)[:, :, None]).astype(np.float32)
    return rgb, proj_g, depth_norm.astype(np.float32)


def subtract_background(vol, method='none', radius_px=0):
    """DISPLAY ONLY. Per-plane local background removal before the MIP:
    'tophat' = white top-hat (rolling-ball-like) with a disk of radius_px — removes everything wider than ~2*radius
    'gauss'  = subtract a Gaussian-blurred copy (sigma = radius_px), clipped at 0 — gentler, keeps wide vessels
    'none'   = unchanged."""
    if method in (None, 'none') or radius_px <= 0:
        return vol
    r = int(round(radius_px)); out = np.empty_like(vol)
    if method == 'tophat':
        if cv2 is not None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            for z in range(vol.shape[2]):
                out[:, :, z] = cv2.morphologyEx(np.ascontiguousarray(vol[:, :, z], dtype=np.float32), cv2.MORPH_TOPHAT, k)
        else:
            from skimage.morphology import white_tophat, disk
            fp = disk(r)
            for z in range(vol.shape[2]):
                out[:, :, z] = white_tophat(vol[:, :, z], fp)
    elif method == 'gauss':
        for z in range(vol.shape[2]):
            out[:, :, z] = np.maximum(vol[:, :, z] - ndi.gaussian_filter(vol[:, :, z], radius_px), 0)
    else:
        raise SystemExit(f"unknown --mip-bg-method {method}")
    return out


def make_mip_volume(disp, z_um, method, radius_px, z_range=None, z_weights=None):
    """Display volume used for every MIP-type image: background-subtracted, planes outside z_range zeroed,
    optional per-range intensity weights [(z_lo, z_hi, factor), ...] (DISPLAY ONLY — biases which plane wins
    the projection; disclose in the caption)."""
    v = subtract_background(disp, method, radius_px)
    if z_range is not None and not (isinstance(z_range, str)):
        lo, hi = z_range
        keep = (z_um >= lo) & (z_um <= hi)
        if not keep.all():
            v = v.copy(); v[:, :, ~keep] = 0
    if z_weights:
        v = v.copy()
        for lo, hi, f in z_weights:
            sel = (z_um >= lo) & (z_um <= hi)
            if sel.any() and f != 1.0:
                v[:, :, sel] *= float(f)
    return v


def parse_z_weights(spec):
    """'-300:-100:1.5' or list of such -> [[lo, hi, f], ...]"""
    out = []
    for item in (spec if isinstance(spec, (list, tuple)) else [spec]):
        if not item:
            continue
        for part in str(item).split(';'):
            tok = re.split(r'[:,]', part.strip())
            if len(tok) == 3:
                lo, hi, f = float(tok[0]), float(tok[1]), float(tok[2])
                out.append([min(lo, hi), max(lo, hi), f])
    return out


def log_mip_percentiles(mip, mask, log):
    pcts = (50, 60, 70, 80, 90, 95, 99, 99.5, 99.9, 100)
    vals = np.percentile(mip[mask], pcts)
    log("[mip] MIP percentiles inside the mask (display units) -> use as --mip-min/--mip-max: "
        + ", ".join(f"p{p:g}={v:.4f}" for p, v in zip(pcts, vals)))


def mip_contact_sheet(disp, z_um, soft, mask, px_um, mipd, out, log,
                      lo_list=('p0', 'p50', 'p60', 'p70', 'p80', 'p90'), gammas=(0.6, 0.8)):
    """Grid of grayscale MIPs: rows = background method (none/gauss/tophat at the current radius) x gamma,
    columns = black level; white level = current --mip-max (or p99.9). -> mip_contact_sheet.png"""
    import matplotlib.pyplot as plt
    r_px = mipd['bg_radius_um'] / px_um
    rows_ = [(m, g) for m in ('none', 'gauss', 'tophat') for g in gammas]
    fig, axes = plt.subplots(len(rows_), len(lo_list), figsize=(2.0 * len(lo_list), 2.5 * len(rows_)))
    axes = np.atleast_2d(axes)
    for i, (m, g) in enumerate(rows_):
        vol = make_mip_volume(disp, z_um, m, r_px, mipd.get('z_range'), mipd.get('z_weights'))
        mip, _ = mip_and_depth(vol, mipd['smooth_px'], **proj_kw(mipd, px_um))
        hv = parse_level(mipd.get('hi_spec') or 'p99.9', mip, mask)
        for j, lo in enumerate(lo_list):
            lv = parse_level(lo, mip, mask)
            ax = axes[i, j]; ax.imshow(apply_levels(mip, lv, hv, g) * soft, cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); ax.set_axis_off()
            ax.set_title(f"bg {m}{'' if m == 'none' else f' R={mipd['bg_radius_um']:.0f}µm'}  gamma {g}\nmin {lo}={lv:.3f}  max={hv:.3f}", fontsize=6, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.03, hspace=0.22)
    fig.savefig(os.path.join(out, 'mip_contact_sheet.png'), dpi=130); plt.close(fig)
    log("[mip] mip_contact_sheet.png written — choose bg method / min / gamma and pass --mip-bg-method/--mip-min/--mip-gamma (or use --tune-mip)")


def _parse_range(spec, default):
    """'a:b:step' or 'a,b,c' -> list of floats."""
    if spec is None:
        return list(default)
    spec = str(spec)
    if ':' in spec:
        a, b, st = [float(v) for v in spec.split(':')]
        n = int(round((b - a) / st)) + 1
        return [round(a + i * st, 6) for i in range(n)]
    return [float(v) for v in spec.split(',')]


def mip_sweep(disp, z_um, soft, mask, px_um, mipd, out, log, gammas, min_pcts, bgs, kind='gray', label=True):
    """Exhaustive trial grid: gamma x background(none/tophat/gauss) x black level (percentile).
    Writes one PNG per combination (sweep/), one contact sheet per background, and sweep_index.csv
    with the exact CLI flags that reproduce each image. Everything else (bg radius, z-range, smoothing,
    projection, white level, upsample) follows the current settings."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    sw = os.path.join(out, 'sweep'); os.makedirs(sw, exist_ok=True)
    up = int(mipd.get('upsample', 2)); soft_up = np.clip(upsample_img(soft, up, 1), 0, 1) if up > 1 else soft
    rows = []; t0 = time.time(); n_img = 0
    all_thumbs = {}; all_levels = {}; all_hv = {}; all_mip = {}; all_vol = {}
    # reference vessel / background pixels for the automatic score (setting-independent: from the top-hat MIP)
    try:
        from skimage.filters import sato
        _vol = make_mip_volume(disp, z_um, 'tophat', mipd['bg_radius_um'] / px_um, mipd.get('z_range'), mipd.get('z_weights'))
        _mip, _ = mip_and_depth(_vol, mipd['smooth_px'], **proj_kw(mipd, px_um))
        vness = sato(_mip.astype(np.float32), sigmas=[max(0.7, 4 / px_um), 8 / px_um, 12 / px_um], black_ridges=False)
        vsel = mask & (vness >= np.percentile(vness[mask], 97))
        bsel = mask & ~ndi.binary_dilation(vsel, iterations=int(round(15 / px_um)) + 1) & (_mip <= np.percentile(_mip[mask], 70))
    except Exception:
        vsel = bsel = None
    for bg in bgs:
        vol = make_mip_volume(disp, z_um, bg, mipd['bg_radius_um'] / px_um, mipd.get('z_range'), mipd.get('z_weights'))
        mip, _ = mip_and_depth(vol, mipd['smooth_px'], **proj_kw(mipd, px_um))
        hv = parse_level(mipd.get('hi_spec') or 'p99.9', mip, mask)
        levels = {pc: parse_level(f'p{pc:g}', mip, mask) for pc in min_pcts}
        levels[0] = parse_level('p0', mip, mask)
        all_levels[bg] = levels; all_hv[bg] = hv; all_mip[bg] = mip; all_vol[bg] = vol
        thumbs = {}
        for g in gammas:
            for pc in min_pcts:
                lv = levels[pc]
                tag = f"bg-{bg}_p{pc:g}_gamma{g:.1f}"
                outs = []
                score = float('nan')
                base_img = apply_levels(mip, lv, hv, g)
                if vsel is not None and vsel.any() and bsel.any():
                    v_m = float(base_img[vsel].mean()); b_m = float(base_img[bsel].mean()); b_s = float(base_img[bsel].std())
                    sat = float((base_img[vsel] >= 0.98).mean())
                    score = v_m - b_m - 0.5 * b_s - 0.5 * sat          # separation - background noise - saturation
                if kind in ('gray', 'both'):
                    img = np.clip(upsample_img(base_img, up, 3), 0, 1) * soft_up
                    fn = f"mip_{tag}.png"; outs.append(fn)
                    im = Image.fromarray((img * 255).astype(np.uint8))
                    if label:
                        ImageDraw.Draw(im).text((6, 6), f"{bg}  p{pc:g}={lv:.3f}  g={g:.1f}", fill=255, font=_font(max(12, im.width // 40)))
                    im.save(os.path.join(sw, fn)); thumbs[(g, pc)] = img
                if kind in ('depth', 'both'):
                    rgb, _, _ = depth_coded_mip(vol, gamma=g, soft_mask=soft, lo=lv, hi=hv, smooth_px=mipd['smooth_px'], upsample=up, **proj_kw(mipd, px_um))
                    fn = f"depth_{tag}.png"; outs.append(fn)
                    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
                    if label:
                        ImageDraw.Draw(im).text((6, 6), f"{bg}  p{pc:g}={lv:.3f}  g={g:.1f}", fill=(255, 255, 255), font=_font(max(12, im.width // 40)))
                    im.save(os.path.join(sw, fn))
                    if kind == 'depth':
                        thumbs[(g, pc)] = np.clip(rgb, 0, 1)
                n_img += len(outs)
                flags = (f"--mip-bg-method {bg} --mip-bg-radius-um {mipd['bg_radius_um']:g} --mip-min p{pc:g} --mip-max {mipd.get('hi_spec') or 'p99.9'} "
                         f"--mip-gamma {g:.1f} --mip-smooth-px {mipd['smooth_px']:g} --mip-proj {mipd.get('proj_mode', 'max')} --mip-upsample {up}"
                         + (f" --mip-z-range={mipd['z_range'][0]:g},{mipd['z_range'][1]:g}" if mipd.get('z_range') else ""))
                rows.append(dict(files=';'.join(outs), bg=bg, min_pct=pc, min_value=round(lv, 5), max_value=round(hv, 5), gamma=g, score=round(score, 4), cli_flags=flags))
        # contact sheet for this background: rows = gamma, cols = black level
        fig, axes = plt.subplots(len(gammas), len(min_pcts), figsize=(1.75 * len(min_pcts), 1.85 * len(gammas)))
        axes = np.atleast_2d(axes)
        for i, g in enumerate(gammas):
            for j, pc in enumerate(min_pcts):
                ax = axes[i, j]; ax.set_axis_off()
                if (g, pc) in thumbs:
                    im_ = thumbs[(g, pc)]
                    ax.imshow(im_, cmap='gray' if im_.ndim == 2 else None, vmin=0, vmax=1, interpolation='bicubic')
                ax.set_title(f"g {g:.1f}  p{pc:g}={levels[pc]:.3f}", fontsize=6, pad=1.5)
        fig.suptitle(f"background: {bg}" + ("" if bg == 'none' else f" (R {mipd['bg_radius_um']:g} µm)") + f"   max = {mipd.get('hi_spec') or 'p99.9'} ({hv:.3f})   "
                     f"smooth {mipd['smooth_px']:g} px   projection {mipd.get('proj_mode', 'max')}   z-range {mipd.get('z_range') or 'all'}", fontsize=8)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.18)
        fig.savefig(os.path.join(sw, f"sweep_contact_bg-{bg}.png"), dpi=130); plt.close(fig)
        all_thumbs[bg] = thumbs
        log(f"[sweep] bg {bg}: {len(gammas) * len(min_pcts)} combinations, levels {', '.join(f'p{pc:g}={levels[pc]:.3f}' for pc in min_pcts)}")
    # ── ONE combined sheet in the --mip-sheet layout: rows = background x gamma, columns = p0 (reference) + black levels
    cols = [0] + list(min_pcts); rlist = [(bg, g) for bg in bgs for g in gammas]
    fig, axes = plt.subplots(len(rlist), len(cols), figsize=(2.0 * len(cols), 2.5 * len(rlist)))
    axes = np.atleast_2d(axes)
    for i, (bg, g) in enumerate(rlist):
        for j, pc in enumerate(cols):
            ax = axes[i, j]; ax.set_axis_off()
            lv = all_levels[bg][pc]; hv = all_hv[bg]
            if pc == 0:
                if kind == 'depth':
                    im_ = np.clip(depth_coded_mip(all_vol[bg], gamma=g, soft_mask=soft, lo=lv, hi=hv, smooth_px=mipd['smooth_px'], upsample=1, **proj_kw(mipd, px_um))[0], 0, 1)
                else:
                    im_ = apply_levels(all_mip[bg], lv, hv, g) * soft
            else:
                im_ = all_thumbs[bg].get((g, pc))
                if im_ is not None and up > 1:
                    pass
            if im_ is not None:
                ax.imshow(im_, cmap='gray' if im_.ndim == 2 else None, vmin=0, vmax=1, interpolation='bicubic')
            ax.set_title(f"bg {bg}{'' if bg == 'none' else f' R={mipd['bg_radius_um']:.0f}µm'}  gamma {g:.1f}\nmin p{pc:g}={lv:.3f}  max={hv:.3f}", fontsize=6, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.03, hspace=0.22)
    fig.savefig(os.path.join(sw, 'sweep_contact_all.png'), dpi=110)
    fig.savefig(os.path.join(out, 'mip_contact_sheet_sweep.png'), dpi=110); plt.close(fig)
    log(f"[sweep] combined sheet ({len(rlist)} rows x {len(cols)} cols, first column = p0 reference): {os.path.join(out, 'mip_contact_sheet_sweep.png')}")
    write_csv(os.path.join(sw, 'sweep_index.csv'), rows)
    scored = [r for r in rows if np.isfinite(r['score'])]
    best = max(scored, key=lambda r: r['score']) if scored else None
    if best:
        log(f"[sweep] automatic pick (max vessel/background separation score): bg {best['bg']}, min p{best['min_pct']:g}, gamma {best['gamma']:.1f}  (score {best['score']:.3f})")
    log(f"[sweep] {n_img} images + {len(bgs)} contact sheets in {sw}  ({time.time() - t0:.0f} s). Pick one and re-run with the flags in sweep_index.csv, e.g.\n"
        f"        python paw_suppfig.py --recon <file>_rl.mat --raw <file>.raw --fps <Hz> --mask-file <out>/mask.png {rows[0]['cli_flags']}")
    return rows, best


def tune_mip_interactive(disp, z_um, soft, mask, px_um, init, out_dir, log):
    """Sliders: black (min), white (max), gamma, 2-D smoothing, background radius; key 'b' cycles the background
    method (none -> gauss -> tophat). Enter = accept -> mip_display.json (auto-reused by later runs in this folder)."""
    if not _interactive_backend(log):
        log("[tune] no GUI backend available — use --mip-sheet and pass --mip-bg-method/--mip-min/--mip-max/--mip-gamma explicitly"); return init
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.colors import Normalize
    from matplotlib import cm
    st = dict(init); methods = ['none', 'gauss', 'tophat']
    cache = {}

    def mip_vol():
        key = (st['bg_method'], round(st['bg_radius_um'], 1))
        if key not in cache:
            cache[key] = make_mip_volume(disp, z_um, st['bg_method'], st['bg_radius_um'] / px_um, st.get('z_range'), st.get('z_weights'))
        return cache[key]
    fig = plt.figure(figsize=(12.5, 7.4))
    ax_g = fig.add_axes([0.03, 0.34, 0.44, 0.62]); ax_c = fig.add_axes([0.50, 0.34, 0.44, 0.62])
    for a in (ax_g, ax_c):
        a.set_axis_off()
    cax = fig.add_axes([0.955, 0.40, 0.012, 0.5])
    sm = cm.ScalarMappable(norm=Normalize(z_um[0], z_um[-1]), cmap='turbo'); sm.set_array([]); fig.colorbar(sm, cax=cax).set_label('z (µm)')

    def render():
        rgb, g, _ = depth_coded_mip(mip_vol(), gamma=st['gamma'], soft_mask=soft, lo=st['lo'], hi=st['hi'], smooth_px=st['smooth_px'], upsample=1, **proj_kw(st, px_um))
        return g, rgb
    g, rgb = render()
    im_g = ax_g.imshow(g, cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); im_c = ax_c.imshow(rgb, interpolation='bicubic')
    title = fig.text(0.03, 0.985, '', fontsize=9, va='top')

    def refresh():
        mip, _ = mip_and_depth(mip_vol(), st['smooth_px'], **proj_kw(st, px_um))
        plo = float((mip[mask] <= st['lo']).mean() * 100); phi = float((mip[mask] <= st['hi']).mean() * 100)
        title.set_text(f"bg {st['bg_method']} R={st['bg_radius_um']:.0f} µm ('b' to cycle)   min {st['lo']:.3f} (p{plo:.1f})   max {st['hi']:.3f} (p{phi:.1f})   "
                       f"gamma {st['gamma']:.2f}   smooth {st['smooth_px']:.2f} px      Enter = accept, r = reset")
        g, rgb = render(); im_g.set_data(g); im_c.set_data(rgb); fig.canvas.draw_idle()
    sl = {}
    specs = [('lo', 'min (black)', 0.0, 0.8, st['lo']), ('hi', 'max (white)', 0.05, 1.0, st['hi']),
             ('gamma', 'gamma', 0.2, 1.5, st['gamma']), ('smooth_px', 'smooth (px)', 0.0, 3.0, st['smooth_px']),
             ('bg_radius_um', 'bg radius (µm)', 20.0, 400.0, max(20.0, st['bg_radius_um']))]
    for i, (k, lab, lo_, hi_, v) in enumerate(specs):
        axs = fig.add_axes([0.14, 0.255 - i * 0.048, 0.70, 0.03]); sl[k] = Slider(axs, lab, lo_, hi_, valinit=v)

        def _cb(val, k=k):
            st[k] = float(val)
            if st['hi'] <= st['lo'] + 1e-3:
                st['hi'] = st['lo'] + 1e-3
            refresh()
        sl[k].on_changed(_cb)

    def on_key(ev):
        if ev.key in ('enter', 'return'):
            plt.close(fig)
        elif ev.key == 'b':
            st['bg_method'] = methods[(methods.index(st['bg_method']) + 1) % len(methods)]; refresh()
        elif ev.key == 'r':
            st['bg_method'] = init['bg_method']
            for k, _, _, _, v in specs:
                sl[k].set_val(init[k] if k != 'bg_radius_um' else max(20.0, init[k]))
    fig.canvas.mpl_connect('key_press_event', on_key)
    refresh(); plt.show()
    res = dict(lo=st['lo'], hi=st['hi'], gamma=st['gamma'], smooth_px=st['smooth_px'], upsample=init.get('upsample', 2),
               bg_method=st['bg_method'], bg_radius_um=st['bg_radius_um'], z_range=init.get('z_range'))
    json.dump(res, open(os.path.join(out_dir, 'mip_display.json'), 'w'), indent=1)
    log(f"[tune] accepted {res} -> mip_display.json")
    return res


def paint_mask_interactive(img01, init_mask, out_dir, log, radius_px=6):
    """Brush editor for the FOV mask on the representative MIP (start from init_mask if given).
    left-drag = paint (include), right-drag = erase, '[' / ']' = brush size, 'z' = undo, 'i' = invert, 'c' = clear,
    'a' = all, Enter = accept -> mask.png/.npy/.json.  NOTE: use it to exclude non-tissue regions (window edge,
    reflections), not to erase background around vessels — the same mask is used for quantification."""
    if not _interactive_backend(log):
        raise SystemExit("no GUI backend available for --paint-mask: draw the mask in Fiji (--export-frame) and pass --mask-file")
    import matplotlib.pyplot as plt
    H, W = img01.shape
    st = dict(mask=(init_mask.copy() if init_mask is not None else np.ones((H, W), bool)), r=int(radius_px), btn=None, last=None, hist=[])
    yy, xx = np.mgrid[0:H, 0:W]
    fig, ax = plt.subplots(figsize=(8.5, 8.5)); ax.set_axis_off()
    ax.imshow(np.clip(img01, 0, 1) ** 0.5, cmap='gray', vmin=0, vmax=1)
    ov = np.zeros((H, W, 4), np.float32); ov[..., 0] = 1.0; ov[..., 1] = 0.35; ov[..., 2] = 0.0    # orange = EXCLUDED
    im_ov = ax.imshow(ov)
    title = ax.set_title('', fontsize=8)

    def refresh():
        ov[..., 3] = np.where(st['mask'], 0.0, 0.35); im_ov.set_data(ov)
        title.set_text(f"brush r={st['r']} px | left=paint(include) right=erase(exclude) | [ ] size | z undo | i invert | c clear | a all | Enter accept"
                       f"\n{int(st['mask'].sum())} px inside ({100 * st['mask'].mean():.1f} %)   orange = excluded")
        fig.canvas.draw_idle()

    def stamp(x, y, val):
        r = st['r']; x0, x1 = max(0, int(x) - r - 1), min(W, int(x) + r + 2); y0, y1 = max(0, int(y) - r - 1), min(H, int(y) + r + 2)
        sub = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= r * r
        st['mask'][y0:y1, x0:x1][sub] = val

    def stroke(x, y, val):
        if st['last'] is not None:
            lx, ly = st['last']; n = int(max(1, np.hypot(x - lx, y - ly) / max(st['r'] / 2.0, 1)))
            for t in np.linspace(0, 1, n + 1):
                stamp(lx + t * (x - lx), ly + t * (y - ly), val)
        else:
            stamp(x, y, val)
        st['last'] = (x, y)

    def on_press(ev):
        if ev.inaxes != ax or ev.xdata is None or ev.button not in (1, 3):
            return
        st['hist'].append(st['mask'].copy()); st['hist'] = st['hist'][-30:]
        st['btn'] = ev.button; st['last'] = None; stroke(ev.xdata, ev.ydata, ev.button == 1); refresh()

    def on_move(ev):
        if st['btn'] is None or ev.inaxes != ax or ev.xdata is None:
            return
        stroke(ev.xdata, ev.ydata, st['btn'] == 1); refresh()

    def on_release(ev):
        st['btn'] = None; st['last'] = None

    def on_key(ev):
        if ev.key in ('enter', 'return'):
            plt.close(fig); return
        if ev.key == ']':
            st['r'] = min(64, st['r'] + 2)
        elif ev.key == '[':
            st['r'] = max(1, st['r'] - 2)
        elif ev.key == 'z' and st['hist']:
            st['mask'] = st['hist'].pop()
        elif ev.key == 'i':
            st['hist'].append(st['mask'].copy()); st['mask'] = ~st['mask']
        elif ev.key == 'c':
            st['hist'].append(st['mask'].copy()); st['mask'] = np.zeros((H, W), bool)
        elif ev.key == 'a':
            st['hist'].append(st['mask'].copy()); st['mask'] = np.ones((H, W), bool)
        refresh()
    for name, fn in (('button_press_event', on_press), ('motion_notify_event', on_move), ('button_release_event', on_release), ('key_press_event', on_key)):
        fig.canvas.mpl_connect(name, fn)
    refresh(); plt.show()
    mask = st['mask']
    if mask.sum() == 0:
        raise SystemExit("empty mask — nothing painted")
    from PIL import Image
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, 'mask.png'))
    np.save(os.path.join(out_dir, 'mask.npy'), mask)
    json.dump(dict(type='painted', brush_px_last=st['r']), open(os.path.join(out_dir, 'mask.json'), 'w'), indent=1)
    log(f"[mask] painted: {int(mask.sum())} px -> mask.png / mask.npy (reuse with --mask-file)")
    return mask


def initial_mipd(args, out, log):
    """MIP display defaults, overridden by --mip-json / <out>/mip_display.json, then by explicit CLI flags."""
    mipd = dict(lo=0.0, hi=None, gamma=args.gamma, smooth_px=0.0, upsample=2, bg_method='none', bg_radius_um=100.0, z_range=None, hi_spec='p99.9',
                proj_mode='max', focus_win_um=40.0, plane_lo=None, plane_hi=None, z_weights=[], depth_bright='intensity', depth_floor=0.5, depth_gamma=None, depth_floor_thr=0.1, depth_local_px=17)
    jpath = args.mip_json or os.path.join(out, 'mip_display.json')
    if os.path.exists(jpath):
        mipd.update({k: v for k, v in json.load(open(jpath)).items() if k in mipd}); log(f"[mip] settings from {jpath}")
    if args.mip_bg_method is not None:
        mipd['bg_method'] = args.mip_bg_method
    if args.mip_bg_radius_um is not None:
        mipd['bg_radius_um'] = float(args.mip_bg_radius_um)
    if args.mip_z_range and str(args.mip_z_range).lower() == 'auto':
        mipd['z_range'] = 'auto'
    elif args.mip_z_range:
        zv = [float(v) for v in str(args.mip_z_range).split(',')]
        mipd['z_range'] = [-abs(zv[0]), abs(zv[0])] if len(zv) == 1 else zv[:2]
    if args.mip_smooth_px is not None:
        mipd['smooth_px'] = float(args.mip_smooth_px)
    if args.mip_gamma is not None:
        mipd['gamma'] = float(args.mip_gamma)
    if args.mip_upsample is not None:
        mipd['upsample'] = int(args.mip_upsample)
    if args.mip_hi is not None:
        mipd['hi_spec'] = args.mip_hi
    if args.z_weight:
        mipd['z_weights'] = parse_z_weights(args.z_weight)
    if args.depth_bright is not None:
        mipd['depth_bright'] = args.depth_bright
    if args.depth_floor is not None:
        mipd['depth_floor'] = float(args.depth_floor)
        if args.depth_bright is None and mipd['depth_bright'] == 'intensity':
            mipd['depth_bright'] = 'floor'
    if args.depth_local_um is not None:
        mipd['depth_local_px'] = max(1, int(round(float(args.depth_local_um) / args.px_um))) if isinstance(args.px_um, (int, float)) else 17
    if args.depth_gamma is not None:
        mipd['depth_gamma'] = float(args.depth_gamma)
    if args.depth_floor_thr is not None:
        mipd['depth_floor_thr'] = float(args.depth_floor_thr)
    if args.mip_proj is not None:
        mipd['proj_mode'] = args.mip_proj
    if args.focus_win_um is not None:
        mipd['focus_win_um'] = float(args.focus_win_um)
    return mipd


# ════════════════════════════════════════════════════════════════════════════
# 4b. Studio GUI — draw/refine the mask + choose the MIP display parameters, then Export
#     (front-end only: it writes mask.png + mip_display.json and the unchanged pipeline runs on them)
# ════════════════════════════════════════════════════════════════════════════
class SuppFigStudio:
    MODES = ['Gray MIP', 'Depth-coded', 'Binned MIPs', 'Single plane']
    PROJ = ['max', 'focus']
    BRIGHT = ['intensity', 'floor', 'local', 'flat']
    TOOLS = ['Off', 'Polygon', 'Ellipse', 'Brush']
    BG = ['none', 'gauss', 'tophat']
    UPS = ['1x', '2x', '3x']

    def __init__(self, disp, z_um, px_um, mipd, mask, out_dir, log, title='', default_video=False):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button, PolygonSelector, EllipseSelector
        from matplotlib.colors import Normalize
        from matplotlib import cm
        self.plt = plt; self.disp = disp; self.z = np.asarray(z_um); self.px = float(px_um); self.out = out_dir; self.log = log
        self.H, self.W, self.D = disp.shape
        self.zmax = float(np.max(np.abs(self.z)))
        m = dict(mipd); m.setdefault('bg_method', 'none'); m.setdefault('bg_radius_um', 100.0); m.setdefault('z_range', None)
        m.setdefault('smooth_px', 0.0); m.setdefault('gamma', 0.5); m.setdefault('upsample', 2); m.setdefault('lo', 0.0)
        m.setdefault('proj_mode', 'max'); m.setdefault('focus_win_um', 40.0); m.setdefault('plane_lo', None); m.setdefault('plane_hi', None)
        m.setdefault('depth_bright', 'intensity'); m.setdefault('depth_floor', 0.5); m.setdefault('depth_gamma', None); m.setdefault('depth_floor_thr', 0.1)
        m.setdefault('depth_local_px', 17); m.setdefault('z_weights', [])
        self.mipd = m; self.mask = mask.copy().astype(bool); self.hist = []; self.cache = {}; self.result = None
        self._bg = None; self._mipcache = {}; self._softcache = None; self._mask_ver = 0
        self.mode = self.MODES[0]; self.tool = 'Off'; self.brush = 8; self.btn = None; self.last = None
        self.default_video = bool(default_video); self.zw_on = bool(m.get('z_weights'))
        if self.mipd.get('hi') is None:
            self.mipd['hi'] = parse_level(self.mipd.get('hi_spec') or 'p99.9', self._mip(), self.mask)
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W]
        # ── figure
        self.fig = plt.figure(figsize=(16, 9.6)); self.fig.canvas.manager.set_window_title(f'NIR-II SLIM Supp-Fig Studio — {title}') if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager else None
        self.ax = self.fig.add_axes([0.01, 0.05, 0.60, 0.90]); self.ax.set_axis_off()
        self.im = self.ax.imshow(np.zeros((self.H, self.W)), cmap='gray', vmin=0, vmax=1, interpolation='bicubic')
        self.ov = np.zeros((self.H, self.W, 4), np.float32); self.ov[..., 0] = 1.0; self.ov[..., 1] = 0.4
        self.im_ov = self.ax.imshow(self.ov, zorder=5)
        from matplotlib.patches import Circle
        self.cursor = Circle((0, 0), radius=self.brush, fill=False, ec='yellow', lw=1.2, ls='--', zorder=6, visible=False); self.ax.add_patch(self.cursor)
        cax = self.fig.add_axes([0.615, 0.30, 0.008, 0.40])
        sm = cm.ScalarMappable(norm=Normalize(self.z[0], self.z[-1]), cmap='turbo'); sm.set_array([])
        self.fig.colorbar(sm, cax=cax).set_label('z (µm)', fontsize=8); cax.tick_params(labelsize=7)
        self.cax = cax; self.zmark = cax.axhline(0.0, color='white', lw=2.0, visible=False); self._planecache = {}
        self.status = self.fig.text(0.65, 0.99, '', fontsize=8, va='top', family='monospace')
        X0 = 0.655
        # ── radios
        self.rb_mode = RadioButtons(self._ax(X0, 0.805, 0.10, 0.10, 'preview'), self.MODES, active=0); self.rb_mode.on_clicked(self._on_mode)
        self.rb_tool = RadioButtons(self._ax(X0 + 0.115, 0.805, 0.10, 0.10, 'mask tool'), self.TOOLS, active=0); self.rb_tool.on_clicked(self._on_tool)
        self.rb_bg = RadioButtons(self._ax(X0 + 0.23, 0.825, 0.09, 0.08, 'background'), self.BG, active=self.BG.index(self.mipd['bg_method'])); self.rb_bg.on_clicked(self._on_bg)
        self.rb_bright = RadioButtons(self._ax(X0, 0.70, 0.10, 0.09, 'depth colour brightness'), self.BRIGHT, active=self.BRIGHT.index(self.mipd['depth_bright'])); self.rb_bright.on_clicked(self._on_bright)
        self.rb_proj = RadioButtons(self._ax(X0 + 0.115, 0.745, 0.09, 0.045, 'projection'), self.PROJ, active=self.PROJ.index(self.mipd['proj_mode'])); self.rb_proj.on_clicked(self._on_proj)
        zw = self.mipd.get('z_weights') or []
        zw_title = 'z-weight ' + (', '.join(f'{lo:g}..{hi:g} x{f:g}' for lo, hi, f in zw) if zw else '(none given)')
        self.rb_zw = RadioButtons(self._ax(X0 + 0.115, 0.695, 0.09, 0.045, zw_title), ['on', 'off'], active=0 if self.zw_on else 1); self.rb_zw.on_clicked(self._on_zw)
        if not zw:
            self.rb_zw.set_active(1)
        # ── sliders
        zr = self.mipd['z_range'] or [float(self.z[0]), float(self.z[-1])]
        self.sl = {}
        specs = [('lo', 'min (black)', 0.0, 0.9, self.mipd['lo'], None), ('hi', 'max (white)', 0.02, 1.0, self.mipd['hi'], None),
                 ('gamma', 'gamma', 0.2, 1.5, self.mipd['gamma'], None), ('smooth_px', 'smooth (px)', 0.0, 3.0, self.mipd['smooth_px'], None),
                 ('bg_radius_um', 'bg radius (µm)', 20.0, 400.0, max(20.0, self.mipd['bg_radius_um']), None),
                 ('focus_win_um', 'focus win (µm)', 10.0, 150.0, float(self.mipd['focus_win_um']), None),
                 ('zmin', 'z min (µm)', float(self.z[0]), float(self.z[-1]), float(max(zr[0], self.z[0])), None),
                 ('zmax', 'z max (µm)', float(self.z[0]), float(self.z[-1]), float(min(zr[1], self.z[-1])), None),
                 ('brush', 'brush (px)', 1, 40, self.brush, 1),
                 ('plane', 'plane z (µm)', float(self.z[0]), float(self.z[-1]), float(self.z[int(np.argmin(np.abs(self.z)))]),
                  float(self.z[1] - self.z[0]) if self.D > 1 else None),
                 ('plane_lo', 'plane min', 0.0, 0.9, float(self.mipd['plane_lo'] if self.mipd['plane_lo'] is not None else 0.0), None),
                 ('plane_hi', 'plane max', 0.02, 1.0, float(self.mipd['plane_hi'] if self.mipd['plane_hi'] is not None else 1.0), None),
                 ('depth_floor', 'colour floor', 0.0, 1.0, float(self.mipd['depth_floor']), None),
                 ('depth_gamma', 'colour gamma', 0.2, 1.5, float(self.mipd['depth_gamma'] or self.mipd['gamma']), None),
                 ('depth_floor_thr', 'colour fade-in', 0.0, 0.4, float(self.mipd['depth_floor_thr']), None)]
        for i, (k, lab, lo_, hi_, v, step) in enumerate(specs):
            axs = self.fig.add_axes([X0 + 0.075, 0.665 - i * 0.026, 0.215, 0.018])
            self.sl[k] = Slider(axs, lab, lo_, hi_, valinit=v, valstep=step); self.sl[k].label.set_fontsize(8); self.sl[k].valtext.set_fontsize(8)
            self.sl[k].on_changed(lambda val, k=k: self._on_slider(k, val))
        self.rb_up = RadioButtons(self._ax(X0 + 0.23, 0.715, 0.09, 0.07, 'upsample'), self.UPS, active=min(int(self.mipd['upsample']) - 1, 2)); self.rb_up.on_clicked(self._on_up)
        # ── buttons
        self.buttons = {}
        row1 = [('Undo (z)', self._undo), ('Clear', self._clear), ('All', self._all), ('Invert', self._invert), ('Load mask', self._load_mask), ('Save mask', self._save_mask)]
        row2 = [('Auto min p60', lambda e: self._auto('lo', 'p60')), ('Auto min p70', lambda e: self._auto('lo', 'p70')), ('Auto max p99.9', lambda e: self._auto('hi', 'p99.9')),
                ('z min = plane', lambda e: self.sl['zmin'].set_val(self._plane_z())), ('z max = plane', lambda e: self.sl['zmax'].set_val(self._plane_z())),
                ('Plane auto', self._plane_auto), ('Save settings', self._save_settings)]
        row3 = [('EXPORT (no video)', lambda e: self._export(False)), ('EXPORT + video', lambda e: self._export(True)), ('Quit', self._quit)]
        for j, (lab, fn) in enumerate(row1):
            b = Button(self.fig.add_axes([X0 + j * 0.0575, 0.255, 0.055, 0.032]), lab); b.label.set_fontsize(7); b.on_clicked(fn); self.buttons[lab] = b
        for j, (lab, fn) in enumerate(row2):
            b = Button(self.fig.add_axes([X0 + j * 0.0487, 0.21, 0.047, 0.036]), lab); b.label.set_fontsize(6.5); b.on_clicked(fn); self.buttons[lab] = b
        for j, (lab, fn) in enumerate(row3):
            b = Button(self.fig.add_axes([X0 + j * 0.112, 0.148, 0.108, 0.048]), lab, color='#dfe8ff' if 'EXPORT' in lab else '0.9'); b.label.set_fontsize(9); b.on_clicked(fn); self.buttons[lab] = b
        self.fig.text(X0, 0.12, "Mask: Polygon = click vertices, click the first one to close (replaces the mask).\n"
                      "Ellipse = drag. Brush: LEFT-drag = paint (include), RIGHT-drag = erase (exclude); the dashed yellow\n"
                      "circle is the brush (toolbar zoom/pan must be OFF). Keys: [ ] size, z undo.\n"
                      "Single plane: 'plane z' (or up/down keys) picks the plane; 'plane min/max' + 'Plane auto' set its contrast; brush on it.\n"
                      "projection 'focus' = per-pixel in-focus plane (removes defocus ghosts). Orange tint = excluded.\n"
                      "Display settings are DISPLAY ONLY -> mip_display.json (state them in the caption);\n"
                      "FWHM / SBR always use the raw reconstruction. EXPORT or simply CLOSING the window = save mask + settings and run; Quit = skip.",
                      fontsize=7.5, va='top')
        # ── selectors
        self.ps = PolygonSelector(self.ax, self._on_poly, useblit=False, props=dict(color='yellow', linewidth=1.5, alpha=0.9),
                                  handle_props=dict(markersize=6, markerfacecolor='yellow', markeredgecolor='k'))
        self.ps.set_active(False); self.ps.set_visible(False)
        self.es = EllipseSelector(self.ax, self._on_ellipse, useblit=False, interactive=True, props=dict(facecolor='none', edgecolor='cyan', linewidth=1.5))
        self.es.set_active(False); self.es.set_visible(False)
        for name, fn in (('button_press_event', self._press), ('motion_notify_event', self._move), ('button_release_event', self._release), ('key_press_event', self._key)):
            self.fig.canvas.mpl_connect(name, fn)
        self.refresh()

    # ── helpers
    def _ax(self, x, y, w, h, title):
        a = self.fig.add_axes([x, y, w, h]); a.set_title(title, fontsize=8); return a

    def _zr(self):
        if not hasattr(self, 'sl'):
            return self.mipd.get('z_range')
        lo, hi = float(self.sl['zmin'].val), float(self.sl['zmax'].val)
        if hi < lo:
            lo, hi = hi, lo
        full = lo <= self.z[0] + 1e-6 and hi >= self.z[-1] - 1e-6
        return None if full else [lo, hi]

    def _vol(self):
        key = (self.mipd['bg_method'], round(float(self.mipd['bg_radius_um']), 1), tuple(self._zr() or ()))
        if key not in self.cache:
            if len(self.cache) > 6:
                self.cache.clear()
            self.cache[key] = make_mip_volume(self.disp, self.z, self.mipd['bg_method'], self.mipd['bg_radius_um'] / self.px, self._zr(),
                                              (self.mipd.get('z_weights') if self.zw_on else None))
        return self.cache[key]

    def _mip(self):
        key = (self.mipd['bg_method'], round(float(self.mipd['bg_radius_um']), 1), tuple(self._zr() or ()), round(float(self.mipd['smooth_px']), 3),
               self.mipd['proj_mode'], round(float(self.mipd['focus_win_um']), 1))
        if key not in self._mipcache:
            if len(self._mipcache) > 6:
                self._mipcache.clear()
            self._mipcache[key] = mip_and_depth(self._vol(), self.mipd['smooth_px'], **proj_kw(self.mipd, self.px))[0]
        return self._mipcache[key]

    def _pct(self, v):
        return float((self._mip()[self.mask] <= v).mean() * 100) if self.mask.any() else float('nan')

    def _soft(self):
        if self._softcache is None or self._softcache[0] != self._mask_ver:
            self._softcache = (self._mask_ver, soft_edge(self.mask))
        return self._softcache[1]

    def _mask_changed(self):
        self._mask_ver += 1; self._bg = None

    def _render(self):
        md = self.mipd; soft = self._soft()
        if self.mode == 'Depth-coded':
            rgb, _, _ = depth_coded_mip(self._vol(), gamma=md['gamma'], soft_mask=soft, lo=md['lo'], hi=md['hi'], smooth_px=md['smooth_px'], upsample=1, **proj_kw(md, self.px), **colour_kw(md))
            return rgb
        if self.mode == 'Binned MIPs':
            edges = np.linspace(self.z[0], self.z[-1], 4); tiles = []
            vol = self._vol()
            for k in range(3):
                sel = (self.z >= edges[k]) & ((self.z < edges[k + 1]) if k < 2 else (self.z <= edges[k + 1]))
                mip = mip_and_depth(vol[:, :, sel], md['smooth_px'], **proj_kw(md, self.px))[0] if sel.any() else np.zeros((self.H, self.W), np.float32)
                tiles.append(apply_levels(mip, md['lo'], md['hi'], md['gamma']) * soft)
                if k < 2:
                    tiles.append(np.full((self.H, 4), 0.25, np.float32))
            return np.concatenate(tiles, axis=1)
        if self.mode == 'Single plane':
            return self._plane_img() * soft
        return apply_levels(self._mip(), md['lo'], md['hi'], md['gamma']) * soft

    # ── single-plane preview (find the focal plane of a vessel, then brush on it)
    def _plane_index(self):
        v = float(self.sl['plane'].val) if hasattr(self, 'sl') and 'plane' in self.sl else 0.0
        return int(np.argmin(np.abs(self.z - v)))

    def _plane_z(self):
        return float(self.z[self._plane_index()])

    def _plane_img(self):
        md = self.mipd; k = self._plane_index()
        key = (k, md['bg_method'], round(float(md['bg_radius_um']), 1), round(float(md['smooth_px']), 3))
        if key not in self._planecache:
            if len(self._planecache) > 40:
                self._planecache.clear()
            pl = subtract_background(self.disp[:, :, k:k + 1], md['bg_method'], md['bg_radius_um'] / self.px)[:, :, 0]
            if md['smooth_px'] > 0:
                pl = ndi.gaussian_filter(pl, md['smooth_px'])
            self._planecache[key] = pl
        pl = self._planecache[key]
        lo = float(self.sl['plane_lo'].val) if hasattr(self, 'sl') else 0.0
        hi = float(self.sl['plane_hi'].val) if hasattr(self, 'sl') else 1.0
        return apply_levels(pl, lo, max(hi, lo + 1e-3), md['gamma'])

    def _plane_auto(self, ev=None):
        """Set plane min/max to the 1st / 99.9th percentile of the CURRENT plane inside the mask."""
        self._plane_img()                                   # make sure the plane is cached
        md = self.mipd; k = self._plane_index()
        key = (k, md['bg_method'], round(float(md['bg_radius_um']), 1), round(float(md['smooth_px']), 3))
        pl = self._planecache[key]; sel = pl[self.mask] if self.mask.any() else pl.ravel()
        lo, hi = float(np.percentile(sel, 1)), float(np.percentile(sel, 99.9))
        self.sl['plane_lo'].set_val(float(np.clip(lo, 0.0, 0.9))); self.sl['plane_hi'].set_val(float(np.clip(hi, 0.02, 1.0)))

    def _on_proj(self, label):
        self.mipd['proj_mode'] = label; self.refresh()

    def _on_zw(self, label):
        self.zw_on = (label == 'on') and bool(self.mipd.get('z_weights'))
        self.cache.clear(); self._mipcache.clear(); self.refresh()

    def _on_bright(self, label):
        self.mipd['depth_bright'] = label
        if self.mode != 'Depth-coded':
            self.rb_mode.set_active(1); return
        self.refresh()

    def refresh(self):
        img = self._render()
        self.im.set_data(img); self.im.set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        if img.ndim == 2:
            self.im.set_cmap('gray'); self.im.set_clim(0, 1)
        show_ov = self.mode != 'Binned MIPs'
        self.ov[..., 3] = np.where(self.mask, 0.0, 0.32) if show_ov else 0.0; self.im_ov.set_data(self.ov)
        self.ax.set_xlim(-0.5, img.shape[1] - 0.5); self.ax.set_ylim(img.shape[0] - 0.5, -0.5)
        md = self.mipd; zr = self._zr() or [self.z[0], self.z[-1]]
        self.status.set_text(f"mask {int(self.mask.sum())} px ({100 * self.mask.mean():.1f} %)   tool: {self.tool}   brush {self.brush} px   (left = paint, right = erase)\n"
                             f"projection {md['proj_mode']}{'' if md['proj_mode'] == 'max' else f' (win {md['focus_win_um']:.0f} µm)'}   bg {md['bg_method']} R={md['bg_radius_um']:.0f} µm   z {zr[0]:.0f} to {zr[1]:.0f} µm   upsample {md['upsample']}x   "
                             f"colour {md['depth_bright']} floor {md['depth_floor']:.2f} g {float(md['depth_gamma'] or md['gamma']):.2f}   z-weight {'ON' if self.zw_on else 'off'}\n"
                             f"min {md['lo']:.3f} (p{self._pct(md['lo']):.1f})   max {md['hi']:.3f} (p{self._pct(md['hi']):.1f})   "
                             f"gamma {md['gamma']:.2f}   smooth {md['smooth_px']:.2f} px = {md['smooth_px'] * self.px:.1f} µm")
        if self.mode == 'Single plane':
            self.zmark.set_ydata([self._plane_z(), self._plane_z()]); self.zmark.set_visible(True)
            self.status.set_text(self.status.get_text() + f"\nSINGLE PLANE  z = {self._plane_z():.0f} µm (plane {self._plane_index() + 1}/{self.D})   plane min {float(self.sl['plane_lo'].val):.3f} / max {float(self.sl['plane_hi'].val):.3f}   keys: up/down = step planes")
        else:
            self.zmark.set_visible(False)
        self.cursor.set_visible(False); self._bg = None
        self.fig.canvas.draw_idle()

    # ── fast path while painting: blit only the overlay + brush cursor onto a cached background
    def _capture_bg(self):
        self.im_ov.set_visible(False); self.cursor.set_visible(False)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.im_ov.set_visible(True)

    def _blit(self, cursor_xy=None):
        c = self.fig.canvas
        try:
            if self._bg is None:
                self._capture_bg()
            c.restore_region(self._bg)
            self.ov[..., 3] = np.where(self.mask, 0.0, 0.32) if self.mode != 'Binned MIPs' else 0.0
            self.im_ov.set_data(self.ov); self.ax.draw_artist(self.im_ov)
            if cursor_xy is not None:
                self.cursor.set_center(cursor_xy); self.cursor.set_radius(self.brush); self.cursor.set_visible(True); self.ax.draw_artist(self.cursor)
            else:
                self.cursor.set_visible(False)
            c.blit(self.ax.bbox); c.flush_events()
        except Exception:                                  # e.g. window resized since capture -> full redraw
            self._bg = None; self.refresh()

    # ── widget callbacks
    def _on_mode(self, label):
        self.mode = label; self.refresh()

    def _on_tool(self, label):
        self.tool = label
        self.ps.set_active(label == 'Polygon'); self.ps.set_visible(label == 'Polygon')
        self.es.set_active(label == 'Ellipse'); self.es.set_visible(label == 'Ellipse')
        if label == 'Brush' and self.mode == 'Binned MIPs':
            self.rb_mode.set_active(0)                     # brushing needs a single-image preview
        self.refresh()

    def _on_bg(self, label):
        self.mipd['bg_method'] = label; self.refresh()

    def _on_up(self, label):
        self.mipd['upsample'] = int(label[0]); self.refresh()

    def _on_slider(self, k, val):
        if k == 'brush':
            self.brush = int(val); self.cursor.set_radius(self.brush)
        elif k in ('plane', 'plane_lo', 'plane_hi'):
            if k == 'plane_lo':
                self.mipd['plane_lo'] = float(val)
            if k == 'plane_hi':
                self.mipd['plane_hi'] = float(val)
            if self.mode != 'Single plane':
                self.rb_mode.set_active(self.MODES.index('Single plane')); return   # refresh happens there
        elif k == 'focus_win_um':
            self.mipd['focus_win_um'] = float(val)
        elif k in ('depth_floor', 'depth_gamma', 'depth_floor_thr'):
            self.mipd[k] = float(val)
            if self.mode != 'Depth-coded':
                self.rb_mode.set_active(1); return
        elif k in ('zmin', 'zmax'):
            self.mipd['z_range'] = self._zr()
        else:
            self.mipd[k] = float(val)
            if self.mipd['hi'] <= self.mipd['lo'] + 1e-3:
                self.mipd['hi'] = self.mipd['lo'] + 1e-3
        self.refresh()

    def _push(self):
        self.hist.append(self.mask.copy()); self.hist = self.hist[-30:]; self._mask_changed()

    def _on_poly(self, verts):
        from matplotlib.path import Path as MplPath
        if len(verts) >= 3:
            self._push()
            self.mask = MplPath(verts).contains_points(np.column_stack([self.xx.ravel(), self.yy.ravel()])).reshape(self.H, self.W)
            try:
                self.ps.clear()
            except Exception:
                pass
            self.refresh()

    def _on_ellipse(self, eclick, erelease):
        x0, x1, y0, y1 = self.es.extents
        if (x1 - x0) > 2 and (y1 - y0) > 2:
            self._push()
            cx, cy, a, b = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            self.mask = ((self.xx - cx) / a) ** 2 + ((self.yy - cy) / b) ** 2 <= 1
            self.refresh()

    def _stamp(self, x, y, val):
        r = self.brush; x0, x1 = max(0, int(x) - r - 1), min(self.W, int(x) + r + 2); y0, y1 = max(0, int(y) - r - 1), min(self.H, int(y) + r + 2)
        sub = (self.xx[y0:y1, x0:x1] - x) ** 2 + (self.yy[y0:y1, x0:x1] - y) ** 2 <= r * r
        self.mask[y0:y1, x0:x1][sub] = val

    def _stroke(self, x, y, val):
        if self.last is not None:
            lx, ly = self.last; n = int(max(1, np.hypot(x - lx, y - ly) / max(self.brush / 2.0, 1)))
            for t in np.linspace(0, 1, n + 1):
                self._stamp(lx + t * (x - lx), ly + t * (y - ly), val)
        else:
            self._stamp(x, y, val)
        self.last = (x, y)

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, 'toolbar', None)
        return bool(getattr(tb, 'mode', ''))

    def _press(self, ev):
        if self.tool != 'Brush' or ev.inaxes != self.ax or ev.xdata is None or ev.button not in (1, 3) or self._toolbar_busy() or self.mode == 'Binned MIPs':
            return
        self._push(); self.btn = int(ev.button); self.last = None
        self._stroke(ev.xdata, ev.ydata, self.btn == 1); self._blit((ev.xdata, ev.ydata))

    def _move(self, ev):
        if self.tool != 'Brush' or self.mode == 'Binned MIPs':
            return
        if ev.inaxes != self.ax or ev.xdata is None:
            if self.cursor.get_visible():
                self._blit(None)
            return
        if self.btn is not None:
            self._stroke(ev.xdata, ev.ydata, self.btn == 1)
        self._blit((ev.xdata, ev.ydata))

    def _release(self, ev):
        if self.btn is not None:
            self.btn = None; self.last = None; self._mask_changed(); self.refresh()

    def _key(self, ev):
        if ev.key == ']':
            self.sl['brush'].set_val(min(40, self.brush + 2))
        elif ev.key == '[':
            self.sl['brush'].set_val(max(1, self.brush - 2))
        elif ev.key == 'z':
            self._undo(None)
        elif ev.key in ('up', 'down', '.', ','):
            k = self._plane_index() + (1 if ev.key in ('up', '.') else -1)
            k = max(0, min(self.D - 1, k)); self.sl['plane'].set_val(float(self.z[k]))

    def _undo(self, ev):
        if self.hist:
            self.mask = self.hist.pop(); self._mask_changed(); self.refresh()

    def _clear(self, ev):
        self._push(); self.mask = np.zeros((self.H, self.W), bool); self.refresh()

    def _all(self, ev):
        self._push(); self.mask = np.ones((self.H, self.W), bool); self.refresh()

    def _invert(self, ev):
        self._push(); self.mask = ~self.mask; self.refresh()

    def _load_mask(self, ev):
        p = os.path.join(self.out, 'mask.png')
        if os.path.exists(p):
            self._push(); self.mask = load_mask_file(p, self.H, self.W, self.log); self.refresh()
        else:
            self.log(f"[gui] no {p} to load")

    def _save_mask(self, ev=None):
        from PIL import Image
        if not self.mask.any():
            self.log("[gui] mask is empty — not saved"); return None
        Image.fromarray((self.mask * 255).astype(np.uint8)).save(os.path.join(self.out, 'mask.png'))
        np.save(os.path.join(self.out, 'mask.npy'), self.mask)
        json.dump(dict(type='studio', px=int(self.mask.sum())), open(os.path.join(self.out, 'mask.json'), 'w'), indent=1)
        self.log(f"[gui] mask saved ({int(self.mask.sum())} px) -> mask.png"); return os.path.join(self.out, 'mask.png')

    def _auto(self, key, spec):
        v = parse_level(spec, self._mip(), self.mask); self.sl[key].set_val(float(np.clip(v, self.sl[key].valmin, self.sl[key].valmax)))

    def settings(self):
        md = self.mipd
        return dict(lo=float(md['lo']), hi=float(md['hi']), gamma=float(md['gamma']), smooth_px=float(md['smooth_px']), upsample=int(md['upsample']),
                    bg_method=md['bg_method'], bg_radius_um=float(md['bg_radius_um']), z_range=self._zr(),
                    proj_mode=md['proj_mode'], focus_win_um=float(md['focus_win_um']),
                    plane_lo=float(self.sl['plane_lo'].val), plane_hi=float(self.sl['plane_hi'].val),
                    depth_bright=md['depth_bright'], depth_floor=float(md['depth_floor']), depth_gamma=float(md['depth_gamma'] or md['gamma']),
                    depth_floor_thr=float(md['depth_floor_thr']), depth_local_px=int(md.get('depth_local_px', 17)),
                    z_weights=(md.get('z_weights') or []) if self.zw_on else [])

    def _save_settings(self, ev=None):
        p = os.path.join(self.out, 'mip_display.json'); json.dump(self.settings(), open(p, 'w'), indent=1)
        self.log(f"[gui] settings saved -> {p}: {self.settings()}"); return p

    def _export(self, video):
        if self._save_mask() is None:
            return
        self._save_settings(); self.result = dict(action='export', video=bool(video), mask=self.mask.copy(), mipd=self.settings())
        self.plt.close(self.fig)

    def _quit(self, ev):
        self.result = dict(action='quit'); self.plt.close(self.fig)

    def run(self):
        self.plt.show()
        if self.result is None:                          # window closed with the [x]: accept the current settings
            if self._save_mask() is None:
                self.log("[gui] window closed with an empty mask -> nothing exported"); return None
            self._save_settings()
            self.result = dict(action='export', video=self.default_video, mask=self.mask.copy(), mipd=self.settings())
            self.log(f"[gui] window closed -> exporting with the current settings (video={self.default_video})")
        return self.result


# ════════════════════════════════════════════════════════════════════════════
# 5. FWHM vs z (main.py helpers) and SBR/CNR vs z
# ════════════════════════════════════════════════════════════════════════════
def perp_profile(img2d, center_xy, tangent_xy, length_px, n_samples=None):
    from scipy.ndimage import map_coordinates
    cx, cy = float(center_xy[0]), float(center_xy[1]); tx, ty = float(tangent_xy[0]), float(tangent_xy[1])
    px, py = -ty, tx; L = float(length_px)
    n = int(round(L) + 1) if n_samples is None else int(n_samples)
    s = np.linspace(-L / 2.0, L / 2.0, n)
    prof = map_coordinates(img2d, np.vstack([cy + s * py, cx + s * px]), order=1, mode='nearest')
    return prof.astype(np.float32), s


def fwhm_from_profile(profile, frac=0.5):
    p = np.asarray(profile, float)
    if p.size < 3:
        return float('nan'), -1, False, False
    base = float(p.min()); peak = float(p.max())
    if peak - base < 1e-9:
        return float('nan'), -1, False, False
    thr = base + frac * (peak - base); above = p >= thr; pk = int(np.argmax(p))
    li = pk
    while li > 0 and above[li - 1]:
        li -= 1
    ri = pk
    while ri < len(p) - 1 and above[ri + 1]:
        ri += 1
    left = 0.0 if li == 0 else (li - 1) + (thr - p[li - 1]) / (p[li] - p[li - 1] + 1e-12)
    right = float(len(p) - 1) if ri == len(p) - 1 else ri + (thr - p[ri]) / (p[ri + 1] - p[ri] + 1e-12)
    w = float(right - left)
    return (w if w > 0 else float('nan')), pk, li > 0, ri < len(p) - 1


def auto_centerline_points(img, mask, px_um, sigmas_um=(4.0, 8.0, 12.0), vessel_pct=97.0,
                           spacing_px=3, nbr_r=4.0, min_nbrs=5, min_component_px=30):
    from skimage.filters import sato
    from skimage.morphology import skeletonize, remove_small_objects
    from scipy.spatial import cKDTree
    sig = [max(0.7, s / px_um) for s in sigmas_um]
    v = sato(img.astype(np.float32), sigmas=sig, black_ridges=False)
    v = np.where(mask, v, 0); vals = v[mask]
    if vals.size == 0 or vals.max() <= 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bw = v > max(float(np.percentile(vals, vessel_pct)), 1e-12)
    try:
        bw = remove_small_objects(bw, max_size=min_component_px - 1)
    except TypeError:
        bw = remove_small_objects(bw, min_size=min_component_px)
    sk = skeletonize(bw); ys, xs = np.nonzero(sk)
    if len(xs) < min_nbrs + 1:
        return np.zeros((0, 2)), np.zeros((0, 2))
    P = np.column_stack([xs, ys]).astype(float); tree = cKDTree(P)
    order = np.lexsort((xs, ys)); keep = np.zeros(len(P), bool); taken = np.zeros(len(P), bool)
    for i in order:
        if taken[i]:
            continue
        keep[i] = True
        for j in tree.query_ball_point(P[i], spacing_px - 1e-6):
            taken[j] = True
    centers, tangents = [], []
    for i in np.nonzero(keep)[0]:
        nb = tree.query_ball_point(P[i], nbr_r)
        if len(nb) < min_nbrs:
            continue
        Q = P[nb] - P[nb].mean(axis=0); w, vec = np.linalg.eigh(Q.T @ Q)
        tv = vec[:, int(np.argmax(w))]
        centers.append(P[i]); tangents.append(tv / (np.linalg.norm(tv) + 1e-12))
    if not centers:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(centers), np.asarray(tangents)


def measure_fwhm_vs_depth(raw, mask, px_um, z_um, profile_len_um=80.0, min_contrast=0.3, max_fwhm_um=150.0,
                          center_tol_px=2.0, focus_halfwin=2, vessels_json=None, log=print, **det_kw):
    H, W, D = raw.shape
    L_px = profile_len_um / px_um
    dist = ndi.distance_transform_edt(mask) if not mask.all() else np.full((H, W), 1e9)
    edge_ok = dist > (L_px / 2.0 + 1)
    rows = []; manual = None
    if vessels_json:
        manual = json.load(open(vessels_json)); log(f"  user centerlines: {len(manual)} polylines")
    for z in range(D):
        img = raw[:, :, z]
        if manual is None:
            C, Tg = auto_centerline_points(img, mask & edge_ok, px_um, **det_kw)
        else:
            C, Tg = [], []
            for item in manual:
                if int(item.get('z', -1)) != z:
                    continue
                pts = np.asarray(item['points'], float)
                if pts.shape[0] < 2:
                    continue
                seg = np.diff(pts, axis=0); seglen = np.hypot(seg[:, 0], seg[:, 1]); cum = np.concatenate([[0], np.cumsum(seglen)])
                for s in np.arange(0, cum[-1], 3.0):
                    k = min(int(np.searchsorted(cum, s, side='right') - 1), len(seg) - 1)
                    tt = (s - cum[k]) / (seglen[k] + 1e-12)
                    C.append(pts[k] + tt * seg[k]); Tg.append(seg[k] / (seglen[k] + 1e-12))
            C = np.asarray(C).reshape(-1, 2); Tg = np.asarray(Tg).reshape(-1, 2)
        n_cand, n_ok = len(C), 0
        for (x, y), tv in zip(C, Tg):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H) or not edge_ok[yi, xi]:
                continue
            z0, z1 = max(0, z - focus_halfwin), min(D, z + focus_halfwin + 1)
            if raw[yi, xi, z] < raw[yi, xi, z0:z1].max() - 1e-9:
                continue
            prof, s = perp_profile(img, (x, y), tv, L_px)
            w_px, pk, l_ok, r_ok = fwhm_from_profile(prof, 0.5)
            if not np.isfinite(w_px) or not (l_ok and r_ok) or abs(s[pk]) > center_tol_px:
                continue
            base = float(prof.min()); peak = float(prof.max())
            contrast = (peak - base) / max(abs(base), 1e-9 * max(peak, 1e-9))
            if contrast < min_contrast:
                continue
            w_um = w_px * px_um
            if w_um > max_fwhm_um:
                continue
            rows.append(dict(plane=z, z_um=float(z_um[z]), x_px=float(x), y_px=float(y), tangent_x=float(tv[0]),
                             tangent_y=float(tv[1]), fwhm_um=float(w_um), peak=peak, base=base, contrast=float(contrast)))
            n_ok += 1
        log(f"  plane {z:2d} (z = {z_um[z]:6.0f} µm): candidates {n_cand:4d} -> accepted {n_ok:4d}")
    return rows


def summarize_fwhm(rows, z_um):
    by = {}
    for r in rows:
        by.setdefault(r['plane'], []).append(r['fwhm_um'])
    out = []
    for z, d in enumerate(z_um):
        v = np.asarray(by.get(z, []), float)
        if v.size:
            out.append(dict(plane=z, z_um=float(d), n=int(v.size), p10=float(np.percentile(v, 10)), p25=float(np.percentile(v, 25)),
                            median=float(np.median(v)), p75=float(np.percentile(v, 75)), min=float(v.min())))
        else:
            out.append(dict(plane=z, z_um=float(d), n=0, p10=np.nan, p25=np.nan, median=np.nan, p75=np.nan, min=np.nan))
    return out


def sbr_cnr_vs_depth(raw, mask, z_um, top_frac=0.01, hp_size=5, jump=1.5):
    H, W, D = raw.shape; rows = []
    for z in range(D):
        img = raw[:, :, z]; vals = img[mask]; n = vals.size; k = max(1, int(round(top_frac * n)))
        idx = np.argpartition(vals, n - k)[n - k:]
        struct = float(vals[idx].mean()); thr = float(vals[idx].min())
        bg_mask = mask & (img < thr)
        bg = float(np.median(img[bg_mask])) if bg_mask.any() else float('nan')
        hp = img - ndi.uniform_filter(img, size=hp_size)
        mad = float(np.median(np.abs(hp[bg_mask] - np.median(hp[bg_mask])))) if bg_mask.any() else float('nan')
        sigma = 1.4826 * mad
        rows.append(dict(plane=z, z_um=float(z_um[z]), structure_mean_top1pct=struct, background_median=bg,
                         sbr=struct / bg if bg > 0 else float('nan'), sigma_mad=sigma,
                         cnr=(struct - bg) / sigma if sigma > 0 else float('nan'), trimmed=False))
    s = [r['sbr'] for r in rows]
    for _ in range(2):
        live = [i for i, r in enumerate(rows) if not r['trimmed']]
        if len(live) < 3:
            break
        for a, b in ((live[0], live[1]), (live[-1], live[-2])):
            if np.isfinite(s[a]) and np.isfinite(s[b]) and (s[a] > jump * s[b] or s[a] < s[b] / jump):
                rows[a]['trimmed'] = True
    return rows


# ════════════════════════════════════════════════════════════════════════════
# 6. Figure panels (Nature style)
# ════════════════════════════════════════════════════════════════════════════
def nature_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False,
        'lines.linewidth': 1.0, 'figure.dpi': 150, 'savefig.dpi': 300, 'mathtext.default': 'regular'})


def add_scalebar(ax, px_um, length_um, W, H, color='white', frac_x=0.05, frac_y=0.93, lw=2.0):
    Lpx = length_um / px_um; x0 = frac_x * W; y0 = frac_y * H
    ax.plot([x0, x0 + Lpx], [y0, y0], color=color, lw=lw, solid_capstyle='butt')


def save_fig(fig, base):
    fig.savefig(base + '.svg'); fig.savefig(base + '.pdf'); fig.savefig(base + '.png', dpi=300)


def save_png8(arr01, path):
    from PIL import Image
    a = np.clip(arr01, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def panel_images(disp, rgb, proj_g, depth_norm, z_um, px_um, dz_um, out, args, soft, log, mipd, disp_mip=None):
    """Panels: MIP (gray), depth-coded MIP, depth-binned MIPs, depth slices + x–z. Returns dict for pptx/metadata.
    mipd = dict(lo, hi, gamma, smooth_px, upsample): MIP display settings (shared by all MIP-type images)."""
    import matplotlib.pyplot as plt
    import tifffile
    from matplotlib.colors import Normalize
    from matplotlib import cm
    H, W, D = disp.shape
    up = int(mipd['upsample']); px_mip = px_um / up; Hm, Wm = proj_g.shape
    res = dict(images={}, px_mip_um=px_mip)
    # layers
    tifffile.imwrite(os.path.join(out, 'panel_b_mip_grayscale_16bit.tif'), (np.clip(proj_g, 0, 1) * 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(out, 'panel_b_depth_map_16bit.tif'), (depth_norm * 65535).astype(np.uint16))
    save_png8(rgb, os.path.join(out, 'img_depth_coded_rgb.png')); res['images']['depth_coded'] = 'img_depth_coded_rgb.png'
    save_png8(proj_g, os.path.join(out, 'img_mip_grayscale.png')); res['images']['mip'] = 'img_mip_grayscale.png'
    # colourbar image for pptx
    fig = plt.figure(figsize=(0.35, 2.2)); cax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    sm = cm.ScalarMappable(norm=Normalize(z_um[0], z_um[-1]), cmap='turbo'); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax); cb.set_ticks([]); cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, 'img_colorbar_turbo.png'), dpi=300); plt.close(fig)
    # b: grayscale MIP
    fig = plt.figure(figsize=(52 * MM, 52 * MM)); ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.imshow(np.clip(proj_g, 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); ax.axis('off')
    add_scalebar(ax, px_mip, args.scalebar_um, Wm, Hm); fig.text(0, 1, 'b', fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_b_MIP_grayscale')); plt.close(fig)
    # b': depth-coded
    fig = plt.figure(figsize=(62 * MM, 52 * MM)); ax = fig.add_axes([0.02, 0.04, 0.78, 0.92])
    ax.imshow(rgb, interpolation='bicubic'); ax.axis('off'); add_scalebar(ax, px_mip, args.scalebar_um, Wm, Hm)
    cax = fig.add_axes([0.83, 0.12, 0.035, 0.72]); cb = fig.colorbar(sm, cax=cax); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2); cax.set_title('z\n(µm)', fontsize=7, pad=3)
    fig.text(0, 1, "b′", fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_b_depth_coded_MIP')); plt.close(fig)
    # depth-binned MIPs
    bins_info = []
    if str(args.mip_bins).lower() != 'none':
        edges = np.linspace(z_um[0], z_um[-1], 4) if str(args.mip_bins).lower() == 'auto' else [float(v) for v in args.mip_bins.split(',')]
        pairs = list(zip(edges[:-1], edges[1:])); n = len(pairs)
        fig = plt.figure(figsize=((42 * n) * MM, 46 * MM))
        for k, (lo, hi) in enumerate(pairs):
            sel = (z_um >= lo) & ((z_um < hi) if k < n - 1 else (z_um <= hi))
            if not sel.any():
                continue
            mip, _ = mip_and_depth((disp_mip if disp_mip is not None else disp)[:, :, sel], mipd['smooth_px'], **proj_kw(mipd, px_um))
            if args.mip_bin_norm == 'global':
                blo, bhi = mipd['lo'], mipd['hi']
            else:
                blo, bhi = mipd['lo'], max(float(np.percentile(mip[mip > mipd['lo']], 99.9)) if (mip > mipd['lo']).any() else 1.0, mipd['lo'] + 1e-6)
            shown = np.clip(upsample_img(apply_levels(mip, blo, bhi, mipd['gamma']), up, 3), 0, 1) * (upsample_img(soft, up, 1) if up > 1 else soft)
            fn = f'img_mip_bin_{k}_{int(lo)}_{int(hi)}um'
            tifffile.imwrite(os.path.join(out, fn + '_16bit.tif'), (np.clip(mip, 0, 1) * 65535).astype(np.uint16))
            save_png8(shown, os.path.join(out, fn + '.png'))
            ax = fig.add_axes([k / n + 0.01, 0.06, 1 / n - 0.02, 0.84]); ax.imshow(shown, cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); ax.axis('off')
            ax.set_title(f'{lo:.0f} to {hi:.0f} µm ({int(sel.sum())} planes)', pad=2)
            if k == 0:
                add_scalebar(ax, px_mip, args.scalebar_um, Wm, Hm)
            bins_info.append(dict(lo_um=float(lo), hi_um=float(hi), n_planes=int(sel.sum()), display_lo=blo, display_hi=bhi, png=fn + '.png'))
        fig.text(0, 1, "b″", fontsize=8, fontweight='bold', va='top')
        save_fig(fig, os.path.join(out, 'panel_mip_depth_bins')); plt.close(fig)
    res['mip_bins'] = bins_info
    # c: slices + x–z  (clipping like the MIP: --slice-min/--slice-max, --xz-min/--xz-max; optional same background removal)
    if str(args.slices).lower() == 'auto':
        slice_z = [z_um[0] + f * (z_um[-1] - z_um[0]) for f in (0.25, 0.5, 0.75)]
    else:
        slice_z = [float(v) for v in args.slices.split(',')]
    idx = [int(np.argmin(np.abs(z_um - zq))) for zq in slice_z]
    mask_hw = soft > 0.5
    if args.xz_row is not None:
        xz_row = int(args.xz_row)
    elif args.ortho_center == 'brightest':
        xz_row = int(np.argmax((disp.max(2) * soft).sum(axis=1)))
    else:                                                            # centre of the mask (FOV centre if no mask)
        ys_, xs_ = np.nonzero(mask_hw) if mask_hw.any() else (np.arange(H), np.arange(W))
        xz_row = int(round(float(np.mean(ys_)))); yz_col_c = int(round(float(np.mean(xs_))))
    s_gamma = args.slice_gamma if args.slice_gamma is not None else args.gamma
    if args.slice_bg == 'same' and mipd.get('bg_method', 'none') != 'none':
        vol_s = subtract_background(disp, mipd['bg_method'], mipd['bg_radius_um'] / px_um)
        if mipd.get('smooth_px', 0) > 0:
            vol_s = ndi.gaussian_filter(vol_s, sigma=(mipd['smooth_px'], mipd['smooth_px'], 0))
        s_bg = f"{mipd['bg_method']} R={mipd['bg_radius_um']:g} µm, smooth {mipd.get('smooth_px', 0):g} px"
    else:
        vol_s = disp; s_bg = 'none'
    # global levels = percentiles over ALL planes inside the mask (one setting for every slice)
    g_lo = g_hi = None
    if args.slice_norm == 'global':
        vals = vol_s[mask_hw]                                            # (N, D)
        g_lo = float(np.percentile(vals, float(args.slice_min[1:]))) if str(args.slice_min).lower().startswith('p') else float(args.slice_min)
        g_hi = float(np.percentile(vals, float(args.slice_max[1:]))) if str(args.slice_max).lower().startswith('p') else float(args.slice_max)
    ncol = len(idx) + 1
    fig = plt.figure(figsize=((42 * ncol) * MM, 46 * MM)); slices_info = []
    for k, zi in enumerate(idx):
        img = vol_s[:, :, zi]
        if args.slice_norm == 'manual' and mipd.get('plane_lo') is not None and mipd.get('plane_hi') is not None:
            vmin_s, vmax = float(mipd['plane_lo']), float(mipd['plane_hi'])
        elif args.slice_norm == 'global':
            vmin_s, vmax = g_lo, g_hi
        else:                                                            # per-plane percentiles of THIS plane inside the mask
            vmin_s = parse_level(args.slice_min, img, mask_hw); vmax = parse_level(args.slice_max, img, mask_hw)
        vmax = max(vmax, vmin_s + 1e-6)
        shown = apply_levels(img, vmin_s, vmax, s_gamma) * soft
        fn = f'img_slice_{k}_z{int(round(z_um[zi]))}um'
        tifffile.imwrite(os.path.join(out, fn + '_16bit.tif'), (np.clip(disp[:, :, zi], 0, 1) * 65535).astype(np.uint16))
        save_png8(shown, os.path.join(out, fn + '.png'))
        ax = fig.add_axes([k / ncol + 0.01, 0.06, 1 / ncol - 0.02, 0.84]); ax.imshow(shown, cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); ax.axis('off')
        ax.set_title(f'z = {z_um[zi]:.0f} µm', pad=2)
        if k == 0:
            add_scalebar(ax, px_um, args.scalebar_um, W, H)
        plo_s = float((img[mask_hw] <= vmin_s).mean() * 100); phi_s = float((img[mask_hw] <= vmax).mean() * 100)
        slices_info.append(dict(plane=int(zi), z_um=float(z_um[zi]), display_vmin=vmin_s, display_vmax=vmax, min_percentile=plo_s, max_percentile=phi_s, png=fn + '.png'))
    r0, r1 = max(0, xz_row - args.xz_slab_px // 2), min(H, xz_row + args.xz_slab_px // 2 + 1)
    xz = vol_s[r0:r1, :, :].max(axis=0).T                                # (D, W)
    xz_mask = np.broadcast_to(mask_hw[xz_row][None, :], xz.shape)
    xz_lo = parse_level(args.xz_min, xz, xz_mask) if xz_mask.any() else 0.0
    xz_hi = max(parse_level(args.xz_max, xz, xz_mask) if xz_mask.any() else float(np.percentile(xz, 99.9)), xz_lo + 1e-6)
    xz_show = apply_levels(xz, xz_lo, xz_hi, s_gamma)
    tifffile.imwrite(os.path.join(out, 'img_xz_section_16bit.tif'), (np.clip(disp[r0:r1, :, :].max(axis=0).T, 0, 1) * 65535).astype(np.uint16))
    save_png8(xz_show, os.path.join(out, 'img_xz_section.png'))
    ax = fig.add_axes([(ncol - 1) / ncol + 0.09, 0.2, 1 / ncol - 0.11, 0.62])
    ax.imshow(xz_show, cmap='gray', vmin=0, vmax=1, interpolation='bicubic',
              extent=[0, W * px_um, z_um[-1] + dz_um / 2, z_um[0] - dz_um / 2])
    ax.set_aspect('equal'); ax.set_xlabel('x (µm)'); ax.set_ylabel('z (µm)'); ax.set_title(f'x–z, y = {xz_row * px_um:.0f} µm', pad=2); ax.tick_params(length=2)
    fig.text(0, 1, 'c', fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_c_depth_slices_xz')); plt.close(fig)
    log(f"[slices] norm {args.slice_norm}, bg {s_bg}, gamma {s_gamma:g}; " + "; ".join(f"z {si['z_um']:.0f}: min {si['display_vmin']:.3f} (p{si['min_percentile']:.0f}) max {si['display_vmax']:.3f} (p{si['max_percentile']:.1f})" for si in slices_info)
        + f"; x–z: min {xz_lo:.3f} max {xz_hi:.3f}  [display only]")
    res.update(slice_display=dict(norm=args.slice_norm, bg=s_bg, gamma=s_gamma, slice_min=args.slice_min, slice_max=args.slice_max, xz_min=args.xz_min, xz_max=args.xz_max,
                                  xz_lo=xz_lo, xz_hi=xz_hi))
    # ── orthogonal views: x–y (MIP, same display as panel b) + x–z (row) + y–z (column), physical aspect
    if args.yz_col is not None:
        yz_col = int(args.yz_col)
    elif args.ortho_center == 'brightest':
        yz_col = int(np.argmax((disp.max(2) * soft).sum(axis=0)))
    else:
        yz_col = yz_col_c
    c0, c1 = max(0, yz_col - args.xz_slab_px // 2), min(W, yz_col + args.xz_slab_px // 2 + 1)
    yz = vol_s[:, c0:c1, :].max(axis=1)                                  # (H, D): y down, z across
    yz_mask = np.broadcast_to(mask_hw[:, yz_col][:, None], yz.shape)
    yz_lo = parse_level(args.xz_min, yz, yz_mask) if yz_mask.any() else 0.0
    yz_hi = max(parse_level(args.xz_max, yz, yz_mask) if yz_mask.any() else float(np.percentile(yz, 99.9)), yz_lo + 1e-6)
    yz_show = apply_levels(yz, yz_lo, yz_hi, s_gamma)
    tifffile.imwrite(os.path.join(out, 'img_yz_section_16bit.tif'), (np.clip(disp[:, c0:c1, :].max(axis=1), 0, 1) * 65535).astype(np.uint16))
    save_png8(yz_show, os.path.join(out, 'img_yz_section.png'))
    k0 = int(np.argmin(np.abs(z_um)))                                    # centre plane (z closest to 0)
    if args.ortho_xy == 'center':
        cimg = vol_s[:, :, k0]
        if args.slice_norm == 'manual' and mipd.get('plane_lo') is not None and mipd.get('plane_hi') is not None:
            c_lo, c_hi = float(mipd['plane_lo']), float(mipd['plane_hi'])
        elif args.slice_norm == 'global':
            c_lo, c_hi = g_lo, g_hi
        else:
            c_lo, c_hi = parse_level(args.slice_min, cimg, mask_hw), parse_level(args.slice_max, cimg, mask_hw)
        xy_gray = apply_levels(cimg, c_lo, max(c_hi, c_lo + 1e-6), s_gamma) * soft
        xy_label = f'x–y (centre plane, z = {z_um[k0]:.0f} µm)'
    else:
        xy_gray = np.clip(proj_g, 0, 1); xy_label = 'x–y (MIP)'
    save_png8(xy_gray, os.path.join(out, 'img_xy_ortho.png'))
    span = float(z_um[-1] - z_um[0] + dz_um)
    Wum, Hum = W * px_um, H * px_um
    fig = plt.figure(figsize=(62 * MM * (Wum + span) / Wum, 62 * MM * (Hum + span) / Hum))
    gs = fig.add_gridspec(2, 2, width_ratios=[Wum, span], height_ratios=[Hum, span], left=0.11, right=0.98, top=0.94, bottom=0.09, wspace=0.04, hspace=0.04)
    ax_xy = fig.add_subplot(gs[0, 0]); ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy); ax_yz = fig.add_subplot(gs[0, 1], sharey=ax_xy)
    ax_xy.imshow(xy_gray, cmap='gray', vmin=0, vmax=1, interpolation='bicubic', extent=[0, Wum, Hum, 0])
    ax_xy.axhline((xz_row + 0.5) * px_um, color=OI['yellow'], lw=0.6, ls='--'); ax_xy.axvline((yz_col + 0.5) * px_um, color=OI['sky'], lw=0.6, ls='--')
    ax_xz.imshow(xz_show, cmap='gray', vmin=0, vmax=1, interpolation='bicubic', extent=[0, Wum, z_um[-1] + dz_um / 2, z_um[0] - dz_um / 2])
    ax_xz.axvline((yz_col + 0.5) * px_um, color=OI['sky'], lw=0.6, ls='--')
    ax_yz.imshow(yz_show, cmap='gray', vmin=0, vmax=1, interpolation='bicubic', extent=[z_um[0] - dz_um / 2, z_um[-1] + dz_um / 2, Hum, 0])
    ax_yz.axhline((xz_row + 0.5) * px_um, color=OI['yellow'], lw=0.6, ls='--')
    def _axes_indicator(ax, h_label, v_label, frac=0.16, color='white'):
        """L-shaped direction arrows in the bottom-left corner: horizontal -> h_label, vertical (down) -> v_label."""
        x0, y0 = 0.06, 0.94                                          # axes fraction: corner of the L near the TOP-left (image y runs downward)
        ap = dict(arrowstyle='-|>', color=color, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0)
        ax.annotate('', xy=(x0 + frac, y0), xytext=(x0, y0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=ap)       # -> right
        ax.annotate('', xy=(x0, y0 - frac), xytext=(x0, y0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=ap)       # -> down
        ax.text(x0 + frac + 0.02, y0, h_label, transform=ax.transAxes, color=color, fontsize=6, va='center', ha='left', fontweight='bold')
        ax.text(x0, y0 - frac - 0.02, v_label, transform=ax.transAxes, color=color, fontsize=6, va='top', ha='center', fontweight='bold')
    for a in (ax_xy, ax_xz, ax_yz):
        a.set_aspect('equal'); a.tick_params(length=2, labelsize=6)
    _axes_indicator(ax_xy, 'x', 'y', 0.16); _axes_indicator(ax_xz, 'x', 'z', 0.16); _axes_indicator(ax_yz, 'z', 'y', 0.16)
    ax_xy.set_ylabel('y (µm)'); ax_xz.set_xlabel('x (µm)'); ax_xz.set_ylabel('z (µm)'); ax_yz.set_xlabel('z (µm)')
    ax_xy.tick_params(labelbottom=False); ax_yz.tick_params(labelleft=False)
    ax_xy.set_title(f'{xy_label}   x–z at y = {(xz_row + 0.5) * px_um:.0f} µm   y–z at x = {(yz_col + 0.5) * px_um:.0f} µm', fontsize=6, pad=2)
    fig.text(0, 1, 'c′', fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_c2_orthogonal_views')); plt.close(fig)
    log(f"[ortho] x–z row {xz_row} (y = {(xz_row + 0.5) * px_um:.0f} µm), y–z column {yz_col} (x = {(yz_col + 0.5) * px_um:.0f} µm); y–z levels {yz_lo:.3f}/{yz_hi:.3f}")
    res.update(ortho=dict(xz_row_px=int(xz_row), yz_col_px=int(yz_col), yz_lo=yz_lo, yz_hi=yz_hi, span_um=span, W=W, H=H, xy=args.ortho_xy, xy_label=xy_label, center_plane_z=float(z_um[k0])))
    # optional: panel b as the centre plane instead of the MIP
    if args.panel_b == 'center':
        fig = plt.figure(figsize=(52 * MM, 52 * MM)); ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
        cb_img = xy_gray if args.ortho_xy == 'center' else apply_levels(vol_s[:, :, k0], *(parse_level(args.slice_min, vol_s[:, :, k0], mask_hw), parse_level(args.slice_max, vol_s[:, :, k0], mask_hw)), s_gamma) * soft
        ax.imshow(cb_img, cmap='gray', vmin=0, vmax=1, interpolation='bicubic'); ax.axis('off')
        add_scalebar(ax, px_um, args.scalebar_um, W, H); fig.text(0, 1, 'b', fontsize=8, fontweight='bold', va='top')
        save_fig(fig, os.path.join(out, 'panel_b_center_plane')); plt.close(fig)
        save_png8(cb_img, os.path.join(out, 'img_center_plane.png')); res['images']['panel_b'] = 'img_center_plane.png'; res['panel_b_label'] = f'Centre plane (z = {z_um[k0]:.0f} µm)'
    else:
        res['images']['panel_b'] = res['images']['mip']; res['panel_b_label'] = 'Maximum-intensity projection'
    res.update(slices=slices_info, xz_row_px=int(xz_row), xz_slab_um=float(args.xz_slab_px * px_um),
               xz_extent_um=[0.0, float(W * px_um), float(z_um[0] - dz_um / 2), float(z_um[-1] + dz_um / 2)])
    return res


def _plot_curve(ax, summ, color, label):
    d = np.array([r['z_um'] for r in summ]); n = np.array([r['n'] for r in summ]); ok = n > 0
    med = np.array([r['median'] for r in summ]); lo = np.array([r['p25'] for r in summ]); hi = np.array([r['p75'] for r in summ]); q = np.array([r['p10'] for r in summ])
    ax.fill_between(d[ok], lo[ok], hi[ok], color=color, alpha=0.18, lw=0)
    ax.plot(d[ok], med[ok], '-o', color=color, ms=2.5, mew=0, label=label)
    ax.plot(d[ok], q[ok], '--', color=color, lw=0.8)


def panel_d(summ, out, label_main, summ_ref=None, label_ref='Ear', lat_res_um=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(58 * MM, 48 * MM)); ax = fig.add_axes([0.2, 0.19, 0.76, 0.68])
    _plot_curve(ax, summ, OI['blue'], label_main)
    if summ_ref is not None:
        _plot_curve(ax, summ_ref, OI['orange'], label_ref)
    if lat_res_um:
        ax.axhline(lat_res_um, color=OI['grey'], ls=':', lw=0.8)
        ax.text(1.0, lat_res_um, 'lateral res.', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=6, color=OI['grey'])
    ax.set_xlabel('Axial position z (µm)'); ax.set_ylabel('Apparent vessel FWHM (µm)'); ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', handlelength=1.6)
    ax.text(0.98, 0.98, 'solid: median (IQR)\ndashed: 10th pct', transform=ax.transAxes, ha='right', va='top', fontsize=6, color=OI['grey'])
    for r in summ:
        if r['n'] > 0:
            ax.text(r['z_um'], 1.0, str(r['n']), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5, color=OI['blue'])
    fig.text(0, 1, 'd', fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_d_fwhm_vs_depth')); plt.close(fig)


def panel_e(rows, out, label_main, rows_ref=None, label_ref='Ear', ref_csv=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(58 * MM, 46 * MM)); ax = fig.add_axes([0.2, 0.2, 0.76, 0.72])

    def _draw(rr, color, lab):
        live = [r for r in rr if not r['trimmed'] and np.isfinite(r['sbr'])]
        ax.plot([r['z_um'] for r in live], [r['sbr'] for r in live], '-o', color=color, ms=2.5, mew=0, label=lab)
        tr = [r for r in rr if r['trimmed'] and np.isfinite(r['sbr'])]
        if tr:
            ax.plot([r['z_um'] for r in tr], [r['sbr'] for r in tr], 'x', color=color, ms=3, mew=0.8, alpha=0.5)
    _draw(rows, OI['blue'], label_main)
    if rows_ref is not None:
        _draw(rows_ref, OI['orange'], label_ref)
    if ref_csv:
        arr = np.genfromtxt(ref_csv, delimiter=',', names=True); cols = list(arr.dtype.names)
        dcol = next((c for c in cols if 'depth' in c.lower() or c.lower().startswith('z')), cols[0]); scol = next((c for c in cols if 'sbr' in c.lower()), cols[1])
        ax.plot(arr[dcol], arr[scol], '-s', color=OI['orange'], ms=2.5, mew=0, label=label_ref)
    ax.axhline(1.0, color=OI['grey'], ls=':', lw=0.8)
    ax.set_xlabel('Axial position z (µm)'); ax.set_ylabel('SBR'); ax.set_ylim(bottom=0); ax.legend(loc='upper right', handlelength=1.6)
    fig.text(0, 1, 'e', fontsize=8, fontweight='bold', va='top')
    save_fig(fig, os.path.join(out, 'panel_e_sbr_vs_depth')); plt.close(fig)


def detections_overlay(rows, H, W, D, px_um, out):
    """Transparent overlay (PNG at native pixel size + SVG) marking every accepted FWHM profile."""
    import matplotlib.pyplot as plt
    if not rows:
        return None
    cmap = plt.colormaps.get_cmap('turbo')
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    xs = np.array([r['x_px'] for r in rows]); ys = np.array([r['y_px'] for r in rows]); zs = np.array([r['plane'] for r in rows])
    tx = np.array([r['tangent_x'] for r in rows]); ty = np.array([r['tangent_y'] for r in rows]); wpx = np.array([r['fwhm_um'] for r in rows]) / px_um
    cols = cmap(zs / max(D - 1, 1))
    for i in range(len(rows)):                                   # measured width as a perpendicular tick
        px, py = -ty[i], tx[i]
        ax.plot([xs[i] - px * wpx[i] / 2, xs[i] + px * wpx[i] / 2], [ys[i] - py * wpx[i] / 2, ys[i] + py * wpx[i] / 2], color=cols[i], lw=0.6, alpha=0.9)
    ax.scatter(xs, ys, s=3, c=cols, linewidths=0)
    fig.savefig(os.path.join(out, 'panel_d_detections_overlay.png'), dpi=100, transparent=True)
    fig.savefig(os.path.join(out, 'panel_d_detections_overlay.svg'), transparent=True); plt.close(fig)
    return 'panel_d_detections_overlay.png'


def write_csv(path, rows, cols=None):
    if not rows:
        open(path, 'w').write(''); return
    cols = cols or list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(('' if (isinstance(r[c], float) and not np.isfinite(r[c])) else str(r[c])) for c in cols) + '\n')


# ════════════════════════════════════════════════════════════════════════════
# 7. PPTX (python-pptx): pictures + native scale bars/labels, native XY charts, parameter table
# ════════════════════════════════════════════════════════════════════════════
def write_pptx(out, stem, ctx, log):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.chart.data import XyChartData
        from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
        from pptx.enum.shapes import MSO_CONNECTOR
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        from pptx.dml.color import RGBColor
    except ImportError:
        log("[pptx] python-pptx not installed (pip install python-pptx) -> skipped"); return None
    from PIL import Image
    px_um = ctx['px_um']; sb_um = ctx['scalebar_um']
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def text(slide, x, y, w, h, s, size=11, bold=False, color=None):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; r = p.add_run(); r.text = s; r.font.name = 'Arial'; r.font.size = Pt(size); r.font.bold = bold
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        return tb

    def picture(slide, png, x, y, w, title=None, scalebar=True, letter=None, px=None):
        path = os.path.join(out, png); px = px or px_um
        im = Image.open(path); Wp, Hp = im.size; h = w * Hp / Wp
        slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
        if scalebar:
            L = (sb_um / px) / Wp * w
            bx, by = x + 0.05 * w, y + 0.93 * h
            ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(bx), Inches(by), Inches(bx + L), Inches(by))
            ln.line.color.rgb = RGBColor(255, 255, 255); ln.line.width = Pt(3)
            text(slide, bx, by - 0.22, 1.2, 0.2, f'{sb_um:g} µm', size=9, color='FFFFFF')
        if title:
            text(slide, x, y - 0.25, w, 0.22, title, size=9)
        if letter:
            text(slide, x - 0.25, y - 0.3, 0.3, 0.3, letter, size=12, bold=True)
        return h

    # slide 1: MIPs
    s = prs.slides.add_slide(blank)
    text(s, 0.4, 0.15, 12, 0.35, f'{stem} — MIP / depth-coded MIP / depth-binned MIPs (native scale bars & labels)', size=13, bold=True)
    pb = ctx['images'].get('panel_b', ctx['images']['mip'])
    picture(s, pb, 0.5, 1.0, 3.2, ctx.get('panel_b_label', 'Maximum-intensity projection'), letter='b', px=(ctx['px_mip_um'] if pb == ctx['images']['mip'] else px_um))
    h = picture(s, ctx['images']['depth_coded'], 4.1, 1.0, 3.2, 'Depth-coded MIP', letter="b′", px=ctx['px_mip_um'])
    s.shapes.add_picture(os.path.join(out, 'img_colorbar_turbo.png'), Inches(7.45), Inches(1.0), height=Inches(h))
    text(s, 7.8, 1.0, 0.8, 0.2, f'{ctx["z_um"][-1]:.0f} µm', size=9); text(s, 7.8, 1.0 + h - 0.2, 0.8, 0.2, f'{ctx["z_um"][0]:.0f} µm', size=9)
    text(s, 7.8, 1.0 + h / 2 - 0.1, 0.8, 0.2, 'z', size=9)
    xb = 0.5
    for b in ctx.get('mip_bins', []):
        picture(s, b['png'], xb, 4.6, 2.3, f"{b['lo_um']:.0f} to {b['hi_um']:.0f} µm ({b['n_planes']} planes)", scalebar=(xb == 0.5), px=ctx['px_mip_um'])
        xb += 2.55
    md = ctx['mipd']
    text(s, 0.5, 7.0, 12, 0.3, f"MIP display (display only): bg {md['bg_method']} R={md['bg_radius_um']:g} µm, z-range {md['z_range'] or 'all'}, min {md['lo']:.3f} / max {md['hi']:.3f} "
                              f"(display units after clip {ctx['display_clip'][0]:.3g}–{ctx['display_clip'][1]:.3g}), gamma {md['gamma']}, 2-D Gaussian {md['smooth_px']} px, {md['upsample']}x bicubic; bin normalisation: {ctx['mip_bin_norm']}", size=8, color='666666')
    # slide 2: slices + xz + detections
    s = prs.slides.add_slide(blank)
    text(s, 0.4, 0.15, 12, 0.35, 'Depth slices, x–z section, and orthogonal views (x–y / x–z / y–z)', size=13, bold=True)
    xb = 0.5
    for k, sl in enumerate(ctx['slices']):
        picture(s, sl['png'], xb, 1.0, 2.6, f"z = {sl['z_um']:.0f} µm", scalebar=(k == 0), letter='c' if k == 0 else None); xb += 2.85
    picture(s, 'img_xz_section.png', 0.5, 4.5, 4.0, f"x–z section (thin-slab MIP, {ctx['xz_slab_um']:.0f} µm), y = {ctx['xz_row_px'] * px_um:.0f} µm; z {ctx['z_um'][0]:.0f} to {ctx['z_um'][-1]:.0f} µm", scalebar=False)
    sd = ctx.get('slice_display') or {}
    text(s, 0.5, 7.15, 12, 0.3, f"slice display (display only): norm {sd.get('norm')}, bg {sd.get('bg')}, gamma {sd.get('gamma')}, levels {sd.get('slice_min')}/{sd.get('slice_max')}; "
                              f"x–z levels {sd.get('xz_min')}/{sd.get('xz_max')} = {sd.get('xz_lo', 0):.3f}/{sd.get('xz_hi', 1):.3f}", size=8, color='666666')
    o = ctx.get('ortho')
    if o:
        wxy = 2.6; hxy = wxy * o['H'] / o['W']; px_um_xy = px_um
        sxz = wxy * o['span_um'] / (o['W'] * px_um); syz = hxy * o['span_um'] / (o['H'] * px_um)
        X, Y = 5.0, 4.35
        picture(s, 'img_xy_ortho.png', X, Y, wxy, o.get('xy_label', 'x–y'), scalebar=False, letter="c′", px=(ctx['px_mip_um'] if o.get('xy') == 'mip' else px_um))
        s.shapes.add_picture(os.path.join(out, 'img_xz_section.png'), Inches(X), Inches(Y + hxy + 0.05), width=Inches(wxy), height=Inches(sxz))
        s.shapes.add_picture(os.path.join(out, 'img_yz_section.png'), Inches(X + wxy + 0.05), Inches(Y), width=Inches(syz), height=Inches(hxy))
        yl = Y + (o['xz_row_px'] + 0.5) / o['H'] * hxy; xl = X + (o['yz_col_px'] + 0.5) / o['W'] * wxy
        for (x1, y1, x2, y2, col) in ((X, yl, X + wxy, yl, 'F0E442'), (xl, Y, xl, Y + hxy, '56B4E9'),
                                      (xl, Y + hxy + 0.05, xl, Y + hxy + 0.05 + sxz, '56B4E9'), (X + wxy + 0.05, yl, X + wxy + 0.05 + syz, yl, 'F0E442')):
            ln = slide_line = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            ln.line.color.rgb = RGBColor.from_string(col); ln.line.width = Pt(1); ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        def arrow(x1, y1, x2, y2, col='FFFFFF'):
            ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            ln.line.color.rgb = RGBColor.from_string(col); ln.line.width = Pt(1.25)
            from pptx.oxml.ns import qn
            from lxml import etree
            lnx = ln.line._get_or_add_ln(); te = lnx.find(qn('a:tailEnd'))
            if te is None:
                te = etree.SubElement(lnx, qn('a:tailEnd'))
            te.set('type', 'triangle'); te.set('w', 'med'); te.set('len', 'med')
            return ln

        def axes_ind(px_, py_, size, hl, vl):
            """L-shaped native arrows at (px_, py_) [top-left corner of the L, inches]: right = hl, down = vl."""
            arrow(px_, py_, px_ + size, py_); arrow(px_, py_, px_, py_ + size)
            text(s, px_ + size + 0.03, py_ - 0.09, 0.3, 0.18, hl, size=8, bold=True, color='FFFFFF')
            text(s, px_ - 0.06, py_ + size + 0.01, 0.3, 0.18, vl, size=8, bold=True, color='FFFFFF')
        sz = min(0.35, 0.18 * wxy)
        axes_ind(X + 0.08, Y + 0.08, sz, 'x', 'y')                           # x–y
        axes_ind(X + 0.08, Y + hxy + 0.05 + 0.06, min(sz, 0.5 * sxz), 'x', 'z')   # x–z (z down)
        axes_ind(X + wxy + 0.05 + 0.06, Y + 0.08, min(sz, 0.5 * syz), 'z', 'y')    # y–z (z right, y down)
        text(s, X, Y + hxy + 0.05 + sxz + 0.02, wxy + syz + 0.1, 0.4,
             f"x–z at y = {(o['xz_row_px'] + 0.5) * px_um:.0f} µm (yellow), y–z at x = {(o['yz_col_px'] + 0.5) * px_um:.0f} µm (blue); z span {o['span_um']:.0f} µm, physical aspect", size=8)
    # slide 3: native charts
    s = prs.slides.add_slide(blank)
    text(s, 0.4, 0.15, 12, 0.35, 'Native charts (editable): apparent vessel FWHM vs z (d) and SBR vs z (e)', size=13, bold=True)

    def xy_chart(slide, x, y, w, h, series, xtitle, ytitle, title):
        cd = XyChartData()
        for name, xs, ys, _, _ in series:
            ser = cd.add_series(name)
            for xv, yv in zip(xs, ys):
                if np.isfinite(xv) and np.isfinite(yv):
                    ser.add_data_point(float(xv), float(yv))
        gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, Inches(x), Inches(y), Inches(w), Inches(h), cd); ch = gf.chart
        ch.has_title = True; ch.chart_title.text_frame.text = title; ch.font.name = 'Arial'; ch.font.size = Pt(10)
        ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
        for ax_, ttl in ((ch.category_axis, xtitle), (ch.value_axis, ytitle)):
            ax_.has_major_gridlines = False; ax_.has_title = True; ax_.axis_title.text_frame.text = ttl
        ch.value_axis.minimum_scale = 0
        for ser, (_, _, _, color, dashed) in zip(ch.plots[0].series, series):
            ser.smooth = False; ser.format.line.color.rgb = RGBColor.from_string(color); ser.format.line.width = Pt(1.5)
            ser.marker.style = XL_MARKER_STYLE.NONE if dashed else XL_MARKER_STYLE.CIRCLE; ser.marker.size = 5
            if not dashed:
                ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
                ser.marker.format.line.color.rgb = RGBColor.from_string(color)
            else:
                ser.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        return ch
    dser = []
    for lab, summ, col in ((ctx['label_main'], ctx['summ'], '0072B2'), (ctx.get('label_ref'), ctx.get('summ_ref'), 'E69F00')):
        if summ is None:
            continue
        z = [r['z_um'] for r in summ if r['n'] > 0]
        dser.append((f'{lab} median', z, [r['median'] for r in summ if r['n'] > 0], col, False))
        dser.append((f'{lab} 10th pct', z, [r['p10'] for r in summ if r['n'] > 0], col, True))
    xy_chart(s, 0.5, 0.8, 6.0, 5.6, dser, 'Axial position z (µm)', 'Apparent vessel FWHM (µm)', 'd  FWHM vs z')
    eser = []
    for lab, rows, col in ((ctx['label_main'], ctx['sbr'], '0072B2'), (ctx.get('label_ref'), ctx.get('sbr_ref'), 'E69F00')):
        if rows is None:
            continue
        live = [r for r in rows if not r['trimmed']]
        eser.append((f'{lab} SBR', [r['z_um'] for r in live], [r['sbr'] for r in live], col, False))
    xy_chart(s, 6.9, 0.8, 6.0, 5.6, eser, 'Axial position z (µm)', 'SBR', 'e  SBR vs z')
    # slide 4: parameters
    s = prs.slides.add_slide(blank)
    text(s, 0.4, 0.15, 12, 0.35, 'Parameters and caption numbers', size=13, bold=True)
    rowsT = [(k, v) for k, v in ctx['table']]
    tbl = s.shapes.add_table(len(rowsT) + 1, 2, Inches(0.5), Inches(0.7), Inches(12.3), Inches(0.3 * (len(rowsT) + 1))).table
    tbl.columns[0].width = Inches(3.8); tbl.columns[1].width = Inches(8.5)
    for j, hdr in enumerate(('parameter', 'value')):
        tbl.cell(0, j).text = hdr
    for i, (k, v) in enumerate(rowsT, start=1):
        tbl.cell(i, 0).text = str(k); tbl.cell(i, 1).text = str(v)
    for r_ in tbl.rows:
        for c in r_.cells:
            for p in c.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9); run.font.name = 'Arial'
    path = os.path.join(out, f'{stem}_suppfig.pptx'); prs.save(path)
    log(f"[pptx] {path}"); return path


# ════════════════════════════════════════════════════════════════════════════
# 8. Video
# ════════════════════════════════════════════════════════════════════════════
def _font(size):
    from PIL import ImageFont
    for f in ('arial.ttf', 'Arial.ttf', 'C:/Windows/Fonts/arial.ttf', '/Library/Fonts/Arial.ttf',
              '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose_frame(img_rgb01, title, subtitle, px_um_eff, scalebar_um, canvas=(1920, 1080), colorbar=None, cb_labels=None):
    from PIL import Image, ImageDraw
    import matplotlib
    Wc, Hc = canvas; h, w = img_rgb01.shape[:2]
    s = min((Hc - 160) / h, (Wc - 520) / w); nw, nh = int(round(w * s)), int(round(h * s))
    im = Image.fromarray((np.clip(img_rgb01, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.BICUBIC)
    frame = Image.new('RGB', (Wc, Hc), (0, 0, 0)); x0 = 120; y0 = (Hc - nh) // 2; frame.paste(im, (x0, y0))
    d = ImageDraw.Draw(frame)
    d.text((40, 30), title, fill=(255, 255, 255), font=_font(40)); d.text((40, 84), subtitle, fill=(220, 220, 220), font=_font(32))
    Lpx = int(round(scalebar_um / px_um_eff * s)); bx, by = x0 + 30, y0 + nh - 40
    d.rectangle([bx, by, bx + Lpx, by + 8], fill=(255, 255, 255)); d.text((bx, by - 40), f'{scalebar_um:g} µm', fill=(255, 255, 255), font=_font(30))
    if colorbar is not None:
        cbx = x0 + nw + 60; cby0 = y0 + 40; cbh = nh - 80
        grad = matplotlib.colormaps['turbo'](np.linspace(1, 0, cbh))[:, :3]
        frame.paste(Image.fromarray((np.repeat(grad[:, None, :], 30, axis=1) * 255).astype(np.uint8)), (cbx, cby0))
        lo, hi = cb_labels
        d.text((cbx + 40, cby0 - 12), f'{hi:g} µm', fill=(255, 255, 255), font=_font(28)); d.text((cbx + 40, cby0 + cbh - 20), f'{lo:g} µm', fill=(255, 255, 255), font=_font(28))
        d.text((cbx - 6, cby0 + cbh + 34), 'z', fill=(255, 255, 255), font=_font(28))
    return np.asarray(frame)


def make_video(disp, z_um, px_um, dz_um, out_path, title, gamma, soft_mask, fps, fly_substeps, hold_s, rot_frames,
               rot_iso_um, scalebar_um, canvas, rgb_mip, rot_floor, log, mipd=None, disp_mip=None):
    mipd = mipd or dict(lo=0.0, hi=1.0, gamma=gamma, smooth_px=0.0, upsample=1)
    disp_mip = disp if disp_mip is None else disp_mip
    import imageio
    H, W, D = disp.shape; smask = soft_mask
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', quality=9, macro_block_size=None)
    n = 0
    try:
        log("  video: fly-through"); nfly = (D - 1) * fly_substeps + 1; hold = int(hold_s * fps)
        for k in range(nfly):
            zf = k / fly_substeps; z0 = int(np.floor(zf)); z1 = min(z0 + 1, D - 1); a = zf - z0
            sl = (1 - a) * disp[:, :, z0] + a * disp[:, :, z1]; g = np.clip(sl, 0, 1) ** gamma * smask
            fr = compose_frame(np.repeat(g[:, :, None], 3, 2), title, f'Depth fly-through   z = {z_um[0] + zf * dz_um:.0f} µm', px_um, scalebar_um, canvas)
            for _ in range(hold if k in (0, nfly - 1) else 1):
                writer.append_data(fr); n += 1
        log("  video: MIP holds")
        g_mip = apply_levels(mip_and_depth(disp_mip, mipd['smooth_px'], **proj_kw(mipd, px_um))[0], mipd['lo'], mipd['hi'], mipd['gamma']) * smask
        fr = compose_frame(np.repeat(g_mip[:, :, None], 3, 2), title, f'Maximum-intensity projection, z {z_um[0]:.0f} to {z_um[-1]:.0f} µm', px_um, scalebar_um, canvas)
        for _ in range(int(hold_s * fps)):
            writer.append_data(fr); n += 1
        fr = compose_frame(rgb_mip, title, f'Depth-coded maximum-intensity projection, z {z_um[0]:.0f} to {z_um[-1]:.0f} µm', px_um / max(int(mipd['upsample']), 1), scalebar_um, canvas, colorbar=True, cb_labels=(z_um[0], z_um[-1]))
        for _ in range(int(2 * hold_s * fps)):
            writer.append_data(fr); n += 1
        log(f"  video: turntable ({rot_frames} frames, {rot_iso_um} µm isotropic voxels)")
        import matplotlib.pyplot as plt
        cmap = plt.colormaps.get_cmap('turbo'); f = (px_um / rot_iso_um, px_um / rot_iso_um, dz_um / rot_iso_um)
        vol_src = disp_mip if mipd['smooth_px'] <= 0 else ndi.gaussian_filter(disp_mip, sigma=(mipd['smooth_px'], mipd['smooth_px'], 0))
        vol_src = apply_levels(vol_src, mipd['lo'], mipd['hi'], 1.0) * smask[:, :, None]; vol_src[vol_src < rot_floor] = 0.0
        vol_iso = ndi.zoom(vol_src, f, order=1); Hi, Wi, Di = vol_iso.shape
        zidx = np.broadcast_to((np.arange(Di) / max(f[2], 1e-9)).astype(np.float32), vol_iso.shape).copy()
        S = int(np.ceil(np.hypot(Wi, Di))) + 4
        pw = ((0, 0), ((S - Wi) // 2, S - Wi - (S - Wi) // 2), ((S - Di) // 2, S - Di - (S - Di) // 2))
        vol_pad = np.pad(vol_iso, pw); z_pad = np.pad(zidx, pw, constant_values=0); t0 = time.time()
        for k in range(rot_frames):
            th = 360.0 * k / rot_frames
            rot = ndi.rotate(vol_pad, th, axes=(1, 2), reshape=False, order=1, prefilter=False)
            zr = ndi.rotate(z_pad, th, axes=(1, 2), reshape=False, order=0, prefilter=False)
            proj = rot.max(axis=2); arg = rot.argmax(axis=2)
            dep = np.take_along_axis(zr, arg[:, :, None], axis=2)[:, :, 0]
            ck = colour_kw(mipd)
            cb = colour_brightness(np.clip(proj, 0, 1), ck['bright_mode'], ck['bright_gamma'] or mipd['gamma'], ck['bright_floor'], ck['bright_thr'], ck['bright_local_px'])
            rgb = cmap(np.clip(dep / max(D - 1, 1), 0, 1))[:, :, :3] * cb[:, :, None]
            fr = compose_frame(rgb, title, f'Depth-coded volume, rotation {th:5.1f}°', rot_iso_um, scalebar_um, canvas, colorbar=True, cb_labels=(z_um[0], z_um[-1]))
            writer.append_data(fr); n += 1
            if k % 30 == 0:
                log(f"    {k}/{rot_frames}  ({time.time() - t0:.0f} s)")
    finally:
        writer.close()
    log(f"  video written: {out_path}  ({n} frames, {n / fps:.1f} s)")


# ════════════════════════════════════════════════════════════════════════════
# 9. Main
# ════════════════════════════════════════════════════════════════════════════
def run_quant(raw, mask, px_um, z_um, args, log, tag):
    log(f"[{tag}] FWHM vs z")
    rows = measure_fwhm_vs_depth(raw, mask, px_um, z_um, profile_len_um=args.profile_len_um, min_contrast=args.min_contrast,
                                 max_fwhm_um=args.max_fwhm_um, vessels_json=(args.vessels if tag == 'main' else None), log=log,
                                 sigmas_um=tuple(args.sigmas_um), vessel_pct=args.vessel_pct)
    summ = summarize_fwhm(rows, z_um)
    log(f"[{tag}] SBR/CNR vs z")
    return rows, summ, sbr_cnr_vs_depth(raw, mask, z_um)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recon', default=None, help='record_*_rl.mat (v5 or v7.3), .tif, or .npy')
    ap.add_argument('--var', default=None); ap.add_argument('--axes', default=None, help="MATLAB-order override, e.g. HWDT / DHW")
    ap.add_argument('--t', default='mid', help="frame to use: 'mid' (default) or an index")
    ap.add_argument('--raw', default=None, help='matching record_*.raw (frame count / fps evidence)')
    ap.add_argument('--raw-shape', default='640,512', help='W,H of raw frames'); ap.add_argument('--raw-dtype', default='int16')
    ap.add_argument('--fps', type=float, default=None, help='acquisition frame rate (Hz) if not stored anywhere')
    ap.add_argument('--sidecar', default=None, help='explicit metadata/log file to search for the frame rate')
    ap.add_argument('--px-um', default='auto', help="lateral µm/px of the recon grid: 'auto' = .mat pixel_size if present, else 4.0")
    ap.add_argument('--z-range-um', type=float, default=600.0, help='FALLBACK only: total axial span assumed for the D planes when the .mat has no step variable (dz = span/(D-1))')
    ap.add_argument('--dz-um', default='auto', help="plane spacing (µm): 'auto' = .mat step variable (depth_size / dz / z_step ...) if present, else --z-range-um/(D-1); or an explicit number")
    ap.add_argument('--surface-index', default='center', help="plane index with z = 0 ('center' = (D-1)/2)")
    ap.add_argument('--mask', default='none', help="fallback FOV mask when no --mask-file: 'none' | 'auto' | 'detect' | 'cx,cy,r'")
    ap.add_argument('--mask-file', default=None, help='binary mask drawn on the representative frame (png/tif/npy)')
    ap.add_argument('--draw-mask', action='store_true', help='open the interactive polygon/ellipse tool on the representative MIP')
    ap.add_argument('--paint-mask', action='store_true', help='brush editor for the FOV mask (starts from --mask-file if given); saves mask.png')
    ap.add_argument('--gui', action='store_true', help='Studio window: draw/refine the mask, set MIP display parameters by eye, then Export runs the unchanged pipeline')
    ap.add_argument('--brush-px', type=int, default=6)
    ap.add_argument('--export-frame', action='store_true', help='write rep_frame_mip_16bit.tif (+png) for drawing a mask in Fiji, then exit')
    ap.add_argument('--pct', default='0.5,99.9'); ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--bg-pct', type=float, default=None, help='DISPLAY-only per-plane background percentile subtraction')
    ap.add_argument('--display-floor', type=float, default=0.0); ap.add_argument('--rot-floor', type=float, default=0.03)
    ap.add_argument('--depth-bright', choices=['intensity', 'floor', 'local', 'flat'], default=None,
                    help="depth-coded MIP colour brightness: 'intensity' = I^gamma (default), 'floor' = smooth lift of dim vessels, 'local' = each vessel normalised to its neighbourhood (dim and bright vessels equally visible), 'flat' = uniform colour")
    ap.add_argument('--depth-floor', type=float, default=None, help="'floor' mode: brightness reached by every vessel pixel well above the black level (default 0.5)")
    ap.add_argument('--depth-local-um', type=float, default=None, help="'local' mode: neighbourhood size in µm (default ~100 µm)")
    ap.add_argument('--depth-gamma', type=float, default=None, help='depth-coded MIP: gamma of the colour brightness (default = --mip-gamma)')
    ap.add_argument('--depth-floor-thr', type=float, default=None, help='depth-coded MIP: normalised level below which pixels fade to black (smoothstep; default 0.1) — keeps speckle dark in floor/local/flat modes')
    ap.add_argument('--depth-smooth-r', type=int, default=3, help='depth-map smoothing radius (px) of the depth-coded MIP (Movie Viewer default 3; 5-7 = smoother colours)')
    # MIP display (black/white levels, smoothing, upsampling) — display only, shared by all MIP-type images and the video
    ap.add_argument('--mip-lo', '--mip-min', dest='mip_lo', default=None, help="MIN / black level: float in display units (0-1) or 'pNN' percentile of the MIP in the mask (default p0 = 0)")
    ap.add_argument('--mip-hi', '--mip-max', dest='mip_hi', default=None, help="MAX / white level: float or 'pNN' (default p99.9)")
    ap.add_argument('--mip-bg-method', choices=['none', 'gauss', 'tophat'], default=None, help='per-plane local background removal before the MIP (display only; default none)')
    ap.add_argument('--mip-bg-radius-um', type=float, default=None, help='radius/sigma for --mip-bg-method in µm (default 100)')
    ap.add_argument('--z-weight', action='append', default=None, help="weight a z range before the projection (display only), e.g. --z-weight=-300:-100:1.5 (repeatable)")
    ap.add_argument('--mip-proj', choices=['max', 'focus'], default=None, help="projection: 'max' (classic MIP, default) or 'focus' (per-pixel in-focus plane; suppresses defocus ghosts)")
    ap.add_argument('--focus-win-um', type=float, default=None, help='window (µm) over which the focus measure is averaged for --mip-proj focus (default 40)')
    ap.add_argument('--mip-z-range', default=None, help="restrict MIP-type images: 'auto' = drop the boundary planes flagged by the SBR 1.5x jump rule, |z| <= value (µm, e.g. 240), or an explicit pair via '--mip-z-range=-240,200'")
    ap.add_argument('--mip-gamma', type=float, default=None, help='gamma for MIP-type images (default = --gamma)')
    ap.add_argument('--mip-smooth-px', type=float, default=None, help='per-plane 2-D Gaussian sigma (px) before MIP; try 0.7-1.5 (default 0)')
    ap.add_argument('--mip-upsample', type=int, default=None, help='bicubic upsampling factor for exported MIP images (default 2)')
    ap.add_argument('--tune-mip', action='store_true', help='open a slider window to set black/white/gamma/smoothing by eye (saved to mip_display.json)')
    ap.add_argument('--mip-sheet', action='store_true', help='write mip_contact_sheet.png (grid of level/gamma choices) for headless checking')
    ap.add_argument('--mip-json', default=None, help='load MIP display settings from a mip_display.json (default: <out>/mip_display.json if present)')
    ap.add_argument('--sweep', action='store_true', help='trial grid: gamma x background x black-level percentile -> <out>/sweep/ (PNG per combo + contact sheets + sweep_index.csv), then exit')
    ap.add_argument('--sweep-gamma', default='0.7:1.5:0.1', help="gamma values 'a:b:step' or comma list (default 0.7:1.5:0.1)")
    ap.add_argument('--sweep-min', default='40:70:5', help="black-level percentiles 'a:b:step' or comma list (default 40:70:5)")
    ap.add_argument('--sweep-bg', default='none,tophat', help="backgrounds to try, comma list of none/tophat/gauss (default none,tophat)")
    ap.add_argument('--sweep-kind', choices=['gray', 'depth', 'both'], default='gray', help='which image to render per combination')
    ap.add_argument('--sweep-nolabel', action='store_true', help='do not burn the parameter label into the sweep PNGs')
    ap.add_argument('--slices', default='auto', help="panel c z positions (µm), e.g. '-150,0,150'; 'auto' = quartiles")
    ap.add_argument('--slice-norm', choices=['per-plane', 'global', 'manual'], default='per-plane',
                    help="panel c slices: 'per-plane' = --slice-min/--slice-max percentiles of each plane (default), 'global' = percentiles over all planes (same levels for every slice), 'manual' = plane_lo/plane_hi from mip_display.json (Studio 'plane min/max')")
    ap.add_argument('--slice-min', default='p0', help="slice black level: float (0-1) or 'pNN' (default p0)")
    ap.add_argument('--slice-max', default='p99.9', help="slice white level: float or 'pNN' (default p99.9)")
    ap.add_argument('--slice-gamma', type=float, default=None, help='gamma for slices and x–z (default = --gamma)')
    ap.add_argument('--slice-bg', choices=['none', 'same'], default='none', help="'same' = apply the MIP's background removal (+smoothing) to slices and x–z (display only; state it in the caption)")
    ap.add_argument('--xz-min', default='p0', help="x–z black level: float or 'pNN' of the section inside the mask (default p0)")
    ap.add_argument('--xz-max', default='p99.9', help="x–z white level (default p99.9)")
    ap.add_argument('--xz-row', type=int, default=None); ap.add_argument('--xz-slab-px', type=int, default=3)
    ap.add_argument('--yz-col', type=int, default=None, help='column (px) for the y–z section')
    ap.add_argument('--ortho-center', choices=['mask', 'brightest'], default='mask', help="where the x–z / y–z cuts go when --xz-row/--yz-col are not given: 'mask' = centre of the FOV mask (default), 'brightest' = brightest row/column")
    ap.add_argument('--ortho-xy', choices=['center', 'mip'], default='center', help="x–y image of the orthogonal views: the centre plane (z ≈ 0, slice display settings; default) or the MIP")
    ap.add_argument('--panel-b', choices=['mip', 'center'], default='mip', help="panel b: grayscale MIP (default) or the centre plane (z ≈ 0)")
    ap.add_argument('--mip-bins', default='auto', help="bin edges in µm, e.g. '-300,-100,100,300'; 'auto' = 3 equal bins; 'none'")
    ap.add_argument('--mip-bin-norm', choices=['global', 'per-bin'], default='global')
    ap.add_argument('--scalebar-um', type=float, default=200.0); ap.add_argument('--lat-res-um', type=float, default=None)
    ap.add_argument('--vessels', default=None); ap.add_argument('--sigmas-um', type=float, nargs='+', default=[4.0, 8.0, 12.0])
    ap.add_argument('--vessel-pct', type=float, default=97.0); ap.add_argument('--profile-len-um', type=float, default=80.0)
    ap.add_argument('--min-contrast', type=float, default=0.3); ap.add_argument('--max-fwhm-um', type=float, default=150.0)
    ap.add_argument('--ref-recon', default=None); ap.add_argument('--ref-label', default='Ear'); ap.add_argument('--ref-px-um', type=float, default=None)
    ap.add_argument('--ref-z-range-um', type=float, default=None); ap.add_argument('--ref-dz-um', type=float, default=None)
    ap.add_argument('--ref-var', default=None); ap.add_argument('--ref-axes', default=None); ap.add_argument('--ref-t', default='mid')
    ap.add_argument('--ref-mask-file', default=None); ap.add_argument('--ref-mask', default='none'); ap.add_argument('--ref-sbr-csv', default=None)
    ap.add_argument('--label', default='Paw')
    ap.add_argument('--no-video', action='store_true'); ap.add_argument('--no-pptx', action='store_true')
    ap.add_argument('--video-title', default='NIR-II SLIM · mouse paw vasculature'); ap.add_argument('--video-fps', type=int, default=30)
    ap.add_argument('--rot-frames', type=int, default=240); ap.add_argument('--rot-iso-um', type=float, default=8.0)
    ap.add_argument('--fly-substeps', type=int, default=8); ap.add_argument('--canvas', default='1920,1080')
    ap.add_argument('--out', default=None, help='output folder (default ./<stem>_suppfig)')
    ap.add_argument('--dry-run', action='store_true')
    # ── batch orchestration (one command: sweep + automatic pick + your picks, each in its own folder)
    ap.add_argument('--batch', default=None, help='folder containing record_*_rl.mat (+ record_*.raw): run the whole workflow for every dataset')
    ap.add_argument('--batch-out', default=None, help='batch output root (default <batch folder>/suppfig_batch)')
    ap.add_argument('--only', default=None, help="comma list of ids to include, e.g. '174304,181541'")
    ap.add_argument('--mask-pattern', default=None, help="comma list of mask file patterns; placeholders {dir} {stem} {id} {out} (default: {out}/mask.png, {dir}/{stem}_suppfig/Mask.tif, {dir}/{stem}_suppfig/mask.png, {dir}/masks/{id}.png|.tif, {dir}/{id}_mask.png|.tif)")
    ap.add_argument('--picks-file', default=None, help="CSV with your choices: id,bg,min,gamma[,zweight]  e.g.  174304,tophat,p55,0.9  or  181541,tophat,p0,1.5,-300:-100:1.5")
    ap.add_argument('--no-prompt', action='store_true', help='do not ask for picks interactively after each sweep')
    ap.add_argument('--redo-sweep', action='store_true', help='re-run the sweep even if sweep_index.csv exists')
    ap.add_argument('--skip-auto', action='store_true', help='do not render the automatic pick')
    ap.add_argument('--gui-final', action='store_true', help='batch: open the Studio for each dataset (starting from its pick / auto setting) so you tune the depth-coded MIP by eye; EXPORT renders into <dataset>/final/. Picks are NOT re-rendered unless --render-picks')
    ap.add_argument('--render-picks', action='store_true', help='with --gui-final: also render the picks/auto folders before opening the Studio')
    ap.add_argument('--final-video', action='store_true', help='Studio: closing the window / EXPORT (no video) still renders the video (EXPORT + video always does)')
    return ap


def run_dataset(args):
    """Single dataset / single configuration (the original command-line behaviour)."""
    if not args.recon:
        raise SystemExit('--recon is required (or use --batch <folder>)')
    stem = os.path.splitext(os.path.basename(args.recon))[0]
    out = args.out or f'./{stem}_suppfig'; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'run_log.txt'), 'a')

    def log(msg):
        print(msg); logf.write(msg + '\n'); logf.flush()
    log('\n$ ' + ' '.join(sys.argv))

    raw, info = load_representative_volume(args.recon, args.var, args.axes, args.t, log)
    H, W, D = raw.shape
    raw_info = check_raw(args.raw, [int(v) for v in args.raw_shape.split(',')], args.raw_dtype, log=log) if args.raw else None
    fps, fps_src = resolve_fps(args, raw_info, info.get('params', {}), log)
    if raw_info and info.get('T', 1) > 1:
        log(f"[frames] .mat T = {info['T']} vs raw frames = {raw_info['n_frames']}" + ("" if raw_info['n_frames'] == info['T'] else "  (recon used a subset / different range)"))
    params = info.get('params', {})
    if str(args.px_um).lower() == 'auto':
        pk = next((k for k in PARAM_NAMES['px'] if k in params and isinstance(params[k], (int, float))), None)
        if pk:
            args.px_um = float(params[pk]); log(f"[px] {args.px_um:g} µm/px (.mat variable '{pk}')")
        else:
            args.px_um = 4.0; log("[px] 4.0 µm/px (default; .mat has no pixel_size — pass --px-um if wrong)")
    else:
        args.px_um = float(args.px_um)
    # ── plane spacing: --dz-um (explicit) > .mat step variable (recon app, authoritative) > --z-range-um/(D-1) (assumption)
    dz_mat = None; dz_key = None
    for k in PARAM_NAMES['dz']:
        if k in params and isinstance(params[k], (int, float)) and float(params[k]) > 0:
            v = float(params[k]); dz_key = k
            dz_mat = v * 1000.0 if v < 1.0 else v            # values < 1 are taken as mm
            break
    dz_assumed = args.z_range_um / max(D - 1, 1)
    if str(args.dz_um).lower() not in ('auto', 'none', ''):
        dz = float(args.dz_um); dz_src = '--dz-um'
    elif dz_mat is not None:
        dz = dz_mat; dz_src = f".mat '{dz_key}'"
    else:
        dz = dz_assumed; dz_src = f"assumed {args.z_range_um:g} µm span / (D-1)"
    if dz_mat is not None and abs(dz_mat - dz_assumed) > 0.01 * dz_mat:
        log(f"  !! .mat '{dz_key}' = {dz_mat:g} µm/plane, but {args.z_range_um:g}/(D-1) would give {dz_assumed:g} µm: "
            f"the {D} planes span {dz_mat * (D - 1):g} µm, not {args.z_range_um:g} µm. Using {dz:g} µm ({dz_src}).")
    for k in PARAM_NAMES['zrange']:
        if k in params:
            log(f"  .mat variable '{k}' = {params[k]}")
    sidx = (D - 1) / 2.0 if str(args.surface_index).lower() == 'center' else float(args.surface_index)
    z_um = z_axis(D, dz, sidx)
    log(f"[geom] {W}×{H} px × {D} planes; {args.px_um} µm/px -> FOV {W * args.px_um:.0f} × {H * args.px_um:.0f} µm; "
        f"dz = {dz:g} µm ({dz_src}); span {dz * (D - 1):g} µm; z from {z_um[0]:g} to {z_um[-1]:g} µm (0 at plane {sidx:g}); frame {info['t_used']}/{info['T']}"
        + (f"; fps {fps:g} Hz ({fps_src})" if fps else "; fps unknown"))
    if args.dry_run:
        log("[dry-run] done"); return

    # representative frame export / mask
    disp_pre, _ = display_volume(raw, 0.5, 99.9)
    rep_mip = disp_pre.max(axis=2)
    if args.export_frame:
        import tifffile
        tifffile.imwrite(os.path.join(out, 'rep_frame_mip_16bit.tif'), (np.clip(raw.max(axis=2), 0, None)).astype(np.float32))
        save_png8(np.clip(rep_mip, 0, 1) ** 0.5, os.path.join(out, 'rep_frame_mip_display.png'))
        log(f"[export] rep_frame_mip_16bit.tif / rep_frame_mip_display.png (frame {info['t_used']}) -> draw the mask in Fiji, then --mask-file"); return
    if args.gui:
        if not _interactive_backend(log):
            raise SystemExit("no GUI backend (tkinter/Qt) available for --gui: run this in your own terminal/Python, or use --export-frame -> Fiji -> --mask-file")
        import matplotlib.pyplot as plt
        pct_lo, pct_hi = [float(v) for v in args.pct.split(',')]
        init_m = load_mask_file(args.mask_file, H, W, log) if args.mask_file else build_mask(H, W, args.mask, raw, log)
        disp0, _ = display_volume(raw, pct_lo, pct_hi, args.bg_pct, init_m)
        _m0 = initial_mipd(args, out, log)
        if _m0.get('z_range') == 'auto':
            _m0['z_range'] = None
        _v0 = make_mip_volume(disp0, z_um, _m0['bg_method'], _m0['bg_radius_um'] / args.px_um, _m0.get('z_range'), _m0.get('z_weights'))
        _mip0, _ = mip_and_depth(_v0, _m0['smooth_px'], **proj_kw(_m0, args.px_um))
        if args.mip_lo is not None:
            _m0['lo'] = parse_level(args.mip_lo, _mip0, init_m)
        if _m0.get('hi') is None:
            _m0['hi'] = parse_level(_m0.get('hi_spec') or 'p99.9', _mip0, init_m)
        studio = SuppFigStudio(disp0, z_um, args.px_um, _m0, init_m, out, log, title=stem, default_video=bool(getattr(args, 'final_video', False)))
        res = studio.run()
        if not res or res.get('action') != 'export':
            log("[gui] closed without export (use Save mask / Save settings buttons to keep work; re-run with --gui to continue)"); return
        args.mask_file = os.path.join(out, 'mask.png'); args.mip_json = os.path.join(out, 'mip_display.json')
        for k in ('mip_lo', 'mip_hi', 'mip_gamma', 'mip_smooth_px', 'mip_upsample', 'mip_bg_method', 'mip_bg_radius_um', 'mip_z_range',
                  'depth_bright', 'depth_floor', 'depth_gamma', 'depth_floor_thr', 'depth_local_um', 'z_weight', 'mip_proj', 'focus_win_um'):
            setattr(args, k, None)
        args.tune_mip = args.mip_sheet = args.draw_mask = args.paint_mask = False; args.no_video = not res['video']
        try:
            plt.close('all'); plt.switch_backend('Agg')
        except Exception:
            pass
        log(f"[gui] export requested (video={res['video']}) -> running the standard pipeline with mask.png + mip_display.json")
    if args.paint_mask:
        init_m = load_mask_file(args.mask_file, H, W, log) if args.mask_file else None
        mask = paint_mask_interactive(rep_mip, init_m, out, log, args.brush_px)
    elif args.draw_mask:
        mask = draw_mask_interactive(rep_mip, out, log)
    elif args.mask_file:
        mask = load_mask_file(args.mask_file, H, W, log)
    else:
        mask = build_mask(H, W, args.mask, raw, log); log(f"[mask] spec '{args.mask}' ({mask.sum()} px)")
    soft = soft_edge(mask)
    pct_lo, pct_hi = [float(v) for v in args.pct.split(',')]
    disp, (vmin, vmax) = display_volume(raw, pct_lo, pct_hi, args.bg_pct, mask)
    log(f"[display] clip [{vmin:g}, {vmax:g}], gamma {args.gamma}, bg_pct {args.bg_pct}")

    nature_style()
    # ── MIP display settings: CLI > --mip-json > <out>/mip_display.json > defaults; then optional sheet / slider tuning
    mipd = initial_mipd(args, out, log)
    if args.mip_sheet:
        mip_contact_sheet(disp, z_um, soft, mask, args.px_um, mipd, out, log)
    if mipd.get('z_range') == 'auto':
        _sbr = sbr_cnr_vs_depth(raw, mask, z_um)
        keep = [r for r in _sbr if not r['trimmed']]
        mipd['z_range'] = [float(keep[0]['z_um']), float(keep[-1]['z_um'])] if keep else None
        log(f"[mip] z-range auto: boundary planes trimmed by the SBR 1.5x jump rule -> planes {[r['plane'] for r in _sbr if r['trimmed']]} excluded, using z {mipd['z_range']}")
    disp_mip = make_mip_volume(disp, z_um, mipd['bg_method'], mipd['bg_radius_um'] / args.px_um, mipd['z_range'], mipd.get('z_weights'))
    mip_s, _ = mip_and_depth(disp_mip, mipd['smooth_px'], **proj_kw(mipd, args.px_um))
    log_mip_percentiles(mip_s, mask, log)
    if mipd.get('z_weights'):
        log(f"[mip] z-weights (DISPLAY ONLY, disclose in caption): " + ", ".join(f"z {lo:g} to {hi:g} µm x{f:g}" for lo, hi, f in mipd['z_weights']))
    if mipd.get('depth_bright', 'intensity') != 'intensity' or mipd.get('depth_gamma'):
        log(f"[mip] depth-coded colour brightness: mode {mipd.get('depth_bright')}, floor {mipd.get('depth_floor'):g}, fade-in below {mipd.get('depth_floor_thr', 0.1):g}, "
            f"gamma {mipd.get('depth_gamma') or mipd['gamma']:g}, local window {mipd.get('depth_local_px')} px")
    if args.mip_lo is not None:
        mipd['lo'] = parse_level(args.mip_lo, mip_s, mask)
    if args.mip_hi is not None or mipd['hi'] is None:
        mipd['hi'] = parse_level(mipd['hi_spec'], mip_s, mask)
    if args.tune_mip:
        mipd = dict(mipd, **tune_mip_interactive(disp, z_um, soft, mask, args.px_um, mipd, out, log))
        disp_mip = make_mip_volume(disp, z_um, mipd['bg_method'], mipd['bg_radius_um'] / args.px_um, mipd['z_range'], mipd.get('z_weights'))
        mip_s, _ = mip_and_depth(disp_mip, mipd['smooth_px'], **proj_kw(mipd, args.px_um))
    if mipd['hi'] <= mipd['lo']:
        mipd['hi'] = mipd['lo'] + 1e-3
    plo = float((mip_s[mask] <= mipd['lo']).mean() * 100); phi = float((mip_s[mask] <= mipd['hi']).mean() * 100)
    log(f"[mip] projection {mipd['proj_mode']}{'' if mipd['proj_mode'] == 'max' else f' (focus window {mipd['focus_win_um']:g} µm)'}, "
        f"bg {mipd['bg_method']} (R {mipd['bg_radius_um']:g} µm), z-range {mipd['z_range'] or 'all'}, min {mipd['lo']:.4f} (p{plo:.1f} of MIP in mask), "
        f"max {mipd['hi']:.4f} (p{phi:.1f}), gamma {mipd['gamma']}, 2-D Gaussian {mipd['smooth_px']} px ({mipd['smooth_px'] * args.px_um:.1f} µm), "
        f"upsample {mipd['upsample']}x  [display only — quantification uses the raw reconstruction]")
    json.dump(mipd, open(os.path.join(out, 'mip_display_used.json'), 'w'), indent=1)
    if args.sweep:
        mip_sweep(disp, z_um, soft, mask, args.px_um, mipd, out, log, _parse_range(args.sweep_gamma, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                  _parse_range(args.sweep_min, [40, 45, 50, 55, 60, 65, 70]), [b.strip() for b in args.sweep_bg.split(',')], args.sweep_kind, not args.sweep_nolabel)
        log("[sweep] done — no panels/video were produced in this run"); return
    rgb, proj_g, depth_norm = depth_coded_mip(disp_mip, gamma=mipd['gamma'], smooth_r=args.depth_smooth_r, soft_mask=soft, floor=args.display_floor,
                                              lo=mipd['lo'], hi=mipd['hi'], smooth_px=mipd['smooth_px'], upsample=mipd['upsample'], **proj_kw(mipd, args.px_um), **colour_kw(mipd))
    imgs = panel_images(disp, rgb, proj_g, depth_norm, z_um, args.px_um, dz, out, args, soft, log, mipd, disp_mip)
    rows, summ, sbr = run_quant(raw, mask, args.px_um, z_um, args, log, 'main')
    write_csv(os.path.join(out, 'sourcedata_panel_d_fwhm_measurements.csv'), rows)
    write_csv(os.path.join(out, 'sourcedata_panel_d_fwhm_summary.csv'), summ)
    write_csv(os.path.join(out, 'sourcedata_panel_e_sbr_cnr.csv'), sbr)
    overlay = detections_overlay(rows, H, W, D, args.px_um, out)
    summ_ref = sbr_ref = None
    if args.ref_recon:
        rraw, rinfo = load_representative_volume(args.ref_recon, args.ref_var, args.ref_axes, args.ref_t, log)
        rH, rW, rD = rraw.shape
        rpk = next((k for k in PARAM_NAMES['px'] if k in rinfo.get('params', {}) and isinstance(rinfo['params'][k], (int, float))), None)
        rpx = args.ref_px_um or (float(rinfo['params'][rpk]) if rpk else args.px_um); log(f"[ref px] {rpx:g} µm/px")
        rdz_mat = next((float(rinfo['params'][k]) for k in PARAM_NAMES['dz'] if k in rinfo.get('params', {}) and isinstance(rinfo['params'][k], (int, float)) and float(rinfo['params'][k]) > 0), None)
        rdz_mat = rdz_mat * 1000.0 if (rdz_mat is not None and rdz_mat < 1.0) else rdz_mat
        rdz = args.ref_dz_um or rdz_mat or (args.ref_z_range_um or args.z_range_um) / max(rD - 1, 1); log(f"[ref dz] {rdz:g} µm")
        rmask = load_mask_file(args.ref_mask_file, rH, rW, log) if args.ref_mask_file else build_mask(rH, rW, args.ref_mask, rraw, log)
        rz = z_axis(rD, rdz, (rD - 1) / 2.0)
        rrows, summ_ref, sbr_ref = run_quant(rraw, rmask, rpx, rz, args, log, 'ref')
        write_csv(os.path.join(out, f'sourcedata_panel_d_fwhm_measurements_{args.ref_label}.csv'), rrows)
        write_csv(os.path.join(out, f'sourcedata_panel_d_fwhm_summary_{args.ref_label}.csv'), summ_ref)
        write_csv(os.path.join(out, f'sourcedata_panel_e_sbr_cnr_{args.ref_label}.csv'), sbr_ref)
    panel_d(summ, out, args.label, summ_ref, args.ref_label, args.lat_res_um)
    panel_e(sbr, out, args.label, sbr_ref, args.ref_label, ref_csv=args.ref_sbr_csv)

    live = [r for r in sbr if not r['trimmed'] and np.isfinite(r['sbr'])]; acc = [r for r in summ if r['n'] > 0]
    meta = dict(source=info, raw=raw_info, fps_hz=fps, fps_source=fps_src, n_planes=int(D), dz_um=float(dz), dz_source=dz_src, z_span_um=float(dz * (D - 1)),
                z_um=[float(v) for v in z_um], z0_plane=sidx, px_um=args.px_um, fov_um=[round(W * args.px_um), round(H * args.px_um)],
                mask=(args.mask_file or ('drawn' if args.draw_mask else args.mask)), mask_px=int(mask.sum()),
                display=dict(clip=[vmin, vmax], gamma=args.gamma, bg_pct=args.bg_pct, floor=args.display_floor,
                             mip=dict(mipd, lo_percentile=plo, hi_percentile=phi, smooth_um=mipd['smooth_px'] * args.px_um)),
                fwhm=dict(n_measurements=len(rows), p10_um_first_last=[acc[0]['p10'], acc[-1]['p10']] if acc else None,
                          median_um_range=[min(r['median'] for r in acc), max(r['median'] for r in acc)] if acc else None,
                          detection=dict(sigmas_um=args.sigmas_um, vessel_pct=args.vessel_pct, profile_len_um=args.profile_len_um,
                                         min_contrast=args.min_contrast, max_fwhm_um=args.max_fwhm_um, focus_halfwin_planes=2)),
                sbr=dict(range=[min(r['sbr'] for r in live), max(r['sbr'] for r in live)] if live else None,
                         trimmed_planes=[r['plane'] for r in sbr if r['trimmed']]),
                panels=imgs)
    json.dump(meta, open(os.path.join(out, 'figure_metadata.json'), 'w'), indent=2, default=float)
    log("[caption numbers] " + json.dumps(dict(z_range=[float(z_um[0]), float(z_um[-1])], dz=float(dz), planes=int(D), frame=f"{info['t_used']}/{info['T']}",
                                             fps=fps, fov_um=meta['fov_um'], fwhm_n=len(rows), fwhm_p10_first_last=meta['fwhm']['p10_um_first_last'],
                                             fwhm_median_range=meta['fwhm']['median_um_range'], sbr_range=meta['sbr']['range'],
                                             sbr_trimmed=meta['sbr']['trimmed_planes']), default=float))
    bgtxt = ('' if mipd['bg_method'] == 'none' else f"each plane was background-subtracted ({'white top-hat' if mipd['bg_method'] == 'tophat' else 'Gaussian high-pass'}, radius {mipd['bg_radius_um']:g} µm), ")
    zr = f"planes within z = {mipd['z_range'][0]:g} to {mipd['z_range'][1]:g} µm" if mipd['z_range'] else "all planes"
    wtxt = ("" if not mipd.get('z_weights') else "planes at " + ", ".join(f"z = {lo:g} to {hi:g} µm were weighted x{f:g}" for lo, hi, f in mipd['z_weights']) + " before projection, ")
    projtxt = "maximum-intensity projected" if mipd['proj_mode'] == 'max' else f"projected by selecting, for each pixel, the plane of highest local focus (|LoG| averaged over {mipd['focus_win_um']:g} µm)"
    log("[caption template] For display only, " + bgtxt + wtxt + f"{projtxt} over {zr}, black/white levels set at the "
        f"{plo:.0f}th/{phi:.1f}th percentiles of the projection, gamma {mipd['gamma']:g}, Gaussian smoothing sigma = {mipd['smooth_px'] * args.px_um:.0f} µm, "
        f"{mipd['upsample']}x bicubic upsampling. Quantification (d, e) used the unprocessed reconstruction.")
    if not args.no_pptx:
        table = [('source file', os.path.basename(args.recon)), ('variable / format', f"{info['variable']} / {info['format']}"),
                 ('volume (H,W,D)', str(list(raw.shape))), ('frame used / T', f"{info['t_used']} / {info['T']}"),
                 ('raw file', os.path.basename(args.raw) if args.raw else '-'), ('raw frames', raw_info['n_frames'] if raw_info else '-'),
                 ('fps (Hz)', f"{fps:g} ({fps_src})" if fps else 'unknown'), ('px (µm)', args.px_um), ('FOV (µm)', f"{meta['fov_um'][0]} × {meta['fov_um'][1]}"),
                 ('z range / dz (µm)', f"{z_um[0]:g} to {z_um[-1]:g} / {dz:g} ({dz_src})"), ('z = 0 plane', sidx), ('mask', f"{meta['mask']} ({mask.sum()} px)"),
                 ('display clip / gamma / bg_pct', f"{vmin:.3g}–{vmax:.3g} / {args.gamma} / {args.bg_pct}"),
                 ('MIP display (display only)', f"bg {mipd['bg_method']} R={mipd['bg_radius_um']:g} µm, z-range {mipd['z_range'] or 'all'}, z-weights {mipd.get('z_weights') or 'none'}, min {mipd['lo']:.3f} (p{plo:.1f}), max {mipd['hi']:.3f} (p{phi:.1f}), gamma {mipd['gamma']}, smooth {mipd['smooth_px']} px, {mipd['upsample']}x bicubic"),
                 ('depth-coded colour brightness', f"mode {mipd.get('depth_bright')}, floor {mipd.get('depth_floor')}, fade-in {mipd.get('depth_floor_thr')}, gamma {mipd.get('depth_gamma') or mipd['gamma']}, local window {mipd.get('depth_local_px')} px"),
                 ('FWHM detection', json.dumps(meta['fwhm']['detection'])), ('FWHM n / p10 first,last (µm)', f"{len(rows)} / {meta['fwhm']['p10_um_first_last']}"),
                 ('SBR range / trimmed planes', f"{meta['sbr']['range']} / {meta['sbr']['trimmed_planes']}"), ('.mat small variables', json.dumps(info.get('params', {}))[:300])]
        ctx = dict(px_um=args.px_um, px_mip_um=imgs['px_mip_um'], mipd=mipd, scalebar_um=args.scalebar_um, z_um=z_um, gamma=args.gamma, display_clip=(vmin, vmax), mip_bin_norm=args.mip_bin_norm,
                   images=imgs['images'], panel_b_label=imgs.get('panel_b_label'), mip_bins=imgs['mip_bins'], slices=imgs['slices'], xz_row_px=imgs['xz_row_px'], xz_slab_um=imgs['xz_slab_um'], slice_display=imgs.get('slice_display'), ortho=imgs.get('ortho'),
                   overlay=overlay, label_main=args.label, summ=summ, sbr=sbr, label_ref=args.ref_label, summ_ref=summ_ref, sbr_ref=sbr_ref, table=table)
        write_pptx(out, stem, ctx, log)
    if not args.no_video:
        make_video(disp, z_um, args.px_um, dz, os.path.join(out, f'SuppVideo_{stem}.mp4'), args.video_title, args.gamma, soft, args.video_fps,
                   args.fly_substeps, 1.5, args.rot_frames, args.rot_iso_um, args.scalebar_um, tuple(int(v) for v in args.canvas.split(',')), rgb, args.rot_floor, log, mipd, disp_mip)
    log("[done] outputs in " + os.path.abspath(out)); logf.close()


# ════════════════════════════════════════════════════════════════════════════
# 10. Batch: sweep -> automatic pick -> your picks, every dataset, every setting in its own folder
# ════════════════════════════════════════════════════════════════════════════
def _ds_id(stem):
    m = re.search(r'record_(\d+_\d+)', stem)
    return m.group(1).split('_')[-1] if m else stem


def _find_mask(patterns, ddir, stem, did, out_ds):
    for pat in patterns:
        p = pat.format(dir=ddir, stem=stem, id=did, out=out_ds)
        if os.path.exists(p):
            return p
    return None


def _parse_picks(text):
    """'tophat p55 0.9; none p60 1.0 -300:-100:1.5' -> [dict(bg, min, gamma, zweight)]  (4th token = optional z-weight)"""
    picks = []
    for part in re.split(r'[;\n]+', text.strip()):
        tok = part.replace(',', ' ').split()
        if len(tok) < 3:
            continue
        bg, mn, g = tok[0].lower(), tok[1].lower(), tok[2]
        if bg not in ('none', 'tophat', 'gauss'):
            continue
        if not mn.startswith('p'):
            mn = 'p' + mn
        picks.append(dict(bg=bg, min=mn, gamma=float(g), zweight=(tok[3] if len(tok) > 3 and ':' in tok[3] else None)))
    return picks


def _tag(pk):
    t = f"{pk['bg']}_{pk['min']}_g{pk['gamma']:.1f}"
    if pk.get('zweight'):
        t += "_zw" + pk['zweight'].replace(':', '_').replace('-', 'm')
    return t


def run_batch(args):
    import copy, glob, shutil
    bdir = args.batch
    files = sorted(glob.glob(os.path.join(bdir, 'record_*_rl*.mat')))
    if args.only:
        keep = [v.strip() for v in args.only.split(',')]
        files = [f for f in files if any(k in os.path.basename(f) for k in keep)]
    if not files:
        raise SystemExit(f"no record_*_rl*.mat in {bdir}")
    broot = args.batch_out or os.path.join(bdir, 'suppfig_batch'); os.makedirs(broot, exist_ok=True)
    blog = open(os.path.join(broot, 'batch_log.txt'), 'a')

    def log(msg):
        print(msg); blog.write(msg + '\n'); blog.flush()
    log('\n$ ' + ' '.join(sys.argv) + f"\n[batch] {len(files)} dataset(s) -> {broot}")
    patterns = ([p.strip() for p in args.mask_pattern.split(',')] if args.mask_pattern else
                ['{out}/mask.png', '{dir}/{stem}_suppfig/Mask.tif', '{dir}/{stem}_suppfig/mask.png', '{dir}/masks/{id}.png', '{dir}/masks/{id}.tif',
                 '{dir}/{id}_mask.png', '{dir}/{id}_mask.tif', '{dir}/../{stem}_suppfig/Mask.tif', '{dir}/../{stem}_suppfig/mask.png'])
    picks_file = {}
    if args.picks_file:
        for line in open(args.picks_file):
            tok = [t.strip() for t in line.strip().split(',')]
            if len(tok) >= 4 and not line.startswith('#'):
                picks_file.setdefault(tok[0], []).extend(_parse_picks(' '.join(tok[1:5])))
    interactive = (not args.no_prompt) and sys.stdin.isatty()
    if args.gui_final:
        if not _interactive_backend(log):
            raise SystemExit("--gui-final needs a GUI backend (tkinter or Qt) and a desktop session. Run this in your own terminal, "
                             "check  python -c \"import tkinter; tkinter.Tk()\"  and  pip install matplotlib  for this interpreter.")
        log("[batch] GUI backend OK — the Studio window will open for each dataset (after its sweep) once the sweep is done")
    same_for_rest = None; summary = []
    for recon in files:
        stem = os.path.splitext(os.path.basename(recon))[0]; did = _ds_id(stem); ddir = os.path.dirname(recon) or '.'
        raw = os.path.join(ddir, stem.split('_rl')[0] + '.raw'); raw = raw if os.path.exists(raw) else None
        out_ds = os.path.join(broot, stem); os.makedirs(out_ds, exist_ok=True)
        mask_src = _find_mask(patterns, ddir, stem, did, out_ds)
        mask_png = os.path.join(out_ds, 'mask.png')
        if mask_src and os.path.abspath(mask_src) != os.path.abspath(mask_png):
            try:
                import tifffile
                from PIL import Image
                m = tifffile.imread(mask_src) if mask_src.lower().endswith(('.tif', '.tiff')) else np.asarray(Image.open(mask_src))
                m = m[..., 0] if m.ndim == 3 else m
                Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(mask_png)
            except Exception as e:
                log(f"[batch] {stem}: could not convert mask {mask_src} ({e}); using it directly"); mask_png = mask_src
        elif not mask_src:
            mask_png = None
        log(f"\n[batch] ===== {stem}  raw: {os.path.basename(raw) if raw else 'not found'}  mask: {mask_src or 'NONE (whole frame)'}")
        base = copy.deepcopy(args); base.batch = None; base.recon = recon; base.raw = raw; base.mask_file = mask_png
        base.gui = base.tune_mip = base.mip_sheet = base.draw_mask = base.paint_mask = base.export_frame = base.dry_run = False
        base.mip_json = os.path.join(out_ds, '__none__.json')          # never inherit a stray mip_display.json
        t0 = time.time()
        # 1) sweep (skipped if already done)
        idx = os.path.join(out_ds, 'sweep', 'sweep_index.csv')
        if args.redo_sweep or not os.path.exists(idx):
            a = copy.deepcopy(base); a.sweep = True; a.out = out_ds; a.no_video = True; a.no_pptx = True
            run_dataset(a)
        else:
            log(f"[batch] sweep exists -> {idx} (use --redo-sweep to redo)")
        rows = []
        for line in open(idx):
            if line.startswith('files'):
                hdr = line.strip().split(','); continue
            tok = line.rstrip('\n').split(',', len(hdr) - 1)
            if len(tok) == len(hdr):
                rows.append(dict(zip(hdr, tok)))
        best = None; scored = [r for r in rows if r.get('score') not in ('', None, 'nan')]
        if scored:
            best = max(scored, key=lambda r: float(r['score']))
        sheet = os.path.join(out_ds, 'mip_contact_sheet_sweep.png')
        log(f"[batch] sheet: {sheet}")
        todo = []
        if best and not args.skip_auto:
            todo.append(('auto', dict(bg=best['bg'], min=f"p{float(best['min_pct']):g}", gamma=float(best['gamma']))))
        for key in (stem, did, '*'):
            for pk in picks_file.get(key, []):
                todo.append(('pick', pk))
        if interactive:
            if same_for_rest:
                todo += [('pick', pk) for pk in same_for_rest]
            else:
                print(f"\n>>> {stem}: look at {sheet}\n    auto pick = {todo[0][1] if todo else 'none'}\n"
                      "    type your setting(s) as  bg min gamma   e.g.  tophat p55 0.9   (several: separate with ';')\n"
                      "    'same' = use the same picks for all remaining datasets, Enter = no extra pick")
                ans = input('    picks > ').strip()
                if ans.lower().startswith('same'):
                    same_for_rest = _parse_picks(ans[4:]) or [pk for kind, pk in todo if kind == 'pick']
                    todo += [('pick', pk) for pk in _parse_picks(ans[4:])]
                else:
                    todo += [('pick', pk) for pk in _parse_picks(ans)]
        # 2) render every setting into its own folder (with --gui-final the picks are only start points unless --render-picks)
        done = set(); rendered = []
        for kind, pk in ([] if (args.gui_final and not args.render_picks) else todo):
            tag = _tag(pk)
            if tag in done:
                continue
            done.add(tag)
            a = copy.deepcopy(base); a.sweep = False; a.out = os.path.join(out_ds, f"{kind}_{tag}")
            a.mip_bg_method = pk['bg']; a.mip_lo = pk['min']; a.mip_gamma = pk['gamma']
            a.z_weight = [pk['zweight']] if pk.get('zweight') else None
            a.mip_hi = a.mip_hi or 'p99.9'; a.mip_proj = a.mip_proj or 'max'
            log(f"[batch] render {kind}: {tag} -> {a.out}")
            try:
                run_dataset(a); rendered.append(f"{kind}:{tag}")
            except SystemExit as e:
                log(f"[batch] !! {stem} {tag} failed: {e}")
            except Exception as e:
                import traceback
                log(f"[batch] !! {stem} {tag} crashed: {e.__class__.__name__}: {e}\n" + traceback.format_exc())
        if args.gui_final:
            start = next((pk for kind, pk in todo if kind == 'pick'), None) or next((pk for kind, pk in todo), None) or dict(bg='tophat', min='p60', gamma=0.8, zweight=None)
            a = copy.deepcopy(base); a.sweep = False; a.gui = True; a.out = os.path.join(out_ds, 'final')
            a.mip_bg_method = start['bg']; a.mip_lo = start['min']; a.mip_gamma = start['gamma']
            a.z_weight = [start['zweight']] if start.get('zweight') else None
            a.mip_hi = a.mip_hi or 'p99.9'; a.mip_proj = a.mip_proj or 'max'
            log(f"[batch] GUI final tuning for {stem} (start: {_tag(start)}{', z-weight ' + start['zweight'] if start.get('zweight') else ''}) -> {a.out}\n"
                f"[batch] >>> opening the Studio window now — adjust, then EXPORT or just close the window; Quit skips this dataset <<<")
            try:
                run_dataset(a); rendered.append('final')
            except SystemExit as e:
                log(f"[batch] !! {stem} final failed: {e}")
            except Exception as e:
                import traceback
                log(f"[batch] !! {stem} final crashed: {e.__class__.__name__}: {e}\n" + traceback.format_exc())
            log(f"[batch] {stem} done -> next dataset")
        summary.append(f"{stem}: mask {'yes' if mask_png else 'NO'}, sweep {len(rows)} combos, rendered [{', '.join(rendered)}]  ({(time.time() - t0) / 60:.1f} min)")
    log("\n[batch] SUMMARY\n  " + "\n  ".join(summary))
    log(f"[batch] to render more settings later: --batch {bdir} --picks-file picks.csv  (rows: id,bg,min,gamma) — sweeps are reused")
    blog.close()


def main():
    args = build_parser().parse_args()
    if args.batch:
        run_batch(args)
    else:
        run_dataset(args)


if __name__ == '__main__':
    main()
