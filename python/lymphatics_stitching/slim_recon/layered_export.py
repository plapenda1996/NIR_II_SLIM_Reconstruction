# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
"""Shared layered / multi-format / multi-frame export for SLIM recon outputs.

Used by both:
  - SLIMGui._play_recon_movie (Movie Viewer save panel)
  - SLIMGui._open_recon_pulse_analyzer (Lymphatic Pulse Analyzer)

Each export request specifies, independently:
  - ``layers``   ⊆ {"grayscale", "depth_coded", "depth_map"}
  - ``formats``  ⊆ {"tif", "svg"}
  - ``frames``   = list[int]

Per (frame × layer × format) one file is written. A single sidecar JSON
(``<stem>_sidecar.json``) records enhancement settings, pixel size, fps, and
the depth-index range so the export is fully reproducible from disk.

Conventions
-----------
TIFF
  • grayscale  → 16-bit (uint16), 0..65535 scaled from 0..1
  • depth_map  → 16-bit (uint16), 0..65535 scaled from depth_norm 0..1
                 (or, when ``depth_index_range`` is passed, from the actual
                 depth-index range; preserved in sidecar metadata)
  • depth_coded → 8-bit RGB (already a colormapped image; 16-bit is cosmetic)
  • All TIFs carry dpi=300 and a `description` JSON string with metadata.

SVG
  • A vector container that embeds the raster layer (base64 PNG) as
    ``<image xlink:href="data:image/png;base64,...">`` and draws the scale
    bar + colorbar + labels as vector primitives (``<rect>``, ``<text>``,
    ``<linearGradient>``). Opens editable in Illustrator/Inkscape.
"""

from __future__ import annotations
import base64
import io
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Frame-spec parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_frame_spec(spec: str, T: int) -> List[int]:
    """Parse a frame-selection string into a sorted unique list[int] in [0, T-1].

    Accepted syntax::

        "12"                   -> [12]
        "0,30,60"              -> [0, 30, 60]
        "0-100"                -> [0, 1, 2, ..., 100]
        "0-100:10"             -> [0, 10, 20, ..., 100]
        "0..100/5"             -> [0, 5, 10, ..., 100]
        " 0, 30-60:5 , 200 "   -> [0, 30, 35, ..., 60, 200]

    Out-of-range indices are clamped to [0, T-1] and de-duplicated.
    """
    if spec is None:
        return []
    out: List[int] = []
    s = str(spec).strip()
    if not s:
        return []
    # Normalize the two range forms: a..b/s == a-b:s
    s = s.replace('..', '-').replace('/', ':')
    for token in re.split(r'\s*,\s*', s):
        if not token:
            continue
        if '-' in token:
            # range form a-b[:s]
            range_part, _, step_part = token.partition(':')
            try:
                a_str, b_str = range_part.split('-', 1)
                a, b = int(a_str), int(b_str)
            except ValueError:
                continue
            step = max(1, int(step_part)) if step_part else 1
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1, step))
        else:
            try:
                out.append(int(token))
            except ValueError:
                continue
    out = [max(0, min(int(T) - 1, i)) for i in out]
    out = sorted(set(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TIFF writers
# ─────────────────────────────────────────────────────────────────────────────
def _to_uint16(img01: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0)
    return (a * 65535.0 + 0.5).astype(np.uint16)


def _to_uint8_rgb(img_rgb01: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(img_rgb01, dtype=np.float32), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def write_tiff_grayscale_u16(path: str, img01: np.ndarray, metadata: dict,
                              dpi: int = 300):
    import tifffile
    arr = _to_uint16(img01)
    tifffile.imwrite(path, arr, photometric='minisblack',
                     resolution=(dpi, dpi), resolutionunit='INCH',
                     description=json.dumps(metadata))


def write_tiff_depth_u16(path: str, depth_norm01: np.ndarray, metadata: dict,
                          dpi: int = 300):
    import tifffile
    arr = _to_uint16(depth_norm01)
    tifffile.imwrite(path, arr, photometric='minisblack',
                     resolution=(dpi, dpi), resolutionunit='INCH',
                     description=json.dumps(metadata))


def write_tiff_rgb_u8(path: str, rgb01: np.ndarray, metadata: dict,
                       dpi: int = 300):
    import tifffile
    arr = _to_uint8_rgb(rgb01)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tifffile.imwrite(path, arr, photometric='rgb',
                         resolution=(dpi, dpi), resolutionunit='INCH',
                         description=json.dumps(metadata))
    else:
        write_tiff_grayscale_u16(path, arr / 255.0, metadata, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# PNG → base64 (for SVG embed)
# ─────────────────────────────────────────────────────────────────────────────
def _png_b64(img01_or_rgb01: np.ndarray) -> str:
    from PIL import Image
    a = np.asarray(img01_or_rgb01)
    if a.ndim == 2:
        pil = Image.fromarray(_to_uint16(a), mode='I;16')
        pil = pil.convert('L')                               # 8-bit for PNG embed compactness
    elif a.ndim == 3 and a.shape[2] == 3:
        pil = Image.fromarray(_to_uint8_rgb(a), mode='RGB')
    elif a.ndim == 3 and a.shape[2] == 4:
        pil = Image.fromarray(_to_uint8_rgb(a[..., :3]), mode='RGB')
    else:
        raise ValueError(f"Unsupported array shape for PNG embed: {a.shape}")
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=False)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ─────────────────────────────────────────────────────────────────────────────
# Turbo colormap (vector colorbar stops without importing matplotlib at write)
# ─────────────────────────────────────────────────────────────────────────────
def _turbo_stops(n: int = 16):
    """Return ``n`` (offset, rgb_hex) pairs for the turbo colormap. We use
    matplotlib if available (consistent with the viewer's depth colormap)."""
    try:
        import matplotlib.pyplot as plt
        cmap = plt.colormaps.get_cmap('turbo')
        stops = []
        for i in range(n):
            f = i / (n - 1)
            r, g, b, _ = cmap(f)
            stops.append((f, f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'))
        return stops
    except Exception:
        # Fallback: linear black→cyan→yellow→red (rough turbo)
        ramp = [(0.0, '#30123b'), (0.25, '#1ab0f4'), (0.5, '#a6fa86'),
                (0.75, '#ffa825'), (1.0, '#7a0402')]
        return ramp


# ─────────────────────────────────────────────────────────────────────────────
# SVG writer
# ─────────────────────────────────────────────────────────────────────────────
def write_svg_layer(path: str, image_array: np.ndarray, *, kind: str,
                    px_um: float, scalebar_um: float = 100.0,
                    depth_range: Optional[Tuple[float, float]] = None,
                    cmap_name: str = 'turbo', metadata: Optional[dict] = None,
                    title: Optional[str] = None):
    """Write an SVG that embeds ``image_array`` as a base64 PNG and overlays
    vector primitives for the scale bar (always) and, when ``kind`` is
    ``depth_coded`` or ``depth_map``, a vector colorbar with tick labels.

    Parameters
    ----------
    kind : 'grayscale' | 'depth_coded' | 'depth_map'
    """
    H, W = image_array.shape[:2]
    pad = 30
    cbar_w = 18 if kind in ('depth_coded', 'depth_map') else 0
    cbar_gap = 14 if cbar_w else 0
    cbar_label_w = 60 if cbar_w else 0
    svg_w = W + 2 * pad + cbar_w + cbar_gap + cbar_label_w
    svg_h = H + 2 * pad

    img_b64 = _png_b64(image_array)

    # Scale bar — bottom-right of the image, vector primitives
    bar_px = float(scalebar_um) / float(max(px_um, 1e-9))
    bar_thick = max(3.0, H * 0.012)
    bar_x1 = pad + W - W * 0.04
    bar_x0 = bar_x1 - bar_px
    bar_y  = pad + H - H * 0.06
    bar_label = f"{scalebar_um:g} µm"

    # Colorbar
    cbar_x0 = pad + W + cbar_gap
    cbar_y0 = pad
    cbar_h  = H

    parts = []
    parts.append(f'<?xml version="1.0" encoding="UTF-8"?>\n')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">\n'
    )
    if metadata:
        parts.append(f'  <metadata>{json.dumps(metadata)}</metadata>\n')

    # defs (turbo gradient) — only if needed
    if cbar_w:
        stops = _turbo_stops(16)
        parts.append('  <defs>\n')
        parts.append('    <linearGradient id="turbo" x1="0%" y1="100%" x2="0%" y2="0%">\n')
        for off, hexcol in stops:
            parts.append(f'      <stop offset="{off*100:.2f}%" stop-color="{hexcol}"/>\n')
        parts.append('    </linearGradient>\n')
        parts.append('  </defs>\n')

    # Embedded raster image
    parts.append(
        f'  <image x="{pad}" y="{pad}" width="{W}" height="{H}" '
        f'xlink:href="data:image/png;base64,{img_b64}"/>\n'
    )

    # Optional title above the image
    if title:
        parts.append(
            f'  <text x="{pad + W/2}" y="{pad - 8}" font-family="Arial" '
            f'font-size="12" text-anchor="middle">{title}</text>\n'
        )

    # Scale bar (white rectangle + black text bbox)
    parts.append(
        f'  <rect x="{bar_x0:.1f}" y="{bar_y:.1f}" width="{bar_px:.1f}" '
        f'height="{bar_thick:.1f}" fill="white" stroke="none"/>\n'
    )
    parts.append(
        f'  <text x="{(bar_x0+bar_x1)/2:.1f}" y="{bar_y - 4:.1f}" '
        f'font-family="Arial" font-size="11" fill="white" '
        f'text-anchor="middle" stroke="black" stroke-width="0.4">{bar_label}</text>\n'
    )

    # Colorbar + ticks
    if cbar_w:
        parts.append(
            f'  <rect x="{cbar_x0}" y="{cbar_y0}" width="{cbar_w}" '
            f'height="{cbar_h}" fill="url(#turbo)" stroke="black" '
            f'stroke-width="0.5"/>\n'
        )
        # Tick labels at top / middle / bottom
        if depth_range is not None:
            lo, hi = float(depth_range[0]), float(depth_range[1])
        else:
            lo, hi = 0.0, 1.0
        for frac, val in [(0.0, hi), (0.5, (lo + hi) / 2.0), (1.0, lo)]:
            ty = cbar_y0 + frac * cbar_h
            parts.append(
                f'  <line x1="{cbar_x0 + cbar_w}" y1="{ty:.1f}" '
                f'x2="{cbar_x0 + cbar_w + 4}" y2="{ty:.1f}" '
                f'stroke="black" stroke-width="0.6"/>\n'
            )
            parts.append(
                f'  <text x="{cbar_x0 + cbar_w + 6}" y="{ty + 4:.1f}" '
                f'font-family="Arial" font-size="10">{val:.2f}</text>\n'
            )
        parts.append(
            f'  <text x="{cbar_x0 + cbar_w/2}" y="{cbar_y0 - 6}" '
            f'font-family="Arial" font-size="10" text-anchor="middle">depth</text>\n'
        )

    parts.append('</svg>\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level export driver
# ─────────────────────────────────────────────────────────────────────────────
def export_layers(layers: Sequence[str], formats: Sequence[str],
                  frames: Sequence[int],
                  get_layer_arrays: Callable[[int], Dict[str, np.ndarray]],
                  out_stem: str, *, px_um: float, fps: float,
                  settings: Optional[dict] = None,
                  depth_index_range: Optional[Tuple[int, int]] = None,
                  cmap_name: str = 'turbo',
                  scalebar_um: float = 100.0,
                  sidecar: bool = True) -> List[str]:
    """Driver for layered/multi-format/multi-frame export.

    Returns the list of written file paths. Also writes a single sidecar
    ``<out_stem>_sidecar.json`` with all metadata when ``sidecar`` is True.
    """
    out_dir = os.path.dirname(out_stem) or '.'
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    layers = [l for l in layers if l in ('grayscale', 'depth_coded', 'depth_map')]
    formats = [f for f in formats if f in ('tif', 'svg')]
    if not layers or not formats or not frames:
        return written

    base_meta = {
        'pixel_size_um': float(px_um),
        'fps_hz': float(fps),
        'cmap': cmap_name,
        'enhance_settings': dict(settings) if settings else None,
        'depth_index_range': list(depth_index_range) if depth_index_range else None,
        'scalebar_um': float(scalebar_um),
    }

    for t in frames:
        arrays = get_layer_arrays(int(t)) or {}
        for layer in layers:
            arr = arrays.get(layer)
            if arr is None:
                continue
            for fmt in formats:
                fname = f"{out_stem}_{layer}_f{int(t):04d}.{fmt}"
                meta = dict(base_meta); meta['frame'] = int(t); meta['layer'] = layer
                if fmt == 'tif':
                    if layer == 'depth_coded':
                        write_tiff_rgb_u8(fname, arr, meta)
                    elif layer == 'depth_map':
                        write_tiff_depth_u16(fname, arr, meta)
                    else:                                  # grayscale
                        write_tiff_grayscale_u16(fname, arr, meta)
                else:                                      # svg
                    dr = None
                    if layer in ('depth_coded', 'depth_map'):
                        if depth_index_range:
                            dr = (float(depth_index_range[0]),
                                  float(depth_index_range[1]))
                        else:
                            dr = (0.0, 1.0)
                    write_svg_layer(fname, arr, kind=layer, px_um=px_um,
                                    scalebar_um=scalebar_um, depth_range=dr,
                                    cmap_name=cmap_name, metadata=meta)
                written.append(fname)

    if sidecar and written:
        side_path = f"{out_stem}_sidecar.json"
        with open(side_path, 'w', encoding='utf-8') as f:
            json.dump({**base_meta, 'frames': list(map(int, frames)),
                       'layers': list(layers), 'formats': list(formats),
                       'files': [os.path.basename(p) for p in written]},
                      f, indent=2)
        written.append(side_path)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-tests
# ─────────────────────────────────────────────────────────────────────────────
def _self_test(tmpdir: str = None, verbose: bool = False):
    """Round-trip 16-bit TIFF + SVG-with-text-and-image, assert sanity."""
    import tempfile, tifffile
    rng = np.random.default_rng(0)
    H, W = 32, 40
    gray = rng.random((H, W)).astype(np.float32)
    rgb = rng.random((H, W, 3)).astype(np.float32)
    depth = (rng.random((H, W)) * 0.95 + 0.02).astype(np.float32)
    td = tmpdir or tempfile.mkdtemp(prefix='layered_export_')
    stem = os.path.join(td, 'test')

    def _layers(t):
        return {'grayscale': gray, 'depth_coded': rgb, 'depth_map': depth}

    files = export_layers(
        layers=['grayscale', 'depth_coded', 'depth_map'],
        formats=['tif', 'svg'],
        frames=[0],
        get_layer_arrays=_layers,
        out_stem=stem,
        px_um=4.0, fps=20.0,
        settings={'gamma': 0.5}, depth_index_range=(0, 31),
        cmap_name='turbo', scalebar_um=50.0,
    )
    # bit-depth check
    tif_g = tifffile.imread(stem + '_grayscale_f0000.tif')
    tif_d = tifffile.imread(stem + '_depth_map_f0000.tif')
    tif_c = tifffile.imread(stem + '_depth_coded_f0000.tif')
    assert tif_g.dtype == np.uint16 and tif_g.min() < tif_g.max(), "grayscale TIF not 16-bit / flat"
    assert tif_d.dtype == np.uint16 and tif_d.min() < tif_d.max(), "depth_map TIF not 16-bit / flat"
    assert tif_c.dtype == np.uint8  and tif_c.ndim == 3, "depth_coded TIF not RGB u8"
    # SVG sanity
    for layer in ('grayscale', 'depth_coded', 'depth_map'):
        with open(stem + f'_{layer}_f0000.svg', 'r', encoding='utf-8') as f:
            svg = f.read()
        assert '<image' in svg, f"SVG {layer}: missing <image"
        assert '<text' in svg, f"SVG {layer}: missing <text"
    if verbose:
        for p in files:
            print(' ', os.path.basename(p))
    return td, files


if __name__ == "__main__":
    td, files = _self_test(verbose=True)
    print(f"OK — {len(files)} files in {td}")
