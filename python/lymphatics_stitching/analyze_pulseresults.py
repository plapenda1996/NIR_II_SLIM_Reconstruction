#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - lymphatic pulse statistics - produces Fig. 3k-m; 885 +/- 149 um/s, transport distance, FWHM
# environment: pulse_py314
"""
analyze_pulseresults.py  (calibrated edition)
=============================================
Aggregate `recon_pulse_v1` pulse-results JSON (one interval = one lymphatic contraction),
recover CALIBRATED acquisition time via the ver2 frame map, fix the flow sign, derive
FWHM-based EF / FPF surrogates, and test for pump run-down.

    python analyze_pulseresults.py *_pulseresults.json --out results_final --flip-sign \
        --framemap concate_view_ver2_framemap.json --fps 30
    python analyze_pulseresults.py --selftest

THE TIMING BRIDGE
    The pulse files use an identity frame_map on the 941-frame stack (root_time_valid=False).
    `--framemap concate_view_ver2_framemap.json` supplies the MONOTONIC 941-local -> ORIGINAL
    root map (root = frame_map[local], root.T=28979, fps=30). Composing it makes real acquisition
    time available, so contraction FREQUENCY / ISI become CALIBRATED (root_time_valid=True).

SIGN
    Flow is antegrade; `--flip-sign` (default ON) negates signed_v and relabels retrograde->
    antegrade. |speed| is unchanged.

POSITIONS (added)
    Each pulse also gets ABSOLUTE start/end positions along the FG line (s_start_um / s_end_um,
    s = arclength from the first FG vertex), a long-format bolus_positions.csv, an [F5] block in
    summary.txt (median end, ±window count, greedy 1-D clusters) and pulse_fig3_positions.png/.svg.
    These are only comparable across pulses if [F3] centerline consistency is OK (same FG line).
    --end-window-um sets the ±window (default 50 µm).

EF_fwhm is a FWHM-derived cylindrical-lumen SURROGATE (EF = 1-(w_min/w_max)^2), NOT a volumetric
measurement — no diameter kymograph is available in these files. FPF is likewise a surrogate.

Dependencies: Python stdlib + numpy + matplotlib (Agg). No GUI, no main.py import.
"""
import argparse
import bisect
import csv
import glob
import json
import os
import sys

import numpy as np

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t975(df):
    return float('nan') if df <= 0 else float(_T975.get(int(df), 1.96))


def _utf8_stdout():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8')
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def load_pulse_files(paths):
    """Read each recon_pulse_v1 JSON. Assert schema. Collect per-interval rows (tag#k if a file
    holds >1 interval). Warn on mixed px_um / fps."""
    rows = []; pxs, fpss = set(), set()
    for p in paths:
        try:
            doc = json.load(open(p))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}", file=sys.stderr); continue
        if doc.get('schema') != 'recon_pulse_v1':
            print(f"[warn] {os.path.basename(p)}: schema {doc.get('schema')!r} != recon_pulse_v1 — skipped",
                  file=sys.stderr); continue
        stem = os.path.splitext(os.path.basename(p))[0]
        px_um = float(doc.get('px_um', float('nan'))); fps = float(doc.get('fps', float('nan')))
        rtv = bool(doc.get('root_time_valid')); cl = doc.get('centerline') or []
        ivs = doc.get('intervals') or []; multi = len(ivs) > 1
        for k, it in enumerate(ivs):
            rows.append(dict(
                tag=(f"{stem}#{k}" if multi else stem), file=p,
                v_um_s=float(it.get('v_um_s', float('nan'))), se=float(it.get('se', float('nan'))),
                r2=float(it.get('r2', float('nan'))), direction=str(it.get('direction', '-')),
                transport_um=float(it.get('transport_um', float('nan'))),
                n_used=int(it.get('n_used', 0)), n_kept=int(it.get('n_kept', 0)),
                local_t0=it.get('local_t0'), local_t1=it.get('local_t1'),
                t_peak_local=it.get('t_peak_local'),
                t_peak_root=it.get('t_peak_root'), root_t0=it.get('root_t0'), root_t1=it.get('root_t1'),
                root_time_valid=rtv, centerline=cl, centerline_n=len(cl),
                present=list(it.get('present_local') or []),
                s_star_um=list(it.get('s_star_um') or []), amp_at_star=list(it.get('amp_at_star') or []),
                fwhm_um=list(it.get('fwhm_um') or []), arclen_um=list(it.get('arclen_um') or []),
                px_um=px_um, fps=fps, diff_mode=doc.get('diff_mode'),
                source_tiff=str(doc.get('source_tiff', ''))))
        if np.isfinite(px_um): pxs.add(round(px_um, 6))
        if np.isfinite(fps): fpss.add(round(fps, 6))
    if len(pxs) > 1: print(f"[warn] MIXED px_um across files: {sorted(pxs)}", file=sys.stderr)
    if len(fpss) > 1: print(f"[warn] MIXED fps across files: {sorted(fpss)}", file=sys.stderr)
    return rows


def load_framemap(path):
    """Load a ver2 framemap JSON: returns dict(frame_map=int[], root_T, fps, px_um, windows)."""
    fmj = json.load(open(path))
    fm = np.asarray(fmj.get('frame_map', []), np.int64)
    root = fmj.get('root') or {}
    windows = []
    for seg in (fmj.get('segments') or []):
        try:
            windows.append((int(seg[0]), int(seg[1])))
        except Exception:
            pass
    return dict(frame_map=fm, root_T=int(root.get('T', (int(fm.max()) + 1 if fm.size else 0))),
                fps=(float(fmj['fps']) if fmj.get('fps') is not None else None),
                px_um=(float(fmj['px_um']) if fmj.get('px_um') is not None else None),
                windows=windows)


def apply_framemap(rows, fm_info, fps):
    """Compose each pulse's LOCAL indices through the ver2 frame_map to ORIGINAL root frames, and
    stamp root_frame / root_time_s / root_time_valid / file_idx. Errors if a local index exceeds
    the map length."""
    fm = fm_info['frame_map']; nmap = int(fm.size)
    wins = fm_info.get('windows') or []
    wstarts = [w[0] for w in wins]
    for r in rows:
        # local_t1 is EXCLUSIVE, so its last real index is local_t1-1
        _t1 = r.get('local_t1')
        li = [r.get('t_peak_local'), r.get('local_t0')] + ([int(_t1) - 1] if _t1 is not None else []) \
            + list(r.get('present') or [])
        li = [int(x) for x in li if x is not None]
        if li and max(li) >= nmap:
            raise ValueError(f"{r['tag']}: local index {max(li)} exceeds frame_map length {nmap} — "
                             f"wrong framemap for this stack.")

        def _root(i):
            return int(fm[int(np.clip(i, 0, nmap - 1))])
        r['root_frame'] = _root(r.get('t_peak_local', 0))
        r['root_time_s'] = r['root_frame'] / float(fps)
        r['root_t0'] = _root(r.get('local_t0', 0)); r['root_t1'] = _root(int(r.get('local_t1', 1)) - 1) + 1
        r['t_peak_root'] = r['root_frame']
        r['present_root'] = [_root(i) for i in (r.get('present') or [])]
        r['root_time_valid'] = True
        r['file_idx'] = (bisect.bisect_right(wstarts, r['root_frame']) - 1) if wstarts else -1
    return rows


# ── Recover ROOT frames from truncated-clip filenames ────────────────────────
# Acquisition order + per-file frame counts (root T = 28979, fps 30). Clips are saved as
# already-truncated stacks (identity frame_map, root_time_valid=False); their real ROOT position
# is encoded in the filename '<tag>_..._truncated_<lo>_<hi>_...'.
SOURCE_FILES_DEFAULT = [
    ("record_30012026_182138_a",  1834), ("record_30012026_182311_b",  1035),
    ("record_30012026_182403_c",  1905), ("record_30012026_182539_d",  1650),
    ("record_30012026_182703_e",  3612), ("record_30012026_183004_f0", 4378),
    ("record_30012026_183344_g",  2443), ("record_30012026_183547_h",  3872),
    ("record_30012026_183901_i0", 4378), ("record_30012026_184241_j",  3872),
]


def recover_root_from_filenames(rows, source_files=SOURCE_FILES_DEFAULT):
    """source_files = [(name, nframes)] in acquisition order. For each row with root_time_valid
    False, parse '<tag>_..._truncated_<lo>_<hi>' from source_tiff, compute the file's cumulative
    offset, and shift interval root_t0/t1/t_peak_root + root_frame/root_time_s to ROOT.
    Returns (rows, n_fixed, root_T)."""
    import re
    off = {}; c = 0
    for name, n in source_files:
        off[str(name).split('_')[-1]] = c; c += int(n)
    root_T = c; n_fixed = 0
    for r in rows:
        if r.get('root_time_valid'):
            continue
        base = os.path.splitext(os.path.basename(r.get('source_tiff', '') or ''))[0]
        m = re.match(r'([A-Za-z0-9]+?)_.*?truncated_(\d+)_(\d+)', base)
        if m:
            tag, lo = m.group(1), int(m.group(2))
        else:
            m2 = re.match(r'([A-Za-z0-9]+?)_', base); tag = m2.group(1) if m2 else None; lo = 1
        if tag not in off:                       # f0/i0 vs f/i: accept either spelling
            tag = (tag + '0') if (tag and (tag + '0') in off) else \
                  (tag[:-1] if (tag and tag[-1:] == '0' and tag[:-1] in off) else tag)
        if tag not in off:
            continue
        clip_root0 = off[tag] + (lo - 1)
        lt0 = int(r.get('local_t0') or 0); lt1 = int(r.get('local_t1') or lt0 + 1)
        tpl = r.get('t_peak_local'); tpk = int(tpl) if tpl is not None else (lt0 + lt1) // 2
        r['root_t0'] = clip_root0 + lt0; r['root_t1'] = clip_root0 + lt1
        r['t_peak_root'] = clip_root0 + tpk; r['root_frame'] = clip_root0 + tpk
        r['root_time_s'] = (clip_root0 + tpk) / float(r.get('fps') or 30.0)
        r['root_time_valid'] = True; n_fixed += 1
    return rows, n_fixed, root_T


# ─────────────────────────────────────────────────────────────────────────────
def _peak_speed(present, s_star, fps):
    s = np.asarray(s_star, float); pr = np.asarray(present, float)
    if s.size < 2 or pr.size != s.size or not np.isfinite(fps) or fps <= 0:
        return float('nan')
    o = np.argsort(pr); s = s[o]; pr = pr[o]
    dt = np.diff(pr) / float(fps); dt = np.where(dt == 0, np.nan, dt)
    v = np.abs(np.diff(s)) / dt
    return float(np.nanmax(v)) if np.isfinite(v).any() else float('nan')


def _bolus_positions(present, s_star, amp=None):
    """ABSOLUTE bolus positions along the FG line (arclength s, µm; s=0 = first FG vertex).
    Time-ordered by `present`; non-finite s* are dropped. s_start/s_end = s* at the first/last
    frame in which the peak was detected; s_at_amp_max = s* at the brightest frame.
    NOTE: s_end is where the tracked leading edge was LAST detected — this is bounded by the
    amplitude gate / SNR and by the clip length, not necessarily by a vessel feature."""
    nan = float('nan')
    out = dict(s_start=nan, s_end=nan, s_min=nan, s_max=nan, s_extent=nan, s_at_amp_max=nan)
    s = np.asarray([(np.nan if x is None else x) for x in (s_star or [])], float)
    pr = np.asarray([(np.nan if x is None else x) for x in (present or [])], float)
    if s.size == 0 or pr.size != s.size:
        return out
    a = np.asarray([(np.nan if x is None else x) for x in (amp or [])], float)
    if a.size != s.size:
        a = np.full(s.size, np.nan)
    m = np.isfinite(s) & np.isfinite(pr)
    if m.sum() == 0:
        return out
    s, pr, a = s[m], pr[m], a[m]
    o = np.argsort(pr, kind='stable'); s, a = s[o], a[o]
    out.update(s_start=float(s[0]), s_end=float(s[-1]), s_min=float(s.min()), s_max=float(s.max()),
               s_extent=float(s.max() - s.min()))
    if np.isfinite(a).any():
        out['s_at_amp_max'] = float(s[int(np.nanargmax(a))])
    return out


def centerline_consistent(table):
    """True if every pulse carries the same FG line (same total arclength to 0.1 µm and the same
    vertex count). Only then are s_* values from different pulses in ONE coordinate system."""
    lens = [round(r['centerline_len_um'], 1) for r in table if np.isfinite(r['centerline_len_um'])]
    npts = set(r['centerline_n'] for r in table)
    return not ((len(set(lens)) > 1) or (len(npts) > 1)), sorted(set(lens)), sorted(npts)


def _clusters_1d(vals, gap_um):
    """Greedy 1-D clustering: sort, split wherever consecutive values differ by more than gap_um.
    Returns a list of clusters (each a list of values)."""
    v = sorted(float(x) for x in vals if x is not None and np.isfinite(x))
    if not v:
        return []
    cl = [[v[0]]]
    for x in v[1:]:
        if x - cl[-1][-1] > gap_um:
            cl.append([x])
        else:
            cl[-1].append(x)
    return cl


def position_summary(table, window_um=50.0):
    """Do bolus END (and START) positions recur at the same place along the FG line?
    Reports per-pulse s_start/s_end, their spread, the fraction of ends within ±window_um of the
    median end, and greedy 1-D clusters (gap > window_um starts a new cluster). Positions are only
    comparable if centerline_consistent() is True."""
    ok, lens, npts = centerline_consistent(table)
    srt = sorted(table, key=lambda z: (z['root_time_s'] if z['root_time_valid'] and z['root_time_s'] is not None else 0))
    ends = [r['s_end_um'] for r in srt]; starts = [r['s_start_um'] for r in srt]
    E = _stats(ends); S = _stats(starts)
    L = float(np.nanmedian([r['centerline_len_um'] for r in srt])) if srt else float('nan')
    fe = [x for x in ends if np.isfinite(x)]
    within = int(sum(1 for x in fe if abs(x - E['median']) <= window_um)) if fe else 0
    clusters = _clusters_1d(fe, window_um)
    return dict(comparable=bool(ok), centerline_lens=lens, centerline_npts=npts,
                centerline_len_um=L, window_um=float(window_um),
                tags=[r['tag'] for r in srt], s_start=starts, s_end=ends,
                end_mean=E['mean'], end_sd=E['sd'], end_median=E['median'], end_n=E['n'],
                start_mean=S['mean'], start_sd=S['sd'], start_median=S['median'],
                end_within_window=within,
                end_clusters=[dict(n=len(c), lo=min(c), hi=max(c), center=float(np.mean(c))) for c in clusters],
                n_end_clusters=len(clusters))


def per_pulse_table(rows, flip_sign=True, fps=None):
    """One row per pulse with calibrated timing (if apply_framemap ran) and FWHM physiology."""
    swap = {'antegrade': 'retrograde', 'retrograde': 'antegrade', '-': '-'}
    out = []
    for r in rows:
        v = r['v_um_s']; direction = r['direction']
        if flip_sign:
            v = -v; direction = swap.get(direction, direction)
        f = float(fps) if fps else r['fps']
        n_kept = r['n_kept']
        fwhm = np.asarray(r['fwhm_um'], float); fwhm = fwhm[np.isfinite(fwhm)]
        w_min = float(np.min(fwhm)) if fwhm.size else float('nan')
        w_max = float(np.max(fwhm)) if fwhm.size else float('nan')
        di = ((w_max - w_min) / w_max) if (np.isfinite(w_max) and w_max > 0) else float('nan')
        ef = (1.0 - (w_min / w_max) ** 2) if (np.isfinite(w_max) and w_max > 0) else float('nan')
        arc = np.asarray(r['arclen_um'], float)
        quality = 'PASS' if (np.isfinite(r['r2']) and r['r2'] >= 0.5 and r['n_used'] >= 5) else 'LOW'
        pos = _bolus_positions(r['present'], r['s_star_um'], r['amp_at_star'])
        out.append(dict(
            tag=r['tag'], speed_um_s=float(abs(v)), signed_v=float(v), se=float(r['se']),
            r2=float(r['r2']), direction=direction, quality=quality,
            transport_um=float(abs(r['transport_um'])),
            duration_s=float(n_kept / f) if (np.isfinite(f) and f > 0) else float('nan'),
            peak_speed=_peak_speed(r['present'], r['s_star_um'], f),
            bolus_fwhm_um=float(np.median(fwhm)) if fwhm.size else float('nan'),
            w_min=w_min, w_max=w_max, DI=di, EF_fwhm=ef,
            s_start_um=pos['s_start'], s_end_um=pos['s_end'], s_min_um=pos['s_min'],
            s_max_um=pos['s_max'], s_extent_um=pos['s_extent'], s_at_amp_max_um=pos['s_at_amp_max'],
            centerline_len_um=float(arc[-1]) if arc.size else float('nan'),
            centerline_n=r['centerline_n'],
            root_frame=r.get('root_frame'), root_time_s=r.get('root_time_s'),
            file_idx=r.get('file_idx', -1),
            root_time_valid=bool(r.get('root_time_valid')),
            n_used=r['n_used'], n_kept=r['n_kept'],
            _s_star=r['s_star_um'], _present=r['present'], _amp=r['amp_at_star'], _fwhm=r['fwhm_um'],
            _t_peak_local=r.get('t_peak_local'), _present_root=list(r.get('present_root') or [])))
    return out


# ─────────────────────────────────────────────────────────────────────────────
def theilsen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n = x.size; sl = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    slope = float(np.median(sl)) if sl else float('nan')
    inter = float(np.median(y - slope * x)) if np.isfinite(slope) else float('nan')
    return slope, inter


def theilsen_ci(x, y, nboot=2000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    slope, _ = theilsen(x, y); n = x.size
    if n < 3:
        return slope, float('nan'), float('nan'), False
    rng = np.random.default_rng(seed); ss = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n); s, _i = theilsen(x[idx], y[idx])
        if np.isfinite(s):
            ss.append(s)
    lo, hi = (float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))) if ss else (float('nan'), float('nan'))
    excl0 = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    return slope, lo, hi, excl0


def _stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float); n = int(v.size)
    if n == 0:
        return dict(n=0, mean=float('nan'), sd=float('nan'), sem=float('nan'), ci_lo=float('nan'),
                    ci_hi=float('nan'), cv_pct=float('nan'), median=float('nan'),
                    iqr_lo=float('nan'), iqr_hi=float('nan'))
    mean = float(v.mean()); sd = float(v.std(ddof=1)) if n >= 2 else 0.0
    sem = sd / np.sqrt(n); tc = _t975(n - 1) if n >= 2 else float('nan')
    half = tc * sem if (n >= 2 and np.isfinite(tc)) else float('nan')
    cv = 100.0 * sd / abs(mean) if mean != 0 else float('nan')
    return dict(n=n, mean=mean, sd=sd, sem=float(sem), ci_lo=mean - half, ci_hi=mean + half,
                cv_pct=cv, median=float(np.median(v)),
                iqr_lo=float(np.percentile(v, 25)), iqr_hi=float(np.percentile(v, 75)))


def aggregate(table, root_T=None, fps=30.0):
    speeds = [r['speed_um_s'] for r in table]
    pass_rows = [r for r in table if r['quality'] == 'PASS']
    A = _stats(speeds); P = _stats([r['speed_um_s'] for r in pass_rows])
    ef = _stats([r['EF_fwhm'] for r in table]); di = _stats([r['DI'] for r in table])
    tr = _stats([r['transport_um'] for r in table]); bf = _stats([r['bolus_fwhm_um'] for r in table])

    n = len(table)
    n_ante = sum(1 for r in table if r['direction'] == 'antegrade')
    n_retro = sum(1 for r in table if r['direction'] == 'retrograde')
    consensus = (f"{n_ante}/{n} antegrade" if n_ante >= n_retro else f"{n_retro}/{n} retrograde")

    calibrated = all(r['root_time_valid'] for r in table) and root_T
    out = dict(n=n, n_pass=len(pass_rows),
               speed_mean=A['mean'], speed_sd=A['sd'], speed_sem=A['sem'],
               speed_ci_lo=A['ci_lo'], speed_ci_hi=A['ci_hi'], speed_cv_pct=A['cv_pct'],
               speed_median=A['median'], speed_iqr_lo=A['iqr_lo'], speed_iqr_hi=A['iqr_hi'],
               pass_speed_mean=P['mean'], pass_speed_sd=P['sd'],
               pass_speed_ci_lo=P['ci_lo'], pass_speed_ci_hi=P['ci_hi'],
               ef_fwhm_mean=ef['mean'], ef_fwhm_sd=ef['sd'], ef_fwhm_ci_lo=ef['ci_lo'], ef_fwhm_ci_hi=ef['ci_hi'],
               di_mean=di['mean'], di_sd=di['sd'],
               transport_mean=tr['mean'], transport_sd=tr['sd'],
               bolus_fwhm_mean=bf['mean'], bolus_fwhm_sd=bf['sd'],
               direction_consensus=consensus, n_antegrade=n_ante, n_retrograde=n_retro,
               root_time_valid=bool(calibrated))

    if calibrated:
        srt = sorted(table, key=lambda r: r['root_time_s'])
        peaks_s = np.asarray([r['root_time_s'] for r in srt], float)
        peaks_f = np.asarray([r['root_frame'] for r in srt], float)
        rec_s = root_T / float(fps)
        isi = np.diff(peaks_s)
        out['frequency_cpm'] = float(60.0 * n / rec_s)
        out['isi_mean_s'] = float(isi.mean()) if isi.size else float('nan')
        out['isi_sd_s'] = float(isi.std(ddof=1)) if isi.size >= 2 else float('nan')
        out['isi_cv'] = float(100.0 * out['isi_sd_s'] / out['isi_mean_s']) if (isi.size >= 2 and out['isi_mean_s'] > 0) else float('nan')
        out['coverage'] = float((peaks_f[-1] - peaks_f[0]) / root_T) if peaks_f.size >= 2 else float('nan')
        out['recording_s'] = float(rec_s)
        dbl = [(srt[i]['tag'], srt[i + 1]['tag'], float(isi[i])) for i in range(isi.size) if isi[i] < 10.0]
        out['doublets'] = dbl
        out['fpf_surrogate'] = float(ef['mean'] * out['frequency_cpm']) if np.isfinite(ef['mean']) else float('nan')
        out['frequency_withheld'] = False
        # run-down: speed / EF / amplitude vs REAL root time
        t = peaks_s
        sp = np.asarray([r['speed_um_s'] for r in srt], float)
        efv = np.asarray([r['EF_fwhm'] for r in srt], float)
        amp = np.asarray([float(np.nanmax(np.asarray(r['_amp'], float))) if len(r['_amp']) else np.nan for r in srt], float)
        for key, yv in (('speed', sp), ('ef', efv), ('amp', amp)):
            s, lo, hi, ex = theilsen_ci(t, yv)
            out[f'rundown_{key}_slope'] = s; out[f'rundown_{key}_ci_lo'] = lo
            out[f'rundown_{key}_ci_hi'] = hi; out[f'rundown_{key}_excludes_zero'] = ex
        out['rundown_slope'] = out['rundown_speed_slope']
        out['rundown_ci'] = [out['rundown_speed_ci_lo'], out['rundown_speed_ci_hi']]
    else:
        out['frequency_withheld'] = True
        out['note'] = ("contraction frequency: WITHHELD (frames are concat-local; "
                      "root_time_valid=False; pass --framemap to calibrate)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
def _nature_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
                         'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'legend.fontsize': 5, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 0.9})
    return plt


def _despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')


MM = 1 / 25.4
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']


def figures(table, agg, outdir):
    plt = _nature_style()
    srt = sorted(table, key=lambda r: (r['root_time_s'] if r['root_time_valid'] and r['root_time_s'] is not None else 0))
    cal = agg.get('root_time_valid')
    # ── FIG 1 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(183 * MM, 120 * MM), dpi=300, facecolor='white')
    ax = [fig.add_subplot(2, 3, i + 1) for i in range(6)]
    # (a) forest
    n = len(table); yy = np.arange(n)[::-1]
    for i, r in enumerate(table):
        c = '#999999' if r['quality'] == 'LOW' else CB[0]
        ax[0].errorbar([r['speed_um_s']], [yy[i]], xerr=[1.96 * (r['se'] if np.isfinite(r['se']) else 0)],
                       fmt='o', ms=3, lw=0.7, capsize=2, color=c)
    m, lo, hi = agg['speed_mean'], agg['speed_ci_lo'], agg['speed_ci_hi']
    ax[0].axvline(m, color=CB[1], ls='--', lw=0.8)
    ax[0].fill_betweenx([-1.6, -0.6], lo, hi, color=CB[1], alpha=0.25, lw=0)
    ax[0].plot([m], [-1.1], 'D', color=CB[1], ms=5)
    ax[0].set_yticks(list(yy) + [-1.1]); ax[0].set_yticklabels([r['tag'][:8] for r in table] + ['pooled'])
    ax[0].set_ylim(-2, n); ax[0].set_xlabel('antegrade speed (µm/s)')
    ax[0].set_title(f"per-pulse speed  ({m:.0f} µm/s)"); _panel(ax[0], 'a')
    # (b) speed vs root time + Theil-Sen
    if cal:
        t = np.asarray([r['root_time_s'] for r in srt], float); y = np.asarray([r['speed_um_s'] for r in srt], float)
        ax[1].plot(t, y, 'o', ms=3, color=CB[0])
        s, blo, bhi, ex = theilsen_ci(t, y)
        if np.isfinite(s):
            b = float(np.median(y - s * t)); xf = np.linspace(t.min(), t.max(), 50)
            ax[1].plot(xf, s * xf + b, '-', color=CB[1], lw=1.0)
            ax[1].annotate(f"slope {s:+.2f} µm/s per s\n95%CI[{blo:+.2f},{bhi:+.2f}] {'*' if ex else 'ns'}",
                           (0.03, 0.05), xycoords='axes fraction', fontsize=5)
        ax[1].set_xlabel('root time (s)')
    else:
        ax[1].text(0.5, 0.5, 'no framemap:\ntime uncalibrated', ha='center', va='center', transform=ax[1].transAxes)
    ax[1].set_ylabel('speed (µm/s)'); ax[1].set_title('run-down: speed vs real time'); _panel(ax[1], 'b')
    # (c) s*(t) overlaid, seconds, zeroed
    for i, r in enumerate(srt):
        s = np.asarray(r['_s_star'], float); pr = np.asarray(r['_present'], float)
        if s.size < 2:
            continue
        o = np.argsort(pr); s = s[o]; pr = pr[o]; tt = (pr - pr[0]) / 30.0
        ax[2].plot(tt, s - s[0], '-', lw=0.7, color=CB[i % len(CB)], label=r['tag'][:6])
    ax[2].set_xlabel('time since first frame (s)'); ax[2].set_ylabel('s* − s*₀ (µm)')
    ax[2].set_title('bolus trajectories'); _panel(ax[2], 'c')
    # (d) FWHM w(t) spaghetti + wmin/wmax
    for i, r in enumerate(srt):
        w = np.asarray(r['_fwhm'], float); pr = np.asarray(r['_present'], float)
        m2 = np.isfinite(w)
        if m2.sum() < 2:
            continue
        o = np.argsort(pr[m2]); tt = (pr[m2][o] - pr[m2][o][0]) / 30.0
        ax[3].plot(tt, w[m2][o], '-', lw=0.6, color=CB[i % len(CB)])
    ax[3].set_xlabel('time (s)'); ax[3].set_ylabel('bolus FWHM (µm)')
    ax[3].set_title('w(t) → EF_fwhm basis'); _panel(ax[3], 'd')
    # (e) EF_fwhm & DI bars
    idx = np.arange(n)
    ax[4].bar(idx - 0.2, [r['EF_fwhm'] for r in table], 0.4, color=CB[2], label='EF_fwhm')
    ax[4].bar(idx + 0.2, [r['DI'] for r in table], 0.4, color=CB[3], label='DI')
    ax[4].axhline(agg['ef_fwhm_mean'], color=CB[2], ls=':', lw=0.8)
    ax[4].set_xticks(idx); ax[4].set_xticklabels([r['tag'][:6] for r in table], rotation=90)
    ax[4].set_ylabel('fraction'); ax[4].set_title('EF_fwhm & DI (surrogate)'); ax[4].legend(frameon=False); _panel(ax[4], 'e')
    # (f) summary table
    ax[5].axis('off')
    freq = f"{agg.get('frequency_cpm', float('nan')):.2f} cpm" if cal else "WITHHELD"
    isi = f"{agg.get('isi_mean_s', float('nan')):.1f}±{agg.get('isi_sd_s', float('nan')):.1f}s" if cal else "—"
    tb = [['metric', 'value'],
          ['n (PASS)', f"{agg['n']} ({agg['n_pass']})"],
          ['speed mean±SD', f"{agg['speed_mean']:.0f}±{agg['speed_sd']:.0f}"],
          ['speed 95%CI', f"[{agg['speed_ci_lo']:.0f},{agg['speed_ci_hi']:.0f}]"],
          ['speed CV%', f"{agg['speed_cv_pct']:.0f}"],
          ['EF_fwhm', f"{agg['ef_fwhm_mean']:.2f}±{agg['ef_fwhm_sd']:.2f}"],
          ['DI', f"{agg['di_mean']:.2f}±{agg['di_sd']:.2f}"],
          ['transport µm', f"{agg['transport_mean']:.0f}±{agg['transport_sd']:.0f}"],
          ['bolus FWHM µm', f"{agg['bolus_fwhm_mean']:.0f}±{agg['bolus_fwhm_sd']:.0f}"],
          ['frequency', freq], ['ISI', isi],
          ['direction', agg['direction_consensus']]]
    t2 = ax[5].table(cellText=tb, loc='center', cellLoc='left')
    t2.auto_set_font_size(False); t2.set_fontsize(5.5); t2.scale(1, 1.15)
    ax[5].set_title('summary'); _panel(ax[5], 'f')
    for a in ax[:5]:
        _despine(a)
    fig.suptitle("Lymphatic pulse kinematics — antegrade; frequency CALIBRATED via ver2 frame map "
                 "(EF_fwhm is a FWHM-derived surrogate, not volumetric)", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f1p = os.path.join(outdir, 'pulse_fig1'); fig.savefig(f1p + '.png'); fig.savefig(f1p + '.svg'); plt.close(fig)

    # ── FIG 2 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(183 * MM, 120 * MM), dpi=300, facecolor='white')
    bx = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    rec_s = agg.get('recording_s', float('nan'))
    if cal:
        # (a) raster on 0..rec_s
        for (r0, r1) in (agg.get('_windows') or []):
            bx[0].axvspan(r0 / 30.0, r1 / 30.0, color='#dddddd', alpha=0.5, lw=0)
        for r in srt:
            bx[0].axvline(r['root_time_s'], color=CB[1], lw=1.0)
            bx[0].text(r['root_time_s'], 1.02, r['tag'][:1], fontsize=5, ha='center', va='bottom', color=CB[1])
        for a2, b2, iv in (agg.get('doublets') or []):
            ta = next((r['root_time_s'] for r in srt if r['tag'] == a2), None)
            tb2 = next((r['root_time_s'] for r in srt if r['tag'] == b2), None)
            if ta is not None and tb2 is not None:
                bx[0].plot([ta, tb2], [0.5, 0.5], color=CB[0], lw=1.5, marker='|')
                bx[0].text((ta + tb2) / 2, 0.55, f"doublet {iv:.1f}s", fontsize=5, ha='center', color=CB[0])
        bx[0].set_xlim(0, rec_s); bx[0].set_ylim(0, 1.15); bx[0].set_yticks([])
        bx[0].set_xlabel('root time (s)')
        bx[0].set_title(f"event raster (coverage {100*agg.get('coverage',0):.0f}%)")
        # (b) ISI sequence + hist
        peaks = np.sort([r['root_time_s'] for r in srt]); isis = np.diff(peaks)
        bx[1].plot(np.arange(1, isis.size + 1), isis, 'o-', color=CB[0], ms=3)
        bx[1].axhline(10, color=CB[1], ls=':', lw=0.8)
        for j in range(isis.size):
            if isis[j] < 10:
                bx[1].plot([j + 1], [isis[j]], 'o', color=CB[1], ms=6, mfc='none')
        bx[1].set_xlabel('interval #'); bx[1].set_ylabel('ISI (s)')
        bx[1].set_title(f"ISI {agg['isi_mean_s']:.0f}±{agg['isi_sd_s']:.0f}s CV{agg['isi_cv']:.0f}%")
        # (c) EF vs time Theil-Sen
        t = np.asarray([r['root_time_s'] for r in srt], float); y = np.asarray([r['EF_fwhm'] for r in srt], float)
        bx[2].plot(t, y, 'o', ms=3, color=CB[2])
        s, lo, hi, ex = theilsen_ci(t, y)
        if np.isfinite(s):
            b = float(np.median(y - s * t)); xf = np.linspace(t.min(), t.max(), 50)
            bx[2].plot(xf, s * xf + b, '-', color=CB[1], lw=1.0)
        bx[2].set_xlabel('root time (s)'); bx[2].set_ylabel('EF_fwhm'); bx[2].set_title('EF_fwhm vs time')
        # (d) amplitude phase-normalized ensemble
        ph = np.linspace(0, 1, 50); ens = []
        for r in srt:
            a = np.asarray(r['_amp'], float)
            if a.size >= 2 and np.isfinite(a).any():
                a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
                b0 = np.percentile(a, 10)
                ens.append(np.interp(ph, np.linspace(0, 1, a.size), (a - b0) / (abs(b0) + 1e-9)))
        if ens:
            E = np.array(ens); mu = E.mean(0); sd = E.std(0)
            bx[3].plot(ph, mu, color=CB[0]); bx[3].fill_between(ph, mu - sd, mu + sd, color=CB[0], alpha=0.25, lw=0)
        bx[3].set_xlabel('phase'); bx[3].set_ylabel('amp dF/F0-like'); bx[3].set_title('amplitude ensemble')
    else:
        for b in bx:
            b.text(0.5, 0.5, 'no framemap', ha='center', va='center', transform=b.transAxes)
    letters = 'abcd'
    for i, a in enumerate(bx):
        _despine(a); _panel(a, letters[i])
    fig.suptitle("Calibrated cohort timing (root_time_valid=True via ver2 frame map)", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f2p = os.path.join(outdir, 'pulse_fig2'); fig.savefig(f2p + '.png'); fig.savefig(f2p + '.svg'); plt.close(fig)
    # ── FIG 3: ABSOLUTE positions along the FG line (do bolus ends recur at one place?) ──────
    ps = position_summary(table, window_um=agg.get('_window_um', 50.0))
    fig = plt.figure(figsize=(183 * MM, 60 * MM), dpi=300, facecolor='white')
    ax = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    L = ps['centerline_len_um']; n = len(srt); yy = np.arange(n)[::-1]
    # (a) start -> end segment per pulse, ordered by real time (top = earliest)
    for i, r in enumerate(srt):
        s0, s1 = r['s_start_um'], r['s_end_um']
        c = '#999999' if r['quality'] == 'LOW' else CB[0]
        if np.isfinite(s0) and np.isfinite(s1):
            ax[0].plot([s0, s1], [yy[i], yy[i]], '-', color=c, lw=1.2)
            ax[0].plot([s0], [yy[i]], 'o', mfc='white', mec=c, ms=3.5, mew=0.8)
            ax[0].plot([s1], [yy[i]], 'o', color=c, ms=3.5)
    if np.isfinite(ps['end_median']):
        ax[0].axvline(ps['end_median'], color=CB[1], ls='--', lw=0.8)
        ax[0].axvspan(ps['end_median'] - ps['window_um'], ps['end_median'] + ps['window_um'],
                      color=CB[1], alpha=0.12, lw=0)
    if np.isfinite(L):
        ax[0].set_xlim(0, L)
    ax[0].set_yticks(list(yy)); ax[0].set_yticklabels([r['tag'][:8] for r in srt])
    ax[0].set_xlabel('position along FG line, s (µm)')
    ax[0].set_title(f"start (open) → end (filled); ends {ps['end_within_window']}/{ps['end_n']} within ±{ps['window_um']:.0f} µm")
    _panel(ax[0], 'a')
    # (b) absolute s*(t) trajectories
    for i, r in enumerate(srt):
        s = np.asarray([(np.nan if x is None else x) for x in r['_s_star']], float)
        pr = np.asarray([(np.nan if x is None else x) for x in r['_present']], float)
        m2 = np.isfinite(s) & np.isfinite(pr)
        if m2.sum() < 2:
            continue
        o = np.argsort(pr[m2]); tt = (pr[m2][o] - pr[m2][o][0]) / 30.0
        ax[1].plot(tt, s[m2][o], '-', lw=0.7, color=CB[i % len(CB)], label=r['tag'][:6])
    if np.isfinite(ps['end_median']):
        ax[1].axhline(ps['end_median'], color=CB[1], ls='--', lw=0.8)
    if np.isfinite(L):
        ax[1].set_ylim(0, L)
    ax[1].set_xlabel('time since first frame (s)'); ax[1].set_ylabel('s (µm, absolute)')
    ax[1].set_title('absolute trajectories'); _panel(ax[1], 'b')
    # (c) where ends / starts fall along the vessel
    fe = [x for x in ps['s_end'] if np.isfinite(x)]; fs = [x for x in ps['s_start'] if np.isfinite(x)]
    if np.isfinite(L) and L > 0:
        bins = np.arange(0, L + 25.0, 25.0)
        if fe: ax[2].hist(fe, bins=bins, color=CB[0], alpha=0.85, label='end')
        if fs: ax[2].hist(fs, bins=bins, color=CB[2], alpha=0.5, label='start')
        ax[2].legend(frameon=False)
    ax[2].set_xlabel('s (µm)'); ax[2].set_ylabel('count')
    ax[2].set_title(f"end clusters (gap>{ps['window_um']:.0f} µm): {ps['n_end_clusters']}"); _panel(ax[2], 'c')
    if not ps['comparable']:
        fig.text(0.5, 0.97, 'WARNING: FG lines differ between pulses — positions not comparable',
                 ha='center', va='top', color=CB[1], fontsize=7, fontweight='bold')
    for a_ in ax:
        _despine(a_); a_.tick_params(direction='out')
    fig.tight_layout()
    f3p = os.path.join(outdir, 'pulse_fig3_positions'); fig.savefig(f3p + '.png'); fig.savefig(f3p + '.svg'); plt.close(fig)
    return f1p, f2p, f3p


# ─────────────────────────────────────────────────────────────────────────────
def write_tables(table, agg, outdir, flip_sign=True, window_um=50.0):
    cols = ['tag', 'speed_um_s', 'signed_v', 'se', 'r2', 'direction', 'quality', 'transport_um',
            'duration_s', 'peak_speed', 'bolus_fwhm_um', 'w_min', 'w_max', 'DI', 'EF_fwhm',
            's_start_um', 's_end_um', 's_min_um', 's_max_um', 's_extent_um', 's_at_amp_max_um',
            'centerline_len_um', 'centerline_n', 'root_frame', 'root_time_s', 'file_idx',
            'root_time_valid', 'n_used', 'n_kept']
    srt = sorted(table, key=lambda z: (z['root_time_s'] if z['root_time_valid'] and z['root_time_s'] is not None else 0))
    with open(os.path.join(outdir, 'per_pulse.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in srt:
            w.writerow([r.get(c) for c in cols])

    # long-format per-frame trajectory in ABSOLUTE FG-line coordinates (one row per detected frame)
    with open(os.path.join(outdir, 'bolus_positions.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tag', 'local_frame', 'root_frame', 't_since_first_s', 's_um', 'amp', 'fwhm_um'])
        for r in srt:
            s = np.asarray([(np.nan if x is None else x) for x in r['_s_star']], float)
            pr = np.asarray([(np.nan if x is None else x) for x in r['_present']], float)
            if s.size == 0 or pr.size != s.size:
                continue
            a = np.asarray([(np.nan if x is None else x) for x in (r['_amp'] or [])], float)
            fw = np.asarray([(np.nan if x is None else x) for x in (r['_fwhm'] or [])], float)
            if a.size != s.size: a = np.full(s.size, np.nan)
            if fw.size != s.size: fw = np.full(s.size, np.nan)
            o = np.argsort(pr, kind='stable')
            t0 = pr[o][0]
            rf0 = r.get('root_frame'); pk = r.get('_t_peak_local')
            proot = list(r.get('_present_root') or [])
            proot = proot if len(proot) == s.size else None      # prefer the composed map when present
            for i in o:
                if not (np.isfinite(s[i]) and np.isfinite(pr[i])):
                    continue
                if proot is not None and proot[i] is not None:
                    root = int(proot[i])
                else:
                    root = (int(rf0) + int(pr[i]) - int(pk)) if (rf0 is not None and pk is not None) else ''
                w.writerow([r['tag'], int(pr[i]), root, f"{(pr[i] - t0) / 30.0:.4f}", f"{s[i]:.2f}",
                            (f"{a[i]:.4g}" if np.isfinite(a[i]) else ''), (f"{fw[i]:.2f}" if np.isfinite(fw[i]) else '')])

    ps = position_summary(table, window_um=window_um)
    cl_ok, lens, npts = centerline_consistent(table)
    cl_bad = not cl_ok
    cal = agg.get('root_time_valid')
    with open(os.path.join(outdir, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write("LYMPHATIC PULSE KINEMATICS + CALIBRATED PHYSIOLOGY — summary\n" + "=" * 64 + "\n\n")
        fh.write(f"n pulses = {agg['n']}  (PASS r2>=0.5 & n_used>=5 = {agg['n_pass']})\n")
        fh.write(f"root_time_valid = {agg['root_time_valid']} "
                 f"({'CALIBRATED via ver2 frame map' if cal else 'concat-order estimate'})\n\n")
        fh.write("PROPAGATION SPEED |v| (biological replicate = pulse):\n")
        fh.write(f"  mean±SD {agg['speed_mean']:.1f}±{agg['speed_sd']:.1f} µm/s   SEM {agg['speed_sem']:.1f}\n")
        fh.write(f"  95%CI(t) [{agg['speed_ci_lo']:.1f},{agg['speed_ci_hi']:.1f}]   CV {agg['speed_cv_pct']:.0f}%\n")
        fh.write(f"  median[IQR] {agg['speed_median']:.1f} [{agg['speed_iqr_lo']:.1f},{agg['speed_iqr_hi']:.1f}]\n")
        fh.write(f"  PASS-only  {agg['pass_speed_mean']:.1f}±{agg['pass_speed_sd']:.1f} (n={agg['n_pass']})\n\n")
        fh.write("FWHM-DERIVED SURROGATES (NOT volumetric — no diameter kymograph):\n")
        fh.write(f"  EF_fwhm = 1-(w_min/w_max)^2 : {agg['ef_fwhm_mean']:.3f}±{agg['ef_fwhm_sd']:.3f}\n")
        fh.write(f"  DI = (w_max-w_min)/w_max    : {agg['di_mean']:.3f}±{agg['di_sd']:.3f}\n")
        fh.write(f"  transport {agg['transport_mean']:.0f}±{agg['transport_sd']:.0f} µm ; "
                 f"bolus FWHM {agg['bolus_fwhm_mean']:.0f}±{agg['bolus_fwhm_sd']:.0f} µm\n\n")
        fh.write(f"DIRECTION consensus: {agg['direction_consensus']}"
                 + (" [flip-sign APPLIED — flow is antegrade]\n\n" if flip_sign else "\n\n"))
        if cal:
            fh.write("CALIBRATED COHORT TIMING (from real root peaks):\n")
            fh.write(f"  contraction frequency = {agg['frequency_cpm']:.3f} cpm "
                     f"(exhaustive over {agg['recording_s']:.0f}s recording)\n")
            fh.write(f"  ISI mean±SD {agg['isi_mean_s']:.1f}±{agg['isi_sd_s']:.1f} s  (CV {agg['isi_cv']:.0f}%)\n")
            fh.write(f"  coverage {100*agg['coverage']:.0f}% of the recording spanned\n")
            fh.write(f"  FPF surrogate = EF_fwhm × freq = {agg['fpf_surrogate']:.3f} /min\n")
            if agg.get('doublets'):
                for a2, b2, iv in agg['doublets']:
                    fh.write(f"  DOUBLET: {a2}+{b2} only {iv:.1f}s apart (<10s)\n")
            fh.write("\nPUMP RUN-DOWN (Theil–Sen vs root time, 95% bootstrap CI):\n")
            for nm, k in (('speed µm/s', 'speed'), ('EF_fwhm', 'ef'), ('amplitude', 'amp')):
                fh.write(f"  {nm:12s} slope {agg[f'rundown_{k}_slope']:+.4f} /s  "
                         f"95%CI[{agg[f'rundown_{k}_ci_lo']:+.4f},{agg[f'rundown_{k}_ci_hi']:+.4f}]  "
                         f"{'EXCLUDES 0 (significant decline)' if agg[f'rundown_{k}_excludes_zero'] else 'includes 0 (trend, not significant)'}\n")
        else:
            fh.write("[F2] " + agg['note'] + "\n")
        fh.write("\nCAVEATS:\n")
        fh.write("  [sign] speed=|velocity|; signed_v is antegrade(+) after flip.\n")
        fh.write("  [F3] centerline consistency: "
                 + (f"WARNING — differ (pts {sorted(npts)}, lens {sorted(set(lens))} µm): different vessel paths.\n"
                    if cl_bad else "OK — consistent.\n"))
        fh.write("  [F4] EF_fwhm / FPF are FWHM-derived SURROGATES; for volumetric EF re-analyse the "
                 "recon .mat with a diameter kymograph.\n")
        fh.write("\n[F5] BOLUS START/END POSITIONS along the FG line (s=0 at the first FG vertex):\n")
        if not ps['comparable']:
            fh.write("  NOT COMPARABLE — pulses carry different FG lines (see [F3]); s_* values below are\n"
                     "  in per-pulse coordinates. Re-analyse all pulses with ONE FG line before interpreting.\n")
        fh.write(f"  FG line length {ps['centerline_len_um']:.0f} µm ; window ±{ps['window_um']:.0f} µm\n")
        for tg, s0, s1 in zip(ps['tags'], ps['s_start'], ps['s_end']):
            fh.write(f"  {tg:>12s}  start {s0:7.1f}  ->  end {s1:7.1f} µm\n")
        fh.write(f"  end  : median {ps['end_median']:.0f}  mean±SD {ps['end_mean']:.0f}±{ps['end_sd']:.0f} µm ; "
                 f"{ps['end_within_window']}/{ps['end_n']} within ±{ps['window_um']:.0f} µm of median\n")
        fh.write(f"  start: median {ps['start_median']:.0f}  mean±SD {ps['start_mean']:.0f}±{ps['start_sd']:.0f} µm\n")
        fh.write(f"  end clusters (gap>{ps['window_um']:.0f} µm splits): {ps['n_end_clusters']} -> "
                 + "; ".join(f"n={c['n']} @ {c['center']:.0f} µm [{c['lo']:.0f}-{c['hi']:.0f}]" for c in ps['end_clusters'])
                 + "\n")
        fh.write("  CAVEAT: s_end is where the tracked leading edge was LAST detected. It is bounded by the\n"
                 "  amplitude gate / SNR and by the clip length, so a recurring end position is SUGGESTIVE of a\n"
                 "  fixed vessel feature (e.g. a valve) but is not evidence for one on its own.\n")

    agg['positions'] = ps
    out_agg = dict(agg); out_agg.pop('_windows', None); out_agg.pop('_window_um', None)
    out_agg['centerline_inconsistent'] = bool(cl_bad); out_agg['flip_sign'] = bool(flip_sign)
    with open(os.path.join(outdir, 'aggregate.json'), 'w') as fh:
        json.dump(out_agg, fh, indent=2,
                  default=lambda o: (float(o) if isinstance(o, np.floating)
                                     else (int(o) if isinstance(o, np.integer) else str(o))))


def _print_summary(agg):
    print(f"n = {agg['n']} (PASS {agg['n_pass']})   root_time_valid = {agg['root_time_valid']}")
    print(f"speed = {agg['speed_mean']:.0f} ± {agg['speed_sd']:.0f} µm/s "
          f"(95%CI [{agg['speed_ci_lo']:.0f},{agg['speed_ci_hi']:.0f}], CV {agg['speed_cv_pct']:.0f}%)  "
          f"{agg['direction_consensus']}")
    print(f"EF_fwhm = {agg['ef_fwhm_mean']:.2f} ± {agg['ef_fwhm_sd']:.2f}  DI = {agg['di_mean']:.2f}")
    if agg.get('root_time_valid'):
        print(f"frequency = {agg['frequency_cpm']:.2f} cpm   ISI = {agg['isi_mean_s']:.0f} ± {agg['isi_sd_s']:.0f} s "
              f"(CV {agg['isi_cv']:.0f}%)   coverage {100*agg['coverage']:.0f}%")
        print(f"run-down speed slope = {agg['rundown_speed_slope']:+.3f} µm/s per s "
              f"95%CI[{agg['rundown_speed_ci_lo']:+.3f},{agg['rundown_speed_ci_hi']:+.3f}] "
              f"({'significant' if agg['rundown_speed_excludes_zero'] else 'ns'})")
        if agg.get('doublets'):
            print("doublets: " + "; ".join(f"{a}+{b} {iv:.1f}s" for a, b, iv in agg['doublets']))
    else:
        print("frequency: WITHHELD (pass --framemap to calibrate)")


def _print_positions(agg):
    ps = agg.get('positions') or {}
    if not ps:
        return
    print("\nBOLUS END POSITIONS along FG line: "
          + ("" if ps.get('comparable') else "[NOT COMPARABLE — different FG lines] ")
          + f"median {ps.get('end_median', float('nan')):.0f} µm, SD {ps.get('end_sd', float('nan')):.0f} µm, "
          f"{ps.get('end_within_window')}/{ps.get('end_n')} within ±{ps.get('window_um', 0):.0f} µm, "
          f"{ps.get('n_end_clusters')} cluster(s)  -> pulse_fig3_positions.png, bolus_positions.csv")


def run(paths, outdir, flip_sign=True, framemap=None, fps=None, recover=True, window_um=50.0):
    rows = load_pulse_files(paths)
    if not rows:
        print("[error] no valid recon_pulse_v1 intervals found.", file=sys.stderr); return None
    root_T = None; win = []
    fm_info = None
    if framemap:
        fm_info = load_framemap(framemap)
        if fps is None:
            fps = fm_info['fps']
        rows = apply_framemap(rows, fm_info, fps or 30.0)
        root_T = fm_info['root_T']; win = fm_info['windows']
        print(f"[framemap] {os.path.basename(framemap)}: composed {len(rows)} pulses -> ROOT "
              f"(root.T={root_T}, fps={fps}); root_time_valid=True")
    elif recover:
        rows, n_fixed, rT = recover_root_from_filenames(rows, SOURCE_FILES_DEFAULT)
        if n_fixed:
            root_T = rT
            print(f"[recover] recovered ROOT frames from filenames for {n_fixed}/{len(rows)} clips "
                  f"(root.T={root_T})")
    fps = fps or (rows[0]['fps'] if rows else 30.0)
    os.makedirs(outdir, exist_ok=True)
    table = per_pulse_table(rows, flip_sign=flip_sign, fps=fps)
    agg = aggregate(table, root_T=root_T, fps=fps)
    agg['_windows'] = win; agg['_window_um'] = float(window_um)
    try:
        figures(table, agg, outdir)
    except Exception as e:
        print(f"[warn] figure generation failed: {e}", file=sys.stderr)
    write_tables(table, agg, outdir, flip_sign=flip_sign, window_um=window_um)
    _print_summary(agg)
    _print_positions(agg)
    print(f"\nOutputs -> {os.path.abspath(outdir)}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
def _synth_doc(v_um_s, direction, tag, n=40, w_min=20.0, w_max=40.0, root_time_valid=False):
    t = np.arange(n); fps = 30.0
    s = (v_um_s / fps) * t + 5.0
    w = np.linspace(w_max, w_min, n)                # FWHM shrinks then... (known w_min/w_max)
    w[: n // 2] = np.linspace(w_min, w_max, n // 2)
    it = dict(idx=0, local_t0=0, local_t1=n, present_local=t.tolist(), present_root=t.tolist(),
              root_t0=0, root_t1=n, t_peak_local=n // 2, t_peak_root=n // 2, root_file=0,
              v_um_s=float(v_um_s), se=8.0, r2=0.9, n_used=n, n_kept=n, direction=direction,
              transport_um=float(s[-1] - s[0]), s_star_um=s.tolist(),
              amp_at_star=(np.ones(n) * 100).tolist(), fwhm_um=w.tolist(),
              arclen_um=np.linspace(0, 200, 80).tolist())
    return dict(schema='recon_pulse_v1', centerline=[[0, 0], [10, 0], [20, 0]],
                direction_convention='antegrade=+ along prox->dist', fps=fps, px_um=4.0,
                diff_mode='consecutive', root_time_valid=root_time_valid,
                frame_map=list(range(n)), root=dict(T=n, boundaries=[0, n]), intervals=[it], n_intervals=1)


def selftest():
    _utf8_stdout(); ok = True

    def chk(c, m):
        nonlocal ok; ok = ok and bool(c); print(f"  [{'OK' if c else 'FAIL'}] {m}")

    import tempfile
    tmp = tempfile.mkdtemp()
    # 4 pulses, speed declines with time (run-down), retrograde (to be flipped -> antegrade)
    specs = [(1000.0, 'A'), (700.0, 'B'), (400.0, 'C'), (200.0, 'D')]
    paths = []
    for v, tg in specs:
        d = _synth_doc(-v, 'retrograde', tg, root_time_valid=False)
        p = os.path.join(tmp, f'{tg}_pulseresults.json'); json.dump(d, open(p, 'w')); paths.append(p)
    # known frame_map: local i -> root i+1000, so peak (n//2=20) -> 1020, spaced by 300 frames/pulse
    # give each file a DIFFERENT map by offsetting; emulate via a single composed map is per-file here,
    # so test apply_framemap on a synthetic map instead:
    rows = load_pulse_files(paths)
    fm = np.arange(40, dtype=np.int64) + 1000
    fmi = dict(frame_map=fm, root_T=5000, fps=30.0, px_um=4.0, windows=[(1000, 1040)])
    rows = apply_framemap(rows, fmi, 30.0)
    chk(all(r['root_frame'] == 1020 for r in rows), "framemap composes local peak 20 -> root 1020")
    chk(all(r['root_time_valid'] for r in rows), "root_time_valid=True after compose")
    tbl = per_pulse_table(rows, flip_sign=True, fps=30.0)
    chk(all(abs(tbl[i]['speed_um_s'] - specs[i][0]) < 1e-6 for i in range(4)), "|speed| recovered")
    chk(all(r['direction'] == 'antegrade' for r in tbl), "flip -> antegrade")
    chk(all(0.0 <= r['EF_fwhm'] <= 1.0 for r in tbl), "EF_fwhm in [0,1]")
    # run-down: give distinct root times so a slope is defined
    for i, r in enumerate(tbl):
        r['root_time_s'] = float(i * 100)
    s, lo, hi, ex = theilsen_ci([r['root_time_s'] for r in tbl], [r['speed_um_s'] for r in tbl])
    chk(np.isfinite(s) and s < 0, f"run-down slope negative (recovered sign): {s:+.2f} /s")
    # aggregate timing withheld without framemap
    rows2 = load_pulse_files(paths)
    agg_nofm = aggregate(per_pulse_table(rows2, flip_sign=True, fps=30.0), root_T=None)
    chk(agg_nofm['frequency_withheld'] is True, "frequency withheld when no framemap")
    # absolute positions: synthetic s = v/fps*t + 5 over t=0..39 (v negative before flip)
    exp_end = [5.0 - v / 30.0 * 39 for v, _ in specs]
    chk(all(abs(r['s_start_um'] - 5.0) < 1e-6 for r in tbl), "s_start_um = first-frame s* (5.0)")
    chk(all(abs(tbl[i]['s_end_um'] - exp_end[i]) < 1e-6 for i in range(4)), "s_end_um = last-frame s*")
    chk(all(abs(r['s_extent_um'] - abs(r['s_end_um'] - r['s_start_um'])) < 1e-6 for r in tbl),
        "s_extent_um = |s_end - s_start| for monotonic trajectories")
    ps = position_summary(tbl, window_um=50.0)
    chk(ps['comparable'] is True, "centerline consistency detected (identical FG lines)")
    chk(ps['n_end_clusters'] == len(_clusters_1d(exp_end, 50.0)), "end clusters match greedy 1-D split")
    chk(_bolus_positions([2, 0, 1], [30.0, 10.0, 20.0])['s_end'] == 30.0, "positions are time-ordered by present")
    chk(np.isnan(_bolus_positions([], [])['s_end']), "empty trajectory -> NaN positions")
    # end-to-end: files written with the new columns
    outd = os.path.join(tmp, 'out'); run(paths, outd, flip_sign=True, framemap=None, fps=30.0, recover=False)
    hdr = open(os.path.join(outd, 'per_pulse.csv')).readline().strip().split(',')
    chk(all(c in hdr for c in ('s_start_um', 's_end_um', 's_extent_um')), "per_pulse.csv has position columns")
    chk(os.path.exists(os.path.join(outd, 'bolus_positions.csv')), "bolus_positions.csv written")
    chk(os.path.exists(os.path.join(outd, 'pulse_fig3_positions.png')), "pulse_fig3_positions.png written")
    chk('[F5] BOLUS START/END POSITIONS' in open(os.path.join(outd, 'summary.txt'), encoding='utf-8').read(),
        "summary.txt has the [F5] section")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrated aggregator for recon_pulse_v1 files.")
    ap.add_argument("inputs", nargs="*", help="recon_pulse_v1 JSON files (globs allowed)")
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--framemap", default=None, help="ver2 framemap JSON: composes local->ROOT time")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--px", type=float, default=None, help="informational px_um override")
    try:
        BA = argparse.BooleanOptionalAction
        ap.add_argument("--flip-sign", action=BA, default=True, help="negate v & swap direction (default ON: flow is antegrade)")
        ap.add_argument("--recover-clip-names", action=BA, default=True,
                        help="recover ROOT frames from '<tag>_..._truncated_<lo>_<hi>' clip filenames (default ON)")
    except AttributeError:
        ap.add_argument("--flip-sign", dest="flip_sign", action="store_true", default=True)
        ap.add_argument("--no-flip-sign", dest="flip_sign", action="store_false")
        ap.add_argument("--recover-clip-names", dest="recover_clip_names", action="store_true", default=True)
        ap.add_argument("--no-recover-clip-names", dest="recover_clip_names", action="store_false")
    ap.add_argument("--end-window-um", type=float, default=50.0,
                    help="±window (µm) for counting bolus END positions near the median and for cluster splitting (default 50)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    _utf8_stdout()
    if args.selftest:
        return selftest()
    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not paths:
        print("[error] no input files (pass *_pulseresults.json or --selftest).", file=sys.stderr); return 2
    agg = run(paths, args.out, flip_sign=args.flip_sign, framemap=args.framemap, fps=args.fps,
              recover=args.recover_clip_names, window_um=args.end_window_um)
    return 0 if agg is not None else 3


if __name__ == "__main__":
    sys.exit(main())