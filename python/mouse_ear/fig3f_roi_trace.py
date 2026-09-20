#!/usr/bin/env python3
"""
fig3f_roi_trace.py — Fig. 3f (mouse-ear ROI fluorescence traces) 재생성 + Source Data CSV
======================================================================================

배경
----
현재 제출용 SourceData_Fig3f_ROI{1,2}.csv 는 논문 그림을 다시 디지타이즈한 값이다
(|Δy| 가 고정 step 의 정수배, dt ≈ 0.0364/0.0360 s = 디지타이저 픽셀 격자, 두 ROI 의 dt 가 다름).
이 모듈은 ROI_analysis_mouse_ear.py 의 "Show Combined Signals" 와 동일한 계산으로
원본 TIFF 에서 프레임별 ROI 트레이스를 다시 뽑아, fps(=20 volumes/s) 기준의 진짜 시간축을 가진
Source Data 와 Nature 스타일 그림(PDF/SVG/PNG)을 만든다.

두 가지 사용법
--------------
(A) GUI 안에서 (숫자가 GUI 의 Show Combined Signals 와 완전히 동일):
        ROI 세 개(foreground, background 1/2) 와 marker 를 잡은 뒤
        >>> import fig3f_roi_trace as f3
        >>> f3.export_from_gui(app, out_dir='fig3f_out', fps=20.0, label='ROI_blue_-300um')
    ROI_analysis_mouse_ear_v2.py 에는 이 호출이 "Export Fig 3f (PDF+CSV)" 버튼으로 들어가 있다.
    ROI 좌표는 같은 폴더에 <label>_roi.json 으로 저장되어 (B) 로 재현 가능하다.

(B) 헤드리스 (기탁/재현용):
        python fig3f_roi_trace.py --tif ear_dynamic.tif --fps 20 \
               --roi ROI_blue_-300um_roi.json --label ROI_blue_-300um \
               --roi ROI_red_+240um_roi.json  --label ROI_red_+240um  --out fig3f_out
    --fps 는 필수 (기본값 없음). GUI 의 기본 30 fps 를 그대로 쓰는 실수를 막기 위함.

계산 (GUI extract_segmented_signals 와 동일)
--------------------------------------------
foreground polyline 을 n_segments 등분한 각 중심점에 지름 fg_width 의 원형 ROI 를 놓고 프레임별 평균 강도를
구한다 (중심점은 GUI 와 같이 int() 절삭). 두 background polyline 도 같은 방식으로 구해 segment 별로 평균하고
segment 축으로 gaussian_filter1d(σ=1) 을 적용한다. 선택적으로 outlier 제거(zscore/iqr/modified_z/morphological)
와 median filter 를 GUI 와 같은 순서로 적용한다. 최종 트레이스:
    fg(t)   = segment 평균 foreground 강도   ← Fig. 3f 에 그려진 값 (raw a.u.)
    bg(t)   = segment 평균 background 강도
    diff(t) = fg − bg
verify_against_gui(app) 로 이 포트와 GUI 메서드의 결과가 동일한지(max |Δ| = 0) 확인할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("cv2 (opencv-python) 가 필요합니다: pip install opencv-python") from e
from scipy import ndimage
from scipy.signal import medfilt
from scipy.interpolate import interp1d

__version__ = "1.1.0 (2026-09-18)"   # breathing exclusion applied to the export + selectable trace style

# --------------------------------------------------------------------------------------
# Nature style
# --------------------------------------------------------------------------------------
OKABE_ITO = {
    'blue': '#0072B2', 'vermilion': '#D55E00', 'green': '#009E73', 'orange': '#E69F00',
    'sky': '#56B4E9', 'purple': '#CC79A7', 'yellow': '#F0E442', 'black': '#000000',
}
# Okabe-Ito 에는 진짜 빨강이 없다 (vermilion 은 주황 쪽). Fig. 3f 오른쪽 패널처럼
# red 트레이스가 필요할 때 쓰도록 몇 가지를 더한다. 이름 -> hex 조회는 COLOR_NAMES 로.
EXTRA_COLORS = {
    'red': '#D62728', 'crimson': '#C0392B', 'dark red': '#A50F15', 'darkred': '#A50F15',
}
COLOR_NAMES = {**OKABE_ITO, **EXTRA_COLORS}
MM = 1 / 25.4


def set_nature_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'lines.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150, 'savefig.dpi': 600,
    })
    return plt


# --------------------------------------------------------------------------------------
# ROI definition (GUI 상태를 그대로 담는 컨테이너)
# --------------------------------------------------------------------------------------
@dataclass
class ROISet:
    foreground: List[List[int]]                 # polyline [[x, y], ...] (image pixel coords, int)
    background_1: List[List[int]]
    background_2: List[List[int]]
    fg_width: int = 5                           # circular sampling ROI diameter (px)
    bg_width: int = 5
    n_segments: int = 10                        # background_segments (fg/bg 공통)
    start_frame: Optional[int] = None           # analysis marker (None → 0)
    end_frame: Optional[int] = None             # analysis marker (None → last)
    excluded_frames: List[int] = field(default_factory=list)  # breathing-excluded (default: not used)
    # 필터 (GUI 기본값: 모두 off)
    remove_outliers: bool = False
    outlier_method: str = 'zscore'              # 'zscore' | 'iqr' | 'modified_z' | 'morphological'
    outlier_threshold: float = 3.0
    morph_size: int = 5
    apply_median: bool = False
    median_window: int = 5
    # 메타 (계산에 쓰이지 않음, provenance 용)
    pixel_size_um: Optional[float] = None
    label: str = 'ROI'
    note: str = ''

    # ---- (de)serialisation ----
    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'schema': 'fig3f_roi_trace.ROISet/1', **asdict(self)}, fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ROISet':
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        d.pop('schema', None)
        return cls(**d)

    @classmethod
    def from_gui(cls, app, label: str = 'ROI') -> 'ROISet':
        """TIFAnalyzer 인스턴스의 현재 ROI/필터 상태를 복사."""
        if app.foreground_roi is None or app.background_roi_1 is None or app.background_roi_2 is None:
            raise ValueError("foreground / background 1 / background 2 ROI 를 모두 먼저 그려야 합니다")
        return cls(
            foreground=np.asarray(app.foreground_roi, dtype=int).tolist(),
            background_1=np.asarray(app.background_roi_1, dtype=int).tolist(),
            background_2=np.asarray(app.background_roi_2, dtype=int).tolist(),
            fg_width=int(app.foreground_line_width),
            bg_width=int(app.background_line_width),
            n_segments=int(app.background_segments),
            start_frame=app.start_marker, end_frame=app.end_marker,
            excluded_frames=sorted(int(i) for i in getattr(app, 'excluded_frames', set())),
            remove_outliers=bool(app.remove_outliers),
            outlier_method=str(app.outlier_method),
            outlier_threshold=float(app.outlier_threshold),
            morph_size=int(getattr(app, 'morph_size', 5)),
            apply_median=bool(app.apply_filter),
            median_window=int(app.median_filter_window),
            pixel_size_um=float(getattr(app, 'pixel_size', np.nan)),
            label=label,
        )


# --------------------------------------------------------------------------------------
# Stack loading (GUI load_tif_stack 과 동일: PIL 다중 프레임 순회)
# --------------------------------------------------------------------------------------
def load_stack(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.array(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    stack = np.array(frames)
    if stack.ndim < 3:
        raise ValueError(f"{path}: 단일 프레임입니다 (shape {stack.shape})")
    return stack


# --------------------------------------------------------------------------------------
# Faithful port of TIFAnalyzer.extract_segmented_signals
# --------------------------------------------------------------------------------------
def _line_length(points) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += np.sqrt(dx * dx + dy * dy)
    return total


def _point_along(points, target_distance):
    """GUI 와 동일: 누적 거리로 위치를 찾고 int() 로 절삭. 폴리라인 밖이면 None."""
    current_distance = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        segment_length = np.sqrt(dx * dx + dy * dy)
        if current_distance + segment_length >= target_distance:
            t = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
            return [int(points[i][0] + t * dx), int(points[i][1] + t * dy)]
        current_distance += segment_length
    return None


def _circle_mean(frame, center, width):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, tuple(center), width // 2, 255, -1)
    px = frame[mask > 0]
    return np.mean(px) if len(px) > 0 else 0


def _remove_outliers(signal, method, threshold):
    """GUI remove_outliers_from_signal (두 번째 정의, 실제로 사용되는 것) 의 포트."""
    if len(signal) < 3:
        return signal, np.zeros(len(signal), dtype=bool)
    outlier_mask = np.zeros(len(signal), dtype=bool)
    if method == 'zscore':
        mean, std = np.mean(signal), np.std(signal)
        if std > 0:
            outlier_mask = np.abs((signal - mean) / std) > threshold
    elif method == 'iqr':
        q1, q3 = np.percentile(signal, 25), np.percentile(signal, 75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (signal < q1 - threshold * iqr) | (signal > q3 + threshold * iqr)
    elif method == 'modified_z':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad > 0:
            outlier_mask = np.abs(0.6745 * (signal - median) / mad) > threshold
    if not np.any(outlier_mask):
        return signal.copy(), outlier_mask
    cleaned = signal.copy()
    valid = np.where(~outlier_mask)[0]
    bad = np.where(outlier_mask)[0]
    if len(valid) < 3:
        return signal.copy(), outlier_mask
    f = interp1d(valid, signal[valid], kind='linear', bounds_error=False, fill_value='extrapolate')
    cleaned[outlier_mask] = f(bad)
    return cleaned, outlier_mask


def _morph_filter(signal, size):
    from scipy.ndimage import grey_opening, grey_closing
    return grey_closing(grey_opening(signal, size=size), size=size)


def frame_range_of(roi: ROISet, n_total: int, exclude_breathing: bool = False) -> List[int]:
    """GUI get_analysis_frame_range + range(start, end+1) 와 동일.

    exclude_breathing=True 면 roi.excluded_frames 를 떨어뜨린다. GUI 의
    TIFAnalyzer._included_frames() 와 같은 규칙이고, 전부 제외되어 비면 GUI 와
    똑같이 전체 구간으로 되돌린다(조용히 빈 배열을 내놓지 않는다).
    """
    s = 0 if roi.start_frame is None else int(roi.start_frame)
    e = n_total - 1 if roi.end_frame is None else int(roi.end_frame)
    s = max(0, min(s, n_total - 1))
    e = max(s, min(e, n_total - 1))
    full = list(range(s, e + 1))
    if not exclude_breathing or not roi.excluded_frames:
        return full
    ex = {int(i) for i in roi.excluded_frames}
    keep = [i for i in full if i not in ex]
    return keep if keep else full


def n_excluded_in_range(roi: ROISet, n_total: int) -> int:
    """분석 구간 안에서 실제로 떨어지는 breathing 프레임 수."""
    return len(frame_range_of(roi, n_total, False)) - len(frame_range_of(roi, n_total, True))


def extract_segmented_signals(stack: np.ndarray, roi: ROISet, exclude_breathing: bool = False):
    """Returns (fg_array, bg_array, frames): arrays shape (n_segments, n_frames).

    exclude_breathing=True 면 호흡/움직임으로 플래그된 프레임을 시간축에서 제거한다.
    """
    fg_pts, bg1_pts, bg2_pts = roi.foreground, roi.background_1, roi.background_2
    fg_len, bg1_len, bg2_len = _line_length(fg_pts), _line_length(bg1_pts), _line_length(bg2_pts)
    N = roi.n_segments
    frames = frame_range_of(roi, len(stack), exclude_breathing)

    fg_all, bg_all = [], []
    for fi in frames:
        frame = stack[fi]
        fg_seg = []
        for k in range(N):
            c = _point_along(fg_pts, (k + 0.5) * fg_len / N)
            fg_seg.append(_circle_mean(frame, c, roi.fg_width) if c is not None else 0)
        bg_seg = []
        for k in range(N):
            vals = []
            c1 = _point_along(bg1_pts, (k + 0.5) * bg1_len / N)
            if c1 is not None:
                vals.append(_circle_mean(frame, c1, roi.bg_width))
            c2 = _point_along(bg2_pts, (k + 0.5) * bg2_len / N)
            if c2 is not None:
                vals.append(_circle_mean(frame, c2, roi.bg_width))
            bg_seg.append(np.mean(vals) if vals else 0)
        if len(bg_seg) > 2:
            bg_seg = ndimage.gaussian_filter1d(bg_seg, sigma=1.0)
        fg_all.append(fg_seg)
        bg_all.append(bg_seg)

    fg = np.array(fg_all).T
    bg = np.array(bg_all).T

    if roi.remove_outliers:
        for i in range(fg.shape[0]):
            if roi.outlier_method == 'morphological':
                fg[i] = _morph_filter(fg[i], roi.morph_size)
                bg[i] = _morph_filter(bg[i], roi.morph_size)
            else:
                fg[i], _ = _remove_outliers(fg[i], roi.outlier_method, roi.outlier_threshold)
                bg[i], _ = _remove_outliers(bg[i], roi.outlier_method, roi.outlier_threshold)
    if roi.apply_median and roi.median_window > 1:
        for i in range(fg.shape[0]):
            fg[i] = medfilt(fg[i], kernel_size=roi.median_window)
            bg[i] = medfilt(bg[i], kernel_size=roi.median_window)
    return fg, bg, np.asarray(frames, dtype=int)


def _break_gaps(t, y, frames):
    """제외된 프레임이 만든 구멍에 NaN 을 끼워 선이 이어지지 않게 한다."""
    f = np.asarray(frames, dtype=int)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.diff(f)
    if not np.any(d > 1):
        return t, y
    ts, ys = [], []
    for i in range(len(f)):
        ts.append(t[i])
        ys.append(y[i])
        if i < len(f) - 1 and d[i] > 1:
            ts.append(np.nan)
            ys.append(np.nan)
    return np.asarray(ts), np.asarray(ys)


def _excluded_spans(frames, time_s):
    """제외 구간마다 (t_start, t_end). 음영 표시용."""
    f = np.asarray(frames, dtype=int)
    out = []
    for i in range(len(f) - 1):
        if f[i + 1] - f[i] > 1:
            out.append((float(time_s[i]), float(time_s[i + 1])))
    return out


def combined_traces(fg_array, bg_array, frames, fps: float,
                    t0_frame: Optional[int] = None) -> dict:
    """GUI show_combined_signals 의 세 트레이스 + 시간축.

    t0_frame 을 주면 그 프레임이 t=0 이 된다 (보통 analysis start marker).
    None 이면 기존처럼 녹화 첫 프레임 기준 절대 시각.
    """
    fg = np.mean(fg_array, axis=0)
    bg = np.mean(bg_array, axis=0)
    frames = np.asarray(frames)
    t0 = 0.0 if t0_frame is None else float(t0_frame)
    return {'frame': frames, 'time_s': (frames - t0) / float(fps),
            'time_s_absolute': frames / float(fps),
            't0_frame': (None if t0_frame is None else int(t0_frame)),
            'fg': fg, 'bg': bg, 'diff': fg - bg}


# --------------------------------------------------------------------------------------
# Smoothing / fitting
#
# 호흡 프레임을 빼고 나면 표본이 시간축에서 '불규칙 간격'이 된다. Savitzky-Golay 나
# 이동평균을 인덱스 기준으로 돌리면 구멍을 지나면서 실제보다 짧은 시간창을 쓰게 되어
# 틀린다. 아래 함수들은 모두 '초' 단위 시간창으로 동작한다.
# --------------------------------------------------------------------------------------
SMOOTH_METHODS = ('none', 'movavg', 'lowess', 'spline', 'savgol')


def _tricube(u):
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u ** 3) ** 3


def _median_dt(t):
    d = np.diff(np.asarray(t, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def smooth_trace(t, y, method: str = 'lowess', window_s: float = 1.0, order: int = 2,
                 spline_s: Optional[float] = None, n_out: Optional[int] = 600,
                 max_gap_s: Optional[float] = None):
    """불규칙 간격에서도 맞는 스무딩. (t_out, y_out) 반환.

    method  : 'none' | 'movavg' | 'lowess' | 'spline' | 'savgol'
    window_s: 평활 시간창(초). lowess 는 이 값이 대역폭.
    n_out   : 출력 격자 점 수. None 이면 입력 t 위에서 평가(= CSV 열용).
    max_gap_s: 실제 표본에서 이보다 멀리 떨어진 출력점은 NaN 으로 비운다.
               제외 구간을 곡선이 가로지르며 '없는 데이터'를 그리는 것을 막는다.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if method in (None, 'none', '') or len(t) < 4:
        return t, y
    to = t.copy() if n_out is None else np.linspace(t[0], t[-1], int(n_out))
    w = float(window_s)
    if w <= 0:
        return t, y

    if method == 'movavg':
        yo = np.full(len(to), np.nan)
        for i, x in enumerate(to):
            m = np.abs(t - x) <= w / 2.0
            if m.any():
                yo[i] = y[m].mean()
    elif method == 'lowess':
        yo = np.full(len(to), np.nan)
        for i, x in enumerate(to):
            d = np.abs(t - x)
            m = d <= w
            if m.sum() < 3:
                m = np.argsort(d)[:min(len(t), 5)]
            dd = d[m]
            scale = dd.max() if dd.max() > 0 else 1.0
            wt = _tricube(dd / max(w, scale) if dd.max() <= w else dd / scale)
            sw = wt.sum()
            if sw <= 0:
                continue
            tt, yy = t[m], y[m]
            mx = (wt * tt).sum() / sw
            my = (wt * yy).sum() / sw
            sxx = (wt * (tt - mx) ** 2).sum()
            sxy = (wt * (tt - mx) * (yy - my)).sum()
            slope = sxy / sxx if sxx > 1e-15 else 0.0
            yo[i] = my + slope * (x - mx)
    elif method == 'spline':
        from scipy.interpolate import UnivariateSpline
        sv = spline_s
        if sv is None:
            sv = len(t) * float(np.var(y)) * 1e-3
        sp = UnivariateSpline(t, y, s=sv, k=min(3, max(1, len(t) - 1)))
        yo = sp(to)
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        dt = _median_dt(t)
        tu = np.arange(t[0], t[-1] + dt * 0.5, dt)
        if len(tu) < 5:
            return t, y
        yu = np.interp(tu, t, y)
        win = int(round(w / dt)) | 1
        win = max(5, min(win, (len(tu) - 1) | 1))
        if win <= order:
            return t, y
        yo = np.interp(to, tu, savgol_filter(yu, win, order))
    else:
        raise ValueError('unknown smoothing method: %r (expected one of %s)'
                         % (method, ', '.join(SMOOTH_METHODS)))

    if max_gap_s:
        j = np.clip(np.searchsorted(t, to), 1, len(t) - 1)
        dist = np.minimum(np.abs(to - t[j - 1]), np.abs(to - t[j]))
        yo = np.where(dist > max_gap_s, np.nan, yo)
    return to, yo


def smooth_label(method: str, window_s: float, order: int = 2) -> str:
    return {'movavg': 'moving average, %.3g s time window' % window_s,
            'lowess': 'LOWESS local linear fit, %.3g s bandwidth' % window_s,
            'spline': 'smoothing spline (cubic)',
            'savgol': 'Savitzky-Golay order %d, %.3g s window (resampled to a uniform grid first)'
                      % (order, window_s),
            }.get(method, 'none')


# --------------------------------------------------------------------------------------
# Output: Source Data CSV + figure
# --------------------------------------------------------------------------------------
def write_source_csv(path: str, tr: dict, roi: ROISet, fps: float, tif_path: str, fps_source: str = '',
                     exclude_breathing: bool = False, n_excluded: int = 0,
                     smooth_info: str = '', y_smoothed=None, signal: str = 'fg'):
    prov = [
        f"# Source data - Fig. 3f, {roi.label}. Regenerated from the reconstructed volume series with fig3f_roi_trace.py v{__version__}",
        f"# tif={os.path.basename(tif_path)}; fps={fps:g} volumes/s{(' (' + fps_source + ')') if fps_source else ''}; "
        f"frames {int(tr['frame'][0])}-{int(tr['frame'][-1])} (n={len(tr['frame'])})",
        ("# time_s = (frame - %d) / fps  -- t=0 at the analysis start marker (frame %d); "
         "absolute time from the first recorded frame = frame / fps"
         % (int(tr['t0_frame']), int(tr['t0_frame']))) if tr.get('t0_frame') is not None else
        "# time_s = frame / fps  -- t=0 at the first recorded frame of the series",
        f"# foreground polyline (x,y px)={roi.foreground}; sampling circle diameter={roi.fg_width} px; segments={roi.n_segments}",
        f"# background_1={roi.background_1}; background_2={roi.background_2}; bg circle diameter={roi.bg_width} px "
        f"(bg = mean of the two lines, gaussian sigma=1 across segments)",
        f"# filters: remove_outliers={roi.remove_outliers} ({roi.outlier_method}, thr={roi.outlier_threshold}, morph={roi.morph_size}); "
        f"median={roi.apply_median} (window={roi.median_window})",
        f"# pixel_size_um={roi.pixel_size_um}; fg_mean_intensity_au = mean camera counts inside the sampling circles (raw, no normalisation) -- PLOTTED",
        "# breathing/motion frame exclusion: "
        + ("ON; %d frame(s) rejected inside the analysis range, so the frame column is "
           "non-contiguous and time_s carries real gaps" % n_excluded
           if exclude_breathing and n_excluded else
           ("ON; no frame was flagged inside this range" if exclude_breathing else
            "OFF; motion-corrupted frames are STILL PRESENT in this table")),
    ]
    if roi.note:
        prov.append(f"# note: {roi.note}")
    cols = [tr['frame'], tr['time_s'], tr['fg'], tr['bg'], tr['diff']]
    names = ['frame', 'time_s', 'fg_mean_intensity_au', 'bg_mean_intensity_au', 'fg_minus_bg_au']
    fmt = ['%d', '%.6f', '%.6f', '%.6f', '%.6f']
    if y_smoothed is not None:
        ys = np.asarray(y_smoothed, dtype=float)
        if len(ys) != len(tr['frame']):
            raise ValueError("smoothed column length %d != %d rows" % (len(ys), len(tr['frame'])))
        cols.append(ys)
        names.append('%s_smoothed_au' % signal)
        fmt.append('%.6f')
        prov.append("# %s_smoothed_au = %s, evaluated at the sampled frames. The four columns before it "
                    "are the RAW measured values -- the smoothing is a display aid, not the source data."
                    % (signal, smooth_info or 'smoothed'))
    else:
        prov.append("# no smoothing applied; the plotted curve is the raw column above")
    arr = np.column_stack(cols)
    header = '\n'.join(prov + [','.join(names)])
    # encoding 을 명시하지 않으면 np.savetxt 는 OS locale 인코딩(Windows: cp1252/cp949)으로 열어
    # 헤더의 non-ASCII 문자에서 UnicodeEncodeError 가 난다 → 항상 UTF-8
    np.savetxt(path, arr, delimiter=',', comments='', header=header,
               fmt=fmt, encoding='utf-8')


def plot_fig3f(traces: Sequence[dict], labels: Sequence[str], out_base: str,
               colors: Optional[Sequence[str]] = None, signal: str = 'fg',
               panel_w_mm: float = 42.0, panel_h_mm: float = 32.0, y_from_zero: bool = False,
               lw: float = 0.9, break_gaps: bool = True, mark_excluded: bool = False,
               excluded_color: str = '0.88', grid: bool = False,
               formats: Sequence[str] = ('pdf', 'svg', 'png'),
               smooth: Optional[dict] = None, show_raw: bool = True,
               raw_alpha: float = 0.30, raw_style: str = 'line'):
    """Fig. 3f 스타일: 패널 하나당 트레이스 하나 (왼쪽 blue ROI, 오른쪽 red ROI).

    - 축 라벨/눈금 포함 (제출된 그림에는 tick label 이 없었음)
    - y 범위는 데이터 기준 (y_from_zero=False). 제출 그림은 0 부터 시작해 패널 절반이 비어 있었음.
    - PDF/SVG(텍스트 보존) + PNG 600 dpi 저장
    """
    plt = set_nature_style()
    n = len(traces)
    if colors is None:
        colors = [COLOR_NAMES['blue'], COLOR_NAMES['red'], COLOR_NAMES['green'],
                  COLOR_NAMES['purple']][:n]
    fig_w = (panel_w_mm * n + 6 * (n - 1) + 14) * MM
    fig_h = (panel_h_mm + 12) * MM
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    for ax, tr, lab, col in zip(axes[0], traces, labels, colors):
        y = tr[signal]
        if mark_excluded:
            for t0, t1 in _excluded_spans(tr['frame'], tr['time_s']):
                ax.axvspan(t0, t1, color=excluded_color, lw=0, zorder=0)
        tp, yp = (_break_gaps(tr['time_s'], y, tr['frame']) if break_gaps
                  else (np.asarray(tr['time_s']), np.asarray(y)))
        sm = dict(smooth or {})
        meth = sm.pop('method', 'none')
        if meth and meth != 'none':
            gap = sm.pop('max_gap_s', None)
            if gap is None and break_gaps:
                gap = 3.0 * _median_dt(tr['time_s'])
            ts, ys = smooth_trace(tr['time_s'], y, method=meth, max_gap_s=gap, **sm)
            if show_raw:
                if raw_style == 'points':
                    ax.plot(tr['time_s'], y, marker='o', ls='none', ms=1.1,
                            color=col, alpha=raw_alpha, zorder=1,
                            markeredgewidth=0)
                else:
                    ax.plot(tp, yp, color=col, lw=lw * 0.6, alpha=raw_alpha,
                            solid_capstyle='round', zorder=1)
            ax.plot(ts, ys, color=col, lw=lw, solid_capstyle='round', zorder=2)
            finite = ys[np.isfinite(ys)]
            y = np.concatenate([np.asarray(y, dtype=float), finite]) if len(finite) else y
        else:
            ax.plot(tp, yp, color=col, lw=lw, solid_capstyle='round')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence intensity (a.u.)' if signal == 'fg' else
                      'Background-subtracted intensity (a.u.)')
        ax.set_title(lab, fontsize=8, pad=3)
        ax.set_xlim(tr['time_s'][0], tr['time_s'][-1])
        if y_from_zero:
            ax.set_ylim(0, np.nanmax(y) * 1.05)
        else:
            lo, hi = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            ax.set_ylim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.ticklabel_format(axis='y', style='plain')
        if grid:
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    for ext in formats:
        fig.savefig(f'{out_base}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return [f'{out_base}.{e}' for e in formats]


# --------------------------------------------------------------------------------------
# GUI integration
# --------------------------------------------------------------------------------------
GUI_DEFAULT_FPS = 30.0   # TIFAnalyzer.__init__ 의 기본값 — 확인 없이 쓰면 안 되는 값


def resolve_fps(app, fps: Optional[float]) -> float:
    """fps 인자가 없으면 GUI 값을 쓰되, GUI 기본값(30) 그대로면 거부."""
    if fps is not None:
        return float(fps)
    try:
        app.frame_rate = float(app.frame_rate_var.get())
    except Exception:
        pass
    f = float(app.frame_rate)
    if abs(f - GUI_DEFAULT_FPS) < 1e-9:
        raise ValueError("GUI 의 Frame Rate 가 기본값 30 fps 그대로입니다. 실제 취득 속도(마우스 귀 dynamic: 20 volumes/s)를 "
                         "입력하거나 export_from_gui(app, fps=20.0) 으로 명시하세요.")
    return f


def export_from_gui(app, out_dir: str = 'fig3f_out', fps: Optional[float] = None,
                    label: str = 'ROI', signal: str = 'fg', fps_source: str = '',
                    make_figure: bool = True, exclude_breathing: bool = True,
                    style: Optional[dict] = None, t0_at_start_marker: bool = True,
                    smooth: Optional[dict] = None) -> dict:
    """GUI 상태 그대로 export. 숫자는 GUI 자신의 extract_segmented_signals() 로 계산 → Show Combined Signals 와 동일.

    exclude_breathing (기본 True) 면 GUI 에서 플래그한 호흡/움직임 프레임을 먼저 떨어뜨리고
    나서 트레이스와 그림을 만든다. GUI 의 videokymograph 가 쓰는 것과 같은 프레임 집합이다.
    style 은 plot_fig3f 로 그대로 넘어가는 dict (colors, lw, break_gaps, mark_excluded,
    grid, y_from_zero, panel_w_mm, panel_h_mm, formats).
    """
    os.makedirs(out_dir, exist_ok=True)
    fps_v = resolve_fps(app, fps)
    roi = ROISet.from_gui(app, label=label)
    # GUI 계산 (동일 숫자 보장) — 제외 플래그를 GUI 메서드에도 그대로 넘긴다
    fg_arr, bg_arr = app.extract_segmented_signals(exclude_breathing=exclude_breathing)
    if fg_arr is None:
        raise ValueError("extract_segmented_signals() 가 None 을 반환 — ROI/스택 확인")
    frames = np.asarray(frame_range_of(roi, len(app.image_stack), exclude_breathing), dtype=int)
    if len(frames) != fg_arr.shape[1]:
        raise ValueError(
            "frame 축 불일치: GUI 가 %d 열을 돌려줬는데 frame_range_of 는 %d 개 인덱스를 냈습니다 "
            "(breathing 제외가 어긋남). ROI 를 다시 잡거나 'Detect Breathing Frames' 를 다시 실행하세요."
            % (fg_arr.shape[1], len(frames)))
    n_ex = n_excluded_in_range(roi, len(app.image_stack)) if exclude_breathing else 0
    # t=0 은 analysis start marker (제외되었을 수도 있으므로 '제외 전' 구간의 첫 프레임을 쓴다)
    t0 = frame_range_of(roi, len(app.image_stack), False)[0] if t0_at_start_marker else None
    tr = combined_traces(fg_arr, bg_arr, frames, fps_v, t0_frame=t0)
    tif_path = getattr(app, 'tif_file_path', 'unknown.tif')

    roi_json = os.path.join(out_dir, f'{label}_roi.json')
    roi.note = (f'exported from GUI; tif={os.path.basename(tif_path)}; '
                f'breathing_exclusion={"on" if exclude_breathing else "off"} ({n_ex} frames)')
    roi.to_json(roi_json)
    # CSV 의 스무딩 열은 '표본 시각 위에서' 평가해 행이 1:1 로 맞게 한다
    sm = dict(smooth or {})
    meth = sm.get('method', 'none')
    y_sm, sm_txt = None, ''
    if meth and meth != 'none':
        kw = {k: v for k, v in sm.items() if k not in ('method', 'max_gap_s')}
        _, y_sm = smooth_trace(tr['time_s'], tr[signal], method=meth, n_out=None,
                               max_gap_s=None, **kw)
        sm_txt = smooth_label(meth, kw.get('window_s', 1.0), kw.get('order', 2))
    csv_path = os.path.join(out_dir, f'SourceData_Fig3f_{label}.csv')
    write_source_csv(csv_path, tr, roi, fps_v, tif_path, fps_source,
                     exclude_breathing=exclude_breathing, n_excluded=n_ex,
                     smooth_info=sm_txt, y_smoothed=y_sm, signal=signal)
    out = {'roi_json': roi_json, 'csv': csv_path, 'traces': tr,
           'n_excluded': n_ex, 'exclude_breathing': bool(exclude_breathing),
           't0_frame': t0, 'smooth': sm_txt}
    if make_figure:
        st = dict(style or {})
        col = st.pop('color', None)
        if col and 'colors' not in st:
            st['colors'] = [col]
        out['figure'] = plot_fig3f([tr], [label], os.path.join(out_dir, f'Fig3f_{label}'),
                                   signal=signal, smooth=smooth, **st)
    return out


def verify_against_gui(app, atol: float = 1e-9, exclude_breathing: bool = False) -> float:
    """이 모듈의 포트와 GUI 메서드의 결과를 비교. max |Δ| 반환 (0 이어야 함)."""
    roi = ROISet.from_gui(app)
    fg_g, bg_g = app.extract_segmented_signals(exclude_breathing=exclude_breathing)
    fg_p, bg_p, _ = extract_segmented_signals(app.image_stack, roi, exclude_breathing)
    d = max(np.abs(fg_g - fg_p).max(), np.abs(bg_g - bg_p).max())
    print(f"verify_against_gui: max |GUI − port| = {d:.3g}  ({'OK' if d <= atol else 'MISMATCH'})")
    return d


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description='Fig. 3f ROI trace regeneration (headless)')
    p.add_argument('--tif', required=True, help='reconstructed dynamic series (multi-frame TIFF, one frame per volume)')
    p.add_argument('--fps', type=float, required=True, help='acquisition volume rate, volumes/s (mouse ear dynamic: 20)')
    p.add_argument('--fps-source', default='', help='provenance note for fps (e.g. "acquisition log 2025-xx-xx")')
    p.add_argument('--roi', action='append', required=True, help='ROISet json (repeatable, one per panel)')
    p.add_argument('--label', action='append', default=None, help='panel label per --roi (repeatable)')
    p.add_argument('--signal', choices=['fg', 'diff'], default='fg', help="plotted quantity: fg (raw ROI mean, Fig 3f) or diff (fg−bg)")
    p.add_argument('--out', default='fig3f_out')
    p.add_argument('--y-from-zero', action='store_true', help='force y axis to start at 0 (submitted figure did this; not recommended)')
    p.add_argument('--exclude-breathing', dest='exclude_breathing', action='store_true', default=True,
                   help='drop the breathing/motion frames listed in the ROI json (default)')
    p.add_argument('--keep-breathing', dest='exclude_breathing', action='store_false',
                   help='keep every frame, including motion-corrupted ones')
    p.add_argument('--color', action='append', default=None,
                   help='trace colour per panel: red, crimson, dark red, or an Okabe-Ito name '
                        '(blue, vermilion, green, orange, sky, purple, yellow, black), '
                        'or any matplotlib colour / #RRGGBB (repeatable)')
    p.add_argument('--lw', type=float, default=0.9, help='trace line width in points (default 0.9)')
    p.add_argument('--grid', action='store_true', help='draw a light background grid')
    p.add_argument('--mark-excluded', action='store_true',
                   help='shade the time spans where frames were rejected')
    p.add_argument('--no-break-gaps', dest='break_gaps', action='store_false', default=True,
                   help='draw a continuous line across rejected spans instead of breaking it')
    p.add_argument('--t0-start-marker', dest='t0_start', action='store_true', default=True,
                   help='put t=0 at the analysis start marker (default)')
    p.add_argument('--t0-absolute', dest='t0_start', action='store_false',
                   help='keep absolute time from the first recorded frame')
    p.add_argument('--smooth', choices=list(SMOOTH_METHODS), default='none',
                   help='smooth/fit the trace for display (source-data columns stay raw)')
    p.add_argument('--smooth-window', type=float, default=1.0,
                   help='smoothing time window / LOWESS bandwidth in SECONDS (default 1.0)')
    p.add_argument('--smooth-order', type=int, default=2, help='polynomial order for savgol (default 2)')
    p.add_argument('--no-raw', dest='show_raw', action='store_false', default=True,
                   help='hide the raw trace behind the smoothed curve')
    p.add_argument('--raw-style', choices=['line', 'points'], default='line',
                   help='how the raw trace is drawn behind the smoothed curve')
    a = p.parse_args(argv)
    colors = None
    if a.color:
        colors = [COLOR_NAMES.get(c, c) for c in a.color]
        if len(colors) != len(a.roi):
            p.error('--color 개수(%d)가 --roi 개수(%d)와 달라요' % (len(colors), len(a.roi)))

    labels = a.label or [os.path.splitext(os.path.basename(r))[0] for r in a.roi]
    if len(labels) != len(a.roi):
        p.error('--label 개수가 --roi 개수와 달라요')
    os.makedirs(a.out, exist_ok=True)

    stack = load_stack(a.tif)
    print(f"stack {stack.shape} {stack.dtype};  fps={a.fps:g}  → dt={1000/a.fps:.2f} ms, record={len(stack)/a.fps:.2f} s")
    traces = []
    for rp, lab in zip(a.roi, labels):
        roi = ROISet.from_json(rp)
        roi.label = lab
        fg, bg, frames = extract_segmented_signals(stack, roi, a.exclude_breathing)
        n_ex = n_excluded_in_range(roi, len(stack)) if a.exclude_breathing else 0
        t0 = frame_range_of(roi, len(stack), False)[0] if a.t0_start else None
        tr = combined_traces(fg, bg, frames, a.fps, t0_frame=t0)
        y_sm, sm_txt = None, ''
        if a.smooth != 'none':
            _, y_sm = smooth_trace(tr['time_s'], tr[a.signal], method=a.smooth,
                                   window_s=a.smooth_window, order=a.smooth_order,
                                   n_out=None, max_gap_s=None)
            sm_txt = smooth_label(a.smooth, a.smooth_window, a.smooth_order)
        csv_path = os.path.join(a.out, f'SourceData_Fig3f_{lab}.csv')
        write_source_csv(csv_path, tr, roi, a.fps, a.tif, a.fps_source,
                         exclude_breathing=a.exclude_breathing, n_excluded=n_ex,
                         smooth_info=sm_txt, y_smoothed=y_sm, signal=a.signal)
        i = int(np.argmax(tr['fg']))
        print(f"  {lab}: n={len(frames)} frames, {tr['time_s'][0]:.2f}-{tr['time_s'][-1]:.2f} s; "
              f"breathing excluded={n_ex if a.exclude_breathing else 'off'}; "
              f"fg baseline(first 20)={tr['fg'][:20].mean():.1f}, peak={tr['fg'][i]:.1f} at {tr['time_s'][i]:.2f} s → {csv_path}")
        traces.append(tr)
    files = plot_fig3f(traces, labels, os.path.join(a.out, 'Fig3f'), signal=a.signal,
                       y_from_zero=a.y_from_zero, colors=colors, lw=a.lw, grid=a.grid,
                       break_gaps=a.break_gaps, mark_excluded=a.mark_excluded,
                       smooth=(None if a.smooth == 'none' else
                               dict(method=a.smooth, window_s=a.smooth_window, order=a.smooth_order)),
                       show_raw=a.show_raw, raw_style=a.raw_style)
    print('figure:', *files)


if __name__ == '__main__':
    main()
