#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - depth-cycle video export - produces Supp. Videos 1-6
# environment: heart_valve_py314
# MAG_PRESETS corrected on 2026-09-20 (see CALIBRATION.md); the released videos were fixed with fix_video_overlays.py
# copied from E:\Selected data\250902_fish\Depth cycle video gui.py.bak on 2026-09-17
"""
depth_cycle_video_gui.py

Depth별 TIFF 스택 → depth-cycle MP4 제작 GUI

기능:
- 폴더에서 *_rl_<n>.tif 자동 인식
- 배율 모드: High mag / Low mag 전환 — 모드별로 z-step(Δz), µm/px,
  scale bar 길이를 따로 저장. 기본값은 High: Δz=20 µm, 11 µm/px, bar 200 µm /
  Low: Δz=100 µm, 30.3 µm/px, bar 1000 µm (Low 값은 실제 광학계에 맞게 수정)
- depth 라벨: 인덱스 i → i × Δz (µm) 선형 매핑 후 반올림
  (High mag 기본: rl_1 = 20 µm, ..., rl_20 = 400 µm)
- Crop: 미리보기에서 드래그로 ROI 지정 (모든 depth에 공통 적용)
- 모션/호흡 제거: 프레임간 |차이| 기반 모션 지표로 임계값 초과 프레임을
  자동 감지(현재 depth 또는 전체 일괄)하거나, 지표 그래프에서 구간을
  가로로 드래그해 수동 제외/복원 — 제외 프레임은 내보내기에서 건너뜀
- Pulse 모드: 동일 depth에서 반복된 pulse TIF들을 세그먼트로 이어붙여
  하나의 영상으로 내보내기 (▲▼ 버튼으로 순서 조정, 기본 cycle=1,
  라벨은 pulse 번호 표기 또는 숨김)
- 정합(De-motion): 사용자가 지정한 기준 항목(★)의 정합 ROI 중앙값 프레임을
  기준으로, gradient magnitude(라인 구조 강조) 기반 위상상관 translation
  정합 — 정합 ROI(청록 점선)는 display crop과 별개로 line structure만
  감싸도록 드래그 지정 가능. "세그먼트별"은 각 세그먼트의 중앙값만 정합
  (펄스 강도 변화에 안전), "프레임별"은 프레임 단위 드리프트까지 보정.
  "정합 검사" 버튼으로 기준(빨강)↔현재(초록) 전/후 오버레이 확인 가능
- 대비 모드:
  · 자동 (기본): 각 depth의 crop 영역 min–max 기반으로 개별 재스케일
    → 모든 depth 영상의 표현(밝기 분포)이 비슷해짐
    (하한/상한 percentile 조정 가능; hot pixel 있으면 0.1/99.9 권장)
  · 수동: 히스토그램(imhist 방식)에서 좌클릭=vmin / 우클릭=vmax
- Scale bar: 1 px = 11 µm 기준, 기본 200 µm (≈18.2 px)
- 오버레이 배치: Z 라벨/Scale bar를 crop 영역 네 corner의 밝기 분석으로
  가장 어두운 corner에 자동 배치 (콤보박스로 수동 지정도 가능)
- depth별 포함 여부 / frame 범위(포함) / cycle 수 지정
- 설정 JSON 저장/불러오기
- H.264 MP4 내보내기 (스트리밍 로드, depth 라벨 + scale bar 오버레이)

필요 패키지:
    pip install tifffile numpy imageio imageio-ffmpeg pillow matplotlib

실행:
    python depth_cycle_video_gui.py
"""

import csv
import json
import queue
import re
import threading
from pathlib import Path

import numpy as np
import tifffile
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------- 기본 설정 -----------------------------
DEFAULT_DIR = r"E:\Selected data\250902_fish\6"
# 배율 모드별 기본값 — 실제 광학계 값에 맞게 수정해서 쓰면 됨
MAG_PRESETS = {
    "high": dict(label="High mag", dz_um=20.0, um_per_px=5.616,   # 256-px frames (= 1438 um / N px; was 11.0, valid for 128-px frames)
                 bar_um=200.0),
    "low":  dict(label="Low mag", dz_um=60.0, um_per_px=17.0,    # 256-px frames (= 4352 um / N px; was dz 100, 30.3)
                 bar_um=600.0),
}
DEFAULT_PERCENTILES = (0.5, 99.8)   # 수동 모드 Auto 버튼용
AUTO_P_DEFAULT = (0.0, 100.0)       # 자동 모드 기본 = 순수 min–max
DEFAULT_CYCLES = 3
DEFAULT_FPS = 20
QUALITY = 8                     # 0~10 (libx264)
FILE_RE = re.compile(r"^(?P<prefix>.*_rl_)(?P<idx>\d+)\.(tif|tiff)$", re.IGNORECASE)

CORNERS = ("tl", "tr", "bl", "br")
POS_LABELS = {"auto": "자동", "tl": "좌상", "tr": "우상",
              "bl": "좌하", "br": "우하"}
POS_CODES = {v: k for k, v in POS_LABELS.items()}
# ---------------------------------------------------------------------


def load_stack(path):
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"{path}: (T, Y, X) 형태가 아닙니다. shape={arr.shape}")
    return arr


def stack_len(path):
    """전체 로드 없이 프레임 수만 확인"""
    with tifffile.TiffFile(str(path)) as tf:
        try:
            s = tf.series[0]
            if len(s.shape) >= 3:
                return int(s.shape[0])
        except Exception:
            pass
        return len(tf.pages)


def load_pulse_boundaries(tif_path):
    """merge_tifs.py가 만든 <stem>_boundaries.csv를 찾아
    [(start, end, pulse_no, 원본파일명), ...] 을 프레임순으로 반환.
    pulse 번호는 원본 파일명의 알파벳 정렬 순서(a, b, c, ... j)로 부여.
    demotion 출력(*_reg_*)이면 '_reg'를 뗀 stem의 csv도 함께 찾음.
    없으면 None."""
    p = Path(tif_path)
    cands = [p.with_name(p.stem + "_boundaries.csv")]
    if "_reg" in p.stem:
        cands.append(p.with_name(p.stem.replace("_reg", "")
                                 + "_boundaries.csv"))
    bp = next((c for c in cands if c.is_file()), None)
    if bp is None:
        return None
    rows = []
    try:
        with open(bp, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append((r["file"], int(r["start_frame"]),
                             int(r["end_frame"])))
    except Exception:
        return None
    if not rows:
        return None
    order = {name: no for no, name in
             enumerate(sorted(x[0] for x in rows), 1)}
    out = [(m0, m1, order[name], name) for name, m0, m1 in rows]
    out.sort(key=lambda x: x[0])
    return out


def pulse_label_at(bounds, k):
    """병합 스택의 frame k가 속한 원본 파일의 pulse 라벨 ('pulse N') 또는 None"""
    if not bounds:
        return None
    for m0, m1, no, _name in bounds:
        if m0 <= k <= m1:
            return f"pulse {no}"
    return None


def sample_values(arr, max_px=500_000):
    flat = np.asarray(arr).reshape(-1)
    step = max(1, flat.size // max_px)
    return flat[::step]


def corner_scores(img, wfrac=0.32, hfrac=0.22):
    """정규화(0~1)된 2D 이미지의 네 corner 평균 밝기 — 낮을수록 오버레이 적합"""
    H, W = img.shape
    cw = max(1, int(W * wfrac))
    ch = max(1, int(H * hfrac))
    return {"tl": float(img[:ch, :cw].mean()),
            "tr": float(img[:ch, W - cw:].mean()),
            "bl": float(img[H - ch:, :cw].mean()),
            "br": float(img[H - ch:, W - cw:].mean())}


def resolve_corners(label_pos, bar_pos, scores):
    """'auto' 항목을 어두운 corner부터 배정 (Z 라벨 우선, 서로 다른 corner 보장)"""
    order = sorted(CORNERS, key=lambda c: scores[c])
    lp, bp = label_pos, bar_pos
    if lp == "auto" and bp == "auto":
        lp = order[0]
        bp = next(c for c in order if c != lp)
    elif lp == "auto":
        lp = next(c for c in order if c != bp)
    elif bp == "auto":
        bp = next(c for c in order if c != lp)
    return lp, bp


def _hann2d(shape):
    hy = np.hanning(shape[0])
    hx = np.hanning(shape[1])
    return (hy[:, None] * hx[None, :]).astype(np.float32)


def _box3(f):
    """3×3 box 평활 (numpy only)"""
    p = np.pad(f, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def prep_reg(img):
    """정합용 전처리: 3×3 평활 → gradient magnitude(라인 구조 강조)
    → 평균 제거 → Hann window. 세그먼트간 밝기/조도 차이에 둔감."""
    f = _box3(img.astype(np.float32))
    gy, gx = np.gradient(f)
    g = np.hypot(gy, gx).astype(np.float32)
    g = g - float(g.mean())
    return g * _hann2d(g.shape)


def register_shift(ref_prep, img, up=20):
    """위상상관 정합: img를 (dy, dx)만큼 이동시키면 ref에 정렬됨.
    정수 피크 탐색 후 국소 업샘플 DFT(1/up px 격자)로 서브픽셀 정밀화."""
    g = prep_reg(img)
    F1 = np.fft.fft2(ref_prep)
    F2 = np.fft.fft2(g)
    R = F1 * np.conj(F2)
    Rw = R / np.maximum(np.abs(R), 1e-12)
    c = np.fft.ifft2(Rw).real
    H, W = c.shape
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    sy = py - H if py > H / 2 else py
    sx = px - W if px > W / 2 else px
    # 국소 업샘플 정밀화 (matrix-multiply DFT, ±1.5 px 범위)
    half = int(1.5 * up)
    ys = sy + np.arange(-half, half + 1) / up
    xs = sx + np.arange(-half, half + 1) / up
    fu = np.fft.fftfreq(H) * H
    fv = np.fft.fftfreq(W) * W
    Ky = np.exp(2j * np.pi * np.outer(fu, ys) / H)
    Kx = np.exp(2j * np.pi * np.outer(fv, xs) / W)
    C = (Ky.T @ R @ Kx).real
    iy, ix = np.unravel_index(int(np.argmax(C)), C.shape)
    return float(ys[iy]), float(xs[ix])


def shift_image(img, dy, dx):
    """bilinear 서브픽셀 이동 (가장자리 clamp). out[y, x] = img[y-dy, x-dx]"""
    H, W = img.shape
    y = np.clip(np.arange(H, dtype=np.float32) - dy, 0, H - 1)
    x = np.clip(np.arange(W, dtype=np.float32) - dx, 0, W - 1)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    f = img.astype(np.float32)
    a = f[np.ix_(y0, x0)]
    b = f[np.ix_(y0, x1)]
    c = f[np.ix_(y1, x0)]
    d = f[np.ix_(y1, x1)]
    return (a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx
            + c * wy * (1 - wx) + d * wy * wx)


def motion_metric(stack):
    """(T,Y,X) 스택의 프레임간 평균 |차이| / 강도 범위 — 호흡 등 큰 모션 지표"""
    T = stack.shape[0]
    m = np.zeros(T, np.float32)
    if T < 2:
        return m
    vals = sample_values(stack, 500_000).astype(np.float32)
    scale = max(float(np.percentile(vals, 99.0) - np.percentile(vals, 1.0)),
                1e-6)
    prev = stack[0].astype(np.float32)
    for k in range(1, T):
        curk = stack[k].astype(np.float32)
        m[k] = float(np.abs(curk - prev).mean()) / scale
        prev = curk
    m[0] = m[1]
    return m


def suggest_threshold(m):
    """robust 임계값 제안: median + 5 × 1.4826 × MAD"""
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med)))
    return med + 5.0 * 1.4826 * max(mad, 1e-9)


def get_font(size):
    for name in ("arial.ttf", "Arial.ttf", "malgun.ttf", "DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class DepthCycleGUI:
    def __init__(self, root):
        self.root = root
        root.title("Depth-Cycle Video Maker")

        self.folder = None
        self.prefix = ""
        self.files = {}     # idx -> Path
        self.meta = {}      # idx -> dict(include,start,end,cycles,vmin,vmax,n,sample)
        self.cur = None
        self.stack = None
        self.frame_i = 0
        self.crop = None    # [x0, x1, y0, y1] (x1/y1 미포함), 전 depth 공통
        self.mode_store = {k: dict(v) for k, v in MAG_PRESETS.items()}
        self._active_mode = "high"
        self.seg_names = {}
        self.pulse_bounds = {}   # str(path) -> boundaries 목록 | None
        self._last_ov_txt = None
        self._active_src = "depth"
        self.reg_ref = None      # 정합 기준 항목 key (None = 첫 포함 항목)
        self.reg_roi = None      # 정합 전용 ROI [x0, x1, y0, y1]
        self._ref_cache = None
        self._drag_target = "crop"
        self.motion_win = None
        self.m_span = None
        self._sel_range = None
        self._motion_batch_btn = None
        self._busy = False
        self._syncing = False
        self._im = None
        self._im_shape = None
        self._overlays_stale = True
        self._crop_patch = None
        self._sb_artists = []
        self._vline_lo = None
        self._vline_hi = None
        self._auto_key = None
        self._auto_val = (None, None)
        self._q = queue.Queue()
        self._exporting = False

        self._build_ui()

        if Path(DEFAULT_DIR).is_dir():
            self.scan_folder(Path(DEFAULT_DIR), quiet=True)

    # ------------------------------ UI ------------------------------
    def _build_ui(self):
        self.root.geometry("1360x820")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        # 상단 바
        top = ttk.Frame(self.root, padding=6)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(top, text="폴더 열기", command=self.open_folder).pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value="폴더를 선택하세요")
        ttk.Label(top, textvariable=self.folder_var).pack(side=tk.LEFT, padx=10)
        self.src_var = tk.StringVar(value="depth")
        ttk.Radiobutton(top, text="Depth", value="depth",
                        variable=self.src_var,
                        command=self.on_src_change).pack(side=tk.LEFT,
                                                         padx=(8, 0))
        ttk.Radiobutton(top, text="Pulse", value="pulse",
                        variable=self.src_var,
                        command=self.on_src_change).pack(side=tk.LEFT)
        self.mag_var = tk.StringVar(value="high")
        ttk.Radiobutton(top, text="High mag", value="high",
                        variable=self.mag_var,
                        command=self.on_mag_change).pack(side=tk.LEFT,
                                                         padx=(10, 0))
        ttk.Radiobutton(top, text="Low mag", value="low",
                        variable=self.mag_var,
                        command=self.on_mag_change).pack(side=tk.LEFT)
        ttk.Label(top, text="  Δz/idx(µm)").pack(side=tk.LEFT)
        self.dz_var = tk.StringVar(value=f'{MAG_PRESETS["high"]["dz_um"]:g}')
        edz = ttk.Entry(top, textvariable=self.dz_var, width=6)
        edz.pack(side=tk.LEFT, padx=2)
        edz.bind("<Return>", self.on_dz_commit)
        edz.bind("<FocusOut>", self.on_dz_commit)
        ttk.Button(top, text="설정 불러오기", command=self.load_settings).pack(side=tk.RIGHT)
        ttk.Button(top, text="설정 저장", command=self.save_settings).pack(side=tk.RIGHT, padx=4)

        # 좌: depth 리스트
        left = ttk.Frame(self.root, padding=(6, 0))
        left.grid(row=1, column=0, sticky="ns")
        ttk.Label(left, text="Depth 목록 (✓=포함)").pack(anchor="w")
        lf = ttk.Frame(left)
        lf.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lf, width=42, font=("Consolas", 9),
                                  exportselection=False)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        mvf = ttk.Frame(left)
        mvf.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(mvf, text="▲ 위로", width=8,
                   command=lambda: self.move_segment(-1)).pack(side=tk.LEFT)
        ttk.Button(mvf, text="▼ 아래로", width=8,
                   command=lambda: self.move_segment(1)).pack(side=tk.LEFT,
                                                              padx=4)
        ttk.Label(mvf, text="(Pulse 순서)",
                  foreground="gray").pack(side=tk.LEFT, padx=4)

        # 중앙: 미리보기 + crop + frame 범위/cycle
        center = ttk.Frame(self.root, padding=6)
        center.grid(row=1, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.fig_img = Figure(figsize=(5.4, 5.0), dpi=100)
        self.fig_img.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.axis("off")
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=center)
        self.canvas_img.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._make_selector()

        self.frame_scale = tk.Scale(center, from_=0, to=0, orient=tk.HORIZONTAL,
                                    label="frame", showvalue=True,
                                    command=self.on_frame_slide)
        self.frame_scale.grid(row=1, column=0, sticky="ew")

        # crop 행
        rowc = ttk.Frame(center)
        rowc.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.cropdrag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(rowc, text="Crop 드래그", variable=self.cropdrag_var,
                        command=self.on_cropdrag_toggle).pack(side=tk.LEFT)
        ttk.Label(rowc, text="  x").pack(side=tk.LEFT)
        self.cx0_var = tk.StringVar()
        self.cx1_var = tk.StringVar()
        self.cy0_var = tk.StringVar()
        self.cy1_var = tk.StringVar()
        ec = []
        for var in (self.cx0_var, self.cx1_var):
            e = ttk.Entry(rowc, textvariable=var, width=6)
            e.pack(side=tk.LEFT, padx=1)
            ec.append(e)
        ttk.Label(rowc, text=" y").pack(side=tk.LEFT)
        for var in (self.cy0_var, self.cy1_var):
            e = ttk.Entry(rowc, textvariable=var, width=6)
            e.pack(side=tk.LEFT, padx=1)
            ec.append(e)
        for e in ec:
            e.bind("<Return>", self.on_crop_entry_commit)
            e.bind("<FocusOut>", self.on_crop_entry_commit)
        ttk.Button(rowc, text="Crop 초기화",
                   command=self.clear_crop).pack(side=tk.LEFT, padx=8)

        # frame 범위 행
        rowf = ttk.Frame(center)
        rowf.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(rowf, text="시작=현재", width=9,
                   command=self.set_start_here).pack(side=tk.LEFT)
        ttk.Button(rowf, text="끝=현재", width=9,
                   command=self.set_end_here).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(rowf, text="frame 범위(포함)").pack(side=tk.LEFT)
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        e1 = ttk.Entry(rowf, textvariable=self.start_var, width=6)
        e1.pack(side=tk.LEFT, padx=2)
        ttk.Label(rowf, text="–").pack(side=tk.LEFT)
        e2 = ttk.Entry(rowf, textvariable=self.end_var, width=6)
        e2.pack(side=tk.LEFT, padx=2)
        for e, cb in ((e1, self.on_start_commit), (e2, self.on_end_commit)):
            e.bind("<Return>", cb)
            e.bind("<FocusOut>", cb)
        ttk.Button(rowf, text="모션/호흡 제거…",
                   command=self.open_motion_dialog).pack(side=tk.RIGHT)

        rowg = ttk.Frame(center)
        rowg.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self.include_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rowg, text="내보내기에 포함", variable=self.include_var,
                        command=self.on_include_toggle).pack(side=tk.LEFT)
        ttk.Label(rowg, text="   cycle").pack(side=tk.LEFT)
        self.cycles_var = tk.StringVar(value=str(DEFAULT_CYCLES))
        sp = ttk.Spinbox(rowg, from_=1, to=99, width=4,
                         textvariable=self.cycles_var,
                         command=self.on_cycles_change)
        sp.pack(side=tk.LEFT, padx=4)
        sp.bind("<Return>", self.on_cycles_change)
        sp.bind("<FocusOut>", self.on_cycles_change)
        ttk.Button(rowg, text="frame범위·cycle 전체 적용",
                   command=self.apply_range_all).pack(side=tk.RIGHT)

        # 우: 히스토그램 + 대비 + scale bar
        right = ttk.Frame(self.root, padding=6)
        right.grid(row=1, column=2, sticky="ns")
        self.fig_hist = Figure(figsize=(4.4, 3.0), dpi=100)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=right)
        self.canvas_hist.get_tk_widget().pack()
        self.canvas_hist.mpl_connect("button_press_event", self.on_hist_click)

        modef = ttk.LabelFrame(right, text="대비 모드", padding=4)
        modef.pack(fill=tk.X, pady=(6, 0))
        self.norm_mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(modef, text="자동: crop 영역 min–max (depth별 재스케일)",
                        variable=self.norm_mode_var, value="auto",
                        command=self.on_norm_mode_change).pack(anchor="w")
        ttk.Radiobutton(modef, text="수동: 히스토그램 vmin/vmax",
                        variable=self.norm_mode_var, value="manual",
                        command=self.on_norm_mode_change).pack(anchor="w")
        rowp = ttk.Frame(modef)
        rowp.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(rowp, text="자동 하한/상한 %").pack(side=tk.LEFT)
        self.plo_var = tk.StringVar(value=f"{AUTO_P_DEFAULT[0]:g}")
        self.phi_var = tk.StringVar(value=f"{AUTO_P_DEFAULT[1]:g}")
        ep1 = ttk.Entry(rowp, textvariable=self.plo_var, width=6)
        ep1.pack(side=tk.LEFT, padx=2)
        ttk.Label(rowp, text="–").pack(side=tk.LEFT)
        ep2 = ttk.Entry(rowp, textvariable=self.phi_var, width=6)
        ep2.pack(side=tk.LEFT, padx=2)
        for e in (ep1, ep2):
            e.bind("<Return>", self.on_auto_p_commit)
            e.bind("<FocusOut>", self.on_auto_p_commit)

        rowv = ttk.Frame(right)
        rowv.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rowv, text="vmin").pack(side=tk.LEFT)
        self.vmin_var = tk.StringVar()
        self.vmax_var = tk.StringVar()
        self.ev1 = ttk.Entry(rowv, textvariable=self.vmin_var, width=9)
        self.ev1.pack(side=tk.LEFT, padx=2)
        ttk.Label(rowv, text="vmax").pack(side=tk.LEFT, padx=(8, 0))
        self.ev2 = ttk.Entry(rowv, textvariable=self.vmax_var, width=9)
        self.ev2.pack(side=tk.LEFT, padx=2)
        for e, cb in ((self.ev1, self.on_vmin_commit),
                      (self.ev2, self.on_vmax_commit)):
            e.bind("<Return>", cb)
            e.bind("<FocusOut>", cb)
        self.ev1.configure(state="disabled")   # 기본 모드 = 자동
        self.ev2.configure(state="disabled")

        roww = ttk.Frame(right)
        roww.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(roww,
                   text=f"Auto {DEFAULT_PERCENTILES[0]}–{DEFAULT_PERCENTILES[1]}%",
                   command=self.auto_contrast).pack(side=tk.LEFT)
        ttk.Button(roww, text="대비 전체 적용",
                   command=self.apply_contrast_all).pack(side=tk.RIGHT)
        ttk.Label(right, foreground="gray",
                  text="수동 모드: 히스토그램 좌클릭=vmin / 우클릭=vmax"
                  ).pack(anchor="w", pady=(4, 0))

        sbf = ttk.LabelFrame(right, text="Scale bar", padding=4)
        sbf.pack(fill=tk.X, pady=(8, 0))
        self.scalebar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sbf, text="표시", variable=self.scalebar_var,
                        command=self.on_sb_change).pack(side=tk.LEFT)
        ttk.Label(sbf, text=" µm/px").pack(side=tk.LEFT)
        self.umpp_var = tk.StringVar(
            value=f'{MAG_PRESETS["high"]["um_per_px"]:g}')
        es1 = ttk.Entry(sbf, textvariable=self.umpp_var, width=6)
        es1.pack(side=tk.LEFT, padx=2)
        ttk.Label(sbf, text="길이(µm)").pack(side=tk.LEFT)
        self.barum_var = tk.StringVar(
            value=f'{MAG_PRESETS["high"]["bar_um"]:g}')
        es2 = ttk.Entry(sbf, textvariable=self.barum_var, width=6)
        es2.pack(side=tk.LEFT, padx=2)
        for e in (es1, es2):
            e.bind("<Return>", self.on_sb_change)
            e.bind("<FocusOut>", self.on_sb_change)

        posf = ttk.LabelFrame(right, text="오버레이 위치 (자동=어두운 corner)",
                              padding=4)
        posf.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(posf, text="Z 라벨").grid(row=0, column=0, sticky="w")
        self.labelpos_var = tk.StringVar(value=POS_LABELS["auto"])
        cb1 = ttk.Combobox(posf, textvariable=self.labelpos_var, width=6,
                           state="readonly", values=list(POS_LABELS.values()))
        cb1.grid(row=0, column=1, padx=(4, 12))
        ttk.Label(posf, text="Scale bar").grid(row=0, column=2, sticky="w")
        self.barpos_var = tk.StringVar(value=POS_LABELS["auto"])
        cb2 = ttk.Combobox(posf, textvariable=self.barpos_var, width=6,
                           state="readonly", values=list(POS_LABELS.values()))
        cb2.grid(row=0, column=3, padx=4)
        for cb in (cb1, cb2):
            cb.bind("<<ComboboxSelected>>", self.on_overlay_pos_change)
        self.label_show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(posf, text="라벨 표시", variable=self.label_show_var,
                        command=self.on_overlay_pos_change).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        demf = ttk.LabelFrame(right, text="정합 (De-motion)", padding=4)
        demf.pack(fill=tk.X, pady=(8, 0))
        rowd1 = ttk.Frame(demf)
        rowd1.pack(fill=tk.X)
        self.demo_var = tk.StringVar(value="none")
        for txt, val in (("없음", "none"), ("세그먼트별", "segment"),
                         ("프레임별", "frame")):
            ttk.Radiobutton(rowd1, text=txt, value=val,
                            variable=self.demo_var).pack(side=tk.LEFT, padx=3)
        rowd2 = ttk.Frame(demf)
        rowd2.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(rowd2, text="현재 항목을 기준으로",
                   command=self.set_reg_reference).pack(side=tk.LEFT)
        ttk.Button(rowd2, text="정합 검사",
                   command=self.demotion_check).pack(side=tk.RIGHT)
        rowd3 = ttk.Frame(demf)
        rowd3.pack(fill=tk.X, pady=(4, 0))
        self.regdrag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(rowd3, text="정합 ROI 드래그(청록)",
                        variable=self.regdrag_var,
                        command=self.on_regdrag_toggle).pack(side=tk.LEFT)
        ttk.Button(rowd3, text="ROI 지우기",
                   command=self.clear_reg_roi).pack(side=tk.RIGHT)
        self.regref_var = tk.StringVar()
        ttk.Label(right, textvariable=self.regref_var,
                  foreground="gray").pack(anchor="w", pady=(2, 0))
        ttk.Label(right, foreground="gray",
                  text="정합 = gradient(라인 구조) 기반 위상상관 — "
                       "ROI는 보고자 하는 line만 감싸게 지정"
                  ).pack(anchor="w")

        # 하단: 내보내기
        bot = ttk.Frame(self.root, padding=6)
        bot.grid(row=2, column=0, columnspan=3, sticky="ew")
        ttk.Label(bot, text="FPS").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value=str(DEFAULT_FPS))
        ttk.Spinbox(bot, from_=1, to=120, width=5,
                    textvariable=self.fps_var).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(bot, text="배율").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="1.0")
        ttk.Entry(bot, textvariable=self.scale_var, width=5).pack(side=tk.LEFT,
                                                                  padx=(2, 10))
        self.roundtrip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bot, text="왕복 재생",
                        variable=self.roundtrip_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(bot, text="출력").pack(side=tk.LEFT)
        self.out_var = tk.StringVar()
        ttk.Entry(bot, textvariable=self.out_var, width=40).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(bot, text="...", width=3,
                   command=self.browse_out).pack(side=tk.LEFT)
        self.export_btn = ttk.Button(bot, text="MP4 내보내기",
                                     command=self.start_export)
        self.export_btn.pack(side=tk.LEFT, padx=8)
        self.progress = ttk.Progressbar(bot, length=150)
        self.progress.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="준비")
        ttk.Label(bot, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)
        self._update_regref_label()

    def _make_selector(self):
        active = bool((getattr(self, "cropdrag_var", None)
                       and self.cropdrag_var.get())
                      or (getattr(self, "regdrag_var", None)
                          and self.regdrag_var.get()))
        kw = dict(useblit=False, button=[1], minspanx=5, minspany=5,
                  spancoords="pixels")
        try:
            self.rect_sel = RectangleSelector(self.ax_img, self.on_crop_select,
                                              interactive=True, **kw)
        except TypeError:
            self.rect_sel = RectangleSelector(self.ax_img, self.on_crop_select,
                                              **kw)
        self.rect_sel.set_active(active)

    def status(self, s):
        self.status_var.set(s)

    # --------------------------- 배율 모드 ---------------------------
    def _dz(self):
        try:
            v = float(self.dz_var.get())
            if v > 0:
                return v
        except (ValueError, tk.TclError):
            pass
        return self.mode_store[self._active_mode]["dz_um"]

    def depth_um(self, i):
        """인덱스 → µm 라벨 (현재 모드의 Δz로 선형 매핑 후 반올림)"""
        return int(round(i * self._dz()))

    def item_title(self, i):
        if self.src_var.get() == "pulse":
            return self.seg_names.get(i, str(i))
        return f"z = {self.depth_um(i)} µm"

    def _overlay_label_text(self):
        if self.cur is None or not self.label_show_var.get():
            return ""
        if self.src_var.get() == "pulse":
            b = self.pulse_bounds.get(str(self.files[self.cur]))
            t = pulse_label_at(b, self.frame_i) if b else None
            if t:
                return t
            return f"pulse {list(self.files).index(self.cur) + 1}"
        return f"z = {self.depth_um(self.cur)} µm"

    def on_src_change(self):
        mode = self.src_var.get()
        if mode == self._active_src:
            return
        self._active_src = mode
        if mode == "pulse" and self.demo_var.get() == "none":
            self.demo_var.set("segment")
        if self.folder is not None:
            self.scan_folder(self.folder, quiet=True)

    def move_segment(self, delta):
        if self.src_var.get() != "pulse" or self.cur is None:
            return
        keys = list(self.files)
        pos = keys.index(self.cur)
        new = pos + delta
        if not 0 <= new < len(keys):
            return
        a, b = keys[pos], keys[new]
        for d in (self.files, self.meta, self.seg_names):
            d[a], d[b] = d[b], d[a]
        if self.reg_ref == a:
            self.reg_ref = b
        elif self.reg_ref == b:
            self.reg_ref = a
        self.cur = b
        self.refresh_listbox()

    # ------------------------- 정합 기준/ROI -------------------------
    def set_reg_reference(self):
        if self.cur is None:
            return
        self.reg_ref = self.cur
        self._ref_cache = None
        self._update_regref_label()
        self.refresh_listbox()
        self.status(f"정합 기준 설정: {self.item_title(self.cur)}")

    def clear_reg_roi(self):
        self.reg_roi = None
        self._overlays_stale = True
        self._update_regref_label()
        if self.stack is not None:
            self.draw_frame()

    def _update_regref_label(self):
        name = (self.item_title(self.reg_ref)
                if self.reg_ref in self.files else "첫 포함 항목(기본)")
        roi = (f"정합ROI x[{self.reg_roi[0]}–{self.reg_roi[1]}] "
               f"y[{self.reg_roi[2]}–{self.reg_roi[3]}]"
               if self.reg_roi else "정합영역: crop/전체")
        self.regref_var.set(f"기준 ★: {name}  |  {roi}")

    def _crop2d_disp(self, fr):
        if self.crop:
            cx0, cx1, cy0, cy1 = self.crop
            return fr[cy0:cy1, cx0:cx1]
        return fr

    def _reg2d(self, fr):
        if self.reg_roi:
            rx0, rx1, ry0, ry1 = self.reg_roi
            return fr[ry0:ry1, rx0:rx1]
        return self._crop2d_disp(fr)

    def _ref_median_full(self):
        """정합 기준 항목의 전체 프레임 중앙값 (캐시)"""
        rk = self.reg_ref if self.reg_ref in self.files else None
        if rk is None:
            inc = [k for k in self.files if self.meta[k]["include"]]
            rk = inc[0] if inc else (next(iter(self.files), None))
        if rk is None:
            return None, None
        m = self.meta[rk]
        sig = (rk, m["start"], m["end"],
               tuple(sorted(int(v) for v in (m.get("excluded") or ()))))
        if self._ref_cache and self._ref_cache[0] == sig:
            return rk, self._ref_cache[1]
        self.status(f"정합 기준 로드 중: {self.item_title(rk)} ...")
        self.root.update_idletasks()
        arr = self.stack if rk == self.cur else load_stack(str(self.files[rk]))
        exc = set(m.get("excluded") or ())
        kept = [k for k in range(m["start"], m["end"] + 1)
                if k not in exc] or list(range(arr.shape[0]))
        rs = kept[::max(1, len(kept) // 30)][:30]
        med = np.median(np.stack(
            [arr[k].astype(np.float32) for k in rs]), axis=0)
        self._ref_cache = (sig, med)
        self.status("정합 기준 준비 완료")
        return rk, med

    def demotion_check(self):
        """기준(빨강) ↔ 현재(초록) 정합 전/후 오버레이 검사"""
        if self.cur is None or self.stack is None:
            return
        rk, ref_med = self._ref_median_full()
        if ref_med is None:
            return
        m = self.meta[self.cur]
        exc = set(m.get("excluded") or ())
        kept = [k for k in range(m["start"], m["end"] + 1)
                if k not in exc] or list(range(self.stack.shape[0]))
        rs = kept[::max(1, len(kept) // 30)][:30]
        cur_med = np.median(np.stack(
            [self.stack[k].astype(np.float32) for k in rs]), axis=0)
        dy, dx = register_shift(prep_reg(self._reg2d(ref_med)),
                                self._reg2d(cur_med))
        cur_corr = shift_image(cur_med, dy, dx) \
            if (abs(dy) > 0.02 or abs(dx) > 0.02) else cur_med

        def _n(img):
            lo, hi = np.percentile(img, (1, 99))
            return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

        r = _n(self._crop2d_disp(ref_med))
        g0 = _n(self._crop2d_disp(cur_med))
        g1 = _n(self._crop2d_disp(cur_corr))
        rgb0 = np.dstack([r, g0, np.zeros_like(r)])
        rgb1 = np.dstack([r, g1, np.zeros_like(r)])
        win = tk.Toplevel(self.root)
        win.title("정합 검사")
        fig = Figure(figsize=(8.6, 4.4), dpi=100)
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax0.imshow(rgb0)
        ax0.axis("off")
        ax0.set_title("정합 전 (빨강=기준, 초록=현재)", fontsize=9)
        ax1.imshow(rgb1)
        ax1.axis("off")
        ax1.set_title(f"정합 후  shift = ({dy:+.2f}, {dx:+.2f}) px",
                      fontsize=9)
        fig.suptitle(f"기준: {self.item_title(rk)}  ↔  현재: "
                     f"{self.item_title(self.cur)}", fontsize=9)
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cv.draw()
        self.status(f"정합 검사: shift = ({dy:+.2f}, {dx:+.2f}) px")

    def _load_mode_entries(self):
        st = self.mode_store[self._active_mode]
        self._syncing = True
        try:
            self.dz_var.set(f'{st["dz_um"]:g}')
            self.umpp_var.set(f'{st["um_per_px"]:g}')
            self.barum_var.set(f'{st["bar_um"]:g}')
        finally:
            self._syncing = False

    def on_mag_change(self):
        mode = self.mag_var.get()
        if mode == self._active_mode:
            return
        self._active_mode = mode
        self._load_mode_entries()
        self._overlays_stale = True
        if self.files:
            self.refresh_listbox()
        if self.stack is not None:
            self.draw_frame()
            self.draw_hist()
        st = self.mode_store[mode]
        self.status(f"{MAG_PRESETS[mode]['label']} 모드 "
                    f"(Δz={st['dz_um']:g} µm, {st['um_per_px']:g} µm/px, "
                    f"bar {st['bar_um']:g} µm)")

    def on_dz_commit(self, _evt=None):
        if self._syncing:
            return
        st = self.mode_store[self._active_mode]
        try:
            v = float(self.dz_var.get())
            if v <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            v = st["dz_um"]
        st["dz_um"] = v
        self.dz_var.set(f"{v:g}")
        self._overlays_stale = True
        if self.files:
            self.refresh_listbox()
        if self.stack is not None:
            self.draw_frame()
            self.draw_hist()

    # --------------------------- 폴더/파일 ---------------------------
    def open_folder(self):
        d = filedialog.askdirectory(
            initialdir=str(self.folder) if self.folder else DEFAULT_DIR)
        if d:
            self.scan_folder(Path(d))

    def scan_folder(self, folder, quiet=False):
        pulse = self.src_var.get() == "pulse"
        if pulse:
            paths = sorted(p for p in Path(folder).iterdir()
                           if p.suffix.lower() in (".tif", ".tiff"))
            if not paths:
                if not quiet:
                    messagebox.showwarning("폴더 열기",
                                           "TIF 파일을 찾지 못했습니다.")
                return
            self.folder = Path(folder)
            self.prefix = ""
            self.files = {k: p for k, p in enumerate(paths)}
            self.seg_names = {k: p.stem for k, p in enumerate(paths)}
            self.pulse_bounds = {}
            for p in paths:
                self.pulse_bounds[str(p)] = load_pulse_boundaries(p)
        else:
            groups = {}
            for p in sorted(Path(folder).iterdir()):
                m = FILE_RE.match(p.name)
                if m:
                    groups.setdefault(m.group("prefix"),
                                      {})[int(m.group("idx"))] = p
            if not groups:
                if not quiet:
                    messagebox.showwarning(
                        "폴더 열기",
                        "*_rl_<n>.tif 형식의 파일을 찾지 못했습니다.")
                return
            prefix = max(groups, key=lambda k: len(groups[k]))
            self.folder = Path(folder)
            self.prefix = prefix
            self.files = dict(sorted(groups[prefix].items()))
            self.seg_names = {}
        self.meta = {}
        for i, p in self.files.items():
            n = stack_len(p)
            self.meta[i] = dict(include=True, start=0, end=n - 1,
                                cycles=(1 if pulse else DEFAULT_CYCLES),
                                vmin=None, vmax=None,
                                n=n, sample=None, motion=None, excluded=set())
        self.cur = None
        self.stack = None
        self._im = None
        self._auto_key = None
        self.reg_ref = None
        self._ref_cache = None
        self._update_regref_label()
        if pulse:
            nb = sum(1 for v in self.pulse_bounds.values() if v)
            extra = (f", boundaries {nb}개 → 프레임별 pulse 라벨"
                     if nb else "")
            self.folder_var.set(
                f"{folder}  |  pulse segments ({len(self.files)} files"
                f"{extra})")
        else:
            self.folder_var.set(
                f"{folder}  |  {self.prefix}*.tif  ({len(self.files)} depths)")
        self.out_var.set(str(Path(folder) / "depth_cycle_video.mp4"))
        self.refresh_listbox(select=0)

    def refresh_listbox(self, select=None):
        keys = list(self.files)
        if select is None and self.cur in keys:
            select = keys.index(self.cur)
        self.listbox.delete(0, tk.END)
        for i in keys:
            m = self.meta[i]
            mark = "✓" if m["include"] else "✗"
            star = "★" if i == self.reg_ref else " "
            exc_n = len(m.get("excluded") or ())
            tail = f" −{exc_n}f" if exc_n else ""
            if self.src_var.get() == "pulse":
                name = self.seg_names.get(i, str(i))[:21]
                head = f"{star}{mark} {name:<21}"
            else:
                head = f"{star}{mark} rl_{i:<2d} z={self.depth_um(i):>4d}µm"
            self.listbox.insert(
                tk.END,
                f'{head} f[{m["start"]}–{m["end"]}] ×{m["cycles"]}{tail}')
        if select is not None and keys:
            select = min(select, len(keys) - 1)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(select)
            self.listbox.see(select)
            self.select_depth(keys[select])

    def on_listbox_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        keys = list(self.files)
        self.select_depth(keys[sel[0]])

    def select_depth(self, i):
        if i == self.cur and self.stack is not None:
            self.sync_controls()
            return
        self.cur = i
        self.status(f"불러오는 중: {self.files[i].name} ...")
        self.root.update_idletasks()
        self.stack = load_stack(self.files[i])
        m = self.meta[i]
        m["n"] = self.stack.shape[0]
        m["end"] = min(m["end"], m["n"] - 1)
        if m["sample"] is None:
            m["sample"] = sample_values(
                self._crop_arr(self.stack)).astype(np.float32)
        if m["vmin"] is None or m["vmax"] is None:
            lo, hi = np.percentile(m["sample"], DEFAULT_PERCENTILES)
            m["vmin"], m["vmax"] = float(lo), float(hi)
        self.frame_i = min(self.frame_i, m["n"] - 1)
        self.frame_scale.configure(to=m["n"] - 1)
        self.frame_scale.set(self.frame_i)
        new_shape = self.stack.shape[1:]
        if self._im is not None and self._im_shape != new_shape:
            self._im = None
        self.sync_controls()
        self.draw_frame()
        self.draw_hist()
        self.status(f"{self.item_title(i)} | {m['n']} frames")
        self._motion_refresh_safe()

    def sync_controls(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        self._syncing = True
        try:
            self.include_var.set(bool(m["include"]))
            self.start_var.set(str(m["start"]))
            self.end_var.set(str(m["end"]))
            self.cycles_var.set(str(m["cycles"]))
            if self.norm_mode_var.get() != "auto":
                self.vmin_var.set("" if m["vmin"] is None else f'{m["vmin"]:.1f}')
                self.vmax_var.set("" if m["vmax"] is None else f'{m["vmax"]:.1f}')
        finally:
            self._syncing = False

    # ------------------------- crop / 대비 계산 -------------------------
    def _crop_arr(self, arr):
        if self.crop:
            x0, x1, y0, y1 = self.crop
            return arr[:, y0:y1, x0:x1]
        return arr

    def _auto_percentiles(self):
        try:
            plo = float(self.plo_var.get())
        except (ValueError, tk.TclError):
            plo = AUTO_P_DEFAULT[0]
        try:
            phi = float(self.phi_var.get())
        except (ValueError, tk.TclError):
            phi = AUTO_P_DEFAULT[1]
        plo = min(max(plo, 0.0), 100.0)
        phi = min(max(phi, 0.0), 100.0)
        if phi <= plo:
            plo, phi = AUTO_P_DEFAULT
        return plo, phi

    @staticmethod
    def _range_of(sub, plo, phi):
        """sub: (T,Y,X) 잘라낸 배열에 대해 (lo, hi) 계산"""
        if plo <= 0.0 and phi >= 100.0:
            lo, hi = float(sub.min()), float(sub.max())
        else:
            s = sample_values(sub, 1_000_000)
            lo, hi = (float(v) for v in np.percentile(s, (plo, phi)))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _auto_range_current(self):
        m = self.meta[self.cur]
        key = (self.cur, m["start"], m["end"],
               tuple(self.crop) if self.crop else None,
               self._auto_percentiles())
        if self._auto_key == key:
            return self._auto_val
        sub = self._crop_arr(self.stack[m["start"]: m["end"] + 1])
        plo, phi = key[-1]
        lo, hi = self._range_of(sub, plo, phi)
        self._auto_key, self._auto_val = key, (lo, hi)
        return lo, hi

    def _effective_range(self):
        m = self.meta[self.cur]
        if self.norm_mode_var.get() == "auto" and self.stack is not None:
            lo, hi = self._auto_range_current()
            self._syncing = True
            try:
                self.vmin_var.set(f"{lo:.1f}")
                self.vmax_var.set(f"{hi:.1f}")
            finally:
                self._syncing = False
            return lo, hi
        return m["vmin"], m["vmax"]

    # ----------------------------- 그리기 -----------------------------
    def draw_frame(self):
        if self.stack is None:
            return
        m = self.meta[self.cur]
        lo, hi = self._effective_range()
        frm = self.stack[self.frame_i]
        created = self._im is None
        if created:
            self.ax_img.clear()
            self.ax_img.axis("off")
            self._im = self.ax_img.imshow(frm, cmap="gray", vmin=lo, vmax=hi)
            self._im_shape = self.stack.shape[1:]
            self._crop_patch = None
            self._sb_artists = []
            self._make_selector()
            self._overlays_stale = True
        else:
            self._im.set_data(frm)
            self._im.set_clim(lo, hi)
        exc_mark = "  [제외 프레임]" if self.frame_i in m.get("excluded", ()) \
            else ""
        self.ax_img.set_title(
            f'{self.item_title(self.cur)}   '
            f'frame {self.frame_i}/{m["n"] - 1}{exc_mark}',
            fontsize=10)
        if self._overlay_label_text() != self._last_ov_txt:
            self._overlays_stale = True
        if self._overlays_stale:
            self._draw_overlays()
            self._overlays_stale = False
        self.canvas_img.draw_idle()

    def _pos_codes(self):
        lp = POS_CODES.get(self.labelpos_var.get(), "auto")
        bp = POS_CODES.get(self.barpos_var.get(), "auto")
        return lp, bp

    def _preview_corner_scores(self, x0, x1, y0, y1):
        """현재 depth 기준 corner 밝기 (프레임 최대 5장 평균, 자동 배치 미리보기용)"""
        m = self.meta[self.cur]
        lo, hi = self._effective_range()
        ks = np.unique(np.linspace(
            m["start"], m["end"],
            min(5, m["end"] - m["start"] + 1)).astype(int))
        acc = None
        for k in ks:
            f = self.stack[int(k), y0:y1, x0:x1].astype(np.float32)
            n = np.clip((f - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            acc = n if acc is None else acc + n
        return corner_scores(acc / len(ks))

    def _draw_overlays(self):
        if self._crop_patch is not None:
            try:
                self._crop_patch.remove()
            except Exception:
                pass
            self._crop_patch = None
        for a in self._sb_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._sb_artists = []
        if self.stack is None:
            return
        H, W = self.stack.shape[1:]
        if self.crop:
            x0, x1, y0, y1 = self.crop
            self._crop_patch = mpatches.Rectangle(
                (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                edgecolor="red", linestyle="--", linewidth=1.2)
            self.ax_img.add_patch(self._crop_patch)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        if self.reg_roi:
            rx0, rx1, ry0, ry1 = self.reg_roi
            rp = mpatches.Rectangle(
                (rx0 - 0.5, ry0 - 0.5), rx1 - rx0, ry1 - ry0, fill=False,
                edgecolor="cyan", linestyle=":", linewidth=1.2)
            self.ax_img.add_patch(rp)
            self._sb_artists.append(rp)
        lp, bp = self._pos_codes()
        if lp == "auto" or (bp == "auto" and self.scalebar_var.get()):
            try:
                scores = self._preview_corner_scores(x0, x1, y0, y1)
            except Exception:
                scores = {c: 0.0 for c in CORNERS}
            lp, bp = resolve_corners(lp, bp, scores)
        if lp == "auto":
            lp = "tl"
        if bp == "auto":
            bp = "br"
        m_ = max(6, (x1 - x0) // 40)
        pes = [pe.withStroke(linewidth=2, foreground="black")]
        ov_txt = self._overlay_label_text()
        self._last_ov_txt = ov_txt
        if ov_txt:
            lx = x0 + m_ if lp in ("tl", "bl") else x1 - m_
            ly = y0 + m_ if lp in ("tl", "tr") else y1 - m_
            t = self.ax_img.text(
                lx, ly, ov_txt, color="white",
                ha="left" if lp in ("tl", "bl") else "right",
                va="top" if lp in ("tl", "tr") else "bottom",
                fontsize=9, path_effects=pes)
            self._sb_artists.append(t)
        if self.scalebar_var.get():
            try:
                umpp = float(self.umpp_var.get())
                bum = float(self.barum_var.get())
            except (ValueError, tk.TclError):
                return
            if umpp <= 0 or bum <= 0:
                return
            blen = bum / umpp
            if bp in ("tr", "br"):
                xr = x1 - m_
                xl = max(x0 + 2, xr - blen)
            else:
                xl = x0 + m_
                xr = min(x1 - 2, xl + blen)
            if bp in ("bl", "br"):
                y = y1 - m_
                ty, tva = y - 5, "bottom"
            else:
                y = y0 + m_ + 4
                ty, tva = y + 5, "top"
            ln, = self.ax_img.plot([xl, xr], [y, y], color="white",
                                   linewidth=3, solid_capstyle="butt")
            ln.set_path_effects(
                [pe.withStroke(linewidth=5, foreground="black")])
            txt = self.ax_img.text(
                (xl + xr) / 2, ty, f"{bum:g} µm", color="white",
                ha="center", va=tva, fontsize=8, path_effects=pes)
            self._sb_artists += [ln, txt]

    def draw_hist(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        if m["sample"] is None:
            if self.stack is None:
                return
            m["sample"] = sample_values(
                self._crop_arr(self.stack)).astype(np.float32)
        lo, hi = self._effective_range()
        self.ax_hist.clear()
        self.ax_hist.hist(m["sample"], bins=256, color="0.4")
        self.ax_hist.set_yscale("log")
        region = "crop 영역" if self.crop else "전체"
        self.ax_hist.set_title(
            f"Histogram ({region}) — {self.item_title(self.cur)}", fontsize=9)
        self._vline_lo = self.ax_hist.axvline(lo, color="tab:blue")
        self._vline_hi = self.ax_hist.axvline(hi, color="tab:red")
        self.fig_hist.tight_layout()
        self.canvas_hist.draw_idle()

    def _update_contrast_visuals(self):
        lo, hi = self._effective_range()
        if self._im is not None:
            self._im.set_clim(lo, hi)
            self.canvas_img.draw_idle()
        if self._vline_lo is not None:
            self._vline_lo.set_xdata([lo, lo])
            self._vline_hi.set_xdata([hi, hi])
            self.canvas_hist.draw_idle()

    # ----------------------------- 콜백 -----------------------------
    def on_frame_slide(self, val):
        if self.stack is None:
            return
        self.frame_i = int(float(val))
        self.draw_frame()

    def on_cropdrag_toggle(self):
        if self.cropdrag_var.get():
            self.regdrag_var.set(False)
            self._drag_target = "crop"
        self._update_selector_active()

    def on_regdrag_toggle(self):
        if self.regdrag_var.get():
            self.cropdrag_var.set(False)
            self._drag_target = "reg"
        self._update_selector_active()

    def _update_selector_active(self):
        on = bool(self.cropdrag_var.get() or self.regdrag_var.get())
        self.rect_sel.set_active(on)
        if not on:
            try:
                self.rect_sel.set_visible(False)
            except Exception:
                pass
        self.canvas_img.draw_idle()

    def on_crop_select(self, eclick, erelease):
        if self.stack is None:
            return
        if None in (eclick.xdata, erelease.xdata,
                    eclick.ydata, erelease.ydata):
            return
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        H, W = self.stack.shape[1:]
        x0 = max(0, int(round(x0)))
        x1 = min(W, int(round(x1)))
        y0 = max(0, int(round(y0)))
        y1 = min(H, int(round(y1)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        if self._drag_target == "reg":
            self.reg_roi = [x0, x1, y0, y1]
            self._overlays_stale = True
            self._update_regref_label()
            self.draw_frame()
            self.status(f"정합 ROI 설정: x[{x0}–{x1}] y[{y0}–{y1}]")
        else:
            self.crop = [x0, x1, y0, y1]
            self._after_crop_changed()

    def on_crop_entry_commit(self, _evt=None):
        if self.stack is None or self._syncing:
            return
        vals = [self.cx0_var.get().strip(), self.cx1_var.get().strip(),
                self.cy0_var.get().strip(), self.cy1_var.get().strip()]
        if all(v == "" for v in vals):
            if self.crop is not None:
                self.clear_crop()
            return
        try:
            x0, x1, y0, y1 = (int(float(v)) for v in vals)
        except ValueError:
            self._sync_crop_entries()
            return
        H, W = self.stack.shape[1:]
        x0 = max(0, min(x0, W - 4))
        x1 = max(x0 + 4, min(x1, W))
        y0 = max(0, min(y0, H - 4))
        y1 = max(y0 + 4, min(y1, H))
        self.crop = [x0, x1, y0, y1]
        self._after_crop_changed()

    def clear_crop(self):
        self.crop = None
        try:
            self.rect_sel.set_visible(False)
        except Exception:
            pass
        self._after_crop_changed()

    def _sync_crop_entries(self):
        self._syncing = True
        try:
            if self.crop:
                x0, x1, y0, y1 = self.crop
                self.cx0_var.set(str(x0))
                self.cx1_var.set(str(x1))
                self.cy0_var.set(str(y0))
                self.cy1_var.set(str(y1))
            else:
                for v in (self.cx0_var, self.cx1_var,
                          self.cy0_var, self.cy1_var):
                    v.set("")
        finally:
            self._syncing = False

    def _after_crop_changed(self):
        # crop 기준으로 모든 depth의 히스토그램 샘플 무효화
        for mm in self.meta.values():
            mm["sample"] = None
            mm["motion"] = None
        self._auto_key = None
        if self.cur is not None and self.stack is not None:
            m = self.meta[self.cur]
            m["sample"] = sample_values(
                self._crop_arr(self.stack)).astype(np.float32)
        self._sync_crop_entries()
        self._overlays_stale = True
        self.draw_frame()
        self.draw_hist()
        self._motion_refresh_safe()

    def on_sb_change(self, _evt=None):
        if self._syncing:
            return
        st = self.mode_store[self._active_mode]
        try:
            v = float(self.umpp_var.get())
            if v > 0:
                st["um_per_px"] = v
        except (ValueError, tk.TclError):
            pass
        try:
            v = float(self.barum_var.get())
            if v > 0:
                st["bar_um"] = v
        except (ValueError, tk.TclError):
            pass
        self._overlays_stale = True
        if self.stack is not None:
            self.draw_frame()

    def on_overlay_pos_change(self, _evt=None):
        self._overlays_stale = True
        if self.stack is not None:
            self.draw_frame()

    def on_norm_mode_change(self):
        auto = self.norm_mode_var.get() == "auto"
        st = "disabled" if auto else "normal"
        self.ev1.configure(state=st)
        self.ev2.configure(state=st)
        if not auto:
            self.sync_controls()
        if self.cur is not None:
            self.draw_frame()
            self.draw_hist()

    def on_auto_p_commit(self, _evt=None):
        if self.cur is not None and self.norm_mode_var.get() == "auto":
            self.draw_frame()
            self.draw_hist()

    def set_start_here(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        m["start"] = min(self.frame_i, m["end"])
        self.sync_controls()
        self.refresh_listbox()
        self.draw_frame()
        self.draw_hist()

    def set_end_here(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        m["end"] = max(self.frame_i, m["start"])
        self.sync_controls()
        self.refresh_listbox()
        self.draw_frame()
        self.draw_hist()

    def on_start_commit(self, _evt=None):
        if self.cur is None or self._syncing:
            return
        m = self.meta[self.cur]
        try:
            v = int(float(self.start_var.get()))
        except ValueError:
            v = m["start"]
        m["start"] = max(0, min(v, m["end"]))
        self.start_var.set(str(m["start"]))
        self.refresh_listbox()
        self.draw_frame()
        self.draw_hist()

    def on_end_commit(self, _evt=None):
        if self.cur is None or self._syncing:
            return
        m = self.meta[self.cur]
        try:
            v = int(float(self.end_var.get()))
        except ValueError:
            v = m["end"]
        m["end"] = max(m["start"], min(v, m["n"] - 1))
        self.end_var.set(str(m["end"]))
        self.refresh_listbox()
        self.draw_frame()
        self.draw_hist()

    def on_cycles_change(self, _evt=None):
        if self.cur is None or self._syncing:
            return
        m = self.meta[self.cur]
        try:
            v = max(1, int(float(self.cycles_var.get())))
        except ValueError:
            v = m["cycles"]
        m["cycles"] = v
        self.cycles_var.set(str(v))
        self.refresh_listbox()

    def on_include_toggle(self):
        if self.cur is None or self._syncing:
            return
        self.meta[self.cur]["include"] = bool(self.include_var.get())
        self.refresh_listbox()

    def on_vmin_commit(self, _evt=None):
        if self.cur is None or self._syncing:
            return
        m = self.meta[self.cur]
        try:
            v = float(self.vmin_var.get())
        except ValueError:
            v = m["vmin"]
        if m["vmax"] is not None:
            v = min(v, m["vmax"] - 1e-6)
        m["vmin"] = v
        self.vmin_var.set(f"{v:.1f}")
        self._update_contrast_visuals()

    def on_vmax_commit(self, _evt=None):
        if self.cur is None or self._syncing:
            return
        m = self.meta[self.cur]
        try:
            v = float(self.vmax_var.get())
        except ValueError:
            v = m["vmax"]
        if m["vmin"] is not None:
            v = max(v, m["vmin"] + 1e-6)
        m["vmax"] = v
        self.vmax_var.set(f"{v:.1f}")
        self._update_contrast_visuals()

    def on_hist_click(self, event):
        if self.cur is None or event.inaxes is not self.ax_hist \
                or event.xdata is None:
            return
        if self.norm_mode_var.get() == "auto":
            self.status("자동 모드에서는 vmin/vmax를 직접 지정할 수 없습니다 "
                        "(수동 모드로 전환하세요)")
            return
        m = self.meta[self.cur]
        x = float(event.xdata)
        if event.button == 1:
            hi = m["vmax"] if m["vmax"] is not None else x + 1.0
            m["vmin"] = min(x, hi - 1e-6)
            self.vmin_var.set(f'{m["vmin"]:.1f}')
        elif event.button == 3:
            lo = m["vmin"] if m["vmin"] is not None else x - 1.0
            m["vmax"] = max(x, lo + 1e-6)
            self.vmax_var.set(f'{m["vmax"]:.1f}')
        else:
            return
        self._update_contrast_visuals()

    def auto_contrast(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        if m["sample"] is None:
            return
        lo, hi = np.percentile(m["sample"], DEFAULT_PERCENTILES)
        m["vmin"], m["vmax"] = float(lo), float(hi)
        if self.norm_mode_var.get() != "auto":
            self.sync_controls()
            self._update_contrast_visuals()
        else:
            self.status("수동 vmin/vmax에 저장됨 (현재는 자동 모드 표시 중)")

    def apply_contrast_all(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        if m["vmin"] is None:
            return
        for mm in self.meta.values():
            mm["vmin"], mm["vmax"] = m["vmin"], m["vmax"]
        self.status("현재 수동 vmin/vmax를 모든 depth에 적용했습니다")

    def apply_range_all(self):
        if self.cur is None:
            return
        m = self.meta[self.cur]
        for mm in self.meta.values():
            mm["start"] = min(m["start"], mm["n"] - 1)
            mm["end"] = min(m["end"], mm["n"] - 1)
            if mm["start"] > mm["end"]:
                mm["start"] = mm["end"]
            mm["cycles"] = m["cycles"]
        self.refresh_listbox()
        self.status("현재 frame 범위/cycle을 모든 depth에 적용했습니다")

    # ------------------------- 모션/호흡 제거 -------------------------
    def _motion_refresh_safe(self):
        if self.motion_win is not None and self.motion_win.winfo_exists():
            self._motion_refresh()

    def _ensure_motion(self):
        """현재 depth의 모션 지표를 crop 기준으로 계산해 meta에 캐시"""
        if self.cur is None or self.stack is None:
            return None
        m = self.meta[self.cur]
        if m.get("motion") is None:
            self.status("모션 지표 계산 중...")
            self.root.update_idletasks()
            m["motion"] = motion_metric(self._crop_arr(self.stack))
            self.status("모션 지표 계산 완료")
        return m["motion"]

    def open_motion_dialog(self):
        if self.motion_win is not None and self.motion_win.winfo_exists():
            self.motion_win.lift()
            self._motion_refresh()
            return
        win = tk.Toplevel(self.root)
        win.title("모션/호흡 프레임 제거")
        win.geometry("880x480")
        win.protocol("WM_DELETE_WINDOW", self._on_motion_close)
        self.motion_win = win

        self.m_fig = Figure(figsize=(8.4, 3.2), dpi=100)
        self.m_ax = self.m_fig.add_subplot(111)
        self.m_canvas = FigureCanvasTkAgg(self.m_fig, master=win)
        self.m_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                           padx=6, pady=(6, 0))

        row1 = ttk.Frame(win, padding=(6, 4))
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="임계값").pack(side=tk.LEFT)
        self.thr_var = tk.StringVar(value="")
        et = ttk.Entry(row1, textvariable=self.thr_var, width=9)
        et.pack(side=tk.LEFT, padx=4)
        et.bind("<Return>", lambda _e: self._motion_refresh())
        ttk.Button(row1, text="자동 제안",
                   command=self.motion_suggest).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="현재 depth 감지",
                   command=self.motion_apply_current).pack(side=tk.LEFT,
                                                           padx=2)
        self._motion_batch_btn = ttk.Button(
            row1, text="모든 depth 일괄 감지", command=self.motion_batch_all)
        self._motion_batch_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="현재 depth 제외 해제",
                   command=self.motion_clear_current).pack(side=tk.RIGHT,
                                                           padx=2)

        row2 = ttk.Frame(win, padding=(6, 0))
        row2.pack(fill=tk.X)
        self.sel_var = tk.StringVar(value="선택 구간: (그래프에서 가로 드래그)")
        ttk.Label(row2, textvariable=self.sel_var).pack(side=tk.LEFT)
        ttk.Button(row2, text="선택 구간 제외",
                   command=self.motion_exclude_sel).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="선택 구간 복원",
                   command=self.motion_restore_sel).pack(side=tk.LEFT)
        ttk.Label(win, foreground="gray", padding=(6, 2),
                  text="지표 = 인접 프레임 |차이| 평균 / 강도 범위 (crop 영역 기준). "
                       "주황선 = 임계값, 빨간 음영 = 제외된 프레임, "
                       "파란 점선 = frame 범위").pack(anchor="w")
        self._motion_refresh()

    def _on_motion_close(self):
        try:
            self.motion_win.destroy()
        except Exception:
            pass
        self.motion_win = None
        self.m_span = None

    def _motion_thr(self):
        try:
            return float(self.thr_var.get())
        except (ValueError, tk.TclError):
            return None

    def _motion_refresh(self):
        mm = self._ensure_motion()
        ax = self.m_ax
        ax.clear()
        if mm is None:
            ax.set_title("depth를 먼저 선택하세요", fontsize=10)
            self.m_canvas.draw_idle()
            return
        meta = self.meta[self.cur]
        exc = meta.get("excluded") or set()
        ax.plot(np.arange(len(mm)), mm, color="0.3", linewidth=0.9)
        if exc:
            idx = sorted(v for v in exc if v < len(mm))
            runs = []
            s = p = None
            for v in idx:
                if s is None:
                    s = p = v
                elif v == p + 1:
                    p = v
                else:
                    runs.append((s, p))
                    s = p = v
            if s is not None:
                runs.append((s, p))
            for a, b in runs:
                ax.axvspan(a - 0.5, b + 0.5, color="red", alpha=0.18)
            if idx:
                ax.plot(idx, mm[idx], "r.", markersize=3)
        thr = self._motion_thr()
        if thr is not None:
            ax.axhline(thr, color="tab:orange", linewidth=1.0)
        ax.axvline(meta["start"] - 0.5, color="tab:blue", linewidth=0.8,
                   linestyle=":")
        ax.axvline(meta["end"] + 0.5, color="tab:blue", linewidth=0.8,
                   linestyle=":")
        ax.set_xlim(-1, len(mm))
        ax.set_xlabel("frame")
        ax.set_ylabel("motion (norm.)")
        ax.set_title(f"{self.item_title(self.cur)}  "
                     f"(제외 {len(exc)}f)", fontsize=10)
        self.m_fig.tight_layout()
        if self.m_span is not None:
            try:
                self.m_span.set_active(False)
            except Exception:
                pass
        try:
            self.m_span = SpanSelector(
                ax, self._on_motion_span, "horizontal", useblit=False,
                props=dict(alpha=0.25, facecolor="tab:orange"))
        except TypeError:
            self.m_span = SpanSelector(
                ax, self._on_motion_span, "horizontal", useblit=False,
                rectprops=dict(alpha=0.25, facecolor="tab:orange"))
        self.m_canvas.draw_idle()

    def _on_motion_span(self, vmin, vmax):
        if self.cur is None:
            return
        n = self.meta[self.cur]["n"]
        a = max(0, int(round(min(vmin, vmax))))
        b = min(n - 1, int(round(max(vmin, vmax))))
        if b < a:
            return
        self._sel_range = (a, b)
        self.sel_var.set(f"선택 구간: {a} – {b}  ({b - a + 1} frames)")

    def motion_suggest(self):
        mm = self._ensure_motion()
        if mm is None:
            return
        self.thr_var.set(f"{suggest_threshold(mm):.4g}")
        self._motion_refresh()

    def motion_apply_current(self):
        mm = self._ensure_motion()
        thr = self._motion_thr()
        if mm is None or thr is None:
            self.status("임계값을 먼저 입력하거나 '자동 제안'을 누르세요")
            return
        m = self.meta[self.cur]
        m["excluded"] = set(int(v) for v in np.nonzero(mm > thr)[0])
        self.refresh_listbox()
        self._motion_refresh()
        self.draw_frame()

    def motion_clear_current(self):
        if self.cur is None:
            return
        self.meta[self.cur]["excluded"] = set()
        self.refresh_listbox()
        self._motion_refresh()
        self.draw_frame()

    def motion_exclude_sel(self):
        self._motion_sel_apply(True)

    def motion_restore_sel(self):
        self._motion_sel_apply(False)

    def _motion_sel_apply(self, exclude):
        if self.cur is None or self._sel_range is None:
            self.status("먼저 그래프에서 구간을 가로 드래그로 선택하세요")
            return
        a, b = self._sel_range
        m = self.meta[self.cur]
        exc = m.get("excluded") or set()
        rng = set(range(a, b + 1))
        m["excluded"] = (exc | rng) if exclude else (exc - rng)
        self.refresh_listbox()
        self._motion_refresh()
        self.draw_frame()

    def motion_batch_all(self):
        if self._exporting or self._busy:
            return
        thr = self._motion_thr()
        if thr is None:
            self.status("임계값을 먼저 입력하거나 '자동 제안'을 누르세요")
            return
        idxs = [(i, str(self.files[i])) for i in self.files
                if self.meta[i]["include"]]
        if not idxs:
            return
        crop = tuple(self.crop) if self.crop else None
        self._busy = True
        if self._motion_batch_btn is not None:
            self._motion_batch_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._motion_batch_worker,
                         args=(idxs, thr, crop), daemon=True).start()
        self.root.after(100, self._poll_queue)

    def _motion_batch_worker(self, idxs, thr, crop):
        try:
            for n, (i, path) in enumerate(idxs, 1):
                self._q.put(("info", f"모션 감지 {n}/{len(idxs)}: rl_{i}"))
                arr = load_stack(path)
                if crop:
                    x0, x1, y0, y1 = crop
                    arr = arr[:, y0:y1, x0:x1]
                mmet = motion_metric(arr)
                exc = set(int(v) for v in np.nonzero(mmet > thr)[0])
                self._q.put(("motion", i, mmet, exc))
            self._q.put(("mdone",))
        except Exception as e:
            self._q.put(("merr", repr(e)))

    # --------------------------- 설정 저장 ---------------------------
    def save_settings(self):
        if not self.files:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".json", initialdir=str(self.folder),
            initialfile="depth_cycle_settings.json",
            filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = dict(
            folder=str(self.folder), prefix=self.prefix,
            mag_mode=self._active_mode,
            modes={k: dict(v) for k, v in self.mode_store.items()},
            fps=self.fps_var.get(), scale=self.scale_var.get(),
            roundtrip=bool(self.roundtrip_var.get()), output=self.out_var.get(),
            crop=list(self.crop) if self.crop else None,
            norm_mode=self.norm_mode_var.get(),
            auto_plo=self.plo_var.get(), auto_phi=self.phi_var.get(),
            scalebar=bool(self.scalebar_var.get()),
            um_per_px=self.umpp_var.get(), bar_um=self.barum_var.get(),
            label_pos=POS_CODES.get(self.labelpos_var.get(), "auto"),
            bar_pos=POS_CODES.get(self.barpos_var.get(), "auto"),
            src_mode=self.src_var.get(),
            demotion=self.demo_var.get(),
            label_show=bool(self.label_show_var.get()),
            seg_order=([self.files[k].name for k in self.files]
                       if self.src_var.get() == "pulse" else None),
            reg_roi=list(self.reg_roi) if self.reg_roi else None,
            reg_ref=(self.files[self.reg_ref].name
                     if (self.src_var.get() == "pulse"
                         and self.reg_ref in self.files)
                     else self.reg_ref),
            depths={(self.files[i].name
                     if self.src_var.get() == "pulse" else str(i)): dict(
                        {k: m[k] for k in
                         ("include", "start", "end", "cycles",
                          "vmin", "vmax")},
                        excluded=sorted(int(v) for v in
                                        (m.get("excluded") or ())))
                    for i, m in self.meta.items()})
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.status(f"설정 저장: {p}")

    def load_settings(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if data.get("src_mode") in ("depth", "pulse"):
            self.src_var.set(data["src_mode"])
            self._active_src = data["src_mode"]
        folder = Path(data.get("folder", ""))
        if folder.is_dir():
            self.scan_folder(folder, quiet=True)
        pulse = self.src_var.get() == "pulse"
        if pulse and self.files:
            order_names = data.get("seg_order") or []
            name2key = {self.files[k].name: k for k in self.files}
            newkeys = [name2key[n] for n in order_names if n in name2key]
            newkeys += [k for k in self.files if k not in newkeys]
            self.files = {n: self.files[k] for n, k in enumerate(newkeys)}
            self.meta = {n: self.meta[k] for n, k in enumerate(newkeys)}
            self.seg_names = {n: self.seg_names[k]
                              for n, k in enumerate(newkeys)}
            self.cur = None
            self.stack = None
        for k, v in data.get("depths", {}).items():
            if pulse:
                match = [kk for kk in self.files
                         if self.files[kk].name == k]
                if not match:
                    continue
                i = match[0]
            else:
                try:
                    i = int(k)
                except ValueError:
                    continue
            if i in self.meta:
                for kk in ("include", "start", "end", "cycles", "vmin", "vmax"):
                    if kk in v:
                        self.meta[i][kk] = v[kk]
                if "excluded" in v:
                    nmax = self.meta[i]["n"]
                    try:
                        self.meta[i]["excluded"] = set(
                            int(x) for x in v["excluded"]
                            if 0 <= int(x) < nmax)
                    except (TypeError, ValueError):
                        pass
                self.meta[i]["end"] = min(self.meta[i]["end"],
                                          self.meta[i]["n"] - 1)
        if "fps" in data:
            self.fps_var.set(str(data["fps"]))
        if "scale" in data:
            self.scale_var.set(str(data["scale"]))
        if "output" in data:
            self.out_var.set(data["output"])
        if "roundtrip" in data:
            self.roundtrip_var.set(bool(data["roundtrip"]))
        if "norm_mode" in data:
            self.norm_mode_var.set(data["norm_mode"])
        if "auto_plo" in data:
            self.plo_var.set(str(data["auto_plo"]))
        if "auto_phi" in data:
            self.phi_var.set(str(data["auto_phi"]))
        if "scalebar" in data:
            self.scalebar_var.set(bool(data["scalebar"]))
        modes = data.get("modes")
        if isinstance(modes, dict):
            for k in ("high", "low"):
                if k in modes and isinstance(modes[k], dict):
                    for kk in ("dz_um", "um_per_px", "bar_um"):
                        if kk in modes[k]:
                            try:
                                self.mode_store[k][kk] = float(modes[k][kk])
                            except (TypeError, ValueError):
                                pass
        if data.get("mag_mode") in MAG_PRESETS:
            self.mag_var.set(data["mag_mode"])
            self._active_mode = data["mag_mode"]
        if modes is None:                      # 구버전 설정 파일 호환
            st = self.mode_store[self._active_mode]
            for key in ("dz_um", "um_per_px", "bar_um"):
                if key in data:
                    try:
                        st[key] = float(data[key])
                    except (TypeError, ValueError):
                        pass
        self._load_mode_entries()
        if "label_pos" in data:
            self.labelpos_var.set(
                POS_LABELS.get(data["label_pos"], POS_LABELS["auto"]))
        if "bar_pos" in data:
            self.barpos_var.set(
                POS_LABELS.get(data["bar_pos"], POS_LABELS["auto"]))
        if data.get("demotion") in ("none", "segment", "frame"):
            self.demo_var.set(data["demotion"])
        if "label_show" in data:
            self.label_show_var.set(bool(data["label_show"]))
        c = data.get("crop")
        self.crop = [int(v) for v in c] if c else None
        rr = data.get("reg_roi")
        self.reg_roi = [int(v) for v in rr] if rr else None
        rref = data.get("reg_ref")
        self.reg_ref = None
        if rref is not None:
            if pulse:
                for kk in self.files:
                    if self.files[kk].name == rref:
                        self.reg_ref = kk
                        break
            else:
                try:
                    r_i = int(rref)
                    if r_i in self.files:
                        self.reg_ref = r_i
                except (TypeError, ValueError):
                    pass
        self._ref_cache = None
        self._update_regref_label()
        self.refresh_listbox(select=0 if (pulse and self.files) else None)
        if self.cur is not None:
            self.on_norm_mode_change()
            self._after_crop_changed()
        self.status(f"설정 불러옴: {p}")

    # --------------------------- 내보내기 ---------------------------
    def browse_out(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialdir=str(self.folder) if self.folder else DEFAULT_DIR,
            initialfile=Path(self.out_var.get()).name if self.out_var.get()
            else "depth_cycle_video.mp4",
            filetypes=[("MP4", "*.mp4")])
        if p:
            self.out_var.set(p)

    def start_export(self):
        if self._exporting or self._busy:
            return
        sel = [i for i in self.files if self.meta[i]["include"]]
        if not sel:
            messagebox.showwarning("내보내기", "포함된 depth가 없습니다.")
            return
        order = sorted(sel)
        if self.roundtrip_var.get() and len(order) > 1:
            order = order + order[-2::-1]
        try:
            fps = max(1, int(float(self.fps_var.get())))
            scale = float(self.scale_var.get())
        except ValueError:
            messagebox.showwarning("내보내기", "FPS/배율 값을 확인하세요.")
            return
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning("내보내기", "출력 경로를 지정하세요.")
            return
        crop = None
        if self.crop:
            x0, x1, y0, y1 = (int(v) for v in self.crop)
            x1 -= (x1 - x0) % 2            # yuv420p용 짝수 크기 스냅
            y1 -= (y1 - y0) % 2
            crop = (x0, x1, y0, y1)
        plo, phi = self._auto_percentiles()
        norm = dict(mode=self.norm_mode_var.get(), plo=plo, phi=phi)
        sbcfg = dict(on=bool(self.scalebar_var.get()))
        try:
            sbcfg["umpp"] = float(self.umpp_var.get())
            sbcfg["bum"] = float(self.barum_var.get())
        except ValueError:
            st = self.mode_store[self._active_mode]
            sbcfg["umpp"], sbcfg["bum"] = st["um_per_px"], st["bar_um"]
        if sbcfg["umpp"] <= 0 or sbcfg["bum"] <= 0:
            sbcfg["on"] = False
        ov = dict(label_pos=POS_CODES.get(self.labelpos_var.get(), "auto"),
                  bar_pos=POS_CODES.get(self.barpos_var.get(), "auto"))
        demo = self.demo_var.get()
        show_label = bool(self.label_show_var.get())
        pulse = self.src_var.get() == "pulse"
        keys_all = list(self.files)
        refspec = None
        regroi = tuple(int(v) for v in self.reg_roi) if self.reg_roi else None
        if demo != "none":
            rk = self.reg_ref if self.reg_ref in self.files else order[0]
            rm = self.meta[rk]
            refspec = dict(path=str(self.files[rk]),
                           start=rm["start"], end=rm["end"],
                           excluded=sorted(int(v) for v in
                                           (rm.get("excluded") or ())),
                           name=self.item_title(rk))
        jobs = []
        for i in order:
            m = self.meta[i]
            exc = sorted(k for k in (m.get("excluded") or ())
                         if m["start"] <= k <= m["end"])
            if (m["end"] - m["start"] + 1) - len(exc) <= 0:
                messagebox.showwarning(
                    "내보내기",
                    f"{self.item_title(i)}: frame 범위 내 모든 프레임이 "
                    f"제외되어 있습니다.")
                return
            if not show_label:
                lab = ""
            elif pulse:
                lab = f"pulse {keys_all.index(i) + 1}"
            else:
                lab = f"z = {self.depth_um(i)} \u00b5m"
            bounds = (self.pulse_bounds.get(str(self.files[i]))
                      if (pulse and show_label) else None)
            jobs.append(dict(idx=i, path=str(self.files[i]),
                             start=m["start"], end=m["end"],
                             cycles=m["cycles"], excluded=exc,
                             vmin=m["vmin"], vmax=m["vmax"],
                             label=lab,
                             bounds=[tuple(b) for b in bounds]
                             if bounds else None))
        self._exporting = True
        self.export_btn.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        threading.Thread(target=self._export_worker,
                         args=(jobs, out, fps, scale, crop, norm, sbcfg,
                               ov, demo, refspec, regroi),
                         daemon=True).start()
        self.root.after(100, self._poll_queue)

    def _prepass_corner_scores(self, jobs, crop, norm):
        """포함 depth마다 프레임 3장을 샘플링해 corner 밝기 점수 합산 (자동 배치용)"""
        sums = {c: 0.0 for c in CORNERS}
        seen = set()
        for j in jobs:
            i = j["idx"]
            if i in seen:
                continue
            seen.add(i)
            frames = []
            try:
                with tifffile.TiffFile(j["path"]) as tf:
                    npg = len(tf.pages)
                    if npg > 1:
                        hi_k = min(j["end"], npg - 1)
                        excset = set(j.get("excluded", ()))
                        cand = [k for k in range(j["start"], hi_k + 1)
                                if k not in excset]
                        if not cand:
                            cand = list(range(j["start"], hi_k + 1))
                        sel = np.unique(np.linspace(
                            0, len(cand) - 1, min(3, len(cand))).astype(int))
                        for k in (cand[int(s)] for s in sel):
                            a = np.asarray(tf.pages[int(k)].asarray())
                            if a.ndim == 3:
                                a = a[..., :3].mean(axis=-1)
                            frames.append(a.astype(np.float32))
            except Exception:
                frames = []
            if not frames:
                arr = load_stack(j["path"])
                excset = set(j.get("excluded", ()))
                cand = [k for k in range(j["start"], j["end"] + 1)
                        if k not in excset]
                if not cand:
                    cand = list(range(j["start"], j["end"] + 1))
                sel = np.unique(np.linspace(
                    0, len(cand) - 1, min(3, len(cand))).astype(int))
                frames = [arr[cand[int(s)]].astype(np.float32) for s in sel]
            sub = np.stack(frames)
            if crop:
                x0, x1, y0, y1 = crop
                sub = sub[:, y0:y1, x0:x1]
            if norm["mode"] == "auto":
                lo, hi = self._range_of(sub, norm["plo"], norm["phi"])
            else:
                lo, hi = j["vmin"], j["vmax"]
                if lo is None or hi is None:
                    lo, hi = self._range_of(sub, *DEFAULT_PERCENTILES)
            nrm = np.clip((sub - lo) / max(hi - lo, 1e-9),
                          0.0, 1.0).mean(axis=0)
            s = corner_scores(nrm)
            for c in CORNERS:
                sums[c] += s[c]
        return sums

    def _export_worker(self, jobs, out, fps, scale, crop, norm, sbcfg, ov,
                       demo, refspec, regroi):
        try:
            total = sum(((j["end"] - j["start"] + 1)
                         - len(j.get("excluded", ()))) * j["cycles"]
                        for j in jobs)
            remaining = {}
            for j in jobs:
                remaining[j["idx"]] = remaining.get(j["idx"], 0) + 1

            # 오버레이 자동 배치: crop 영역 corner 밝기 분석 → 어두운 corner 선택
            lp, bp = ov["label_pos"], ov["bar_pos"]
            if lp == "auto" or (bp == "auto" and sbcfg["on"]):
                self._q.put(("info", "오버레이 위치 계산 중 (corner 밝기 분석)..."))
                scores = self._prepass_corner_scores(jobs, crop, norm)
                lp, bp = resolve_corners(lp, bp, scores)
            if lp == "auto":
                lp = "tl"
            if bp == "auto":
                bp = "br"
            self._q.put(("info",
                         f"오버레이 배치 — Z 라벨: {POS_LABELS[lp]}, "
                         f"scale bar: {POS_LABELS[bp]}"))

            writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                        quality=QUALITY,
                                        pixelformat="yuv420p",
                                        macro_block_size=2)
            cache = {}
            done = 0
            meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
            ref_prep = None

            def reg2d(fr):
                if regroi:
                    rx0, rx1, ry0, ry1 = regroi
                    return fr[ry0:ry1, rx0:rx1]
                if crop:
                    cx0, cx1, cy0, cy1 = crop
                    return fr[cy0:cy1, cx0:cx1]
                return fr

            if demo != "none" and refspec is not None:
                self._q.put(("info", f"정합 기준 로드: {refspec['name']}"))
                arr0 = load_stack(refspec["path"])
                exc0 = set(refspec["excluded"])
                kept0 = [k for k in range(refspec["start"],
                                          refspec["end"] + 1)
                         if k not in exc0] or list(range(arr0.shape[0]))
                rs0 = kept0[::max(1, len(kept0) // 30)][:30]
                ref_prep = prep_reg(np.median(np.stack(
                    [reg2d(arr0[k]).astype(np.float32) for k in rs0]),
                    axis=0))
                del arr0
                self._q.put(("info",
                             f"정합 기준 준비 완료 "
                             f"(영역={'정합ROI' if regroi else 'crop/전체'})"))
            for j in jobs:
                i = j["idx"]
                if i not in cache:
                    cache[i] = load_stack(j["path"])
                stack_full = cache[i]
                excset = set(j.get("excluded", ()))
                kept = [k for k in range(j["start"], j["end"] + 1)
                        if k not in excset]

                def crop2d(fr):
                    if crop:
                        cx0, cx1, cy0, cy1 = crop
                        return fr[cy0:cy1, cx0:cx1]
                    return fr

                if demo != "none" and ref_prep is not None:
                    if demo == "segment":
                        rs = kept[::max(1, len(kept) // 30)][:30]
                        med = np.median(np.stack(
                            [reg2d(stack_full[k]).astype(np.float32)
                             for k in rs]), axis=0)
                        dy0, dx0 = register_shift(ref_prep, med)
                        shifts = [(dy0, dx0)] * len(kept)
                        self._q.put(("info",
                                     f"정합 {j['label'] or j['idx']}: "
                                     f"({dy0:+.1f}, {dx0:+.1f}) px"))
                    else:
                        shifts = [register_shift(ref_prep,
                                                 reg2d(stack_full[k]))
                                  for k in kept]
                        mags = [abs(a_) + abs(b_) for a_, b_ in shifts]
                        self._q.put(("info",
                                     f"정합(프레임별) {j['label'] or j['idx']}"
                                     f": max |shift| = {max(mags):.1f} px"))
                    frs = []
                    for k, (dy_, dx_) in zip(kept, shifts):
                        fr = stack_full[k]
                        if abs(dy_) > 0.05 or abs(dx_) > 0.05:
                            fr = shift_image(fr, dy_, dx_)
                        frs.append(np.asarray(crop2d(fr), np.float32))
                    sub = np.stack(frs)
                else:
                    sub = stack_full[j["start"]: j["end"] + 1]
                    if crop:
                        cx0, cx1, cy0, cy1 = crop
                        sub = sub[:, cy0:cy1, cx0:cx1]
                    if excset:
                        keep = [k - j["start"] for k in kept]
                        sub = sub[keep]

                # depth별 대비 범위
                if norm["mode"] == "auto":
                    lo, hi = self._range_of(sub, norm["plo"], norm["phi"])
                else:
                    lo, hi = j["vmin"], j["vmax"]
                    if lo is None or hi is None:
                        lo, hi = self._range_of(sub, *DEFAULT_PERCENTILES)

                h, w = sub.shape[1:]
                ow = max(2, int(w * scale)) // 2 * 2
                oh = max(2, int(h * scale)) // 2 * 2
                need_resize = (ow, oh) != (w, h)
                ratio = ow / w

                marg = max(8, min(ow, oh) // 30)
                label = j["label"]
                bounds = j.get("bounds")
                labels_seq = None
                if label and bounds:
                    labels_seq = [pulse_label_at(bounds, k) or label
                                  for k in kept]
                    self._q.put((
                        "info",
                        f"{Path(j['path']).stem}: boundaries 기반 "
                        f"프레임별 pulse 라벨 ({len(bounds)}구간)"))
                label_font = get_font(max(12, int(oh * 0.06)))
                lth = getattr(label_font, "size", 14)
                ly = marg if lp in ("tl", "tr") \
                    else max(2, int(oh - marg - lth * 1.3))
                _lx_cache = {}

                def _label_x(txt):
                    if txt not in _lx_cache:
                        try:
                            ltw = meas.textlength(txt, font=label_font)
                        except Exception:
                            ltw = len(txt) * lth * 0.6
                        _lx_cache[txt] = marg if lp in ("tl", "bl") \
                            else max(2, int(ow - marg - ltw))
                    return _lx_cache[txt]
                if sbcfg["on"]:
                    bar_px = max(2, int(round(
                        sbcfg["bum"] / sbcfg["umpp"] * ratio)))
                    bar_px = min(bar_px, ow - 2 * marg)
                    bh = max(3, oh // 150)
                    sb_font = get_font(max(10, int(oh * 0.05)))
                    sb_txt = f'{sbcfg["bum"]:g} \u00b5m'
                    try:
                        stw = meas.textlength(sb_txt, font=sb_font)
                    except Exception:
                        stw = len(sb_txt) * getattr(sb_font, "size", 12) * 0.6
                    sth = getattr(sb_font, "size", 12)
                    bx0 = marg if bp in ("tl", "bl") else ow - marg - bar_px
                    if bp in ("bl", "br"):
                        by1 = oh - marg
                        by0 = by1 - bh
                        sty = by0 - sth - 5
                    else:
                        sty = marg
                        by0 = marg + sth + 5
                        by1 = by0 + bh
                    stx = max(2, min(bx0 + bar_px / 2 - stw / 2,
                                     ow - stw - 2))

                for _ in range(j["cycles"]):
                    for k in range(sub.shape[0]):
                        f = sub[k].astype(np.float32)
                        u8 = np.clip((f - lo) / max(hi - lo, 1e-9) * 255.0,
                                     0, 255).astype(np.uint8)
                        img = Image.fromarray(u8).convert("RGB")
                        if need_resize:
                            img = img.resize((ow, oh), Image.BILINEAR)
                        d = ImageDraw.Draw(img)
                        cur_lab = (labels_seq[k] if labels_seq else label)
                        if cur_lab:
                            d.text((_label_x(cur_lab), ly), cur_lab,
                                   font=label_font,
                                   fill="white", stroke_width=2,
                                   stroke_fill="black")
                        if sbcfg["on"]:
                            d.rectangle([bx0, by0, bx0 + bar_px, by1],
                                        fill="white", outline="black")
                            d.text((stx, sty), sb_txt, font=sb_font,
                                   fill="white", stroke_width=2,
                                   stroke_fill="black")
                        writer.append_data(np.asarray(img))
                        done += 1
                        if done % 20 == 0 or done == total:
                            self._q.put(("prog", done, total))
                remaining[i] -= 1
                if remaining[i] == 0:
                    del cache[i]
            writer.close()
            self._q.put(("done", out, total, fps))
        except Exception as e:
            self._q.put(("err", repr(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "prog":
                    _, done, total = msg
                    self.progress.configure(maximum=total, value=done)
                    self.status_var.set(f"내보내는 중... {done}/{total}")
                elif msg[0] == "info":
                    self.status_var.set(msg[1])
                elif msg[0] == "done":
                    _, out, total, fps = msg
                    self._exporting = False
                    self.export_btn.configure(state=tk.NORMAL)
                    self.progress["value"] = self.progress["maximum"]
                    self.status_var.set("완료")
                    messagebox.showinfo(
                        "완료",
                        f"{out}\n{total} frames, {total / fps:.1f} s @ {fps} fps")
                elif msg[0] == "err":
                    self._exporting = False
                    self.export_btn.configure(state=tk.NORMAL)
                    self.status_var.set("오류")
                    messagebox.showerror("내보내기 오류", msg[1])
                elif msg[0] == "motion":
                    _, i, marr, exc = msg
                    if i in self.meta:
                        self.meta[i]["motion"] = marr
                        self.meta[i]["excluded"] = set(exc)
                elif msg[0] == "mdone":
                    self._busy = False
                    if self._motion_batch_btn is not None:
                        try:
                            self._motion_batch_btn.configure(state=tk.NORMAL)
                        except Exception:
                            pass
                    self.refresh_listbox()
                    self._motion_refresh_safe()
                    self.draw_frame()
                    self.status_var.set("모션 일괄 감지 완료")
                elif msg[0] == "merr":
                    self._busy = False
                    if self._motion_batch_btn is not None:
                        try:
                            self._motion_batch_btn.configure(state=tk.NORMAL)
                        except Exception:
                            pass
                    self.status_var.set("모션 감지 오류")
                    messagebox.showerror("모션 감지 오류", msg[1])
        except queue.Empty:
            pass
        if self._exporting or self._busy:
            self.root.after(100, self._poll_queue)


def main():
    root = tk.Tk()
    DepthCycleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()