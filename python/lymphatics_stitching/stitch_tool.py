# NIR-II SLIM - extended-field stitching - produces Fig. 3d, 3j
# environment: stitch_py311
"""SLIM Reconstruction Stitching Tool.

Interactive tool to:
  1. Preview raw data with frame scrollbar / step buttons / direct entry
     (multiple raw files concatenate into one continuous timeline)
  2. Reconstruct selected frames
  3. Drag reconstructed tiles around the canvas to position the stitch
  4. Blend overlapping regions
  5. Export as 3D volume or depth-coded MIP
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import os

from slim_recon.pipeline import SLIMPipeline
from slim_recon import data_io


class StitchTool:
    """Interactive stitching tool for SLIM reconstruction."""

    DEFAULT_RAW = r"E:\Selected data\260209_mouse_lym\Stitch\record_09022026_142008_20hz.raw"
    DEFAULT_GEO = r"E:\Selected data\260209_mouse_lym\260209_geo_1.mat"
    DEFAULT_PSF = r"E:\Selected data\260209_mouse_lym\260209_psf_1.mat"

    def __init__(self, root):
        self.root = root
        self.root.title("SLIM Stitch Tool")
        self.root.geometry("1400x900")

        # raw_files: list of {'path': str, 'data': (H,W,Ni), 'num_frames': int}.
        # The frame slider/entry index is a *virtual* index into the concatenation
        # of every loaded file's frames, so stitching can span multiple raws seamlessly.
        self.raw_files = []
        self.pipe = None          # SLIMPipeline (shared calibration)
        self.tiles = []           # list of {'frame_idx', 'volume', 'mip', 'mip_idx', 'x', 'y', 'label'}
        self.canvas_size = [4096, 4096]  # canvas (W, H)
        self.selected_tiles = set()  # multi-select: set of tile indices
        self.drag_offset = None
        self.drag_start = None       # for rubber-band selection
        self.rubber_rect = None
        self.dragging = False        # True while a tile drag is in progress
        self._mouse_cids = []        # matplotlib mpl_connect ids for the stitch canvas
        self._configure_after_id = None  # debounce token for Tk <Configure>
        # Visible region in canvas-pixel coords as (x0, x1, y0, y1); None = fit to canvas.
        self.zoom_view = None

        self._build_gui()

    def _build_gui(self):
        # ── Top: file paths ──
        pf = ttk.LabelFrame(self.root, text="File Paths")
        pf.pack(fill=tk.X, padx=8, pady=4)

        self.raw_var = tk.StringVar(value=self.DEFAULT_RAW)
        self.geo_var = tk.StringVar(value=self.DEFAULT_GEO)
        self.psf_var = tk.StringVar(value=self.DEFAULT_PSF)

        # Raw: multi-file listbox so frames from several raw files concatenate into one timeline.
        raw_row = ttk.Frame(pf); raw_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(raw_row, text="Raw:", width=5).pack(side=tk.LEFT, anchor=tk.N)
        self.raw_listbox = tk.Listbox(raw_row, height=3,
                                       selectmode=tk.EXTENDED, exportselection=False)
        self.raw_listbox.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        raw_btns = ttk.Frame(raw_row); raw_btns.pack(side=tk.LEFT, padx=2)
        ttk.Button(raw_btns, text="Add...", width=8,
                   command=self._add_raw_files).pack(pady=1)
        ttk.Button(raw_btns, text="Remove", width=8,
                   command=self._remove_raw_file).pack(pady=1)
        ttk.Button(raw_btns, text="Clear", width=8,
                   command=self._clear_raw_files).pack(pady=1)

        # Geo / PSF: single-file pickers.
        for lbl, var, ft in [
            ("Geo:", self.geo_var, [("MAT", "*.mat"), ("All", "*.*")]),
            ("PSF:", self.psf_var, [("MAT", "*.mat"), ("All", "*.*")]),
        ]:
            row = ttk.Frame(pf); row.pack(fill=tk.X, padx=4, pady=1)
            ttk.Label(row, text=lbl, width=5).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=70).pack(side=tk.LEFT, padx=2)
            ttk.Button(row, text="...",
                       command=lambda v=var, f=ft: self._browse(v, f)).pack(side=tk.LEFT)

        # ── Parameters row ──
        par = ttk.Frame(pf); par.pack(fill=tk.X, padx=4, pady=3)

        self.num_frames_var = tk.IntVar(value=1000)
        self.clip_lo_var = tk.DoubleVar(value=0.3)
        self.clip_hi_var = tk.DoubleVar(value=0.999)
        self.n_depth_var = tk.IntVar(value=32)
        self.psf_n_var = tk.IntVar(value=2)
        self.n_iters_var = tk.IntVar(value=7)

        for lbl, var, w in [("Frames:", self.num_frames_var, 5),
                             ("Clip Lo:", self.clip_lo_var, 5),
                             ("Clip Hi:", self.clip_hi_var, 5),
                             ("N Depth:", self.n_depth_var, 4),
                             ("PSF N:", self.psf_n_var, 3),
                             ("Iters:", self.n_iters_var, 4)]:
            ttk.Label(par, text=lbl).pack(side=tk.LEFT, padx=(4, 0))
            ttk.Entry(par, textvariable=var, width=w).pack(side=tk.LEFT, padx=(2, 6))

        ttk.Button(par, text="Load Calib", command=self._load_calib).pack(side=tk.LEFT, padx=5)

        # Scale bar settings
        ttk.Separator(par, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Label(par, text="px(um):").pack(side=tk.LEFT)
        self.pixel_size_var = tk.DoubleVar(value=11.0)
        ttk.Entry(par, textvariable=self.pixel_size_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(par, text="Bar(um):").pack(side=tk.LEFT)
        self.scalebar_um_var = tk.DoubleVar(value=100.0)
        ttk.Entry(par, textvariable=self.scalebar_um_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(par, text="Color:").pack(side=tk.LEFT)
        self.scalebar_color_var = tk.StringVar(value='white')
        ttk.Combobox(par, textvariable=self.scalebar_color_var, width=6, state='readonly',
                     values=['white', 'red', 'yellow', 'cyan', 'green']).pack(side=tk.LEFT, padx=(2, 0))

        # Depth range (μm) drives the numerical labels on the depth colorbar
        # in the MIP exports. Defaults are placeholders — set these to the
        # acquisition's real near/far depth so the saved figure's colorbar
        # ticks read the right physical values.
        ttk.Label(par, text="Depth Lo(um):").pack(side=tk.LEFT, padx=(8, 0))
        self.depth_lo_um_var = tk.DoubleVar(value=0.0)
        ttk.Entry(par, textvariable=self.depth_lo_um_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(par, text="Depth Hi(um):").pack(side=tk.LEFT)
        self.depth_hi_um_var = tk.DoubleVar(value=200.0)
        ttk.Entry(par, textvariable=self.depth_hi_um_var, width=5).pack(side=tk.LEFT, padx=(2, 4))

        # ── Middle: left=preview, right=canvas ──
        # Grid (with equal-weight uniform columns) guarantees both panels split
        # the available width 50/50 on resize. Plain pack() can give the panel
        # with the wider natural minimum (the canvas, due to its many control
        # buttons) less of the extra space when the window is maximized.
        mid = ttk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        mid.columnconfigure(0, weight=1, uniform='mid')
        mid.columnconfigure(1, weight=1, uniform='mid')
        mid.rowconfigure(0, weight=1)

        # Left panel: raw frame preview
        left = ttk.LabelFrame(mid, text="Raw Frame Preview")
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        prev_ctrl = ttk.Frame(left); prev_ctrl.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(prev_ctrl, text="Frame:").pack(side=tk.LEFT)
        self.frame_idx_var = tk.IntVar(value=0)
        self.frame_entry_var = tk.StringVar(value='0')

        ttk.Button(prev_ctrl, text="◂◂", width=3,
                   command=lambda: self._step_frame(-10)).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(prev_ctrl, text="◂", width=2,
                   command=lambda: self._step_frame(-1)).pack(side=tk.LEFT)

        self.frame_slider = ttk.Scale(prev_ctrl, from_=0, to=0,
                                       variable=self.frame_idx_var,
                                       command=self._on_frame_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        ttk.Button(prev_ctrl, text="▸", width=2,
                   command=lambda: self._step_frame(1)).pack(side=tk.LEFT)
        ttk.Button(prev_ctrl, text="▸▸", width=3,
                   command=lambda: self._step_frame(10)).pack(side=tk.LEFT, padx=(0, 4))

        self.frame_entry = ttk.Entry(prev_ctrl, textvariable=self.frame_entry_var, width=6)
        self.frame_entry.pack(side=tk.LEFT)
        self.frame_entry.bind('<Return>', lambda e: self._jump_to_frame())
        self.frame_entry.bind('<FocusOut>', lambda e: self._jump_to_frame())
        self.frame_label = ttk.Label(prev_ctrl, text="/ 0")
        self.frame_label.pack(side=tk.LEFT, padx=(2, 4))

        ttk.Button(prev_ctrl, text="Reconstruct & Add",
                   command=self._reconstruct_frame).pack(side=tk.LEFT, padx=4)

        # Keyboard shortcuts: arrows step, Shift+arrows = ±10, Home/End = endpoints.
        self.root.bind('<Left>', lambda e: self._kbd_step(-1))
        self.root.bind('<Right>', lambda e: self._kbd_step(1))
        self.root.bind('<Shift-Left>', lambda e: self._kbd_step(-10))
        self.root.bind('<Shift-Right>', lambda e: self._kbd_step(10))
        self.root.bind('<Home>', lambda e: self._kbd_step(-10**9))
        self.root.bind('<End>', lambda e: self._kbd_step(10**9))
        # Mouse wheel over the preview steps one frame per notch.
        self.preview_wheel_target = None  # set after preview_canvas is built

        self.preview_fig = Figure(figsize=(4, 4), dpi=100, facecolor='black')
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax.axis('off')
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, left)
        prev_widget = self.preview_canvas.get_tk_widget()
        prev_widget.pack(fill=tk.BOTH, expand=True)
        prev_widget.bind('<MouseWheel>', self._on_preview_wheel)
        prev_widget.bind('<Button-4>', lambda e: self._step_frame(-1))  # X11 scroll up
        prev_widget.bind('<Button-5>', lambda e: self._step_frame(1))   # X11 scroll down
        self.preview_im = None

        # Right panel: stitch canvas
        right = ttk.LabelFrame(mid, text="Stitch Canvas (drag tiles to reposition)")
        right.grid(row=0, column=1, sticky='nsew', padx=(4, 0))

        canvas_ctrl = ttk.Frame(right); canvas_ctrl.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(canvas_ctrl, text="Canvas W:").pack(side=tk.LEFT)
        self.canvas_w_var = tk.IntVar(value=4096)
        ttk.Entry(canvas_ctrl, textvariable=self.canvas_w_var, width=6).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(canvas_ctrl, text="H:").pack(side=tk.LEFT)
        self.canvas_h_var = tk.IntVar(value=4096)
        ttk.Entry(canvas_ctrl, textvariable=self.canvas_h_var, width=6).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(canvas_ctrl, text="Resize", command=self._resize_canvas).pack(side=tk.LEFT, padx=4)
        ttk.Button(canvas_ctrl, text="Refresh", command=self._redraw_canvas).pack(side=tk.LEFT, padx=4)

        # Zoom controls. Mouse wheel over the canvas zooms around the cursor;
        # buttons zoom around the centre of the visible area.
        ttk.Button(canvas_ctrl, text="Zoom −", width=7,
                   command=self._zoom_out).pack(side=tk.LEFT, padx=(8, 1))
        ttk.Button(canvas_ctrl, text="Zoom +", width=7,
                   command=self._zoom_in).pack(side=tk.LEFT, padx=1)
        ttk.Button(canvas_ctrl, text="Fit", width=4,
                   command=self._zoom_fit).pack(side=tk.LEFT, padx=1)

        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(canvas_ctrl, text="Show outlines & labels",
                        variable=self.show_labels_var,
                        command=self._redraw_canvas).pack(side=tk.LEFT, padx=8)

        # Blend toggle: ON = overlap-averaged (export-equivalent preview);
        # OFF = max-projection per pixel so each tile stays fully visible
        # while you're aligning. Exports always use the blended path.
        self.blend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(canvas_ctrl, text="Blend overlaps",
                        variable=self.blend_var,
                        command=self._redraw_canvas).pack(side=tk.LEFT, padx=4)

        self.stitch_fig = Figure(figsize=(5, 5), dpi=100, facecolor='black')
        self.stitch_ax = self.stitch_fig.add_subplot(111)
        self.stitch_ax.axis('off')
        self.stitch_mpl_canvas = FigureCanvasTkAgg(self.stitch_fig, right)
        stitch_widget = self.stitch_mpl_canvas.get_tk_widget()
        stitch_widget.pack(fill=tk.BOTH, expand=True)
        self.stitch_im = None

        # Mouse events for dragging — bound through a helper so we can
        # cleanly re-establish them after a panel/canvas resize.
        self._bind_canvas_events()
        # When the Tk widget hosting the figure is resized (window resize or
        # explicit canvas Resize), reset drag state and re-bind once the resize
        # settles so drag-to-move keeps working. Use add='+' so this hook runs
        # *in addition to* matplotlib's own <Configure> handler — without it,
        # our handler replaces FigureCanvasTkAgg's internal resize callback and
        # the stitch figure stops growing with its widget on full-screen.
        stitch_widget.bind('<Configure>', self._on_canvas_widget_configure, add='+')

        # ── Bottom: tile list + export ──
        bot = ttk.Frame(self.root)
        bot.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(bot, text="Tiles:").pack(side=tk.LEFT)
        self.tile_listbox = tk.Listbox(bot, height=3, width=60)
        self.tile_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_frame = ttk.Frame(bot)
        btn_frame.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_tile).pack(pady=1)
        ttk.Button(btn_frame, text="Export MIP (Gray)", command=self._export_mip).pack(pady=1)
        ttk.Button(btn_frame, text="Export MIP (Color)", command=self._export_mip_color).pack(pady=1)
        ttk.Button(btn_frame, text="Export 3D Volume", command=self._export_volume).pack(pady=1)

        proj_frame = ttk.Frame(bot)
        proj_frame.pack(side=tk.LEFT, padx=4)
        ttk.Button(proj_frame, text="Save Project",
                   command=self._save_project).pack(pady=1)
        ttk.Button(proj_frame, text="Load Project",
                   command=self._load_project).pack(pady=1)

        # Status
        self.status_var = tk.StringVar(value="Ready — Load raw data to begin")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, padx=8, pady=(0, 4))

        # Init canvas
        self._redraw_canvas()

    # ── Helpers ────────────────────────────────────────────────
    def _browse(self, var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes)
        if p:
            var.set(p)

    def _status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ── Load raw data (multi-file) ─────────────────────────────
    def _add_raw_files(self):
        """Open a multi-select file dialog and append each chosen file's frames to the
        timeline. Accepts raw .raw stacks OR denoised/processed .tif/.tiff stacks (the
        output of the Shot Noise Suppression batch, e.g. '<stem>_denoised.tif' in a
        'denoised' subfolder) — both load to the same (H,W,T) in-memory format, so the
        timeline, preview, and reconstruction work identically regardless of source."""
        # Prefer a 'denoised' subfolder next to the last file if one exists, so the
        # next "Add..." dialog opens where the denoise batch wrote its outputs.
        last = self.raw_var.get()
        base_dir = os.path.dirname(last) or "."
        denoised_dir = os.path.join(base_dir, "denoised")
        initial_dir = denoised_dir if os.path.isdir(denoised_dir) else base_dir
        paths = filedialog.askopenfilenames(
            title="Add Raw / Denoised TIFF Files",
            initialdir=initial_dir,
            filetypes=[("Raw / TIFF", "*.raw *.tif *.tiff"),
                       ("Raw", "*.raw"),
                       ("TIFF", "*.tif *.tiff"),
                       ("All", "*.*")])
        if not paths:
            return
        self._status(f"Loading {len(paths)} file(s)...")
        n_loaded = 0
        for p in paths:
            if not os.path.exists(p):
                messagebox.showerror("Error", f"File not found:\n{p}")
                continue
            # Skip duplicates so re-adding the same file does not double-count frames.
            if any(f['path'] == p for f in self.raw_files):
                continue
            try:
                # load_data_auto picks raw vs TIFF by extension; both return (H,W,N).
                data = data_io.load_data_auto(p, num_frames=self.num_frames_var.get())
            except Exception as e:
                messagebox.showerror("Error",
                                     f"Failed to load {os.path.basename(p)}:\n{e}")
                continue
            self.raw_files.append({'path': p, 'data': data,
                                    'num_frames': data.shape[2]})
            self.raw_listbox.insert(tk.END,
                f"{os.path.basename(p)}  ({data.shape[2]} frames)")
            self.raw_var.set(p)  # next "Add..." dialog opens nearby
            n_loaded += 1
        self._refresh_frame_range()
        if n_loaded:
            H, W = self.raw_files[0]['data'].shape[:2]
            self._status(f"Loaded {len(self.raw_files)} file(s), "
                         f"{self._total_frames()} total frames ({H}x{W})")
        else:
            self._status("No new files loaded")

    def _remove_raw_file(self):
        """Remove the listbox-selected raw file(s) from the timeline.

        Reconstructed tiles already on the canvas keep their pixel data, but their
        ``frame_idx`` label refers to the virtual index at reconstruction time and
        will no longer line up with the current timeline.
        """
        sel = list(self.raw_listbox.curselection())
        if not sel:
            messagebox.showinfo("Info", "Select raw files in the list to remove")
            return
        for i in sorted(sel, reverse=True):
            self.raw_files.pop(i)
            self.raw_listbox.delete(i)
        self._refresh_frame_range()
        self._status(f"{len(self.raw_files)} raw file(s), "
                     f"{self._total_frames()} total frames")

    def _clear_raw_files(self):
        self.raw_files = []
        self.raw_listbox.delete(0, tk.END)
        self._refresh_frame_range()
        self._status("Raw file list cleared")

    def _total_frames(self):
        return sum(f['num_frames'] for f in self.raw_files)

    def _resolve_frame(self, virtual_idx):
        """Map a virtual frame index across all loaded raw files to (file_idx, local_idx)."""
        offset = 0
        for fi, f in enumerate(self.raw_files):
            if virtual_idx < offset + f['num_frames']:
                return fi, virtual_idx - offset
            offset += f['num_frames']
        raise IndexError(f"Frame {virtual_idx} out of range (total {offset})")

    def _refresh_frame_range(self):
        """Resync the slider/entry/preview after the raw file list changes."""
        N = self._total_frames()
        if N == 0:
            self.frame_slider.config(to=0)
            self.frame_idx_var.set(0)
            self.frame_entry_var.set("0")
            self.frame_label.config(text="/ 0")
            if self.preview_im is not None:
                self.preview_ax.clear()
                self.preview_ax.axis('off')
                self.preview_ax.set_title('')
                self.preview_im = None
                self.preview_canvas.draw_idle()
            return
        self.frame_slider.config(to=N - 1)
        cur = max(0, min(self.frame_idx_var.get(), N - 1))
        self.frame_idx_var.set(cur)
        self.frame_entry_var.set(str(cur))
        self.frame_label.config(text=f"/ {N - 1}")
        self._show_preview_frame(cur)

    # ── Load calibration ───────────────────────────────────────
    def _load_calib(self):
        geo = self.geo_var.get()
        psf = self.psf_var.get()
        if not os.path.exists(geo) or not os.path.exists(psf):
            messagebox.showerror("Error", "Geo or PSF file not found")
            return
        self._status("Loading calibration...")
        try:
            self.pipe = SLIMPipeline()
            self.pipe.load_calibration(geo, psf)
            self._status("Calibration loaded")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._status("Error loading calibration")

    # ── Preview frame ──────────────────────────────────────────
    def _on_frame_change(self, val=None):
        idx = int(float(val)) if val is not None else self.frame_idx_var.get()
        if self.raw_files:
            N = self._total_frames()
            idx = max(0, min(idx, N - 1))
            self.frame_label.config(text=f"/ {N - 1}")
            self.frame_entry_var.set(str(idx))
            self._show_preview_frame(idx)

    def _step_frame(self, delta):
        """Advance the previewed frame by ``delta`` across the concatenated timeline."""
        if not self.raw_files:
            return
        N = self._total_frames()
        cur = self.frame_idx_var.get()
        new = max(0, min(N - 1, cur + delta))
        if new == cur:
            return
        self.frame_idx_var.set(new)
        self._on_frame_change(new)

    def _jump_to_frame(self):
        """Read the entry box and seek to that virtual frame index."""
        if not self.raw_files:
            return
        try:
            idx = int(self.frame_entry_var.get())
        except (TypeError, ValueError):
            self.frame_entry_var.set(str(self.frame_idx_var.get()))
            return
        N = self._total_frames()
        idx = max(0, min(N - 1, idx))
        self.frame_idx_var.set(idx)
        self._on_frame_change(idx)

    def _kbd_step(self, delta):
        """Step from a global key binding, ignoring keys typed in text widgets."""
        w = self.root.focus_get()
        if w is not None and w.winfo_class() in (
            'Entry', 'TEntry', 'Spinbox', 'TSpinbox', 'Text',
            'Combobox', 'TCombobox',
        ):
            return
        self._step_frame(delta)

    def _on_preview_wheel(self, event):
        # Windows/macOS: event.delta in multiples of 120; up = previous frame.
        self._step_frame(-1 if event.delta > 0 else 1)

    def _show_preview_frame(self, idx):
        if not self.raw_files:
            return
        try:
            file_idx, local_idx = self._resolve_frame(idx)
        except IndexError:
            return
        frame = self.raw_files[file_idx]['data'][:, :, local_idx]
        vmin, vmax = np.percentile(frame, [1, 99.5])
        disp = np.clip((frame - vmin) / (vmax - vmin + 1e-10), 0, 1)

        if self.preview_im is None:
            self.preview_im = self.preview_ax.imshow(disp, cmap='gray',
                                                      vmin=0, vmax=1)
            self.preview_ax.axis('off')
        else:
            self.preview_im.set_data(disp)
        if len(self.raw_files) > 1:
            fname = os.path.basename(self.raw_files[file_idx]['path'])
            title = f'Frame {idx}  ({fname}:{local_idx})'
        else:
            title = f'Frame {idx}'
        self.preview_ax.set_title(title, color='white', fontsize=9)
        self.preview_canvas.draw_idle()

    # ── Reconstruct a frame ────────────────────────────────────
    def _reconstruct_frame(self):
        if not self.raw_files:
            messagebox.showwarning("Warning", "Load raw data first")
            return
        if self.pipe is None:
            messagebox.showwarning("Warning", "Load calibration first")
            return

        idx = self.frame_idx_var.get()
        file_idx, local_idx = self._resolve_frame(idx)
        file_info = self.raw_files[file_idx]
        fname = os.path.basename(file_info['path'])
        if len(self.raw_files) > 1:
            self._status(f"Reconstructing frame {idx} ({fname}:{local_idx})...")
        else:
            self._status(f"Reconstructing frame {idx}...")

        try:
            # Create a fresh pipeline copy with shared calibration
            pipe = SLIMPipeline()
            pipe.geo_cal = self.pipe.geo_cal
            pipe.psf_cal = self.pipe.psf_cal
            pipe.im_data = file_info['data']

            pipe.preprocess_frame(frame_idx=local_idx, num_avg=1,
                                   clip_range=(self.clip_lo_var.get(),
                                               self.clip_hi_var.get()))
            pipe.extract_views()
            pipe.prepare_reconstruction(depth_range=(0.0, 1.0),
                                         n_depth=self.n_depth_var.get(),
                                         psf_n=self.psf_n_var.get())
            pipe.reconstruct(n_iters=self.n_iters_var.get())

            vol = pipe.recon_result  # (H, W, D)
            # Normalize
            vmin, vmax = vol.min(), vol.max()
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            vol = vol.astype(np.float32)

            # MIP + depth index for canvas display
            mip = np.max(vol, axis=2)
            mip_idx = np.argmax(vol, axis=2)

            # Always spawn new tiles at the canvas centre. Earlier versions
            # staggered the spawn to avoid perfect stacking, but that drifted
            # toward the bottom-right and eventually wrapped or fell off the
            # canvas. The most-recently-added tile is drawn last (i.e., on
            # top), so the user can grab it and drag it immediately.
            cw = self.canvas_w_var.get()
            ch = self.canvas_h_var.get()
            th, tw = mip.shape[:2]
            x = max(0, (cw - tw) // 2)
            y = max(0, (ch - th) // 2)

            tile = {
                'frame_idx': idx,                 # virtual index across all loaded raws
                'local_idx': local_idx,           # frame index within source_file
                'source_file': file_info['path'],
                'volume': vol,
                'mip': mip,
                'mip_idx': mip_idx,
                'x': x,
                'y': y,
                'label': f'F{idx}',
            }
            self.tiles.append(tile)
            self._update_tile_list()
            self._redraw_canvas()
            self._status(f"Frame {idx} reconstructed and added ({tw}x{th}, {vol.shape[2]} depths)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Reconstruction failed:\n{e}")
            self._status("Reconstruction error")

    # ── Canvas rendering ───────────────────────────────────────
    def _redraw_canvas(self):
        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()
        blend = self.blend_var.get()
        show_chrome = self.show_labels_var.get()

        if not self.tiles:
            canvas_img = np.zeros((ch, cw), dtype=np.float32)
        elif blend:
            # Overlap-averaged preview — matches export behaviour.
            canvas_acc = np.zeros((ch, cw), dtype=np.float64)
            canvas_cnt = np.zeros((ch, cw), dtype=np.float64)
            for tile in self.tiles:
                mip = tile['mip']
                th, tw = mip.shape[:2]
                tx, ty = tile['x'], tile['y']
                x0 = max(0, tx); y0 = max(0, ty)
                x1 = min(cw, tx + tw); y1 = min(ch, ty + th)
                if x1 <= x0 or y1 <= y0:
                    continue
                sx0 = x0 - tx; sy0 = y0 - ty
                sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
                canvas_acc[y0:y1, x0:x1] += mip[sy0:sy1, sx0:sx1]
                canvas_cnt[y0:y1, x0:x1] += 1.0
            mask = canvas_cnt > 0
            canvas_img = np.zeros((ch, cw), dtype=np.float32)
            canvas_img[mask] = (canvas_acc[mask] / canvas_cnt[mask]).astype(np.float32)
        else:
            # Stitching preview — max projection keeps every tile fully visible
            # so overlapping regions don't look washed out while aligning.
            canvas_img = np.zeros((ch, cw), dtype=np.float32)
            for tile in self.tiles:
                mip = tile['mip']
                th, tw = mip.shape[:2]
                tx, ty = tile['x'], tile['y']
                x0 = max(0, tx); y0 = max(0, ty)
                x1 = min(cw, tx + tw); y1 = min(ch, ty + th)
                if x1 <= x0 or y1 <= y0:
                    continue
                sx0 = x0 - tx; sy0 = y0 - ty
                sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
                np.maximum(canvas_img[y0:y1, x0:x1],
                           mip[sy0:sy1, sx0:sx1].astype(np.float32),
                           out=canvas_img[y0:y1, x0:x1])

        # Display
        self.stitch_ax.clear()
        self.stitch_ax.imshow(canvas_img, cmap='gray', vmin=0, vmax=1,
                               origin='upper', extent=[0, cw, ch, 0])

        # Draw tile outlines and labels — gated by the toggle, but always draw
        # an outline for selected tiles so selection feedback survives the hide.
        for i, tile in enumerate(self.tiles):
            th, tw = tile['mip'].shape[:2]
            tx, ty = tile['x'], tile['y']
            is_sel = i in self.selected_tiles
            if not (show_chrome or is_sel):
                continue
            color = 'cyan' if is_sel else 'yellow'
            lw = 2.0 if is_sel else 1.0
            rect = plt.Rectangle((tx, ty), tw, th,
                                  linewidth=lw, edgecolor=color,
                                  facecolor='none', linestyle='--')
            self.stitch_ax.add_patch(rect)
            if show_chrome:
                self.stitch_ax.text(tx + 3, ty + 12, tile['label'],
                                    color=color, fontsize=8, fontweight='bold')

        vx0, vx1, vy0, vy1 = self._current_view_bounds()
        self.stitch_ax.set_xlim(vx0, vx1)
        self.stitch_ax.set_ylim(vy1, vy0)  # origin upper
        self.stitch_ax.axis('off')
        mode = 'blend' if blend else 'max'
        zoom_pct = int(round(cw / max(vx1 - vx0, 1e-9) * 100))
        self.stitch_ax.set_title(
            f'Canvas ({len(self.tiles)} tiles, {mode}, {zoom_pct}%)',
            color='white', fontsize=9)
        self.stitch_mpl_canvas.draw_idle()

    def _resize_canvas(self):
        # Drag state is in canvas-pixel coordinates; after a resize those
        # coordinates may no longer match what the user sees, so clear it.
        self._reset_drag_state()
        self.zoom_view = None  # zoom bounds reference the old canvas size
        self._bind_canvas_events()
        self._redraw_canvas()

    def _bind_canvas_events(self):
        """(Re)connect press / motion / release callbacks on the stitch canvas.

        Resizing the embedded Tk widget can disturb FigureCanvas event state on
        some matplotlib backends; rebinding makes drag-to-move robust to that.
        """
        for cid in self._mouse_cids:
            try:
                self.stitch_mpl_canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mouse_cids = [
            self.stitch_mpl_canvas.mpl_connect(
                'button_press_event', self._on_canvas_press),
            self.stitch_mpl_canvas.mpl_connect(
                'motion_notify_event', self._on_canvas_drag),
            self.stitch_mpl_canvas.mpl_connect(
                'button_release_event', self._on_canvas_release),
            self.stitch_mpl_canvas.mpl_connect(
                'scroll_event', self._on_canvas_scroll),
        ]

    def _on_canvas_widget_configure(self, event=None):
        # <Configure> fires for every pixel of a window-drag resize; debounce.
        if self._configure_after_id is not None:
            try:
                self.root.after_cancel(self._configure_after_id)
            except Exception:
                pass
        self._configure_after_id = self.root.after(150, self._on_resize_settled)

    def _on_resize_settled(self):
        self._configure_after_id = None
        self._reset_drag_state()
        self._bind_canvas_events()
        # The Tk widget grew/shrank; nudge matplotlib to redraw at the new size
        # so the figure content actually fills the panel after a window resize.
        try:
            self.stitch_mpl_canvas.draw_idle()
        except Exception:
            pass

    def _reset_drag_state(self):
        self.drag_offset = None
        self.drag_start = None
        self.dragging = False
        try:
            self.stitch_mpl_canvas.get_tk_widget().config(cursor='')
        except Exception:
            pass

    # ── Zoom ───────────────────────────────────────────────────
    ZOOM_STEP = 1.25

    def _current_view_bounds(self):
        """Return (x0, x1, y0, y1) of the visible region in canvas-pixel coords."""
        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()
        if self.zoom_view is None:
            return (0.0, float(cw), 0.0, float(ch))
        return self.zoom_view

    def _zoom_in(self):
        self._zoom_canvas(self.ZOOM_STEP)

    def _zoom_out(self):
        self._zoom_canvas(1.0 / self.ZOOM_STEP)

    def _zoom_fit(self):
        self.zoom_view = None
        self._redraw_canvas()
        self._status("Zoom: fit")

    def _zoom_canvas(self, factor, anchor_x=None, anchor_y=None):
        """Scale the visible area by ``factor`` (>1 zoom in) around an anchor.

        ``anchor_x`` / ``anchor_y`` are in canvas-pixel data coordinates; if
        omitted the centre of the current view is used.
        """
        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()
        x0, x1, y0, y1 = self._current_view_bounds()
        if anchor_x is None:
            anchor_x = (x0 + x1) / 2
        if anchor_y is None:
            anchor_y = (y0 + y1) / 2

        new_w = (x1 - x0) / factor
        new_h = (y1 - y0) / factor
        # Cap zoom-in (keep tiles draggable) and zoom-out (don't exceed canvas).
        new_w = max(min(new_w, cw), 16.0)
        new_h = max(min(new_h, ch), 16.0)

        # Keep the anchor under the same fractional position in the new view.
        fx = (anchor_x - x0) / max(x1 - x0, 1e-9)
        fy = (anchor_y - y0) / max(y1 - y0, 1e-9)
        nx0 = anchor_x - fx * new_w
        ny0 = anchor_y - fy * new_h
        nx1 = nx0 + new_w
        ny1 = ny0 + new_h
        # Clamp inside the canvas.
        if nx0 < 0:
            nx0, nx1 = 0.0, new_w
        if ny0 < 0:
            ny0, ny1 = 0.0, new_h
        if nx1 > cw:
            nx1, nx0 = float(cw), float(cw) - new_w
        if ny1 > ch:
            ny1, ny0 = float(ch), float(ch) - new_h

        if abs(new_w - cw) < 0.5 and abs(new_h - ch) < 0.5:
            self.zoom_view = None  # snapped back to fit
        else:
            self.zoom_view = (nx0, nx1, ny0, ny1)
        self._redraw_canvas()
        zoom_pct = int(round(cw / max(new_w, 1e-9) * 100))
        self._status(f"Zoom: {zoom_pct}%")

    def _on_canvas_scroll(self, event):
        """Mouse-wheel over the stitch canvas zooms around the cursor.

        Plain wheel and Ctrl/Cmd+wheel both zoom — matplotlib delivers the
        scroll regardless of modifier, which honors the Ctrl/Cmd convention
        without breaking plain-wheel zoom for users who expect it.
        """
        if event.inaxes != self.stitch_ax or event.xdata is None:
            return
        factor = self.ZOOM_STEP if event.button == 'up' else 1.0 / self.ZOOM_STEP
        self._zoom_canvas(factor, anchor_x=event.xdata, anchor_y=event.ydata)

    # ── Tile list ──────────────────────────────────────────────
    def _update_tile_list(self):
        self.tile_listbox.delete(0, tk.END)
        for i, t in enumerate(self.tiles):
            th, tw = t['mip'].shape[:2]
            self.tile_listbox.insert(tk.END,
                f"[{i}] {t['label']}  pos=({t['x']},{t['y']})  size={tw}x{th}  D={t['volume'].shape[2]}")

    def _remove_tile(self):
        # Remove from listbox selection or canvas selection
        sel = self.tile_listbox.curselection()
        if sel:
            indices = sorted(sel, reverse=True)
            for idx in indices:
                self.tiles.pop(idx)
        elif self.selected_tiles:
            for idx in sorted(self.selected_tiles, reverse=True):
                if idx < len(self.tiles):
                    self.tiles.pop(idx)
        else:
            messagebox.showinfo("Info", "Select tiles to remove")
            return
        self.selected_tiles.clear()
        self._update_tile_list()
        self._redraw_canvas()

    # ── Mouse drag on canvas (multi-select + multi-drag) ──────
    def _find_tile_at(self, mx, my):
        """Find the topmost tile at canvas coordinates (mx, my)."""
        for i in range(len(self.tiles) - 1, -1, -1):
            t = self.tiles[i]
            th, tw = t['mip'].shape[:2]
            if t['x'] <= mx <= t['x'] + tw and t['y'] <= my <= t['y'] + th:
                return i
        return None

    def _find_tiles_in_rect(self, x0, y0, x1, y1):
        """Find all tiles overlapping a rectangle."""
        result = set()
        for i, t in enumerate(self.tiles):
            th, tw = t['mip'].shape[:2]
            # Check overlap
            if t['x'] + tw > x0 and t['x'] < x1 and t['y'] + th > y0 and t['y'] < y1:
                result.add(i)
        return result

    def _on_canvas_press(self, event):
        if event.inaxes != self.stitch_ax or event.xdata is None:
            return
        mx, my = event.xdata, event.ydata
        idx = self._find_tile_at(mx, my)

        if idx is not None:
            # Clicked on a tile
            if idx in self.selected_tiles:
                # Already selected — start dragging all selected
                pass
            else:
                # New selection (replace unless Ctrl held — but matplotlib
                # doesn't easily expose modifiers, so just select this one
                # unless it's already in the set)
                self.selected_tiles = {idx}
            # Record drag offset relative to the clicked tile
            t = self.tiles[idx]
            self.drag_offset = (mx - t['x'], my - t['y'])
            self.drag_start = None
            self.dragging = True
            self.stitch_mpl_canvas.get_tk_widget().config(cursor='fleur')
            self._status(
                f"Dragging {len(self.selected_tiles)} tile(s) — "
                f"release to drop")
            self._redraw_canvas()
        else:
            # Clicked on empty space — start rubber-band selection
            self.selected_tiles.clear()
            self.drag_offset = None
            self.drag_start = (mx, my)
            self._redraw_canvas()

    def _on_canvas_drag(self, event):
        if event.inaxes != self.stitch_ax or event.xdata is None:
            return
        mx, my = event.xdata, event.ydata

        if self.drag_offset is not None and self.selected_tiles:
            # Dragging selected tiles together
            # Find the reference tile (first in selection)
            ref_idx = min(self.selected_tiles)
            ref_t = self.tiles[ref_idx]
            dx = int(mx - self.drag_offset[0]) - ref_t['x']
            dy = int(my - self.drag_offset[1]) - ref_t['y']
            for i in self.selected_tiles:
                self.tiles[i]['x'] += dx
                self.tiles[i]['y'] += dy
            # Update offset for next drag event
            ref_t = self.tiles[ref_idx]
            self.drag_offset = (mx - ref_t['x'], my - ref_t['y'])
            self._redraw_canvas()

        elif self.drag_start is not None:
            # Rubber-band selection — just redraw with rectangle
            self._redraw_canvas()
            x0 = min(self.drag_start[0], mx)
            y0 = min(self.drag_start[1], my)
            x1 = max(self.drag_start[0], mx)
            y1 = max(self.drag_start[1], my)
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=1, edgecolor='white',
                                  facecolor='white', alpha=0.1, linestyle=':')
            self.stitch_ax.add_patch(rect)
            self.stitch_mpl_canvas.draw_idle()

    def _on_canvas_release(self, event):
        if self.drag_start is not None and event.xdata is not None:
            # Finish rubber-band: select tiles inside rectangle
            mx, my = event.xdata, event.ydata
            x0 = min(self.drag_start[0], mx)
            y0 = min(self.drag_start[1], my)
            x1 = max(self.drag_start[0], mx)
            y1 = max(self.drag_start[1], my)
            self.selected_tiles = self._find_tiles_in_rect(x0, y0, x1, y1)
            self._status(f"Selected {len(self.selected_tiles)} tiles")
        elif self.dragging and self.selected_tiles:
            anchor = self.tiles[min(self.selected_tiles)]
            self._status(
                f"Dropped {len(self.selected_tiles)} tile(s) — "
                f"anchor at ({anchor['x']}, {anchor['y']})")

        if self.selected_tiles:
            self._update_tile_list()
        self._reset_drag_state()
        self._redraw_canvas()

    # ── Shared export annotation helpers ───────────────────────
    @staticmethod
    def _format_scalebar_length(um):
        """Format a scale-bar length, switching μm→mm at ≥ 1000 μm.

        ``1500`` → ``"1.5 mm"``, ``2000`` → ``"2 mm"``, ``150.0`` → ``"150 μm"``.
        """
        if um >= 1000.0:
            mm = um / 1000.0
            if abs(mm - round(mm)) < 1e-6:
                return f"{int(round(mm))} mm"
            return f"{mm:g} mm"
        return f"{um:.0f} μm"

    def _draw_export_annotations(self, pil_img, cmap_obj):
        """Draw scale bar + depth colorbar onto ``pil_img`` (mutates).

        Returns an ``overlay_meta`` dict describing every primitive drawn so
        :meth:`_save_export_svg` can re-emit the same overlay as true vector
        elements in SVG output. Position / styling matches the original
        in-place annotation code from :meth:`_export_mip_color`.
        """
        from PIL import ImageDraw, ImageFont
        cw, ch = pil_img.size  # (width, height) per PIL
        draw = ImageDraw.Draw(pil_img)

        pixel_size = self.pixel_size_var.get()
        scalebar_um = self.scalebar_um_var.get()
        scalebar_px = (int(scalebar_um / pixel_size)
                        if pixel_size > 0 else int(cw * 0.1))

        color_map = {
            'white': (255, 255, 255), 'red': (255, 50, 50),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255),
            'green': (0, 255, 0),
        }
        bar_color = color_map.get(self.scalebar_color_var.get(),
                                    (255, 255, 255))

        fs = max(12, int(ch * 0.025))
        fs_s = max(10, int(ch * 0.018))
        fnt = fnt_s = None
        for fn in ['arial.ttf', 'Arial.ttf', 'C:/Windows/Fonts/arial.ttf',
                    'DejaVuSans.ttf']:
            try:
                fnt = ImageFont.truetype(fn, fs)
                fnt_s = ImageFont.truetype(fn, fs_s)
                break
            except (IOError, OSError):
                continue
        if fnt is None:
            fnt = ImageFont.load_default()
            fnt_s = fnt

        margin = int(cw * 0.03)
        shadow = max(1, fs // 12)

        # Scale bar (bottom-right).
        thick = max(3, int(ch * 0.008))
        bar_y = ch - margin
        bar_x1 = cw - margin - scalebar_px
        bar_x2 = cw - margin
        draw.rectangle([bar_x1, bar_y - thick, bar_x2, bar_y], fill=bar_color)

        # Scale bar label (centered on the bar). Auto-switches to "mm" once
        # the scale-bar length reaches 1000 μm.
        lbl = self._format_scalebar_length(scalebar_um)
        try:
            bb = draw.textbbox((0, 0), lbl, font=fnt_s)
            tw = bb[2] - bb[0]
        except AttributeError:
            tw = len(lbl) * fs_s * 0.6
        lx = bar_x1 + (scalebar_px - tw) / 2
        ly = bar_y - thick - fs_s - 3
        draw.text((lx + shadow, ly + shadow), lbl, fill=(0, 0, 0), font=fnt_s)
        draw.text((lx, ly), lbl, fill=bar_color, font=fnt_s)

        # Depth colorbar (right edge, vertical). Top = shallow, bottom = deep.
        cbar_w = max(10, int(cw * 0.015))
        cbar_h = int(ch * 0.35)
        cbar_x = cw - margin - cbar_w
        cbar_y0 = margin
        cbar_y1 = cbar_y0 + cbar_h
        for row in range(cbar_h):
            frac = row / max(cbar_h - 1, 1)
            rgb = cmap_obj(frac)[:3]
            c = tuple(int(v * 255) for v in rgb)
            draw.rectangle(
                [cbar_x, cbar_y0 + row, cbar_x + cbar_w, cbar_y0 + row + 1],
                fill=c)
        draw.rectangle([cbar_x - 1, cbar_y0 - 1,
                        cbar_x + cbar_w + 1, cbar_y1 + 1],
                        outline=bar_color, width=1)

        # Numerical depth ticks at top / middle / bottom of the colorbar.
        # Top of the bar is the shallow (small-z) end, bottom is the deep
        # (large-z) end — matching the cmap orientation drawn above.
        depth_lo = float(self.depth_lo_um_var.get())
        depth_hi = float(self.depth_hi_um_var.get())
        depth_mid = (depth_lo + depth_hi) / 2.0

        def _fmt_depth(um):
            if abs(um) >= 1000.0:
                mm = um / 1000.0
                if abs(mm - round(mm)) < 1e-6:
                    return f"{int(round(mm))} mm"
                return f"{mm:g} mm"
            if abs(um - round(um)) < 1e-6:
                return f"{int(round(um))} μm"
            return f"{um:g} μm"

        tick_texts = [_fmt_depth(depth_lo), _fmt_depth(depth_mid),
                       _fmt_depth(depth_hi)]
        tick_ys = [int(cbar_y0),
                    int(cbar_y0 + cbar_h / 2 - fs_s / 2),
                    int(cbar_y1 - fs_s)]

        # Right-align all tick labels at a common x just left of the colorbar.
        widths = []
        for t in tick_texts:
            try:
                bb = draw.textbbox((0, 0), t, font=fnt_s)
                widths.append(bb[2] - bb[0])
            except AttributeError:
                widths.append(int(len(t) * fs_s * 0.6))
        max_lw = max(widths) if widths else int(fs_s * 4)
        label_right = cbar_x - 6  # 6 px gap between text and colorbar
        tick_xs = [label_right - widths[i] for i in range(len(tick_texts))]

        cb_labels_meta = []
        for txt, lx_i, ly_i, lw_i in zip(tick_texts, tick_xs, tick_ys, widths):
            draw.text((lx_i + shadow, ly_i + shadow), txt,
                      fill=(0, 0, 0), font=fnt_s)
            draw.text((lx_i, ly_i), txt, fill=bar_color, font=fnt_s)
            cb_labels_meta.append({
                'text': txt,
                'x': int(lx_i), 'y': int(ly_i),
                'color_rgb': tuple(bar_color),
                'font_size': int(fs_s),
            })
            # Small tick mark joining label to colorbar.
            draw.line([(label_right + 1, ly_i + fs_s // 2),
                        (cbar_x - 1, ly_i + fs_s // 2)],
                       fill=bar_color, width=1)

        cmap_name = getattr(cmap_obj, 'name', 'turbo')
        return {
            'cw': int(cw), 'ch': int(ch),
            'shadow_dx': int(shadow), 'shadow_dy': int(shadow),
            'font_family': 'Arial, Helvetica, sans-serif',
            'scalebar': {
                'bar': {'x': int(bar_x1), 'y': int(bar_y - thick),
                         'w': int(scalebar_px), 'h': int(thick),
                         'color_rgb': tuple(bar_color)},
                'label': {'text': lbl,
                          'cx': int(bar_x1 + scalebar_px / 2),
                          'y': int(ly),
                          'color_rgb': tuple(bar_color),
                          'font_size': int(fs_s)},
            },
            'colorbar': {
                'x': int(cbar_x), 'y0': int(cbar_y0),
                'w': int(cbar_w), 'h': int(cbar_h),
                'border_color': tuple(bar_color),
                'cmap_name': cmap_name,
                'labels': cb_labels_meta,
                'ticks': [
                    {'x0': int(label_right + 1), 'x1': int(cbar_x - 1),
                     'y': int(ly_i + fs_s // 2),
                     'color_rgb': tuple(bar_color)}
                    for ly_i in tick_ys
                ],
            },
        }

    def _save_export_svg(self, path, pil_clean, overlay_meta, n_grad_stops=24):
        """Write a vector SVG: embedded clean raster + vector annotations.

        Scale bar → ``<rect>``. Scale-bar / colorbar labels → ``<text>`` with
        a black drop-shadow copy underneath. Depth colorbar → vertical
        ``<linearGradient>`` filled into a ``<rect>``, plus a border rect.
        The raster contains no annotations so the SVG opens with editable
        vector overlays on top of the data image.
        """
        import base64
        import io as _io
        from xml.sax.saxutils import escape as _escape

        buf = _io.BytesIO()
        pil_clean.save(buf, format='PNG')
        png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        w, h = pil_clean.size
        sb = overlay_meta['scalebar']
        cb = overlay_meta['colorbar']
        sdx = int(overlay_meta.get('shadow_dx', 1))
        sdy = int(overlay_meta.get('shadow_dy', 1))
        ff = overlay_meta.get('font_family', 'Arial, Helvetica, sans-serif')

        def _rgb(rgb):
            r, g, b = rgb
            return f"rgb({int(r)},{int(g)},{int(b)})"

        # Sample the matplotlib colormap to build SVG gradient stops.
        cmap = plt.colormaps.get_cmap(cb.get('cmap_name', 'turbo'))
        stops_xml = []
        for i in range(n_grad_stops):
            frac = i / (n_grad_stops - 1)
            r, g, b = (int(v * 255) for v in cmap(frac)[:3])
            stops_xml.append(
                f'      <stop offset="{frac * 100:.2f}%" '
                f'stop-color="rgb({r},{g},{b})"/>')
        grad_id = 'depth-cbar-grad'

        def _emit_text(t, anchor='start'):
            fs = int(t['font_size'])
            col = _rgb(t['color_rgb'])
            x = int(t['cx' if anchor == 'middle' else 'x'])
            y = int(t['y'])
            base = (f'font-family="{ff}" font-size="{fs}" '
                    f'font-weight="bold" text-anchor="{anchor}" '
                    f'dominant-baseline="text-before-edge"')
            txt = _escape(str(t['text']))
            return [
                f'    <text x="{x + sdx}" y="{y + sdy}" '
                f'fill="rgb(0,0,0)" {base}>{txt}</text>',
                f'    <text x="{x}" y="{y}" fill="{col}" {base}>{txt}</text>',
            ]

        anno = ['  <g id="annotations">']
        bar = sb['bar']
        anno.append(
            f'    <rect x="{bar["x"]}" y="{bar["y"]}" '
            f'width="{bar["w"]}" height="{bar["h"]}" '
            f'fill="{_rgb(bar["color_rgb"])}"/>')
        anno.extend(_emit_text(sb['label'], anchor='middle'))
        anno.append(
            f'    <rect x="{cb["x"]}" y="{cb["y0"]}" '
            f'width="{cb["w"]}" height="{cb["h"]}" '
            f'fill="url(#{grad_id})"/>')
        anno.append(
            f'    <rect x="{cb["x"] - 1}" y="{cb["y0"] - 1}" '
            f'width="{cb["w"] + 2}" height="{cb["h"] + 2}" '
            f'fill="none" stroke="{_rgb(cb["border_color"])}" '
            f'stroke-width="1"/>')
        # Tick marks (joining each numeric label to the colorbar edge).
        for tk in cb.get('ticks', []) or []:
            anno.append(
                f'    <line x1="{tk["x0"]}" y1="{tk["y"]}" '
                f'x2="{tk["x1"]}" y2="{tk["y"]}" '
                f'stroke="{_rgb(tk["color_rgb"])}" stroke-width="1"/>')
        for lbl in cb['labels']:
            anno.extend(_emit_text(lbl, anchor='start'))
        anno.append('  </g>')

        svg_text = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            '  <defs>\n'
            f'    <linearGradient id="{grad_id}" x1="0%" y1="0%" '
            'x2="0%" y2="100%">\n'
            + '\n'.join(stops_xml) + '\n'
            '    </linearGradient>\n'
            '  </defs>\n'
            f'  <image x="0" y="0" width="{w}" height="{h}" '
            f'preserveAspectRatio="none" '
            f'xlink:href="data:image/png;base64,{png_b64}"/>\n'
            + '\n'.join(anno) + '\n'
            '</svg>\n'
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(svg_text)

    # ── Export blended MIP ─────────────────────────────────────
    def _export_mip(self, blend_mode='linear'):
        """Export the grayscale mosaic with seam-aware blending.

        ``blend_mode`` is ``'linear'`` (feather + valid-mask) or ``'multiband'``
        (Burt-Adelson Laplacian pyramid). Both apply per-tile background
        subtraction and a global gain match before compositing.
        """
        if not self.tiles:
            messagebox.showwarning("Warning", "No tiles to export")
            return

        from slim_recon import blend_export

        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()

        # Pre-blend (uniform-average, valid-mask gated) for the seam diagnostic
        # so the user sees row/col |diff| spikes shrink after blending.
        old = blend_export.blend_linear(
            self.tiles, [t['mip'] for t in self.tiles], (cw, ch), margin=1)
        blend_export.print_seam_report("BEFORE (uniform avg)", old)

        self._status(f"Blending {len(self.tiles)} tiles ({blend_mode})...")
        result = blend_export.stitch_gray(self.tiles, (cw, ch),
                                            blend=blend_mode, normalize=True)
        blend_export.print_seam_report(f"AFTER  ({blend_mode})", result)

        # Normalize to 99.5-percentile of nonzero pixels for uint8 output.
        nz = result[result > 0]
        vmax = float(np.percentile(nz, 99.5)) if nz.size else 1.0
        result_norm = np.clip(result / max(vmax, 1e-9), 0.0, 1.0)

        path = filedialog.asksaveasfilename(
            title="Export Blended MIP",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG (vector)", "*.svg"),
                        ("TIFF", "*.tif"), ("All", "*.*")])
        if not path:
            return

        # Build a 3-channel RGB raster so the same overlay code (scale bar +
        # depth colorbar) can draw coloured primitives onto the grayscale MIP.
        from PIL import Image
        img_u8 = (result_norm * 255).astype(np.uint8)
        pil_clean = Image.fromarray(img_u8, mode='L').convert('RGB')
        pil_annotated = pil_clean.copy()
        cmap_obj = plt.colormaps.get_cmap('turbo')
        overlay_meta = self._draw_export_annotations(pil_annotated, cmap_obj)

        if path.lower().endswith('.svg'):
            # Vector SVG: embed clean raster + vector annotations.
            self._save_export_svg(path, pil_clean, overlay_meta)
            self._status(f"MIP exported as SVG ({blend_mode}): {path}")
            messagebox.showinfo(
                "Saved",
                f"Blended MIP saved (vector SVG):\n{path}\n"
                f"blend={blend_mode}")
        else:
            pil_annotated.save(path, dpi=(300, 300))
            self._status(f"MIP exported ({blend_mode}): {path}")
            messagebox.showinfo(
                "Saved",
                f"Blended MIP saved:\n{path}\nblend={blend_mode}")

    # ── Export depth-coded color MIP with scale bar ──────────
    def _export_mip_color(self, blend_mode='linear', weight_power=1.0,
                          top_k=3, sigma_z=1.5, attenuation_correct=True,
                          conf_k=2.0, conf_bg_pct=10.0,
                          use_vesselness=True, cmap_name='turbo',
                          saturation_boost=1.2, depth_gain=0.0,
                          **kwargs):
        """Export the depth-coded color mosaic.

        Iteration 3 of the depth encoder: ``H = cmap(depth), S = conf,
        V = MIP_norm`` so brightness exactly tracks the gray MIP and
        low-confidence pixels go gray (not black). Depth itself comes from
        a top-K weighted mean over per-slice attenuation-corrected volumes,
        regularized with vessel-aware skeleton smoothing. See
        ``slim_recon.blend_export.stitch_color`` for the full pipeline and
        all tunable knobs (passed through ``**kwargs``).
        """
        if not self.tiles:
            messagebox.showwarning("Warning", "No tiles to export")
            return

        from PIL import Image, ImageDraw, ImageFont
        from slim_recon import blend_export

        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()

        self._status(
            f"Building depth-coded color ({len(self.tiles)} tiles, "
            f"blend={blend_mode})...")
        color_img = blend_export.stitch_color(
            self.tiles, (cw, ch),
            weight_power=weight_power,
            top_k=top_k,
            sigma_z=sigma_z,
            attenuation_correct=attenuation_correct,
            conf_k=conf_k,
            conf_bg_pct=conf_bg_pct,
            use_vesselness=use_vesselness,
            cmap=cmap_name,
            saturation_boost=saturation_boost,
            depth_gain_alpha=depth_gain,
            blend=blend_mode,
            normalize=True,
            **kwargs,
        )
        cmap = plt.colormaps.get_cmap(cmap_name)

        # Clean (un-annotated) raster + annotated copy. The shared helper
        # also returns metadata so the SVG path emits the scale bar / depth
        # colorbar as vector primitives instead of PNG-baked pixels.
        pil_clean = Image.fromarray(color_img)
        pil_annotated = pil_clean.copy()
        overlay_meta = self._draw_export_annotations(pil_annotated, cmap)

        path = filedialog.asksaveasfilename(
            title="Export Depth-Coded Color MIP",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG (vector)", "*.svg"),
                        ("TIFF", "*.tif"), ("All", "*.*")])
        if not path:
            return

        if path.lower().endswith('.svg'):
            self._save_export_svg(path, pil_clean, overlay_meta)
            self._status(f"Color MIP exported as SVG: {path}")
            messagebox.showinfo("Saved",
                f"Depth-coded color MIP saved (vector SVG):\n{path}")
        else:
            pil_annotated.save(path, dpi=(300, 300))
            self._status(f"Color MIP exported: {path}")
            messagebox.showinfo("Saved",
                f"Depth-coded color MIP saved:\n{path}")

    # ── Export blended 3D volume ───────────────────────────────
    def _export_volume(self):
        if not self.tiles:
            messagebox.showwarning("Warning", "No tiles to export")
            return

        cw = self.canvas_w_var.get()
        ch = self.canvas_h_var.get()

        # Find max depth across all tiles
        max_d = max(t['volume'].shape[2] for t in self.tiles)
        self._status(f"Building 3D canvas ({cw}x{ch}x{max_d})...")

        canvas_acc = np.zeros((ch, cw, max_d), dtype=np.float64)
        canvas_cnt = np.zeros((ch, cw, max_d), dtype=np.float64)

        for tile in self.tiles:
            vol = tile['volume']
            th, tw, td = vol.shape
            tx, ty = tile['x'], tile['y']
            x0, y0 = max(0, tx), max(0, ty)
            x1, y1 = min(cw, tx + tw), min(ch, ty + th)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0, sy0 = x0 - tx, y0 - ty

            # Resample depth if needed
            if td != max_d:
                from scipy.ndimage import zoom
                vol_r = zoom(vol[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0), :],
                              [1, 1, max_d / td], order=1)
            else:
                vol_r = vol[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0), :]

            canvas_acc[y0:y1, x0:x1, :vol_r.shape[2]] += vol_r
            canvas_cnt[y0:y1, x0:x1, :vol_r.shape[2]] += 1.0

        mask = canvas_cnt > 0
        result = np.zeros_like(canvas_acc, dtype=np.float32)
        result[mask] = (canvas_acc[mask] / canvas_cnt[mask]).astype(np.float32)

        path = filedialog.asksaveasfilename(
            title="Export 3D Volume",
            defaultextension=".mat",
            filetypes=[("MAT", "*.mat"), ("NPY", "*.npy"), ("All", "*.*")])
        if not path:
            return

        self._status("Saving 3D volume...")
        if path.endswith('.mat'):
            import scipy.io as sio
            sio.savemat(path, {'volume': result}, do_compression=True)
        elif path.endswith('.npy'):
            np.save(path, result)
        else:
            np.save(path + '.npy', result)

        self._status(f"Volume exported: {path}")
        messagebox.showinfo("Saved",
            f"3D volume saved:\n{path}\n"
            f"Shape: {result.shape} (H, W, D)")

    # ── Project save / load ────────────────────────────────────
    PROJECT_VERSION = 1

    def _save_project(self):
        """Pickle the full session state (tiles + canvas + paths + settings).

        Tile volumes/MIPs are bundled so the project reloads without re-running
        reconstruction. Raw / calibration files are stored as paths only and
        are reloaded by their original locations on load.
        """
        if not self.tiles and not self.raw_files:
            messagebox.showinfo("Info", "Nothing to save yet")
            return
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".stitchproj",
            filetypes=[("Stitch Project", "*.stitchproj"),
                        ("All", "*.*")])
        if not path:
            return

        state = {
            'version': self.PROJECT_VERSION,
            'tiles': self.tiles,
            'canvas_w': self.canvas_w_var.get(),
            'canvas_h': self.canvas_h_var.get(),
            'raw_paths': [f['path'] for f in self.raw_files],
            'geo_path': self.geo_var.get(),
            'psf_path': self.psf_var.get(),
            'pixel_size': self.pixel_size_var.get(),
            'scalebar_um': self.scalebar_um_var.get(),
            'scalebar_color': self.scalebar_color_var.get(),
            'show_labels': self.show_labels_var.get(),
            'blend': self.blend_var.get(),
            'frame_idx': self.frame_idx_var.get(),
            'recon_params': {
                'num_frames': self.num_frames_var.get(),
                'clip_lo': self.clip_lo_var.get(),
                'clip_hi': self.clip_hi_var.get(),
                'n_depth': self.n_depth_var.get(),
                'psf_n': self.psf_n_var.get(),
                'n_iters': self.n_iters_var.get(),
            },
        }

        import pickle
        import gzip
        self._status(f"Saving project ({len(self.tiles)} tiles)...")
        try:
            with gzip.open(path, 'wb', compresslevel=4) as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")
            self._status("Save failed")
            return

        size_mb = os.path.getsize(path) / (1024 * 1024)
        self._status(
            f"Project saved ({size_mb:.1f} MB, {len(self.tiles)} tiles): {path}")
        messagebox.showinfo(
            "Saved",
            f"Project saved:\n{path}\n"
            f"{len(self.tiles)} tile(s), {len(self.raw_files)} raw file(s), "
            f"{size_mb:.1f} MB")

    def _load_project(self):
        path = filedialog.askopenfilename(
            title="Load Project",
            filetypes=[("Stitch Project", "*.stitchproj"),
                        ("Pickle", "*.pkl"),
                        ("All", "*.*")])
        if not path or not os.path.exists(path):
            return

        if (self.tiles or self.raw_files) and not messagebox.askyesno(
                "Confirm",
                "Loading will replace your current tiles, raw files, and "
                "canvas settings. Continue?"):
            return

        import pickle
        import gzip
        self._status(f"Loading project from {os.path.basename(path)}...")
        # Try gzip first; fall back to raw pickle for forward-compat.
        state = None
        for opener in (lambda p: gzip.open(p, 'rb'), lambda p: open(p, 'rb')):
            try:
                with opener(path) as f:
                    state = pickle.load(f)
                break
            except Exception:
                continue
        if state is None:
            messagebox.showerror("Error", f"Could not read:\n{path}")
            self._status("Load failed")
            return

        if not isinstance(state, dict) or 'tiles' not in state:
            messagebox.showerror("Error", "Not a valid stitch project file")
            self._status("Load failed")
            return

        # ── Restore canvas + settings ──
        if 'canvas_w' in state:
            self.canvas_w_var.set(int(state['canvas_w']))
        if 'canvas_h' in state:
            self.canvas_h_var.set(int(state['canvas_h']))
        if 'pixel_size' in state:
            self.pixel_size_var.set(float(state['pixel_size']))
        if 'scalebar_um' in state:
            self.scalebar_um_var.set(float(state['scalebar_um']))
        if 'scalebar_color' in state:
            self.scalebar_color_var.set(str(state['scalebar_color']))
        if 'show_labels' in state:
            self.show_labels_var.set(bool(state['show_labels']))
        if 'blend' in state:
            self.blend_var.set(bool(state['blend']))
        rp = state.get('recon_params', {})
        if 'num_frames' in rp:
            self.num_frames_var.set(int(rp['num_frames']))
        if 'clip_lo' in rp:
            self.clip_lo_var.set(float(rp['clip_lo']))
        if 'clip_hi' in rp:
            self.clip_hi_var.set(float(rp['clip_hi']))
        if 'n_depth' in rp:
            self.n_depth_var.set(int(rp['n_depth']))
        if 'psf_n' in rp:
            self.psf_n_var.set(int(rp['psf_n']))
        if 'n_iters' in rp:
            self.n_iters_var.set(int(rp['n_iters']))

        if 'geo_path' in state:
            self.geo_var.set(state['geo_path'])
        if 'psf_path' in state:
            self.psf_var.set(state['psf_path'])

        # ── Restore raw files (paths only; reload from disk if present) ──
        self.raw_files = []
        self.raw_listbox.delete(0, tk.END)
        missing = []
        for p in state.get('raw_paths', []):
            if not os.path.exists(p):
                missing.append(p)
                continue
            try:
                data = data_io.load_data_auto(p, num_frames=self.num_frames_var.get())
            except Exception as e:
                missing.append(f"{p} ({e})")
                continue
            self.raw_files.append(
                {'path': p, 'data': data, 'num_frames': data.shape[2]})
            self.raw_listbox.insert(
                tk.END, f"{os.path.basename(p)}  ({data.shape[2]} frames)")
            self.raw_var.set(p)
        self._refresh_frame_range()
        if 'frame_idx' in state and self.raw_files:
            try:
                idx = max(0, min(int(state['frame_idx']),
                                  self._total_frames() - 1))
                self.frame_idx_var.set(idx)
                self._on_frame_change(idx)
            except Exception:
                pass

        # ── Restore tiles ──
        self.tiles = list(state.get('tiles', []))
        self.selected_tiles.clear()
        self.zoom_view = None
        self._update_tile_list()
        self._redraw_canvas()

        n_tiles = len(self.tiles)
        n_raw = len(self.raw_files)
        msg = (f"Project loaded:\n{path}\n"
               f"{n_tiles} tile(s), {n_raw} raw file(s)")
        if missing:
            msg += ("\n\nWarning: could not reload these raw files:\n  "
                    + "\n  ".join(missing[:5]))
            if len(missing) > 5:
                msg += f"\n  ... ({len(missing) - 5} more)"
        self._status(f"Project loaded: {n_tiles} tiles, {n_raw} raw file(s)")
        messagebox.showinfo("Loaded", msg)


def main():
    root = tk.Tk()
    app = StitchTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
